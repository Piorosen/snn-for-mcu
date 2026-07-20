/*
 * TFLM 경로 전체 파이프라인 검증 (호스트, 실제 snn_tflm.cc + 실제 모델 데이터 +
 * TFLM 런타임 + CMSIS-NN C 커널).
 *
 * 레퍼런스: 동일 시맨틱의 나이브 구현
 *   conv5x5(픽셀별 requantize) -> avgpool(반올림) -> LIF -> ... -> FC -> LIF_out
 * 30스텝 x 3발화율로 최종 출력 스파이크/막전위 비교.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "snn.h"             /* TFLM 경로 공개 API (snn_tflm.cc) */
#include "snn_weights_fused.h" /* snn_fc_weight_hwc (레퍼런스 FC용) */

extern "C" size_t snn_tflm_arena_used(void);

/* ---------- 양자화 헬퍼 (snn 경로와 동일) ---------- */
static void qmult(double m, int32_t *mult, int32_t *shift) {
    int exp; double q = frexp(m, &exp);
    int64_t q31 = llround(q * 2147483648.0);
    if (q31 == 2147483648LL) { q31 >>= 1; exp++; }
    *mult = (int32_t)q31; *shift = exp;
}
static int32_t red_mult(int32_t m) { return (m < 0x7FFF0000) ? ((m + (1 << 15)) >> 16) : 0x7FFF; }
static int32_t requant64(int64_t v, int32_t rm, int32_t sh) {
    int32_t r = (int32_t)((v * rm) >> (14 - sh));
    return (r + 1) >> 1;
}
static int16_t clamp16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v; }

/* 퓨전(6x6/s2) raw 누산: 5x5 CHW 원본에서 즉석 합산 (생성기와 독립 구현) */
static int64_t fused6_acc(const int16_t *in, int H, int W, int Ci,
                          const int8_t *w5, int co, int oh, int ow) {
    int64_t acc = 0;
    for (int a = 0; a < 6; ++a) {
        int r = 2 * oh - 2 + a; if (r < 0 || r >= H) continue;
        for (int b = 0; b < 6; ++b) {
            int c = 2 * ow - 2 + b; if (c < 0 || c >= W) continue;
            for (int ci = 0; ci < Ci; ++ci) {
                int w6 = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        int ka = a - dy, kb = b - dx;
                        if (ka >= 0 && ka < 5 && kb >= 0 && kb < 5)
                            w6 += w5[((co * Ci + ci) * 5 + ka) * 5 + kb];
                    }
                acc += (int32_t)in[(r * W + c) * Ci + ci] * w6;
            }
        }
    }
    return acc;
}

static void ref_lif(int n, const int16_t *in, int16_t *spk_out,
                    int16_t *mem, int16_t *prev,
                    int16_t beta, int16_t thr, int16_t sv) {
    for (int i = 0; i < n; ++i) {
        int32_t t = ((int32_t)beta * mem[i]) >> 15;
        int16_t mt = clamp16(t + in[i]);
        int16_t sr = 0, mn = mt;
        if (mt >= thr) { sr = sv; mn = (int16_t)(mt - thr); }
        spk_out[i] = prev[i]; prev[i] = sr; mem[i] = mn;
    }
}

/* 퓨전 레퍼런스: 6x6 raw acc + 단일 requantize(배율 ws/(32768*4)) + LIF */
static void ref_conv_pool_lif(const int16_t *in, int H, int W, int Ci, int Co,
                              const int8_t *w5, const int8_t *bias8, int bs,
                              int16_t ws_q15, int16_t *mem, int16_t *prev,
                              int16_t beta, int16_t thr, int16_t sv, int16_t *spk_out) {
    int Ho = H / 2, Wo = W / 2;
    static int16_t pooled[32 * 16 * 16];
    double m = (double)ws_q15 / 32768.0 / 4.0;
    int32_t mult, shift; qmult(m, &mult, &shift);
    int32_t rm = red_mult(mult);
    for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow)
            for (int co = 0; co < Co; ++co) {
                int64_t b64 = llround(((double)bias8[co] * bs / 32768.0) / ((1.0 / 32768.0) * m));
                pooled[(oh * Wo + ow) * Co + co] = clamp16(
                    requant64(fused6_acc(in, H, W, Ci, w5, co, oh, ow) + b64, rm, shift));
            }
    ref_lif(Ho * Wo * Co, pooled, spk_out, mem, prev, beta, thr, sv);
}

int main() {
    srand(42);

    snn_accel_init();
    printf("arena used: %zu bytes\n", snn_tflm_arena_used());

    /* 레퍼런스 상태 */
    static int16_t rm1[8192], rp1[8192], rm2[4096], rp2[4096], rmo[10], rpo[10];
    static int16_t rs1[8192], rs2[4096];

    static int16_t x[32 * 32 * 3];
    int16_t spk[10], mem[10], rspk[10];

    long long out_spk_diff = 0, out_mem_maxdiff = 0, n_out = 0;
    long long pred_diff = 0;
    const double rates[] = { 0.1, 0.45, 0.8 };

    for (int ri = 0; ri < 3; ++ri) {
        snn_reset_state();
        memset(rm1, 0, sizeof(rm1)); memset(rp1, 0, sizeof(rp1));
        memset(rm2, 0, sizeof(rm2)); memset(rp2, 0, sizeof(rp2));
        memset(rmo, 0, sizeof(rmo)); memset(rpo, 0, sizeof(rpo));
        int calc_t[10] = {0}, calc_r[10] = {0};

        for (int t = 0; t < 30; ++t) {
            for (int i = 0; i < 32 * 32 * 3; ++i)
                x[i] = (((double)rand() / RAND_MAX) < rates[ri]) ? 32767 : 0;

            snn_forward_step(x, spk, mem);

            /* 레퍼런스 파이프라인 */
            ref_conv_pool_lif(x, 32, 32, 3, 32, (const int8_t *)snn_conv1_weight,
                              (const int8_t *)snn_conv1_bias, snn_conv1_bias_scale,
                              snn_conv1_weight_scale, rm1, rp1,
                              snn_lif1_beta, snn_lif1_threshold, 32767, rs1);
            ref_conv_pool_lif(rs1, 16, 16, 32, 64, (const int8_t *)snn_conv2_weight,
                              (const int8_t *)snn_conv2_bias, snn_conv2_bias_scale,
                              snn_conv2_weight_scale, rm2, rp2,
                              snn_lif2_beta, snn_lif2_threshold, 32767, rs2);
            int16_t fc[10];
            {
                double m = (double)snn_fc_weight_scale / 32768.0;
                int32_t mult, shift; qmult(m, &mult, &shift);
                int32_t rmu = red_mult(mult);
                for (int o = 0; o < 10; ++o) {
                    int64_t acc = 0;
                    for (int i = 0; i < 4096; ++i)
                        acc += (int32_t)rs2[i] * (int32_t)snn_fc_weight_hwc[o * 4096 + i];
                    int64_t b64 = llround((double)snn_fc_bias[o] * snn_fc_bias_scale * 32768.0 /
                                          (double)snn_fc_weight_scale);
                    fc[o] = clamp16(requant64(acc + b64, rmu, shift));
                }
            }
            ref_lif(10, fc, rspk, rmo, rpo, snn_lif_out_beta, snn_lif_out_threshold, 32767);

            for (int o = 0; o < 10; ++o) {
                n_out++;
                if ((spk[o] != 0) != (rspk[o] != 0)) out_spk_diff++;
                long long d = llabs((long long)mem[o] - rmo[o]);
                if (d > out_mem_maxdiff) out_mem_maxdiff = d;
                calc_t[o] += spk[o] > 16000;
                calc_r[o] += rspk[o] > 16000;
            }
        }
        int pt = 0, pr = 0;
        for (int o = 1; o < 10; ++o) {
            if (calc_t[o] > calc_t[pt]) pt = o;
            if (calc_r[o] > calc_r[pr]) pr = o;
        }
        pred_diff += (pt != pr);
        printf("rate %.2f: pred tflm=%d ref=%d, votes tflm[%d]=%d ref[%d]=%d\n",
               rates[ri], pt, pr, pt, calc_t[pt], pr, calc_r[pr]);
    }

    printf("output spikes: %lld/%lld disagree (%.3f%%), max |mem diff| = %lld LSB, pred mismatch %lld/3\n",
           out_spk_diff, n_out, 100.0 * out_spk_diff / n_out, out_mem_maxdiff, pred_diff);
    printf(out_spk_diff * 100 < n_out && pred_diff == 0 ? "PASS\n" : "CHECK\n");
    return 0;
}
