/*
 * Conv+Pool+LIF 퓨전 파이프라인 검증 (실제 가중치/실제 커널 소스, CMSIS-NN API).
 *
 * 1) 항등식: 생성된 6x6 퓨전 가중치의 raw 누산 == 원본 5x5 conv 4위치 합
 * 2) FC HWC 퍼뮤트 배열 == 원본 배열 재배열
 * 3) 커널 == 나이브 퓨전 레퍼런스, 30스텝 상태 포함 비트 일치
 * 4) 드리프트: 기존 경로 대비 스파이크 불일치율 정량화
 * 5) leftover 경로: Wo 홀수(리뷰 반례 H=4,W=6) + Co 홀수 지오메트리 비트 일치
 *
 * 컴파일:
 *  스칼라: clang -O2 -I <Inc> -I <Src> -I <cmsis_nn/Include> test_fused.c -lm
 *  DSP목:  위에 -DARM_MATH_DSP -I simd_mock 추가
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "snn_weights.c"        /* 원본 가중치/스케일 */
#include "snn_weights_fused.c"  /* 생성된 퓨전 가중치 */
#include "snn_lif_s16.c"        /* 실제 LIF 커널 */
#include "snn_conv_pool_lif_s16.c" /* 실제 퓨전 커널 (CMSIS-NN API) */

#define SPK1 16384

/* ---------- snn.c와 동일한 양자화 매핑 ---------- */
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

/* ---------- 커널 호출 헬퍼 (CMSIS-NN API) ---------- */
static void run_fused(int H, int W, int Ci, int Co, const int16_t *in, const int16_t *w6,
                      const int64_t *b64, const int32_t *mults, const int32_t *shifts,
                      int16_t *mem, int16_t *prev, int16_t beta, int16_t thr, int16_t sv,
                      int16_t *spk_out, int16_t *im2col) {
    const cmsis_nn_context ctx = { .buf = im2col, .size = (int32_t)(2 * 36 * Ci * 2) };
    const cmsis_nn_conv_params cp = {
        .input_offset = 0, .output_offset = 0,
        .stride = { .w = 2, .h = 2 }, .padding = { .w = 2, .h = 2 },
        .dilation = { .w = 1, .h = 1 }, .activation = { .min = -32768, .max = 32767 },
    };
    const cmsis_nn_per_channel_quant_params qp = { (int32_t *)mults, (int32_t *)shifts };
    const cmsis_nn_dims in_d  = { 1, H, W, Ci };
    const cmsis_nn_dims flt_d = { Co, 6, 6, Ci };
    const cmsis_nn_dims out_d = { 1, H / 2, W / 2, Co };
    const cmsis_nn_dims bias_d = { 0, 0, 0, 0 };
    const snn_lif_ctx lif = { mem, prev, beta, thr, sv };
    if (snn_convolve_pool_lif_s16(&ctx, &cp, &qp, &in_d, in, &flt_d, w6,
                                  &bias_d, b64, &lif, &out_d, spk_out) != ARM_CMSIS_NN_SUCCESS) {
        printf("KERNEL ARG ERROR\n"); exit(1);
    }
}

/* ---------- raw 누산 레퍼런스 ---------- */
static int64_t conv5_acc(const int16_t *in, int H, int W, int Ci,
                         const int8_t *w5, int co, int r, int c) {
    int64_t acc = 0;
    for (int kh = 0; kh < 5; ++kh) {
        int rr = r + kh - 2; if (rr < 0 || rr >= H) continue;
        for (int kw = 0; kw < 5; ++kw) {
            int cc = c + kw - 2; if (cc < 0 || cc >= W) continue;
            for (int ci = 0; ci < Ci; ++ci)
                acc += (int32_t)in[(rr * W + cc) * Ci + ci] *
                       (int32_t)w5[((co * Ci + ci) * 5 + kh) * 5 + kw];
        }
    }
    return acc;
}

static int64_t fused6_acc(const int16_t *in, int H, int W, int Ci,
                          const int16_t *w6, int co, int oh, int ow) {
    int64_t acc = 0;
    for (int a = 0; a < 6; ++a) {
        int r = 2 * oh - 2 + a; if (r < 0 || r >= H) continue;
        for (int b = 0; b < 6; ++b) {
            int c = 2 * ow - 2 + b; if (c < 0 || c >= W) continue;
            for (int ci = 0; ci < Ci; ++ci)
                acc += (int32_t)in[(r * W + c) * Ci + ci] *
                       (int32_t)w6[((co * 6 + a) * 6 + b) * Ci + ci];
        }
    }
    return acc;
}

/* ---------- 원본 LIF 시맨틱 (독립 레퍼런스) ---------- */
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

/* 나이브 퓨전 레이어 레퍼런스 (커널과 같은 수식, 다른 구현, per-channel quant) */
static void ref_fused_layer(const int16_t *in, int H, int W, int Ci, int Co,
                            const int16_t *w6, const int64_t *b64,
                            const int32_t *mults, const int32_t *shifts,
                            int16_t *mem, int16_t *prev,
                            int16_t beta, int16_t thr, int16_t sv, int16_t *spk_out) {
    int Ho = H / 2, Wo = W / 2;
    static int16_t tmp[32 * 16 * 16];
    for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow)
            for (int co = 0; co < Co; ++co)
                tmp[(oh * Wo + ow) * Co + co] =
                    clamp16(requant64(fused6_acc(in, H, W, Ci, w6, co, oh, ow) + b64[co],
                                      red_mult(mults[co]), shifts[co]));
    ref_lif(Ho * Wo * Co, tmp, spk_out, mem, prev, beta, thr, sv);
}

/* 기존(퓨전 전) 경로: 픽셀별 requantize -> 2x2 평균(반올림) -> LIF */
static void old_path_layer(const int16_t *in, int H, int W, int Ci, int Co,
                           const int8_t *w5, const int8_t *bias8, int bias_scale,
                           double m_conv, int16_t *mem, int16_t *prev,
                           int16_t beta, int16_t thr, int16_t sv, int16_t *spk_out) {
    int Ho = H / 2, Wo = W / 2;
    static int16_t convout[32 * 32 * 32];
    static int16_t pooled[32 * 16 * 16];
    int32_t mult, shift; qmult(m_conv, &mult, &shift);
    int32_t rm = red_mult(mult);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            for (int co = 0; co < Co; ++co) {
                int64_t b64 = llround((double)bias8[co] * bias_scale / m_conv);
                convout[(r * W + c) * Co + co] =
                    clamp16(requant64(conv5_acc(in, H, W, Ci, w5, co, r, c) + b64, rm, shift));
            }
    for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow)
            for (int co = 0; co < Co; ++co) {
                int32_t s = convout[((2 * oh) * W + 2 * ow) * Co + co]
                          + convout[((2 * oh) * W + 2 * ow + 1) * Co + co]
                          + convout[((2 * oh + 1) * W + 2 * ow) * Co + co]
                          + convout[((2 * oh + 1) * W + 2 * ow + 1) * Co + co];
                pooled[(oh * Wo + ow) * Co + co] = clamp16((s + ((s >= 0) ? 2 : -2)) >> 2);
            }
    ref_lif(Ho * Wo * Co, pooled, spk_out, mem, prev, beta, thr, sv);
}

static void rand_spikes(int16_t *v, int n, double rate, int16_t val) {
    for (int i = 0; i < n; ++i) v[i] = (((double)rand() / RAND_MAX) < rate) ? val : 0;
}

int main(void) {
    srand(2026);
    int ok = 1;

    /* ---------- 1) 항등식: 생성 가중치 == 5x5 4위치 합 ---------- */
    {
        static int16_t in1[32 * 32 * 3], in2[16 * 16 * 32];
        long long bad = 0;
        for (int seed = 0; seed < 3; ++seed) {
            for (int i = 0; i < 32 * 32 * 3; ++i) in1[i] = (int16_t)((rand() % 65536) - 32768);
            for (int i = 0; i < 16 * 16 * 32; ++i) in2[i] = (int16_t)((rand() % 65536) - 32768);
            for (int co = 0; co < 32; ++co)
                for (int oh = 0; oh < 16; ++oh)
                    for (int ow = 0; ow < 16; ++ow) {
                        int64_t s4 = 0;
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx)
                                s4 += conv5_acc(in1, 32, 32, 3, (const int8_t *)snn_conv1_weight,
                                                co, 2 * oh + dy, 2 * ow + dx);
                        if (s4 != fused6_acc(in1, 32, 32, 3, snn_conv1_fused_w, co, oh, ow)) bad++;
                    }
            for (int co = 0; co < 64; ++co)
                for (int oh = 0; oh < 8; ++oh)
                    for (int ow = 0; ow < 8; ++ow) {
                        int64_t s4 = 0;
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx)
                                s4 += conv5_acc(in2, 16, 16, 32, (const int8_t *)snn_conv2_weight,
                                                co, 2 * oh + dy, 2 * ow + dx);
                        if (s4 != fused6_acc(in2, 16, 16, 32, snn_conv2_fused_w, co, oh, ow)) bad++;
                    }
        }
        printf("1) fusion identity: %s (%lld mismatch)\n", bad ? "FAIL" : "PASS", bad);
        ok &= !bad;
    }

    /* ---------- 2) FC 퍼뮤트 ---------- */
    {
        long long bad = 0;
        for (int o = 0; o < 10; ++o)
            for (int c = 0; c < 64; ++c)
                for (int h = 0; h < 8; ++h)
                    for (int w = 0; w < 8; ++w)
                        if (snn_fc_weight_hwc[o * 4096 + (h * 8 + w) * 64 + c] !=
                            snn_fc_weight[o][c * 64 + h * 8 + w]) bad++;
        printf("2) fc permute: %s\n", bad ? "FAIL" : "PASS");
        ok &= !bad;
    }

    /* 양자화 파라미터 (snn.c와 동일, per-channel 복제) */
    const double m1 = ((double)snn_conv1_weight_scale / 32768.0) / 4.0;
    const double m2 = ((double)snn_conv2_weight_scale / 32768.0) * (32767.0 / SPK1) / 4.0;
    int32_t mu1[32], sh1[32], mu2[64], sh2[64];
    int64_t b1[32], b2[64];
    for (int c = 0; c < 32; ++c) {
        qmult(m1, &mu1[c], &sh1[c]);
        b1[c] = llround((double)snn_conv1_bias[c] * snn_conv1_bias_scale / m1);
    }
    for (int c = 0; c < 64; ++c) {
        qmult(m2, &mu2[c], &sh2[c]);
        b2[c] = llround((double)snn_conv2_bias[c] * snn_conv2_bias_scale / m2);
    }

    /* ---------- 3) 커널 == 나이브 레퍼런스 (30스텝, 실제 가중치) ---------- */
    {
        static __attribute__((aligned(4))) int16_t in1[32 * 32 * 3], spk_k[16 * 16 * 32],
            mem_k[16 * 16 * 32], prev_k[16 * 16 * 32], im2col[2 * 6 * 6 * 32];
        static int16_t spk_r[16 * 16 * 32], mem_r[16 * 16 * 32], prev_r[16 * 16 * 32];
        static __attribute__((aligned(4))) int16_t in2[16 * 16 * 32], spk2_k[8 * 8 * 64],
            mem2_k[8 * 8 * 64], prev2_k[8 * 8 * 64];
        static int16_t spk2_r[8 * 8 * 64], mem2_r[8 * 8 * 64], prev2_r[8 * 8 * 64];

        long long bad = 0;
        const double rates[] = { 0.05, 0.3, 1.0 };
        for (int ri = 0; ri < 3; ++ri) {
            memset(mem_k, 0, sizeof(mem_k)); memset(prev_k, 0, sizeof(prev_k));
            memset(mem_r, 0, sizeof(mem_r)); memset(prev_r, 0, sizeof(prev_r));
            memset(mem2_k, 0, sizeof(mem2_k)); memset(prev2_k, 0, sizeof(prev2_k));
            memset(mem2_r, 0, sizeof(mem2_r)); memset(prev2_r, 0, sizeof(prev2_r));
            for (int t = 0; t < 30; ++t) {
                rand_spikes(in1, 32 * 32 * 3, rates[ri], 32767);
                run_fused(32, 32, 3, 32, in1, snn_conv1_fused_w, b1, mu1, sh1,
                          mem_k, prev_k, snn_lif1_beta, snn_lif1_threshold, SPK1, spk_k, im2col);
                ref_fused_layer(in1, 32, 32, 3, 32, snn_conv1_fused_w, b1, mu1, sh1,
                                mem_r, prev_r, snn_lif1_beta, snn_lif1_threshold, SPK1, spk_r);
                bad += memcmp(spk_k, spk_r, sizeof(spk_r)) != 0;
                bad += memcmp(mem_k, mem_r, sizeof(mem_r)) != 0;
                bad += memcmp(prev_k, prev_r, sizeof(prev_r)) != 0;

                rand_spikes(in2, 16 * 16 * 32, rates[ri], SPK1);
                run_fused(16, 16, 32, 64, in2, snn_conv2_fused_w, b2, mu2, sh2,
                          mem2_k, prev2_k, snn_lif2_beta, snn_lif2_threshold, 32767, spk2_k, im2col);
                ref_fused_layer(in2, 16, 16, 32, 64, snn_conv2_fused_w, b2, mu2, sh2,
                                mem2_r, prev2_r, snn_lif2_beta, snn_lif2_threshold, 32767, spk2_r);
                bad += memcmp(spk2_k, spk2_r, sizeof(spk2_r)) != 0;
                bad += memcmp(mem2_k, mem2_r, sizeof(mem2_r)) != 0;
                bad += memcmp(prev2_k, prev2_r, sizeof(prev2_r)) != 0;
            }
        }
        printf("3) kernel vs naive ref (30 steps x 3 rates): %s (%lld mismatch)\n",
               bad ? "FAIL" : "PASS", bad);
        ok &= !bad;
    }

    /* ---------- 4) 드리프트: 기존 경로 대비 (conv1, 30스텝) ---------- */
    {
        static __attribute__((aligned(4))) int16_t in1[32 * 32 * 3], spk_f[16 * 16 * 32],
            mem_f[16 * 16 * 32], prev_f[16 * 16 * 32], im2col[2 * 6 * 6 * 32];
        static int16_t spk_o[16 * 16 * 32], mem_o[16 * 16 * 32], prev_o[16 * 16 * 32];
        memset(mem_f, 0, sizeof(mem_f)); memset(prev_f, 0, sizeof(prev_f));
        memset(mem_o, 0, sizeof(mem_o)); memset(prev_o, 0, sizeof(prev_o));

        long long spk_diff = 0, n_tot = 0;
        for (int t = 0; t < 30; ++t) {
            rand_spikes(in1, 32 * 32 * 3, 0.45, 32767);
            run_fused(32, 32, 3, 32, in1, snn_conv1_fused_w, b1, mu1, sh1,
                      mem_f, prev_f, snn_lif1_beta, snn_lif1_threshold, SPK1, spk_f, im2col);
            old_path_layer(in1, 32, 32, 3, 32, (const int8_t *)snn_conv1_weight,
                           (const int8_t *)snn_conv1_bias, snn_conv1_bias_scale,
                           m1 * 4.0, mem_o, prev_o,
                           snn_lif1_beta, snn_lif1_threshold, SPK1, spk_o);
            for (int i = 0; i < 16 * 16 * 32; ++i) {
                n_tot++;
                if ((spk_f[i] != 0) != (spk_o[i] != 0)) spk_diff++;
            }
        }
        printf("4) drift vs old path: spike disagree %.4f%%\n", 100.0 * spk_diff / n_tot);
    }

    /* ---------- 5) leftover 경로: Wo 홀수(H=4,W=6) + Co 홀수(5) ---------- */
    {
        enum { H = 4, W = 6, CI = 2, CO = 5, N = (H / 2) * (W / 2) * CO };
        static __attribute__((aligned(4))) int16_t in[H * W * CI], w6[CO * 36 * CI],
            spk_k[N], mem_k[N], prev_k[N], im2col[2 * 36 * CI];
        static int16_t spk_r[N], mem_r[N], prev_r[N];
        int32_t mu[CO], sh[CO];
        int64_t b64[CO];
        for (int c = 0; c < CO; ++c) {
            qmult(m1, &mu[c], &sh[c]);
            b64[c] = (rand() % 20001) - 10000;
        }
        for (int i = 0; i < CO * 36 * CI; ++i) w6[i] = (int16_t)((rand() % 1017) - 508);

        long long bad = 0;
        memset(mem_k, 0, sizeof(mem_k)); memset(prev_k, 0, sizeof(prev_k));
        memset(mem_r, 0, sizeof(mem_r)); memset(prev_r, 0, sizeof(prev_r));
        for (int t = 0; t < 50; ++t) {
            rand_spikes(in, H * W * CI, 0.5, 32767);
            run_fused(H, W, CI, CO, in, w6, b64, mu, sh,
                      mem_k, prev_k, 29491, 32767, 16384, spk_k, im2col);
            ref_fused_layer(in, H, W, CI, CO, w6, b64, mu, sh,
                            mem_r, prev_r, 29491, 32767, 16384, spk_r);
            bad += memcmp(spk_k, spk_r, sizeof(spk_r)) != 0;
            bad += memcmp(mem_k, mem_r, sizeof(mem_r)) != 0;
            bad += memcmp(prev_k, prev_r, sizeof(prev_r)) != 0;
        }
        printf("5) leftover paths (Wo=3, Co=5, 50 steps): %s (%lld mismatch)\n",
               bad ? "FAIL" : "PASS", bad);
        ok &= !bad;
    }

    printf(ok ? "ALL PASS\n" : "SOME FAIL\n");
    return !ok;
}
