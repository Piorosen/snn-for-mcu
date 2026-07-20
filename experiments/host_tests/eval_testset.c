/*
 * 정수(배포 퓨전) 파이프라인의 CIFAR-10 테스트셋 평가 드라이버 (실험 E1~E3).
 *
 * 디바이스와 동일한 실제 커널(snn_conv_pool_lif_s16, snn_lif_s16, snn_encode,
 * CMSIS-NN FC)을 링크하고, snn.c와 동일한 양자화 파라미터를 구성해 실행한다.
 * (이 조합의 디바이스 동치성은 experiments/host_tests의 H4/H5가 비트 단위로 검증)
 *
 * 측정:
 *  E1: T=30 정수 정확도
 *  E2: T=1..30 누적 투표 체크포인트별 정확도 곡선 (단일 패스)
 *  E3: 레이어별 평균 스파이크 발화율 (입력/LIF1/LIF2)
 *  옵션 --unfused: 비퓨전 정수 시맨틱(픽셀별 requantize+풀링 반올림) 경로를
 *                  함께 실행해 T=30 최종 예측 일치율 측정 (실험 E5)
 *
 * 사용: eval_testset <testset_hwc.bin> <start> <count> [--unfused]
 * 출력(stdout 1줄): RESULT n=<n> correct30=<c> curve=<c1,...,c30>
 *                   spikes_in=<..> spikes_l1=<..> spikes_l2=<..> steps=<..>
 *                   [agree=<a> unfused30=<c>]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "snn.h"
#include "snn_conv_pool_lif_s16.h"
#include "snn_lif_s16.h"
#include "snn_weights_fused.h"
#include "arm_nnfunctions.h"

#define T_STEPS 30

/* ---------- snn.c와 동일한 양자화 파라미터 구성 ---------- */
static int64_t g_bias1[32], g_bias2[64], g_biasfc[10];
static int32_t g_mult1[32], g_shift1[32], g_mult2[64], g_shift2[64], g_multfc, g_shiftfc;

static void quantize_mult(double m, int32_t *mult, int32_t *shift) {
    if (m <= 0.0) { *mult = 0; *shift = 0; return; }
    int exp;
    double q = frexp(m, &exp);
    int64_t q31 = llround(q * 2147483648.0);
    if (q31 == 2147483648LL) { q31 >>= 1; exp++; }
    *mult = (int32_t)q31; *shift = exp;
}

static void init_params(void) {
    const double m1 = ((double)snn_conv1_weight_scale / 32768.0) / 4.0;
    const double m2 = ((double)snn_conv2_weight_scale / 32768.0) * ((double)ONE_Q15 / 32767.0) / 4.0;
    const double mf = (double)snn_fc_weight_scale / 32768.0;
    for (int c = 0; c < 32; ++c) {
        quantize_mult(m1, &g_mult1[c], &g_shift1[c]);
        g_bias1[c] = llround((double)snn_conv1_bias[c] * (double)snn_conv1_bias_scale / m1);
    }
    for (int c = 0; c < 64; ++c) {
        quantize_mult(m2, &g_mult2[c], &g_shift2[c]);
        g_bias2[c] = llround((double)snn_conv2_bias[c] * (double)snn_conv2_bias_scale / m2);
    }
    quantize_mult(mf, &g_multfc, &g_shiftfc);
    for (int o = 0; o < 10; ++o)
        g_biasfc[o] = llround((double)snn_fc_bias[o] * (double)snn_fc_bias_scale / mf);
}

/* ---------- 퓨전 경로 상태/버퍼 ---------- */
static int16_t spikes[3072] __attribute__((aligned(4)));
static int16_t spk1[8192] __attribute__((aligned(4))), spk2[4096] __attribute__((aligned(4)));
static int16_t l1_mem[8192] __attribute__((aligned(4))), l1_prev[8192] __attribute__((aligned(4)));
static int16_t l2_mem[4096] __attribute__((aligned(4))), l2_prev[4096] __attribute__((aligned(4)));
static int16_t lo_mem[10] __attribute__((aligned(4))), lo_prev[10] __attribute__((aligned(4)));
static int16_t im2col_buf[2 * 6 * 6 * 32] __attribute__((aligned(4)));

static void fused_step(const int16_t *x, int16_t *spk_out) {
    const cmsis_nn_context ctx = { .buf = im2col_buf, .size = sizeof(im2col_buf) };
    const cmsis_nn_conv_params cp = {
        .input_offset = 0, .output_offset = 0,
        .stride = { .w = 2, .h = 2 }, .padding = { .w = 2, .h = 2 },
        .dilation = { .w = 1, .h = 1 }, .activation = { .min = -32768, .max = 32767 },
    };
    const cmsis_nn_dims bias_d = { 0, 0, 0, 0 };
    {
        const cmsis_nn_per_channel_quant_params qp = { g_mult1, g_shift1 };
        const cmsis_nn_dims in_d = { 1, 32, 32, 3 }, flt_d = { 32, 6, 6, 3 }, out_d = { 1, 16, 16, 32 };
        const snn_lif_ctx lif = { l1_mem, l1_prev, snn_lif1_beta, snn_lif1_threshold, ONE_Q15 };
        snn_convolve_pool_lif_s16(&ctx, &cp, &qp, &in_d, x, &flt_d, snn_conv1_fused_w,
                                  &bias_d, g_bias1, &lif, &out_d, spk1);
    }
    {
        const cmsis_nn_per_channel_quant_params qp = { g_mult2, g_shift2 };
        const cmsis_nn_dims in_d = { 1, 16, 16, 32 }, flt_d = { 64, 6, 6, 32 }, out_d = { 1, 8, 8, 64 };
        const snn_lif_ctx lif = { l2_mem, l2_prev, snn_lif2_beta, snn_lif2_threshold, ONE_Q15 };
        snn_convolve_pool_lif_s16(&ctx, &cp, &qp, &in_d, spk1, &flt_d, snn_conv2_fused_w,
                                  &bias_d, g_bias2, &lif, &out_d, spk2);
    }
    int16_t fc_out[10];
    {
        const cmsis_nn_context fctx = { .buf = NULL, .size = 0 };
        const cmsis_nn_fc_params fp = { .input_offset = 0, .filter_offset = 0,
                                        .output_offset = 0, .activation = { -32768, 32767 } };
        const cmsis_nn_per_tensor_quant_params qp = { g_multfc, g_shiftfc };
        const cmsis_nn_dims in_d = { 1, 1, 1, 4096 }, flt_d = { 4096, 1, 1, 10 },
                            out_d = { 1, 1, 1, 10 }, b_d = { 0, 0, 0, 0 };
        arm_fully_connected_s16(&fctx, &fp, &qp, &in_d, spk2, &flt_d, snn_fc_weight_hwc,
                                &b_d, g_biasfc, &out_d, fc_out);
    }
    snn_lif_s16(10, fc_out, spk_out, lo_mem, lo_prev,
                snn_lif_out_beta, snn_lif_out_threshold, ONE_Q15);
}

/* ---------- 비퓨전 정수 시맨틱 (실험 E5용 레퍼런스, test_fused와 동일) ---------- */
#define RED_MULT(m) (((m) < 0x7FFF0000) ? (((m) + (1 << 15)) >> 16) : 0x7FFF)
static int32_t requant64(int64_t v, int32_t rm, int32_t sh) {
    int32_t r = (int32_t)((v * rm) >> (14 - sh));
    return (r + 1) >> 1;
}
static int16_t clamp16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v; }

static void ref_lif(int n, const int16_t *in, int16_t *spk_out, int16_t *mem, int16_t *prev,
                    int16_t beta, int16_t thr) {
    for (int i = 0; i < n; ++i) {
        int32_t t = ((int32_t)beta * mem[i]) >> 15;
        int16_t mt = clamp16(t + in[i]);
        int16_t sr = 0, mn = mt;
        if (mt >= thr) { sr = ONE_Q15; mn = (int16_t)(mt - thr); }
        spk_out[i] = prev[i]; prev[i] = sr; mem[i] = mn;
    }
}

static void unfused_layer(const int16_t *in, int H, int W, int Ci, int Co,
                          const int8_t *w5, const int8_t *b8, int bs, int16_t ws,
                          int16_t *mem, int16_t *prev, int16_t beta, int16_t thr,
                          int16_t *spk_out, int16_t *convout /* H*W*Co */) {
    int32_t mult, shift;
    quantize_mult((double)ws / 32768.0, &mult, &shift);
    int32_t rm = RED_MULT(mult);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            for (int co = 0; co < Co; ++co) {
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
                int64_t b64 = llround((double)b8[co] * bs * 32768.0 / (double)ws);
                convout[(r * W + c) * Co + co] = clamp16(requant64(acc + b64, rm, shift));
            }
    int Ho = H / 2, Wo = W / 2;
    for (int oh = 0; oh < Ho; ++oh)
        for (int ow = 0; ow < Wo; ++ow)
            for (int co = 0; co < Co; ++co) {
                int32_t s = convout[((2 * oh) * W + 2 * ow) * Co + co]
                          + convout[((2 * oh) * W + 2 * ow + 1) * Co + co]
                          + convout[((2 * oh + 1) * W + 2 * ow) * Co + co]
                          + convout[((2 * oh + 1) * W + 2 * ow + 1) * Co + co];
                /* pooled를 convout 앞부분에 재기록 (Ho*Wo*Co <= H*W*Co) */
                convout[(oh * Wo + ow) * Co + co] = clamp16((s + ((s >= 0) ? 2 : -2)) >> 2);
            }
    ref_lif(Ho * Wo * Co, convout, spk_out, mem, prev, beta, thr);
}

static int16_t u_spk1[8192], u_spk2[4096];
static int16_t u1_mem[8192], u1_prev[8192], u2_mem[4096], u2_prev[4096], uo_mem[10], uo_prev[10];
static int16_t u_conv[32 * 32 * 32];

static void unfused_step(const int16_t *x, int16_t *spk_out) {
    unfused_layer(x, 32, 32, 3, 32, (const int8_t *)snn_conv1_weight,
                  (const int8_t *)snn_conv1_bias, snn_conv1_bias_scale, snn_conv1_weight_scale,
                  u1_mem, u1_prev, snn_lif1_beta, snn_lif1_threshold, u_spk1, u_conv);
    unfused_layer(u_spk1, 16, 16, 32, 64, (const int8_t *)snn_conv2_weight,
                  (const int8_t *)snn_conv2_bias, snn_conv2_bias_scale, snn_conv2_weight_scale,
                  u2_mem, u2_prev, snn_lif2_beta, snn_lif2_threshold, u_spk2, u_conv);
    int16_t fc_out[10];
    for (int o = 0; o < 10; ++o) { /* FC: int64 누산 + 동일 requantize (test_fused와 동일) */
        int64_t acc = 0;
        for (int i = 0; i < 4096; ++i)
            acc += (int32_t)u_spk2[i] * (int32_t)snn_fc_weight_hwc[o * 4096 + i];
        int32_t mult, shift;
        quantize_mult((double)snn_fc_weight_scale / 32768.0, &mult, &shift);
        fc_out[o] = clamp16(requant64(acc + g_biasfc[o], RED_MULT(mult), shift));
    }
    ref_lif(10, fc_out, spk_out, uo_mem, uo_prev, snn_lif_out_beta, snn_lif_out_threshold);
}

/* ---------- 메인 ---------- */
static int argmax_votes(const int *v) {
    int p = 0;
    for (int i = 1; i < 10; ++i) if (v[i] > v[p]) p = i; /* main.cc와 동일: 최초 최대 */
    return p;
}
static long long count_nz(const int16_t *v, int n) {
    long long c = 0;
    for (int i = 0; i < n; ++i) c += (v[i] != 0);
    return c;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <bin> <start> <count> [--unfused]\n", argv[0]); return 2; }
    const char *path = argv[1];
    long start = atol(argv[2]), count = atol(argv[3]);
    int do_unfused = (argc > 4 && strcmp(argv[4], "--unfused") == 0);

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    init_params();

    long correct30 = 0, curve[T_STEPS];
    long long sp_in = 0, sp_l1 = 0, sp_l2 = 0, steps = 0;
    long agree = 0, u_correct30 = 0;
    memset(curve, 0, sizeof(curve));
    uint8_t rec[1 + 3072];
    int16_t spk_out[10], u_spk_out[10];

    fseek(f, start * (long)sizeof(rec), SEEK_SET);
    for (long i = 0; i < count; ++i) {
        if (fread(rec, sizeof(rec), 1, f) != 1) { count = i; break; }
        int label = rec[0];
        int votes[10] = { 0 }, u_votes[10] = { 0 };

        memset(l1_mem, 0, sizeof(l1_mem)); memset(l1_prev, 0, sizeof(l1_prev));
        memset(l2_mem, 0, sizeof(l2_mem)); memset(l2_prev, 0, sizeof(l2_prev));
        memset(lo_mem, 0, sizeof(lo_mem)); memset(lo_prev, 0, sizeof(lo_prev));
        if (do_unfused) {
            memset(u1_mem, 0, sizeof(u1_mem)); memset(u1_prev, 0, sizeof(u1_prev));
            memset(u2_mem, 0, sizeof(u2_mem)); memset(u2_prev, 0, sizeof(u2_prev));
            memset(uo_mem, 0, sizeof(uo_mem)); memset(uo_prev, 0, sizeof(uo_prev));
        }

        for (int t = 0; t < T_STEPS; ++t) {
            spiking_rate(rec + 1, spikes, T_STEPS, 1, 3, 32, 32, 1.0f, 0.0f);
            fused_step(spikes, spk_out);
            for (int o = 0; o < 10; ++o) votes[o] += spk_out[o] > 16000;
            curve[t] += (argmax_votes(votes) == label);
            sp_in += count_nz(spikes, 3072);
            sp_l1 += count_nz(spk1, 8192);
            sp_l2 += count_nz(spk2, 4096);
            steps++;
            if (do_unfused) { /* 동일 스파이크 입력을 비퓨전 경로에도 공급 */
                unfused_step(spikes, u_spk_out);
                for (int o = 0; o < 10; ++o) u_votes[o] += u_spk_out[o] > 16000;
            }
        }
        int pred = argmax_votes(votes);
        correct30 += (pred == label);
        if (do_unfused) {
            int up = argmax_votes(u_votes);
            agree += (up == pred);
            u_correct30 += (up == label);
        }
    }
    fclose(f);

    printf("RESULT n=%ld correct30=%ld curve=", count, correct30);
    for (int t = 0; t < T_STEPS; ++t) printf("%ld%s", curve[t], t == T_STEPS - 1 ? "" : ",");
    printf(" spikes_in=%lld spikes_l1=%lld spikes_l2=%lld steps=%lld", sp_in, sp_l1, sp_l2, steps);
    if (do_unfused) printf(" agree=%ld unfused30=%ld", agree, u_correct30);
    printf("\n");
    return 0;
}
