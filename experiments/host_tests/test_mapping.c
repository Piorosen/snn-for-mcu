/*
 * 호스트 검증: 원본 Q15 conv/FC (CHW, 절단 시프트) vs
 * CMSIS-NN 시맨틱 (NHWC, int32 누산 + bias64 + requantize_s64) 비교.
 *
 * snn.c의 quantize_mult / bias64 매핑과 동일한 코드를 사용해
 * 실제 스케일 값(231/349, 534/697, 639/256)으로 랜덤 스파이크 입력을 검증한다.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define ONE_Q15 32767
#define Q15_SHIFT 15
#define SPK1_VAL 16384

static inline int16_t sat16_from32(int32_t x) {
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

/* ---------- 원본 레퍼런스 (snn.c 이전 버전과 동일) ---------- */
static void ref_conv2d_5x5_pad2_q15(
    int C_in, int C_out, int H, int W,
    const int16_t *in, int16_t *out,
    const int8_t *weight, int16_t weight_scale,
    const int8_t *bias, int16_t bias_scale)
{
    for (int co = 0; co < C_out; ++co)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int64_t sum_w_q15 = 0;
                for (int ci = 0; ci < C_in; ++ci)
                    for (int kh = -2; kh <= 2; ++kh) {
                        int ih = h + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = -2; kw <= 2; ++kw) {
                            int iw = w + kw;
                            if (iw < 0 || iw >= W) continue;
                            int in_idx = (ci * H + ih) * W + iw;
                            int w_idx = (((co * C_in) + ci) * 5 + kh + 2) * 5 + kw + 2;
                            sum_w_q15 += (int32_t)in[in_idx] * (int32_t)weight[w_idx];
                        }
                    }
                int64_t tmp = sum_w_q15 * (int64_t)weight_scale;
                int32_t y_w_q15 = (int32_t)(tmp >> Q15_SHIFT);
                int32_t y_b_q15 = (int32_t)bias[co] * (int32_t)bias_scale;
                out[(co * H + h) * W + w] = sat16_from32(y_w_q15 + y_b_q15);
            }
}

static void ref_fc(int IN, int OUT, const int16_t *in, int16_t *out,
                   const int8_t *w, int16_t ws, const int8_t *b, int16_t bs)
{
    for (int o = 0; o < OUT; ++o) {
        int64_t sum = 0;
        for (int i = 0; i < IN; ++i)
            sum += (int32_t)in[i] * (int32_t)w[o * IN + i];
        int32_t y_w = (int32_t)((sum * (int64_t)ws) >> Q15_SHIFT);
        int32_t y_b = (int32_t)b[o] * (int32_t)bs;
        out[o] = sat16_from32(y_w + y_b);
    }
}

/* ---------- CMSIS-NN requantize 시맨틱 (arm_nnsupportfunctions.h와 동일) ---------- */
#define REDUCE_MULTIPLIER(m) (((m) < 0x7FFF0000) ? (((m) + (1 << 15)) >> 16) : 0x7FFF)

static int32_t requantize_s64(int64_t val, int32_t reduced_mult, int32_t shift) {
    const int64_t new_val = val * reduced_mult;
    int32_t result = (int32_t)(new_val >> (14 - shift));
    result = (result + 1) >> 1;
    return result;
}

/* ---------- snn.c와 동일한 매핑 ---------- */
static void quantize_mult(double m, int32_t *mult, int32_t *shift) {
    if (m <= 0.0) { *mult = 0; *shift = 0; return; }
    int exp;
    double q = frexp(m, &exp);
    int64_t q31 = llround(q * 2147483648.0);
    if (q31 == 2147483648LL) { q31 >>= 1; exp++; }
    *mult = (int32_t)q31;
    *shift = exp;
}

/* CMSIS 커널 시뮬레이션: NHWC conv, int32 누산 + bias64 + requantize_s64 */
static void sim_cmsis_conv_nhwc(
    int C_in, int C_out, int H, int W,
    const int16_t *in_hwc, int16_t *out_hwc,
    const int8_t *w_ohwi, const int64_t *bias64,
    int32_t mult, int32_t shift, int64_t *max_abs_acc)
{
    int32_t rm = REDUCE_MULTIPLIER(mult);
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int co = 0; co < C_out; ++co) {
                int32_t acc = 0;
                for (int kh = -2; kh <= 2; ++kh) {
                    int ih = h + kh;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = -2; kw <= 2; ++kw) {
                        int iw = w + kw;
                        if (iw < 0 || iw >= W) continue;
                        for (int ci = 0; ci < C_in; ++ci) {
                            int64_t chk = (int64_t)acc +
                                (int32_t)in_hwc[(ih * W + iw) * C_in + ci] *
                                (int32_t)w_ohwi[((co * 5 + kh + 2) * 5 + kw + 2) * C_in + ci];
                            if (llabs(chk) > *max_abs_acc) *max_abs_acc = llabs(chk);
                            acc = (int32_t)chk; /* int32 누산 시뮬레이션 */
                        }
                    }
                }
                int32_t r = requantize_s64((int64_t)acc + bias64[co], rm, shift);
                if (r > 32767) r = 32767;
                if (r < -32768) r = -32768;
                out_hwc[(h * W + w) * C_out + co] = (int16_t)r;
            }
}

static int rand_i8(void) { return (rand() % 255) - 127; }

static int test_conv(const char *name, int C_in, int C_out, int H, int W,
                     int16_t ws, int16_t bs, int spike_val_cand, double spike_rate)
{
    int N_in = C_in * H * W, N_out = C_out * H * W;
    int8_t  *w_chw  = malloc(C_out * C_in * 25);
    int8_t  *w_ohwi = malloc(C_out * C_in * 25);
    int8_t  *bias   = malloc(C_out);
    int16_t *in_chw = malloc(N_in * 2), *in_hwc = malloc(N_in * 2);
    int16_t *out_ref = malloc(N_out * 2), *out_cand = malloc(N_out * 2);
    int64_t *bias64 = malloc(C_out * 8);

    for (int i = 0; i < C_out * C_in * 25; ++i) w_chw[i] = rand_i8();
    for (int i = 0; i < C_out; ++i) bias[i] = rand_i8();

    /* snn.c와 동일한 퍼뮤트 [Co][Ci][Kh][Kw] -> [Co][Kh][Kw][Ci] */
    for (int co = 0; co < C_out; ++co)
        for (int ci = 0; ci < C_in; ++ci)
            for (int kh = 0; kh < 5; ++kh)
                for (int kw = 0; kw < 5; ++kw)
                    w_ohwi[((co * 5 + kh) * 5 + kw) * C_in + ci] =
                        w_chw[((co * C_in + ci) * 5 + kh) * 5 + kw];

    /* 랜덤 스파이크: 레퍼런스는 ONE_Q15, 후보는 spike_val_cand */
    for (int c = 0; c < C_in; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int spk = ((double)rand() / RAND_MAX) < spike_rate;
                in_chw[(c * H + h) * W + w] = spk ? ONE_Q15 : 0;
                in_hwc[(h * W + w) * C_in + c] = spk ? spike_val_cand : 0;
            }

    /* snn.c와 동일한 양자화 파라미터 */
    double m = ((double)ws / 32768.0) * ((double)ONE_Q15 / (double)spike_val_cand);
    int32_t mult, shift;
    quantize_mult(m, &mult, &shift);
    for (int c = 0; c < C_out; ++c)
        bias64[c] = llround((double)bias[c] * (double)bs / m);

    ref_conv2d_5x5_pad2_q15(C_in, C_out, H, W, in_chw, out_ref, w_chw, ws, bias, bs);

    int64_t max_abs_acc = 0;
    sim_cmsis_conv_nhwc(C_in, C_out, H, W, in_hwc, out_cand, w_ohwi, bias64, mult, shift, &max_abs_acc);

    int max_diff = 0;
    long long sum_diff = 0;
    for (int co = 0; co < C_out; ++co)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int d = abs(out_ref[(co * H + h) * W + w] - out_cand[(h * W + w) * C_out + co]);
                if (d > max_diff) max_diff = d;
                sum_diff += d;
            }
    printf("%s: max_diff=%d LSB, mean_diff=%.4f, max|acc32|=%lld (INT32_MAX=%d) %s\n",
           name, max_diff, (double)sum_diff / N_out, (long long)max_abs_acc, INT32_MAX,
           (max_diff <= 2 && max_abs_acc < INT32_MAX) ? "PASS" : "FAIL");

    free(w_chw); free(w_ohwi); free(bias); free(in_chw); free(in_hwc);
    free(out_ref); free(out_cand); free(bias64);
    return max_diff <= 2 && max_abs_acc < INT32_MAX;
}

static int test_fc(void)
{
    enum { IN = 4096, OUT = 10, C = 64, HW = 8 };
    static int8_t w[OUT * IN], w_perm[OUT * IN], bias[OUT];
    static int16_t in_chw[IN], in_hwc[IN], out_ref[OUT], out_cand[OUT];
    int16_t ws = 639, bs = 256;

    for (int i = 0; i < OUT * IN; ++i) w[i] = rand_i8();
    for (int i = 0; i < OUT; ++i) bias[i] = rand_i8();

    /* snn.c와 동일한 FC 퍼뮤트 */
    for (int o = 0; o < OUT; ++o)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < HW; ++h)
                for (int wd = 0; wd < HW; ++wd)
                    w_perm[o * IN + (h * HW + wd) * C + c] = w[o * IN + c * 64 + h * HW + wd];

    for (int c = 0; c < C; ++c)
        for (int h = 0; h < HW; ++h)
            for (int wd = 0; wd < HW; ++wd) {
                int spk = ((double)rand() / RAND_MAX) < 0.3;
                in_chw[c * 64 + h * HW + wd] = spk ? ONE_Q15 : 0;
                in_hwc[(h * HW + wd) * C + c] = spk ? ONE_Q15 : 0;
            }

    double m = (double)ws / 32768.0;
    int32_t mult, shift;
    quantize_mult(m, &mult, &shift);
    int32_t rm = REDUCE_MULTIPLIER(mult);

    ref_fc(IN, OUT, in_chw, out_ref, w, ws, bias, bs);

    /* arm_nn_vec_mat_mult_t_s16 시맨틱: int64 누산(512 초과분) + requantize_s64 */
    for (int o = 0; o < OUT; ++o) {
        int64_t acc = 0;
        for (int i = 0; i < IN; ++i)
            acc += (int32_t)in_hwc[i] * (int32_t)w_perm[o * IN + i];
        int64_t bias64 = llround((double)bias[o] * (double)bs / m);
        int32_t r = requantize_s64(acc + bias64, rm, shift);
        if (r > 32767) r = 32767;
        if (r < -32768) r = -32768;
        out_cand[o] = (int16_t)r;
    }

    int max_diff = 0;
    for (int o = 0; o < OUT; ++o) {
        int d = abs(out_ref[o] - out_cand[o]);
        if (d > max_diff) max_diff = d;
    }
    printf("fc: max_diff=%d LSB %s\n", max_diff, max_diff <= 2 ? "PASS" : "FAIL");
    return max_diff <= 2;
}

int main(void)
{
    srand(12345);
    int ok = 1;
    /* conv1: 실제 스케일 ws=231 bs=349, 입력 스파이크 밀도 다양하게 */
    ok &= test_conv("conv1 (r=0.3)", 3, 32, 32, 32, 231, 349, ONE_Q15, 0.3);
    ok &= test_conv("conv1 (r=1.0)", 3, 32, 32, 32, 231, 349, ONE_Q15, 1.0);
    /* conv2: 실제 스케일 ws=534 bs=697, 후보는 SPK1_VAL=16384 + 배율 보상 */
    ok &= test_conv("conv2 (r=0.3)", 32, 64, 16, 16, 534, 697, SPK1_VAL, 0.3);
    ok &= test_conv("conv2 (r=1.0, worst)", 32, 64, 16, 16, 534, 697, SPK1_VAL, 1.0);
    ok &= test_fc();
    printf(ok ? "ALL PASS\n" : "SOME FAIL\n");
    return !ok;
}
