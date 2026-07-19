/*
 * SNN forward — 레이어 퓨전 + CMSIS-NN 가속 버전.
 *
 * 타임스텝당 파이프라인 (NHWC):
 *   spiking_rate (snn_encode.c, LUT+xorshift32)
 *   -> [Conv1+Pool1+LIF1 퓨전]  snn_conv_pool_lif_s16 (6x6/s2 SMLALD 커널)
 *   -> [Conv2+Pool2+LIF2 퓨전]  snn_conv_pool_lif_s16
 *   -> FC (CMSIS-NN arm_fully_connected_s16)
 *   -> LIF_out
 *
 * conv/pool 중간 텐서는 존재하지 않는다 (커널 내부 2픽셀 타일뿐).
 * 퓨전 가중치(int16, 5x5 커널 4개의 합)는 빌드타임 생성되어 FLASH에 상주
 * (tools/gen_fused_weights.c 참고).
 *
 * 양자화 매핑 (원본: out_q15 = (acc * ws_q15) >> 15 + b * bs_q15, 이후 /4 평균):
 *   실배율 M_fused = ws_q15/2^15 * (32767/spike_val) / 4
 *   bias64 = round(b * bs_q15 / M_fused)
 *   out = requantize_s64(acc_fused + bias64)  — 반올림 1회라 원본보다 오히려 정밀
 */

#include "snn.h"
#include <string.h> // memset
#include <math.h>

#include "arm_nnfunctions.h"
#include "snn_conv_pool_lif_s16.h"
#include "snn_weights_fused.h"

// LIF1 스파이크 크기: FC와 달리 conv2는 과거 int32 누산 제약으로 Q14(16384)를
// 썼고, SMLALD 전환 후에도 유지한다 (배율 보상은 M2에 포함, 정확도 무손실).
#define SPK1_VAL 16384

// --------- 내부 상태 (LIF, elementwise라 레이아웃 무관 → flat) ---------

// 4바이트 정렬: snn_lif_s16의 SIMD(32비트 페어 로드) 경로 보장
static int16_t g_lif1_mem[32 * 16 * 16] __attribute__((aligned(4)));
static int16_t g_lif1_spk_prev[32 * 16 * 16] __attribute__((aligned(4)));
static int16_t g_lif2_mem[64 * 8 * 8] __attribute__((aligned(4)));
static int16_t g_lif2_spk_prev[64 * 8 * 8] __attribute__((aligned(4)));
static int16_t g_lif_out_mem[10];
static int16_t g_lif_out_spk_prev[10];

// --------- 중간 버퍼: 레이어 간 스파이크 텐서만 남는다 ---------

static int16_t g_spk1[32 * 16 * 16] __attribute__((aligned(4)));  // LIF1 출력 [16][16][32]
static int16_t g_spk2[64 * 8 * 8] __attribute__((aligned(4)));    // LIF2 출력 [8][8][64]

// im2col: 2컬럼 * 6*6*Ci, conv2가 최대 (2*1152)
static int16_t g_im2col_buf[2 * 6 * 6 * 32] __attribute__((aligned(4)));

// --------- 양자화 파라미터 (snn_accel_init에서 계산) ---------

// per-channel 배열 (CMSIS-NN 컨벤션; per-layer 스케일 값을 채널 수만큼 복제)
static int64_t g_bias1[32];
static int32_t g_mult1[32], g_shift1[32];
static int64_t g_bias2[64];
static int32_t g_mult2[64], g_shift2[64];
static int64_t g_biasfc[10];
static int32_t g_multfc, g_shiftfc;

static int g_accel_inited = 0;

// --------- 유틸 ---------

static inline int16_t sat16_from32(int32_t x) {
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

// 실배율 m -> (Q31 multiplier, shift), m = mult/2^31 * 2^shift
static void quantize_mult(double m, int32_t *mult, int32_t *shift) {
    if (m <= 0.0) { *mult = 0; *shift = 0; return; }
    int exp;
    double q = frexp(m, &exp);          // m = q * 2^exp, q in [0.5, 1)
    int64_t q31 = llround(q * 2147483648.0);
    if (q31 == 2147483648LL) { q31 >>= 1; exp++; }
    *mult = (int32_t)q31;
    *shift = exp;
}

// --------- 가속 초기화: 양자화 파라미터 (가중치는 빌드타임 생성, FLASH) ---------

void snn_accel_init(void) {
    if (g_accel_inited) return;

    // 퓨전 배율: conv 배율 * (32767/스파이크값) / 4(avgpool)
    const double m1 = ((double)snn_conv1_weight_scale / 32768.0) / 4.0;
    const double m2 = ((double)snn_conv2_weight_scale / 32768.0) *
                      ((double)ONE_Q15 / (double)SPK1_VAL) / 4.0;
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
    for (int o = 0; o < 10; ++o) {
        g_biasfc[o] = llround((double)snn_fc_bias[o] * (double)snn_fc_bias_scale / mf);
    }

    g_accel_inited = 1;
}

// 입력 스파이크 인코딩(spiking_rate)은 snn_encode.c (LUT + xorshift32 가속 버전)

// --------- LIF_out: in[10], out spk_prev[10], mem_out[10] ---------

static void lif_out_step(const int16_t in[10],
                         int16_t spk_out[10],
                         int16_t mem_out[10]) {
    for (int i = 0; i < 10; ++i) {
        int16_t mem_prev = g_lif_out_mem[i];
        int16_t spk_prev = g_lif_out_spk_prev[i];

        int32_t tmp = ((int32_t)snn_lif_out_beta * (int32_t)mem_prev) >> Q15_SHIFT;
        int32_t mem_tilde = tmp + in[i];
        int16_t mem_tilde_q15 = sat16_from32(mem_tilde);

        int16_t spk_raw_q15 = 0;
        int16_t mem_next_q15 = mem_tilde_q15;

        if (mem_tilde_q15 >= snn_lif_out_threshold) {
            spk_raw_q15 = ONE_Q15;
            mem_next_q15 = mem_tilde_q15 - snn_lif_out_threshold;
        }

        spk_out[i] = spk_prev;
        mem_out[i] = mem_next_q15;

        g_lif_out_spk_prev[i] = spk_raw_q15;
        g_lif_out_mem[i]      = mem_next_q15;
    }
}

// --------- 상태 리셋 ---------

void snn_reset_state(void) {
    memset(g_lif1_mem,         0, sizeof(g_lif1_mem));
    memset(g_lif1_spk_prev,    0, sizeof(g_lif1_spk_prev));
    memset(g_lif2_mem,         0, sizeof(g_lif2_mem));
    memset(g_lif2_spk_prev,    0, sizeof(g_lif2_spk_prev));
    memset(g_lif_out_mem,      0, sizeof(g_lif_out_mem));
    memset(g_lif_out_spk_prev, 0, sizeof(g_lif_out_spk_prev));
}

// --------- 전체 한 step forward (B=1, NHWC) ---------

void snn_forward_step(const int16_t* x,
                      int16_t* spk_out,
                      int16_t* mem_out) {
    if (!g_accel_inited) snn_accel_init();

    const cmsis_nn_context ctx = { .buf = g_im2col_buf, .size = sizeof(g_im2col_buf) };
    // 퓨전 6x6 커널, stride 2 (= 원본 conv 5x5/pad2 + avgpool 2x2/s2)
    const cmsis_nn_conv_params conv_params = {
        .input_offset  = 0,
        .output_offset = 0,
        .stride        = { .w = 2, .h = 2 },
        .padding       = { .w = 2, .h = 2 },
        .dilation      = { .w = 1, .h = 1 },
        .activation    = { .min = -32768, .max = 32767 },
    };
    const cmsis_nn_dims bias_d = { 0, 0, 0, 0 }; // 미사용

    // Conv1+Pool1+LIF1: x [32][32][3] -> g_spk1 [16][16][32] (0 또는 SPK1_VAL)
    {
        const cmsis_nn_per_channel_quant_params qp = { g_mult1, g_shift1 };
        const cmsis_nn_dims in_d  = { .n = 1,  .h = 32, .w = 32, .c = 3 };
        const cmsis_nn_dims flt_d = { .n = 32, .h = 6,  .w = 6,  .c = 3 };
        const cmsis_nn_dims out_d = { .n = 1,  .h = 16, .w = 16, .c = 32 };
        const snn_lif_ctx lif1 = { g_lif1_mem, g_lif1_spk_prev,
                                   snn_lif1_beta, snn_lif1_threshold, SPK1_VAL };
        snn_convolve_pool_lif_s16(&ctx, &conv_params, &qp,
                                  &in_d, x, &flt_d, snn_conv1_fused_w,
                                  &bias_d, g_bias1, &lif1, &out_d, g_spk1);
    }

    // Conv2+Pool2+LIF2: g_spk1 -> g_spk2 [8][8][64] (0 또는 ONE_Q15)
    {
        const cmsis_nn_per_channel_quant_params qp = { g_mult2, g_shift2 };
        const cmsis_nn_dims in_d  = { .n = 1,  .h = 16, .w = 16, .c = 32 };
        const cmsis_nn_dims flt_d = { .n = 64, .h = 6,  .w = 6,  .c = 32 };
        const cmsis_nn_dims out_d = { .n = 1,  .h = 8,  .w = 8,  .c = 64 };
        const snn_lif_ctx lif2 = { g_lif2_mem, g_lif2_spk_prev,
                                   snn_lif2_beta, snn_lif2_threshold, ONE_Q15 };
        snn_convolve_pool_lif_s16(&ctx, &conv_params, &qp,
                                  &in_d, g_spk1, &flt_d, snn_conv2_fused_w,
                                  &bias_d, g_bias2, &lif2, &out_d, g_spk2);
    }

    // FC: g_spk2 (4096) -> fc_out[10]
    // (arm_fully_connected_s16은 내부적으로 512컬럼까지만 int32 누산, 이후 int64 → 4096 안전)
    int16_t fc_out[10];
    {
        const cmsis_nn_context ctx = { .buf = NULL, .size = 0 };
        const cmsis_nn_fc_params fc_params = {
            .input_offset  = 0,
            .filter_offset = 0,
            .output_offset = 0,
            .activation    = { .min = -32768, .max = 32767 },
        };
        const cmsis_nn_per_tensor_quant_params qp = { g_multfc, g_shiftfc };
        const cmsis_nn_dims in_d   = { .n = 1,    .h = 1, .w = 1, .c = 4096 };
        const cmsis_nn_dims flt_d  = { .n = 4096, .h = 1, .w = 1, .c = 10 };
        const cmsis_nn_dims out_d  = { .n = 1,    .h = 1, .w = 1, .c = 10 };
        const cmsis_nn_dims bias_d = { 0, 0, 0, 0 };
        arm_fully_connected_s16(&ctx, &fc_params, &qp,
                                &in_d, g_spk2, &flt_d, snn_fc_weight_hwc,
                                &bias_d, g_biasfc, &out_d, fc_out);
    }

    // LIF_out: fc_out[10] -> spk_out[10], mem_out[10]
    lif_out_step(fc_out, spk_out, mem_out);
}
