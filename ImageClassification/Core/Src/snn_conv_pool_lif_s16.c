/*
 * Conv+AvgPool+LIF 퓨전 커널 — arm_convolve_fast_s16.c(CMSIS-NN, Apache-2.0,
 * Copyright 2010-2023 Arm Limited)의 im2col/matmul 구조를 따르고, CMSIS-NN
 * 지원 함수(arm_nn_requantize_s64, REDUCE_MULTIPLIER, arm_memcpy_s8/arm_memset_s8,
 * arm_nn_read_q15x2_ia)를 사용한다.
 *
 * CMSIS-NN 자체로 커버되지 않는 부분 두 가지만 직접 구현한다:
 *  1) s16 가중치 x s16 활성값 내적 — CMSIS-NN의 s16 커널은 int8 가중치 전용
 *     이라 퓨전 가중치(|값| <= 508)를 담을 수 없다. __SMLALD(듀얼 16x16 MAC,
 *     64비트 누산, CMSIS 코어 인트린식)로 처리해 오버플로를 원천 차단한다.
 *  2) LIF 에필로그 — requantize된 2픽셀 x Co 타일에 snn_lif_s16을 즉시 적용.
 *
 * im2col은 2컬럼이 차면 즉시 flush하므로(행 경계와 무관) 출력 픽셀 수가
 * 홀수여도 leftover 1컬럼 경로로 처리된다. 출력 채널 홀수도 tail 처리된다.
 */

#include "snn_conv_pool_lif_s16.h"
#include "snn_lif_s16.h"

#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_DSP)
#include "cmsis_compiler.h"
#endif

/* 컬럼 1~2개 x 채널 1~2개의 내적 (int64 누산) */
static void dot_rows_cols(const int16_t *w_row0, const int16_t *w_row1,
                          const int16_t *col0, const int16_t *col1,
                          int32_t rhs_cols, int64_t acc[4])
{
    int64_t a00 = 0, a01 = 0, a10 = 0, a11 = 0;
#if defined(ARM_MATH_DSP)
    const int16_t *x0 = col0, *x1 = col1, *w0 = w_row0, *w1 = w_row1;
    for (int32_t k = rhs_cols >> 1; k > 0; --k) {
        uint32_t xp0 = arm_nn_read_q15x2_ia(&x0);
        uint32_t xp1 = arm_nn_read_q15x2_ia(&x1);
        uint32_t wp0 = arm_nn_read_q15x2_ia(&w0);
        uint32_t wp1 = arm_nn_read_q15x2_ia(&w1);
        a00 = (int64_t)__SMLALD(wp0, xp0, (uint64_t)a00);
        a01 = (int64_t)__SMLALD(wp0, xp1, (uint64_t)a01);
        a10 = (int64_t)__SMLALD(wp1, xp0, (uint64_t)a10);
        a11 = (int64_t)__SMLALD(wp1, xp1, (uint64_t)a11);
    }
    if (rhs_cols & 1) {
        int32_t k = rhs_cols - 1;
        a00 += (int32_t)w_row0[k] * col0[k];
        a01 += (int32_t)w_row0[k] * col1[k];
        a10 += (int32_t)w_row1[k] * col0[k];
        a11 += (int32_t)w_row1[k] * col1[k];
    }
#else
    for (int32_t k = 0; k < rhs_cols; ++k) {
        a00 += (int32_t)w_row0[k] * col0[k];
        a01 += (int32_t)w_row0[k] * col1[k];
        a10 += (int32_t)w_row1[k] * col0[k];
        a11 += (int32_t)w_row1[k] * col1[k];
    }
#endif
    acc[0] = a00; acc[1] = a01; acc[2] = a10; acc[3] = a11;
}

/*
 * 컬럼 n_cols(1|2)개에 대한 matmul + requantize + LIF.
 * out은 스파이크 출력 버퍼 내 현재 위치, base = out - output_start.
 */
static int16_t *mat_mult_lif(const int16_t *filter, const int16_t *cols,
                             int32_t n_cols, int32_t output_ch, int32_t rhs_cols,
                             const int32_t *output_mult, const int32_t *output_shift,
                             const int64_t *bias, int32_t act_min, int32_t act_max,
                             const snn_lif_ctx *lif, int32_t base, int16_t *out)
{
    /* 픽셀 2개 x 최대 64채널 타일 (헤더 제약: output_ch <= 64) */
    int16_t tile[2 * 64] __attribute__((aligned(4)));
    const int16_t *col0 = cols;
    const int16_t *col1 = (n_cols == 2) ? cols + rhs_cols : cols; /* 1컬럼이면 중복 계산 후 버림 */

    int32_t co = 0;
    for (; co + 1 < output_ch; co += 2) {
        int64_t acc[4];
        dot_rows_cols(&filter[co * rhs_cols], &filter[(co + 1) * rhs_cols],
                      col0, col1, rhs_cols, acc);
        const int32_t rm0 = REDUCE_MULTIPLIER(output_mult[co]);
        const int32_t rm1 = REDUCE_MULTIPLIER(output_mult[co + 1]);
        int32_t v;
        v = arm_nn_requantize_s64(acc[0] + bias[co], rm0, output_shift[co]);
        tile[co] = (int16_t)MIN(MAX(v, act_min), act_max);
        v = arm_nn_requantize_s64(acc[1] + bias[co], rm0, output_shift[co]);
        tile[output_ch + co] = (int16_t)MIN(MAX(v, act_min), act_max);
        v = arm_nn_requantize_s64(acc[2] + bias[co + 1], rm1, output_shift[co + 1]);
        tile[co + 1] = (int16_t)MIN(MAX(v, act_min), act_max);
        v = arm_nn_requantize_s64(acc[3] + bias[co + 1], rm1, output_shift[co + 1]);
        tile[output_ch + co + 1] = (int16_t)MIN(MAX(v, act_min), act_max);
    }
    if (co < output_ch) { /* 출력 채널 홀수 tail */
        int64_t acc[4];
        dot_rows_cols(&filter[co * rhs_cols], &filter[co * rhs_cols],
                      col0, col1, rhs_cols, acc);
        const int32_t rm = REDUCE_MULTIPLIER(output_mult[co]);
        int32_t v;
        v = arm_nn_requantize_s64(acc[0] + bias[co], rm, output_shift[co]);
        tile[co] = (int16_t)MIN(MAX(v, act_min), act_max);
        v = arm_nn_requantize_s64(acc[1] + bias[co], rm, output_shift[co]);
        tile[output_ch + co] = (int16_t)MIN(MAX(v, act_min), act_max);
    }

    snn_lif_s16(n_cols * output_ch, tile, out,
                &lif->mem[base], &lif->spk_prev[base],
                lif->beta, lif->threshold, lif->spike_val);
    return out + n_cols * output_ch;
}

arm_cmsis_nn_status snn_convolve_pool_lif_s16(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int64_t *bias_data,
                                              const snn_lif_ctx *lif,
                                              const cmsis_nn_dims *output_dims,
                                              int16_t *output_data)
{
    (void)bias_dims;

    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t rhs_cols = input_ch * kernel_y * kernel_x;

    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;

    if (ctx->buf == NULL || input_dims->n != 1 || output_ch > 64) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int16_t *buffer_a = (int16_t *)ctx->buf;

    /* im2col: 2컬럼이 차면 즉시 flush (arm_convolve_fast_s16과 동일 구조) */
    int16_t *two_column_buf = buffer_a;
    int16_t *out = output_data;
    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
            for (int32_t i_ker_y = i_out_y * stride_y - pad_y;
                 i_ker_y < i_out_y * stride_y - pad_y + kernel_y; i_ker_y++) {
                for (int32_t i_ker_x = i_out_x * stride_x - pad_x;
                     i_ker_x < i_out_x * stride_x - pad_x + kernel_x; i_ker_x++) {
                    if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x) {
                        /* 패딩: 0 채움 (원본 conv pad와 동일) */
                        arm_memset_s8((int8_t *)two_column_buf, 0, sizeof(int16_t) * input_ch);
                    } else {
                        arm_memcpy_s8((int8_t *)two_column_buf,
                                      (const int8_t *)(input_data +
                                                       (i_ker_y * input_x + i_ker_x) * input_ch),
                                      input_ch * sizeof(int16_t));
                    }
                    two_column_buf += input_ch;
                }
            }
            if (two_column_buf == buffer_a + 2 * rhs_cols) {
                out = mat_mult_lif(filter_data, buffer_a, 2, output_ch, rhs_cols,
                                   quant_params->multiplier, quant_params->shift,
                                   bias_data, act_min, act_max,
                                   lif, (int32_t)(out - output_data), out);
                two_column_buf = buffer_a;
            }
        }
    }

    /* 출력 픽셀 수 홀수 leftover: 1컬럼 */
    if (two_column_buf != buffer_a) {
        out = mat_mult_lif(filter_data, buffer_a, 1, output_ch, rhs_cols,
                           quant_params->multiplier, quant_params->shift,
                           bias_data, act_min, act_max,
                           lif, (int32_t)(out - output_data), out);
    }

    return ARM_CMSIS_NN_SUCCESS;
}
