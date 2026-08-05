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
 *
 * SNN_OPT_KERNEL: 희소(spike-driven) matmul. 입력이 이진 {0, S}이므로
 *   Σ w[k]·x[k] = S · Σ_{x[k]≠0} w[k]
 * 컬럼당 비영 인덱스를 한 번 만들고 채널마다 해당 가중치만 int32로 합산한다.
 * 정수 연산 결합법칙에 의해 dense와 비트 단위 동일. 이 모델의 발화율은
 * 입력 47.7% / LIF1 4.1%라 conv2에서 로드·MAC의 ~96%가 사라진다.
 */

#include "snn_conv_pool_lif_s16.h"
#include "snn_lif_s16.h"
#include "snn_hot.h"

#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_DSP)
#include "cmsis_compiler.h"
#endif

#if !defined(SNN_OPT_KERNEL) || (SNN_OPT_KERNEL == 0)

/* 컬럼 1~2개 x 채널 1~2개의 내적 (int64 누산) */
static SNN_ITCM_CODE void dot_rows_cols(const int16_t *w_row0, const int16_t *w_row1,
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
 * 컬럼 n_cols(1|2)개에 대한 matmul + requantize + LIF (dense 경로).
 * out은 스파이크 출력 버퍼 내 현재 위치, base = out - output_start.
 */
static SNN_ITCM_CODE int16_t *mat_mult_lif(const int16_t *filter, const int16_t *cols,
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

#endif /* !SNN_OPT_KERNEL (dense 경로 헬퍼) */

#if defined(SNN_OPT_KERNEL) && (SNN_OPT_KERNEL != 0)

/* 희소 인덱스 버퍼 (출력 픽셀 1개분). DTCM이면 로드 2사이클 확정. */
static uint16_t s_nz_idx[SNN_CPL_MAX_RHS_COLS] SNN_DTCM_BSS;

/*
 * 출력 픽셀 하나의 수용 영역에서 비영(발화) 입력의 im2col 인덱스 목록을
 * 직접 만든다. 값은 전부 in_spike_val이므로 im2col 컬럼을 실체화할 필요가
 * 없다 — dense 경로가 매 스텝 쓰고 읽던 컬럼 버퍼 트래픽(conv2 기준
 * 147KB/스텝)이 통째로 사라진다.
 *
 * 인덱스 k = (ky*kernel_x + kx)*input_ch + c 는 im2col 컬럼 레이아웃과
 * 동일하고, 패딩 셀은 0(비발화)이므로 건너뛰는 것과 같다. 정수 합은
 * 순서와 무관하므로 dense와 비트 단위 동일하다.
 */
static SNN_ITCM_CODE int32_t build_nz_idx_win(const int16_t *input,
                                              int32_t input_x, int32_t input_y, int32_t input_ch,
                                              int32_t kernel_x, int32_t kernel_y,
                                              int32_t base_y, int32_t base_x,
                                              uint16_t *idx)
{
    int32_t m = 0;
    for (int32_t ky = 0; ky < kernel_y; ++ky) {
        const int32_t iy = base_y + ky;
        if (iy < 0 || iy >= input_y) continue;                 /* 패딩 행 */
        for (int32_t kx = 0; kx < kernel_x; ++kx) {
            const int32_t ix = base_x + kx;
            if (ix < 0 || ix >= input_x) continue;             /* 패딩 열 */
            const int16_t *src = input + (iy * input_x + ix) * input_ch;
            const int32_t k0 = (ky * kernel_x + kx) * input_ch;
            if ((input_ch & 1) == 0 && (((uintptr_t)src) & 3u) == 0u) {
                /* 짝수 채널 + 정렬: 32비트 페어 검사. 발화율이 낮을수록
                 * pair==0 fast path가 대부분을 처리한다. */
                const uint32_t *p = (const uint32_t *)src;
                for (int32_t c = 0; c < input_ch; c += 2) {
                    uint32_t pair = *p++;
                    if (pair == 0) continue;
                    if (pair & 0xFFFFu) idx[m++] = (uint16_t)(k0 + c);
                    if (pair >> 16)     idx[m++] = (uint16_t)(k0 + c + 1);
                }
            } else {                                           /* conv1: 3채널 */
                /* 밀도 ~48%라 분기가 예측 불가 → branchless append
                 * (항상 쓰고 조건부로 m만 증가; 결과 목록은 동일) */
                for (int32_t c = 0; c < input_ch; ++c) {
                    idx[m] = (uint16_t)(k0 + c);
                    m += (src[c] != 0);
                }
            }
        }
    }
    return m;
}

/*
 * 희소 채널 합 + requantize → trow[0..output_ch). dense와 비트 단위 동일:
 *   dense  acc64 = Σ w[k]·x[k]            (x[k] ∈ {0, S})
 *   sparse acc64 = S · Σ_{x≠0} w[k]       (Σw는 int32: |Σw| ≤ 1152·508 < 2^31)
 * 이후 + bias, requantize, clamp 함수·순서 모두 동일하다 (채널별 합산은
 * 어느 레이아웃이든 j 오름차순 — int32 랩 덧셈은 순서 무관).
 *
 * filter_t는 전치 레이아웃 [rhs_cols][output_ch]: 활성 인덱스 k 하나의
 * 채널 전체(64ch=128B)가 연속이라 캐시라인이 온전히 쓰인다. 원래
 * [output_ch][rhs_cols] 레이아웃은 희소 gather가 채널마다 2.3KB 간격의
 * 라인을 건드려 144KB 가중치에서 미스를 양산했다.
 */
static SNN_ITCM_CODE void sparse_row_sums(const int16_t *filter_t, int32_t output_ch,
                                          const uint16_t *idx, int32_t m,
                                          const int32_t *output_mult, const int32_t *output_shift,
                                          const int64_t *bias, int32_t act_min, int32_t act_max,
                                          int32_t in_spike_val, int16_t *trow)
{
    int32_t co = 0;
    if ((output_ch & 7) == 0) {
        /* 8채널 레지스터 블로킹: 누산기가 전부 레지스터에 머물고,
         * 인덱스당 가중치 16B(8ch)가 연속이라 페어 로드 4회 + SXTAH 8회.
         * 인접 블록 쌍이 같은 캐시라인(32B=16ch)을 쓰므로 재사용도 남는다. */
        for (; co < output_ch; co += 8) {
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
            const int16_t *wb = filter_t + co;
            for (int32_t j = 0; j < m; ++j) {
                /* 4바이트 정렬: base aligned(4), idx*output_ch+co 짝수 */
                const uint32_t *wp = (const uint32_t *)(wb + (int32_t)idx[j] * output_ch);
                uint32_t p;
                p = wp[0]; s0 += (int16_t)(p & 0xFFFFu); s1 += (int16_t)(p >> 16);
                p = wp[1]; s2 += (int16_t)(p & 0xFFFFu); s3 += (int16_t)(p >> 16);
                p = wp[2]; s4 += (int16_t)(p & 0xFFFFu); s5 += (int16_t)(p >> 16);
                p = wp[3]; s6 += (int16_t)(p & 0xFFFFu); s7 += (int16_t)(p >> 16);
            }
            const int32_t s[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
            for (int32_t u = 0; u < 8; ++u) {
                const int32_t rm = REDUCE_MULTIPLIER(output_mult[co + u]);
                int32_t v = arm_nn_requantize_s64((int64_t)s[u] * in_spike_val + bias[co + u],
                                                  rm, output_shift[co + u]);
                trow[co + u] = (int16_t)MIN(MAX(v, act_min), act_max);
            }
        }
        return;
    }
    /* output_ch가 8의 배수가 아닐 때: 스칼라 (정렬 보장 없음) */
    for (; co < output_ch; ++co) {
        int32_t s0 = 0;
        for (int32_t j = 0; j < m; ++j) s0 += filter_t[(int32_t)idx[j] * output_ch + co];
        const int32_t rm = REDUCE_MULTIPLIER(output_mult[co]);
        int32_t v = arm_nn_requantize_s64((int64_t)s0 * in_spike_val + bias[co],
                                          rm, output_shift[co]);
        trow[co] = (int16_t)MIN(MAX(v, act_min), act_max);
    }
}

#endif /* SNN_OPT_KERNEL */

arm_cmsis_nn_status SNN_ITCM_CODE snn_convolve_pool_lif_s16(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int64_t *bias_data,
                                              const snn_lif_ctx *lif,
                                              int32_t in_spike_val,
                                              const cmsis_nn_dims *output_dims,
                                              int16_t *output_data)
{
    (void)bias_dims;
    (void)in_spike_val;

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

#if defined(SNN_OPT_KERNEL) && (SNN_OPT_KERNEL != 0)
    /* --- 희소(spike-driven) 경로: im2col 없이 인덱스 직접 생성 --- */
    if (rhs_cols > SNN_CPL_MAX_RHS_COLS) {
        return ARM_CMSIS_NN_ARG_ERROR; /* 인덱스 버퍼 크기 전제 위반 */
    }
    (void)ctx;

    /* dense와 동일한 2픽셀 타일 + LIF 구조 (LIF 호출 단위까지 동일) */
    int16_t tile[2 * 64] __attribute__((aligned(4)));
    int16_t *out = output_data;
    int32_t px_in_tile = 0;
    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
            const int32_t m = build_nz_idx_win(input_data, input_x, input_y, input_ch,
                                               kernel_x, kernel_y,
                                               i_out_y * stride_y - pad_y,
                                               i_out_x * stride_x - pad_x, s_nz_idx);
            sparse_row_sums(filter_data, output_ch, s_nz_idx, m,
                            quant_params->multiplier, quant_params->shift,
                            bias_data, act_min, act_max, in_spike_val,
                            &tile[px_in_tile * output_ch]);
            if (++px_in_tile == 2) {
                snn_lif_s16(2 * output_ch, tile, out,
                            &lif->mem[out - output_data], &lif->spk_prev[out - output_data],
                            lif->beta, lif->threshold, lif->spike_val);
                out += 2 * output_ch;
                px_in_tile = 0;
            }
        }
    }
    if (px_in_tile) { /* 출력 픽셀 수 홀수 leftover */
        snn_lif_s16(output_ch, tile, out,
                    &lif->mem[out - output_data], &lif->spk_prev[out - output_data],
                    lif->beta, lif->threshold, lif->spike_val);
    }

#else
    /* --- dense 경로: im2col 2컬럼 flush (arm_convolve_fast_s16과 동일 구조) --- */
    int16_t *buffer_a = (int16_t *)ctx->buf;
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
#endif /* SNN_OPT_KERNEL */

    return ARM_CMSIS_NN_SUCCESS;
}
