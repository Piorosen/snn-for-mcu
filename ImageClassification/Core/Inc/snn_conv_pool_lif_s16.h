#ifndef SNN_CONV_POOL_LIF_S16_H
#define SNN_CONV_POOL_LIF_S16_H

#include "arm_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* LIF 파라미터/상태 (CMSIS-NN에 없는 SNN 고유 확장) */
typedef struct {
    int16_t *mem;       /* [output pixels * output_ch] membrane, 4바이트 정렬 */
    int16_t *spk_prev;  /* [output pixels * output_ch] 직전 스파이크, 4바이트 정렬 */
    int16_t beta;       /* Q15 감쇠 */
    int16_t threshold;  /* Q15, >= 0 */
    int16_t spike_val;  /* 발화 시 출력 스파이크 크기 */
} snn_lif_ctx;

/*
 * Conv(pad) + AvgPool(2x2,s2) + LIF 퓨전 — arm_convolve_fast_s16과 같은 구조의
 * CMSIS-NN 스타일 커널.
 *
 * conv+pool은 "원본 5x5 커널 4개(2x2 시프트)의 합인 6x6 커널을 stride 2로
 * 적용 후 /4"와 동일하며(/4는 requantize 배율에 흡수, snn_weights_fused 참고),
 * 여기서는 filter_dims/conv_params가 그 6x6/s2 형상을 받는다.
 *
 * arm_convolve_fast_s16과의 차이:
 *  - filter_data가 int16 (퓨전 가중치 |값| <= 508은 int8 범위 초과).
 *    CMSIS-NN에는 s16 가중치 matmul이 없어 내적만 __SMLALD(64비트 누산,
 *    오버플로 원천 차단)로 수행하고, requantize 등 나머지는 CMSIS-NN
 *    지원 함수(arm_nn_requantize_s64 등)를 사용한다.
 *  - requantize 직후 출력 픽셀 단위(2픽셀 x output_ch 타일)로 LIF를 즉시
 *    적용(snn_lif_s16)해 conv/pool 중간 텐서가 존재하지 않는다.
 *    output_data에는 LIF 스파이크(reset_delay: 직전 스텝 스파이크)가 담긴다.
 *
 * in_spike_val: 입력 스파이크 진폭 S. 이 커널의 입력은 항상 이진 {0, S}다
 * (인코더 출력 또는 LIF 스파이크 — 패딩 0도 비발화와 동일). SNN_OPT_KERNEL
 * 빌드에서는 이를 이용해 희소(spike-driven) 경로로 계산한다:
 *   Σ w[k]·x[k] = S · Σ_{x[k]≠0} w[k]
 * 정수 연산이라 dense 경로와 비트 단위 동일하다 (Σ|w| ≤ rhs_cols·508 ≤
 * 585,216으로 int32 안전, S 곱은 int64). dense 경로에서는 무시된다.
 *
 * 가중치 레이아웃 주의: SNN_OPT_KERNEL 빌드의 filter_data는 **전치본**
 * [rhs_cols][output_ch] 이어야 한다 (희소 gather의 캐시 지역성 확보;
 * 호출자가 초기화 시점에 한 번 전치한다). dense 빌드는 원래 레이아웃
 * [output_ch][kernel_y][kernel_x][input_ch] 그대로다.
 *
 * 제약:
 *  - batch(input_dims->n) == 1, output_ch <= 64 (내부 타일 크기)
 *  - rhs_cols(= kernel_x*kernel_y*input_ch)는 짝수, <= SNN_CPL_MAX_RHS_COLS
 *  - ctx->buf: int16 2 * rhs_cols 이상, 4바이트 정렬
 *  - lif->threshold >= 0, 버퍼들은 서로 겹치지 않아야 한다
 *  - 출력 픽셀 수 홀수/output_ch 홀수는 leftover 경로로 처리된다 (제약 아님)
 */
#define SNN_CPL_MAX_RHS_COLS 1152   /* conv2: 6*6*32 (희소 인덱스 버퍼 크기) */

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
                                              int32_t in_spike_val,
                                              const cmsis_nn_dims *output_dims,
                                              int16_t *output_data);

#ifdef __cplusplus
}
#endif

#endif /* SNN_CONV_POOL_LIF_S16_H */
