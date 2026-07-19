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
 * 제약:
 *  - batch(input_dims->n) == 1, output_ch <= 64 (내부 타일 크기)
 *  - ctx->buf: int16 2 * (kernel_x*kernel_y*input_ch) 이상, 4바이트 정렬
 *  - lif->threshold >= 0, 버퍼들은 서로 겹치지 않아야 한다
 *  - 출력 픽셀 수 홀수/output_ch 홀수는 leftover 경로로 처리된다 (제약 아님)
 */
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
                                              int16_t *output_data);

#ifdef __cplusplus
}
#endif

#endif /* SNN_CONV_POOL_LIF_S16_H */
