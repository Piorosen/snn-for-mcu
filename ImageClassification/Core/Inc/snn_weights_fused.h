/* tools/gen_fused_weights.c 가 생성한 파일 — 직접 수정 금지. */
#ifndef SNN_WEIGHTS_FUSED_H
#define SNN_WEIGHTS_FUSED_H

#include <stdint.h>

/* Conv+AvgPool 퓨전 6x6/s2 커널, [Co][6][6][Ci], 5x5 커널 4개의 합.
 * 실배율 = (원본 conv 배율) / 4 — requantize에서 처리 */
extern const int16_t snn_conv1_fused_w[32*6*6*3];
extern const int16_t snn_conv2_fused_w[64*6*6*32];

/* FC 가중치, 입력 HWC flatten 순서 [10][4096] */
extern const int8_t snn_fc_weight_hwc[10*4096];

#endif
