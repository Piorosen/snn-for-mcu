#ifndef CHACHA_SNN_INT_H
#define CHACHA_SNN_INT_H

#include <stdint.h>
#include "snn_weights.h"  // export_quant_snn_to_c_and_h 로 생성된 헤더

#ifdef __cplusplus
extern "C" {
#endif

#define IMG_H   32
#define IMG_W   32

// Q15에서 1.0 에 해당하는 값 (32767 사용)
#define ONE_Q15      32767
#define Q15_SHIFT    15

// LIF 상태 리셋
void snn_reset_state(void);

// data  : [B*C*H*W] 크기의 입력 (0~255)
// spikes: [T * B * C * H * W] 크기의 출력 버퍼 (0 또는 1 저장)
void spiking_rate(
    const uint8_t* data,   // 입력 데이터 (0 ~ 255)
    int16_t* spikes,       // 출력 스파이크 (0 또는 1)
    int T, int B, int C, int H, int W,
    float gain,
    float offset
);

/**
 * 단일 time-step forward
 *
 * @param x       입력 이미지 [3][32][32], Q15 (실제값 ~= x / 2^15)
 * @param spk_out 출력 스파이크 [10], Q15 (0 또는 ONE_Q15)
 * @param mem_out 출력 membrane [10], Q15
 */
void snn_forward_step(const int16_t* x,
                      int16_t* spk_out,
                      int16_t* mem_out);

#ifdef __cplusplus
}
#endif

#endif /* CHACHA_SNN_INT_H */
