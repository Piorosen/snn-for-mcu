#ifndef CHACHA_SNN_INT_H
#define CHACHA_SNN_INT_H

#include <stdint.h>
#include "snn_weights.h"  // export_quant_snn_to_c_and_h 로 생성된 헤더

#ifdef __cplusplus
extern "C" {
#endif

// Q15에서 1.0 에 해당하는 값 (32767 사용)
#define ONE_Q15      32767
#define Q15_SHIFT    15

// LIF 상태 리셋
void snn_reset_state(void);

// CMSIS-NN 가속 초기화: requantize 파라미터(mult/shift/bias64) 계산.
// (퓨전 가중치는 빌드타임 생성되어 FLASH 상주 — tools/gen_fused_weights.c)
// snn_forward_step 최초 호출 시 자동 수행되지만, 부팅 시 명시 호출 권장.
void snn_accel_init(void);

// 스파이크 인코딩 (rate coding, 구현: snn_encode.c)
// data  : [B*H*W*C] 크기의 입력 (0~255), T는 무시됨 (한 타임스텝 분량 생성)
// spikes: [B*H*W*C] 크기의 출력 버퍼 (0 또는 ONE_Q15 저장)
void spiking_rate(
    const uint8_t* data,   // 입력 데이터 (0 ~ 255)
    int16_t* spikes,       // 출력 스파이크 (0 또는 ONE_Q15)
    int T, int B, int C, int H, int W,
    float gain,
    float offset
);

/**
 * 단일 time-step forward
 *
 * @param x       입력 스파이크 [32][32][3] (HWC, JPEG 디코더 인터리브 출력과
 *                동일 레이아웃), Q15 (0 또는 ONE_Q15)
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
