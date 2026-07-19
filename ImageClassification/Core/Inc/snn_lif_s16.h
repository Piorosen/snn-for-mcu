#ifndef SNN_LIF_S16_H
#define SNN_LIF_S16_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * LIF(Leaky Integrate-and-Fire) step — CMSIS-NN 스타일 s16 커널.
 *
 * 원소별 시맨틱 (reset_mechanism="subtract", reset_delay=True):
 *   mem_tilde = sat16((beta * mem[i]) >> 15 + in[i])
 *   spike     = (mem_tilde >= threshold)
 *   spk_out[i]  = spk_prev[i]                  // 지연된 스파이크 출력
 *   spk_prev[i] = spike ? spike_val : 0
 *   mem[i]      = spike ? mem_tilde - threshold : mem_tilde
 *
 * ARM_MATH_DSP 정의 시 Cortex-M SIMD(QADD16 포화 덧셈, SSUB16의 GE 플래그 +
 * SEL 조건 선택)로 2원소/반복 브랜치리스 처리. 결과는 스칼라와 비트 단위 동일.
 *
 * 제약: threshold >= 0. 버퍼들은 서로 겹치지 않아야 한다.
 * 4바이트 정렬이 아니면 자동으로 스칼라 경로로 처리된다.
 */
void snn_lif_s16(int32_t n,
                 const int16_t *in,
                 int16_t *spk_out,
                 int16_t *mem,
                 int16_t *spk_prev,
                 int16_t beta,
                 int16_t threshold,
                 int16_t spike_val);

#ifdef __cplusplus
}
#endif

#endif /* SNN_LIF_S16_H */
