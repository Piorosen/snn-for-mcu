/*
 * 호스트 검증용 Cortex-M SIMD 인트린식 시뮬레이션.
 * ARM ARM 시맨틱 기준:
 *  - QADD16: 레인별 int16 포화 덧셈
 *  - SSUB16: 레인별 랩어라운드 감산, GE 플래그는 랩되지 않은 실제 차이의 부호
 *  - SEL: GE 플래그 기준 레인별 선택 (GE=1이면 첫 번째 피연산자)
 *  - PKHBT: (a의 하위 16비트) | (b << shift 의 상위 16비트)
 */
#ifndef SIM_CMSIS_COMPILER_H
#define SIM_CMSIS_COMPILER_H

#include <stdint.h>

static int sim_ge_lo, sim_ge_hi;

static inline uint32_t __QADD16(uint32_t a, uint32_t b) {
    int32_t lo = (int32_t)(int16_t)(a & 0xFFFF) + (int32_t)(int16_t)(b & 0xFFFF);
    int32_t hi = (int32_t)(int16_t)(a >> 16)    + (int32_t)(int16_t)(b >> 16);
    if (lo > 32767) lo = 32767; if (lo < -32768) lo = -32768;
    if (hi > 32767) hi = 32767; if (hi < -32768) hi = -32768;
    return ((uint32_t)(uint16_t)hi << 16) | (uint16_t)lo;
}

static inline uint32_t __SSUB16(uint32_t a, uint32_t b) {
    int32_t lo = (int32_t)(int16_t)(a & 0xFFFF) - (int32_t)(int16_t)(b & 0xFFFF);
    int32_t hi = (int32_t)(int16_t)(a >> 16)    - (int32_t)(int16_t)(b >> 16);
    sim_ge_lo = (lo >= 0);
    sim_ge_hi = (hi >= 0);
    return ((uint32_t)(uint16_t)hi << 16) | (uint16_t)lo; /* 랩어라운드 저장 */
}

static inline uint32_t __SEL(uint32_t a, uint32_t b) {
    uint32_t lo = sim_ge_lo ? (a & 0xFFFF) : (b & 0xFFFF);
    uint32_t hi = sim_ge_hi ? (a & 0xFFFF0000u) : (b & 0xFFFF0000u);
    return hi | lo;
}

#define __PKHBT(ARG1, ARG2, ARG3) \
    ((((uint32_t)(ARG1)) & 0x0000FFFFu) | ((((uint32_t)(ARG2)) << (ARG3)) & 0xFFFF0000u))

/* SMLALD: 듀얼 signed 16x16 곱을 64비트 누산기에 더함 */
static inline uint64_t __SMLALD(uint32_t x, uint32_t y, uint64_t acc) {
    int32_t xb = (int16_t)(x & 0xFFFF), xt = (int16_t)(x >> 16);
    int32_t yb = (int16_t)(y & 0xFFFF), yt = (int16_t)(y >> 16);
    return (uint64_t)((int64_t)acc + (int64_t)xb * yb + (int64_t)xt * yt);
}

#endif
