/*
 * 호스트 DSP목 테스트용 arm_nnsupportfunctions.h 심(shim).
 * 실제 헤더는 ARM_MATH_DSP 정의 시 호스트에 없는 SXTB16 계열을 요구하므로,
 * snn_conv_pool_lif_s16.c가 사용하는 심볼만 CMSIS-NN과 동일 시맨틱으로 제공한다.
 */
#ifndef SIM_ARM_NNSUPPORTFUNCTIONS_H
#define SIM_ARM_NNSUPPORTFUNCTIONS_H

#include <stdint.h>
#include <string.h>
#include "arm_nn_types.h"   /* 실제 CMSIS-NN 타입 (뒤쪽 -I 경로에서 발견) */

#ifndef MAX
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif
#ifndef MIN
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

#define REDUCE_MULTIPLIER(_mult) ((_mult < 0x7FFF0000) ? ((_mult + (1 << 15)) >> 16) : 0x7FFF)

static inline int32_t arm_nn_requantize_s64(const int64_t val,
                                            const int32_t reduced_multiplier,
                                            const int32_t shift)
{
    const int64_t new_val = val * reduced_multiplier;
    int32_t result = (int32_t)(new_val >> (14 - shift));
    result = (result + 1) >> 1;
    return result;
}

static inline uint32_t arm_nn_read_q15x2_ia(const int16_t **in_q15)
{
    uint32_t val;
    memcpy(&val, *in_q15, 4);
    *in_q15 += 2;
    return val;
}

static inline void arm_memcpy_s8(int8_t *dst, const int8_t *src, uint32_t block_size)
{
    memcpy(dst, src, block_size);
}

static inline void arm_memset_s8(int8_t *dst, const int8_t val, uint32_t block_size)
{
    memset(dst, val, block_size);
}

#endif
