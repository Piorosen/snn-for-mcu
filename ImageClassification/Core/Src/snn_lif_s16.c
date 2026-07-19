/*
 * LIF step — Cortex-M SIMD 최적화 커널 (CMSIS-NN 커널과 같은 방식).
 *
 * 듀얼 레인(int16 x2) 처리:
 *   1) beta * mem >> 15 : 레인별 SMULBB/SMULTB (컴파일러 생성) 후 PKHBT로 재패킹
 *   2) + in, sat16      : QADD16 (레인별 포화 덧셈)
 *   3) >= threshold     : SSUB16이 레인별 GE 플래그 설정 (플래그는 랩되지 않은
 *                         실제 차이의 부호 기준). GE 레인에서 diff = tilde - thr는
 *                         thr >= 0이면 [0, 32767] 범위라 랩되지 않음.
 *   4) 조건 선택        : SEL로 mem(감산 리셋)과 spike 값을 브랜치 없이 선택
 *
 * SSUB16/SEL은 cmsis_gcc.h에서 volatile asm이므로 서로 재배치되지 않고,
 * 사이의 일반 ALU 연산은 GE 플래그를 건드리지 않는다.
 */

#include "snn_lif_s16.h"

#if defined(ARM_MATH_DSP)
#include "cmsis_compiler.h"
#endif

void snn_lif_s16(int32_t n,
                 const int16_t *in,
                 int16_t *spk_out,
                 int16_t *mem,
                 int16_t *spk_prev,
                 int16_t beta,
                 int16_t threshold,
                 int16_t spike_val)
{
    int32_t i = 0;

#if defined(ARM_MATH_DSP)
    if (((((uintptr_t)in) | ((uintptr_t)spk_out) | ((uintptr_t)mem) | ((uintptr_t)spk_prev)) & 3u) == 0u)
    {
        const uint32_t thr_pair = ((uint32_t)(uint16_t)threshold << 16) | (uint16_t)threshold;
        const uint32_t spk_pair = ((uint32_t)(uint16_t)spike_val << 16) | (uint16_t)spike_val;

        const uint32_t *in32   = (const uint32_t *)in;
        uint32_t       *out32  = (uint32_t *)spk_out;
        uint32_t       *mem32  = (uint32_t *)mem;
        uint32_t       *prev32 = (uint32_t *)spk_prev;

        int32_t pairs = n >> 1;
        while (pairs > 0)
        {
            uint32_t mem_pair = *mem32;

            /* 레인별 beta * mem >> 15 (SMULBB/SMULTB로 컴파일됨) */
            int32_t s_lo = ((int32_t)(int16_t)mem_pair * (int32_t)beta) >> 15;
            int32_t s_hi = (((int32_t)mem_pair >> 16) * (int32_t)beta) >> 15;
            uint32_t scaled = __PKHBT(s_lo, s_hi, 16);

            uint32_t tilde = __QADD16(scaled, *in32++);

            uint32_t diff = __SSUB16(tilde, thr_pair); /* GE[lane] = (tilde >= thr) */
            uint32_t mem_next = __SEL(diff, tilde);
            uint32_t spk_raw = __SEL(spk_pair, 0u);

            *out32++  = *prev32;   /* reset_delay: 이전 스파이크를 출력 */
            *prev32++ = spk_raw;
            *mem32++  = mem_next;

            pairs--;
        }
        i = (n >> 1) << 1;
    }
#endif

    /* 스칼라 폴백 + 홀수 꼬리 (SIMD 경로와 비트 단위 동일) */
    for (; i < n; ++i)
    {
        int32_t tmp = ((int32_t)beta * (int32_t)mem[i]) >> 15;
        int32_t mem_tilde = tmp + in[i];
        if (mem_tilde > 32767)  mem_tilde = 32767;
        if (mem_tilde < -32768) mem_tilde = -32768;

        int16_t mem_tilde_q15 = (int16_t)mem_tilde;
        int16_t spk_raw = 0;
        int16_t mem_next = mem_tilde_q15;

        if (mem_tilde_q15 >= threshold)
        {
            spk_raw = spike_val;
            mem_next = (int16_t)(mem_tilde_q15 - threshold);
        }

        spk_out[i]  = spk_prev[i];
        spk_prev[i] = spk_raw;
        mem[i]      = mem_next;
    }
}
