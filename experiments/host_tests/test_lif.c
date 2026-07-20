/*
 * snn_lif_s16.c 실제 소스를 SIMD 목(mock)과 함께 컴파일해
 * 독립 스칼라 레퍼런스(원본 snn.c의 lif 시맨틱)와 비트 단위 비교.
 *
 * 1) 전수 레인 테스트: mem 전 범위(65536) x in(경계값+랜덤) x {beta, thr, spk} 조합
 * 2) 벡터 테스트: 실제 호출 형태(n=8192/4096/홀수)로 상태 갱신까지 비교
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* 실제 커널 소스를 그대로 포함 (ARM_MATH_DSP + 목 인트린식) */
#include "snn_lif_s16.c"

/* ---------- 원본 snn.c의 LIF 시맨틱 (독립 레퍼런스) ---------- */
static void ref_lif(int n, const int16_t *in, int16_t *spk_out,
                    int16_t *mem, int16_t *spk_prev,
                    int16_t beta, int16_t threshold, int16_t spike_val)
{
    for (int i = 0; i < n; ++i) {
        int16_t mem_prev = mem[i];
        int32_t tmp = ((int32_t)beta * (int32_t)mem_prev) >> 15;
        int32_t mem_tilde = tmp + in[i];
        int16_t mt;
        if (mem_tilde > 32767) mt = 32767;
        else if (mem_tilde < -32768) mt = -32768;
        else mt = (int16_t)mem_tilde;

        int16_t spk_raw = 0;
        int16_t mem_next = mt;
        if (mt >= threshold) { spk_raw = spike_val; mem_next = (int16_t)(mt - threshold); }

        spk_out[i] = spk_prev[i];
        spk_prev[i] = spk_raw;
        mem[i] = mem_next;
    }
}

static const int16_t EDGE_IN[] = {
    -32768, -32767, -32766, -16385, -16384, -16383, -2, -1, 0, 1, 2,
    16383, 16384, 16385, 32765, 32766, 32767
};
#define N_EDGE ((int)(sizeof(EDGE_IN) / sizeof(EDGE_IN[0])))

static long long g_fail = 0, g_checked = 0;

/* 페어 단위(2원소) 비교: SIMD 경로 강제 (정렬 버퍼) */
static void check_pair(int16_t mem0, int16_t in0,
                       int16_t beta, int16_t thr, int16_t spk_val)
{
    /* 두 레인 모두 같은 값으로 채워 레인별 결과를 동시 검증 */
    __attribute__((aligned(4))) int16_t in_a[2]  = { in0, in0 };
    __attribute__((aligned(4))) int16_t out_a[2], mem_a[2] = { mem0, mem0 };
    __attribute__((aligned(4))) int16_t prev_a[2] = { (int16_t)0x1234, (int16_t)0x5678 };

    int16_t in_r[2] = { in0, in0 };
    int16_t out_r[2], mem_r[2] = { mem0, mem0 };
    int16_t prev_r[2] = { (int16_t)0x1234, (int16_t)0x5678 };

    snn_lif_s16(2, in_a, out_a, mem_a, prev_a, beta, thr, spk_val);
    ref_lif(2, in_r, out_r, mem_r, prev_r, beta, thr, spk_val);

    g_checked++;
    if (memcmp(out_a, out_r, 4) || memcmp(mem_a, mem_r, 4) || memcmp(prev_a, prev_r, 4)) {
        if (g_fail < 5)
            printf("FAIL mem=%d in=%d beta=%d thr=%d spk=%d | mem: %d,%d vs %d,%d  prev: %d,%d vs %d,%d\n",
                   mem0, in0, beta, thr, spk_val,
                   mem_a[0], mem_a[1], mem_r[0], mem_r[1],
                   prev_a[0], prev_a[1], prev_r[0], prev_r[1]);
        g_fail++;
    }
}

int main(void)
{
    srand(777);

    const int16_t BETAS[] = { 29491, 0, 1, 32767 };
    const int16_t THRS[]  = { 32767, 12000, 1, 0 };
    const int16_t SPKS[]  = { 16384, 32767 };

    /* 1) 전수 레인: mem 전 범위 x (경계 in + 랜덤 in) */
    for (int bi = 0; bi < 4; ++bi)
        for (int ti = 0; ti < 4; ++ti) {
            int16_t spk = SPKS[(bi + ti) & 1];
            for (int32_t m = -32768; m <= 32767; ++m) {
                for (int e = 0; e < N_EDGE; ++e)
                    check_pair((int16_t)m, EDGE_IN[e], BETAS[bi], THRS[ti], spk);
                check_pair((int16_t)m, (int16_t)((rand() % 65536) - 32768),
                           BETAS[bi], THRS[ti], spk);
            }
        }
    printf("lane sweep: %lld checks, %lld fail\n", g_checked, g_fail);

    /* 2) 벡터 테스트: 실제 크기 + 홀수 n + 다중 스텝 상태 누적 */
    enum { NMAX = 8192 };
    static __attribute__((aligned(4))) int16_t in_a[NMAX], out_a[NMAX], mem_a[NMAX], prev_a[NMAX];
    static int16_t in_r[NMAX], out_r[NMAX], mem_r[NMAX], prev_r[NMAX];
    int sizes[] = { 8192, 4096, 4095, 3, 1 };
    long long vfail = 0;

    for (int si = 0; si < 5; ++si) {
        int n = sizes[si];
        memset(mem_a, 0, sizeof(mem_a)); memset(prev_a, 0, sizeof(prev_a));
        memset(mem_r, 0, sizeof(mem_r)); memset(prev_r, 0, sizeof(prev_r));

        for (int step = 0; step < 50; ++step) {
            for (int i = 0; i < n; ++i) {
                /* 스파이크성 입력 분포: 0/16384/32767 + 랜덤 */
                int k = rand() % 4;
                in_r[i] = in_a[i] = (k == 0) ? 0 : (k == 1) ? 16384 : (k == 2) ? 32767
                                      : (int16_t)((rand() % 65536) - 32768);
            }
            snn_lif_s16(n, in_a, out_a, mem_a, prev_a, 29491, 32767, 16384);
            ref_lif(n, in_r, out_r, mem_r, prev_r, 29491, 32767, 16384);
            if (memcmp(out_a, out_r, n * 2) || memcmp(mem_a, mem_r, n * 2) ||
                memcmp(prev_a, prev_r, n * 2)) {
                vfail++;
                printf("vector FAIL n=%d step=%d\n", n, step);
                break;
            }
        }
    }
    printf("vector: %s\n", vfail ? "FAIL" : "PASS");

    int ok = (g_fail == 0 && vfail == 0);
    printf(ok ? "ALL PASS\n" : "SOME FAIL\n");
    return !ok;
}
