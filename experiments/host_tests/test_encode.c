/*
 * snn_encode.c(실제 소스 포함) 검증:
 * 1) 결정적 경계: p=0 -> 스파이크 없음, p=1 -> 항상 스파이크
 * 2) clamp 동작 (offset/gain 극단값)
 * 3) 통계: 픽셀값별 발화율이 기준 확률 p의 5-시그마 이내
 * 4) 출력값은 0 또는 ONE_Q15만
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "snn_encode.c"   /* 실제 구현 */

static double ref_p(int v, double gain, double offset) {
    double p = gain * (v / 255.0) + offset;
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    return p;
}

int main(void)
{
    static uint8_t data[3072];
    static int16_t spikes[3072];
    int ok = 1;

    /* 1) 경계: gain=1, offset=0 에서 v=0 -> p=0, v=255 -> p=1 */
    for (int i = 0; i < 3072; ++i) data[i] = (i % 2) ? 255 : 0;
    long long z_spk = 0, o_miss = 0;
    for (int rep = 0; rep < 1000; ++rep) {
        spiking_rate(data, spikes, 30, 1, 3, 32, 32, 1.0f, 0.0f);
        for (int i = 0; i < 3072; ++i) {
            if (data[i] == 0 && spikes[i] != 0) z_spk++;
            if (data[i] == 255 && spikes[i] != ONE_Q15) o_miss++;
            if (spikes[i] != 0 && spikes[i] != ONE_Q15) { ok = 0; }
        }
    }
    printf("boundary: p=0 spikes=%lld (want 0), p=1 misses=%lld (want 0)\n", z_spk, o_miss);
    ok &= (z_spk == 0 && o_miss == 0);

    /* 2) clamp: offset=-2 -> 전부 0, offset=+2 -> 전부 발화 */
    for (int i = 0; i < 3072; ++i) data[i] = (uint8_t)(i % 256);
    spiking_rate(data, spikes, 30, 1, 3, 32, 32, 1.0f, -2.0f);
    long long c1 = 0;
    for (int i = 0; i < 3072; ++i) c1 += (spikes[i] != 0);
    spiking_rate(data, spikes, 30, 1, 3, 32, 32, 1.0f, 2.0f);
    long long c2 = 0;
    for (int i = 0; i < 3072; ++i) c2 += (spikes[i] == ONE_Q15);
    printf("clamp: low=%lld (want 0), high=%lld (want 3072)\n", c1, c2);
    ok &= (c1 == 0 && c2 == 3072);

    /* 3) 통계: (gain, offset) 두 조합에서 픽셀값별 발화율 vs 기준 p, 5-sigma */
    const double cases[][2] = { {1.0, 0.0}, {0.7, 0.1} };
    for (int c = 0; c < 2; ++c) {
        double gain = cases[c][0], offset = cases[c][1];
        static long long fire[256], seen[256];
        for (int v = 0; v < 256; ++v) fire[v] = seen[v] = 0;

        const int REPS = 4000; /* 값당 표본 = 3072/256 * 4000 = 48000 */
        for (int rep = 0; rep < REPS; ++rep) {
            spiking_rate(data, spikes, 30, 1, 3, 32, 32, (float)gain, (float)offset);
            for (int i = 0; i < 3072; ++i) {
                seen[data[i]]++;
                fire[data[i]] += (spikes[i] == ONE_Q15);
            }
        }
        double worst_sig = 0;
        int worst_v = 0;
        for (int v = 0; v < 256; ++v) {
            double p = ref_p(v, gain, offset);
            double n = (double)seen[v];
            double rate = fire[v] / n;
            double sd = sqrt(p * (1 - p) / n);
            double sig = (sd > 0) ? fabs(rate - p) / sd : (fabs(rate - p) > 0 ? 999 : 0);
            if (sig > worst_sig) { worst_sig = sig; worst_v = v; }
        }
        printf("stats gain=%.1f offset=%.1f: worst |rate-p| = %.2f sigma (v=%d) %s\n",
               gain, offset, worst_sig, worst_v, worst_sig < 5.0 ? "PASS" : "FAIL");
        ok &= (worst_sig < 5.0);
    }

    printf(ok ? "ALL PASS\n" : "SOME FAIL\n");
    return !ok;
}
