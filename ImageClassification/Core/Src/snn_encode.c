/*
 * 입력 스파이크 인코딩 (rate coding) — 정수화 가속 버전.
 *
 * 원본: 원소마다 float 정규화/곱셈 + newlib rand() 호출 + float 나눗셈 비교.
 * 가속: 픽셀값(0~255)당 발화 확률이 고정이므로 호출당 256-엔트리 임계값 LUT를
 *       만들고, 원소별로는 xorshift32 난수와 정수 비교 한 번만 수행한다.
 *
 *   spike ⇔ U(0,1) < p  ⇔  rand32 < round(p * 2^32)   (thr는 uint64로 p=1 정확 처리)
 *
 * 확률 오차는 LUT 양자화(≤ 2^-32)뿐으로 통계적으로 원본과 동일하다.
 * 난수원이 newlib LCG에서 xorshift32로 바뀌므로 부팅마다 재현되는 고정
 * 시퀀스 자체는 이전 펌웨어와 다르다 (동일 품질의 균등 분포).
 */

#include "snn.h"

static uint32_t g_encode_rng = 0x12345678u;

static inline uint32_t xorshift32(void) {
    uint32_t x = g_encode_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_encode_rng = x;
    return x;
}

// data  : [B*C*H*W] 크기의 입력 (0~255)
// spikes: [B*C*H*W] 크기의 출력 (0 또는 ONE_Q15)
void spiking_rate(
    const uint8_t* data,
    int16_t* spikes,
    int T, int B, int C, int H, int W,
    float gain,
    float offset
){
    (void)T;
    int spatial = B * C * H * W;

    // 발화 확률 p = clamp(gain * v/255 + offset, 0, 1) -> 임계값 p * 2^32
    uint64_t thr[256];
    for (int v = 0; v < 256; ++v) {
        double p = (double)gain * ((double)v / 255.0) + (double)offset;
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        thr[v] = (uint64_t)(p * 4294967296.0);
    }

    for (int i = 0; i < spatial; ++i) {
        spikes[i] = ((uint64_t)xorshift32() < thr[data[i]]) ? ONE_Q15 : 0;
    }
}
