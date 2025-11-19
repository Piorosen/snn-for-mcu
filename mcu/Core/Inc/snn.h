#ifndef CHACHA_SNN_H_
#define CHACHA_SNN_H_

#include <snn_weights.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------- Spike encoding (rate coding) --------- */
/*
 * data   : 입력 데이터 (크기 B*C*H*W, float)
 * spikes : 출력 스파이크 버퍼 (크기 T*B*C*H*W, float)
 * T, B, C, H, W : 시간/배치/채널/높이/너비
 * gain, offset  : prob = gain * data + offset (0~1로 클리핑)
 * first_spike_time : 처음 t < first_spike_time 구간은 강제로 0
 */
void spiking_rate(
    const float* data,
    float* spikes,
	int i,
    int T, int B, int C, int H, int W,
    float gain,
    float offset
);

/* --------- Conv2D 5x5, padding=2 --------- */
/* 입력: in[3][32][32], 출력: out[32][32][32] */
void conv1_forward(
    const float*,
    float*
);

/* 입력: in[32][16][16], 출력: out[64][16][16] */
void conv2_forward(
		const float*,
		    float*
);

/* --------- AvgPool2D(2x2, stride=2), Conv1 출력용 --------- */
/* 입력: in[32][32][32], 출력: out[32][16][16] */
void avgpool2d_2x2_32x32(const float* in,// [32][32][32],
                                float* out);//[32][16][16])

/* --------- AvgPool2D(2x2, stride=2), Conv2 출력용 --------- */
/* 입력: in[64][16][16], 출력: out[64][8][8] */
void avgpool2d_2x2_16x16(const float* in,//[64][16][16],
                                float* out); // [64][8][8])

///* --------- Flatten: [64][8][8] -> [4096] --------- */
//void flatten_64x8x8_to_4096(
//    const float in[64][8][8],
//    float out[4096]
//);

/* --------- Linear: 4096 -> 10 --------- */
void linear_fc_4096_10(
    const float* in, //[4096],
    float* out //[10]
);

/* --------- LIF 단계 함수들 --------- */
/* LIF1: in[32][16][16] -> spk_out[32][16][16] */
void lif1_step(
    const float in[32][16][16],
    float spk_out[32][16][16]
);

/* LIF2: in[64][8][8] -> spk_out[64][8][8] */
void lif2_step(
    const float in[64][8][8],
    float spk_out[64][8][8]
);

/* LIF_out: in[10] -> spk_out[10], mem_out[10] */
void lif_out_step(
    const float in[10],
    float spk_out[10],
    float mem_out[10]
);

/* --------- SNN 상태 초기화 & 단일 타임스텝 forward --------- */
void snn_reset_state(void);

/* x: [3][32][32], spk_out: [10], mem_out: [10] */
void snn_forward_step(
    const float x[3][32][32],
    float spk_out[10],
    float mem_out[10]
);

#ifdef __cplusplus
}
#endif
#endif // CHACHA_SNN_H_
