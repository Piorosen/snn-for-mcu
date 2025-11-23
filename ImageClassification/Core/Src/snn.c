
#include "snn.h"
#include <string.h> // memset
#include <stdlib.h>

// --------- 내부 상태 (LIF) ---------

static int16_t g_lif1_mem[32][16][16];
static int16_t g_lif1_spk_prev[32][16][16];
static int16_t g_lif2_mem[64][8][8];
static int16_t g_lif2_spk_prev[64][8][8];
static int16_t g_lif_out_mem[10];
static int16_t g_lif_out_spk_prev[10];

// 작업용 버퍼 (중간 feature 공유)
// 최대 크기: Conv1 출력 [32][32][32] = 32768
#define MAX_FEATURE_SIZE (96 * 32 * 32)
static int16_t g_workbuf[MAX_FEATURE_SIZE];

// --------- 유틸: saturate ---------

static inline int16_t sat16_from32(int32_t x) {
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

static inline int16_t sat16_from64(int64_t x) {
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

// --------- Conv 5x5, padding=2, Q15 ---------

#include <stdlib.h>


// data  : [B*C*H*W] 크기의 입력 (0~255)
// spikes: [T * B * C * H * W] 크기의 출력 버퍼 (0 또는 1 저장)
void spiking_rate(
    const uint8_t* data,   // 입력 데이터 (0 ~ 255)
    int16_t* spikes,       // 출력 스파이크 (0 또는 1)
    int T, int B, int C, int H, int W,
    float gain,
    float offset
){ 
    int spatial = B * C * H * W;   // 예: 1 * 3 * 32 * 32 = 3072


    for (int i = 0; i < spatial; ++i) {
        // 0~255 → 0~1 로 정규화
        float x = (float)data[i] / 255.0f;

        // 발화 확률 p = gain * x + offset
        float p = gain * x + offset;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;

        // [0,1) 균등 난수
        float r = (float)rand() / ((float)RAND_MAX + 1.0f);

        // 스파이크 발생 여부 (int16_t로 0 또는 1)
        spikes[i] = (r < p) ? 32767 : 0;
    }
}


/*
 * in  : Q15, shape [C_in][H][W]
 * out : Q15, shape [C_out][H][W]
 * weight : int8, shape [C_out][C_in][5][5]
 * weight_scale : Q15 (export된 값), per-layer 스케일 가정
 * bias   : int8, shape [C_out]
 * bias_scale : Q15
 *
 * float 기준 표현:
 *   y_real ≈ sum( x_real * (w_int8 * s_w) ) + b_int8 * s_b
 *
 * Q15 기준:
 *   x_real  ≈ x_q15 / 2^15
 *   s_w     ≈ s_w_q15 / 2^15
 *   s_b     ≈ s_b_q15 / 2^15
 */
static void conv2d_5x5_pad2_q15(
    int C_in, int C_out,
    int H, int W,
    const int16_t *in,      // [C_in][H][W]
    int16_t *out,           // [C_out][H][W]
    const int8_t *weight,   // [C_out][C_in][5][5]
    int16_t weight_scale,   // Q15
    const int8_t *bias,     // [C_out]
    int16_t bias_scale      // Q15
) {
    for (int co = 0; co < C_out; ++co) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {

                // 1) 가중치 부분 누산 (Q15)
                int64_t sum_w_q15 = 0;

                for (int ci = 0; ci < C_in; ++ci) {
                    for (int kh = -2; kh <= 2; ++kh) {
                        int ih = h + kh;
                        if (ih < 0 || ih >= H) continue;

                        for (int kw = -2; kw <= 2; ++kw) {
                            int iw = w + kw;
                            if (iw < 0 || iw >= W) continue;

                            int in_idx = (ci * H + ih) * W + iw;
                            int w_kh = kh + 2;
                            int w_kw = kw + 2;
                            int w_idx = (((co * C_in) + ci) * 5 + w_kh) * 5 + w_kw;

                            int16_t x_q15 = in[in_idx];
                            int8_t  w_q0  = weight[w_idx];

                            // x_q15 (Q15) * w_q0 (Q0) -> Q15
                            sum_w_q15 += (int32_t)x_q15 * (int32_t)w_q0;
                        }
                    }
                }

                // 2) weight scale 적용: Q15 * Q15 -> Q30, >>15 -> Q15
                int64_t tmp = sum_w_q15 * (int64_t)weight_scale;
                int32_t y_w_q15 = (int32_t)(tmp >> Q15_SHIFT);

                // 3) bias 부분: bias_int8 * bias_scale_q15 -> Q15
                int32_t y_b_q15 = (int32_t)bias[co] * (int32_t)bias_scale;

                // 4) 합산 및 saturate
                int32_t y_q15 = y_w_q15 + y_b_q15;
                int out_idx = (co * H + h) * W + w;
                out[out_idx] = sat16_from32(y_q15);
            }
        }
    }
}

// Conv1 래퍼
static void conv1_forward(const int16_t *in, int16_t *out) {
    conv2d_5x5_pad2_q15(
        3, 32, 32, 32,
        in, out,
        (const int8_t*)snn_conv1_weight,
        snn_conv1_weight_scale,
        (const int8_t*)snn_conv1_bias,
        snn_conv1_bias_scale
    );
}

// Conv2 래퍼
static void conv2_forward(const int16_t *in, int16_t *out) {
    conv2d_5x5_pad2_q15(
        32, 64, 16, 16,
        in, out,
        (const int8_t*)snn_conv2_weight,
        snn_conv2_weight_scale,
        (const int8_t*)snn_conv2_bias,
        snn_conv2_bias_scale
    );
}

// --------- AvgPool2D 2x2, stride=2, Q15 ---------

static void avgpool2d_2x2_s2_q15(
    int C, int H, int W,
    const int16_t *in,
    int16_t *out
) {
    int H_out = H / 2;
    int W_out = W / 2;

    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                int h0 = oh * 2;
                int w0 = ow * 2;

                int idx00 = (c * H + (h0 + 0)) * W + (w0 + 0);
                int idx01 = (c * H + (h0 + 0)) * W + (w0 + 1);
                int idx10 = (c * H + (h0 + 1)) * W + (w0 + 0);
                int idx11 = (c * H + (h0 + 1)) * W + (w0 + 1);

                int32_t sum = 0;
                sum += in[idx00];
                sum += in[idx01];
                sum += in[idx10];
                sum += in[idx11];

                // 평균 = sum / 4 (rounding 포함)
                int32_t avg = (sum + ((sum >= 0) ? 2 : -2)) >> 2;

                int out_idx = (c * H_out + oh) * W_out + ow;
                out[out_idx] = sat16_from32(avg);
            }
        }
    }
}

// Conv1 출력용: [32][32][32] -> [32][16][16]
static void avgpool2d_2x2_32x32(const int16_t *in, int16_t *out) {
    avgpool2d_2x2_s2_q15(32, 32, 32, in, out);
}

// Conv2 출력용: [64][16][16] -> [64][8][8]
static void avgpool2d_2x2_16x16(const int16_t *in, int16_t *out) {
    avgpool2d_2x2_s2_q15(64, 16, 16, in, out);
}

// --------- FC 4096 -> 10, Q15 ---------

static void linear_fc_4096_10(const int16_t *in_q15, int16_t *out_q15) {
    for (int o = 0; o < 10; ++o) {
        int64_t sum_w_q15 = 0;

        for (int i = 0; i < 4096; ++i) {
            int16_t x_q15 = in_q15[i];
            int8_t  w_q0  = snn_fc_weight[o][i];
            // x_q15(Q15) * w_q0(Q0) -> Q15
            sum_w_q15 += (int32_t)x_q15 * (int32_t)w_q0;
        }

        // weight scale: Q15 * Q15 -> Q30 >>15 -> Q15
        int64_t tmp = sum_w_q15 * (int64_t)snn_fc_weight_scale;
        int32_t y_w_q15 = (int32_t)(tmp >> Q15_SHIFT);

        // bias: int8 * Q15 -> Q15
        int32_t y_b_q15 = (int32_t)snn_fc_bias[o] * (int32_t)snn_fc_bias_scale;

        int32_t y_q15 = y_w_q15 + y_b_q15;
        out_q15[o] = sat16_from32(y_q15);
    }
}

// --------- LIF step (Q15) ---------

/*
 * reset_mechanism="subtract", reset_delay=True
 * mem, in, threshold, beta 모두 Q15
 * spike는 Q15에서 0 또는 ONE_Q15
 */

// LIF1: in [32][16][16], out spk_prev[32][16][16]
static void lif1_step(const int16_t *in, int16_t *spk_out) {
    int H = 16, W = 16;

    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = (c * H + h) * W + w;

                int16_t mem_prev = g_lif1_mem[c][h][w];
                int16_t spk_prev = g_lif1_spk_prev[c][h][w];
                int16_t in_val   = in[idx];

                // mem_tilde = beta * mem_prev + in (Q15)
                int32_t tmp = ((int32_t)snn_lif1_beta * (int32_t)mem_prev) >> Q15_SHIFT;
                int32_t mem_tilde = tmp + in_val;
                int16_t mem_tilde_q15 = sat16_from32(mem_tilde);

                int16_t spk_raw_q15 = 0;
                int16_t mem_next_q15 = mem_tilde_q15;

                if (mem_tilde_q15 >= snn_lif1_threshold) {
                    spk_raw_q15 = ONE_Q15;
                    mem_next_q15 = mem_tilde_q15 - snn_lif1_threshold;
                }

                // reset_delay=True -> 현재 step 출력은 이전 spk
                spk_out[idx] = spk_prev;

                g_lif1_spk_prev[c][h][w] = spk_raw_q15;
                g_lif1_mem[c][h][w]      = mem_next_q15;
            }
        }
    }
}

// LIF2: in [64][8][8], out spk_prev[64][8][8]
static void lif2_step(const int16_t *in, int16_t *spk_out) {
    int H = 8, W = 8;

    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = (c * H + h) * W + w;

                int16_t mem_prev = g_lif2_mem[c][h][w];
                int16_t spk_prev = g_lif2_spk_prev[c][h][w];
                int16_t in_val   = in[idx];

                int32_t tmp = ((int32_t)snn_lif2_beta * (int32_t)mem_prev) >> Q15_SHIFT;
                int32_t mem_tilde = tmp + in_val;
                int16_t mem_tilde_q15 = sat16_from32(mem_tilde);

                int16_t spk_raw_q15 = 0;
                int16_t mem_next_q15 = mem_tilde_q15;

                if (mem_tilde_q15 >= snn_lif2_threshold) {
                    spk_raw_q15 = ONE_Q15;
                    mem_next_q15 = mem_tilde_q15 - snn_lif2_threshold;
                }

                spk_out[idx] = spk_prev;

                g_lif2_spk_prev[c][h][w] = spk_raw_q15;
                g_lif2_mem[c][h][w]      = mem_next_q15;
            }
        }
    }
}

// LIF_out: in[10], out spk_prev[10], mem_out[10]
static void lif_out_step(const int16_t in[10],
                         int16_t spk_out[10],
                         int16_t mem_out[10]) {
    for (int i = 0; i < 10; ++i) {
        int16_t mem_prev = g_lif_out_mem[i];
        int16_t spk_prev = g_lif_out_spk_prev[i];
        int16_t in_val   = in[i];

        int32_t tmp = ((int32_t)snn_lif_out_beta * (int32_t)mem_prev) >> Q15_SHIFT;
        int32_t mem_tilde = tmp + in_val;
        int16_t mem_tilde_q15 = sat16_from32(mem_tilde);

        int16_t spk_raw_q15 = 0;
        int16_t mem_next_q15 = mem_tilde_q15;

        if (mem_tilde_q15 >= snn_lif_out_threshold) {
            spk_raw_q15 = ONE_Q15;
            mem_next_q15 = mem_tilde_q15 - snn_lif_out_threshold;
        }

        spk_out[i] = spk_prev;
        mem_out[i] = mem_next_q15;

        g_lif_out_spk_prev[i] = spk_raw_q15;
        g_lif_out_mem[i]      = mem_next_q15;
    }
}

// --------- 상태 리셋 ---------

void snn_reset_state(void) {
    memset(g_lif1_mem,         0, sizeof(g_lif1_mem));
    memset(g_lif1_spk_prev,    0, sizeof(g_lif1_spk_prev));
    memset(g_lif2_mem,         0, sizeof(g_lif2_mem));
    memset(g_lif2_spk_prev,    0, sizeof(g_lif2_spk_prev));
    memset(g_lif_out_mem,      0, sizeof(g_lif_out_mem));
    memset(g_lif_out_spk_prev, 0, sizeof(g_lif_out_spk_prev));
}

// --------- 전체 한 step forward (B=1) ---------

void snn_forward_step(const int16_t* x,
                      int16_t* spk_out,
                      int16_t* mem_out) {
    // g_workbuf 재사용
    int16_t *buf = g_workbuf;

    // Conv1: 입력 [3][32][32] -> buf[0 .. 32*32*32-1]
    conv1_forward(x, buf);

    // Pool1: Conv1 출력 [32][32][32] -> pool1_out [32][16][16]
    int16_t *pool1_out = &buf[MAX_FEATURE_SIZE - 32 * 16 * 16];
    avgpool2d_2x2_32x32(buf, pool1_out);

    // LIF1: in [32][16][16] -> spk1_out [32][16][16] (buf 앞쪽 재사용)
    int16_t *lif1_spk = buf;  // 크기 32*16*16
    lif1_step(pool1_out, lif1_spk);

    // Conv2: in lif1_spk [32][16][16] -> conv2_out [64][16][16]
    int16_t *conv2_out = &buf[MAX_FEATURE_SIZE - 64 * 16 * 16];
    conv2_forward(lif1_spk, conv2_out);

    // Pool2: [64][16][16] -> pool2_out [64][8][8] (buf 앞쪽 재사용)
    int16_t *pool2_out = buf; // 크기 64*8*8
    avgpool2d_2x2_16x16(conv2_out, pool2_out);

    // LIF2: in [64][8][8] -> spk2_out [64][8][8] (buf 뒤쪽 일부 사용)
    int16_t *lif2_spk = &buf[MAX_FEATURE_SIZE - 64 * 8 * 8];
    lif2_step(pool2_out, lif2_spk);

    // FC: in lif2_spk (연속 4096 = 64*8*8) -> fc_out[10] (buf 앞쪽)
    int16_t *fc_out = buf;
    linear_fc_4096_10(lif2_spk, fc_out);

    // LIF_out: fc_out[10] -> spk_out[10], mem_out[10]
    lif_out_step(fc_out, spk_out, mem_out);
}
