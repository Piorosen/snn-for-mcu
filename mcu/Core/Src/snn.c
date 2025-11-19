
#include <stdlib.h>  // malloc, free, rand, srand
#include <stdio.h>
#include <string.h> // memset
#include <stdint.h>
#include <snn.h>

#define IMG_H   32
#define IMG_W   32

/* Conv1 출력: [32, 32, 32] */

static float g_lif1_mem[32][16][16];
static float g_lif1_spk_prev[32][16][16];
static float g_lif2_mem[64][8][8];
static float g_lif2_spk_prev[64][8][8];
static float g_lif_out_mem[10];
static float g_lif_out_spk_prev[10];


/* --------- 유틸: 0으로 초기화 --------- */
void spiking_rate(
    const float* data,
    float* spikes,
	int index,
    int T, int B, int C, int H, int W,
    float gain,
    float offset
) {
    int spatial = B * C * H * W;       // 1 * 3 * 32 * 32 = 3072
    float* prob = (float*)malloc(sizeof(float) * spatial);
    if (!prob) {
        fprintf(stderr, "malloc 실패\n");
        return;
    }

    for (int i = 0; i < spatial; ++i) {
        float p = gain * data[i] + offset;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        prob[i] = p;
    }

        for (int i = 0; i < spatial; ++i) {
            // [0,1) 균등분포
            float r = (float)rand() / ((float)RAND_MAX + 1.0f);
            float p = prob[i];  // broadcast
            spikes[/*t * spatial + */i] = (r < p) ? 1.0f : 0.0f;
        }
    free(prob);
}



/* --------- Conv2D 5x5, padding=2 구현 (Conv1) --------- */
/* 입력: in[3][32][32], 출력: out[32][32][32] */
void conv2d_5x5_pad2_forward(int C_in, int C_out,
                             int H, int W,
                             const float *in,
                             float *out,
                             const int8_t *weight,
                             const float weight_scale,
                             const int8_t *bias,
                             const float bias_scale)
{
    int co, ci, h, w, kh, kw;

    for (co = 0; co < C_out; ++co)
    {
        for (h = 0; h < H; ++h)
        {
            for (w = 0; w < W; ++w)
            {
                float sum = (float)bias[co] * bias_scale;

                for (ci = 0; ci < C_in; ++ci)
                {
                    /* 이 (h, w)에 대해 실제로 유효한 kh, kw 범위만 계산 */
                    int kh_start = (h <= 2) ? (2 - h) : 0;
                    int kh_end   = (h >= H - 3) ? (H - 1 + 2 - h) : 4;

                    int kw_start = (w <= 2) ? (2 - w) : 0;
                    int kw_end   = (w >= W - 3) ? (W - 1 + 2 - w) : 4;

                    int base_in   = ci * H * W;
                    int base_w_co_ci = ((co * C_in) + ci) * 25; // 5*5 = 25

                    for (kh = kh_start; kh <= kh_end; ++kh)
                    {
                        int ih = h + kh - 2; // 실제 입력 위치 (row)
                        for (kw = kw_start; kw <= kw_end; ++kw)
                        {
                            int iw = w + kw - 2; // 실제 입력 위치 (col)

                            // 여기까지 온 kh, kw는 항상 유효 범위만 돌도록 만들어져 있음
                            float in_val = in[base_in + ih * W + iw];

                            int w_idx = base_w_co_ci + kh * 5 + kw;
                            float w_val = (float)weight[w_idx] * weight_scale;

                            sum += in_val * w_val;
                        }
                    }
                }

                out[(co * H + h) * W + w] = sum;
            }
        }
    }
}


void conv1_forward(const float* in,
                          float* out)
{
	conv2d_5x5_pad2_forward(3, 32, 32, 32, in, out, (int8_t*)snn_conv1_weight, snn_conv1_weight_scale, (int8_t*)snn_conv1_bias, snn_conv1_bias_scale);
}

/* --------- Conv2D 5x5, padding=2 구현 (Conv2) --------- */
/* 입력: in[32][16][16], 출력: out[64][16][16] */
void conv2_forward(const float* in,
                          float* out)
{
	conv2d_5x5_pad2_forward(32, 64, 16, 16, in, out, (int8_t*)snn_conv2_weight, snn_conv2_weight_scale, (int8_t*)snn_conv2_bias, snn_conv2_bias_scale);
}


/* --------- AvgPool2D(2x2, stride=2), Conv1 출력용 --------- */
/* 입력: in[32][32][32], 출력: out[32][16][16] */
void avgpool2d_2x2_s2_forward(int C, int H, int W,
                              const float *in,
                              float *out)
{
    int c, oh, ow;
    int H_out = H / 2;
    int W_out = W / 2;

    for (c = 0; c < C; ++c) {
        for (oh = 0; oh < H_out; ++oh) {
            for (ow = 0; ow < W_out; ++ow) {
                int h0 = oh * 2;
                int w0 = ow * 2;
                float sum = 0.0f;

                sum += in[(c * H + (h0 + 0)) * W + (w0 + 0)];
                sum += in[(c * H + (h0 + 0)) * W + (w0 + 1)];
                sum += in[(c * H + (h0 + 1)) * W + (w0 + 0)];
                sum += in[(c * H + (h0 + 1)) * W + (w0 + 1)];

                out[(c * H_out + oh) * W_out + ow] = sum * 0.25f;
            }
        }
    }
}

/* --------- AvgPool2D(2x2, stride=2), Conv1 출력용 --------- */
/* 입력: in[32][32][32], 출력: out[32][16][16] */
void avgpool2d_2x2_32x32(const float* in,// [32][32][32],
                                float* out)//[32][16][16])
{
	avgpool2d_2x2_s2_forward(32,32,32, (float*)in, (float*)out);
}

/* --------- AvgPool2D(2x2, stride=2), Conv2 출력용 --------- */
/* 입력: in[64][16][16], 출력: out[64][8][8] */
void avgpool2d_2x2_16x16(const float* in,//[64][16][16],
                                float* out) // [64][8][8])
{
	avgpool2d_2x2_s2_forward(64,16,16, (float*)in, (float*)out);
}

// /* --------- Flatten: [64][8][8] -> [4096] --------- */

// void flatten_64x8x8_to_4096(const float in[64][8][8],
//                                    float out[4096])
// {
//     int idx = 0;
//     for (int c = 0; c < 64; ++c) {
//         for (int h = 0; h < 8; ++h) {
//             for (int w = 0; w < 8; ++w) {
//                 out[idx++] = in[c][h][w];
//             }
//         }
//     }
// }

/* --------- Linear: 4096 -> 10 --------- */
void linear_fc_4096_10(const float* in,
                              float* out)
{
    for (int o = 0; o < 10; ++o) {
        float sum = (snn_fc_bias[o] * snn_fc_bias_scale);
        for (int i = 0; i < 4096; ++i) {
            sum += (snn_fc_weight[o][i] * snn_fc_weight_scale) * in[i];
        }
        out[o] = sum;
    }
}

/* --------- LeakyNP step (reset_mechanism="subtract", reset_delay=True) --------- */
/* LIF1: 입력 in[32][16][16], 출력 spk_out[32][16][16] (spk_prev 출력) */

void lif1_step(const float in[32][16][16],
               float spk_out[32][16][16])
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 16; ++h) {
            for (int w = 0; w < 16; ++w) {
                float mem_prev = g_lif1_mem[c][h][w];
                float spk_prev = g_lif1_spk_prev[c][h][w];

                float mem_tilde = snn_lif1_beta * mem_prev + in[c][h][w];
                float spk_raw   = (mem_tilde >= snn_lif1_threshold) ? 1.0f : 0.0f;
                float mem_next  = mem_tilde - spk_raw * snn_lif1_threshold;

                spk_out[c][h][w] = spk_prev;

                g_lif1_spk_prev[c][h][w] = spk_raw;
                g_lif1_mem[c][h][w]      = mem_next;
            }
        }
    }
}

/* LIF2: in[64][8][8], out spk_out[64][8][8] */

void lif2_step(const float in[64][8][8],
                      float spk_out[64][8][8])
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 8; ++h) {
            for (int w = 0; w < 8; ++w) {
            	float mem_prev = g_lif2_mem[c][h][w];
            	float spk_prev = g_lif2_spk_prev[c][h][w];

            	float mem_tilde = snn_lif2_beta * mem_prev + (in[c][h][w]);
                float spk_raw   = (mem_tilde >= (snn_lif2_threshold)) ? 1.0f : 0.0f;
                float mem_next  = mem_tilde - spk_raw * (snn_lif2_threshold);

                spk_out[c][h][w] = spk_prev;

                g_lif2_spk_prev[c][h][w] = spk_raw;
                g_lif2_mem[c][h][w]      = mem_next;
            }
        }
    }
}

/* LIF_out: in[10], out spk_out[10], mem_out[10] */

void lif_out_step(const float in[10],
                         float spk_out[10],
                         float mem_out[10])
{
    for (int i = 0; i < 10; ++i) {
    	float mem_prev = g_lif_out_mem[i];
    	float spk_prev = g_lif_out_spk_prev[i];

    	float mem_tilde = snn_lif_out_beta * mem_prev + (in[i]);
        float spk_raw   = (mem_tilde >= (snn_lif_out_threshold)) ? 1.0f : 0.0f;
        float mem_next  = mem_tilde - spk_raw * (snn_lif_out_threshold);

        spk_out[i] = spk_prev;
        mem_out[i] = mem_next;

        g_lif_out_spk_prev[i] = spk_raw;
        g_lif_out_mem[i]      = mem_next;
    }
}
void snn_reset_state(void)
{
    memset(g_lif1_mem, 0, sizeof(g_lif1_mem));
    memset(g_lif1_spk_prev, 0, sizeof(g_lif1_spk_prev));

    memset(g_lif2_mem, 0, sizeof(g_lif2_mem));
    memset(g_lif2_spk_prev, 0, sizeof(g_lif2_spk_prev));

    memset(g_lif_out_mem, 0, sizeof(g_lif_out_mem));
    memset(g_lif_out_spk_prev, 0, sizeof(g_lif_out_spk_prev));
}


/* --------- 전체 한 step forward (B=1) --------- */

void snn_forward_step(const float x[3][32][32],
                      float spk_out[10],
                      float mem_out[10])
{
	float in_buffer1[64][32][32];
	float in_buffer2[64][32][32];

    /* Conv1 */
    conv1_forward(x, in_buffer1);
    /* Pool1 */
    avgpool2d_2x2_32x32(in_buffer1, in_buffer2);
    /* LIF1 */
    lif1_step(in_buffer2, in_buffer1 /* 임시 버퍼 재사용: 모양[32][16][16] */);

    /* Conv2: 입력은 LIF1의 spk_out -> g_conv2_out에 다시 덮어쓰지 않도록 주의 */
    conv2_forward((const float (*)[16][16])in_buffer1, in_buffer2);
    /* Pool2 */
    avgpool2d_2x2_16x16(in_buffer2, in_buffer1);
    /* LIF2 */
    lif2_step(in_buffer1, (float (*)[8][8])in_buffer2 /* 임시 [64][8][8] 공간 필요하면 별도로 두어도 됨 */);
    /* Flatten */
    // flatten_64x8x8_to_4096((const float (*)[8][8])g_pool1_out, g_flat);
    /* FC */
    linear_fc_4096_10(in_buffer2, in_buffer1);
    /* LIF_out */
    lif_out_step(in_buffer1, spk_out, mem_out);
}
