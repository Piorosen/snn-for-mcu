
#include <stdlib.h>  // malloc, free, rand, srand
#include <stdio.h>
#include <string.h> // memset

#include <snn.h>

#define IMG_H   32
#define IMG_W   32

/* Conv1 출력: [32, 32, 32] */
static float g_conv1_out[32][IMG_H][IMG_W];
/* Pool1 출력: [32, 16, 16] */
static float g_pool1_out[32][16][16];
/* LIF1 spk/mem: [32, 16, 16] */
static float g_lif1_mem[32][16][16];
static float g_lif1_spk_prev[32][16][16];

/* Conv2 출력: [64, 16, 16] */
static float g_conv2_out[64][16][16];
/* Pool2 출력: [64, 8, 8] */
static float g_pool2_out[64][8][8];
/* LIF2 spk/mem: [64, 8, 8] */
static float g_lif2_mem[64][8][8];
static float g_lif2_spk_prev[64][8][8];

/* Flatten 출력: [4096] */
static float g_flat[4096];
/* FC 출력: [10] */
static float g_fc_out[10];
/* LIF_out spk/mem: [10] */
static float g_lif_out_mem[10];
static float g_lif_out_spk_prev[10];



/* --------- 유틸: 0으로 초기화 --------- */
void spiking_rate(
    const float* data,
    float* spikes,
    int T, int B, int C, int H, int W,
    float gain,
    float offset,
    int first_spike_time
) {
    int spatial = B * C * H * W;       // 1 * 3 * 32 * 32 = 3072
    int total_out = T * spatial;       // 30 * 3072 = 92160

    // 1) prob = gain * data + offset, clip [0, 1]
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

    // 2) broadcast_to((T, B, C, H, W)) + 랜덤 샘플
    //    spikes[t, :, :, :, :] = (rand < prob) ? 1 : 0
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < spatial; ++i) {
            // [0,1) 균등분포
            float r = (float)rand() / ((float)RAND_MAX + 1.0f);
            float p = prob[i];  // broadcast
            spikes[t * spatial + i] = (r < p) ? 1.0f : 0.0f;
        }
    }

    // 3) first_spike_time 처리: spikes[:t0] = 0.0
    if (first_spike_time > 0) {
        int t0 = (first_spike_time < T) ? first_spike_time : T;
        for (int t = 0; t < t0; ++t) {
            for (int i = 0; i < spatial; ++i) {
                spikes[t * spatial + i] = 0.0f;
            }
        }
    }

    free(prob);
}


/* --------- Conv2D 5x5, padding=2 구현 (Conv1) --------- */
/* 입력: in[3][32][32], 출력: out[32][32][32] */
void conv1_forward(const float in[3][32][32],
                          float out[32][32][32])
{
    /* zero padding: [3][36][36] */
    float in_pad[3][IMG_H + 4][IMG_W + 4] = {0};

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < IMG_H; ++h) {
            for (int w = 0; w < IMG_W; ++w) {
                in_pad[c][h + 2][w + 2] = in[c][h][w];
            }
        }
    }

    for (int co = 0; co < 32; ++co) {
        for (int h = 0; h < IMG_H; ++h) {
            for (int w = 0; w < IMG_W; ++w) {
                float sum = snn_conv1_bias[co];
                for (int ci = 0; ci < 3; ++ci) {
                    for (int kh = 0; kh < 5; ++kh) {
                        for (int kw = 0; kw < 5; ++kw) {
                            float x = in_pad[ci][h + kh][w + kw];
                            float wgt = snn_conv1_weight[co][ci][kh][kw];
                            sum += x * wgt;
                        }
                    }
                }
                out[co][h][w] = sum;
            }
        }
    }
}

/* --------- Conv2D 5x5, padding=2 구현 (Conv2) --------- */
/* 입력: in[32][16][16], 출력: out[64][16][16] */
void conv2_forward(const float in[32][16][16],
                          float out[64][16][16])
{
    const int H = 16;
    const int W = 16;
    float in_pad[32][H + 4][W + 4];
    memset(in_pad, 0, sizeof(in_pad));

    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                in_pad[c][h + 2][w + 2] = in[c][h][w];
            }
        }
    }

    for (int co = 0; co < 64; ++co) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float sum = snn_conv2_bias[co];
                for (int ci = 0; ci < 32; ++ci) {
                    for (int kh = 0; kh < 5; ++kh) {
                        for (int kw = 0; kw < 5; ++kw) {
                            float x = in_pad[ci][h + kh][w + kw];
                            float wgt = snn_conv2_weight[co][ci][kh][kw];
                            sum += x * wgt;
                        }
                    }
                }
                out[co][h][w] = sum;
            }
        }
    }
}

/* --------- AvgPool2D(2x2, stride=2), Conv1 출력용 --------- */
/* 입력: in[32][32][32], 출력: out[32][16][16] */
void avgpool2d_2x2_32x32(const float in[32][32][32],
                                float out[32][16][16])
{
    for (int c = 0; c < 32; ++c) {
        for (int oh = 0; oh < 16; ++oh) {
            for (int ow = 0; ow < 16; ++ow) {
                int h0 = oh * 2;
                int w0 = ow * 2;
                float sum = 0.0f;
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        sum += in[c][h0 + kh][w0 + kw];
                    }
                }
                out[c][oh][ow] = sum * 0.25f; // mean
            }
        }
    }
}

/* --------- AvgPool2D(2x2, stride=2), Conv2 출력용 --------- */
/* 입력: in[64][16][16], 출력: out[64][8][8] */
void avgpool2d_2x2_16x16(const float in[64][16][16],
                                float out[64][8][8])
{
    for (int c = 0; c < 64; ++c) {
        for (int oh = 0; oh < 8; ++oh) {
            for (int ow = 0; ow < 8; ++ow) {
                int h0 = oh * 2;
                int w0 = ow * 2;
                float sum = 0.0f;
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        sum += in[c][h0 + kh][w0 + kw];
                    }
                }
                out[c][oh][ow] = sum * 0.25f;
            }
        }
    }
}

/* --------- Flatten: [64][8][8] -> [4096] --------- */

void flatten_64x8x8_to_4096(const float in[64][8][8],
                                   float out[4096])
{
    int idx = 0;
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 8; ++h) {
            for (int w = 0; w < 8; ++w) {
                out[idx++] = in[c][h][w];
            }
        }
    }
}

/* --------- Linear: 4096 -> 10 --------- */

void linear_fc_4096_10(const float in[4096],
                              float out[10])
{
    for (int o = 0; o < 10; ++o) {
        float sum = snn_fc_bias[o];
        for (int i = 0; i < 4096; ++i) {
            sum += snn_fc_weight[o][i] * in[i];
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

                /* reset_delay=True 이므로 출력은 이전 스파이크 */
                spk_out[c][h][w] = spk_prev;

                /* 다음 타임스텝을 위해 상태 업데이트 */
                g_lif1_spk_prev[c][h][w] = spk_raw;   // graded_spikes_factor=1
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

                float mem_tilde = snn_lif2_beta * mem_prev + in[c][h][w];
                float spk_raw   = (mem_tilde >= snn_lif2_threshold) ? 1.0f : 0.0f;
                float mem_next  = mem_tilde - spk_raw * snn_lif2_threshold;

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

        float mem_tilde = snn_lif_out_beta * mem_prev + in[i];
        float spk_raw   = (mem_tilde >= snn_lif_out_threshold) ? 1.0f : 0.0f;
        float mem_next  = mem_tilde - spk_raw * snn_lif_out_threshold;

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
    /* Conv1 */
    conv1_forward(x, g_conv1_out);
    /* Pool1 */
    avgpool2d_2x2_32x32(g_conv1_out, g_pool1_out);
    /* LIF1 */
    lif1_step(g_pool1_out, g_conv2_out /* 임시 버퍼 재사용: 모양[32][16][16] */);

    /* Conv2: 입력은 LIF1의 spk_out -> g_conv2_out에 다시 덮어쓰지 않도록 주의 */
    conv2_forward((const float (*)[16][16])g_conv2_out, g_conv2_out);
    /* Pool2 */
    avgpool2d_2x2_16x16(g_conv2_out, g_pool2_out);
    /* LIF2 */
    lif2_step(g_pool2_out, (float (*)[8][8])g_pool1_out /* 임시 [64][8][8] 공간 필요하면 별도로 두어도 됨 */);

    /* Flatten */
    flatten_64x8x8_to_4096((const float (*)[8][8])g_pool1_out, g_flat);
    /* FC */
    linear_fc_4096_10(g_flat, g_fc_out);
    /* LIF_out */
    lif_out_step(g_fc_out, spk_out, mem_out);
}