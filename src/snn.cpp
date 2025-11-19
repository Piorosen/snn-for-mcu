
#include <stdlib.h>  // malloc, free, rand, srand
#include <stdio.h>


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


