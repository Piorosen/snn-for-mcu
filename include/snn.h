#ifndef CHACHA_SNN_H_
#define CHACHA_SNN_H_

void spiking_rate(
    const float* data,
    float* spikes,
    int T, int B, int C, int H, int W,
    float gain,
    float offset,
    int first_spike_time
);

#endif // CHACHA_SNN_H_
