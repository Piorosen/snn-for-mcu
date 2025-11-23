/* Auto-generated quantized SNN weights (declaration, int-only) */
#ifndef CHACHA_SNN_WEIGHTS_H
#define CHACHA_SNN_WEIGHTS_H

#include <stdint.h>

/* All scales and LIF parameters are Q15 fixed-point (value / 2^15). */

// Conv1
extern const int8_t snn_conv1_weight[32][3][5][5];
extern const int16_t snn_conv1_weight_scale;
extern const int8_t snn_conv1_bias[32];
extern const int16_t snn_conv1_bias_scale;

// Conv2
extern const int8_t snn_conv2_weight[64][32][5][5];
extern const int16_t snn_conv2_weight_scale;
extern const int8_t snn_conv2_bias[64];
extern const int16_t snn_conv2_bias_scale;

// FC
extern const int8_t snn_fc_weight[10][4096];
extern const int16_t snn_fc_weight_scale;
extern const int8_t snn_fc_bias[10];
extern const int16_t snn_fc_bias_scale;

// LIF parameters (Q15 fixed-point)
extern const int16_t snn_lif1_beta;
extern const int16_t snn_lif2_beta;
extern const int16_t snn_lif_out_beta;
extern const int16_t snn_lif1_threshold;
extern const int16_t snn_lif2_threshold;
extern const int16_t snn_lif_out_threshold;

#endif /* CHACHA_SNN_WEIGHTS_H */
