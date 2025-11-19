/* auto-generated SNN weights (declaration) */

#ifndef SNN_WEIGHTS_H
#define SNN_WEIGHTS_H

#include <stdint.h>

/* Auto-generated SNN weight declarations */

extern const float snn_conv1_weight[32][3][5][5];
extern const float snn_conv1_bias[32];

extern const float snn_conv2_weight[64][32][5][5];
extern const float snn_conv2_bias[64];

extern const float snn_fc_weight[10][4096];
extern const float snn_fc_bias[10];

// LIF parameters
extern const float snn_lif1_beta;
extern const float snn_lif2_beta;
extern const float snn_lif_out_beta;
extern const float snn_lif1_threshold;
extern const float snn_lif2_threshold;
extern const float snn_lif_out_threshold;

#endif /* SNN_WEIGHTS_H */
