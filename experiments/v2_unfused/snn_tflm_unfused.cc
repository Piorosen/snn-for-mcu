/*
 * [실험 V2 전용] SNN forward — TFLite Micro 비퓨전 경로.
 * CONV_2D / AVERAGE_POOL_2D / FULLY_CONNECTED = TFLM CMSIS-NN int16x8 커널,
 * LIF만 커스텀 오퍼(SNN_LIF1/2/OUT). 논문의 "CMSIS-NN(TFLM 표준)" 비교군.
 */

#include "snn.h"
#include "snn_lif_s16.h"
#include "snn_model_data.h"

#include <string.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

static int16_t g_lif1_mem[32 * 16 * 16] __attribute__((aligned(4)));
static int16_t g_lif1_spk_prev[32 * 16 * 16] __attribute__((aligned(4)));
static int16_t g_lif2_mem[64 * 8 * 8] __attribute__((aligned(4)));
static int16_t g_lif2_spk_prev[64 * 8 * 8] __attribute__((aligned(4)));
static int16_t g_lif_out_mem[10] __attribute__((aligned(4)));
static int16_t g_lif_out_spk_prev[10] __attribute__((aligned(4)));

#ifndef SNN_TFLM_ARENA_SIZE
#define SNN_TFLM_ARENA_SIZE (128 * 1024)
#endif

namespace {

alignas(16) uint8_t g_tensor_arena[SNN_TFLM_ARENA_SIZE];
tflite::MicroInterpreter *g_interpreter = nullptr;
int g_ready = 0;

int32_t EvalElementCount(const TfLiteEvalTensor *t) {
    int32_t n = 1;
    for (int i = 0; i < t->dims->size; ++i) n *= t->dims->data[i];
    return n;
}

TfLiteStatus LifPrepare(TfLiteContext *, TfLiteNode *) { return kTfLiteOk; }

TfLiteStatus LifEvalCommon(TfLiteContext *context, TfLiteNode *node,
                           int16_t *mem, int16_t *spk_prev,
                           int16_t beta, int16_t threshold, int16_t spike_val) {
    const TfLiteEvalTensor *in = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor *out = tflite::micro::GetEvalOutput(context, node, 0);
    snn_lif_s16(EvalElementCount(in), in->data.i16, out->data.i16,
                mem, spk_prev, beta, threshold, spike_val);
    return kTfLiteOk;
}

TfLiteStatus Lif1Eval(TfLiteContext *c, TfLiteNode *n) {
    return LifEvalCommon(c, n, g_lif1_mem, g_lif1_spk_prev,
                         snn_lif1_beta, snn_lif1_threshold, ONE_Q15);
}
TfLiteStatus Lif2Eval(TfLiteContext *c, TfLiteNode *n) {
    return LifEvalCommon(c, n, g_lif2_mem, g_lif2_spk_prev,
                         snn_lif2_beta, snn_lif2_threshold, ONE_Q15);
}
TfLiteStatus LifOutEval(TfLiteContext *c, TfLiteNode *n) {
    return LifEvalCommon(c, n, g_lif_out_mem, g_lif_out_spk_prev,
                         snn_lif_out_beta, snn_lif_out_threshold, ONE_Q15);
}

TFLMRegistration MakeLifReg(TfLiteStatus (*invoke)(TfLiteContext *, TfLiteNode *)) {
    TFLMRegistration r = {};
    r.prepare = LifPrepare;
    r.invoke = invoke;
    return r;
}

TFLMRegistration g_lif1_reg = MakeLifReg(Lif1Eval);
TFLMRegistration g_lif2_reg = MakeLifReg(Lif2Eval);
TFLMRegistration g_lif_out_reg = MakeLifReg(LifOutEval);

} // namespace

extern "C" void snn_accel_init(void) {
    if (g_ready) return;
    const tflite::Model *model = tflite::GetModel(g_snn_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return;

    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddCustom("SNN_LIF1", &g_lif1_reg);
    resolver.AddCustom("SNN_LIF2", &g_lif2_reg);
    resolver.AddCustom("SNN_LIF_OUT", &g_lif_out_reg);

    static tflite::MicroInterpreter interpreter(model, resolver,
                                                g_tensor_arena, SNN_TFLM_ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed");
        return;
    }
    g_interpreter = &interpreter;
    g_ready = 1;
}

extern "C" void snn_reset_state(void) {
    memset(g_lif1_mem, 0, sizeof(g_lif1_mem));
    memset(g_lif1_spk_prev, 0, sizeof(g_lif1_spk_prev));
    memset(g_lif2_mem, 0, sizeof(g_lif2_mem));
    memset(g_lif2_spk_prev, 0, sizeof(g_lif2_spk_prev));
    memset(g_lif_out_mem, 0, sizeof(g_lif_out_mem));
    memset(g_lif_out_spk_prev, 0, sizeof(g_lif_out_spk_prev));
}

extern "C" void snn_forward_step(const int16_t *x,
                                 int16_t *spk_out,
                                 int16_t *mem_out) {
    if (!g_ready) snn_accel_init();
    if (!g_ready) {
        memset(spk_out, 0, 10 * sizeof(int16_t));
        memset(mem_out, 0, 10 * sizeof(int16_t));
        return;
    }
    memcpy(g_interpreter->input(0)->data.i16, x, 32 * 32 * 3 * sizeof(int16_t));
    if (g_interpreter->Invoke() != kTfLiteOk) {
        memset(spk_out, 0, 10 * sizeof(int16_t));
        memset(mem_out, 0, 10 * sizeof(int16_t));
        return;
    }
    memcpy(spk_out, g_interpreter->output(0)->data.i16, 10 * sizeof(int16_t));
    memcpy(mem_out, g_lif_out_mem, 10 * sizeof(int16_t));
}

extern "C" size_t snn_tflm_arena_used(void) {
    return g_ready ? g_interpreter->arena_used_bytes() : 0;
}
