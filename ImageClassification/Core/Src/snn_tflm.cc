/*
 * SNN forward — TFLite Micro 런타임 경로 (Conv+Pool+LIF 퓨전 커스텀 오퍼).
 *
 * 그래프: input -> SNN_CONV_POOL_LIF1 -> SNN_CONV_POOL_LIF2
 *         -> FULLY_CONNECTED(TFLM CMSIS-NN int16x8) -> SNN_LIF_OUT -> [10]
 *
 * 퓨전 커스텀 오퍼는 검증된 snn_convolve_pool_lif_s16(6x6/s2 SMLALD 커널)을
 * 호출하고, TFLM 메모리 기능을 그대로 활용한다:
 *   - requantize 파라미터(per-channel mult/shift): AllocatePersistentBuffer
 *   - LIF 막전위/직전 스파이크 상태:              AllocatePersistentBuffer
 *   - im2col 버퍼:                                RequestScratchBufferInArena
 *   - 활성값 텐서:                                아레나 플래너가 배치
 * 즉 SNN 관련 RAM이 전부 텐서 아레나 안에서 관리된다 (정적 배열 없음).
 *
 * 양자화: Prepare에서 텐서 스케일(활성 1/32768, 가중치 ws/(32768*4))로
 * 실배율을 계산해 QuantizeMultiplier로 mult/shift를 만든다. 스파이크 값은
 * 전 레이어 32767(Q15 1.0) — 커널이 SMLALD 64비트 누산이라 오버플로 없음.
 */

#include "snn.h"
#include "snn_conv_pool_lif_s16.h"
#include "snn_lif_s16.h"
#include "snn_model_data.h"

#include <string.h>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --------- 텐서 아레나 / 인터프리터 ---------

// 현재 모델 실사용 80,288B (arena_used_bytes). 최대 확보 정책:
// RAM 320KB - 아레나 외 정적 ~6KB - 스택/힙(libjpeg malloc 포함) 여유 64KB
#ifndef SNN_TFLM_ARENA_SIZE
#define SNN_TFLM_ARENA_SIZE (256 * 1024)
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

// --------- LIF 상태 레지스트리 (아레나 퍼시스턴트 버퍼, 리셋용 추적) ---------

struct LifState {
    int16_t *mem;
    int16_t *spk_prev;
    int32_t n;
};
LifState g_lif_states[3];   // [0]=CPL1, [1]=CPL2, [2]=LIF_OUT
int16_t *g_lif_out_mem = nullptr; // mem_out 복사용

TfLiteStatus AllocLifState(TfLiteContext *context, int slot, int32_t n) {
    LifState &s = g_lif_states[slot];
    s.n = n;
    s.mem = (int16_t *)context->AllocatePersistentBuffer(context, n * sizeof(int16_t));
    s.spk_prev = (int16_t *)context->AllocatePersistentBuffer(context, n * sizeof(int16_t));
    if (!s.mem || !s.spk_prev) return kTfLiteError;
    memset(s.mem, 0, n * sizeof(int16_t));
    memset(s.spk_prev, 0, n * sizeof(int16_t));
    return kTfLiteOk;
}

// --------- SNN_CONV_POOL_LIF 커스텀 커널 ---------

struct CplOpData {
    int32_t *mult;      // per-channel (persistent)
    int32_t *shift;
    int scratch_idx;    // im2col
    int slot;           // LIF 상태 슬롯
    int16_t beta, threshold, spike_val;
};

CplOpData g_cpl[2];

TfLiteStatus CplPrepareImpl(TfLiteContext *context, TfLiteNode *node, int slot,
                            int16_t beta, int16_t threshold) {
    CplOpData &d = g_cpl[slot];
    d.slot = slot;
    d.beta = beta;
    d.threshold = threshold;
    d.spike_val = ONE_Q15;

    tflite::MicroContext *mc = tflite::GetMicroContext(context);
    TfLiteTensor *in = mc->AllocateTempInputTensor(node, 0);
    TfLiteTensor *w = mc->AllocateTempInputTensor(node, 1);
    TfLiteTensor *out = mc->AllocateTempOutputTensor(node, 0);
    if (!in || !w || !out) return kTfLiteError;

    const int32_t Ci = w->dims->data[3];
    const int32_t Co = w->dims->data[0];
    const int32_t rhs_cols = w->dims->data[1] * w->dims->data[2] * Ci;
    const int32_t out_elems = out->dims->data[1] * out->dims->data[2] * out->dims->data[3];

    // per-channel requantize 파라미터 (퍼시스턴트 아레나)
    d.mult = (int32_t *)context->AllocatePersistentBuffer(context, Co * sizeof(int32_t));
    d.shift = (int32_t *)context->AllocatePersistentBuffer(context, Co * sizeof(int32_t));
    if (!d.mult || !d.shift) return kTfLiteError;

    const auto *in_q = (const TfLiteAffineQuantization *)in->quantization.params;
    const auto *w_q = (const TfLiteAffineQuantization *)w->quantization.params;
    const auto *out_q = (const TfLiteAffineQuantization *)out->quantization.params;
    const double in_scale = in_q->scale->data[0];
    const double out_scale = out_q->scale->data[0];
    for (int32_t c = 0; c < Co; ++c) {
        const double eff = in_scale * (double)w_q->scale->data[c] / out_scale;
        int mult_shift;
        tflite::QuantizeMultiplier(eff, &d.mult[c], &mult_shift);
        d.shift[c] = mult_shift;
    }

    // im2col 스크래치 (아레나 플래너 관리)
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, 2 * rhs_cols * sizeof(int16_t), &d.scratch_idx));

    // LIF 상태 (퍼시스턴트 아레나)
    TF_LITE_ENSURE_STATUS(AllocLifState(context, slot, out_elems));

    mc->DeallocateTempTfLiteTensor(in);
    mc->DeallocateTempTfLiteTensor(w);
    mc->DeallocateTempTfLiteTensor(out);
    return kTfLiteOk;
}

TfLiteStatus CplEvalImpl(TfLiteContext *context, TfLiteNode *node, int slot) {
    const CplOpData &d = g_cpl[slot];
    const TfLiteEvalTensor *in = tflite::micro::GetEvalInput(context, node, 0);
    const TfLiteEvalTensor *w = tflite::micro::GetEvalInput(context, node, 1);
    const TfLiteEvalTensor *bias = tflite::micro::GetEvalInput(context, node, 2);
    TfLiteEvalTensor *out = tflite::micro::GetEvalOutput(context, node, 0);

    const cmsis_nn_context ctx = {
        .buf = context->GetScratchBuffer(context, d.scratch_idx),
        .size = 0,
    };
    const cmsis_nn_conv_params conv_params = {
        .input_offset  = 0,
        .output_offset = 0,
        .stride        = { .w = 2, .h = 2 },
        .padding       = { .w = 2, .h = 2 },
        .dilation      = { .w = 1, .h = 1 },
        .activation    = { .min = -32768, .max = 32767 },
    };
    const cmsis_nn_per_channel_quant_params qp = { d.mult, d.shift };
    const cmsis_nn_dims in_d = { 1, in->dims->data[1], in->dims->data[2], in->dims->data[3] };
    const cmsis_nn_dims flt_d = { w->dims->data[0], w->dims->data[1],
                                  w->dims->data[2], w->dims->data[3] };
    const cmsis_nn_dims out_d = { 1, out->dims->data[1], out->dims->data[2], out->dims->data[3] };
    const cmsis_nn_dims bias_d = { 0, 0, 0, 0 };
    const snn_lif_ctx lif = { g_lif_states[d.slot].mem, g_lif_states[d.slot].spk_prev,
                              d.beta, d.threshold, d.spike_val };

    if (snn_convolve_pool_lif_s16(&ctx, &conv_params, &qp,
                                  &in_d, in->data.i16, &flt_d, w->data.i16,
                                  &bias_d, bias->data.i64, &lif,
                                  &out_d, out->data.i16) != ARM_CMSIS_NN_SUCCESS) {
        return kTfLiteError;
    }
    return kTfLiteOk;
}

TfLiteStatus Cpl1Prepare(TfLiteContext *c, TfLiteNode *n) {
    return CplPrepareImpl(c, n, 0, snn_lif1_beta, snn_lif1_threshold);
}
TfLiteStatus Cpl2Prepare(TfLiteContext *c, TfLiteNode *n) {
    return CplPrepareImpl(c, n, 1, snn_lif2_beta, snn_lif2_threshold);
}
TfLiteStatus Cpl1Eval(TfLiteContext *c, TfLiteNode *n) { return CplEvalImpl(c, n, 0); }
TfLiteStatus Cpl2Eval(TfLiteContext *c, TfLiteNode *n) { return CplEvalImpl(c, n, 1); }

// --------- SNN_LIF_OUT 커스텀 커널 (FC 출력 -> 출력 스파이크) ---------

TfLiteStatus LifOutPrepare(TfLiteContext *context, TfLiteNode *node) {
    tflite::MicroContext *mc = tflite::GetMicroContext(context);
    TfLiteTensor *out = mc->AllocateTempOutputTensor(node, 0);
    if (!out) return kTfLiteError;
    const int32_t n = out->dims->data[out->dims->size - 1];
    mc->DeallocateTempTfLiteTensor(out);
    TF_LITE_ENSURE_STATUS(AllocLifState(context, 2, n));
    g_lif_out_mem = g_lif_states[2].mem;
    return kTfLiteOk;
}

TfLiteStatus LifOutEval(TfLiteContext *context, TfLiteNode *node) {
    const TfLiteEvalTensor *in = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor *out = tflite::micro::GetEvalOutput(context, node, 0);
    snn_lif_s16(EvalElementCount(in), in->data.i16, out->data.i16,
                g_lif_states[2].mem, g_lif_states[2].spk_prev,
                snn_lif_out_beta, snn_lif_out_threshold, ONE_Q15);
    return kTfLiteOk;
}

TFLMRegistration MakeReg(TfLiteStatus (*prepare)(TfLiteContext *, TfLiteNode *),
                         TfLiteStatus (*invoke)(TfLiteContext *, TfLiteNode *)) {
    TFLMRegistration r = {};
    r.prepare = prepare;
    r.invoke = invoke;
    return r;
}

TFLMRegistration g_cpl1_reg = MakeReg(Cpl1Prepare, Cpl1Eval);
TFLMRegistration g_cpl2_reg = MakeReg(Cpl2Prepare, Cpl2Eval);
TFLMRegistration g_lif_out_reg = MakeReg(LifOutPrepare, LifOutEval);

} // namespace

// --------- 공개 API (snn.h) ---------

extern "C" void snn_accel_init(void) {
    if (g_ready) return;

    const tflite::Model *model = tflite::GetModel(g_snn_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("model schema %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();  // -> CMSIS-NN Register_FULLY_CONNECTED
    resolver.AddCustom("SNN_CONV_POOL_LIF1", &g_cpl1_reg);
    resolver.AddCustom("SNN_CONV_POOL_LIF2", &g_cpl2_reg);
    resolver.AddCustom("SNN_LIF_OUT", &g_lif_out_reg);

    static tflite::MicroInterpreter interpreter(model, resolver,
                                                g_tensor_arena, SNN_TFLM_ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed (arena %d)", SNN_TFLM_ARENA_SIZE);
        return;
    }
    g_interpreter = &interpreter;
    g_ready = 1;
}

extern "C" void snn_reset_state(void) {
    if (!g_ready) snn_accel_init();
    for (const LifState &s : g_lif_states) {
        if (s.mem) memset(s.mem, 0, s.n * sizeof(int16_t));
        if (s.spk_prev) memset(s.spk_prev, 0, s.n * sizeof(int16_t));
    }
}

extern "C" void snn_forward_step(const int16_t *x,
                                 int16_t *spk_out,
                                 int16_t *mem_out) {
    if (!g_ready) snn_accel_init();
    if (!g_ready) { // 초기화 실패 시 무해한 출력
        memset(spk_out, 0, 10 * sizeof(int16_t));
        memset(mem_out, 0, 10 * sizeof(int16_t));
        return;
    }

    memcpy(g_interpreter->input(0)->data.i16, x, 32 * 32 * 3 * sizeof(int16_t));

    if (g_interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        memset(spk_out, 0, 10 * sizeof(int16_t));
        memset(mem_out, 0, 10 * sizeof(int16_t));
        return;
    }

    memcpy(spk_out, g_interpreter->output(0)->data.i16, 10 * sizeof(int16_t));
    memcpy(mem_out, g_lif_out_mem, 10 * sizeof(int16_t));
}

// 호스트 검증용: 아레나 실사용량
extern "C" size_t snn_tflm_arena_used(void) {
    return g_ready ? g_interpreter->arena_used_bytes() : 0;
}
