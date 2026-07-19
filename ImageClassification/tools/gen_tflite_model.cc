/*
 * SNN TFLite 모델 빌드타임 생성기 (호스트에서 실행).
 *
 * 타임스텝 1회분 그래프. Conv+AvgPool+LIF는 퓨전 커스텀 오퍼 하나로 내장한다:
 *   input[1,32,32,3] -> SNN_CONV_POOL_LIF1 -> [1,16,16,32]
 *   -> SNN_CONV_POOL_LIF2 -> [1,8,8,64]
 *   -> FULLY_CONNECTED(int16x8, TFLM CMSIS-NN) -> SNN_LIF_OUT -> [1,10]
 *
 * 퓨전 오퍼 입력: (활성값, int16 퓨전 가중치 [Co,6,6,Ci], int64 bias)
 *   - 가중치 = 5x5 커널 4개(2x2 시프트)의 합 (6x6/s2와 등가, snn_weights_fused와 동일)
 *   - 가중치 scale = ws_q15/(32768*4) — avgpool의 /4를 스케일에 흡수
 *   - bias scale = in_scale * w_scale, 값 = round(4*b*bs*32768/ws)
 *   - 활성값 scale = 1/32768 (Q15, zero_point 0), 스파이크 값 32767
 * 커널(snn_tflm.cc)은 Prepare에서 텐서 스케일로 requantize 파라미터를 계산하고
 * 아레나(퍼시스턴트/스크래치)를 할당한 뒤 snn_convolve_pool_lif_s16을 호출한다.
 *
 * 실행 (snn_weights.c 변경 시 재실행):
 *   cd ImageClassification
 *   clang++ -std=c++17 -O2 -I tensorflow-lite -I third_party/flatbuffers/include \
 *       -o /tmp/gentfl tools/gen_tflite_model.cc && /tmp/gentfl
 *   (Core/Src/snn_model_data.cc 와 Core/Inc/snn_model_data.h 갱신)
 */
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "../Core/Src/snn_weights.c"        /* 원본 int8 가중치/스케일 */
#include "../Core/Src/snn_weights_fused.c"  /* snn_fc_weight_hwc (HWC 퍼뮤트 FC) */
}

using namespace tflite;
using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;

static constexpr double kActScale = 1.0 / 32768.0; /* Q15 */

struct TensorSpec {
    Offset<Tensor> off;
};

int main() {
    /* TFLM 트림판 flatbuffers는 null allocator 폴백이 제거되어 있어 명시 전달 필요 */
    flatbuffers::DefaultAllocator alloc;
    FlatBufferBuilder fbb(1 << 21, &alloc);

    std::vector<Offset<Buffer>> buffers;
    buffers.push_back(CreateBuffer(fbb)); /* buffer 0: 빈 센티널 */

    auto add_buffer = [&](const void *data, size_t bytes) -> uint32_t {
        auto vec = fbb.CreateVector(reinterpret_cast<const uint8_t *>(data), bytes);
        buffers.push_back(CreateBuffer(fbb, vec));
        return (uint32_t)(buffers.size() - 1);
    };

    std::vector<Offset<Tensor>> tensors;
    auto add_act_i16 = [&](std::vector<int32_t> shape, const char *name) -> int32_t {
        auto q = CreateQuantizationParameters(
            fbb, 0, 0,
            fbb.CreateVector<float>({(float)kActScale}),
            fbb.CreateVector<int64_t>({0}));
        tensors.push_back(CreateTensor(fbb, fbb.CreateVector(shape), TensorType_INT16,
                                       /*buffer=*/0, fbb.CreateString(name), q));
        return (int32_t)(tensors.size() - 1);
    };

    /* 퓨전 가중치: [Co][Ci][5][5](CHW) int8 -> 6x6 합산 int16 [Co][6][6][Ci].
     * scale = ws/(32768*4) per-channel 복제 (/4 = avgpool 평균) */
    auto add_fused_weights = [&](const int8_t *w5, int Co, int Ci, int16_t ws_q15,
                                 const char *name) -> int32_t {
        std::vector<int16_t> w6((size_t)Co * 36 * Ci);
        for (int co = 0; co < Co; ++co)
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    for (int ci = 0; ci < Ci; ++ci) {
                        int v = 0;
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx) {
                                int ka = a - dy, kb = b - dx;
                                if (ka >= 0 && ka < 5 && kb >= 0 && kb < 5)
                                    v += w5[(((size_t)co * Ci + ci) * 5 + ka) * 5 + kb];
                            }
                        w6[(((size_t)co * 6 + a) * 6 + b) * Ci + ci] = (int16_t)v;
                    }
        std::vector<float> scales(Co, (float)((double)ws_q15 / 32768.0 / 4.0));
        std::vector<int64_t> zps(Co, 0);
        auto q = CreateQuantizationParameters(fbb, 0, 0, fbb.CreateVector(scales),
                                              fbb.CreateVector(zps),
                                              QuantizationDetails_NONE, 0,
                                              /*quantized_dimension=*/0);
        uint32_t buf = add_buffer(w6.data(), w6.size() * sizeof(int16_t));
        tensors.push_back(CreateTensor(fbb, fbb.CreateVector<int32_t>({Co, 6, 6, Ci}),
                                       TensorType_INT16, buf, fbb.CreateString(name), q));
        return (int32_t)(tensors.size() - 1);
    };

    /* bias(int64): 실값 b*bs/2^15 를 scale(in*w)로 양자화. w_scale 인자는 실배율 */
    auto add_bias = [&](const int8_t *b, int n, int16_t bs_q15, double w_scale,
                        const char *name) -> int32_t {
        std::vector<int64_t> bq(n);
        std::vector<float> scales(n, (float)(kActScale * w_scale));
        std::vector<int64_t> zps(n, 0);
        for (int i = 0; i < n; ++i)
            bq[i] = llround(((double)b[i] * (double)bs_q15 / 32768.0) / (kActScale * w_scale));
        auto q = CreateQuantizationParameters(fbb, 0, 0, fbb.CreateVector(scales),
                                              fbb.CreateVector(zps),
                                              QuantizationDetails_NONE, 0, 0);
        uint32_t buf = add_buffer(bq.data(), bq.size() * sizeof(int64_t));
        tensors.push_back(CreateTensor(fbb, fbb.CreateVector<int32_t>({n}),
                                       TensorType_INT64, buf, fbb.CreateString(name), q));
        return (int32_t)(tensors.size() - 1);
    };

    /* ---- 텐서 ---- */
    int32_t t_in    = add_act_i16({1, 32, 32, 3}, "input_spikes");
    int32_t t_w1    = add_fused_weights((const int8_t *)snn_conv1_weight, 32, 3,
                                        snn_conv1_weight_scale, "cpl1_w");
    int32_t t_b1    = add_bias((const int8_t *)snn_conv1_bias, 32, snn_conv1_bias_scale,
                               (double)snn_conv1_weight_scale / 32768.0 / 4.0, "cpl1_b");
    int32_t t_s1    = add_act_i16({1, 16, 16, 32}, "lif1_spikes");
    int32_t t_w2    = add_fused_weights((const int8_t *)snn_conv2_weight, 64, 32,
                                        snn_conv2_weight_scale, "cpl2_w");
    int32_t t_b2    = add_bias((const int8_t *)snn_conv2_bias, 64, snn_conv2_bias_scale,
                               (double)snn_conv2_weight_scale / 32768.0 / 4.0, "cpl2_b");
    int32_t t_s2    = add_act_i16({1, 8, 8, 64}, "lif2_spikes");

    /* FC 가중치 [10,4096]: 입력이 NHWC flatten이므로 HWC 퍼뮤트본 사용, per-tensor */
    int32_t t_wf;
    {
        std::vector<float> sc = {(float)((double)snn_fc_weight_scale / 32768.0)};
        std::vector<int64_t> zp = {0};
        auto q = CreateQuantizationParameters(fbb, 0, 0, fbb.CreateVector(sc),
                                              fbb.CreateVector(zp));
        uint32_t buf = add_buffer(snn_fc_weight_hwc, 10 * 4096);
        tensors.push_back(CreateTensor(fbb, fbb.CreateVector<int32_t>({10, 4096}),
                                       TensorType_INT8, buf, fbb.CreateString("fc_w"), q));
        t_wf = (int32_t)(tensors.size() - 1);
    }
    int32_t t_bf    = add_bias((const int8_t *)snn_fc_bias, 10, snn_fc_bias_scale,
                               (double)snn_fc_weight_scale / 32768.0, "fc_b");
    int32_t t_fc    = add_act_i16({1, 10}, "fc_out");
    int32_t t_out   = add_act_i16({1, 10}, "spk_out");

    /* ---- 오퍼레이터 코드 ---- */
    std::vector<Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(fbb, /*deprecated=*/9, 0, /*version=*/7, BuiltinOperator_FULLY_CONNECTED),
        CreateOperatorCode(fbb, /*deprecated=*/32, fbb.CreateString("SNN_CONV_POOL_LIF1"), 1, BuiltinOperator_CUSTOM),
        CreateOperatorCode(fbb, /*deprecated=*/32, fbb.CreateString("SNN_CONV_POOL_LIF2"), 1, BuiltinOperator_CUSTOM),
        CreateOperatorCode(fbb, /*deprecated=*/32, fbb.CreateString("SNN_LIF_OUT"), 1, BuiltinOperator_CUSTOM),
    };

    auto fc_opts = CreateFullyConnectedOptions(fbb, ActivationFunctionType_NONE,
                                               FullyConnectedOptionsWeightsFormat_DEFAULT,
                                               false, false);

    auto op = [&](uint32_t opcode_idx, std::vector<int32_t> in, std::vector<int32_t> out,
                  BuiltinOptions bt = BuiltinOptions_NONE,
                  flatbuffers::Offset<void> opts = 0) {
        return CreateOperator(fbb, opcode_idx, fbb.CreateVector(in), fbb.CreateVector(out),
                              bt, opts);
    };

    std::vector<Offset<Operator>> ops = {
        op(1, {t_in, t_w1, t_b1}, {t_s1}),
        op(2, {t_s1, t_w2, t_b2}, {t_s2}),
        op(0, {t_s2, t_wf, t_bf}, {t_fc}, BuiltinOptions_FullyConnectedOptions, fc_opts.Union()),
        op(3, {t_fc}, {t_out}),
    };

    auto subgraph = CreateSubGraph(fbb, fbb.CreateVector(tensors),
                                   fbb.CreateVector<int32_t>({t_in}),
                                   fbb.CreateVector<int32_t>({t_out}),
                                   fbb.CreateVector(ops), fbb.CreateString("snn_step"));

    auto model = CreateModel(fbb, /*TFLITE_SCHEMA_VERSION=*/3, fbb.CreateVector(opcodes),
                             fbb.CreateVector<Offset<SubGraph>>({subgraph}),
                             fbb.CreateString("SNN CIFAR-10 single timestep (fused Conv+Pool+LIF custom ops)"),
                             fbb.CreateVector(buffers));
    FinishModelBuffer(fbb, model);

    /* ---- C 배열로 방출 ---- */
    const uint8_t *data = fbb.GetBufferPointer();
    size_t size = fbb.GetSize();

    FILE *f = fopen("Core/Src/snn_model_data.cc", "w");
    if (!f) { perror("snn_model_data.cc"); return 1; }
    fprintf(f, "/* tools/gen_tflite_model.cc 가 생성한 파일 — 직접 수정 금지.\n"
               " * snn_weights.c 변경 시 생성기를 재실행할 것. */\n"
               "#include \"snn_model_data.h\"\n\n"
               "alignas(16) const unsigned char g_snn_model_data[%zu] = {\n", size);
    for (size_t i = 0; i < size; ++i)
        fprintf(f, "0x%02x,%s", data[i], (i % 16 == 15) ? "\n" : " ");
    fprintf(f, "};\nconst unsigned int g_snn_model_data_len = %zu;\n", size);
    fclose(f);

    f = fopen("Core/Inc/snn_model_data.h", "w");
    if (!f) { perror("snn_model_data.h"); return 1; }
    fprintf(f, "/* tools/gen_tflite_model.cc 가 생성한 파일 — 직접 수정 금지. */\n"
               "#ifndef SNN_MODEL_DATA_H\n#define SNN_MODEL_DATA_H\n\n"
               "alignas(16) extern const unsigned char g_snn_model_data[%zu];\n"
               "extern const unsigned int g_snn_model_data_len;\n\n#endif\n", size);
    fclose(f);

    printf("generated: snn_model_data.cc (%zu bytes)\n", size);
    return 0;
}
