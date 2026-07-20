/*
 * [실험 V2 전용] 비퓨전 TFLite 모델 생성기 — 표준 int16x8 그래프:
 *   input -> CONV_2D(5x5,SAME) -> AVERAGE_POOL_2D(2x2,s2) -> SNN_LIF1(custom)
 *   -> CONV_2D -> AVERAGE_POOL_2D -> SNN_LIF2(custom)
 *   -> FULLY_CONNECTED -> SNN_LIF_OUT(custom)
 * (논문의 "CMSIS-NN(TFLM 표준 커널)" 비교군. 퓨전 버전과 동일 양자화 정합)
 */
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "../Core/Src/snn_weights.c"
#include "../Core/Src/snn_weights_fused.c"
}

using namespace tflite;
using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;

static constexpr double kActScale = 1.0 / 32768.0;

int main() {
    flatbuffers::DefaultAllocator alloc;
    FlatBufferBuilder fbb(1 << 21, &alloc);

    std::vector<Offset<Buffer>> buffers;
    buffers.push_back(CreateBuffer(fbb));

    auto add_buffer = [&](const void *data, size_t bytes) -> uint32_t {
        auto vec = fbb.CreateVector(reinterpret_cast<const uint8_t *>(data), bytes);
        buffers.push_back(CreateBuffer(fbb, vec));
        return (uint32_t)(buffers.size() - 1);
    };

    std::vector<Offset<Tensor>> tensors;
    auto add_act_i16 = [&](std::vector<int32_t> shape, const char *name) -> int32_t {
        auto q = CreateQuantizationParameters(
            fbb, 0, 0, fbb.CreateVector<float>({(float)kActScale}),
            fbb.CreateVector<int64_t>({0}));
        tensors.push_back(CreateTensor(fbb, fbb.CreateVector(shape), TensorType_INT16,
                                       0, fbb.CreateString(name), q));
        return (int32_t)(tensors.size() - 1);
    };

    auto add_conv_weights = [&](const int8_t *w5, int Co, int Ci, int16_t ws_q15,
                                const char *name) -> int32_t {
        std::vector<int8_t> ohwi((size_t)Co * 25 * Ci);
        for (int co = 0; co < Co; ++co)
            for (int ci = 0; ci < Ci; ++ci)
                for (int kh = 0; kh < 5; ++kh)
                    for (int kw = 0; kw < 5; ++kw)
                        ohwi[(((size_t)co * 5 + kh) * 5 + kw) * Ci + ci] =
                            w5[(((size_t)co * Ci + ci) * 5 + kh) * 5 + kw];
        std::vector<float> scales(Co, (float)((double)ws_q15 / 32768.0));
        std::vector<int64_t> zps(Co, 0);
        auto q = CreateQuantizationParameters(fbb, 0, 0, fbb.CreateVector(scales),
                                              fbb.CreateVector(zps),
                                              QuantizationDetails_NONE, 0, 0);
        uint32_t buf = add_buffer(ohwi.data(), ohwi.size());
        tensors.push_back(CreateTensor(fbb, fbb.CreateVector<int32_t>({Co, 5, 5, Ci}),
                                       TensorType_INT8, buf, fbb.CreateString(name), q));
        return (int32_t)(tensors.size() - 1);
    };

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

    int32_t t_in = add_act_i16({1, 32, 32, 3}, "input_spikes");
    int32_t t_w1 = add_conv_weights((const int8_t *)snn_conv1_weight, 32, 3,
                                    snn_conv1_weight_scale, "conv1_w");
    int32_t t_b1 = add_bias((const int8_t *)snn_conv1_bias, 32, snn_conv1_bias_scale,
                            (double)snn_conv1_weight_scale / 32768.0, "conv1_b");
    int32_t t_c1 = add_act_i16({1, 32, 32, 32}, "conv1_out");
    int32_t t_p1 = add_act_i16({1, 16, 16, 32}, "pool1_out");
    int32_t t_s1 = add_act_i16({1, 16, 16, 32}, "lif1_spikes");
    int32_t t_w2 = add_conv_weights((const int8_t *)snn_conv2_weight, 64, 32,
                                    snn_conv2_weight_scale, "conv2_w");
    int32_t t_b2 = add_bias((const int8_t *)snn_conv2_bias, 64, snn_conv2_bias_scale,
                            (double)snn_conv2_weight_scale / 32768.0, "conv2_b");
    int32_t t_c2 = add_act_i16({1, 16, 16, 64}, "conv2_out");
    int32_t t_p2 = add_act_i16({1, 8, 8, 64}, "pool2_out");
    int32_t t_s2 = add_act_i16({1, 8, 8, 64}, "lif2_spikes");

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
    int32_t t_bf = add_bias((const int8_t *)snn_fc_bias, 10, snn_fc_bias_scale,
                            (double)snn_fc_weight_scale / 32768.0, "fc_b");
    int32_t t_fc = add_act_i16({1, 10}, "fc_out");
    int32_t t_out = add_act_i16({1, 10}, "spk_out");

    std::vector<Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(fbb, 3, 0, 3, BuiltinOperator_CONV_2D),
        CreateOperatorCode(fbb, 1, 0, 3, BuiltinOperator_AVERAGE_POOL_2D),
        CreateOperatorCode(fbb, 9, 0, 7, BuiltinOperator_FULLY_CONNECTED),
        CreateOperatorCode(fbb, 32, fbb.CreateString("SNN_LIF1"), 1, BuiltinOperator_CUSTOM),
        CreateOperatorCode(fbb, 32, fbb.CreateString("SNN_LIF2"), 1, BuiltinOperator_CUSTOM),
        CreateOperatorCode(fbb, 32, fbb.CreateString("SNN_LIF_OUT"), 1, BuiltinOperator_CUSTOM),
    };

    auto conv_opts = CreateConv2DOptions(fbb, Padding_SAME, 1, 1,
                                         ActivationFunctionType_NONE, 1, 1);
    auto pool_opts = CreatePool2DOptions(fbb, Padding_VALID, 2, 2, 2, 2,
                                         ActivationFunctionType_NONE);
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
        op(0, {t_in, t_w1, t_b1}, {t_c1}, BuiltinOptions_Conv2DOptions, conv_opts.Union()),
        op(1, {t_c1}, {t_p1}, BuiltinOptions_Pool2DOptions, pool_opts.Union()),
        op(3, {t_p1}, {t_s1}),
        op(0, {t_s1, t_w2, t_b2}, {t_c2}, BuiltinOptions_Conv2DOptions, conv_opts.Union()),
        op(1, {t_c2}, {t_p2}, BuiltinOptions_Pool2DOptions, pool_opts.Union()),
        op(4, {t_p2}, {t_s2}),
        op(2, {t_s2, t_wf, t_bf}, {t_fc}, BuiltinOptions_FullyConnectedOptions, fc_opts.Union()),
        op(5, {t_fc}, {t_out}),
    };

    auto subgraph = CreateSubGraph(fbb, fbb.CreateVector(tensors),
                                   fbb.CreateVector<int32_t>({t_in}),
                                   fbb.CreateVector<int32_t>({t_out}),
                                   fbb.CreateVector(ops), fbb.CreateString("snn_step"));
    auto model = CreateModel(fbb, 3, fbb.CreateVector(opcodes),
                             fbb.CreateVector<Offset<SubGraph>>({subgraph}),
                             fbb.CreateString("SNN unfused (int16x8 + LIF custom ops)"),
                             fbb.CreateVector(buffers));
    FinishModelBuffer(fbb, model);

    const uint8_t *data = fbb.GetBufferPointer();
    size_t size = fbb.GetSize();
    FILE *f = fopen("Core/Src/snn_model_data.cc", "w");
    if (!f) { perror("snn_model_data.cc"); return 1; }
    fprintf(f, "/* [실험 V2] 비퓨전 모델 — gen_tflite_model_unfused.cc 생성 */\n"
               "#include \"snn_model_data.h\"\n\n"
               "alignas(16) const unsigned char g_snn_model_data[%zu] = {\n", size);
    for (size_t i = 0; i < size; ++i)
        fprintf(f, "0x%02x,%s", data[i], (i % 16 == 15) ? "\n" : " ");
    fprintf(f, "};\nconst unsigned int g_snn_model_data_len = %zu;\n", size);
    fclose(f);
    f = fopen("Core/Inc/snn_model_data.h", "w");
    fprintf(f, "#ifndef SNN_MODEL_DATA_H\n#define SNN_MODEL_DATA_H\n#include <stdint.h>\n"
               "alignas(16) extern const unsigned char g_snn_model_data[%zu];\n"
               "extern const unsigned int g_snn_model_data_len;\n#endif\n", size);
    fclose(f);
    printf("generated unfused model: %zu bytes\n", size);
    return 0;
}
