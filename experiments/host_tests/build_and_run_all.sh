#!/bin/bash
# 호스트 검증 테스트 전체 빌드+실행 (EXPERIMENTS.md의 실험 H1~H5)
# 사용: experiments/host_tests/build_and_run_all.sh   (저장소 어디서 실행해도 됨)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IC="$REPO/ImageClassification"
OUT="$HERE/bin"; mkdir -p "$OUT"

echo "== H1: 양자화 매핑 (test_mapping) =="
clang -O2 -o "$OUT/test_mapping" "$HERE/test_mapping.c"
"$OUT/test_mapping"

echo "== H2: LIF SIMD 전수 검증 (test_lif, DSP목) =="
clang -O2 -DARM_MATH_DSP -I "$HERE/simd_mock" -I "$IC/Core/Inc" -I "$IC/Core/Src" \
      -o "$OUT/test_lif" "$HERE/test_lif.c"
"$OUT/test_lif"

echo "== H3: 인코더 (test_encode) =="
clang -O2 -I "$IC/Core/Inc" -I "$IC/Core/Src" -o "$OUT/test_encode" "$HERE/test_encode.c" -lm
"$OUT/test_encode"

echo "== H4: 퓨전 커널 (test_fused, 스칼라) =="
clang -O2 -I "$IC/Core/Inc" -I "$IC/Core/Src" -I "$IC/third_party/cmsis_nn/Include" \
      -o "$OUT/test_fused_scalar" "$HERE/test_fused.c" -lm
"$OUT/test_fused_scalar"
echo "== H4: 퓨전 커널 (test_fused, SIMD목) =="
clang -O2 -DARM_MATH_DSP -I "$HERE/simd_mock" -I "$IC/Core/Inc" -I "$IC/Core/Src" \
      -I "$IC/third_party/cmsis_nn/Include" -o "$OUT/test_fused_dsp" "$HERE/test_fused.c" -lm
"$OUT/test_fused_dsp"

echo "== H5: TFLM 전체 파이프라인 (test_tflm) =="
OBJ="$OUT/tflm_obj"; mkdir -p "$OBJ"
CXXFLAGS="-std=c++17 -O2 -DTF_LITE_STATIC_MEMORY -DCMSIS_NN \
  -I $IC/tensorflow-lite -I $IC/third_party/flatbuffers/include \
  -I $IC/third_party/gemmlowp -I $IC/third_party/ruy \
  -I $IC/third_party/cmsis_nn -I $IC/third_party/cmsis_nn/Include -I $IC/Core/Inc -w"
CFLAGS="-std=c11 -O2 -I $IC/third_party/cmsis_nn/Include -I $IC/Core/Inc -w"

CXX_SOURCES=$(ls "$IC"/tensorflow-lite/tensorflow/lite/micro/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/micro/arena_allocator/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/micro/memory_planner/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/micro/tflite_bridge/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/micro/kernels/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/micro/kernels/cmsis_nn/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/core/c/common.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/core/api/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/kernels/internal/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/kernels/internal/reference/*.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/kernels/kernel_util.cc \
  "$IC"/tensorflow-lite/tensorflow/lite/schema/schema_utils.cc \
  | grep -vE '(test_helpers|test_helper_custom_ops|fake_micro_context|mock_micro_graph|recording_micro|kernel_runner)')
for k in add conv depthwise_conv fully_connected mul pooling softmax svdf unidirectional_sequence_lstm; do
  CXX_SOURCES=$(echo "$CXX_SOURCES" | grep -v "/micro/kernels/$k\.cc$")
done
CXX_SOURCES="$CXX_SOURCES $IC/Core/Src/snn_tflm.cc $IC/Core/Src/snn_model_data.cc"
C_SOURCES="$(ls "$IC"/third_party/cmsis_nn/Source/*/*.c) $IC/Core/Src/snn_lif_s16.c $IC/Core/Src/snn_conv_pool_lif_s16.c $IC/Core/Src/snn_weights.c $IC/Core/Src/snn_weights_fused.c"

compile_one() {
  f="$1"; o="$OBJ/$(echo "$f" | shasum | cut -c1-12).o"
  if [ ! -f "$o" ] || [ "$f" -nt "$o" ]; then
    case "$f" in
      *.cc) clang++ $CXXFLAGS -c "$f" -o "$o" ;;
      *.c)  clang $CFLAGS -c "$f" -o "$o" ;;
    esac
  fi
}
export -f compile_one 2>/dev/null || true
for f in $CXX_SOURCES $C_SOURCES; do compile_one "$f"; done
clang++ $CXXFLAGS -o "$OUT/test_tflm" "$HERE/test_tflm.cc" "$OBJ"/*.o
"$OUT/test_tflm"

echo "ALL HOST TESTS DONE"
