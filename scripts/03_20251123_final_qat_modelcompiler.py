#%%
import torch
import torch.nn as nn
import numpy as np
import os
import snntorch as snn
from snntorch import surrogate, utils
from snntorch import spikegen

import brevitas.nn as qnn       # ✨ QAT/양자화 레이어
from chacha_py import NumpySNN  # 사용 중인 numpy 기반 SNN 클래스

# ---------------------------------------------------------
# 1. Quant SNN 정의 + weight 로드
# ---------------------------------------------------------
beta = 0.9
spike_grad = surrogate.fast_sigmoid()

net = nn.Sequential(
    # 입력: [B, 3, 32, 32]
    qnn.QuantConv2d(  # Conv2d 양자화 버전
        3, 32, 5, padding=2,
        weight_bit_width=8,  # 8-bit weight
        bias=True,
    ),
    nn.AvgPool2d(2),  # 32x32 -> 16x16
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    qnn.QuantConv2d(
        32, 64, 5, padding=2,
        weight_bit_width=8,
        bias=True
    ),
    nn.AvgPool2d(2),  # 16x16 -> 8x8
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Flatten(),           # [B, 64*8*8]
    qnn.QuantLinear(        # Linear 양자화 버전
        64 * 8 * 8, 10,
        weight_bit_width=8,
        bias=True
    ),

    # 마지막 레이어: spikes와 membrane state를 모두 반환
    snn.Leaky(beta=beta, spike_grad=spike_grad,
              init_hidden=True, output=True)
)

# 학습된 QAT 체크포인트 로드
utils.reset(net)
net.load_state_dict(torch.load("0096_acc50_snn_cifar10.pth", map_location="cpu"))
net.eval()

# ---------------------------
# 헬퍼 함수들
# ---------------------------

def tensor_to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()

def shape_to_c_dims(arr: np.ndarray) -> str:
    """numpy shape -> C 배열 차원 문자열 ex) (32,3,5,5) -> [32][3][5][5]"""
    return "".join(f"[{d}]" for d in arr.shape)

def numpy_to_c_int_array(name: str, arr: np.ndarray, c_type: str = "int8_t") -> str:
    arr = np.asarray(arr)
    dim_str = shape_to_c_dims(arr)
    flat = arr.flatten()

    lines = []
    lines.append(f"const {c_type} {name}{dim_str} = {{")

    per_line = 16
    line = "    "
    for i, v in enumerate(flat):
        line += f"{int(v)}"
        if i != len(flat) - 1:
            line += ", "
        if (i + 1) % per_line == 0:
            lines.append(line)
            line = "    "
    if line.strip():
        lines.append(line)

    lines.append("};")
    lines.append("")
    return "\n".join(lines)

# (float 기반 배열/스칼라는 더 이상 사용하지 않지만, 필요시를 위해 남겨둠)
def numpy_to_c_float_array(name: str, arr: np.ndarray, c_type: str = "float") -> str:
    arr = np.asarray(arr, dtype=np.float32)
    dim_str = shape_to_c_dims(arr)
    flat = arr.flatten()

    lines = []
    lines.append(f"const {c_type} {name}{dim_str} = {{")

    per_line = 8
    line = "    "
    for i, v in enumerate(flat):
        line += f"{float(v):.8f}f"
        if i != len(flat) - 1:
            line += ", "
        if (i + 1) % per_line == 0:
            lines.append(line)
            line = "    "
    if line.strip():
        lines.append(line)

    lines.append("};")
    lines.append("")
    return "\n".join(lines)

def numpy_to_c_float_scalar(name: str, value: float, c_type: str = "float") -> str:
    return f"const {c_type} {name} = {float(value):.8f}f;\n"

def quantize_int8(arr: np.ndarray):
    """
    float bias를 int8 + scale 로 변환
    float ≈ int8 * scale
    """
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        scale = 1.0
        q = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = max_abs / 127.0
        q = np.round(arr / scale).astype(np.int32)
        q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale

# ---- 새로 추가: float -> Q15(int16) 변환, int 스칼라 출력 ----

def float_to_q15(x: np.ndarray) -> np.ndarray:
    """
    float -> Q15(int16) 고정소수점 변환
    실제 값 ≈ q / 2^15
    """
    arr = np.asarray(x, dtype=np.float32)
    q = np.round(arr * (1 << 15)).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q

def numpy_to_c_int_scalar(name: str, value: int, c_type: str = "int16_t") -> str:
    """
    단일 정수 스칼라를 C 코드로 출력
    ex) const int16_t snn_lif1_beta = 25600;
    """
    return f"const {c_type} {name} = {int(value)};\n"

#%%
def export_quant_snn_to_c_and_h(net: nn.Sequential,
                                c_filename: str = "quant_snn_weights.c",
                                h_filename: str = "quant_snn_weights.h",
                                var_prefix: str = "snn",
                                header_guard: str = "QUANT_SNN_WEIGHTS_H"):
    """
    MCU / CMSIS-NN을 고려한 순수 int export.

    Conv / Linear:
      - weight      : layer.int_weight() -> int8_t
      - weight_scale: weight_quant.scale() (float) -> Q15(int16_t)
                      실제값 ≈ weight_scale / 2^15
      - bias        : float bias -> (int8_t bias, Q15(int16_t) bias_scale)

    LIF:
      - beta, threshold : float scalar -> Q15(int16_t)

    생성되는 C/H 코드에는 float 타입/리터럴이 없음.
    """

    # net 구조 인덱스 고정
    conv1    = net[0]
    lif1     = net[2]
    conv2    = net[3]
    lif2     = net[5]
    fc       = net[7]
    lif_out  = net[8]

    # ----------------- C 소스(.c) -----------------
    c_parts = []
    c_parts.append("/* Auto-generated quantized SNN weights (definition, int-only) */")
    c_parts.append(f'#include "{os.path.basename(h_filename)}"')
    c_parts.append("")
    c_parts.append("/* All scales and LIF parameters are Q15 fixed-point (value / 2^15). */")
    c_parts.append("")

    # ---- Conv1 ----
    conv1_w_int = tensor_to_numpy(conv1.int_weight())              # int8
    conv1_w_scale = tensor_to_numpy(conv1.weight_quant.scale())    # float or tensor
    conv1_b_float = tensor_to_numpy(conv1.bias)                    # float
    conv1_b_int, conv1_b_scale = quantize_int8(conv1_b_float)      # int8 + float scale

    conv1_w_scale_q15 = float_to_q15(np.array(conv1_w_scale, ndmin=1))
    conv1_b_scale_q15 = float_to_q15(np.array([conv1_b_scale]))[0]

    c_parts.append("// Conv1")
    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv1_weight", conv1_w_int, "int8_t"))

    if conv1_w_scale_q15.size == 1:
        c_parts.append(
            numpy_to_c_int_scalar(f"{var_prefix}_conv1_weight_scale",
                                  int(conv1_w_scale_q15[0]), "int16_t")
        )
    else:
        c_parts.append(
            numpy_to_c_int_array(f"{var_prefix}_conv1_weight_scale",
                                 conv1_w_scale_q15, "int16_t")
        )

    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv1_bias", conv1_b_int, "int8_t"))
    c_parts.append(
        numpy_to_c_int_scalar(f"{var_prefix}_conv1_bias_scale",
                              int(conv1_b_scale_q15), "int16_t")
    )
    c_parts.append("")

    # ---- Conv2 ----
    conv2_w_int = tensor_to_numpy(conv2.int_weight())
    conv2_w_scale = tensor_to_numpy(conv2.weight_quant.scale())
    conv2_b_float = tensor_to_numpy(conv2.bias)
    conv2_b_int, conv2_b_scale = quantize_int8(conv2_b_float)

    conv2_w_scale_q15 = float_to_q15(np.array(conv2_w_scale, ndmin=1))
    conv2_b_scale_q15 = float_to_q15(np.array([conv2_b_scale]))[0]

    c_parts.append("// Conv2")
    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv2_weight", conv2_w_int, "int8_t"))

    if conv2_w_scale_q15.size == 1:
        c_parts.append(
            numpy_to_c_int_scalar(f"{var_prefix}_conv2_weight_scale",
                                  int(conv2_w_scale_q15[0]), "int16_t")
        )
    else:
        c_parts.append(
            numpy_to_c_int_array(f"{var_prefix}_conv2_weight_scale",
                                 conv2_w_scale_q15, "int16_t")
        )

    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv2_bias", conv2_b_int, "int8_t"))
    c_parts.append(
        numpy_to_c_int_scalar(f"{var_prefix}_conv2_bias_scale",
                              int(conv2_b_scale_q15), "int16_t")
    )
    c_parts.append("")

    # ---- FC (QuantLinear) ----
    fc_w_int = tensor_to_numpy(fc.int_weight())
    fc_w_scale = tensor_to_numpy(fc.weight_quant.scale())
    fc_b_float = tensor_to_numpy(fc.bias)
    fc_b_int, fc_b_scale = quantize_int8(fc_b_float)

    fc_w_scale_q15 = float_to_q15(np.array(fc_w_scale, ndmin=1))
    fc_b_scale_q15 = float_to_q15(np.array([fc_b_scale]))[0]

    c_parts.append("// FC")
    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_fc_weight", fc_w_int, "int8_t"))

    if fc_w_scale_q15.size == 1:
        c_parts.append(
            numpy_to_c_int_scalar(f"{var_prefix}_fc_weight_scale",
                                  int(fc_w_scale_q15[0]), "int16_t")
        )
    else:
        c_parts.append(
            numpy_to_c_int_array(f"{var_prefix}_fc_weight_scale",
                                 fc_w_scale_q15, "int16_t")
        )

    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_fc_bias", fc_b_int, "int8_t"))
    c_parts.append(
        numpy_to_c_int_scalar(f"{var_prefix}_fc_bias_scale",
                              int(fc_b_scale_q15), "int16_t")
    )
    c_parts.append("")

    # ---- LIF parameters (Q15) ----
    lif1_beta_q15      = float_to_q15(np.array([float(lif1.beta)]))[0]
    lif2_beta_q15      = float_to_q15(np.array([float(lif2.beta)]))[0]
    lif_out_beta_q15   = float_to_q15(np.array([float(lif_out.beta)]))[0]
    lif1_th_q15        = float_to_q15(np.array([float(lif1.threshold)]))[0]
    lif2_th_q15        = float_to_q15(np.array([float(lif2.threshold)]))[0]
    lif_out_th_q15     = float_to_q15(np.array([float(lif_out.threshold)]))[0]

    c_parts.append("// LIF parameters (Q15 fixed-point)")
    c_parts.append(numpy_to_c_int_scalar(f"{var_prefix}_lif1_beta",
                                         int(lif1_beta_q15), "int16_t"))
    c_parts.append(numpy_to_c_int_scalar(f"{var_prefix}_lif2_beta",
                                         int(lif2_beta_q15), "int16_t"))
    c_parts.append(numpy_to_c_int_scalar(f"{var_prefix}_lif_out_beta",
                                         int(lif_out_beta_q15), "int16_t"))
    c_parts.append(numpy_to_c_int_scalar(f"{var_prefix}_lif1_threshold",
                                         int(lif1_th_q15), "int16_t"))
    c_parts.append(numpy_to_c_int_scalar(f"{var_prefix}_lif2_threshold",
                                         int(lif2_th_q15), "int16_t"))
    c_parts.append(numpy_to_c_int_scalar(f"{var_prefix}_lif_out_threshold",
                                         int(lif_out_th_q15), "int16_t"))

    c_src = "\n".join(c_parts)

    # ----------------- 헤더(.h) -----------------
    h_parts = []
    h_parts.append("/* Auto-generated quantized SNN weights (declaration, int-only) */")
    h_parts.append(f"#ifndef {header_guard}")
    h_parts.append(f"#define {header_guard}")
    h_parts.append("")
    h_parts.append("#include <stdint.h>")
    h_parts.append("")
    h_parts.append("/* All scales and LIF parameters are Q15 fixed-point (value / 2^15). */")
    h_parts.append("")

    # Conv1
    conv1_w_dims = shape_to_c_dims(conv1_w_int)
    conv1_b_dims = shape_to_c_dims(conv1_b_int)

    h_parts.append("// Conv1")
    h_parts.append(
        f"extern const int8_t {var_prefix}_conv1_weight{conv1_w_dims};")

    if conv1_w_scale_q15.size == 1:
        h_parts.append(
            f"extern const int16_t {var_prefix}_conv1_weight_scale;")
    else:
        conv1_ws_dims = shape_to_c_dims(conv1_w_scale_q15)
        h_parts.append(
            f"extern const int16_t {var_prefix}_conv1_weight_scale{conv1_ws_dims};")

    h_parts.append(
        f"extern const int8_t {var_prefix}_conv1_bias{conv1_b_dims};")
    h_parts.append(
        f"extern const int16_t {var_prefix}_conv1_bias_scale;")
    h_parts.append("")

    # Conv2
    conv2_w_dims = shape_to_c_dims(conv2_w_int)
    conv2_b_dims = shape_to_c_dims(conv2_b_int)

    h_parts.append("// Conv2")
    h_parts.append(
        f"extern const int8_t {var_prefix}_conv2_weight{conv2_w_dims};")

    if conv2_w_scale_q15.size == 1:
        h_parts.append(
            f"extern const int16_t {var_prefix}_conv2_weight_scale;")
    else:
        conv2_ws_dims = shape_to_c_dims(conv2_w_scale_q15)
        h_parts.append(
            f"extern const int16_t {var_prefix}_conv2_weight_scale{conv2_ws_dims};")

    h_parts.append(
        f"extern const int8_t {var_prefix}_conv2_bias{conv2_b_dims};")
    h_parts.append(
        f"extern const int16_t {var_prefix}_conv2_bias_scale;")
    h_parts.append("")

    # FC
    fc_w_dims = shape_to_c_dims(fc_w_int)
    fc_b_dims = shape_to_c_dims(fc_b_int)

    h_parts.append("// FC")
    h_parts.append(
        f"extern const int8_t {var_prefix}_fc_weight{fc_w_dims};")

    if fc_w_scale_q15.size == 1:
        h_parts.append(
            f"extern const int16_t {var_prefix}_fc_weight_scale;")
    else:
        fc_ws_dims = shape_to_c_dims(fc_w_scale_q15)
        h_parts.append(
            f"extern const int16_t {var_prefix}_fc_weight_scale{fc_ws_dims};")

    h_parts.append(
        f"extern const int8_t {var_prefix}_fc_bias{fc_b_dims};")
    h_parts.append(
        f"extern const int16_t {var_prefix}_fc_bias_scale;")
    h_parts.append("")

    # LIF (Q15)
    h_parts.append("// LIF parameters (Q15 fixed-point)")
    h_parts.append(f"extern const int16_t {var_prefix}_lif1_beta;")
    h_parts.append(f"extern const int16_t {var_prefix}_lif2_beta;")
    h_parts.append(f"extern const int16_t {var_prefix}_lif_out_beta;")
    h_parts.append(f"extern const int16_t {var_prefix}_lif1_threshold;")
    h_parts.append(f"extern const int16_t {var_prefix}_lif2_threshold;")
    h_parts.append(f"extern const int16_t {var_prefix}_lif_out_threshold;")
    h_parts.append("")

    h_parts.append(f"#endif /* {header_guard} */")
    h_parts.append("")

    h_src = "\n".join(h_parts)

    # ----------------- 파일 저장 -----------------
    with open(c_filename, "w") as f:
        f.write(c_src)

    with open(h_filename, "w") as f:
        f.write(h_src)

    print(f"Generated (int-only): {c_filename}, {h_filename}")

# %%
if __name__ == "__main__":
    export_quant_snn_to_c_and_h(
        net,
        c_filename="snn_weights.c",
        h_filename="snn_weights.h",
        var_prefix="snn",
        header_guard="CHACHA_SNN_WEIGHTS_H"
    )

# %%
