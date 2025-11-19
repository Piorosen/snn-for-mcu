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
net.load_state_dict(torch.load("0097_acc49_snn_cifar10.pth", map_location="cpu"))
net.eval()
# net[0].weight_quant.scale()
# net[0].int_weight()

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

#%%
def export_quant_snn_to_c_and_h(net: nn.Sequential,
                                c_filename: str = "quant_snn_weights.c",
                                h_filename: str = "quant_snn_weights.h",
                                var_prefix: str = "snn",
                                header_guard: str = "QUANT_SNN_WEIGHTS_H"):
    """
    Conv/Linear:
      - weight : layer.int_weight(), layer.weight_quant.scale()
      - bias   : float bias를 직접 int8 + scale 로 양자화 (quantize_int8 사용)
    LIF:
      - beta, threshold : float scalar 로 export
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
    c_parts.append("/* Auto-generated quantized SNN weights (definition) */")
    c_parts.append(f'#include "{os.path.basename(h_filename)}"')
    c_parts.append("")

    # ---- Conv1 ----
    conv1_w_int = tensor_to_numpy(conv1.int_weight())              # int8
    conv1_w_scale = tensor_to_numpy(conv1.weight_quant.scale())    # float or tensor
    conv1_b_float = tensor_to_numpy(conv1.bias)                    # float
    conv1_b_int, conv1_b_scale = quantize_int8(conv1_b_float)      # int8 + scale

    c_parts.append("// Conv1")
    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv1_weight", conv1_w_int, "int8_t"))

    if conv1_w_scale.size == 1:
        c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_conv1_weight_scale",
                                               float(conv1_w_scale)))
    else:
        c_parts.append(numpy_to_c_float_array(f"{var_prefix}_conv1_weight_scale",
                                              conv1_w_scale, "float"))

    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv1_bias", conv1_b_int, "int8_t"))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_conv1_bias_scale", conv1_b_scale))

    # ---- Conv2 ----
    conv2_w_int = tensor_to_numpy(conv2.int_weight())
    conv2_w_scale = tensor_to_numpy(conv2.weight_quant.scale())
    conv2_b_float = tensor_to_numpy(conv2.bias)
    conv2_b_int, conv2_b_scale = quantize_int8(conv2_b_float)

    c_parts.append("// Conv2")
    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv2_weight", conv2_w_int, "int8_t"))

    if conv2_w_scale.size == 1:
        c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_conv2_weight_scale",
                                               float(conv2_w_scale)))
    else:
        c_parts.append(numpy_to_c_float_array(f"{var_prefix}_conv2_weight_scale",
                                              conv2_w_scale, "float"))

    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_conv2_bias", conv2_b_int, "int8_t"))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_conv2_bias_scale", conv2_b_scale))

    # ---- FC (QuantLinear) ----
    fc_w_int = tensor_to_numpy(fc.int_weight())
    fc_w_scale = tensor_to_numpy(fc.weight_quant.scale())
    fc_b_float = tensor_to_numpy(fc.bias)
    fc_b_int, fc_b_scale = quantize_int8(fc_b_float)

    c_parts.append("// FC")
    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_fc_weight", fc_w_int, "int8_t"))

    if fc_w_scale.size == 1:
        c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_fc_weight_scale",
                                               float(fc_w_scale)))
    else:
        c_parts.append(numpy_to_c_float_array(f"{var_prefix}_fc_weight_scale",
                                              fc_w_scale, "float"))

    c_parts.append(numpy_to_c_int_array(f"{var_prefix}_fc_bias", fc_b_int, "int8_t"))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_fc_bias_scale", fc_b_scale))

    # ---- LIF parameters ----
    c_parts.append("// LIF parameters (float)")
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_lif1_beta", float(lif1.beta)))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_lif2_beta", float(lif2.beta)))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_lif_out_beta", float(lif_out.beta)))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_lif1_threshold", float(lif1.threshold)))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_lif2_threshold", float(lif2.threshold)))
    c_parts.append(numpy_to_c_float_scalar(f"{var_prefix}_lif_out_threshold", float(lif_out.threshold)))

    c_src = "\n".join(c_parts)

    # ----------------- 헤더(.h) -----------------
    h_parts = []
    h_parts.append("/* Auto-generated quantized SNN weights (declaration) */")
    h_parts.append(f"#ifndef {header_guard}")
    h_parts.append(f"#define {header_guard}")
    h_parts.append("")
    h_parts.append("#include <stdint.h>")
    h_parts.append("")

    # Conv1
    conv1_w_dims = shape_to_c_dims(conv1_w_int)
    conv1_b_dims = shape_to_c_dims(conv1_b_int)

    h_parts.append("// Conv1")
    h_parts.append(
        f"extern const int8_t {var_prefix}_conv1_weight{conv1_w_dims};")

    if conv1_w_scale.size == 1:
        h_parts.append(
            f"extern const float {var_prefix}_conv1_weight_scale;")
    else:
        conv1_ws_dims = shape_to_c_dims(conv1_w_scale)
        h_parts.append(
            f"extern const float {var_prefix}_conv1_weight_scale{conv1_ws_dims};")

    h_parts.append(
        f"extern const int8_t {var_prefix}_conv1_bias{conv1_b_dims};")
    h_parts.append(
        f"extern const float {var_prefix}_conv1_bias_scale;")
    h_parts.append("")

    # Conv2
    conv2_w_dims = shape_to_c_dims(conv2_w_int)
    conv2_b_dims = shape_to_c_dims(conv2_b_int)

    h_parts.append("// Conv2")
    h_parts.append(
        f"extern const int8_t {var_prefix}_conv2_weight{conv2_w_dims};")

    if conv2_w_scale.size == 1:
        h_parts.append(
            f"extern const float {var_prefix}_conv2_weight_scale;")
    else:
        conv2_ws_dims = shape_to_c_dims(conv2_w_scale)
        h_parts.append(
            f"extern const float {var_prefix}_conv2_weight_scale{conv2_ws_dims};")

    h_parts.append(
        f"extern const int8_t {var_prefix}_conv2_bias{conv2_b_dims};")
    h_parts.append(
        f"extern const float {var_prefix}_conv2_bias_scale;")
    h_parts.append("")

    # FC
    fc_w_dims = shape_to_c_dims(fc_w_int)
    fc_b_dims = shape_to_c_dims(fc_b_int)

    h_parts.append("// FC")
    h_parts.append(
        f"extern const int8_t {var_prefix}_fc_weight{fc_w_dims};")

    if fc_w_scale.size == 1:
        h_parts.append(
            f"extern const float {var_prefix}_fc_weight_scale;")
    else:
        fc_ws_dims = shape_to_c_dims(fc_w_scale)
        h_parts.append(
            f"extern const float {var_prefix}_fc_weight_scale{fc_ws_dims};")

    h_parts.append(
        f"extern const int8_t {var_prefix}_fc_bias{fc_b_dims};")
    h_parts.append(
        f"extern const float {var_prefix}_fc_bias_scale;")
    h_parts.append("")

    # LIF
    h_parts.append("// LIF parameters (float)")
    h_parts.append(f"extern const float {var_prefix}_lif1_beta;")
    h_parts.append(f"extern const float {var_prefix}_lif2_beta;")
    h_parts.append(f"extern const float {var_prefix}_lif_out_beta;")
    h_parts.append(f"extern const float {var_prefix}_lif1_threshold;")
    h_parts.append(f"extern const float {var_prefix}_lif2_threshold;")
    h_parts.append(f"extern const float {var_prefix}_lif_out_threshold;")
    h_parts.append("")

    h_parts.append(f"#endif /* {header_guard} */")
    h_parts.append("")

    h_src = "\n".join(h_parts)

    # ----------------- 파일 저장 -----------------
    with open(c_filename, "w") as f:
        f.write(c_src)

    with open(h_filename, "w") as f:
        f.write(h_src)

    print(f"Generated: {c_filename}, {h_filename}")

# %%
if __name__ == "__main__":
    # utils.reset(net)
    # net.load_state_dict(torch.load("acc33_0001_snn_cifar10.pth", map_location="cpu"))
    # net.eval()

    export_quant_snn_to_c_and_h(
        net,
        c_filename="snn_weights.c",
        h_filename="snn_weights.h",
        var_prefix="snn",
        header_guard="CHACHA_SNN_WEIGHTS_H"
    )

# %%
