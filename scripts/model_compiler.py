#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import nir
from snntorch.export_nir import export_to_nir
import snntorch as snn
from snntorch import surrogate, utils
from snntorch import spikegen   # ✅ 인코딩용 추가
from chacha_py import rate_numpy
import numpy as np
from chacha_py import NumpySNN


def build_numpy_snn_from_torch(net_torch, beta=0.5):
    np_net = NumpySNN(beta=beta)

    # PyTorch 레이어 꺼내기 (Sequential 인덱스 기반)
    conv1_t = net_torch[0]
    pool1_t = net_torch[1]
    lif1_t  = net_torch[2]
    conv2_t = net_torch[3]
    pool2_t = net_torch[4]
    lif2_t  = net_torch[5]
    flatten_t = net_torch[6]
    fc_t      = net_torch[7]
    lif_out_t = net_torch[8]

    # Conv1
    np_net.conv1.weight = conv1_t.weight.detach().cpu().numpy().astype(np.float32)
    np_net.conv1.bias   = conv1_t.bias.detach().cpu().numpy().astype(np.float32)

    # Conv2
    np_net.conv2.weight = conv2_t.weight.detach().cpu().numpy().astype(np.float32)
    np_net.conv2.bias   = conv2_t.bias.detach().cpu().numpy().astype(np.float32)

    # FC
    np_net.fc.weight = fc_t.weight.detach().cpu().numpy().astype(np.float32)
    np_net.fc.bias   = fc_t.bias.detach().cpu().numpy().astype(np.float32)

    # beta, threshold (필요하면 복사 – 여기서는 beta만 예시)
    np_net.lif1.beta     = float(lif1_t.beta)
    np_net.lif2.beta     = float(lif2_t.beta)
    np_net.lif_out.beta  = float(lif_out_t.beta)
    np_net.lif1.threshold    = float(lif1_t.threshold)
    np_net.lif2.threshold    = float(lif2_t.threshold)
    np_net.lif_out.threshold = float(lif_out_t.threshold)

    return np_net

def numpy_shape_to_c_dims(arr):
    """
    numpy 배열의 shape를 C 스타일 차원 문자열로 변환
    예: shape = (16, 1, 3, 3) -> "[16][1][3][3]"
    """
    arr = np.asarray(arr)
    return "".join(f"[{d}]" for d in arr.shape)

def numpy_array_to_c_int8(name, arr):
    """
    name : C 변수 이름 (예: "snn_conv1_weight_q")
    arr  : numpy 배열 (float32 등)
    반환 : int8_t 배열 + scale 을 담은 C 코드 문자열
    """
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    shape = arr.shape

    # scale 계산 (대칭 양자화)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0.0:
        scale = 1.0  # 전부 0이면 아무 scale 이나 상관 없음
    else:
        # float ≈ int8 * scale  (int8 범위: [-128, 127])
        scale = max_abs / 127.0

    # 양자화: q = round(x / scale), [-128, 127] 클리핑
    q = np.round(arr / scale).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)

    dim_str = "".join(f"[{d}]" for d in shape)

    lines = []
    # int8_t 배열
    lines.append(f"const int8_t {name}{dim_str} = {{")

    flat = q.flatten()
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
    # dequantization 용 scale 상수 (float)
    lines.append(f"const float {name}_scale = {scale:.8f}f;")
    lines.append("")

    return "\n".join(lines)


def numpy_array_to_c(name, arr, dtype="float"):
    """
    name: C 변수 이름
    arr : numpy 배열
    """
    import numpy as np

    arr = np.asarray(arr)
    shape = arr.shape

    # shape -> [O][I][H][W] 이런 식으로 변환
    dim_str = "".join(f"[{d}]" for d in shape)

    # 헤더
    lines = []
    lines.append(f"const {dtype} {name}{dim_str} = {{")

    flat = arr.flatten()
    # 보기 좋게 줄 바꿈
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
    lines.append("")  # 마지막 빈 줄

    return "\n".join(lines)


def export_snn_to_c(np_net, var_prefix="snn"):
    """
    build_numpy_snn_from_torch 로 만든 np_net을
    C 코드 문자열로 변환
    """

    parts = []

    # Conv1
    parts.append(numpy_array_to_c_int8(
        f"{var_prefix}_conv1_weight",
        np_net.conv1.weight
    ))
    parts.append(numpy_array_to_c_int8(
        f"{var_prefix}_conv1_bias",
        np_net.conv1.bias
    ))

    # Conv2
    parts.append(numpy_array_to_c_int8(
        f"{var_prefix}_conv2_weight",
        np_net.conv2.weight
    ))
    parts.append(numpy_array_to_c_int8(
        f"{var_prefix}_conv2_bias",
        np_net.conv2.bias
    ))

    # FC
    parts.append(numpy_array_to_c_int8(
        f"{var_prefix}_fc_weight",
        np_net.fc.weight
    ))
    parts.append(numpy_array_to_c_int8(
        f"{var_prefix}_fc_bias",
        np_net.fc.bias
    ))

    # LIF 파라미터 (beta, threshold)
    lif_lines = []
    lif_lines.append("// LIF parameters")
    lif_lines.append(
        f"const float {var_prefix}_lif1_beta      = {float(np_net.lif1.beta):.8f}f;")
    lif_lines.append(
        f"const float {var_prefix}_lif2_beta      = {float(np_net.lif2.beta):.8f}f;")
    lif_lines.append(
        f"const float {var_prefix}_lif_out_beta   = {float(np_net.lif_out.beta):.8f}f;")
    lif_lines.append(
        f"const float {var_prefix}_lif1_threshold = {float(np_net.lif1.threshold):.8f}f;")
    lif_lines.append(
        f"const float {var_prefix}_lif2_threshold = {float(np_net.lif2.threshold):.8f}f;")
    lif_lines.append(
        f"const float {var_prefix}_lif_out_threshold = {float(np_net.lif_out.threshold):.8f}f;")
    lif_lines.append("")

    parts.append("\n".join(lif_lines))

    # 전부 합쳐서 하나의 .c 텍스트로 반환
    c_code = "\n".join(parts)
    return c_code

def export_snn_to_h(np_net, var_prefix="snn", header_guard="SNN_WEIGHTS_H"):
    """
    NumpySNN(np_net)을 C 헤더(.h)용 extern 선언 코드로 변환.
    int8 대칭 양자화(per-tensor) 사용:
      float ≈ int8 * scale
    각 weight/bias 에 대해 int8 배열 + float scale 를 extern 으로 선언.
    """

    parts = []
    parts.append(f"#ifndef {header_guard}")
    parts.append(f"#define {header_guard}")
    parts.append("")
    parts.append("#include <stdint.h>")
    parts.append("")
    parts.append("/* Auto-generated SNN weight declarations (int8 quantized) */")
    parts.append("/* Quantization: per-tensor symmetric, float ~= int8 * scale */")
    parts.append("")

    # --- Conv1 ---
    conv1_w_dims = numpy_shape_to_c_dims(np_net.conv1.weight)
    conv1_b_dims = numpy_shape_to_c_dims(np_net.conv1.bias)
    parts.append(
        f"extern const int8_t {var_prefix}_conv1_weight{conv1_w_dims};")
    parts.append(
        f"extern const float {var_prefix}_conv1_weight_scale;")
    parts.append(
        f"extern const int8_t {var_prefix}_conv1_bias{conv1_b_dims};")
    parts.append(
        f"extern const float {var_prefix}_conv1_bias_scale;")
    parts.append("")

    # --- Conv2 ---
    conv2_w_dims = numpy_shape_to_c_dims(np_net.conv2.weight)
    conv2_b_dims = numpy_shape_to_c_dims(np_net.conv2.bias)
    parts.append(
        f"extern const int8_t {var_prefix}_conv2_weight{conv2_w_dims};")
    parts.append(
        f"extern const float {var_prefix}_conv2_weight_scale;")
    parts.append(
        f"extern const int8_t {var_prefix}_conv2_bias{conv2_b_dims};")
    parts.append(
        f"extern const float {var_prefix}_conv2_bias_scale;")
    parts.append("")

    # --- FC ---
    fc_w_dims = numpy_shape_to_c_dims(np_net.fc.weight)
    fc_b_dims = numpy_shape_to_c_dims(np_net.fc.bias)
    parts.append(
        f"extern const int8_t {var_prefix}_fc_weight{fc_w_dims};")
    parts.append(
        f"extern const float {var_prefix}_fc_weight_scale;")
    parts.append(
        f"extern const int8_t {var_prefix}_fc_bias{fc_b_dims};")
    parts.append(
        f"extern const float {var_prefix}_fc_bias_scale;")
    parts.append("")

    # --- LIF parameters (scalar, float 그대로) ---
    parts.append("// LIF parameters (not quantized)")
    parts.append(
        f"extern const float {var_prefix}_lif1_beta;")
    parts.append(
        f"extern const float {var_prefix}_lif2_beta;")
    parts.append(
        f"extern const float {var_prefix}_lif_out_beta;")
    parts.append(
        f"extern const float {var_prefix}_lif1_threshold;")
    parts.append(
        f"extern const float {var_prefix}_lif2_threshold;")
    parts.append(
        f"extern const float {var_prefix}_lif_out_threshold;")
    parts.append("")

    parts.append(f"#endif /* {header_guard} */")
    parts.append("")

    return "\n".join(parts)



beta = 0.9
spike_grad = surrogate.fast_sigmoid()
net = nn.Sequential(
    # 입력: [B, 3, 32, 32]
    nn.Conv2d(3, 32, 5, padding=2),
    nn.AvgPool2d(2),  # 32x32 -> 16x16
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Conv2d(32, 64, 5, padding=2),
    nn.AvgPool2d(2),  # 16x16 -> 8x8
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),           # [B, 64*8*8]
    nn.Linear(64 * 8 * 8, 10),
    # 마지막 레이어: spikes와 membrane state를 모두 반환
    snn.Leaky(beta=beta, spike_grad=spike_grad,
              init_hidden=True, output=True)
)
# net = nn.Sequential(
#     # 입력: [B, 3, 32, 32]
#     nn.Conv2d(3, 16, 5, padding=2),
#     nn.AvgPool2d(2),  # 32x32 -> 16x16
#     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

#     nn.Conv2d(16, 32, 5, padding=2),
#     nn.AvgPool2d(2),  # 16x16 -> 8x8
#     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

#     nn.Flatten(),           # [B, 32*8*8]
#     nn.Linear(32 * 8 * 8, 10),
#     # 마지막 레이어: spikes와 membrane state를 모두 반환
#     snn.Leaky(beta=beta, spike_grad=spike_grad,
#               init_hidden=True, output=True)
# )

utils.reset(net)
net.load_state_dict(torch.load("/home/chacha/Desktop/git/snn-for-mcu/resources/acc44_snn_cifar10.pth"))
net.eval()

# 1) Torch 모델 → NumpySNN
np_net = build_numpy_snn_from_torch(net, beta=0.9)

# 2) C 소스 코드(.c) 생성
c_src = export_snn_to_c(np_net, var_prefix="snn")

with open("snn_weights.c", "w") as f:
    f.write("/* auto-generated SNN weights (definition) */\n\n")
    f.write('#include "snn_weights.h"\n\n')
    f.write(c_src)

# 3) 헤더 코드(.h) 생성
h_src = export_snn_to_h(np_net, var_prefix="snn", header_guard="SNN_WEIGHTS_H")

with open("snn_weights.h", "w") as f:
    f.write("/* auto-generated SNN weights (declaration) */\n\n")
    f.write(h_src)


# %%
