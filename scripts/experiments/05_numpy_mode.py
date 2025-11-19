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

transform = transforms.Compose([
    transforms.ToTensor(),  # 0~1
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)


beta = 0.9
num_steps = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
utils.reset(net)
net.load_state_dict(torch.load("snn_cifar10.pth"))
net.eval()

net.compile()
#%%

def summarize_model(model: nn.Module, input_size, device="cpu"):
    """
    input_size: (C, H, W) 또는 (features,) 처럼 '배치 제외' 크기
    예) 이미지: (3, 224, 224)
        MLP:   (100,)
    """
    model = model.to(device)
    model.eval()

    summary = []
    hooks = []

    def hook(module, input, output):
        # 모듈 이름
        class_name = module.__class__.__name__
        idx = len(summary)
        layer_name = f"{class_name}-{idx}"

        def get_shape(x):
            if isinstance(x, torch.Tensor):
                return list(x.shape)
            elif isinstance(x, (list, tuple)):
                return [list(t.shape) for t in x if isinstance(t, torch.Tensor)]
            else:
                return str(type(x))

        # input, output은 튜플일 수 있음
        in_obj = input[0] if isinstance(input, (list, tuple)) and len(input) > 0 else input
        out_obj = output

        summary.append({
            "layer": layer_name,
            "input_shape": get_shape(in_obj),
            "output_shape": get_shape(out_obj),
        })

    # leaf 모듈(실제 연산하는 레이어)들만 hook
    for name, module in model.named_modules():
        # 자식 모듈 있으면 container로 보고 skip
        if len(list(module.children())) > 0:
            continue
        hooks.append(module.register_forward_hook(hook))

    # 더미 입력 생성 (batch size = 1)
    x = torch.randn(input_size).to(device)

    with torch.no_grad():
        _ = model(x)

    # hook 제거
    for h in hooks:
        h.remove()

    return summary


def print_summary(summary):
    print("{:<25} {:<35} {:<35}".format("Layer", "Input shape", "Output shape"))
    print("-" * 100)
    for item in summary:
        print(
            "{:<25} {:<35} {:<35}".format(
                item["layer"],
                str(item["input_shape"]),
                str(item["output_shape"]),
            )
        )
        
    
model_summary = summarize_model(net, input_size=(1, 3, 32, 32), device=device)
print_summary(model_summary)

#%%
#%%
from chacha_py import Conv2D, AvgPool2D, LeakyNP, Flatten, Linear, rate_numpy
class NumpySNN:
    """
    PyTorch net_torch와 같은 구조:
    Conv2D -> AvgPool2D -> LeakyNP ->
    Conv2D -> AvgPool2D -> LeakyNP ->
    Flatten -> Linear -> LeakyNP (output=True)
    """
    def __init__(self, beta=0.9):
        self.conv1 = Conv2D(3, 32, kernel_size=5, padding=2)
        self.pool1 = AvgPool2D(2)
        self.lif1  = LeakyNP(beta=beta, init_hidden=True, reset_mechanism="subtract")

        self.conv2 = Conv2D(32, 64, kernel_size=5, padding=2)
        self.pool2 = AvgPool2D(2)
        self.lif2  = LeakyNP(beta=beta, init_hidden=True, reset_mechanism="subtract")

        self.flatten = Flatten()
        self.fc      = Linear(64 * 8 * 8, 10)
        self.lif_out = LeakyNP(beta=beta, init_hidden=True, reset_mechanism="subtract")

    def reset_state(self):
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif_out.reset_state()

    def forward_step(self, x_np):
        """
        한 타임스텝용 forward.
        x_np: [B, 3, 32, 32] NumPy 배열 (spike 입력)
        return: spk_out, mem_out (마지막 레이어 기준)
        """
        # 첫 블록
        z = self.conv1.forward(x_np)
        z = self.pool1.forward(z)
        spk1, _ = self.lif1.forward(z)

        # 두 번째 블록
        z = self.conv2.forward(spk1)
        z = self.pool2.forward(z)
        spk2, _ = self.lif2.forward(z)

        # FC + 마지막 Leaky
        z = self.flatten.forward(spk2)
        z = self.fc.forward(z)
        spk_out, mem_out = self.lif_out.forward(z)

        return spk_out, mem_out

#%%
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
#%%
np_snn = build_numpy_snn_from_torch(net, beta=beta)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
#%%

img, label = train_dataset[0]
img = img.unsqueeze(0)
data_spk = rate_numpy(img, num_steps=num_steps)
#%%
data_spk.shape
#%%
spk_rec = []
for step in range(num_steps):
    spk_out, mem_out = np_snn.forward_step(data_spk[step])
    spk_rec.append(spk_out)
    


# %%
sum(spk_rec)
# %%
