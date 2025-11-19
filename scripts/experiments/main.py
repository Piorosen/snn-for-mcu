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
#%%
from chacha_py import NumpySNN, rate_numpy

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

img, label = train_dataset[1]
img = img.unsqueeze(0)
data_spk = rate_numpy(img, num_steps=num_steps)
spk_rec = []
for step in range(num_steps):
    spk_out, mem_out = np_snn.forward_step(data_spk[step])
    spk_rec.append(spk_out)
    
print(sum(spk_rec), label)
# %%





# %%
