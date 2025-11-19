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

transform = transforms.Compose([
    transforms.ToTensor(),  # 0~1
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=True, drop_last=True)

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


#%%
from PIL import Image

img_path = "6_cifar10_tensor_sample.png"   # 혹은 "cifar10_tensor_sample.png"
img = Image.open(img_path).convert("RGB")  # RGB로 맞춰주기
transform = transforms.Compose([transforms.ToTensor()])
tensor = transform(img)
tensor = tensor.unsqueeze(0) 
data_spk = spikegen.rate(tensor, num_steps=num_steps)

# data_spk.shape

spk_rec = []
for step in range(num_steps):
    spk_out, mem_out = net(data_spk[step])
    spk_rec.append(spk_out)
    
spk_rec = torch.stack(spk_rec)
# # %%
# spk_sum = spk_rec.sum(dim=0)    # [B, 10]
# _, predicted = spk_sum.max(1)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 이미 있던 코드 ---
spk_rec = []
for step in range(num_steps):
    spk_out, mem_out = net(data_spk[step])  # spk_out: [B, 10]
    spk_rec.append(spk_out)

spk_rec = torch.stack(spk_rec)   # [T, B, 10]

# --- 시간 방향 누적합 ---
cum_spk_sum = spk_rec.cumsum(dim=0)   # [T, B, 10]


# ----- 여기부터: 한 샘플에 대해 10개 클래스 곡선 그리기 -----
sample_idx = 0   # 배치에서 보고 싶은 샘플 인덱스 (원하면 바꿔도 됨)

# [T, 10] : 시간(t)마다 10개 클래스 값
values = cum_spk_sum[:, sample_idx, :]           # softmax 전 (누적 스파이크 값)
probs  = F.softmax(values, dim=1)                # softmax 적용 (각 t마다)

# CPU + numpy로 변환
values_np = values.detach().cpu().numpy()   # [T, 10]
probs_np  = probs.detach().cpu().numpy()    # [T, 10]

steps = range(1, num_steps + 1)

# ----- 그래프 그리기 -----
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1) 위: softmax 적용한 확률
for cls in range(10):
    axes[0].plot(steps, probs_np[:, cls], label=f"class {cls}")
axes[0].set_ylabel("Softmax probability")
axes[0].set_title("Softmax(cumulative spikes) over time")
axes[0].grid(True)
axes[0].legend(loc="upper right", ncol=2, fontsize=8)

# 2) 아래: softmax 안 한 누적 스파이크 값
for cls in range(10):
    axes[1].plot(steps, values_np[:, cls], label=f"class {cls}")
axes[1].set_xlabel("Time step (T)")
axes[1].set_ylabel("Cumulative spike value")
axes[1].set_title("Raw cumulative spikes over time")
axes[1].grid(True)
axes[1].legend(loc="upper left", ncol=2, fontsize=8)

plt.tight_layout()
plt.show()

