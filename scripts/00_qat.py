#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import brevitas.nn as qnn

import snntorch as snn
from snntorch import utils
from snntorch import surrogate
import snntorch.functional as SF

import nir
from snntorch.export_nir import export_to_nir
import snntorch as snn
from snntorch import surrogate, utils
from snntorch import functional as SF
from snntorch import spikegen   # ✅ 인코딩용 추가

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 100
num_steps = 30       # ✅ 타임스텝 30
beta = 0.9
lr = 2e-3

# ---------------------------------------------------------
# 1. 환경 설정
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2. CIFAR-10 데이터셋 & DataLoader
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # 0~1
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, drop_last=False)

spike_grad = surrogate.fast_sigmoid()

net = nn.Sequential(
    # 입력: [B, 3, 32, 32]
    # nn.Conv2d(3, 32, 5, padding=2),
    qnn.QuantConv2d(  # Conv2d 양자화 버전
        3, 32, 5, padding=2,
        weight_bit_width=8,  # 8-bit weight
        bias=True
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
    qnn.QuantLinear(      # Linear 양자화 버전
        64 * 8 * 8, 10,
        weight_bit_width=8,
        bias=True
    ),

    # 마지막 레이어: spikes와 membrane state를 모두 반환
    snn.Leaky(beta=beta, spike_grad=spike_grad,
              init_hidden=True, output=True)
).to(device)


# ---------------------------------------------------------
# 4. Forward pass (num_steps 동안 SNN 시뮬레이션)
#    - spikegen.rate 로 SNN 인코딩 후, 각 타임스텝마다 스파이크 입력
# ---------------------------------------------------------
def forward_pass(net, data, num_steps):
    """
    data: [B, 3, 32, 32]  (0~1 범위 권장)
    return:
      spk_rec: [T, B, 10]
      mem_rec: [T, B, 10]
    """
    spk_rec = []
    mem_rec = []

    # 모든 LIF neuron의 hidden state reset
    utils.reset(net)

    # 1) SNN 인코딩 (rate / Poisson)
    #    결과 shape: [T, B, 3, 32, 32]
    data_spk = spikegen.rate(data, num_steps=num_steps)

    # 2) 각 타임스텝마다 스파이크 프레임 하나씩 네트워크에 입력
    for step in range(num_steps):
        # data_spk[step]: [B, 3, 32, 32]
        spk_out, mem_out = net(data_spk[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

loss_fn = SF.ce_rate_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))


def evaluate(net, loader, num_steps):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec, mem_rec = forward_pass(net, data, num_steps)
            spk_sum = spk_rec.sum(dim=0)    # [B, 10]
            _, predicted = spk_sum.max(1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

#%%

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        spk_rec, mem_rec = forward_pass(net, data, num_steps)  # ✅ num_steps=30
        loss = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Iter {i+1}/{len(train_loader)} "
                  f"Loss: {running_loss / 50:.4f}")
            running_loss = 0.0
    test_acc = evaluate(net, test_loader, num_steps)   # ✅ num_steps=30
    torch.save(net.state_dict(), f"{epoch+1:04}_acc{int(test_acc)}_snn_cifar10.pth")





# %%
