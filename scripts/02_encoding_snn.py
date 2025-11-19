import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from snntorch import spikegen
from torchvision import datasets, transforms

num_steps = 30

transform = transforms.Compose([
    transforms.ToTensor(),  # 0~1
])

img_path = "6_cifar10_tensor_sample.png"   # 혹은 "cifar10_tensor_sample.png"
img = Image.open(img_path).convert("RGB")  # RGB로 맞춰주기
transform = transforms.Compose([transforms.ToTensor()])
tensor = transform(img)
tensor = tensor.unsqueeze(0)  # 배치 차원 추가

data_spk = spikegen.rate(tensor, num_steps=num_steps)


#%%
# net(data_spk)

# %%
data_spk.shape
# %%


data_spk.shape

frames = data_spk  # 네가 가진 이름으로 바꿔줘
frames = frames.detach().cpu()  # GPU -> CPU

fig, ax = plt.subplots()
ax.axis("off")

# 첫 프레임을 [H, W, C]로 바꿔서 초기 이미지 설정
first_frame = frames[0].permute(1, 2, 0).numpy()
im = ax.imshow(first_frame)

def update(frame_idx):
    frame = frames[frame_idx].permute(1, 2, 0).numpy()
    im.set_data(frame)
    return [im]

# interval: 프레임 간 시간 (ms), fps: 초당 프레임
ani = FuncAnimation(
    fig,
    update,
    frames=30,
    interval=100,   # 100ms -> 0.1초마다 다음 프레임
    blit=True
)

# GIF로 저장 (PillowWriter 필요: pip install pillow)
writer = PillowWriter(fps=30)  # 초당 10프레임
ani.save("output.gif", writer=writer)

plt.close(fig)
print("GIF 저장 완료: output.gif")