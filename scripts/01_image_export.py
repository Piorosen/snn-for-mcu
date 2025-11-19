
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 1) ToTensor 사용해서 로드
transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,  # img: torch.Tensor (C, H, W)
)

# 2) 이미지 하나 가져오기
img, label = dataset[0]   # img: [3, 32, 32] tensor, 값 범위 [0, 1]

# 3) 텐서를 이미지 파일로 저장
save_image(img, "cifar10_tensor_sample.png")  # 자동으로 PNG로 저장
print("label:", label)
