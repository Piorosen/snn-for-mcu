#%%
import os
from torchvision import datasets, transforms

# -----------------------------------
# 1) CIFAR10 로드 (ToTensor: [0,1] float32, shape [3,32,32])
# -----------------------------------
transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# -----------------------------------
# 2) .bin 저장 함수
# -----------------------------------
def save_cifar_bin(dataset, out_dir, max_count=None):
    os.makedirs(out_dir, exist_ok=True)

    # 이미지 id는 1부터 시작
    img_id = 1

    for idx, (img, label) in enumerate(dataset):
        if max_count is not None and img_id > max_count:
            break

        # img: torch.Tensor [3,32,32], float32, [0,1]
        img_np = img.numpy().astype("float32")  # (C,H,W)

        # 파일 이름: label과 id를 4자리 zero-padding
        filename = f"{label:04d}_{img_id:04d}.bin"
        path = os.path.join(out_dir, filename)

        # 바이너리로 쓰기 (순수 float32 배열만)
        # 방법1: tofile
        img_np.tofile(path)

        # 방법2 (참고): struct 사용
        # with open(path, "wb") as f:
        #     f.write(img_np.tobytes())

        img_id += 1

    print(f"Saved {img_id - 1} images to {out_dir}")


if __name__ == "__main__":
    # 예: train set에서 앞 100장만 저장
    save_cifar_bin(dataset, out_dir="./cifar_bin", max_count=100)
    

# import torch
# from torchvision import datasets, transforms
# from torchvision.utils import save_image

# # 1) ToTensor 사용해서 로드
# transform = transforms.ToTensor()

# dataset = datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform,  # img: torch.Tensor (C, H, W)
# )

# # 2) 이미지 하나 가져오기
# img, label = dataset[0]   # img: [3, 32, 32] tensor, 값 범위 [0, 1]

# # 3) 텐서를 이미지 파일로 저장
# save_image(img, "cifar10_tensor_sample.png")  # 자동으로 PNG로 저장
# print("label:", label)

# %%
