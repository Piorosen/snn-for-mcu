import torch
import numpy as np
kl_list = []


for i in range(1000):
    img, label = train_dataset[i]      # img: [3, 32, 32] tensor, [0,1]
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = img.unsqueeze(0)          # [1, 3, 32, 32]
    tensor_np = tensor.detach().cpu().numpy()

    # 1. Torch 쪽 스파이크 생성
    torch.manual_seed(123)
    data_spk_torch = spikegen.rate(tensor, num_steps=num_steps)  # [T, 1, 3, 32, 32]
    data_spk_torch_np = data_spk_torch.detach().cpu().numpy()

    # 2. NumPy 쪽 스파이크 생성
    rng = np.random.default_rng(seed=123)
    data_spk_np = rate_numpy(tensor_np, num_steps=num_steps, rng=rng)  # [T, 1, 3, 32, 32]

    # --- 통계 비교 ---
    print(f"\n=== Sample {i} (label={label}) ===")
    print("Torch spike mean:", data_spk_torch_np.mean())
    print("NumPy spike mean:", data_spk_np.mean())
    print("Torch spike var :", data_spk_torch_np.var())
    print("NumPy spike var :", data_spk_np.var())

    # --- KL 다이버전스 계산 ---
    # 시간축(T)에 대해 평균 내서 각 뉴런의 발화 확률 p, q 추정
    # data_spk_* : [T, 1, 3, 32, 32]
    p = data_spk_torch_np.mean(axis=0)  # [1, 3, 32, 32]
    q = data_spk_np.mean(axis=0)        # [1, 3, 32, 32]

    # 수치 안정성용 epsilon
    eps = 1e-7
    p_clipped = np.clip(p, eps, 1.0 - eps)
    q_clipped = np.clip(q, eps, 1.0 - eps)

    # Bernoulli KL: D_KL(Bern(p) || Bern(q))
    kl_per_neuron = (
        p_clipped * np.log(p_clipped / q_clipped)
        + (1.0 - p_clipped) * np.log((1.0 - p_clipped) / (1.0 - q_clipped))
    )  # [1, 3, 32, 32]

    # 전체 뉴런에 대해 평균 KL (스칼라)
    kl_mean = kl_per_neuron.mean()
    kl_list.append(kl_mean)

    print("KL divergence (Bernoulli, Torch || NumPy):", float(kl_mean))

# 10개 샘플에 대한 평균 KL
kl_array = np.array(kl_list, dtype=np.float32)
print("\n=== Overall KL stats over 10 samples ===")
print("mean KL:", kl_array.mean())
print("std KL :", kl_array.std())
