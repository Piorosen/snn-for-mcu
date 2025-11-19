#%%
import torch
import torch.nn as nn
import snntorch as snn  # pip install snntorch

# ----------------- 모델 정의 -----------------
beta = 0.9
spike_grad = None  # 실제로 쓰는 grad 함수 있으면 여기 넣으면 됩니다.

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

# ----------------- hook 유틸 -----------------
def register_shape_hooks(model):
    hooks = []

    def shape_of(x):
        import torch

        if isinstance(x, torch.Tensor):
            return list(x.shape)
        if isinstance(x, (tuple, list)):
            return [shape_of(xx) for xx in x]
        return str(type(x))

    def hook(module, inputs, output):
        name = module.__class__.__name__

        # input은 tuple로 들어올 수 있음
        in_obj = inputs if not isinstance(inputs, (list, tuple)) else inputs
        if isinstance(in_obj, (list, tuple)) and len(in_obj) == 1:
            in_obj = in_obj[0]

        in_shape = shape_of(in_obj)
        out_shape = shape_of(output)

        w = getattr(module, "weight", None)
        w_shape = shape_of(w) if w is not None else None

        print(f"{name}")
        print(f"  input : {in_shape}")
        print(f"  output: {out_shape}")
        print(f"  weight: {w_shape}")
        print("-" * 50)

    # leaf 모듈(실제 연산 레이어)에만 hook
    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook))

    return hooks

# ----------------- 사용 예 -----------------
if __name__ == "__main__":
    B = 1
    x = torch.randn(B, 3, 32, 32)  # 입력 예시

    hooks = register_shape_hooks(net)
    out = net(x)   # forward 하면서 각 레이어 shape 출력

    # 필요 없으면 hook 제거
    for h in hooks:
        h.remove()
