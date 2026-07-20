#!/usr/bin/env python3
"""CIFAR-10 테스트셋(10,000장)을 정수 평가 드라이버용 바이너리로 내보낸다.
포맷: 이미지당 3073바이트 = label(1B) + HWC RGB uint8(32*32*3).
사용: .venv/bin/python experiments/export_testset.py  (저장소 루트에서)
출력: experiments/testset_hwc.bin
"""
import numpy as np
import torchvision

ds = torchvision.datasets.CIFAR10("./data", train=False, download=True)
data = ds.data            # [10000, 32, 32, 3] uint8, HWC, RGB
labels = np.array(ds.targets, dtype=np.uint8)

with open("experiments/testset_hwc.bin", "wb") as f:
    for i in range(len(ds)):
        f.write(bytes([labels[i]]))
        f.write(data[i].tobytes())
print(f"wrote experiments/testset_hwc.bin: {len(ds)} images")
