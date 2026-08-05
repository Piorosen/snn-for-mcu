#!/usr/bin/env python3
"""
SNN 벤치마크용 고정 입력 이미지를 C 배열로 내보낸다.

벤치마크가 SD카드·JPEG 디코딩·LCD에 의존하면 측정에 그 비용이 섞이고,
매 실행마다 조건이 흔들린다. 입력을 플래시에 박아 두면 추론 경로만 남는다.

입력: experiments/testset_hwc.bin  (이미지당 3073B = label 1B + HWC RGB uint8 3072B)
출력: ImageClassification/Core/Src/snn_bench_images.c

사용: .venv/bin/python info/code/gen_bench_images.py [--count 8]
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "experiments" / "testset_hwc.bin"
DST = ROOT / "ImageClassification" / "Core" / "Src" / "snn_bench_images.c"

REC = 1 + 32 * 32 * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=8,
                    help="내보낼 이미지 수 (플래시 사용량 = count x 3072B)")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"테스트셋이 없다: {SRC}\n"
                         f"  .venv/bin/python experiments/export_testset.py 로 생성할 것")

    raw = SRC.read_bytes()
    total = len(raw) // REC
    n = min(args.count, total)

    lines = []
    lines.append("/*")
    lines.append(" * SNN 벤치마크용 고정 입력 이미지 (자동 생성 — 직접 편집하지 말 것).")
    lines.append(" *")
    lines.append(" * 생성: info/code/gen_bench_images.py")
    lines.append(f" * 출처: experiments/testset_hwc.bin 의 앞 {n}장 (CIFAR-10 테스트셋)")
    lines.append(" * 레이아웃: HWC RGB uint8, 32x32x3 = 3072 bytes/장")
    lines.append(" *")
    lines.append(" * 입력을 플래시에 박아 두는 이유: SD카드 읽기·JPEG 디코딩·LCD 출력이")
    lines.append(" * 측정에 섞이지 않게 하고, 실행마다 완전히 같은 입력을 쓰기 위해서다.")
    lines.append(" */")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define SNN_BENCH_IMAGE_COUNT {n}")
    lines.append("")
    lines.append(f"const int32_t g_bench_labels[{n}] = {{")

    labels = []
    for i in range(n):
        labels.append(raw[i * REC])
    lines.append("    " + ", ".join(str(v) for v in labels))
    lines.append("};")
    lines.append("")
    lines.append(f"const uint8_t g_bench_images[{n}][32 * 32 * 3] = {{")

    for i in range(n):
        px = raw[i * REC + 1: (i + 1) * REC]
        lines.append(f"  {{ /* [{i}] label={labels[i]} */")
        for off in range(0, len(px), 24):
            chunk = px[off:off + 24]
            lines.append("    " + ",".join(f"{b:3d}" for b in chunk) + ",")
        lines.append("  },")
    lines.append("};")
    lines.append("")

    DST.write_text("\n".join(lines))
    size = DST.stat().st_size
    print(f"생성: {DST}")
    print(f"  이미지 {n}장, 플래시 사용 {n * 3072:,} bytes, 소스 {size:,} bytes")
    print(f"  정답 레이블: {labels}")


if __name__ == "__main__":
    main()
