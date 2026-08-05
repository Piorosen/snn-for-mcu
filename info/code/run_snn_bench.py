#!/usr/bin/env python3
"""
SNN 추론 벤치마크 자동화: 변형별 빌드 → 플래싱 → DWT 사이클 결과 회수 → 비교표.

main_bench.cc(SNN_BENCH=ON 빌드)가 g_snn_bench 전역 구조체에 남긴 결과를
ST-LINK HOTPLUG 메모리 판독으로 회수한다. 시리얼이 필요 없고, 같은 바이너리를
다시 돌리면 사이클 수까지 거의 그대로 재현된다 (입력·RNG 시드 고정).

변형(누적 적용):
    v0     최적화 없음 (기존 조건 재현: ART OFF, 코드 플래시, 데이터 기본 배치)
    art    + ART accelerator/프리페치 (FLASH_ACR.ARTEN/PRFTEN)
    itcm   + 핫 커널 코드를 ITCM RAM(16KB)으로
    dtcm   + conv 가중치·im2col 스크래치를 DTCM(64KB)으로
    full   + 희소(spike-driven) conv 커널

회귀 검증: 모든 변형의 spike_checksum과 preds[]는 비트 단위로 동일해야 한다.
다르면 그 최적화는 계산 결과를 바꾼 것이므로 표에 FAIL로 찍힌다.

사용법:
    python3 run_snn_bench.py                  # 전 변형 빌드+측정
    python3 run_snn_bench.py --variants v0 art
    python3 run_snn_bench.py --skip-build     # 기존 빌드 재사용
    python3 run_snn_bench.py --fetch-only     # 현재 보드의 결과 버퍼만 회수
"""

import argparse
import glob
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

PROG = (
    "/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/"
    "com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.macosaarch64_1.0.100.202603051304"
    "/tools/bin/STM32_Programmer_CLI"
)
TOOLCHAIN_BIN = (
    "/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/"
    "com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.macosaarch64_1.0.0.202602081740"
    "/tools/bin"
)
NM = f"{TOOLCHAIN_BIN}/arm-none-eabi-nm"

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent / "ImageClassification"
TOOLCHAIN_FILE = PROJ / "cmake" / "arm-none-eabi.cmake"
RESULTS_DIR = HERE.parent / "results"

MAGIC = 0x424E4E53  # 'SNNB'
IMAGES = 8

# main_bench.cc snn_bench_t 레이아웃 (uint32 워드 단위)
#  0 magic  1 done  2 opt_flags  3 sysclk_hz  4 images  5 timesteps
#  6 cyc_per_image  7 cyc_encode  8 cyc_invoke
#  9 cyc_cpl1  10 cyc_cpl2  11 cyc_lifout  12 cyc_memcpy
#  13..20 preds[8]  21..28 labels[8]  29 correct  30 spike_checksum
#  31 flash_acr  32 scb_ccr  33 arena_used  34 itcm_code_bytes
N_WORDS = 35

VARIANTS = {
    "v0":   [],
    "art":  ["ART"],
    "itcm": ["ART", "ITCM"],
    "dtcm": ["ART", "ITCM", "DTCM"],
    "full": ["ART", "ITCM", "DTCM", "KERNEL"],
}


def cli(args, timeout=300):
    p = subprocess.run([PROG] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, re.sub(r"\033\[[0-9;]*m", "", p.stdout + p.stderr)


def build(name, opts):
    bdir = PROJ / f"build-bench-{name}"
    flags = [
        f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN_FILE}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSNN_BENCH=ON",
    ]
    for o in ("ART", "ITCM", "DTCM", "KERNEL"):
        flags.append(f"-DSNN_OPT_{o}={'ON' if o in opts else 'OFF'}")
    if not (bdir / "build.ninja").exists():
        subprocess.run(["cmake", "-S", str(PROJ), "-B", str(bdir), "-G", "Ninja"] + flags,
                       check=True, capture_output=True, text=True)
    p = subprocess.run(["cmake", "--build", str(bdir)], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit(f"[{name}] 빌드 실패")
    return bdir


def bench_addr(elf):
    out = subprocess.run([NM, str(elf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^([0-9a-f]+)\s+[BbDd]\s+g_snn_bench$", out, re.M)
    if not m:
        raise SystemExit(f"g_snn_bench 심볼이 없다: {elf}")
    return int(m.group(1), 16)


def flash(binpath):
    rc, out = cli(["-c", "port=SWD", "mode=UR", "-w", str(binpath), "0x08000000", "-v", "-rst"])
    if "Verification OK" not in out and "successfully" not in out.lower():
        print(out[-2000:])
        raise SystemExit("플래싱 실패")


def read_words(addr, n):
    rc, out = cli(["-q", "-c", "port=SWD", "mode=HOTPLUG", "-r32", hex(addr), hex(n * 4)])
    words = []
    for line in out.splitlines():
        m = re.match(r"\s*0x([0-9A-Fa-f]{8})\s*:\s*(.*)", line)
        if m:
            words += [int(w, 16) for w in re.findall(r"\b[0-9A-Fa-f]{8}\b", m.group(2))]
    return words[:n]


def wait_and_fetch(addr, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        w = read_words(addr, 2)
        if len(w) == 2 and w[0] == MAGIC and w[1] == 1:
            return read_words(addr, N_WORDS)
        time.sleep(2.0)
    raise SystemExit(f"시간 초과({timeout_s}s): 벤치가 끝나지 않았다 (magic/done 미충족)")


def parse(w):
    if len(w) < N_WORDS or w[0] != MAGIC:
        raise SystemExit(f"결과 버퍼가 비정상이다: {w[:4]}")
    def s32(x):
        return x - (1 << 32) if x >= (1 << 31) else x
    return {
        "opt_flags": w[2], "sysclk_hz": w[3], "images": w[4], "timesteps": w[5],
        "cyc_per_image": w[6], "cyc_encode": w[7], "cyc_invoke": w[8],
        "cyc_cpl1": w[9], "cyc_cpl2": w[10], "cyc_lifout": w[11], "cyc_memcpy": w[12],
        "preds": [s32(x) for x in w[13:21]], "labels": [s32(x) for x in w[21:29]],
        "correct": w[29], "spike_checksum": w[30],
        "flash_acr": w[31], "scb_ccr": w[32],
        "arena_used": w[33], "itcm_code_bytes": w[34],
    }


def ms(cyc, hz):
    return cyc / hz * 1000.0


def report(rows):
    if not rows:
        return
    base = rows[0][1]
    print()
    print(f"{'변형':<6} {'ms/img':>9} {'배율':>6} {'encode':>8} {'invoke':>9} "
          f"{'cpl1':>8} {'cpl2':>8} {'lifout':>7} {'memcpy':>7} {'FLASH_ACR':>10} {'ITCM':>6} {'검증':>4}")
    for name, r in rows:
        hz = r["sysclk_hz"]
        ok = (r["spike_checksum"] == base["spike_checksum"] and r["preds"] == base["preds"])
        print(f"{name:<6} {ms(r['cyc_per_image'], hz):>9.1f} "
              f"{base['cyc_per_image'] / r['cyc_per_image']:>5.2f}x "
              f"{ms(r['cyc_encode'], hz):>7.1f} {ms(r['cyc_invoke'], hz):>8.1f} "
              f"{ms(r['cyc_cpl1'], hz):>7.1f} {ms(r['cyc_cpl2'], hz):>7.1f} "
              f"{ms(r['cyc_lifout'], hz):>6.1f} {ms(r['cyc_memcpy'], hz):>6.1f} "
              f"0x{r['flash_acr']:08X} {r['itcm_code_bytes']:>6} {'OK' if ok else 'FAIL':>4}")
    print(f"\n기준 checksum=0x{base['spike_checksum']:08X}  preds={base['preds']}  "
          f"labels={base['labels']}  correct={base['correct']}/{base['images']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--fetch-only", action="store_true",
                    help="플래시 없이 현재 보드 결과만 회수 (어느 변형이 올라가 있는지 모르므로 "
                         "빌드된 모든 변형의 g_snn_bench 주소를 차례로 짚어 magic이 맞는 것을 쓴다)")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    if args.fetch_only:
        # 보드에 어떤 변형이 올라가 있는지 모르므로, 빌드된 모든 변형의 심볼
        # 주소를 한 번씩 읽어 magic+done이 맞는 것을 찾는다. (변형에 따라
        # .dtcm_bss가 .bss를 밀어 g_snn_bench 주소가 ~60KB 다르다.)
        seen = set()
        for name in args.variants:
            elf = PROJ / f"build-bench-{name}" / "ImageClassification.elf"
            if not elf.exists():
                continue
            addr = bench_addr(elf)
            if addr in seen:
                continue
            seen.add(addr)
            w = read_words(addr, 2)
            if len(w) == 2 and w[0] == MAGIC and w[1] == 1:
                r = parse(read_words(addr, N_WORDS))
                hz = r["sysclk_hz"]
                print(f"[fetch] 0x{addr:08X} ({name} 레이아웃, opt_flags=0x{r['opt_flags']:x}): "
                      f"{ms(r['cyc_per_image'], hz):.1f} ms/img  "
                      f"checksum 0x{r['spike_checksum']:08X}  correct {r['correct']}/{r['images']}")
                return
        raise SystemExit("완료된 결과를 찾지 못했다 — 시도한 주소에 magic/done 없음")

    rows = []
    for name in args.variants:
        opts = VARIANTS[name]
        print(f"\n=== [{name}] SNN_OPT: {'+'.join(opts) or '(없음)'} ===")
        bdir = PROJ / f"build-bench-{name}"
        if not args.skip_build:
            print("  빌드...")
            build(name, opts)
        elf = bdir / "ImageClassification.elf"
        addr = bench_addr(elf)
        if args.build_only:
            print(f"  빌드 완료 (g_snn_bench @ 0x{addr:08X})")
            continue
        print(f"  플래싱... (g_snn_bench @ 0x{addr:08X})")
        flash(bdir / "ImageClassification.bin")
        time.sleep(1.0)
        print("  측정 대기...")
        r = parse(wait_and_fetch(addr, args.timeout))
        rows.append((name, r))
        hz = r["sysclk_hz"]
        print(f"  {ms(r['cyc_per_image'], hz):.1f} ms/img  "
              f"(invoke {ms(r['cyc_invoke'], hz):.1f}, encode {ms(r['cyc_encode'], hz):.1f})  "
              f"checksum 0x{r['spike_checksum']:08X}")

    if rows and not args.build_only:
        report(rows)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        outpath = RESULTS_DIR / "snn_bench_results.json"
        payload = {name: r for name, r in rows}
        outpath.write_text(json.dumps(payload, indent=2))
        print(f"\n저장: {outpath}")


if __name__ == "__main__":
    sys.exit(main())
