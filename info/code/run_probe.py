#!/usr/bin/env python3
"""
프로브 펌웨어 빌드 → 플래싱 → 실행 대기 → 결과 회수를 한 번에 수행한다.

결과는 UART(ST-LINK VCP)로도 나가지만, 회수는 0x20040000의 결과 버퍼를
ST-LINK로 직접 읽는 방식을 쓴다. 시리얼 포트 오픈 타이밍이나 드롭에 좌우되지
않고, 펌웨어가 도중에 죽어도 그때까지 쌓인 로그를 그대로 건질 수 있다.

버퍼 헤더 레이아웃 (probe.h의 probe_result_t):
    +0  magic    0x50524231("PRB1")=실행 중 / 0x50524244("PRBD")=정상 종료
    (버퍼는 0x20035000, 진단 영역은 0x20034000에 따로 있다)
    +4  length   text[]에 쌓인 바이트 수
    +8  overflow 버퍼 초과로 버려진 바이트 수
    +12 sysclk_hz
    +16 text[]   로그 본문

사용법:
    python3 run_probe.py                 # 빌드+플래시+회수
    python3 run_probe.py --no-build      # 이미 빌드된 바이너리로 플래시
    python3 run_probe.py --fetch-only    # 플래시 없이 현재 버퍼만 회수
    python3 run_probe.py --restore       # 원래 SNN 펌웨어로 복원
"""

import argparse
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
TOOLCHAIN = "/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/cmake/arm-none-eabi.cmake"

HERE = Path(__file__).resolve().parent
FW_DIR = HERE / "fw_hwprobe"
BUILD = FW_DIR / "build"
RESULTS = HERE.parent / "results"
RAW = HERE.parent / "raw"

RESULT_ADDR = 0x20035000
RESULT_SIZE = 92 * 1024
MAGIC_RUNNING = 0x50524231
MAGIC_DONE = 0x50524244

ANSI = re.compile(r"\033\[[0-9;]*m")


def cli(args, timeout=300):
    p = subprocess.run([PROG] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, ANSI.sub("", p.stdout + p.stderr)


def build():
    print("[1] 펌웨어 빌드")
    if not (BUILD / "build.ninja").exists():
        subprocess.run(
            ["cmake", "-S", str(FW_DIR), "-B", str(BUILD), "-G", "Ninja",
             f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN}", "-DCMAKE_BUILD_TYPE=Release"],
            check=True, capture_output=True, text=True,
        )
    p = subprocess.run(["cmake", "--build", str(BUILD)], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit("빌드 실패")
    for line in (p.stdout + p.stderr).splitlines():
        if any(k in line for k in ("Memory region", "ITCM:", "DTCM:", "RAM:", "DIAG:",
                                   "RESULT:", "SRAM2:", "FLASH:", "text\t", "error", "Error")):
            print("   " + line.rstrip())
    return BUILD / "hwprobe.bin"


def flash(binpath):
    print(f"[2] 플래싱: {binpath}")
    rc, out = cli(["-c", "port=SWD", "mode=UR", "-w", str(binpath), "0x08000000", "-v", "-rst"])
    ok = "successfully" in out.lower() or "Verification OK" in out
    for line in out.splitlines():
        if any(k in line for k in ("Download", "Verification", "Error", "File download",
                                   "RUNNING", "successfully")):
            print("   " + line.strip())
    if not ok:
        print(out[-2000:])
        raise SystemExit("플래싱 실패")


def read_header():
    rc, out = cli(["-q", "-c", "port=SWD", "mode=HOTPLUG", "-r32", hex(RESULT_ADDR), "0x10"])
    m = re.search(r"0x%08X\s*:\s*([0-9A-Fa-f]{8})\s+([0-9A-Fa-f]{8})\s+([0-9A-Fa-f]{8})\s+([0-9A-Fa-f]{8})"
                  % RESULT_ADDR, out)
    if not m:
        return None
    return [int(g, 16) for g in m.groups()]


def wait_done(timeout_s):
    print("[3] 측정 완료 대기 (결과 버퍼 magic 폴링)")
    t0 = time.time()
    last_len = -1
    while time.time() - t0 < timeout_s:
        hdr = read_header()
        if hdr:
            magic, length, overflow, sysclk = hdr
            tag = {MAGIC_RUNNING: "실행 중", MAGIC_DONE: "완료"}.get(magic, f"0x{magic:08X}")
            if length != last_len:
                print(f"   [{time.time()-t0:5.1f}s] {tag}  로그 {length} bytes  "
                      f"overflow {overflow}  SYSCLK {sysclk} Hz")
                last_len = length
            if magic == MAGIC_DONE:
                return hdr
        time.sleep(2.0)
    print("   시간 초과 — 그때까지 쌓인 내용을 회수한다")
    return read_header()


def fetch(hdr):
    print("[4] 결과 회수")
    if not hdr:
        raise SystemExit("결과 버퍼 헤더를 읽지 못했다")
    magic, length, overflow, sysclk = hdr
    if length == 0 or length > RESULT_SIZE:
        print(f"   경고: length={length} 가 비정상. 버퍼 전체를 뜬다.")
        length = RESULT_SIZE - 16
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    binpath = RAW / "probe_result_raw.bin"
    # 헤더 16바이트 + 본문. 4바이트 정렬로 올림.
    total = (16 + length + 3) & ~3
    rc, out = cli(["-q", "-c", "port=SWD", "mode=HOTPLUG",
                   "-r", hex(RESULT_ADDR), hex(total), str(binpath)])
    if not binpath.exists():
        print(out[-2000:])
        raise SystemExit("결과 버퍼 읽기 실패")
    data = binpath.read_bytes()
    text = data[16:16 + length].decode("utf-8", errors="replace")
    outpath = RESULTS / "probe_output.txt"
    header = (
        f"# STM32F746G-DISCO 온디바이스 프로브 결과\n"
        f"# magic=0x{magic:08X} ({'정상 종료' if magic == MAGIC_DONE else '미완/실행중'})\n"
        f"# 로그 {length} bytes, 버퍼 초과로 버려진 {overflow} bytes\n"
        f"# 펌웨어가 보고한 SYSCLK = {sysclk} Hz\n"
        f"# 원시 바이너리: {binpath}\n"
        f"{'=' * 70}\n\n"
    )
    outpath.write_text(header + text)
    print(f"   저장: {outpath}  ({length} bytes, overflow {overflow})")
    if overflow:
        print(f"   경고: {overflow} bytes가 버퍼를 넘쳐 잘렸다. 출력량을 줄이거나 버퍼를 키워야 한다.")
    return outpath


def read_fault_log():
    """probe_system.c의 폴트 핸들러가 진단 영역(0x20034010)에 남긴 기록을 확인한다."""
    rc, out = cli(["-q", "-c", "port=SWD", "mode=HOTPLUG", "-r32", "0x20034010", "0x20"])
    words = [int(w, 16) for w in re.findall(r"\b([0-9A-F]{8})\b", out)]
    if words and words[0] == 0xDEADBEEF:
        kinds = {1: "NMI", 2: "HardFault", 3: "MemManage", 4: "BusFault", 5: "UsageFault"}
        print("\n!! 폴트 발생 기록 있음 !!")
        print(f"   종류   : {kinds.get(words[1], words[1])}")
        print(f"   CFSR   : 0x{words[2]:08X}")
        print(f"   HFSR   : 0x{words[3]:08X}")
        print(f"   MMFAR  : 0x{words[4]:08X}")
        print(f"   BFAR   : 0x{words[5]:08X}")
        print(f"   SHCSR  : 0x{words[6]:08X}")
        print(f"   ABFSR  : 0x{words[7]:08X}")
        return True
    return False


def restore():
    orig = Path("/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/build/ImageClassification.bin")
    if not orig.exists():
        raise SystemExit(f"원본 펌웨어가 없다: {orig}")
    print(f"[복원] 원래 SNN 펌웨어로 되돌린다: {orig}")
    rc, out = cli(["-c", "port=SWD", "mode=UR", "-w", str(orig), "0x08000000", "-v", "-rst"])
    print("   " + ("성공" if "successfully" in out.lower() else "실패"))
    if "successfully" not in out.lower():
        print(out[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--timeout", type=float, default=180.0, help="측정 완료 대기 시간(초)")
    args = ap.parse_args()

    if args.restore:
        restore()
        return

    if args.fetch_only:
        hdr = read_header()
        fetch(hdr)
        read_fault_log()
        return

    binpath = BUILD / "hwprobe.bin" if args.no_build else build()
    flash(binpath)
    time.sleep(1.0)
    hdr = wait_done(args.timeout)
    fetch(hdr)
    read_fault_log()
    print("\n다음: python3 run_probe.py --restore  로 원래 펌웨어를 되돌릴 수 있다.")


if __name__ == "__main__":
    sys.exit(main())
