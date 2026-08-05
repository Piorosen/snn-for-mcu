#!/usr/bin/env python3
"""
STM32F746G-DISCO ST-LINK 메모리/레지스터 전방위 덤프 도구.

STM32_Programmer_CLI 의 임의 주소 읽기(-r / -r32)를 이용해, 펌웨어를 플래싱하지
않고도 코어(SCS/디버그 블록), 팩토리 프로그램 영역, 주요 페리페럴 레지스터를
그대로 떠온다. 결과는 info/raw/regions/*.bin 에 원시 바이너리로 남기고,
region_index.json 에 성공/실패와 메타데이터를 기록한다.

주의: HOTPLUG 모드는 "현재 보드에서 돌고 있는 펌웨어의 상태"를 읽는다.
      코어/팩토리 영역은 펌웨어와 무관하므로 하드웨어 고유값이지만,
      RCC/FLASH/GPIO 등 페리페럴 값은 실행 중 펌웨어의 설정임에 유의.
      리셋 직후 기본값이 필요하면 --mode UR 로 다시 돌린다.

사용법:
    python3 stlink_dump.py [--mode HOTPLUG|UR] [--tag NAME]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROG = (
    "/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/"
    "com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.macosaarch64_1.0.100.202603051304"
    "/tools/bin/STM32_Programmer_CLI"
)

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"

ANSI = re.compile(r"\033\[[0-9;]*m")

# (이름, 시작주소, 바이트수, 설명)
# PPB(0xE00.....)와 팩토리 영역은 페리페럴 클럭과 무관하게 항상 읽힌다.
# 페리페럴 영역은 해당 RCC 클럭이 꺼져 있으면 0 또는 접근 실패가 날 수 있다.
REGIONS = [
    # ---- Cortex-M7 Private Peripheral Bus: 코어 자체 정보 (하드웨어 고유) ----
    ("ppb_itm",       0xE0000000, 0x1000, "ITM (Instrumentation Trace Macrocell)"),
    ("ppb_dwt",       0xE0001000, 0x1000, "DWT (Data Watchpoint & Trace) - 사이클 카운터 등"),
    ("ppb_fpb",       0xE0002000, 0x1000, "FPB (Flash Patch & Breakpoint)"),
    ("ppb_scs",       0xE000E000, 0x1000, "SCS: SysTick/NVIC/SCB/MPU/FPU/캐시제어 전체"),
    ("ppb_tpiu",      0xE0040000, 0x1000, "TPIU (Trace Port Interface Unit)"),
    ("ppb_etm",       0xE0041000, 0x1000, "ETM (Embedded Trace Macrocell)"),
    ("ppb_dbgmcu",    0xE0042000, 0x0100, "DBGMCU: IDCODE(디바이스/리비전 ID), 디버그 프리즈"),
    ("ppb_romtable",  0xE00FF000, 0x1000, "CoreSight ROM table: 탑재된 디버그 컴포넌트 목록"),

    # ---- 팩토리 프로그램 영역 (ST가 공장에서 기록, 읽기 전용, 하드웨어 고유) ----
    ("factory_uid_area", 0x1FF0F000, 0x1000, "UID/Flash size/ADC 캘리브레이션이 있는 팩토리 영역"),
    ("factory_pkg_area", 0x1FF07800, 0x0800, "패키지 데이터 레지스터 영역(0x1FF07BF0)"),
    ("sysmem_bootloader", 0x1FF00000, 0x8000, "시스템 메모리 부트로더 코드 (ST 내장)"),

    # ---- 옵션 바이트 ----
    ("option_bytes", 0x1FFF0000, 0x0020, "옵션 바이트 (RDP/BOR/WDG/BOOT 설정)"),

    # ---- 시스템 제어 페리페럴 (항상 클럭 공급) ----
    ("rcc",     0x40023800, 0x0100, "RCC: 클럭 트리 현재 설정 + 페리페럴 클럭 인에이블 상태"),
    ("flash_if", 0x40023C00, 0x0100, "FLASH 인터페이스: ACR(wait state/ART), OPTCR"),
    ("pwr",     0x40007000, 0x0020, "PWR: 전압 스케일링/over-drive 상태"),
    ("syscfg",  0x40013800, 0x0040, "SYSCFG: 메모리 리맵, EXTI 소스"),
    ("crc",     0x40023000, 0x0020, "CRC 계산 유닛"),

    # ---- GPIO 포트 A~K (현재 핀 설정 = 보드 배선 추론 근거) ----
    ("gpio_all", 0x40020000, 0x2C00, "GPIOA~GPIOK (포트당 0x400, 11포트)"),

    # ---- DMA ----
    ("dma1", 0x40026000, 0x0100, "DMA1 컨트롤러"),
    ("dma2", 0x40026400, 0x0100, "DMA2 컨트롤러"),
    ("dma2d", 0x4002B000, 0x0100, "DMA2D (Chrom-ART 그래픽 가속기)"),

    # ---- 외부 메모리 컨트롤러 ----
    ("fmc",    0xA0000000, 0x0200, "FMC: SDRAM/NOR/NAND 컨트롤러 (SDCR/SDTR 타이밍)"),
    ("quadspi", 0xA0001000, 0x0100, "QUADSPI 컨트롤러"),

    # ---- 디스플레이/이미지 ----
    ("ltdc", 0x40016800, 0x0200, "LTDC LCD-TFT 컨트롤러 (해상도/타이밍 현재 설정)"),
    ("dcmi", 0x50050000, 0x0100, "DCMI 카메라 인터페이스"),
    ("jpeg", 0x50051000, 0x0100, "JPEG 하드웨어 코덱"),

    # ---- 통신/기타 페리페럴 (존재 및 현재 설정 확인) ----
    ("usart1", 0x40011000, 0x0040, "USART1 (ST-LINK VCP 후보)"),
    ("usart6", 0x40011400, 0x0040, "USART6"),
    ("uart_bank_apb1", 0x40004400, 0x0400, "USART2/3, UART4/5 (APB1 연속 영역)"),
    ("i2c_bank", 0x40005400, 0x0C00, "I2C1/2/3 (APB1)"),
    ("spi_bank_apb1", 0x40003800, 0x0C00, "SPI2/3, SPDIFRX 영역"),
    ("adc_bank", 0x40012000, 0x0400, "ADC1/2/3 + 공통 레지스터"),
    ("sdmmc1", 0x40012C00, 0x0100, "SDMMC1 (microSD)"),
    ("rng",  0x50060800, 0x0020, "RNG 난수 발생기"),
    ("rtc",  0x40002800, 0x0100, "RTC + 백업 레지스터"),
    ("iwdg", 0x40003000, 0x0020, "독립 워치독"),
    ("wwdg", 0x40002C00, 0x0020, "윈도우 워치독"),
    ("tim1", 0x40010000, 0x0100, "TIM1 (고급 제어 타이머)"),
    ("tim2", 0x40000000, 0x0100, "TIM2 (32비트 범용 타이머)"),
    ("eth",  0x40028000, 0x0400, "Ethernet MAC"),
    ("usb_hs_core", 0x40040000, 0x0400, "USB OTG HS 코어 레지스터"),
    ("usb_fs_core", 0x50000000, 0x0400, "USB OTG FS 코어 레지스터"),

    # ---- 메모리 내용 샘플 (내용 자체보다 '접근 가능한가'가 정보) ----
    ("mem_itcm_head",  0x00000000, 0x0040, "ITCM 선두 (0x00000000)"),
    ("mem_flash_itcm_alias", 0x00200000, 0x0040, "Flash의 ITCM 별칭 (0x00200000)"),
    ("mem_flash_axim", 0x08000000, 0x0040, "Flash AXIM 별칭 (0x08000000) = 벡터 테이블"),
    ("mem_dtcm_head",  0x20000000, 0x0040, "DTCM 선두 (0x20000000)"),
    ("mem_sram1_head", 0x20010000, 0x0040, "SRAM1 선두 (0x20010000)"),
    ("mem_sram2_head", 0x2004C000, 0x0040, "SRAM2 선두 (0x2004C000)"),
    ("mem_bkpsram",    0x40024000, 0x0040, "백업 SRAM (0x40024000)"),
    ("mem_sdram_head", 0xC0000000, 0x0040, "외부 SDRAM 선두 (FMC Bank5/6, 0xC0000000)"),
    ("mem_qspi_head",  0x90000000, 0x0040, "QSPI 메모리맵 모드 선두 (0x90000000)"),
]

# 개별로 읽을 32비트 레지스터: (이름, 주소, 설명)
# 블록 덤프로 이미 커버되는 것도 있지만, 교차 확인용으로 명시적으로 한 번 더 읽는다.
SINGLE_REGS = [
    ("CPUID",        0xE000ED00, "코어 식별 (Cortex-M7 r0p1 = 0x410FC271)"),
    ("DBGMCU_IDCODE", 0xE0042000, "DEV_ID[11:0] / REV_ID[31:16]"),
    ("FLASH_SIZE",   0x1FF0F442, "Flash 용량(KB), 하위 16비트"),
    ("PACKAGE",      0x1FF07BF0, "패키지 데이터 레지스터"),
    ("UID0",         0x1FF0F420, "Unique device ID [31:0]"),
    ("UID1",         0x1FF0F424, "Unique device ID [63:32]"),
    ("UID2",         0x1FF0F428, "Unique device ID [95:64]"),
    ("CLIDR",        0xE000ED78, "Cache Level ID"),
    ("CTR",          0xE000ED7C, "Cache Type Register"),
    ("CCSIDR",       0xE000ED80, "Cache Size ID (CSSELR로 선택된 캐시)"),
    ("CSSELR",       0xE000ED84, "Cache Size Selection"),
    ("MPU_TYPE",     0xE000ED90, "MPU 리전 수"),
    ("MVFR0",        0xE000EF40, "FPU 기능 레지스터 0"),
    ("MVFR1",        0xE000EF44, "FPU 기능 레지스터 1"),
    ("MVFR2",        0xE000EF48, "FPU 기능 레지스터 2"),
    ("ITCMCR",       0xE000EF90, "ITCM 제어"),
    ("DTCMCR",       0xE000EF94, "DTCM 제어"),
    ("AHBPCR",       0xE000EF98, "AHBP(페리페럴 포트) 제어"),
    ("CACR",         0xE000EF9C, "L1 캐시 제어 (FORCEWT/ECCEN/SIWT)"),
    ("AHBSCR",       0xE000EFA0, "AHBS(슬레이브 포트) 제어"),
    ("ABFSR",        0xE000EFA8, "Auxiliary Bus Fault Status"),
    ("ACTLR",        0xE000E008, "Auxiliary Control (분기폴딩/스토어버퍼 비활성 비트)"),
    ("SCB_CCR",      0xE000ED14, "Configuration & Control (IC/DC 인에이블 비트)"),
    ("SCB_CPACR",    0xE000ED88, "코프로세서 접근 (FPU 인에이블)"),
    ("SYST_CALIB",   0xE000E01C, "SysTick 캘리브레이션 (TENMS)"),
    ("ICTR",         0xE000E004, "인터럽트 컨트롤러 타입 (지원 IRQ 수)"),
    ("FP_CTRL",      0xE0002000, "FPB: 하드웨어 브레이크포인트 개수"),
    ("DWT_CTRL",     0xE0001000, "DWT: 지원 비교기 수 및 기능"),
    ("ID_PFR0",      0xE000ED40, "프로세서 기능 레지스터 0"),
    ("ID_PFR1",      0xE000ED44, "프로세서 기능 레지스터 1"),
    ("ID_DFR0",      0xE000ED48, "디버그 기능 레지스터 0"),
    ("ID_AFR0",      0xE000ED4C, "보조 기능 레지스터 0"),
    ("ID_MMFR0",     0xE000ED50, "메모리 모델 기능 0"),
    ("ID_MMFR1",     0xE000ED54, "메모리 모델 기능 1"),
    ("ID_MMFR2",     0xE000ED58, "메모리 모델 기능 2"),
    ("ID_MMFR3",     0xE000ED5C, "메모리 모델 기능 3"),
    ("ID_ISAR0",     0xE000ED60, "명령어 집합 속성 0"),
    ("ID_ISAR1",     0xE000ED64, "명령어 집합 속성 1"),
    ("ID_ISAR2",     0xE000ED68, "명령어 집합 속성 2"),
    ("ID_ISAR3",     0xE000ED6C, "명령어 집합 속성 3"),
    ("ID_ISAR4",     0xE000ED70, "명령어 집합 속성 4"),
    ("RCC_CR",       0x40023800, "RCC 제어: HSE/HSI/PLL 상태"),
    ("RCC_PLLCFGR",  0x40023804, "PLL 설정 (M/N/P/Q)"),
    ("RCC_CFGR",     0x40023808, "SYSCLK 소스, AHB/APB 프리스케일러"),
    ("FLASH_ACR",    0x40023C00, "Flash wait state + ART accelerator"),
    ("FLASH_OPTCR",  0x40023C14, "옵션 바이트 제어 (RDP/BOR/nRST)"),
    ("PWR_CR1",      0x40007000, "전압 스케일링/over-drive"),
    ("PWR_CSR1",     0x40007004, "전압 스케일링 준비 플래그"),
]


def run_cli(args, timeout=180):
    cmd = [PROG] + args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = ANSI.sub("", p.stdout + p.stderr)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def dump_regions(mode, tag):
    outdir = RAW / f"regions_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, addr, size, desc in REGIONS:
        path = outdir / f"{name}.bin"
        rc, out = run_cli([
            "-q", "-c", f"port=SWD", f"mode={mode}",
            "-r", hex(addr), hex(size), str(path),
        ])
        ok = path.exists() and path.stat().st_size == size and "Error" not in out
        entry = {
            "name": name, "addr": f"0x{addr:08X}", "size": size,
            "desc": desc, "ok": ok,
            "file": str(path.relative_to(RAW.parent)) if ok else None,
        }
        if not ok:
            entry["error"] = out.strip().splitlines()[-3:] if out else ["no output"]
            if path.exists():
                path.unlink()
        index.append(entry)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:22s} 0x{addr:08X} +{size:#07x}  {desc}")
    return index, outdir


def dump_single_regs(mode, tag):
    """개별 32비트 레지스터를 한 번의 CLI 호출로 몰아서 읽는다."""
    args = ["-q", "-c", "port=SWD", f"mode={mode}"]
    for _, addr, _ in SINGLE_REGS:
        args += ["-r32", hex(addr), "0x4"]
    rc, out = run_cli(args)
    values = {}
    for line in out.splitlines():
        m = re.match(r"\s*(0x[0-9A-Fa-f]{8})\s*:\s*([0-9A-Fa-f]{8})", line.strip())
        if m:
            values[int(m.group(1), 16)] = int(m.group(2), 16)
    result = []
    for name, addr, desc in SINGLE_REGS:
        v = values.get(addr)
        result.append({
            "name": name, "addr": f"0x{addr:08X}",
            "value": f"0x{v:08X}" if v is not None else None,
            "desc": desc,
        })
        print(f"  {name:14s} @ 0x{addr:08X} = " + (f"0x{v:08X}" if v is not None else "READ FAIL") + f"   {desc}")
    return result


def dump_cache_ids(tag):
    """CSSELR을 조작하여 D-cache/I-cache의 CCSIDR을 각각 읽는다.

    CSSELR 쓰기가 필요하므로 코어를 halt한 상태에서 수행한다.
    (캐시 유지관리 선택 레지스터일 뿐이라 실행 중인 코드의 동작에는 영향 없음)
    """
    out_all = []
    res = {}
    for label, sel in (("dcache_L1", 0), ("icache_L1", 1)):
        rc, out = run_cli([
            "-q", "-c", "port=SWD", "mode=HOTPLUG", "-halt",
            "-w32", "0xE000ED84", hex(sel),
            "-r32", "0xE000ED80", "0x4",
            "-r32", "0xE000ED78", "0x4",
            "-r32", "0xE000ED7C", "0x4",
        ])
        out_all.append(f"--- CSSELR={sel} ({label}) ---\n{out}")
        vals = re.findall(r"0x([0-9A-Fa-f]{8})\s*:\s*([0-9A-Fa-f]{8})", out)
        got = {int(a, 16): int(b, 16) for a, b in vals}
        res[label] = {
            "CSSELR_written": sel,
            "CCSIDR": f"0x{got[0xE000ED80]:08X}" if 0xE000ED80 in got else None,
            "CLIDR": f"0x{got[0xE000ED78]:08X}" if 0xE000ED78 in got else None,
            "CTR": f"0x{got[0xE000ED7C]:08X}" if 0xE000ED7C in got else None,
        }
        print(f"  {label}: CCSIDR={res[label]['CCSIDR']}  CLIDR={res[label]['CLIDR']}  CTR={res[label]['CTR']}")
    # 코어를 다시 실행 상태로 되돌린다
    run_cli(["-q", "-c", "port=SWD", "mode=HOTPLUG", "-run"])
    (RAW / f"cache_id_raw_{tag}.txt").write_text("\n".join(out_all))
    return res


def probe_banner(mode):
    """CubeProgrammer가 보고하는 연결 배너(디바이스 ID/리비전/전압) 원문 저장."""
    rc, out = run_cli(["-c", "port=SWD", f"mode={mode}"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="HOTPLUG", choices=["HOTPLUG", "UR", "NORMAL"],
                    help="HOTPLUG=실행 중 상태 읽기, UR=리셋 상태에서 읽기(기본값 확인용)")
    ap.add_argument("--tag", default=None, help="출력 디렉터리 접미사 (기본: mode 소문자)")
    ap.add_argument("--skip-cache-ids", action="store_true",
                    help="CSSELR 쓰기(코어 halt)를 건너뛴다")
    args = ap.parse_args()
    tag = args.tag or args.mode.lower()

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"=== STM32F746 ST-LINK 덤프 (mode={args.mode}, tag={tag}) ===\n")

    print("[0] 연결 배너")
    banner = probe_banner(args.mode)
    (RAW / f"connect_banner_{tag}.txt").write_text(banner)
    for line in banner.splitlines():
        if any(k in line for k in ("ST-LINK", "Board", "Voltage", "Device", "Revision", "NVM", "BL Version", "SWD freq")):
            print("   " + line.strip())

    print("\n[1] 블록 덤프")
    index, outdir = dump_regions(args.mode, tag)

    print("\n[2] 개별 레지스터")
    singles = dump_single_regs(args.mode, tag)

    cache_ids = None
    if not args.skip_cache_ids:
        print("\n[3] 캐시 ID 레지스터 (CSSELR 조작)")
        cache_ids = dump_cache_ids(tag)

    meta = {
        "mode": args.mode,
        "tag": tag,
        "programmer": PROG,
        "regions": index,
        "single_regs": singles,
        "cache_ids": cache_ids,
        "note": (
            "PPB(0xE00.....)/팩토리 영역은 하드웨어 고유값. "
            "페리페럴 레지스터는 덤프 시점에 보드에서 실행 중이던 펌웨어의 설정임."
        ),
    }
    outjson = RAW / f"stlink_dump_{tag}.json"
    outjson.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    nok = sum(1 for r in index if r["ok"])
    print(f"\n완료: 영역 {nok}/{len(index)} 성공, 개별 레지스터 {len(singles)}개")
    print(f"  인덱스: {outjson}")
    print(f"  바이너리: {outdir}")


if __name__ == "__main__":
    sys.exit(main())
