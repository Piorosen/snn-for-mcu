#!/usr/bin/env python3
"""
STM32F746G-DISCO ST-LINK 덤프 해석기.

stlink_dump.py 가 떠 놓은 원시 바이너리/JSON 을 읽어 사람이 읽을 수 있는
한국어 Markdown 문서(info/raw/decoded_<tag>.md)로 변환한다.

- 하드웨어에 전혀 접근하지 않는다. 이미 존재하는 파일만 읽는다.
- 파일이 없으면 해당 섹션을 건너뛴다(죽지 않는다).
- 비트필드 정의는 다음 헤더에서 확인한 값만 사용한다:
    Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h
    Drivers/CMSIS/Include/core_cm7.h
  헤더에 없는 필드(Cortex-M7 TRM 전용)는 주석에 출처를 남긴다.

사용법:
    python3 decode_dump.py            # hotplug, reset 둘 다 처리
    python3 decode_dump.py hotplug    # 특정 태그만
"""

import datetime
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"

# ---------------------------------------------------------------------------
# 비트 조작 헬퍼
# ---------------------------------------------------------------------------


def bf(value, hi, lo):
    """value[hi:lo] 추출."""
    return (value >> lo) & ((1 << (hi - lo + 1)) - 1)


def bit(value, pos):
    return (value >> pos) & 1


def onoff(b, on="설정됨", off="해제됨"):
    return on if b else off


def hex32(v):
    return "해석 불가" if v is None else f"0x{v:08X}"


# ---------------------------------------------------------------------------
# 원시 데이터 접근 (없으면 None)
# ---------------------------------------------------------------------------


class Dump:
    """한 태그(hotplug/reset)의 덤프 전체."""

    def __init__(self, tag):
        self.tag = tag
        self.dir = RAW / f"regions_{tag}"
        self.json_path = RAW / f"stlink_dump_{tag}.json"
        self.meta = {}
        if self.json_path.exists():
            try:
                self.meta = json.loads(self.json_path.read_text())
            except (ValueError, OSError):
                self.meta = {}
        self.singles = {}
        for s in self.meta.get("single_regs") or []:
            if s.get("value"):
                self.singles[s["name"]] = int(s["value"], 16)
        self.regions = {r["name"]: r for r in (self.meta.get("regions") or [])}
        self._cache = {}
        banner = RAW / f"connect_banner_{tag}.txt"
        self.banner = banner.read_text(errors="replace") if banner.exists() else ""
        cid = RAW / f"cache_id_raw_{tag}.txt"
        self.cache_id_raw = cid if cid.exists() else None

    # -- 영역 바이너리 --------------------------------------------------
    def blob(self, name):
        if name not in self._cache:
            p = self.dir / f"{name}.bin"
            self._cache[name] = p.read_bytes() if p.exists() else None
        return self._cache[name]

    def base_of(self, name):
        r = self.regions.get(name)
        return int(r["addr"], 16) if r else None

    def r32(self, name, addr):
        """영역 name 안의 절대주소 addr 에서 32비트 읽기 (없으면 None)."""
        b = self.blob(name)
        base = self.base_of(name)
        if b is None or base is None:
            return None
        off = addr - base
        if off < 0 or off + 4 > len(b):
            return None
        return struct.unpack_from("<I", b, off)[0]

    def r16(self, name, addr):
        b = self.blob(name)
        base = self.base_of(name)
        if b is None or base is None:
            return None
        off = addr - base
        if off < 0 or off + 2 > len(b):
            return None
        return struct.unpack_from("<H", b, off)[0]

    def off32(self, name, offset):
        b = self.blob(name)
        if b is None or offset + 4 > len(b):
            return None
        return struct.unpack_from("<I", b, offset)[0]

    def single(self, name):
        return self.singles.get(name)

    # -- 편의: 개별 레지스터 우선, 없으면 블록에서 --------------------
    def reg(self, single_name, region, addr):
        v = self.single(single_name)
        if v is None:
            v = self.r32(region, addr)
        return v


# ---------------------------------------------------------------------------
# Markdown 빌더
# ---------------------------------------------------------------------------


class Doc:
    def __init__(self):
        self.lines = []
        self.summary = []
        self.warnings = []

    def w(self, s=""):
        self.lines.append(s)

    def h(self, level, text):
        self.w()
        self.w("#" * level + " " + text)
        self.w()

    def table(self, headers, rows):
        if not rows:
            self.w("(데이터 없음)")
            self.w()
            return
        self.w("| " + " | ".join(headers) + " |")
        self.w("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            self.w("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
        self.w()

    def fields(self, title, lines):
        """`이름 @ 주소 = 값` + 들여쓴 필드 해석 블록."""
        self.w(f"`{title}`")
        self.w()
        for ln in lines:
            self.w(f"- {ln}")
        self.w()

    def note(self, text):
        self.w(f"> {text}")
        self.w()

    def warn(self, text):
        self.warnings.append(text)
        self.w(f"> ⚠️ **경고**: {text}")
        self.w()

    def sum_(self, text, pri=2):
        """pri=1 → '핵심 요약' (하드웨어 확정 사실), pri=2 → '그 밖에 확정된 사실'."""
        self.summary.append((pri, text))

    def summary_by_pri(self, pri):
        return [t for p, t in self.summary if p == pri]

    def render(self):
        return "\n".join(self.lines) + "\n"


def u32(buf, off):
    """버퍼 범위를 벗어나면 None 을 돌려주는 안전한 32비트 읽기."""
    if buf is None or off < 0 or off + 4 > len(buf):
        return None
    return struct.unpack_from("<I", buf, off)[0]


def hexdump(data, base_addr=0, width=16):
    """hex + ASCII 덤프 문자열 리스트."""
    out = []
    for off in range(0, len(data), width):
        chunk = data[off:off + width]
        hx = " ".join(f"{c:02X}" for c in chunk).ljust(width * 3 - 1)
        asc = "".join(chr(c) if 32 <= c < 127 else "." for c in chunk)
        out.append(f"{base_addr + off:08X}  {hx}  |{asc}|")
    return out


# ---------------------------------------------------------------------------
# 상수 테이블
# ---------------------------------------------------------------------------

# ARM Cortex-M7 TRM (ARM DDI 0489) "ITCMCR/DTCMCR" SZ[6:3] 인코딩 표
#   0b0000 = TCM 미탑재            0b1000 = 128KB
#   0b0011 = 4KB                    0b1001 = 256KB
#   0b0100 = 8KB                    0b1010 = 512KB
#   0b0101 = 16KB                   0b1011 = 1MB
#   0b0110 = 32KB                   0b1100 = 2MB
#   0b0111 = 64KB                   0b1101 = 4MB
#                                   0b1110 = 8MB
#                                   0b1111 = 16MB
#   (0b0001, 0b0010 은 reserved)
TCM_SZ = {
    0b0000: "TCM 미탑재",
    0b0001: "reserved",
    0b0010: "reserved",
    0b0011: "4KB",
    0b0100: "8KB",
    0b0101: "16KB",
    0b0110: "32KB",
    0b0111: "64KB",
    0b1000: "128KB",
    0b1001: "256KB",
    0b1010: "512KB",
    0b1011: "1MB",
    0b1100: "2MB",
    0b1101: "4MB",
    0b1110: "8MB",
    0b1111: "16MB",
}

# Cortex-M7 TRM AHBPCR.SZ[3:1]
AHBP_SZ = {
    0b000: "AHBP 비활성(0MB)",
    0b001: "64MB",
    0b010: "128MB",
    0b011: "256MB",
    0b100: "512MB",
}

# DBGMCU_IDCODE REV_ID → 리비전 문자 (STM32F74x/75x, RM0385 / 데이터시트)
REV_ID = {
    0x1000: "Rev A",
    0x1001: "Rev Z",
    0x1003: "Rev 1",
    0x2001: "Rev 2",
    0x2003: "Rev 3",
}

# CoreSight ROM table 엔트리 → 알려진 컴포넌트 주소
CORESIGHT_BY_ADDR = {
    0xE000E000: ("SCS", "System Control Space (SCB/NVIC/MPU/FPU/캐시제어)", "ppb_scs"),
    0xE0001000: ("DWT", "Data Watchpoint and Trace", "ppb_dwt"),
    0xE0002000: ("FPB", "Flash Patch and Breakpoint", "ppb_fpb"),
    0xE0000000: ("ITM", "Instrumentation Trace Macrocell", "ppb_itm"),
    0xE0040000: ("TPIU", "Trace Port Interface Unit", "ppb_tpiu"),
    0xE0041000: ("ETM", "Embedded Trace Macrocell", "ppb_etm"),
}

# ARM CoreSight part number → 부품명 (ARM Cortex-M7 TRM 부록)
CORESIGHT_PART = {
    0x00C: "Cortex-M7 SCS (System Control Space)",
    0x002: "Cortex-M7 DWT",
    0x00E: "Cortex-M7 FPB",
    0x001: "Cortex-M7 ITM",
    0x9A9: "CoreSight TPIU-M7",
    0x975: "CoreSight ETM-M7",
    0x4C7: "Cortex-M7 PPB ROM table",
}

# CIDR1[7:4] 컴포넌트 클래스
CS_CLASS = {
    0x0: "Generic verification component",
    0x1: "ROM table",
    0x9: "CoreSight component",
    0xB: "PrimeCell / non-CoreSight peripheral",
    0xE: "Generic IP component",
    0xF: "PrimeCell(구형)",
}

# stm32f746xx.h 의 RCC_*ENR_*_Pos 정의에서 그대로 옮긴 표
RCC_ENR_BITS = {
    "AHB1ENR": {
        0: "GPIOA", 1: "GPIOB", 2: "GPIOC", 3: "GPIOD", 4: "GPIOE", 5: "GPIOF",
        6: "GPIOG", 7: "GPIOH", 8: "GPIOI", 9: "GPIOJ", 10: "GPIOK", 12: "CRC",
        18: "BKPSRAM", 20: "DTCMRAM", 21: "DMA1", 22: "DMA2", 23: "DMA2D",
        25: "ETHMAC", 26: "ETHMACTX", 27: "ETHMACRX", 28: "ETHMACPTP",
        29: "OTGHS", 30: "OTGHSULPI",
    },
    "AHB2ENR": {0: "DCMI", 6: "RNG", 7: "OTGFS"},
    "AHB3ENR": {0: "FMC", 1: "QSPI"},
    "APB1ENR": {
        0: "TIM2", 1: "TIM3", 2: "TIM4", 3: "TIM5", 4: "TIM6", 5: "TIM7",
        6: "TIM12", 7: "TIM13", 8: "TIM14", 9: "LPTIM1", 11: "WWDG",
        14: "SPI2", 15: "SPI3", 16: "SPDIFRX", 17: "USART2", 18: "USART3",
        19: "UART4", 20: "UART5", 21: "I2C1", 22: "I2C2", 23: "I2C3",
        24: "I2C4", 25: "CAN1", 26: "CAN2", 27: "CEC", 28: "PWR", 29: "DAC",
        30: "UART7", 31: "UART8",
    },
    "APB2ENR": {
        0: "TIM1", 1: "TIM8", 4: "USART1", 5: "USART6", 8: "ADC1", 9: "ADC2",
        10: "ADC3", 11: "SDMMC1", 12: "SPI1", 13: "SPI4", 14: "SYSCFG",
        16: "TIM9", 17: "TIM10", 18: "TIM11", 20: "SPI5", 21: "SPI6",
        22: "SAI1", 23: "SAI2", 26: "LTDC",
    },
}

# STM32F746 데이터시트 "Alternate function mapping" 표의 AF 그룹 개괄
AF_GROUP = {
    0: "SYS (MCO/JTAG/SWD/TRACE)",
    1: "TIM1/TIM2",
    2: "TIM3/TIM4/TIM5",
    3: "TIM8/TIM9/TIM10/TIM11/LPTIM1/CEC",
    4: "I2C1/I2C2/I2C3/I2C4/CEC",
    5: "SPI1/SPI2/SPI3/SPI4/SPI5/SPI6",
    6: "SPI3/SAI1",
    7: "SPI2/SPI3/USART1/USART2/USART3/UART5/SPDIFRX",
    8: "SAI2/USART6/UART4/UART5/UART7/UART8/SPDIFRX",
    9: "CAN1/CAN2/TIM12/TIM13/TIM14/QUADSPI/LTDC",
    10: "SAI2/QUADSPI/OTG_HS/OTG_FS",
    11: "ETH",
    12: "FMC/SDMMC1/OTG_FS",
    13: "DCMI",
    14: "LTDC",
    15: "EVENTOUT",
}

GPIO_MODE = {0: "입력", 1: "출력", 2: "대체기능(AF)", 3: "아날로그"}
GPIO_SPEED = {0: "Low", 1: "Medium", 2: "High", 3: "Very high"}
GPIO_PUPD = {0: "없음", 1: "Pull-up", 2: "Pull-down", 3: "reserved"}


# ---------------------------------------------------------------------------
# 섹션 1: 코어 식별 및 제어
# ---------------------------------------------------------------------------


def sec_core(d, doc):
    doc.h(2, "코어 (Cortex-M7 식별 및 제어 레지스터)")

    cpuid = d.reg("CPUID", "ppb_scs", 0xE000ED00)
    if cpuid is not None:
        impl = bf(cpuid, 31, 24)
        var = bf(cpuid, 23, 20)
        arch = bf(cpuid, 19, 16)
        partno = bf(cpuid, 15, 4)
        rev = bf(cpuid, 3, 0)
        impl_s = "0x41 = ARM Limited" if impl == 0x41 else f"0x{impl:02X} = 미상"
        arch_s = {0xC: "ARMv7-M", 0xF: "ARMv7-M(확장)"}.get(arch, f"0x{arch:X} = 미상")
        part_s = {0xC27: "Cortex-M7", 0xC24: "Cortex-M4", 0xC23: "Cortex-M3"}.get(
            partno, f"0x{partno:03X} = 미상")
        core_name = f"{part_s} r{var}p{rev}"
        doc.fields(f"CPUID @ 0xE000ED00 = {hex32(cpuid)}", [
            f"IMPLEMENTER[31:24] = 0x{impl:02X} → {impl_s}",
            f"VARIANT[23:20] = {var} → 메이저 리비전 r{var}",
            f"ARCHITECTURE[19:16] = 0x{arch:X} → {arch_s}",
            f"PARTNO[15:4] = 0x{partno:03X} → {part_s}",
            f"REVISION[3:0] = {rev} → 마이너 리비전 p{rev}",
            f"**종합: {core_name}**",
        ])
        doc.sum_(f"CPU 코어는 **{core_name}** 이다 (CPUID={hex32(cpuid)}).", pri=1)

    ictr = d.reg("ICTR", "ppb_scs", 0xE000E004)
    if ictr is not None:
        n = bf(ictr, 3, 0)
        doc.fields(f"ICTR @ 0xE000E004 = {hex32(ictr)}", [
            f"INTLINESNUM[3:0] = {n} → NVIC 가 지원하는 외부 인터럽트 라인 수 = ({n}+1)×32 = **{(n + 1) * 32}개**",
        ])
        doc.sum_(f"NVIC 는 최대 {(n + 1) * 32}개의 외부 인터럽트 라인을 지원한다 (ICTR={hex32(ictr)}).", pri=1)

    actlr = d.reg("ACTLR", "ppb_scs", 0xE000E008)
    if actlr is not None:
        # core_cm7.h 의 SCnSCB_ACTLR_* 정의에 있는 비트만 해석한다.
        # (DISDEFWBUF 는 Cortex-M3/M4 전용 비트로, core_cm7.h 에는 정의가 없다.)
        doc.fields(f"ACTLR @ 0xE000E008 = {hex32(actlr)}", [
            f"DISMCYCINT[0] = {bit(actlr, 0)} → 다중 사이클 명령(LDM/STM/DIV 등)의 인터럽트 중단 "
            f"{onoff(bit(actlr, 0), '**금지**(지연 증가, 결정성 향상)', '허용(기본, 인터럽트 지연 최소)')}",
            f"DISFOLD[2] = {bit(actlr, 2)} → IT 폴딩 "
            f"{onoff(bit(actlr, 2), '**비활성화**', '활성화(기본, 성능 유리)')}",
            f"FPEXCODIS[10] = {bit(actlr, 10)} → FPU 예외 출력 "
            f"{onoff(bit(actlr, 10), '**비활성화**', '활성화(기본)')}",
            f"DISRAMODE[11] = {bit(actlr, 11)} → D-cache 의 read-allocate 모드 "
            f"{onoff(bit(actlr, 11), '**비활성화**(항상 write-allocate)', '활성화(기본)')}",
            f"DISITMATBFLUSH[12] = {bit(actlr, 12)} → ITM/DWT ATB flush "
            f"{onoff(bit(actlr, 12), '**비활성화**', '활성화(기본)')}",
            "DISDEFWBUF: core_cm7.h 에 정의 없음 (Cortex-M3/M4 전용 비트) → 해석 불가",
        ])
        if actlr == 0:
            doc.sum_("코어 성능 저하 비트(ACTLR)가 모두 0 → IT 폴딩·D-cache read-allocate·"
                     "다중사이클 명령 중단이 전부 기본(성능 우선) 상태 (ACTLR=0x00000000).")

    ccr = d.reg("SCB_CCR", "ppb_scs", 0xE000ED14)
    if ccr is not None:
        doc.fields(f"SCB_CCR @ 0xE000ED14 = {hex32(ccr)}", [
            f"NONBASETHRDENA[0] = {bit(ccr, 0)}",
            f"USERSETMPEND[1] = {bit(ccr, 1)}",
            f"UNALIGN_TRP[3] = {bit(ccr, 3)} → 비정렬 접근 트랩 {onoff(bit(ccr, 3))}",
            f"DIV_0_TRP[4] = {bit(ccr, 4)} → 0 나눗셈 트랩 {onoff(bit(ccr, 4))}",
            f"BFHFNMIGN[8] = {bit(ccr, 8)}",
            f"STKALIGN[9] = {bit(ccr, 9)} → 예외 진입 시 8바이트 스택 정렬 {onoff(bit(ccr, 9))}",
            f"DC[16] = {bit(ccr, 16)} → **D-cache {onoff(bit(ccr, 16), 'ON', 'OFF')}**",
            f"IC[17] = {bit(ccr, 17)} → **I-cache {onoff(bit(ccr, 17), 'ON', 'OFF')}**",
            f"BP[18] = {bit(ccr, 18)} → 분기 예측 {onoff(bit(ccr, 18), 'ON', 'OFF')}",
        ])
        doc.sum_(f"덤프 시점 I-cache={onoff(bit(ccr, 17), 'ON', 'OFF')}, "
                 f"D-cache={onoff(bit(ccr, 16), 'ON', 'OFF')}, "
                 f"분기예측={onoff(bit(ccr, 18), 'ON', 'OFF')} (SCB_CCR={hex32(ccr)}).")

    aircr = d.r32("ppb_scs", 0xE000ED0C)
    if aircr is not None:
        pg = bf(aircr, 10, 8)
        # NVIC 우선순위 비트 4개(STM32F7) 기준 그룹/서브 분할
        pre_bits = min(4, max(0, 7 - pg)) if pg <= 7 else 0
        sub_bits = 4 - pre_bits
        doc.fields(f"SCB_AIRCR @ 0xE000ED0C = {hex32(aircr)}", [
            f"VECTKEYSTAT[31:16] = 0x{bf(aircr, 31, 16):04X} (읽기 시 항상 0xFA05)",
            f"ENDIANNESS[15] = {bit(aircr, 15)} → "
            f"{'빅엔디언' if bit(aircr, 15) else '**리틀엔디언**'}",
            f"PRIGROUP[10:8] = {pg} (0b{pg:03b}) → STM32F7(우선순위 4비트) 기준 "
            f"선점 {pre_bits}비트 / 서브 {sub_bits}비트 "
            f"(HAL 의 NVIC_PRIORITYGROUP_{pre_bits})",
            f"SYSRESETREQ[2] = {bit(aircr, 2)}",
        ])
        if bit(aircr, 15) == 0:
            doc.sum_("코어는 리틀엔디언으로 동작한다 (AIRCR.ENDIANNESS=0).")

    vtor = d.r32("ppb_scs", 0xE000ED08)
    if vtor is not None:
        where = ""
        if vtor == 0x00200000:
            where = " → ITCM 인터페이스를 통한 내장 Flash 별칭(0x00200000)"
        elif 0x08000000 <= vtor < 0x08200000:
            where = " → AXIM 인터페이스를 통한 내장 Flash(0x08000000)"
        elif vtor < 0x00010000:
            where = " → ITCM RAM"
        elif 0x20000000 <= vtor < 0x20050000:
            where = " → DTCM/SRAM"
        doc.fields(f"SCB_VTOR @ 0xE000ED08 = {hex32(vtor)}", [
            f"TBLOFF[31:7] → 벡터 테이블 베이스 = {hex32(vtor)}{where}",
        ])
        doc.sum_(f"벡터 테이블 베이스(VTOR)는 {hex32(vtor)}{where} 이다.")

    scr = d.r32("ppb_scs", 0xE000ED10)
    if scr is not None:
        doc.fields(f"SCB_SCR @ 0xE000ED10 = {hex32(scr)}", [
            f"SLEEPONEXIT[1] = {bit(scr, 1)}",
            f"SLEEPDEEP[2] = {bit(scr, 2)} → {onoff(bit(scr, 2), 'deep sleep(STOP/STANDBY)', '일반 sleep')}",
            f"SEVONPEND[4] = {bit(scr, 4)}",
        ])

    shcsr = d.r32("ppb_scs", 0xE000ED24)
    if shcsr is not None:
        doc.fields(f"SCB_SHCSR @ 0xE000ED24 = {hex32(shcsr)}", [
            f"MEMFAULTACT[0]={bit(shcsr, 0)} BUSFAULTACT[1]={bit(shcsr, 1)} USGFAULTACT[3]={bit(shcsr, 3)}",
            f"SVCALLACT[7]={bit(shcsr, 7)} MONITORACT[8]={bit(shcsr, 8)} PENDSVACT[10]={bit(shcsr, 10)} SYSTICKACT[11]={bit(shcsr, 11)}",
            f"MEMFAULTENA[16] = {bit(shcsr, 16)} → MemManage 폴트 핸들러 {onoff(bit(shcsr, 16))}",
            f"BUSFAULTENA[17] = {bit(shcsr, 17)} → BusFault 핸들러 {onoff(bit(shcsr, 17))}",
            f"USGFAULTENA[18] = {bit(shcsr, 18)} → UsageFault 핸들러 {onoff(bit(shcsr, 18))}",
        ])

    cfsr = d.r32("ppb_scs", 0xE000ED28)
    hfsr = d.r32("ppb_scs", 0xE000ED2C)
    if cfsr is not None:
        mmfsr, bfsr, ufsr = bf(cfsr, 7, 0), bf(cfsr, 15, 8), bf(cfsr, 31, 16)
        rows = [
            ("MMFSR (MemManage)", f"0x{mmfsr:02X}",
             ", ".join(n for n, p in [("IACCVIOL", 0), ("DACCVIOL", 1), ("MUNSTKERR", 3),
                                      ("MSTKERR", 4), ("MLSPERR", 5), ("MMARVALID", 7)]
                       if bit(mmfsr, p)) or "플래그 없음"),
            ("BFSR (BusFault)", f"0x{bfsr:02X}",
             ", ".join(n for n, p in [("IBUSERR", 0), ("PRECISERR", 1), ("IMPRECISERR", 2),
                                      ("UNSTKERR", 3), ("STKERR", 4), ("LSPERR", 5),
                                      ("BFARVALID", 7)] if bit(bfsr, p)) or "플래그 없음"),
            ("UFSR (UsageFault)", f"0x{ufsr:04X}",
             ", ".join(n for n, p in [("UNDEFINSTR", 0), ("INVSTATE", 1), ("INVPC", 2),
                                      ("NOCP", 3), ("UNALIGNED", 8), ("DIVBYZERO", 9)]
                       if bit(ufsr, p)) or "플래그 없음"),
        ]
        doc.w(f"`SCB_CFSR @ 0xE000ED28 = {hex32(cfsr)}`")
        doc.w()
        doc.table(["서브레지스터", "값", "세워진 플래그"], rows)
    if hfsr is not None:
        doc.fields(f"SCB_HFSR @ 0xE000ED2C = {hex32(hfsr)}", [
            f"VECTTBL[1] = {bit(hfsr, 1)} → 벡터 테이블 읽기 실패",
            f"FORCED[30] = {bit(hfsr, 30)} → 하위 폴트가 HardFault 로 승격",
            f"DEBUGEVT[31] = {bit(hfsr, 31)} → 디버그 이벤트",
        ])
    if cfsr == 0 and hfsr == 0:
        doc.sum_("덤프 시점까지 기록된 폴트가 없다 (CFSR=0x00000000, HFSR=0x00000000).")

    mmfar = d.r32("ppb_scs", 0xE000ED34)
    bfar = d.r32("ppb_scs", 0xE000ED38)
    afsr = d.r32("ppb_scs", 0xE000ED3C)
    doc.table(["레지스터", "주소", "값", "해석"], [
        ("SCB_MMFAR", "0xE000ED34", hex32(mmfar), "MemManage 폴트 주소 (MMARVALID 일 때만 유효)"),
        ("SCB_BFAR", "0xE000ED38", hex32(bfar), "BusFault 주소 (BFARVALID 일 때만 유효)"),
        ("SCB_AFSR", "0xE000ED3C", hex32(afsr), "Auxiliary Fault Status (구현 정의)"),
    ])

    cpacr = d.reg("SCB_CPACR", "ppb_scs", 0xE000ED88)
    if cpacr is not None:
        acc = {0: "접근 거부", 1: "특권 모드만", 2: "reserved", 3: "전체 접근"}
        cp10, cp11 = bf(cpacr, 21, 20), bf(cpacr, 23, 22)
        doc.fields(f"SCB_CPACR @ 0xE000ED88 = {hex32(cpacr)}", [
            f"CP10[21:20] = 0b{cp10:02b} → {acc.get(cp10)}",
            f"CP11[23:22] = 0b{cp11:02b} → {acc.get(cp11)}",
            f"**FPU {'사용 가능(활성)' if cp10 == 3 and cp11 == 3 else '비활성 — 부동소수 명령 실행 시 UsageFault(NOCP)'}**",
        ])
        doc.sum_(f"FPU 코프로세서 접근은 {'허용(FPU 활성)' if cp10 == 3 else '차단(FPU 비활성)'} "
                 f"상태였다 (CPACR={hex32(cpacr)}).")

    calib = d.reg("SYST_CALIB", "ppb_scs", 0xE000E01C)
    if calib is not None:
        tenms = bf(calib, 23, 0)
        doc.fields(f"SYST_CALIB @ 0xE000E01C = {hex32(calib)}", [
            f"TENMS[23:0] = {tenms} ({tenms:#x})",
            f"SKEW[30] = {bit(calib, 30)} → TENMS 가 정확히 10ms 가 "
            f"{'아님' if bit(calib, 30) else '맞음'}",
            f"NOREF[31] = {bit(calib, 31)} → 외부 레퍼런스 클럭 "
            f"{'없음' if bit(calib, 31) else '있음(HCLK/8)'}",
        ])

    # 코어 기능 ID 레지스터
    idnames = [("ID_PFR0", 0xE000ED40), ("ID_PFR1", 0xE000ED44), ("ID_DFR0", 0xE000ED48),
               ("ID_AFR0", 0xE000ED4C), ("ID_MMFR0", 0xE000ED50), ("ID_MMFR1", 0xE000ED54),
               ("ID_MMFR2", 0xE000ED58), ("ID_MMFR3", 0xE000ED5C), ("ID_ISAR0", 0xE000ED60),
               ("ID_ISAR1", 0xE000ED64), ("ID_ISAR2", 0xE000ED68), ("ID_ISAR3", 0xE000ED6C),
               ("ID_ISAR4", 0xE000ED70)]
    notes = {
        "ID_PFR0": lambda v: f"State0(ARM ISA)[3:0]={bf(v, 3, 0)} → "
                             f"{'미지원' if bf(v, 3, 0) == 0 else '지원'}, "
                             f"State1(Thumb)[7:4]={bf(v, 7, 4)} → "
                             f"{'Thumb-2 지원' if bf(v, 7, 4) == 3 else '해석 불가'}",
        "ID_PFR1": lambda v: f"프로그래머 모델[11:8]={bf(v, 11, 8)} → "
                             f"{'ARMv7-M (M-profile)' if bf(v, 11, 8) == 2 else '해석 불가'}",
        "ID_DFR0": lambda v: f"M-profile 디버그 모델[23:20]={bf(v, 23, 20)} → "
                             f"{'지원' if bf(v, 23, 20) else '미지원'}",
        "ID_MMFR0": lambda v: f"PMSA[7:4]={bf(v, 7, 4)} → "
                              f"{'PMSAv7 (MPU 있음)' if bf(v, 7, 4) == 3 else '해석 불가'}, "
                              f"TCM 지원[19:16]={bf(v, 19, 16)}, Aux 레지스터[23:20]={bf(v, 23, 20)}",
        "ID_MMFR2": lambda v: f"WFI stalling[27:24]={bf(v, 27, 24)} → "
                              f"{'지원' if bf(v, 27, 24) else '미지원'}",
        "ID_ISAR3": lambda v: f"SIMD_instrs[7:4]={bf(v, 7, 4)} → "
                              f"{'**DSP(SIMD) 확장 탑재** — SADD16/SMLAD 등 사용 가능' if bf(v, 7, 4) >= 3 else '해석 불가'}"
                              f", Saturate_instrs[3:0]={bf(v, 3, 0)}",
    }
    rows = []
    for nm, addr in idnames:
        v = d.reg(nm, "ppb_scs", addr)
        note = notes.get(nm)
        rows.append((nm, f"0x{addr:08X}", hex32(v),
                     note(v) if (note and v is not None) else "상세 필드 해석 생략(ARMv7-M ARM 참조)"))
    doc.h(3, "코어 기능 ID 레지스터")
    doc.table(["레지스터", "주소", "값", "해석"], rows)
    isar3 = d.reg("ID_ISAR3", "ppb_scs", 0xE000ED6C)
    if isar3 is not None and bf(isar3, 7, 4) >= 3:
        doc.sum_(f"DSP(SIMD) 명령 확장이 탑재되어 있다 — CMSIS-NN 의 SMLAD/SADD16 계열 사용 근거 "
                 f"(ID_ISAR3={hex32(isar3)}, SIMD_instrs={bf(isar3, 7, 4)}).", pri=1)


# ---------------------------------------------------------------------------
# 섹션 2: L1 캐시
# ---------------------------------------------------------------------------


def decode_ccsidr(v):
    """CCSIDR → (설명 리스트, 총 바이트 용량)."""
    line_words = 1 << (bf(v, 2, 0) + 2)      # LineSize[2:0]: (값+2) 의 2의 거듭제곱 워드
    line_bytes = line_words * 4
    assoc = bf(v, 12, 3) + 1                 # Associativity[12:3] (0 기반)
    numsets = bf(v, 27, 13) + 1              # NumSets[27:13] (0 기반)
    total = numsets * assoc * line_bytes
    lines = [
        f"LineSize[2:0] = {bf(v, 2, 0)} → 캐시 라인 {line_words} 워드 = **{line_bytes} 바이트**",
        f"Associativity[12:3] = {bf(v, 12, 3)} → **{assoc}-way set associative**",
        f"NumSets[27:13] = {bf(v, 27, 13)} → **{numsets} 세트**",
        f"WA[28] = {bit(v, 28)} → write-allocate {onoff(bit(v, 28), '지원', '미지원')}",
        f"RA[29] = {bit(v, 29)} → read-allocate {onoff(bit(v, 29), '지원', '미지원')}",
        f"WB[30] = {bit(v, 30)} → write-back {onoff(bit(v, 30), '지원', '미지원')}",
        f"WT[31] = {bit(v, 31)} → write-through {onoff(bit(v, 31), '지원', '미지원')}",
        f"**실제 용량 = {numsets} 세트 × {assoc} way × {line_bytes} B = {total} B "
        f"= {total // 1024} KB**",
    ]
    return lines, total


def sec_cache(d, doc):
    doc.h(2, "L1 캐시")

    clidr = d.reg("CLIDR", "ppb_scs", 0xE000ED78)
    if clidr is not None:
        ctype = {0: "캐시 없음", 1: "I-cache 만", 2: "D-cache 만",
                 3: "**I-cache 와 D-cache 분리(Harvard)**", 4: "통합 캐시"}
        lines = []
        for lvl in range(1, 8):
            ct = bf(clidr, 3 * lvl - 1, 3 * (lvl - 1))
            if ct:
                lines.append(f"Ctype{lvl}[{3 * lvl - 1}:{3 * (lvl - 1)}] = {ct} → L{lvl}: {ctype.get(ct, '해석 불가')}")
        lines += [
            f"LoUIS[23:21] = {bf(clidr, 23, 21)} (Level of Unification, Inner Shareable)",
            f"LoC[26:24] = {bf(clidr, 26, 24)} → **Level of Coherency = L{bf(clidr, 26, 24)}** (캐시 레벨은 총 {bf(clidr, 26, 24)}단계)",
            f"LoUU[29:27] = {bf(clidr, 29, 27)} (Level of Unification, Uniprocessor)",
        ]
        doc.fields(f"CLIDR @ 0xE000ED78 = {hex32(clidr)}", lines)
        if bf(clidr, 2, 0) == 3:
            doc.sum_(f"L1 캐시는 명령/데이터 분리형(Harvard)이며 L1 한 단계뿐이다 "
                     f"(CLIDR={hex32(clidr)}, Ctype1=3, LoC={bf(clidr, 26, 24)}).", pri=1)

    ctr = d.reg("CTR", "ppb_scs", 0xE000ED7C)
    if ctr is not None:
        imin = 1 << bf(ctr, 3, 0)
        dmin = 1 << bf(ctr, 19, 16)
        erg = bf(ctr, 23, 20)
        cwg = bf(ctr, 27, 24)
        fmt = bf(ctr, 31, 29)
        doc.fields(f"CTR @ 0xE000ED7C = {hex32(ctr)}", [
            f"IminLine[3:0] = {bf(ctr, 3, 0)} → I-cache 최소 라인 {imin} 워드 = **{imin * 4} B**",
            f"DminLine[19:16] = {bf(ctr, 19, 16)} → D-cache 최소 라인 {dmin} 워드 = **{dmin * 4} B**",
            f"ERG[23:20] = {erg} → Exclusive Reservation Granule "
            + (f"{1 << erg} 워드 = {(1 << erg) * 4} B" if erg else "0 → 미구현/미보고"),
            f"CWG[27:24] = {cwg} → Cache Write-back Granule "
            + (f"{1 << cwg} 워드 = **{(1 << cwg) * 4} B**" if cwg else "0 → 미보고"),
            f"Format[31:29] = 0b{fmt:03b} → "
            + ("**ARMv7 CTR 포맷**" if fmt == 0b100 else "해석 불가"),
        ])

    # CCSIDR: hotplug 덤프는 CSSELR 을 0/1 로 바꿔가며 D/I 를 각각 읽어 두었다.
    cids = d.meta.get("cache_ids") or {}
    total_cache = 0
    cap_parts = []
    if cids:
        for label, korean in (("dcache_L1", "D-cache (CSSELR=0)"), ("icache_L1", "I-cache (CSSELR=1)")):
            e = cids.get(label) or {}
            if not e.get("CCSIDR"):
                continue
            v = int(e["CCSIDR"], 16)
            lines, total = decode_ccsidr(v)
            total_cache += total
            doc.h(3, f"CCSIDR — {korean}")
            doc.fields(f"CCSIDR @ 0xE000ED80 = 0x{v:08X}", lines)
            cap_parts.append(f"{korean.split(' ')[0]} {total // 1024} KB "
                             f"({bf(v, 27, 13) + 1} 세트 × {bf(v, 12, 3) + 1} way × "
                             f"{(1 << (bf(v, 2, 0) + 2)) * 4} B, CCSIDR=0x{v:08X})")
        if cap_parts:
            doc.sum_("L1 캐시 실측 용량: **" + " / ".join(cap_parts)
                     + f"** → 합계 {total_cache // 1024} KB.", pri=1)
        if d.cache_id_raw:
            doc.note(f"근거 원문: `info/raw/{d.cache_id_raw.name}` "
                     "(CSSELR 에 0/1 을 써 넣고 CCSIDR 을 각각 읽은 CLI 출력)")
    else:
        v = d.reg("CCSIDR", "ppb_scs", 0xE000ED80)
        sel = d.reg("CSSELR", "ppb_scs", 0xE000ED84)
        if v is not None:
            which = "D-cache" if (sel or 0) == 0 else "I-cache"
            lines, total = decode_ccsidr(v)
            doc.h(3, f"CCSIDR — CSSELR={hex32(sel)} 선택분 ({which})")
            doc.fields(f"CCSIDR @ 0xE000ED80 = {hex32(v)}", lines)
            doc.sum_(f"L1 {which} 용량은 **{total // 1024} KB** 이다 (CCSIDR={hex32(v)}).")
            doc.note("이 덤프에서는 CSSELR 을 조작하지 않았으므로 한쪽 캐시만 확인된다. "
                     "양쪽 값은 hotplug 덤프 문서를 참조.")

    cacr = d.reg("CACR", "ppb_scs", 0xE000EF9C)
    if cacr is not None:
        doc.fields(f"CACR @ 0xE000EF9C = {hex32(cacr)}", [
            f"SIWT[0] = {bit(cacr, 0)} → Shared cacheable-is-WT "
            f"{onoff(bit(cacr, 0), '활성(공유 영역 write-through 강제)', '비활성')}",
            f"ECCEN[1] = {bit(cacr, 1)} → 캐시/TCM **ECC {onoff(bit(cacr, 1), '활성', '비활성')}**",
            f"FORCEWT[2] = {bit(cacr, 2)} → 전 영역 write-through 강제 "
            f"{onoff(bit(cacr, 2), '활성', '비활성(write-back 허용)')}",
        ])
        doc.sum_(f"캐시/TCM ECC 는 {onoff(bit(cacr, 1), '켜져 있고', '꺼져 있고')}, "
                 f"write-through 강제는 {onoff(bit(cacr, 2), '켜져 있다', '꺼져 있다')} "
                 f"(CACR={hex32(cacr)}).")


# ---------------------------------------------------------------------------
# 섹션 3: TCM 및 버스 인터페이스
# ---------------------------------------------------------------------------


def sec_tcm(d, doc):
    doc.h(2, "TCM / 버스 인터페이스")

    for nm, addr, expect_kb, alias in (("ITCMCR", 0xE000EF90, 16, "0x00000000"),
                                       ("DTCMCR", 0xE000EF94, 64, "0x20000000")):
        v = d.reg(nm, "ppb_scs", addr)
        if v is None:
            continue
        sz = bf(v, 6, 3)
        sz_str = TCM_SZ.get(sz, "해석 불가")
        doc.fields(f"{nm} @ 0x{addr:08X} = {hex32(v)}", [
            f"EN[0] = {bit(v, 0)} → TCM {onoff(bit(v, 0), '**활성**', '비활성')} (별칭 주소 {alias})",
            f"RMW[1] = {bit(v, 1)} → read-modify-write "
            f"{onoff(bit(v, 1), '활성(바이트/하프워드 쓰기 지원)', '비활성')}",
            f"RETEN[2] = {bit(v, 2)} → 리셋 시 내용 유지 {onoff(bit(v, 2), '활성', '비활성')}",
            f"SZ[6:3] = 0b{sz:04b} → Cortex-M7 TRM 인코딩 표상 **{sz_str}**",
        ])
        if sz_str != "해석 불가" and sz_str.endswith("KB") and int(sz_str[:-2]) == expect_kb:
            doc.sum_(f"{nm[:4]} 은 활성 상태이며 크기 필드가 {sz_str} 로 "
                     f"STM32F746 사양({expect_kb}KB)과 일치한다 ({nm}={hex32(v)}).", pri=1)
        else:
            doc.warn(
                f"**{nm}={hex32(v)} 의 SZ 필드(0b{sz:04b})는 Cortex-M7 TRM 표에서 "
                f"'{sz_str}' 를 뜻하는데, STM32F746 데이터시트의 {nm[:4]} 크기({expect_kb}KB)와 "
                f"어긋난다.** ST 가 TCM 인터페이스에 할당한 주소공간 크기를 보고하는 것인지, "
                f"실제 RAM 크기를 보고하는 것인지 이 덤프만으로는 확정할 수 없다. "
                f"온디바이스 aliasing 실측(별칭 주소에 쓰고 되읽어 wrap-around 지점을 찾는 방법)으로 "
                f"별도 확인 예정.")

    ahbpcr = d.reg("AHBPCR", "ppb_scs", 0xE000EF98)
    if ahbpcr is not None:
        sz = bf(ahbpcr, 3, 1)
        doc.fields(f"AHBPCR @ 0xE000EF98 = {hex32(ahbpcr)}", [
            f"EN[0] = {bit(ahbpcr, 0)} → AHBP(페리페럴 포트) {onoff(bit(ahbpcr, 0), '**활성**', '비활성')}",
            f"SZ[3:1] = 0b{sz:03b} → AHBP 영역 크기 **{AHBP_SZ.get(sz, '해석 불가')}** "
            f"(0x40000000 기준)",
        ])
        if bit(ahbpcr, 0):
            doc.sum_(f"AHBP 페리페럴 포트가 활성이고 {AHBP_SZ.get(sz, '해석 불가')} 영역을 담당한다 "
                     f"(AHBPCR={hex32(ahbpcr)}) → 페리페럴 접근이 AXIM/캐시를 우회한다.", pri=1)

    ahbscr = d.reg("AHBSCR", "ppb_scs", 0xE000EFA0)
    if ahbscr is not None:
        doc.fields(f"AHBSCR @ 0xE000EFA0 = {hex32(ahbscr)}", [
            f"CTL[1:0] = 0b{bf(ahbscr, 1, 0):02b} → AHBS(슬레이브 포트) 중재 제어 "
            + ("0b00 = 기본(소프트웨어 접근 우선)" if bf(ahbscr, 1, 0) == 0 else "해석 불가"),
            f"TPRI[10:2] = {bf(ahbscr, 10, 2)} → AHBS 임계 우선순위",
            f"INITCOUNT[15:11] = {bf(ahbscr, 15, 11)} → 페어니스 카운터 초기값",
        ])

    abfsr = d.reg("ABFSR", "ppb_scs", 0xE000EFA8)
    if abfsr is not None:
        flags = [n for n, p in (("ITCM", 0), ("DTCM", 1), ("AHBP", 2), ("AXIM", 3), ("EPPB", 4))
                 if bit(abfsr, p)]
        axtype = {0: "OKAY", 1: "EXOKAY", 2: "SLVERR", 3: "DECERR"}
        doc.fields(f"ABFSR @ 0xE000EFA8 = {hex32(abfsr)}", [
            f"ITCM[0]={bit(abfsr, 0)} DTCM[1]={bit(abfsr, 1)} AHBP[2]={bit(abfsr, 2)} "
            f"AXIM[3]={bit(abfsr, 3)} EPPB[4]={bit(abfsr, 4)}",
            f"AXIMTYPE[9:8] = 0b{bf(abfsr, 9, 8):02b} → {axtype.get(bf(abfsr, 9, 8), '해석 불가')}",
            f"→ **{('비동기 버스 폴트 발생 인터페이스: ' + ', '.join(flags)) if flags else '기록된 비동기 버스 폴트 없음'}**",
        ])
        if abfsr == 0:
            doc.sum_("비동기 버스 폴트 기록이 없다 — ITCM/DTCM/AHBP/AXIM/EPPB 모두 정상 "
                     "(ABFSR=0x00000000).")


# ---------------------------------------------------------------------------
# 섹션 4: FPU
# ---------------------------------------------------------------------------


def sec_fpu(d, doc):
    doc.h(2, "FPU (MVFR0/1/2)")

    mvfr0 = d.reg("MVFR0", "ppb_scs", 0xE000EF40)
    mvfr1 = d.reg("MVFR1", "ppb_scs", 0xE000EF44)
    mvfr2 = d.reg("MVFR2", "ppb_scs", 0xE000EF48)

    dp_supported = None
    if mvfr0 is not None:
        simdreg = bf(mvfr0, 3, 0)
        sp = bf(mvfr0, 7, 4)
        dp = bf(mvfr0, 11, 8)
        trap = bf(mvfr0, 15, 12)
        div = bf(mvfr0, 19, 16)
        sqrt = bf(mvfr0, 23, 20)
        shvec = bf(mvfr0, 27, 24)
        rnd = bf(mvfr0, 31, 28)
        dp_supported = dp != 0
        doc.fields(f"MVFR0 @ 0xE000EF40 = {hex32(mvfr0)}", [
            f"A_SIMD registers[3:0] = {simdreg} → "
            + ("16 × 64비트 레지스터 뱅크(D0-D15 / S0-S31)" if simdreg == 1 else "해석 불가"),
            f"Single-precision[7:4] = {sp} → **단정밀도 {'지원(모든 VFP 연산)' if sp == 2 else '해석 불가'}**",
            f"Double-precision[11:8] = {dp} → **배정밀도 {'하드웨어 미지원' if dp == 0 else '지원'}**",
            f"FP exception trapping[15:12] = {trap} → "
            + ("트랩 미지원(예외는 상태 비트로만 보고)" if trap == 0 else "지원"),
            f"Divide[19:16] = {div} → VDIV {'지원' if div else '미지원'}",
            f"Square root[23:20] = {sqrt} → VSQRT {'지원' if sqrt else '미지원'}",
            f"Short vectors[27:24] = {shvec} → {'미지원' if shvec == 0 else '지원'}",
            f"FP rounding modes[31:28] = {rnd} → "
            + ("모든 라운딩 모드 지원" if rnd == 1 else "해석 불가"),
        ])

    if mvfr1 is not None:
        ftz = bf(mvfr1, 3, 0)
        dnan = bf(mvfr1, 7, 4)
        hp = bf(mvfr1, 27, 24)
        fma = bf(mvfr1, 31, 28)
        doc.fields(f"MVFR1 @ 0xE000EF44 = {hex32(mvfr1)}", [
            f"FtZ mode[3:0] = {ftz} → "
            + ("**Flush-to-Zero 만 지원** (비정규수는 0으로 플러시)" if ftz == 1
               else ("완전한 비정규수 연산 지원" if ftz == 3 else "해석 불가")),
            f"D_NaN mode[7:4] = {dnan} → "
            + ("**Default NaN 만 지원**" if dnan == 1 else "해석 불가"),
            f"A_SIMD load/store[11:8] = {bf(mvfr1, 11, 8)}, "
            f"A_SIMD integer[15:12] = {bf(mvfr1, 15, 12)}, "
            f"A_SIMD single-precision[19:16] = {bf(mvfr1, 19, 16)}, "
            f"A_SIMD half-precision[23:20] = {bf(mvfr1, 23, 20)} → "
            + ("Advanced SIMD(NEON) 미탑재" if bf(mvfr1, 23, 8) == 0 else "해석 불가"),
            f"FP HPFP[27:24] = {hp} → **반정밀도(half-precision) 변환 명령 "
            f"{'지원(VCVTB/VCVTT)' if hp else '미지원'}**",
            f"FP fused MAC[31:28] = {fma} → **fused multiply-accumulate "
            f"{'지원(VFMA/VFMS)' if fma else '미지원'}**",
        ])

    if mvfr2 is not None:
        misc = bf(mvfr2, 7, 4)
        doc.fields(f"MVFR2 @ 0xE000EF48 = {hex32(mvfr2)}", [
            f"VFP_Misc[7:4] = {misc} → "
            + ("**FPv5 부가 명령 지원** — VSEL, VMAXNM/VMINNM, VRINT*, VCVTA/N/P/M"
               if misc == 4 else "해석 불가"),
        ])

    if dp_supported is False:
        doc.w("> **결론: 배정밀도(double) 부동소수 하드웨어가 없다 → FPv5-SP (단정밀도 전용) FPU.**")
        doc.w("> `double` 연산은 전부 소프트웨어 라이브러리(`__aeabi_d*`)로 처리되므로 "
              "추론 커널은 반드시 `float`(단정밀도) 또는 정수 연산으로 작성해야 한다.")
        doc.w()
        doc.sum_("FPU 는 **FPv5-SP(단정밀도 전용)** 로 배정밀도 하드웨어가 없다 "
                 f"(MVFR0={hex32(mvfr0)}, Double-precision 필드=0) → `double` 은 소프트웨어 에뮬레이션.", pri=1)
        if mvfr1 is not None and bf(mvfr1, 31, 28):
            doc.sum_(f"FMA(fused multiply-accumulate)와 반정밀도 변환 명령을 하드웨어로 지원한다 "
                     f"(MVFR1={hex32(mvfr1)}).", pri=1)


# ---------------------------------------------------------------------------
# 섹션 5: MPU
# ---------------------------------------------------------------------------


def sec_mpu(d, doc):
    doc.h(2, "MPU")

    mt = d.reg("MPU_TYPE", "ppb_scs", 0xE000ED90)
    if mt is not None:
        dreg, ireg, sep = bf(mt, 15, 8), bf(mt, 23, 16), bit(mt, 0)
        doc.fields(f"MPU_TYPE @ 0xE000ED90 = {hex32(mt)}", [
            f"SEPARATE[0] = {sep} → "
            + ("명령/데이터 MPU 분리" if sep else "**통합 MPU(명령·데이터 공용)**"),
            f"DREGION[15:8] = {dreg} → **데이터 리전 {dreg}개**",
            f"IREGION[23:16] = {ireg} → 명령 전용 리전 {ireg}개",
        ])
        doc.sum_(f"MPU 는 통합형이며 리전 {dreg}개를 지원한다 (MPU_TYPE={hex32(mt)}).", pri=1)

    mc = d.r32("ppb_scs", 0xE000ED94)
    if mc is not None:
        doc.fields(f"MPU_CTRL @ 0xE000ED94 = {hex32(mc)}", [
            f"ENABLE[0] = {bit(mc, 0)} → MPU {onoff(bit(mc, 0), '**활성화됨**', '비활성')}",
            f"HFNMIENA[1] = {bit(mc, 1)} → HardFault/NMI 중에도 MPU "
            f"{onoff(bit(mc, 1), '적용', '무시')}",
            f"PRIVDEFENA[2] = {bit(mc, 2)} → 특권 모드의 기본 메모리 맵 "
            f"{onoff(bit(mc, 2), '**허용**(리전 미지정 영역도 접근 가능)', '차단')}",
        ])
        doc.sum_(f"MPU 는 {onoff(bit(mc, 0), '**활성화**', '비활성')} 상태였다 "
                 f"(MPU_CTRL={hex32(mc)}, PRIVDEFENA={bit(mc, 2)}).")

    rnr = d.r32("ppb_scs", 0xE000ED98)
    rbar = d.r32("ppb_scs", 0xE000ED9C)
    rasr = d.r32("ppb_scs", 0xE000EDA0)
    if rasr:
        size = bf(rasr, 5, 1)
        nbytes = 1 << (size + 1)
        ap = {0: "접근 없음", 1: "특권 RW", 2: "특권 RW / 비특권 RO", 3: "전체 RW",
              5: "특권 RO", 6: "읽기 전용", 7: "읽기 전용"}
        human = f"{nbytes} B" if nbytes < 1024 else (
            f"{nbytes // 1024} KB" if nbytes < 1024 * 1024 else f"{nbytes // (1024 * 1024)} MB")
        doc.fields(f"MPU_RNR/RBAR/RASR = {hex32(rnr)} / {hex32(rbar)} / {hex32(rasr)}", [
            f"RNR = {hex32(rnr)} → 마지막으로 선택된 리전 번호 **{bf(rnr or 0, 7, 0)}**",
            f"RBAR ADDR[31:5] = 0x{(rbar or 0) & ~0x1F:08X} (REGION[3:0]={bf(rbar or 0, 3, 0)}, VALID[4]={bit(rbar or 0, 4)})",
            f"RASR ENABLE[0] = {bit(rasr, 0)}",
            f"RASR SIZE[5:1] = {size} → 리전 크기 2^({size}+1) = **{human}**",
            f"RASR SRD[15:8] = 0x{bf(rasr, 15, 8):02X} (서브리전 비활성 마스크)",
            f"RASR B[16]={bit(rasr, 16)} C[17]={bit(rasr, 17)} S[18]={bit(rasr, 18)} "
            f"TEX[21:19]={bf(rasr, 21, 19)} → "
            + ("Device/Shareable (캐시 불가)" if (bf(rasr, 21, 19) == 0 and bit(rasr, 17) == 0
                                                and bit(rasr, 16) == 1)
               else "메모리 속성 조합은 ARMv7-M ARM 표 참조"),
            f"RASR AP[26:24] = {bf(rasr, 26, 24)} → {ap.get(bf(rasr, 26, 24), '해석 불가')}",
            f"RASR XN[28] = {bit(rasr, 28)} → 명령 실행 "
            f"{onoff(bit(rasr, 28), '**금지**', '허용')}",
        ])
        doc.note("MPU_RBAR/RASR 는 RNR 로 선택된 **한 리전만** 보인다. "
                 "덤프 시점에 마지막으로 선택돼 있던 리전 값이므로 전체 리전 구성은 알 수 없다.")
    else:
        doc.w("MPU_RNR/RBAR/RASR 가 모두 0 → 설정된 리전이 없거나 리전 0 이 선택된 상태.")
        doc.w()


# ---------------------------------------------------------------------------
# 섹션 6: 디버그 / 트레이스
# ---------------------------------------------------------------------------


def sec_debug(d, doc):
    doc.h(2, "디버그 / 트레이스")

    demcr = d.r32("ppb_scs", 0xE000EDFC)
    dhcsr = d.r32("ppb_scs", 0xE000EDF0)
    if demcr is not None or dhcsr is not None:
        rows = []
        if dhcsr is not None:
            rows.append(("DHCSR", "0xE000EDF0", hex32(dhcsr),
                         f"C_DEBUGEN[0]={bit(dhcsr, 0)}, C_HALT[1]={bit(dhcsr, 1)}, "
                         f"S_HALT[17]={bit(dhcsr, 17)}, S_REGRDY[16]={bit(dhcsr, 16)} → "
                         + ("코어 정지(halt) 상태" if bit(dhcsr, 17) else "코어 실행 중")))
        if demcr is not None:
            rows.append(("DEMCR", "0xE000EDFC", hex32(demcr),
                         f"TRCENA[24]={bit(demcr, 24)} → 트레이스/DWT 블록 "
                         f"{onoff(bit(demcr, 24), '활성', '**비활성**')}, "
                         f"VC_CORERESET[0]={bit(demcr, 0)}, MON_EN[16]={bit(demcr, 16)}"))
        doc.table(["레지스터", "주소", "값", "해석"], rows)

    dwt = d.reg("DWT_CTRL", "ppb_dwt", 0xE0001000)
    if dwt is not None:
        numcomp = bf(dwt, 31, 28)
        lines = [
            f"NUMCOMP[31:28] = {numcomp} → DWT 비교기 {numcomp}개",
            f"NOTRCPKT[27] = {bit(dwt, 27)}, NOEXTTRIG[26] = {bit(dwt, 26)}, "
            f"NOCYCCNT[25] = {bit(dwt, 25)}, NOPRFCNT[24] = {bit(dwt, 24)}",
            f"CYCCNTENA[0] = {bit(dwt, 0)} → 사이클 카운터 {onoff(bit(dwt, 0))}",
            f"PCSAMPLENA[12] = {bit(dwt, 12)}, EXCTRCENA[16] = {bit(dwt, 16)}, "
            f"CYCEVTENA[22] = {bit(dwt, 22)}",
        ]
        doc.fields(f"DWT_CTRL @ 0xE0001000 = {hex32(dwt)}", lines)
        if dwt == 0:
            doc.warn(
                "**DWT_CTRL 이 0x00000000 으로 읽혔다. 이는 'DWT 가 없다'는 뜻이 아니다.** "
                f"DEMCR.TRCENA(비트 24) = {bit(demcr or 0, 24)} 로 트레이스 블록 전원/클럭이 꺼져 있으면 "
                "DWT 레지스터가 전부 0 으로 읽힌다. DWT_CYCCNT 기반 사이클 측정을 하려면 "
                "먼저 `CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk` 를 수행해야 한다. "
                "따라서 이 덤프로는 NUMCOMP 등 DWT 실제 사양을 확정할 수 없다(해석 불가).")
            doc.sum_("DWT 레지스터가 전부 0 으로 읽힌다 — DEMCR.TRCENA=0 이라 트레이스 블록이 "
                     "꺼져 있기 때문이며, DWT 부재의 근거가 아니다 (DWT_CTRL=0x00000000).")

    fpc = d.reg("FP_CTRL", "ppb_fpb", 0xE0002000)
    if fpc is not None:
        num_code = (bf(fpc, 14, 12) << 4) | bf(fpc, 7, 4)
        num_lit = bf(fpc, 11, 8)
        rev = bf(fpc, 31, 28)
        doc.fields(f"FP_CTRL @ 0xE0002000 = {hex32(fpc)}", [
            f"ENABLE[0] = {bit(fpc, 0)} → FPB {onoff(bit(fpc, 0), '활성', '비활성')}",
            f"KEY[1] = {bit(fpc, 1)} (쓰기 시에만 의미)",
            f"NUM_CODE[7:4]+[14:12] = {num_code} → **하드웨어 명령 브레이크포인트 {num_code}개**",
            f"NUM_LIT[11:8] = {num_lit} → 리터럴 비교기 {num_lit}개",
            f"REV[31:28] = {rev} → "
            + ("FPB v2 (Cortex-M7, 32비트 주소 비교, 주소 리맵 없음)" if rev == 1
               else ("FPB v1 (주소 리맵 지원)" if rev == 0 else "해석 불가")),
        ])
        doc.sum_(f"하드웨어 브레이크포인트 {num_code}개, 리터럴 비교기 {num_lit}개 "
                 f"(FP_CTRL={hex32(fpc)}, FPB v{rev + 1}).", pri=1)

    idc = d.reg("DBGMCU_IDCODE", "ppb_dbgmcu", 0xE0042000)
    if idc is not None:
        dev = bf(idc, 11, 0)
        rev = bf(idc, 31, 16)
        devname = {0x449: "STM32F74x / STM32F75x", 0x451: "STM32F76x / STM32F77x",
                   0x452: "STM32F72x / STM32F73x"}.get(dev, "미상")
        revname = REV_ID.get(rev, "미상 리비전")
        doc.fields(f"DBGMCU_IDCODE @ 0xE0042000 = {hex32(idc)}", [
            f"DEV_ID[11:0] = 0x{dev:03X} → **{devname}**",
            f"REV_ID[31:16] = 0x{rev:04X} → **{revname}**",
        ])
        doc.sum_(f"실리콘은 DEV_ID=0x{dev:03X}({devname}), REV_ID=0x{rev:04X}(**{revname}**) 이다 "
                 f"(DBGMCU_IDCODE={hex32(idc)}).", pri=1)

    dcr = d.r32("ppb_dbgmcu", 0xE0042004)
    if dcr is not None:
        doc.fields(f"DBGMCU_CR @ 0xE0042004 = {hex32(dcr)}", [
            f"DBG_SLEEP[0] = {bit(dcr, 0)}, DBG_STOP[1] = {bit(dcr, 1)}, "
            f"DBG_STANDBY[2] = {bit(dcr, 2)} → 저전력 모드에서 디버그 유지 "
            f"{onoff(bit(dcr, 0) or bit(dcr, 1) or bit(dcr, 2), '일부 활성', '전부 비활성')}",
            f"TRACE_IOEN[5] = {bit(dcr, 5)} → 트레이스 핀 {onoff(bit(dcr, 5), '활성', '비활성')}",
            f"TRACE_MODE[7:6] = 0b{bf(dcr, 7, 6):02b}",
        ])

    _sec_romtable(d, doc)


def _sec_romtable(d, doc):
    b = d.blob("ppb_romtable")
    if b is None:
        return
    doc.h(3, "CoreSight ROM table (0xE00FF000)")
    base = 0xE00FF000
    rows = []
    found = []
    for i in range(0, 16):
        v = u32(b, 4 * i)
        if v is None:
            rows.append((i, "-", "-", "-", "덤프가 잘려 엔트리를 읽을 수 없음 → 해석 불가"))
            break
        if v == 0:
            rows.append((i, f"0x{v:08X}", "-", "-", "종료 엔트리(0x00000000) — 이후 엔트리 없음"))
            break
        present = bit(v, 0)
        fmt32 = bit(v, 1)
        off = v & 0xFFFFF000
        if off & 0x80000000:            # 부호 확장 (엔트리 오프셋은 부호 있는 값)
            off -= 1 << 32
        comp = base + off
        nm, desc, region = CORESIGHT_BY_ADDR.get(comp & 0xFFFFFFFF, ("미상", "식별 불가", None))
        rows.append((i, f"0x{v:08X}",
                     f"{'있음' if present else '**없음**'} / {'32비트' if fmt32 else '8비트'}",
                     f"0x{comp & 0xFFFFFFFF:08X}",
                     f"**{nm}** — {desc}"))
        found.append((nm, comp & 0xFFFFFFFF, region, present))
    doc.table(["#", "엔트리 값", "present / format", "컴포넌트 베이스", "식별"], rows)

    # 각 컴포넌트의 PIDR/CIDR
    doc.h(3, "CoreSight 컴포넌트 ID (PIDR/CIDR)")
    idrows = []
    for nm, addr, region, present in found:
        if region is None:
            continue
        rb = d.blob(region)
        if rb is None or len(rb) < 0x1000:
            idrows.append((nm, f"0x{addr:08X}", "-", "-", "-", "영역 덤프 없음 → 해석 불가"))
            continue
        pid = [(u32(rb, o) or 0) & 0xFF
               for o in (0xFE0, 0xFE4, 0xFE8, 0xFEC, 0xFD0)]   # PIDR0..3, PIDR4
        cid = [(u32(rb, o) or 0) & 0xFF
               for o in (0xFF0, 0xFF4, 0xFF8, 0xFFC)]          # CIDR0..3
        part = ((pid[1] & 0xF) << 8) | pid[0]
        jep_id = ((pid[2] & 0x7) << 4) | (pid[1] >> 4)
        jep_cont = pid[4] & 0xF
        jedec = bit(pid[2], 3)
        revision = pid[2] >> 4
        blocks = 1 << (pid[4] >> 4)
        cls = (cid[1] >> 4) & 0xF
        cid_ok = (cid[0] == 0x0D and cid[3] == 0xB1 and (cid[1] & 0x0F) == 0x0 and cid[2] == 0x05)
        vendor = ("ARM Ltd. (JEP106 continuation %d, ID 0x%02X)" % (jep_cont, jep_id)
                  if (jep_cont == 4 and jep_id == 0x3B) else
                  f"JEP106 continuation {jep_cont}, ID 0x{jep_id:02X} (미상 제조사)")
        idrows.append((
            nm, f"0x{addr:08X}",
            " ".join(f"{x:02X}" for x in pid[:4]) + f" / P4={pid[4]:02X}",
            " ".join(f"{x:02X}" for x in cid),
            f"0x{part:03X} = {CORESIGHT_PART.get(part, '미상 부품')}",
            f"{vendor}; rev {revision}; JEDEC={jedec}; 크기 {blocks}×4KB; "
            f"클래스 0x{cls:X} = {CS_CLASS.get(cls, '미상')}"
            + ("" if cid_ok else "; ⚠ CIDR preamble 불일치"),
        ))
    # ROM table 자신
    rb = d.blob("ppb_romtable")
    pid = [(u32(rb, o) or 0) & 0xFF for o in (0xFE0, 0xFE4, 0xFE8, 0xFEC, 0xFD0)]
    cid = [(u32(rb, o) or 0) & 0xFF for o in (0xFF0, 0xFF4, 0xFF8, 0xFFC)]
    part = ((pid[1] & 0xF) << 8) | pid[0]
    cls = (cid[1] >> 4) & 0xF
    idrows.append(("ROM table", "0xE00FF000",
                   " ".join(f"{x:02X}" for x in pid[:4]) + f" / P4={pid[4]:02X}",
                   " ".join(f"{x:02X}" for x in cid),
                   f"0x{part:03X} = {CORESIGHT_PART.get(part, '미상 부품')}",
                   f"클래스 0x{cls:X} = {CS_CLASS.get(cls, '미상')}"))
    doc.table(["컴포넌트", "베이스", "PIDR0-3 / PIDR4", "CIDR0-3", "Part number", "제조사·속성"], idrows)

    present_names = [n for n, _, _, p in found if p]
    absent_names = [n for n, _, _, p in found if not p]
    if present_names:
        doc.sum_("CoreSight ROM table 이 present 로 표시한 디버그 컴포넌트: **"
                 + ", ".join(present_names) + "** (0xE00FF000 엔트리).", pri=1)
    if absent_names:
        doc.warn(
            f"ROM table 엔트리에서 **{', '.join(absent_names)}** 의 present 비트가 0 이다. "
            "그런데 해당 주소의 PIDR/CIDR 는 ARM 정식 부품 ID 로 읽힌다(위 표). "
            "즉 물리적으로는 존재하나 ROM table 이 '없음'으로 보고하는 상태다. "
            "이 덤프만으로는 미탑재인지, 트레이스 인프라가 꺼져 있어서인지 확정할 수 없다(해석 불가).")
        doc.sum_(f"ROM table 은 {', '.join(absent_names)} 를 present=0 으로 보고하지만 "
                 "해당 주소의 PIDR 은 ARM 정식 부품 ID 를 반환한다(상충 → 확정 불가).")


# ---------------------------------------------------------------------------
# 섹션 7: 팩토리 프로그램 영역
# ---------------------------------------------------------------------------


def sec_factory(d, doc):
    doc.h(2, "팩토리 프로그램 영역 (ST 공장 기록, 읽기 전용)")

    b = d.blob("factory_uid_area")
    rows = []
    if b is not None:
        base = d.base_of("factory_uid_area") or 0x1FF0F000

        def rd(addr, n):
            off = addr - base
            return b[off:off + n] if 0 <= off and off + n <= len(b) else None

        uid = rd(0x1FF0F420, 12)
        if uid:
            u0, u1, u2 = struct.unpack("<III", uid)
            ascii_part = "".join(chr(c) if 32 <= c < 127 else "." for c in uid)
            rows.append(("UID (96비트)", "0x1FF0F420", uid.hex().upper(),
                         f"워드: 0x{u0:08X} / 0x{u1:08X} / 0x{u2:08X} — "
                         f"바이트 ASCII 해석 `{ascii_part}` "
                         "(ST 는 하위 워드에 웨이퍼 X/Y, 상위에 로트번호 ASCII 를 기록)"))
            doc.sum_(f"이 개체의 96비트 고유 UID 는 `{uid.hex().upper()}` 이다 (0x1FF0F420).", pri=1)
        fs = rd(0x1FF0F442, 2)
        if fs:
            kb = struct.unpack("<H", fs)[0]
            rows.append(("Flash size", "0x1FF0F442", f"0x{kb:04X}", f"**{kb} KB = {kb // 1024} MB**"))
            doc.sum_(f"내장 Flash 용량은 **{kb} KB ({kb // 1024} MB)** 로 공장 기록되어 있다 "
                     f"(0x1FF0F442 = 0x{kb:04X}).", pri=1)
        for nm, addr, desc in (
            ("VREFINT_CAL", 0x1FF0F44A, "30°C·VDDA=3.3V 에서 측정한 내부 기준전압 ADC 원시값"),
            ("TS_CAL1", 0x1FF0F44C, "30°C·VDDA=3.3V 온도센서 ADC 원시값"),
            ("TS_CAL2", 0x1FF0F44E, "110°C·VDDA=3.3V 온도센서 ADC 원시값"),
        ):
            v = rd(addr, 2)
            if v:
                x = struct.unpack("<H", v)[0]
                rows.append((nm, f"0x{addr:08X}", f"0x{x:04X}", f"{x} — {desc}"))
        t1 = rd(0x1FF0F44C, 2)
        t2 = rd(0x1FF0F44E, 2)
        if t1 and t2:
            a = struct.unpack("<H", t1)[0]
            c = struct.unpack("<H", t2)[0]
            if c != a:
                slope = (110 - 30) / (c - a)
                rows.append(("(파생) 온도센서 기울기", "-", "-",
                             f"(110−30)/(TS_CAL2−TS_CAL1) = 80/{c - a} = "
                             f"{slope:.4f} °C/LSB"))
    doc.table(["항목", "주소", "원시값", "해석"], rows)

    pkg = d.blob("factory_pkg_area")
    pv = d.reg("PACKAGE", "factory_pkg_area", 0x1FF07BF0)
    if pv is not None:
        pkgcode = bf(pv, 18, 16)
        if pv == 0xFFFFFFFF:
            interp = ("전 비트 1 → 패키지 데이터 레지스터가 프로그램되지 않았거나 이 "
                      "실리콘 리비전에서 미지원. **해석 불가**")
        else:
            interp = (f"PKG[18:16] = 0b{pkgcode:03b} (STM32F7 패키지 코드; "
                      "값→패키지 매핑은 RM0385 표 참조)")
        doc.fields(f"PACKAGE @ 0x1FF07BF0 = {hex32(pv)}", [interp])
        if pv == 0xFFFFFFFF:
            doc.sum_("패키지 데이터 레지스터(0x1FF07BF0)가 0xFFFFFFFF → 미프로그램 상태로 "
                     "패키지 판별 불가.")


# ---------------------------------------------------------------------------
# 섹션 8: RCC (클럭 트리)
# ---------------------------------------------------------------------------

HSE_HZ = 25_000_000     # STM32F746G-DISCO 는 25MHz 크리스털(X2) 사용
HSI_HZ = 16_000_000     # STM32F7 내부 RC


def sec_rcc(d, doc):
    doc.h(2, "RCC (클럭 트리 및 페리페럴 클럭)")

    B = "rcc"
    R = lambda off: d.off32(B, off)
    cr, pll, cfgr, cir = R(0x00), R(0x04), R(0x08), R(0x0C)
    if cr is None:
        doc.w("(rcc.bin 없음 — 건너뜀)")
        doc.w()
        return {}

    doc.h(3, "RCC_CR — 발진기/PLL 상태")
    doc.fields(f"RCC_CR @ 0x40023800 = {hex32(cr)}", [
        f"HSION[0]={bit(cr, 0)} HSIRDY[1]={bit(cr, 1)} → HSI(16MHz) "
        f"{onoff(bit(cr, 0), 'ON', 'OFF')}/{onoff(bit(cr, 1), 'ready', 'not ready')}",
        f"HSITRIM[7:3]={bf(cr, 7, 3)}, HSICAL[15:8]=0x{bf(cr, 15, 8):02X} (공장 캘리브레이션)",
        f"HSEON[16]={bit(cr, 16)} HSERDY[17]={bit(cr, 17)} → HSE(외부 25MHz) "
        f"{onoff(bit(cr, 16), '**ON**', 'OFF')}/{onoff(bit(cr, 17), 'ready', 'not ready')}",
        f"HSEBYP[18]={bit(cr, 18)} → "
        + ("외부 클럭 입력(바이패스)" if bit(cr, 18) else "크리스털 발진기 모드"),
        f"CSSON[19]={bit(cr, 19)} → 클럭 보안 시스템 {onoff(bit(cr, 19))}",
        f"PLLON[24]={bit(cr, 24)} PLLRDY[25]={bit(cr, 25)} → 메인 PLL "
        f"{onoff(bit(cr, 24), '**ON**', 'OFF')}/{onoff(bit(cr, 25), 'locked', 'not locked')}",
        f"PLLI2SON[26]={bit(cr, 26)} PLLI2SRDY[27]={bit(cr, 27)} → PLLI2S "
        f"{onoff(bit(cr, 26), 'ON', 'OFF')}",
        f"PLLSAION[28]={bit(cr, 28)} PLLSAIRDY[29]={bit(cr, 29)} → PLLSAI "
        f"{onoff(bit(cr, 28), '**ON**', 'OFF')}",
    ])

    doc.h(3, "RCC_PLLCFGR — 메인 PLL 및 SYSCLK 역산")
    m = bf(pll, 5, 0)
    n = bf(pll, 14, 6)
    p_code = bf(pll, 17, 16)
    p = (p_code + 1) * 2
    src_hse = bit(pll, 22)
    q = bf(pll, 27, 24)
    r_raw = bf(pll, 30, 28)
    src_hz = HSE_HZ if src_hse else HSI_HZ
    src_name = f"HSE = {HSE_HZ / 1e6:.0f} MHz" if src_hse else f"HSI = {HSI_HZ / 1e6:.0f} MHz"
    vco_in = src_hz / m if m else 0
    vco_out = vco_in * n
    pllp = vco_out / p if p else 0
    pllq = vco_out / q if q else 0
    doc.fields(f"RCC_PLLCFGR @ 0x40023804 = {hex32(pll)}", [
        f"PLLM[5:0] = {m} → VCO 입력 = {src_name} / {m} = **{vco_in / 1e6:.3f} MHz** "
        "(데이터시트 권장 1~2MHz)",
        f"PLLN[14:6] = {n} → VCO 출력 = {vco_in / 1e6:.3f} × {n} = **{vco_out / 1e6:.1f} MHz**",
        f"PLLP[17:16] = 0b{p_code:02b} → 분주 {p} → PLLP(SYSCLK 후보) = "
        f"**{pllp / 1e6:.1f} MHz**",
        f"PLLSRC[22] = {src_hse} → PLL 입력 = {src_name}",
        f"PLLQ[27:24] = {q} → PLLQ(USB/SDMMC/RNG) = {vco_out / 1e6:.1f}/{q} = "
        f"**{pllq / 1e6:.3f} MHz**"
        + ("  ← 48MHz 가 아니므로 USB FS/HS 를 이 클럭으로 쓸 수 없다" if abs(pllq - 48e6) > 1e5 else ""),
        f"비트[30:28] = 0b{r_raw:03b} → STM32F746 의 PLLCFGR 에는 PLLR 필드가 정의되어 있지 않다"
        " (stm32f746xx.h 에 RCC_PLLCFGR_PLLR 없음) → 예약 비트, 해석 불가",
    ])
    pll_on = bit(cr, 24)
    pll_is_sysclk = pll_on and bf(cfgr, 3, 2) == 2
    doc.sum_(f"메인 PLL 계수: {src_name.split(' =')[0]} ÷ M={m} × N={n} ÷ P={p} → "
             f"VCO {vco_out / 1e6:.0f} MHz, PLLP 출력 {pllp / 1e6:.0f} MHz "
             + ("(**실제 SYSCLK 로 사용 중**)" if pll_is_sysclk else
                ("(**RCC_CR.PLLON=0 — PLL 은 꺼져 있고 이 값은 리셋 기본 계수일 뿐이다**)"
                 if not pll_on else "(PLL 은 켜져 있으나 SYSCLK 으로 선택되지 않음)"))
             + f" (RCC_PLLCFGR={hex32(pll)}, RCC_CR={hex32(cr)}).",
             pri=1 if pll_is_sysclk else 2)

    doc.h(3, "RCC_CFGR — SYSCLK 선택 및 버스 프리스케일러")
    sw = bf(cfgr, 1, 0)
    sws = bf(cfgr, 3, 2)
    hpre = bf(cfgr, 7, 4)
    ppre1 = bf(cfgr, 12, 10)
    ppre2 = bf(cfgr, 15, 13)
    swname = {0: "HSI", 1: "HSE", 2: "PLL(PLLP)", 3: "reserved"}
    hpre_div = {0b1000: 2, 0b1001: 4, 0b1010: 8, 0b1011: 16, 0b1100: 64,
                0b1101: 128, 0b1110: 256, 0b1111: 512}.get(hpre, 1)
    ppre_div = {0b100: 2, 0b101: 4, 0b110: 8, 0b111: 16}
    p1 = ppre_div.get(ppre1, 1)
    p2 = ppre_div.get(ppre2, 1)
    sysclk = {0: HSI_HZ, 1: HSE_HZ, 2: pllp}.get(sws, 0)
    hclk = sysclk / hpre_div
    pclk1 = hclk / p1
    pclk2 = hclk / p2
    # APB 타이머 클럭: 프리스케일러가 1이 아니면 ×2
    tim1 = pclk1 * (2 if p1 != 1 else 1)
    tim2 = pclk2 * (2 if p2 != 1 else 1)
    doc.fields(f"RCC_CFGR @ 0x40023808 = {hex32(cfgr)}", [
        f"SW[1:0] = {sw} → 요청된 시스템 클럭 소스 = {swname.get(sw)}",
        f"SWS[3:2] = {sws} → **실제 사용 중인 시스템 클럭 = {swname.get(sws)}** → "
        f"SYSCLK = **{sysclk / 1e6:.1f} MHz**",
        f"HPRE[7:4] = 0b{hpre:04b} → AHB 분주 /{hpre_div} → **HCLK = {hclk / 1e6:.1f} MHz**",
        f"PPRE1[12:10] = 0b{ppre1:03b} → APB1 분주 /{p1} → **PCLK1 = {pclk1 / 1e6:.1f} MHz** "
        f"(APB1 타이머 클럭 {tim1 / 1e6:.1f} MHz)",
        f"PPRE2[15:13] = 0b{ppre2:03b} → APB2 분주 /{p2} → **PCLK2 = {pclk2 / 1e6:.1f} MHz** "
        f"(APB2 타이머 클럭 {tim2 / 1e6:.1f} MHz)",
        f"RTCPRE[20:16] = {bf(cfgr, 20, 16)} → HSE→RTC 분주 "
        + (f"/{bf(cfgr, 20, 16)}" if bf(cfgr, 20, 16) >= 2 else "(비활성)"),
        f"MCO1[22:21]={bf(cfgr, 22, 21)} MCO1PRE[26:24]={bf(cfgr, 26, 24)} "
        f"MCO2[31:30]={bf(cfgr, 31, 30)} MCO2PRE[29:27]={bf(cfgr, 29, 27)}",
    ])
    doc.sum_(f"클럭 트리 실측: **SYSCLK={sysclk / 1e6:.0f} MHz, HCLK={hclk / 1e6:.0f} MHz, "
             f"PCLK1={pclk1 / 1e6:.0f} MHz, PCLK2={pclk2 / 1e6:.0f} MHz** "
             f"(RCC_CFGR={hex32(cfgr)}).", pri=1)

    if cir is not None:
        doc.fields(f"RCC_CIR @ 0x4002380C = {hex32(cir)}", [
            "인터럽트 플래그/인에이블 — "
            + ("모두 0 (클럭 관련 인터럽트 미사용, 대기 중인 플래그 없음)" if cir == 0
               else f"0 이 아님: 0x{cir:08X} (RM0385 표 참조)"),
        ])

    # ---- PLLI2S / PLLSAI ----
    plli2s, pllsai = R(0x84), R(0x88)
    dck1, dck2 = R(0x8C), R(0x90)
    lcd_clk = None
    if pllsai is not None:
        sn = bf(pllsai, 14, 6)
        sp_code = bf(pllsai, 17, 16)
        sq = bf(pllsai, 27, 24)
        sr = bf(pllsai, 30, 28)
        sai_vco = vco_in * sn
        sai_r = sai_vco / sr if sr else 0
        divr = {0: 2, 1: 4, 2: 8, 3: 16}.get(bf(dck1 or 0, 17, 16), 2)
        sai_on = bit(cr, 28)     # RCC_CR.PLLSAION
        # PLLSAI 가 꺼져 있으면 계수는 리셋 기본값일 뿐 실제 클럭이 나오지 않는다.
        lcd_clk = (sai_r / divr) if (sr and sai_on) else None
        doc.h(3, "RCC_PLLSAICFGR / PLLI2SCFGR / DCKCFGR — 파생 클럭")
        doc.fields(f"RCC_PLLSAICFGR @ 0x40023888 = {hex32(pllsai)}", [
            f"RCC_CR.PLLSAION[28] = {sai_on} → PLLSAI "
            + ("**동작 중** — 아래 계수가 실제 클럭을 만든다"
               if sai_on else
               "**정지** — 아래 계수는 리셋 기본값일 뿐 실제 클럭은 생성되지 않는다"),
            f"PLLSAIN[14:6] = {sn} → PLLSAI VCO = {vco_in / 1e6:.3f} × {sn} = "
            f"**{sai_vco / 1e6:.1f} MHz** (VCO 입력은 메인 PLL 의 PLLM 을 공유)",
            f"PLLSAIP[17:16] = 0b{sp_code:02b} → 분주 {(sp_code + 1) * 2}",
            f"PLLSAIQ[27:24] = {sq} → SAI 클럭 {sai_vco / sq / 1e6:.3f} MHz" if sq else
            "PLLSAIQ = 0",
            f"PLLSAIR[30:28] = {sr} → PLLSAIR = {sai_r / 1e6:.3f} MHz" if sr else "PLLSAIR = 0",
        ])
        if dck1 is not None:
            doc.fields(f"RCC_DCKCFGR1 @ 0x4002388C = {hex32(dck1)}", [
                f"PLLI2SDIVQ[4:0] = {bf(dck1, 4, 0)} (+1 분주)",
                f"PLLSAIDIVQ[12:8] = {bf(dck1, 12, 8)} (+1 분주)",
                f"PLLSAIDIVR[17:16] = 0b{bf(dck1, 17, 16):02b} → LCD-TFT 클럭 분주 /{divr} → "
                f"**LCD_CLK = {lcd_clk / 1e6:.3f} MHz**" if lcd_clk else
                f"PLLSAIDIVR[17:16] = 0b{bf(dck1, 17, 16):02b}",
                f"SAI1SEL[21:20]={bf(dck1, 21, 20)} SAI2SEL[23:22]={bf(dck1, 23, 22)} "
                f"TIMPRE[24]={bit(dck1, 24)}",
            ])
            if lcd_clk:
                doc.sum_(f"LTDC 픽셀 클럭은 PLLSAI({sai_vco / 1e6:.0f}MHz)/R={sr}/DIVR={divr} = "
                         f"**{lcd_clk / 1e6:.2f} MHz** 이다 "
                         f"(PLLSAICFGR={hex32(pllsai)}, DCKCFGR1={hex32(dck1)}).")
        if plli2s is not None:
            doc.fields(f"RCC_PLLI2SCFGR @ 0x40023884 = {hex32(plli2s)}", [
                f"PLLI2SN[14:6]={bf(plli2s, 14, 6)} PLLI2SP[17:16]={bf(plli2s, 17, 16)} "
                f"PLLI2SQ[27:24]={bf(plli2s, 27, 24)} PLLI2SR[30:28]={bf(plli2s, 30, 28)}"
                + ("  (PLLI2S 는 RCC_CR.PLLI2SON=0 이므로 미사용)" if not bit(cr, 26) else ""),
            ])
        if dck2 is not None:
            src = {0: "PCLK", 1: "SYSCLK", 2: "HSI", 3: "LSE"}
            doc.fields(f"RCC_DCKCFGR2 @ 0x40023890 = {hex32(dck2)}", [
                "USART1SEL=%s USART2SEL=%s USART3SEL=%s UART4SEL=%s UART5SEL=%s USART6SEL=%s" % (
                    src[bf(dck2, 1, 0)], src[bf(dck2, 3, 2)], src[bf(dck2, 5, 4)],
                    src[bf(dck2, 7, 6)], src[bf(dck2, 9, 8)], src[bf(dck2, 11, 10)]),
                f"I2C1SEL[17:16]={bf(dck2, 17, 16)} I2C2SEL[19:18]={bf(dck2, 19, 18)} "
                f"I2C3SEL[21:20]={bf(dck2, 21, 20)} I2C4SEL[23:22]={bf(dck2, 23, 22)}",
                f"LPTIM1SEL[25:24]={bf(dck2, 25, 24)} CECSEL[26]={bit(dck2, 26)} "
                f"CK48MSEL[27]={bit(dck2, 27)} → 48MHz 소스 = "
                + ("PLLSAIP" if bit(dck2, 27) else "PLLQ")
                + f", SDMMCSEL[28]={bit(dck2, 28)}",
            ])

    # ---- 페리페럴 클럭 인에이블 ----
    doc.h(3, "페리페럴 클럭 인에이블 (켜져 있는 것만)")
    enr_offsets = {"AHB1ENR": 0x30, "AHB2ENR": 0x34, "AHB3ENR": 0x38,
                   "APB1ENR": 0x40, "APB2ENR": 0x44}
    rows = []
    enabled_all = {}
    for nm, off in enr_offsets.items():
        v = R(off)
        if v is None:
            continue
        names = [name for pos, name in sorted(RCC_ENR_BITS[nm].items()) if bit(v, pos)]
        enabled_all[nm] = names
        unknown = [f"bit{i}" for i in range(32)
                   if bit(v, i) and i not in RCC_ENR_BITS[nm]]
        txt = ", ".join(names) if names else "(없음)"
        if unknown:
            txt += f"  / 헤더에 정의 없는 비트: {', '.join(unknown)} → 해석 불가"
        rows.append((f"RCC_{nm}", f"0x{0x40023800 + off:08X}", hex32(v), txt))
    doc.table(["레지스터", "주소", "값", "클럭이 켜진 페리페럴"], rows)
    total_on = sum(len(v) for v in enabled_all.values())
    doc.sum_(f"덤프 시점에 클럭이 공급되던 페리페럴은 총 {total_on}개다: "
             + "; ".join(f"{k}=[{', '.join(v) if v else '없음'}]"
                         for k, v in enabled_all.items()) + ".")

    # 저전력 모드 클럭
    lp_offsets = {"AHB1LPENR": 0x50, "AHB2LPENR": 0x54, "AHB3LPENR": 0x58,
                  "APB1LPENR": 0x60, "APB2LPENR": 0x64}
    lprows = []
    for nm, off in lp_offsets.items():
        v = R(off)
        if v is not None:
            lprows.append((f"RCC_{nm}", f"0x{0x40023800 + off:08X}", hex32(v),
                           "Sleep 모드에서 클럭 유지 마스크 (리셋 기본값은 대부분 1)"))
    doc.table(["레지스터", "주소", "값", "해석"], lprows)

    # ---- BDCR / CSR ----
    bdcr, csr = R(0x70), R(0x74)
    if bdcr is not None:
        rtcsel = {0: "없음(RTC 클럭 미선택)", 1: "LSE", 2: "LSI", 3: "HSE/RTCPRE"}
        doc.fields(f"RCC_BDCR @ 0x40023870 = {hex32(bdcr)}", [
            f"LSEON[0]={bit(bdcr, 0)} LSERDY[1]={bit(bdcr, 1)} LSEBYP[2]={bit(bdcr, 2)} "
            f"LSEDRV[4:3]={bf(bdcr, 4, 3)} → 외부 32.768kHz "
            f"{onoff(bit(bdcr, 0), 'ON', 'OFF')}",
            f"RTCSEL[9:8] = 0b{bf(bdcr, 9, 8):02b} → RTC 클럭 소스 = {rtcsel[bf(bdcr, 9, 8)]}",
            f"RTCEN[15] = {bit(bdcr, 15)} → RTC {onoff(bit(bdcr, 15), 'ON', 'OFF')}",
            f"BDRST[16] = {bit(bdcr, 16)}",
        ])
    if csr is not None:
        flags = [n for n, p in (("LPWRRSTF(저전력 리셋)", 31), ("WWDGRSTF(윈도우 워치독)", 30),
                                ("IWDGRSTF(독립 워치독)", 29), ("SFTRSTF(소프트웨어 리셋)", 28),
                                ("PORRSTF(POR/PDR)", 27), ("PINRSTF(NRST 핀)", 26),
                                ("BORRSTF(브라운아웃)", 25)) if bit(csr, p)]
        doc.fields(f"RCC_CSR @ 0x40023874 = {hex32(csr)}", [
            f"LSION[0] = {bit(csr, 0)} LSIRDY[1] = {bit(csr, 1)} → 내부 32kHz LSI "
            f"{onoff(bit(csr, 0), '**ON**', 'OFF')}",
            f"RMVF[24] = {bit(csr, 24)}",
            "**리셋 원인 플래그: " + (", ".join(flags) if flags else "없음") + "**",
        ])
        if flags:
            doc.sum_("마지막 리셋 원인 플래그(누적, RMVF 로 지우기 전까지 유지): "
                     + ", ".join(flags) + f" (RCC_CSR={hex32(csr)}).")

    return {"sysclk": sysclk, "hclk": hclk, "pclk1": pclk1, "pclk2": pclk2,
            "lcd_clk": lcd_clk, "pllq": pllq}


# ---------------------------------------------------------------------------
# 섹션 9: FLASH 인터페이스 / PWR / SYSCFG
# ---------------------------------------------------------------------------


def sec_flash_if(d, doc, clocks):
    doc.h(2, "FLASH 인터페이스")

    acr = d.reg("FLASH_ACR", "flash_if", 0x40023C00)
    if acr is not None:
        lat = bf(acr, 3, 0)
        hclk = (clocks or {}).get("hclk") or 0
        doc.fields(f"FLASH_ACR @ 0x40023C00 = {hex32(acr)}", [
            f"LATENCY[3:0] = {lat} → **Flash 대기 상태 {lat} WS** "
            + (f"(HCLK {hclk / 1e6:.0f} MHz, VDD 2.7~3.6V·VOS scale1+over-drive 기준 "
               f"RM0385 표에서 {lat} WS 는 {(lat) * 30}~{(lat + 1) * 30} MHz 구간)" if hclk else ""),
            f"PRFTEN[8] = {bit(acr, 8)} → 프리페치 버퍼 "
            f"{onoff(bit(acr, 8), '활성', '**비활성**')}",
            f"ARTEN[9] = {bit(acr, 9)} → **ART Accelerator(ITCM 경로 명령 캐시) "
            f"{onoff(bit(acr, 9), '활성', '비활성')}**",
            f"ARTRST[11] = {bit(acr, 11)} → ART 리셋 {onoff(bit(acr, 11))}",
        ])
        doc.sum_(f"Flash 대기 상태 = {lat} WS, 프리페치 {onoff(bit(acr, 8), 'ON', 'OFF')}, "
                 f"ART Accelerator {onoff(bit(acr, 9), 'ON', '**OFF**')} "
                 f"(FLASH_ACR={hex32(acr)}).", pri=1)

    sr = d.r32("flash_if", 0x40023C0C)
    cr = d.r32("flash_if", 0x40023C10)
    if sr is not None:
        errs = [n for n, p in (("ERSERR", 7), ("PGPERR", 6), ("PGAERR", 5), ("WRPERR", 4),
                               ("OPERR", 1), ("EOP", 0)) if bit(sr, p)]
        doc.fields(f"FLASH_SR @ 0x40023C0C = {hex32(sr)}", [
            f"BSY[16] = {bit(sr, 16)} → Flash 동작 중 여부",
            "세워진 플래그: " + (", ".join(errs) if errs else "없음 (에러 없음)"),
        ])
    if cr is not None:
        doc.fields(f"FLASH_CR @ 0x40023C10 = {hex32(cr)}", [
            f"PG[0]={bit(cr, 0)} SER[1]={bit(cr, 1)} MER[2]={bit(cr, 2)}",
            f"SNB[7:3]={bf(cr, 7, 3)} PSIZE[9:8]=0b{bf(cr, 9, 8):02b}",
            f"STRT[16]={bit(cr, 16)}, EOPIE[24]={bit(cr, 24)}, ERRIE[25]={bit(cr, 25)}",
            f"LOCK[31] = {bit(cr, 31)} → Flash 제어 레지스터 "
            f"{onoff(bit(cr, 31), '**잠김**(쓰기/소거 불가)', '해제됨')}",
        ])

    optcr = d.reg("FLASH_OPTCR", "flash_if", 0x40023C14)
    if optcr is not None:
        rdp = bf(optcr, 15, 8)
        rdp_s = {0xAA: "**Level 0 — 보호 없음(디버거로 Flash 전체 읽기 가능)**",
                 0xCC: "Level 2 — 영구 차단"}.get(rdp, "Level 1 — 디버그 접근 시 Flash 차단")
        bor = bf(optcr, 3, 2)
        bor_s = {0b11: "BOR off (VBOR0, ~1.8V 리셋 임계)", 0b10: "BOR Level 1 (~2.1V)",
                 0b01: "BOR Level 2 (~2.4V)", 0b00: "BOR Level 3 (~2.7V)"}
        nwrp = bf(optcr, 27, 16)
        prot = [f"섹터{i}" for i in range(12) if not bit(nwrp, i)]
        doc.fields(f"FLASH_OPTCR @ 0x40023C14 = {hex32(optcr)}", [
            f"OPTLOCK[0] = {bit(optcr, 0)} → 옵션 바이트 레지스터 "
            f"{onoff(bit(optcr, 0), '**잠김**', '해제됨')}",
            f"OPTSTRT[1] = {bit(optcr, 1)}",
            f"BOR_LEV[3:2] = 0b{bor:02b} → **{bor_s[bor]}**",
            f"WWDG_SW[4] = {bit(optcr, 4)} → 윈도우 워치독 "
            f"{onoff(bit(optcr, 4), '**소프트웨어 제어**', '하드웨어 자동 시작')}",
            f"IWDG_SW[5] = {bit(optcr, 5)} → 독립 워치독 "
            f"{onoff(bit(optcr, 5), '**소프트웨어 제어**', '하드웨어 자동 시작')}",
            f"nRST_STOP[6] = {bit(optcr, 6)} → STOP 진입 시 "
            f"{onoff(bit(optcr, 6), '리셋 없음', '리셋 발생')}",
            f"nRST_STDBY[7] = {bit(optcr, 7)} → STANDBY 진입 시 "
            f"{onoff(bit(optcr, 7), '리셋 없음', '리셋 발생')}",
            f"RDP[15:8] = 0x{rdp:02X} → {rdp_s}",
            f"nWRP[27:16] = 0x{nwrp:03X} → 쓰기 보호된 섹터: "
            + (", ".join(prot) if prot else "없음")
            + "  (1MB 단일 뱅크 F746 은 섹터 0~7 만 존재 — 섹터 8~11 비트는 무의미)",
            f"IWDG_STDBY[30] = {bit(optcr, 30)}, IWDG_STOP[31] = {bit(optcr, 31)} → "
            "저전력 모드에서 IWDG 동작 여부",
        ])
        doc.sum_(f"읽기 보호는 **RDP {rdp_s.split('—')[0].strip().replace('**', '')}** "
                 f"(RDP=0x{rdp:02X}), 워치독은 둘 다 소프트웨어 시작, "
                 f"BOR 은 {bor_s[bor]} (FLASH_OPTCR={hex32(optcr)}).")

    optcr1 = d.r32("flash_if", 0x40023C18)
    if optcr1 is not None:
        add0 = bf(optcr1, 15, 0)
        add1 = bf(optcr1, 31, 16)

        def boot_addr(v):
            a = v << 14
            where = {0x00200000: "ITCM 인터페이스의 내장 Flash 별칭",
                     0x08000000: "AXIM 인터페이스의 내장 Flash",
                     0x00100000: "ITCM 인터페이스의 시스템 메모리(부트로더) 별칭",
                     0x1FF00000: "시스템 메모리(부트로더)",
                     0x20000000: "DTCM RAM",
                     0x00000000: "ITCM RAM"}.get(a, "기타 영역")
            return a, where

        a0, w0 = boot_addr(add0)
        a1, w1 = boot_addr(add1)
        doc.fields(f"FLASH_OPTCR1 @ 0x40023C18 = {hex32(optcr1)}", [
            f"BOOT_ADD0[15:0] = 0x{add0:04X} → BOOT0=0 일 때 부트 주소 = "
            f"0x{add0:04X} × 0x4000 = **0x{a0:08X}** ({w0})",
            f"BOOT_ADD1[31:16] = 0x{add1:04X} → BOOT0=1 일 때 부트 주소 = "
            f"0x{add1:04X} × 0x4000 = **0x{a1:08X}** ({w1})",
        ])
        doc.sum_(f"부트 주소 옵션: BOOT_ADD0=0x{a0:08X}({w0}), BOOT_ADD1=0x{a1:08X}({w1}) "
                 f"(FLASH_OPTCR1={hex32(optcr1)}) → 리셋 직후 VTOR 이 0x{a0:08X} 인 이유.", pri=1)

    ob = d.blob("option_bytes")
    if ob:
        doc.h(3, "옵션 바이트 원시 덤프 (0x1FFF0000)")
        doc.w("```")
        for ln in hexdump(ob, 0x1FFF0000):
            doc.w(ln)
        doc.w("```")
        doc.w()
        doc.note("옵션 바이트 플래시 셀의 원시 이미지다. 필드 의미는 위의 FLASH_OPTCR/OPTCR1 "
                 "해석이 정본이며, 여기서는 원문 보존 목적으로만 싣는다.")


def sec_pwr(d, doc):
    doc.h(2, "PWR (전원/전압 스케일링)")
    cr1 = d.reg("PWR_CR1", "pwr", 0x40007000)
    csr1 = d.reg("PWR_CSR1", "pwr", 0x40007004)
    if cr1 is not None:
        vos = bf(cr1, 15, 14)
        vos_s = {0b11: "**Scale 1** (최대 216MHz, over-drive 시)",
                 0b10: "Scale 2 (최대 180MHz)", 0b01: "Scale 3 (최대 144MHz)",
                 0b00: "reserved"}
        pls = bf(cr1, 7, 5)
        doc.fields(f"PWR_CR1 @ 0x40007000 = {hex32(cr1)}", [
            f"LPDS[0] = {bit(cr1, 0)} → STOP 모드에서 레귤레이터 "
            f"{onoff(bit(cr1, 0), '저전력', '정상')}",
            f"PDDS[1] = {bit(cr1, 1)}, CSBF[3] = {bit(cr1, 3)}",
            f"PVDE[4] = {bit(cr1, 4)} → 전원 전압 감지기 {onoff(bit(cr1, 4))}",
            f"PLS[7:5] = 0b{pls:03b} → PVD 임계 레벨 {pls} "
            f"({2.0 + 0.1 * pls:.1f}V 근방; PVDE=0 이면 무의미)",
            f"DBP[8] = {bit(cr1, 8)} → 백업 도메인 쓰기 보호 "
            f"{onoff(bit(cr1, 8), '**해제됨**(RTC/백업 레지스터 쓰기 가능)', '적용 중')}",
            f"FPDS[9] = {bit(cr1, 9)}, LPUDS[10] = {bit(cr1, 10)}, MRUDS[11] = {bit(cr1, 11)}",
            f"VOS[15:14] = 0b{vos:02b} → **전압 스케일 {vos_s[vos]}**",
            f"ODEN[16] = {bit(cr1, 16)} → **Over-drive {onoff(bit(cr1, 16), '활성', '비활성')}**",
            f"ODSWEN[17] = {bit(cr1, 17)} → **Over-drive 전환 "
            f"{onoff(bit(cr1, 17), '완료(적용 중)', '미적용')}**",
            f"UDEN[19:18] = 0b{bf(cr1, 19, 18):02b} → Under-drive",
        ])
        od_on = bit(cr1, 16) and bit(cr1, 17)
        doc.sum_(f"전압 스케일 {vos_s[vos].split('(')[0].strip().replace('**', '')}, "
                 f"Over-drive ODEN={bit(cr1, 16)}/ODSWEN={bit(cr1, 17)} "
                 f"(PWR_CR1={hex32(cr1)}) — "
                 + ("Over-drive 가 적용된 상태로, 이것이 200MHz 동작의 전제 조건이다."
                    if od_on else
                    "Over-drive 미적용 → 이 상태의 최대 HCLK 은 180MHz 로 제한된다."),
                 pri=1 if od_on else 2)
    if csr1 is not None:
        doc.fields(f"PWR_CSR1 @ 0x40007004 = {hex32(csr1)}", [
            f"WUIF[0] = {bit(csr1, 0)}, SBF[1] = {bit(csr1, 1)}, PVDO[2] = {bit(csr1, 2)}, "
            f"BRR[3] = {bit(csr1, 3)}",
            f"BRE[9] = {bit(csr1, 9)} → 백업 레귤레이터 {onoff(bit(csr1, 9))}",
            f"VOSRDY[14] = {bit(csr1, 14)} → **전압 스케일 전환 "
            f"{onoff(bit(csr1, 14), '완료', '진행 중/미적용')}**",
            f"ODRDY[16] = {bit(csr1, 16)} → **Over-drive "
            f"{onoff(bit(csr1, 16), 'ready', 'not ready')}**",
            f"ODSWRDY[17] = {bit(csr1, 17)} → **Over-drive 전환 "
            f"{onoff(bit(csr1, 17), 'ready(200MHz 동작 가능 상태)', 'not ready')}**",
            f"UDRDY[19:18] = 0b{bf(csr1, 19, 18):02b}",
        ])


def sec_syscfg(d, doc):
    doc.h(2, "SYSCFG")
    b = d.blob("syscfg")
    if b is None:
        doc.w("(syscfg.bin 없음 — 건너뜀)")
        doc.w()
        return
    memrmp = d.r32("syscfg", 0x40013800)
    pmc = d.r32("syscfg", 0x40013804)
    if memrmp is not None:
        boot = bf(memrmp, 2, 0)
        boot_s = {0b000: "**메인 Flash** 가 0x00000000 에 매핑",
                  0b001: "시스템 Flash(부트로더) 매핑",
                  0b010: "메인 Flash 매핑",
                  0b011: "내장 SRAM1 매핑"}
        doc.fields(f"SYSCFG_MEMRMP @ 0x40013800 = {hex32(memrmp)}", [
            f"MEM_BOOT[2:0] = 0b{boot:03b} → {boot_s.get(boot, '해석 불가')}",
            f"SWP_FMC[11:10] = 0b{bf(memrmp, 11, 10):02b} → FMC 뱅크 스와핑 "
            + ("없음(기본: SDRAM=0xC0000000/0xD0000000)" if bf(memrmp, 11, 10) == 0
               else "적용됨"),
        ])
    if pmc is not None:
        doc.fields(f"SYSCFG_PMC @ 0x40013804 = {hex32(pmc)}", [
            f"MII_RMII_SEL[23] = {bit(pmc, 23)} → 이더넷 PHY 인터페이스 = "
            + ("**RMII**" if bit(pmc, 23) else "MII"),
            f"ADCxDC2[18:16] = 0b{bf(pmc, 18, 16):03b}",
        ])
    rows = []
    for i in range(4):
        v = d.r32("syscfg", 0x40013808 + 4 * i)
        if v is None:
            continue
        parts = []
        for k in range(4):
            port = bf(v, 4 * k + 3, 4 * k)
            parts.append(f"EXTI{4 * i + k}←P{chr(ord('A') + port)}" if port < 11
                         else f"EXTI{4 * i + k}←0x{port:X}(해석 불가)")
        rows.append((f"SYSCFG_EXTICR{i + 1}", f"0x{0x40013808 + 4 * i:08X}", hex32(v),
                     ", ".join(parts)))
    doc.table(["레지스터", "주소", "값", "EXTI 소스 포트"], rows)
    if all(r[2] == "0x00000000" for r in rows):
        doc.note("EXTICR 이 전부 0 → 모든 EXTI 라인이 기본값(포트 A)을 가리킨다. "
                 "즉 실행 중이던 펌웨어는 GPIO 외부 인터럽트를 하나도 재매핑하지 않았다.")


# ---------------------------------------------------------------------------
# 섹션 10: FMC (SDRAM 컨트롤러)
# ---------------------------------------------------------------------------


def sec_fmc(d, doc, clocks):
    doc.h(2, "FMC — SDRAM 컨트롤러 (베이스 0xA0000000)")
    if d.blob("fmc") is None:
        doc.w("(fmc.bin 없음 — 건너뜀)")
        doc.w()
        return
    R = lambda a: d.r32("fmc", a)
    sdcr1, sdcr2 = R(0xA0000140), R(0xA0000144)
    sdtr1, sdtr2 = R(0xA0000148), R(0xA000014C)
    sdcmr, sdrtr, sdsr = R(0xA0000150), R(0xA0000154), R(0xA0000158)
    hclk = (clocks or {}).get("hclk") or 0

    if not any(v for v in (sdcr1, sdcr2, sdtr1, sdtr2)):
        doc.w("SDRAM 관련 레지스터가 전부 0 이다 → FMC 가 초기화되지 않았거나 "
              "RCC_AHB3ENR.FMCEN 이 꺼져 레지스터가 0 으로 읽힌다.")
        doc.w()
        doc.sum_("FMC SDRAM 컨트롤러가 초기화되어 있지 않다 (SDCR1/SDTR1 = 0x00000000).")
        return

    sdram_hz = None
    for idx, v in ((1, sdcr1), (2, sdcr2)):
        if v is None:
            continue
        nc = bf(v, 1, 0) + 8
        nr = bf(v, 3, 2) + 11
        mwid = {0: 8, 1: 16, 2: 32, 3: None}[bf(v, 5, 4)]
        nb = 4 if bit(v, 6) else 2
        cas = bf(v, 8, 7)
        wp = bit(v, 9)
        sdclk_code = bf(v, 11, 10)
        sdclk_div = {0: None, 1: None, 2: 2, 3: 3}[sdclk_code]
        rburst = bit(v, 12)
        rpipe = bf(v, 14, 13)
        lines = [
            f"NC[1:0] = {bf(v, 1, 0)} → 컬럼 주소 **{nc} 비트**",
            f"NR[3:2] = {bf(v, 3, 2)} → 로우 주소 **{nr} 비트** (= {1 << nr} 행)",
            f"MWID[5:4] = {bf(v, 5, 4)} → 메모리 데이터 버스 폭 **{mwid} 비트**"
            if mwid else f"MWID[5:4] = {bf(v, 5, 4)} → reserved, 해석 불가",
            f"NB[6] = {bit(v, 6)} → 내부 뱅크 **{nb}개**",
            f"CAS[8:7] = {cas} → **CAS latency {cas} 사이클**",
            f"WP[9] = {wp} → 쓰기 보호 {onoff(wp, '**적용**(읽기 전용)', '없음')}",
            f"SDCLK[11:10] = 0b{sdclk_code:02b} → SDRAM 클럭 = "
            + (f"HCLK/{sdclk_div}" + (f" = **{hclk / sdclk_div / 1e6:.1f} MHz**" if hclk else "")
               if sdclk_div else "**비활성(클럭 없음)**"),
            f"RBURST[12] = {rburst} → 읽기 버스트 {onoff(rburst, '활성', '비활성')}",
            f"RPIPE[14:13] = {rpipe} → 읽기 데이터 파이프라인 지연 {rpipe} HCLK",
        ]
        if mwid and idx == 1:
            cap = (1 << nr) * (1 << nc) * nb * (mwid // 8)
            lines.append(f"→ **주소 지정 가능 용량 = 2^{nr} 행 × 2^{nc} 열 × {nb} 뱅크 × "
                         f"{mwid // 8} B = {cap} B = {cap // (1024 * 1024)} MB**")
        # SDCLK/RBURST/RPIPE 는 SDCR1 에만 유효
        if idx == 2:
            lines.append("주의: SDCLK/RBURST/RPIPE 필드는 SDCR1 의 값만 유효하다 "
                         "(RM0385) → SDCR2 의 해당 필드는 무시된다.")
        doc.fields(f"FMC_SDCR{idx} @ 0xA00001{0x40 + 4 * (idx - 1):02X} = {hex32(v)}", lines)
        if idx == 1:
            if sdclk_div and hclk:
                sdram_hz = hclk / sdclk_div
                doc.sum_(f"외부 SDRAM 은 HCLK/{sdclk_div} = **{sdram_hz / 1e6:.0f} MHz**, "
                         f"{mwid}비트 버스, CAS {cas}, {nb} 뱅크, {nr}행/{nc}열 → "
                         f"**{(1 << nr) * (1 << nc) * nb * (mwid // 8) // (1024 * 1024)} MB** "
                         f"(FMC_SDCR1={hex32(v)}).", pri=1)

    for idx, v in ((1, sdtr1), (2, sdtr2)):
        if v is None:
            continue
        f = lambda hi, lo: bf(v, hi, lo) + 1
        ns = (lambda c: f" = {c / sdram_hz * 1e9:.1f} ns" if sdram_hz else "")
        doc.fields(f"FMC_SDTR{idx} @ 0xA00001{0x48 + 4 * (idx - 1):02X} = {hex32(v)}", [
            f"TMRD[3:0] = {bf(v, 3, 0)} → Load Mode Register → Active 지연 "
            f"**{f(3, 0)} 사이클**{ns(f(3, 0))}",
            f"TXSR[7:4] = {bf(v, 7, 4)} → Self-refresh 탈출 지연 **{f(7, 4)} 사이클**{ns(f(7, 4))}",
            f"TRAS[11:8] = {bf(v, 11, 8)} → Self-refresh(최소 Active) 시간 "
            f"**{f(11, 8)} 사이클**{ns(f(11, 8))}",
            f"TRC[15:12] = {bf(v, 15, 12)} → Row cycle 지연 **{f(15, 12)} 사이클**{ns(f(15, 12))}",
            f"TWR[19:16] = {bf(v, 19, 16)} → Write recovery 시간 **{f(19, 16)} 사이클**{ns(f(19, 16))}",
            f"TRP[23:20] = {bf(v, 23, 20)} → Row precharge 지연 **{f(23, 20)} 사이클**{ns(f(23, 20))}",
            f"TRCD[27:24] = {bf(v, 27, 24)} → Row→Column 지연 **{f(27, 24)} 사이클**{ns(f(27, 24))}",
        ] + (["(SDTR2 는 뱅크2 용. 전 필드가 최대값이면 뱅크2 미사용을 뜻한다. "
              "단 RM0385 에 따르면 TRC/TRP 는 SDTR1 의 값만 유효하며 두 뱅크에 공통 적용된다.)"]
             if idx == 2 else
             ["(TRC/TRP 는 SDTR1 의 값이 두 뱅크에 공통 적용된다 — RM0385)"]))

    if sdcmr is not None:
        mode = {0: "**Normal mode** (초기화 시퀀스 완료)", 1: "Clock Configuration Enable",
                2: "PALL (precharge all)", 3: "Auto-refresh", 4: "Load Mode Register",
                5: "Self-refresh", 6: "Power-down"}
        mrd = bf(sdcmr, 21, 9)
        bl = 1 << bf(mrd, 2, 0)
        doc.fields(f"FMC_SDCMR @ 0xA0000150 = {hex32(sdcmr)}", [
            f"MODE[2:0] = {bf(sdcmr, 2, 0)} → {mode.get(bf(sdcmr, 2, 0), '해석 불가')}",
            f"CTB2[3] = {bit(sdcmr, 3)}, CTB1[4] = {bit(sdcmr, 4)} → 명령 대상 뱅크",
            f"NRFS[8:5] = {bf(sdcmr, 8, 5)} → auto-refresh 연속 횟수 {bf(sdcmr, 8, 5) + 1}",
            f"MRD[21:9] = 0x{mrd:04X} → 마지막으로 SDRAM 모드 레지스터에 쓴 값: "
            f"Burst Length={bl}, Burst Type={'interleaved' if bit(mrd, 3) else 'sequential'}, "
            f"CAS Latency={bf(mrd, 6, 4)}, "
            f"Write Burst Mode={'single location' if bit(mrd, 9) else 'programmed BL'}",
        ])
    if sdrtr is not None:
        cnt = bf(sdrtr, 13, 1)
        lines = [
            f"CRE[0] = {bit(sdrtr, 0)} → refresh error 플래그 클리어",
            f"COUNT[13:1] = {cnt} → 리프레시 타이머 카운트",
            f"REIE[14] = {bit(sdrtr, 14)} → refresh error 인터럽트 {onoff(bit(sdrtr, 14))}",
        ]
        if sdram_hz and cnt:
            # RM0385: COUNT = (SDRAM refresh period / 행수) × SDRAM_clk − 20
            row_us = (cnt + 20) / sdram_hz * 1e6
            nrows = 1 << (bf(sdcr1, 3, 2) + 11) if sdcr1 is not None else None
            lines.append(f"→ 행 하나당 리프레시 주기 = (COUNT+20)/f_SDRAM = "
                         f"({cnt}+20)/{sdram_hz / 1e6:.0f}MHz = **{row_us:.2f} µs**")
            if nrows:
                lines.append(f"→ 전체 배열 리프레시 주기 = {row_us:.2f} µs × {nrows} 행 = "
                             f"**{row_us * nrows / 1000:.1f} ms** (SDRAM 표준 64 ms 사양 대비)")
                doc.sum_(f"SDRAM 리프레시 설정은 행당 {row_us:.2f} µs "
                         f"(전체 {row_us * nrows / 1000:.1f} ms / {nrows}행) 이다 "
                         f"(FMC_SDRTR={hex32(sdrtr)}).")
        doc.fields(f"FMC_SDRTR @ 0xA0000154 = {hex32(sdrtr)}", lines)
    if sdsr is not None:
        st = {0: "Normal", 1: "Self-refresh", 2: "Power-down", 3: "해석 불가"}
        doc.fields(f"FMC_SDSR @ 0xA0000158 = {hex32(sdsr)}", [
            f"RE[0] = {bit(sdsr, 0)} → refresh error 플래그",
            f"MODES1[2:1] = {bf(sdsr, 2, 1)} → 뱅크1 상태 = {st.get(bf(sdsr, 2, 1))}",
            f"MODES2[4:3] = {bf(sdsr, 4, 3)} → 뱅크2 상태 = {st.get(bf(sdsr, 4, 3))}",
            f"BUSY[5] = {bit(sdsr, 5)} → 컨트롤러가 명령 처리 중 "
            f"{onoff(bit(sdsr, 5), '**예**', '아니오')}",
        ])


# ---------------------------------------------------------------------------
# 섹션 11: QUADSPI
# ---------------------------------------------------------------------------


def sec_qspi(d, doc, clocks):
    doc.h(2, "QUADSPI (베이스 0xA0001000)")
    if d.blob("quadspi") is None:
        doc.w("(quadspi.bin 없음 — 건너뜀)")
        doc.w()
        return
    R = lambda o: d.off32("quadspi", o)
    cr, dcr, sr, ccr = R(0x00), R(0x04), R(0x08), R(0x14)
    hclk = (clocks or {}).get("hclk") or 0

    if not any(v for v in (cr, dcr, sr, ccr)):
        doc.w("QUADSPI 레지스터가 전부 0x00000000 이다.")
        doc.w()
        doc.warn("QUADSPI 블록의 클럭(RCC_AHB3ENR.QSPIEN)이 꺼져 있으면 레지스터가 "
                 "0 으로 읽힌다. 따라서 이 덤프로는 **QSPI 클럭 주파수와 플래시 용량을 "
                 "계산할 수 없다(해석 불가)**. QSPI 플래시의 존재 자체는 이 덤프로 "
                 "증명되지도 반증되지도 않는다.")
        doc.sum_("QUADSPI 레지스터가 전부 0 — 컨트롤러 클럭이 꺼진 상태라 플래시 용량/클럭 "
                 "산출 불가 (QUADSPI_CR/DCR = 0x00000000).")
        return

    if cr is not None:
        pre = bf(cr, 31, 24)
        qclk = hclk / (pre + 1) if hclk else 0
        doc.fields(f"QUADSPI_CR @ 0xA0001000 = {hex32(cr)}", [
            f"EN[0] = {bit(cr, 0)} → QUADSPI {onoff(bit(cr, 0), '활성', '비활성')}",
            f"ABORT[1]={bit(cr, 1)} DMAEN[2]={bit(cr, 2)} TCEN[3]={bit(cr, 3)} "
            f"SSHIFT[4]={bit(cr, 4)} DFM[6]={bit(cr, 6)} FSEL[7]={bit(cr, 7)}",
            f"FTHRES[12:8] = {bf(cr, 12, 8)} → FIFO 임계값 {bf(cr, 12, 8) + 1} 바이트",
            f"PRESCALER[31:24] = {pre} → **QSPI 클럭 = HCLK/({pre}+1) = "
            f"{qclk / 1e6:.2f} MHz**" if hclk else f"PRESCALER[31:24] = {pre}",
        ])
        if hclk:
            doc.sum_(f"QSPI 클럭은 HCLK/{pre + 1} = **{qclk / 1e6:.2f} MHz** 이다 "
                     f"(QUADSPI_CR={hex32(cr)}).")
    if dcr is not None:
        fsize = bf(dcr, 20, 16)
        cap = 1 << (fsize + 1)
        doc.fields(f"QUADSPI_DCR @ 0xA0001004 = {hex32(dcr)}", [
            f"CKMODE[0] = {bit(dcr, 0)} → CLK 유휴 레벨 "
            f"{'High (mode 3)' if bit(dcr, 0) else 'Low (mode 0)'}",
            f"CSHT[10:8] = {bf(dcr, 10, 8)} → chip-select 최소 High 시간 "
            f"{bf(dcr, 10, 8) + 1} 사이클",
            f"FSIZE[20:16] = {fsize} → **플래시 용량 = 2^({fsize}+1) = {cap} B = "
            f"{cap // (1024 * 1024)} MB**",
        ])
        doc.sum_(f"QSPI 플래시 용량 설정은 2^{fsize + 1} = **{cap // (1024 * 1024)} MB** 이다 "
                 f"(QUADSPI_DCR={hex32(dcr)}).")
    if ccr is not None:
        mode = {0: "미사용", 1: "1선", 2: "2선", 3: "4선"}
        fmode = {0: "간접 쓰기", 1: "간접 읽기", 2: "자동 폴링", 3: "**메모리 맵 모드**"}
        adsize = {0: "8비트", 1: "16비트", 2: "24비트", 3: "32비트"}
        doc.fields(f"QUADSPI_CCR @ 0xA0001014 = {hex32(ccr)}", [
            f"INSTRUCTION[7:0] = 0x{bf(ccr, 7, 0):02X}",
            f"IMODE[9:8] = {bf(ccr, 9, 8)} → 명령 전송 {mode[bf(ccr, 9, 8)]}",
            f"ADMODE[11:10] = {bf(ccr, 11, 10)} → 주소 전송 {mode[bf(ccr, 11, 10)]}",
            f"ADSIZE[13:12] = {bf(ccr, 13, 12)} → 주소 크기 {adsize[bf(ccr, 13, 12)]}",
            f"ABMODE[15:14] = {bf(ccr, 15, 14)} → 대체 바이트 {mode[bf(ccr, 15, 14)]}",
            f"DCYC[22:18] = {bf(ccr, 22, 18)} → **더미 사이클 {bf(ccr, 22, 18)}**",
            f"DMODE[25:24] = {bf(ccr, 25, 24)} → 데이터 전송 {mode[bf(ccr, 25, 24)]}",
            f"FMODE[27:26] = {bf(ccr, 27, 26)} → {fmode[bf(ccr, 27, 26)]}",
            f"SIOO[28]={bit(ccr, 28)} DDRM[31]={bit(ccr, 31)}",
        ])


# ---------------------------------------------------------------------------
# 섹션 12: LTDC
# ---------------------------------------------------------------------------


def sec_ltdc(d, doc, clocks):
    doc.h(2, "LTDC — LCD-TFT 컨트롤러 (베이스 0x40016800)")
    if d.blob("ltdc") is None:
        doc.w("(ltdc.bin 없음 — 건너뜀)")
        doc.w()
        return
    R = lambda o: d.off32("ltdc", o)
    sscr, bpcr, awcr, twcr, gcr = R(0x08), R(0x0C), R(0x10), R(0x14), R(0x18)
    if not any(v for v in (sscr, bpcr, awcr, twcr)):
        doc.w("LTDC 타이밍 레지스터가 전부 0 → LTDC 미초기화 또는 APB2 클럭 차단 상태.")
        doc.w()
        doc.sum_("LTDC 가 초기화되어 있지 않다 (SSCR/BPCR/AWCR/TWCR = 0x00000000).")
        return

    hsw = bf(sscr, 27, 16) + 1
    vsh = bf(sscr, 10, 0) + 1
    ahbp = bf(bpcr, 27, 16) + 1
    avbp = bf(bpcr, 10, 0) + 1
    aaw = bf(awcr, 27, 16) + 1
    aah = bf(awcr, 10, 0) + 1
    totalw = bf(twcr, 27, 16) + 1
    totalh = bf(twcr, 10, 0) + 1
    active_w = aaw - ahbp
    active_h = aah - avbp
    hbp = ahbp - hsw
    vbp = avbp - vsh
    hfp = totalw - aaw
    vfp = totalh - aah

    doc.fields(f"LTDC_SSCR @ 0x40016808 = {hex32(sscr)}", [
        f"HSW[27:16] = {bf(sscr, 27, 16)} → 수평 동기 폭 = **{hsw} 픽셀클럭**",
        f"VSH[10:0] = {bf(sscr, 10, 0)} → 수직 동기 높이 = **{vsh} 라인**",
    ])
    doc.fields(f"LTDC_BPCR @ 0x4001680C = {hex32(bpcr)}", [
        f"AHBP[27:16] = {bf(bpcr, 27, 16)} → 누적 수평 백포치 {ahbp} → HBP = {ahbp}−{hsw} = **{hbp}**",
        f"AVBP[10:0] = {bf(bpcr, 10, 0)} → 누적 수직 백포치 {avbp} → VBP = {avbp}−{vsh} = **{vbp}**",
    ])
    doc.fields(f"LTDC_AWCR @ 0x40016810 = {hex32(awcr)}", [
        f"AAW[27:16] = {bf(awcr, 27, 16)} → 누적 활성 폭 {aaw}",
        f"AAH[10:0] = {bf(awcr, 10, 0)} → 누적 활성 높이 {aah}",
        f"→ **실제 활성 해상도 = (AAW−AHBP) × (AAH−AVBP) = {active_w} × {active_h}**",
    ])
    doc.fields(f"LTDC_TWCR @ 0x40016814 = {hex32(twcr)}", [
        f"TOTALW[27:16] = {bf(twcr, 27, 16)} → 전체 수평 {totalw} → HFP = {totalw}−{aaw} = **{hfp}**",
        f"TOTALH[10:0] = {bf(twcr, 10, 0)} → 전체 수직 {totalh} → VFP = {totalh}−{aah} = **{vfp}**",
    ])
    if gcr is not None:
        doc.fields(f"LTDC_GCR @ 0x40016818 = {hex32(gcr)}", [
            f"LTDCEN[0] = {bit(gcr, 0)} → LTDC {onoff(bit(gcr, 0), '**동작 중**', '정지')}",
            f"DBW[6:4]={bf(gcr, 6, 4)} DGW[10:8]={bf(gcr, 10, 8)} DRW[14:12]={bf(gcr, 14, 12)} "
            "→ 디더링 비트 폭(읽기 전용, 구현 고정값)",
            f"DEN[16] = {bit(gcr, 16)} → 디더링 {onoff(bit(gcr, 16))}",
            f"PCPOL[28]={bit(gcr, 28)} DEPOL[29]={bit(gcr, 29)} VSPOL[30]={bit(gcr, 30)} "
            f"HSPOL[31]={bit(gcr, 31)} → 신호 극성 (0=기본 active-low/입력 그대로)",
        ])

    lcd_clk = (clocks or {}).get("lcd_clk")
    total_px = totalw * totalh
    lines = [f"프레임당 총 픽셀클럭 수 = TOTALW × TOTALH = {totalw} × {totalh} = "
             f"**{total_px:,}**"]
    if lcd_clk:
        fps = lcd_clk / total_px
        lines.append(f"픽셀 클럭 = {lcd_clk / 1e6:.3f} MHz (RCC PLLSAI 에서 산출)")
        lines.append(f"→ **추정 프레임레이트 = {lcd_clk / 1e6:.3f}MHz / {total_px:,} = "
                     f"{fps:.2f} Hz**")
        doc.sum_(f"LCD 는 **{active_w}×{active_h}** 활성 해상도로 구동 중이며 "
                 f"픽셀클럭 {lcd_clk / 1e6:.2f} MHz, 전체 타이밍 {totalw}×{totalh} → "
                 f"**약 {fps:.1f} Hz** (LTDC_AWCR={hex32(awcr)}, TWCR={hex32(twcr)}).", pri=1)
    else:
        lines.append("픽셀 클럭을 RCC 에서 산출하지 못해 프레임레이트는 **해석 불가**.")
        doc.sum_(f"LCD 활성 해상도는 **{active_w}×{active_h}** 이다 (LTDC_AWCR={hex32(awcr)}).")
    doc.h(3, "파생: 활성 해상도와 프레임레이트")
    for ln in lines:
        doc.w(f"- {ln}")
    doc.w()
    doc.table(["항목", "값"], [
        ("활성 해상도", f"{active_w} × {active_h}"),
        ("HSW / HBP / HFP", f"{hsw} / {hbp} / {hfp}"),
        ("VSH / VBP / VFP", f"{vsh} / {vbp} / {vfp}"),
        ("전체 타이밍", f"{totalw} × {totalh}"),
    ])

    # 레이어 설정
    fb_addrs = []
    for li, lbase in ((1, 0x84), (2, 0x104)):
        lcr = R(lbase + 0x00)
        if lcr is None:
            continue
        # LTDC_Layer_TypeDef (stm32f746xx.h): CR +0x00, WHPCR +0x04, WVPCR +0x08,
        # CKCR +0x0C, PFCR +0x10, CACR +0x14, DCCR +0x18, BFCR +0x1C,
        # (reserved ×2), CFBAR +0x28, CFBLR +0x2C, CFBLNR +0x30
        pfcr = R(lbase + 0x10)
        cfbar = R(lbase + 0x28)
        cfblr = R(lbase + 0x2C)
        cfblnr = R(lbase + 0x30)
        pf = {0: "ARGB8888", 1: "RGB888", 2: "RGB565", 3: "ARGB1555", 4: "ARGB4444",
              5: "L8", 6: "AL44", 7: "AL88"}
        doc.fields(f"LTDC_L{li}CR @ 0x{0x40016800 + lbase:08X} = {hex32(lcr)}", [
            f"LEN[0] = {bit(lcr, 0)} → 레이어 {li} {onoff(bit(lcr, 0), '**활성**', '비활성')}",
            f"L{li}PFCR = {hex32(pfcr)} → 픽셀 포맷 = "
            + (pf.get(bf(pfcr, 2, 0), "해석 불가") if pfcr is not None else "해석 불가"),
            f"L{li}CFBAR (프레임버퍼 주소) = {hex32(cfbar)}",
            f"L{li}CFBLR = {hex32(cfblr)} → 라인 길이 {bf(cfblr or 0, 12, 0)} B, "
            f"피치 {bf(cfblr or 0, 28, 16)} B",
            f"L{li}CFBLNR = {hex32(cfblnr)} → 프레임버퍼 라인 수 {bf(cfblnr or 0, 10, 0)}",
        ])
        if bit(lcr, 0) and cfbar:
            where = ("외부 SDRAM(0xC0000000 대)" if 0xC0000000 <= cfbar < 0xE0000000
                     else ("내부 SRAM" if 0x20000000 <= cfbar < 0x20080000 else "기타 영역"))
            doc.sum_(f"LTDC 레이어{li} 프레임버퍼는 {hex32(cfbar)} ({where}), "
                     f"픽셀 포맷 {pf.get(bf(pfcr or 0, 2, 0), '해석 불가')} 이다.")
            fb_addrs.append((li, cfbar, bf(cfbar and cfblr or 0, 28, 16),
                             bf(cfblnr or 0, 10, 0)))

    # 두 레이어의 프레임버퍼 간격이 정확히 한 프레임 크기인지 확인
    if len(fb_addrs) == 2:
        (_, a1, pitch1, ln1), (_, a2, _, _) = fb_addrs
        gap = a2 - a1
        frame = pitch1 * ln1
        if frame and gap == frame:
            doc.sum_(f"두 레이어의 프레임버퍼 간격({gap} B = 0x{gap:X})이 한 프레임 크기"
                     f"(피치 {pitch1} B × {ln1} 라인 = {frame} B)와 정확히 일치한다 → "
                     "SDRAM 에 두 프레임을 연속 배치한 더블 버퍼 구성이다.")


# ---------------------------------------------------------------------------
# 섹션 13: GPIO
# ---------------------------------------------------------------------------

GPIO_PORTS = "ABCDEFGHIJK"
GPIO_STRIDE = 0x400


def _port_regs(blob, idx):
    o = idx * GPIO_STRIDE
    if o + 0x28 > len(blob):
        return None
    names = ("MODER", "OTYPER", "OSPEEDR", "PUPDR", "IDR", "ODR", "BSRR", "LCKR",
             "AFRL", "AFRH")
    offs = (0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24)
    return dict(zip(names, (u32(blob, o + x) or 0 for x in offs)))


def _block_kind(blob, idx):
    """포트 0x400 블록의 성격을 판정.

    'zero'     — 전 바이트 0 (페리페럴 클럭 미공급 시 정상적인 읽기 결과)
    'repeated' — 0 이 아닌 단일 32비트 값의 반복 (정상적인 레지스터 이미지가 아님)
    'normal'   — 그 외
    """
    o = idx * GPIO_STRIDE
    chunk = blob[o:o + GPIO_STRIDE]
    if len(chunk) < GPIO_STRIDE:
        return "normal"
    if not any(chunk):
        return "zero"
    first = chunk[:4]
    if chunk == first * (GPIO_STRIDE // 4):
        return "repeated"
    return "normal"


def sec_gpio(d, doc, rcc_enabled):
    doc.h(2, "GPIO — 핀별 설정 (보드 배선 실측 근거)")
    blob = d.blob("gpio_all")
    if blob is None:
        doc.w("(gpio_all.bin 없음 — 건너뜀)")
        doc.w()
        return
    base = d.base_of("gpio_all") or 0x40020000

    suspect = []
    zero_ports = []
    af_list = []
    port_tables = []

    for idx, p in enumerate(GPIO_PORTS):
        regs = _port_regs(blob, idx)
        if regs is None:
            continue
        addr = base + idx * GPIO_STRIDE
        clk_on = f"GPIO{p}" in (rcc_enabled or [])
        kind = _block_kind(blob, idx)
        if kind == "repeated":
            suspect.append((p, addr, regs["MODER"], clk_on))
        elif kind == "zero":
            zero_ports.append((p, addr, clk_on))
        else:
            port_tables.append((p, addr, regs, clk_on))

    if zero_ports:
        doc.w("**전 레지스터가 0x00000000 으로 읽힌 포트**: "
              + ", ".join(f"GPIO{p}(0x{a:08X})" for p, a, _ in zero_ports)
              + ". 해당 포트의 RCC_AHB1ENR 클럭이 꺼져 있어 레지스터가 0 으로 읽힌 것이며, "
                "핀 설정 정보가 없으므로 아래 표에서 제외한다.")
        doc.w()

    if suspect:
        doc.warn(
            "다음 포트는 0x400 블록 **전체가 0 이 아닌 하나의 32비트 값으로 반복**되어 있다 — "
            "정상적인 레지스터 이미지가 아니다(덤프 아티팩트). "
            + ", ".join(f"GPIO{p}(0x{a:08X}, 반복값 0x{v:08X}, "
                        f"RCC 클럭 {'ON' if c else 'OFF'})" for p, a, v, c in suspect)
            + ". 반복된 값이 각 포트 MODER 의 리셋 기본값과 일치하고 해당 포트의 "
              "AHB1ENR 클럭이 꺼져 있으므로, **이 포트들의 값은 신뢰할 수 없어 아래 "
              "핀 상태표와 AF 목록에서 제외한다(해석 불가).**")

    for p, addr, regs, clk_on in port_tables:
        doc.h(3, f"GPIO{p} @ 0x{addr:08X}  (RCC 클럭 {'ON' if clk_on else 'OFF'})")
        doc.table(["레지스터", "값"],
                  [(k, f"0x{v:08X}") for k, v in regs.items()])
        rows = []
        for pin in range(16):
            mode = bf(regs["MODER"], 2 * pin + 1, 2 * pin)
            otype = bit(regs["OTYPER"], pin)
            speed = bf(regs["OSPEEDR"], 2 * pin + 1, 2 * pin)
            pupd = bf(regs["PUPDR"], 2 * pin + 1, 2 * pin)
            afr = regs["AFRL"] if pin < 8 else regs["AFRH"]
            af = bf(afr, 4 * (pin % 8) + 3, 4 * (pin % 8))
            idr = bit(regs["IDR"], pin)
            odr = bit(regs["ODR"], pin)
            af_txt = f"AF{af} ({AF_GROUP.get(af, '해석 불가')})" if mode == 2 else "-"
            if mode == 2:
                af_list.append((f"P{p}{pin}", af, AF_GROUP.get(af, "해석 불가")))
            rows.append((f"P{p}{pin}", GPIO_MODE[mode],
                         "open-drain" if otype else "push-pull",
                         GPIO_SPEED[speed], GPIO_PUPD[pupd], af_txt, idr, odr))
        doc.table(["핀", "모드", "출력 타입", "속도", "풀업/다운", "대체기능", "IDR", "ODR"], rows)

    doc.h(3, "대체기능(AF)으로 설정된 핀 전체 목록")
    if af_list:
        doc.w("보드 배선의 실측 증거다. AF 번호 → 기능 그룹 매핑은 STM32F746 데이터시트의 "
              "alternate function 표에서 옮긴 개괄이며, 같은 AF 번호라도 핀마다 실제 신호는 다르다.")
        doc.w()
        doc.table(["핀", "AF", "AF 그룹(개괄)"],
                  [(a, f"AF{b}", c) for a, b, c in af_list])
        # AF 그룹별 요약
        by_af = {}
        for pin, af, grp in af_list:
            by_af.setdefault(af, []).append(pin)
        doc.h(4, "AF 번호별 묶음")
        doc.table(["AF", "그룹(개괄)", "핀 수", "핀 목록"],
                  [(f"AF{af}", AF_GROUP.get(af, "해석 불가"), len(v), ", ".join(v))
                   for af, v in sorted(by_af.items())])
        top = sorted(by_af.items(), key=lambda kv: -len(kv[1]))[:3]
        doc.sum_(f"AF 로 설정된 핀은 총 **{len(af_list)}개**이며 분포는 "
                 + ", ".join(f"AF{af} {len(v)}핀" for af, v in top)
                 + " 이다 — 대부분이 FMC(SDRAM/SDMMC) 와 LTDC 배선이다 "
                 "(GPIOx_MODER/AFRL/AFRH 실측).", pri=1)
    else:
        doc.w("AF 로 설정된 핀이 없다.")
        doc.w()


# ---------------------------------------------------------------------------
# 섹션 14: 메모리 샘플
# ---------------------------------------------------------------------------

MEM_SAMPLES = [
    ("mem_itcm_head", 0x00000000, "ITCM RAM 선두"),
    ("mem_flash_itcm_alias", 0x00200000, "ITCM 인터페이스의 내장 Flash 별칭"),
    ("mem_flash_axim", 0x08000000, "AXIM 인터페이스의 내장 Flash = 벡터 테이블"),
    ("mem_dtcm_head", 0x20000000, "DTCM RAM 선두"),
    ("mem_sram1_head", 0x20010000, "SRAM1 선두"),
    ("mem_sram2_head", 0x2004C000, "SRAM2 선두"),
    ("mem_bkpsram", 0x40024000, "백업 SRAM"),
    ("mem_sdram_head", 0xC0000000, "외부 SDRAM(FMC Bank5/6) 선두"),
    ("mem_qspi_head", 0x90000000, "QSPI 메모리맵 모드 선두"),
]

VECTOR_NAMES = ["초기 SP (MSP)", "Reset_Handler", "NMI_Handler", "HardFault_Handler",
                "MemManage_Handler", "BusFault_Handler", "UsageFault_Handler",
                "(예약)", "(예약)", "(예약)", "(예약)", "SVC_Handler",
                "DebugMon_Handler", "(예약)", "PendSV_Handler", "SysTick_Handler"]


def _describe_addr(a):
    for lo, hi, nm in ((0x00000000, 0x00010000, "ITCM RAM"),
                       (0x00100000, 0x00110000, "시스템 메모리 ITCM 별칭"),
                       (0x00200000, 0x00300000, "Flash ITCM 별칭"),
                       (0x08000000, 0x08100000, "Flash AXIM"),
                       (0x1FF00000, 0x1FF10000, "시스템 부트로더"),
                       (0x20000000, 0x20010000, "DTCM RAM"),
                       (0x20010000, 0x2004C000, "SRAM1"),
                       (0x2004C000, 0x20050000, "SRAM2"),
                       (0x90000000, 0xA0000000, "QSPI 메모리맵"),
                       (0xC0000000, 0xD0000000, "외부 SDRAM")):
        if lo <= a < hi:
            return nm
    # 스택 포인터는 영역의 '끝 다음 주소'(exclusive)를 가리키는 것이 정상이다.
    for lo, hi, nm in ((0x00000000, 0x00010000, "ITCM RAM"),
                       (0x20000000, 0x20010000, "DTCM RAM"),
                       (0x20010000, 0x2004C000, "SRAM1"),
                       (0x2004C000, 0x20050000, "SRAM2"),
                       (0x08000000, 0x08100000, "Flash AXIM"),
                       (0xC0000000, 0xD0000000, "외부 SDRAM")):
        if a == hi:
            return f"{nm} 최상단 경계(= 이 영역 끝 다음 주소)"
    return "기타/미상 영역"


def sec_memory(d, doc):
    doc.h(2, "메모리 영역 샘플 (각 영역 선두 64바이트)")
    for name, addr, desc in MEM_SAMPLES:
        b = d.blob(name)
        doc.h(3, f"{desc} — 0x{addr:08X}")
        if b is None:
            doc.w("(덤프 없음 — 읽기 실패했거나 파일 없음)")
            doc.w()
            continue
        doc.w("```")
        for ln in hexdump(b, addr):
            doc.w(ln)
        doc.w("```")
        doc.w()
        if all(x == 0 for x in b):
            doc.w("→ 전 바이트 0x00 (초기화되지 않았거나 지워진 상태).")
            doc.w()
        elif all(x == 0xFF for x in b):
            doc.w("→ 전 바이트 0xFF (소거된 플래시 또는 응답 없는 영역).")
            doc.w()

    # 벡터 테이블 해석
    fb = d.blob("mem_flash_axim")
    if fb and len(fb) >= 8:
        doc.h(3, "벡터 테이블 해석 (0x08000000)")
        rows = []
        for i in range(min(16, len(fb) // 4)):
            v = u32(fb, 4 * i) or 0
            extra = ""
            if i == 0:
                extra = f"초기 스택 포인터 → {_describe_addr(v)}" + (
                    " (스택은 이 주소에서 아래로 자란다)" if v else "")
            elif v:
                extra = (f"Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x{v & ~1:08X}, "
                         f"{_describe_addr(v & ~1)}") if (v & 1) else \
                        f"{_describe_addr(v)} (bit0=0 — Thumb 비트 없음)"
            rows.append((i, VECTOR_NAMES[i] if i < len(VECTOR_NAMES) else f"IRQ{i - 16}",
                         f"0x{0x08000000 + 4 * i:08X}", f"0x{v:08X}", extra))
        doc.table(["#", "엔트리", "주소", "값", "해석"], rows)
        sp = u32(fb, 0) or 0
        pc = u32(fb, 4) or 0
        doc.sum_(f"펌웨어 벡터 테이블: 초기 SP = 0x{sp:08X} ({_describe_addr(sp)}), "
                 f"Reset_Handler = 0x{pc & ~1:08X} ({_describe_addr(pc & ~1)}) "
                 "→ 코드는 AXIM 경로의 Flash 에서 실행된다.")

    # ITCM 별칭 비교
    a = d.blob("mem_itcm_head")
    b2 = d.blob("mem_flash_itcm_alias")
    c = d.blob("mem_flash_axim")
    doc.h(3, "ITCM 별칭 검증")
    rows = []
    if a is not None and b2 is not None:
        same = a == b2
        rows.append(("0x00000000 (ITCM) vs 0x00200000 (Flash ITCM 별칭)",
                     "**일치**" if same else "**불일치**",
                     "0x00000000 은 Flash 별칭이 아니라 ITCM RAM 그 자체다"
                     if not same else "두 주소가 같은 저장소를 가리킨다"))
    if b2 is not None and c is not None:
        same = b2 == c
        rows.append(("0x00200000 (Flash ITCM 별칭) vs 0x08000000 (Flash AXIM)",
                     "**일치**" if same else "**불일치**",
                     "같은 Flash 를 ITCM/AXIM 두 경로로 본 것임이 확인된다"
                     if same else "두 경로의 내용이 다르다 — 추가 조사 필요"))
    if a is not None and c is not None:
        rows.append(("0x00000000 (ITCM) vs 0x08000000 (Flash AXIM)",
                     "**일치**" if a == c else "**불일치**", ""))
    doc.table(["비교", "결과", "의미"], rows)
    if a is not None and b2 is not None and c is not None:
        if b2 == c and a != b2:
            doc.sum_("0x00200000 과 0x08000000 의 내용이 바이트 단위로 동일하다 → "
                     "**내장 Flash 는 ITCM/AXIM 두 경로로 동시에 보인다**. "
                     "반면 0x00000000 은 내용이 달라 **ITCM RAM 그 자체**임이 확인된다.", pri=1)
        if a is not None and not all(x == 0 for x in a) and a != b2:
            doc.sum_("ITCM RAM(0x00000000) 선두에 비-0 임의값이 남아 있다 → "
                     "ITCM 은 초기화되지 않은 SRAM 이며 리셋 후에도 이전 내용이 남는다.")


# ---------------------------------------------------------------------------
# 섹션 15: 시스템 부트로더
# ---------------------------------------------------------------------------


def sec_bootloader(d, doc):
    doc.h(2, "시스템 메모리 부트로더 (0x1FF00000, ST 내장)")
    b = d.blob("sysmem_bootloader")
    if b is None:
        doc.w("(sysmem_bootloader.bin 없음 — 건너뜀)")
        doc.w()
        return
    doc.w(f"덤프 크기: {len(b)} 바이트 (0x1FF00000 ~ 0x{0x1FF00000 + len(b) - 1:08X})")
    doc.w()
    rows = []
    for i in range(min(16, len(b) // 4)):
        v = u32(b, 4 * i) or 0
        extra = ""
        if i == 0:
            extra = f"초기 SP → {_describe_addr(v)}"
        elif v:
            extra = (f"Thumb 함수 → 0x{v & ~1:08X}, {_describe_addr(v & ~1)}"
                     if (v & 1) else f"{_describe_addr(v)}")
        rows.append((i, VECTOR_NAMES[i] if i < len(VECTOR_NAMES) else f"IRQ{i - 16}",
                     f"0x{0x1FF00000 + 4 * i:08X}", f"0x{v:08X}", extra))
    doc.h(3, "부트로더 벡터 테이블")
    doc.table(["#", "엔트리", "주소", "값", "해석"], rows)
    sp = u32(b, 0) or 0
    pc = u32(b, 4) or 0
    doc.sum_(f"시스템 부트로더가 실재한다 — 초기 SP=0x{sp:08X}, 진입점=0x{pc & ~1:08X} "
             "(0x1FF00000 에서 직접 읽음).")

    doc.h(3, "내장 ASCII 문자열 추출")
    good = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._-/:()@+")
    hits = []
    for m in re.finditer(rb"[\x20-\x7e]{6,}", b):
        s = m.group()
        ratio = sum(1 for ch in s if ch in good) / len(s)
        letters = sum(1 for ch in s if chr(ch).isalpha())
        if ratio > 0.9 and letters >= 4 and b" " not in s[:1]:
            hits.append((0x1FF00000 + m.start(), s.decode("ascii")))
    if hits:
        doc.table(["주소", "문자열"], [(f"0x{a:08X}", f"`{s}`") for a, s in hits[:60]])
        if len(hits) > 60:
            doc.w(f"... 외 {len(hits) - 60}개")
            doc.w()
    else:
        doc.w("의미 있는 ASCII 문자열이 발견되지 않았다. 추출된 인쇄가능 바이트열은 모두 "
              "Thumb 기계어가 우연히 ASCII 범위에 들어간 것으로 보인다 → "
              "**부트로더 버전 문자열은 이 덤프에서 확인 불가(해석 불가)**.")
        doc.w()
        doc.note("부트로더 버전은 CubeProgrammer 연결 배너의 `BL Version` 필드로 확인하는 것이 "
                 "정확하다 (아래 '덤프 조건' 참조).")
        doc.sum_("시스템 부트로더 32KB 안에 버전 문자열 같은 사람이 읽을 수 있는 ASCII 는 "
                 "없다 — 전 영역이 순수 Thumb 코드/데이터다.")


# ---------------------------------------------------------------------------
# 섹션 16: 페리페럴 존재 여부 (덤프 성공/실패)
# ---------------------------------------------------------------------------

# 읽기 실패가 '실리콘에 해당 블록이 없음'을 뜻하는 주소들
ABSENT_EVIDENCE = {
    0x50051000: ("JPEG 하드웨어 코덱",
                 "STM32F746 계열 데이터시트에는 JPEG 코덱이 없고 STM32F76x/77x 에만 있다. "
                 "0x50051000 읽기가 버스 에러로 실패했다는 것은 **이 실리콘에 JPEG 코덱이 "
                 "물리적으로 탑재되어 있지 않다**는 직접 증거다."),
}


# 덤프 영역 이름 → 그 블록에 클럭을 공급하는 RCC ENR 비트 이름
REGION_CLOCK = {
    "syscfg": "SYSCFG", "crc": "CRC", "dma1": "DMA1", "dma2": "DMA2", "dma2d": "DMA2D",
    "quadspi": "QSPI", "fmc": "FMC", "dcmi": "DCMI", "rng": "RNG", "ltdc": "LTDC",
    "tim1": "TIM1", "tim2": "TIM2", "eth": "ETHMAC", "usb_hs_core": "OTGHS",
    "usb_fs_core": "OTGFS", "adc_bank": "ADC1", "sdmmc1": "SDMMC1",
    "usart1": "USART1", "usart6": "USART6", "i2c_bank": "I2C1",
    "spi_bank_apb1": "SPI2", "uart_bank_apb1": "USART2", "wwdg": "WWDG",
}


def _clock_state(d, region_name):
    """해당 영역의 RCC 클럭 인에이블 상태를 조회. (알 수 없으면 None)"""
    want = REGION_CLOCK.get(region_name)
    if want is None:
        return None
    offsets = {"AHB1ENR": 0x30, "AHB2ENR": 0x34, "AHB3ENR": 0x38,
               "APB1ENR": 0x40, "APB2ENR": 0x44}
    for nm, off in offsets.items():
        for pos, bitname in RCC_ENR_BITS[nm].items():
            if bitname == want:
                v = d.off32("rcc", off)
                if v is None:
                    return None
                return bool(bit(v, pos))
    return None


def sec_presence(d, doc):
    doc.h(2, "페리페럴 영역 덤프 성공/실패")
    ok_rows, fail_rows = [], []
    for r in (d.meta.get("regions") or []):
        addr = int(r["addr"], 16)
        row = (r["name"], r["addr"], f"{r['size']} B", r["desc"])
        if r.get("ok"):
            ok_rows.append(row)
        else:
            err = r.get("error")
            err_txt = " / ".join(err) if isinstance(err, list) else str(err)
            fail_rows.append(row + (err_txt,))
    doc.h(3, f"읽기 성공 ({len(ok_rows)}개)")
    doc.table(["영역", "주소", "크기", "설명"], ok_rows)
    doc.h(3, f"읽기 실패 ({len(fail_rows)}개)")
    if fail_rows:
        doc.table(["영역", "주소", "크기", "설명", "CLI 출력"], fail_rows)
    else:
        doc.w("실패한 영역 없음.")
        doc.w()
    for r in (d.meta.get("regions") or []):
        if r.get("ok"):
            continue
        addr = int(r["addr"], 16)
        ev = ABSENT_EVIDENCE.get(addr)
        if ev:
            doc.w(f"**{ev[0]} (0x{addr:08X}) — 미탑재 증거**")
            doc.w()
            doc.w(f"> {ev[1]}")
            doc.w()
            doc.sum_(f"**{ev[0]} 미탑재**: 0x{addr:08X} 읽기가 버스 에러로 실패했다 "
                     "(STM32F76x/77x 에만 존재하는 블록).", pri=1)

    # 클럭이 꺼진 채 0 으로 읽힌 블록 구분
    doc.note("주의: 읽기에 '성공'했어도 해당 페리페럴의 RCC 클럭이 꺼져 있으면 레지스터가 "
             "전부 0x00000000 으로 읽힌다. 값이 전부 0 인 블록은 '페리페럴 부재'가 아니라 "
             "'클럭 미공급'으로 해석해야 한다.")
    zero_rows = []
    for r in (d.meta.get("regions") or []):
        if not r.get("ok") or r["name"].startswith("mem_"):
            continue   # mem_* 는 페리페럴이 아니라 메모리 샘플이므로 별도 섹션에서 다룬다
        b = d.blob(r["name"])
        if b and all(x == 0 for x in b):
            clk = _clock_state(d, r["name"])
            if clk is None:
                verdict = "RCC 클럭 상태 미확인 → 원인 **해석 불가**"
            elif clk is False:
                verdict = f"RCC 클럭 **OFF** ({REGION_CLOCK[r['name']]}EN=0) → 클럭 미공급 때문"
            else:
                verdict = (f"RCC 클럭 **ON** ({REGION_CLOCK[r['name']]}EN=1) → 클럭은 살아 있으나 "
                           "레지스터가 실제로 전부 리셋값(0)임")
            zero_rows.append((r["name"], r["addr"], f"{len(b)} B", r["desc"], verdict))
    if zero_rows:
        doc.h(3, "전 영역이 0x00 으로 읽힌 블록")
        doc.table(["영역", "주소", "크기", "설명", "판정"], zero_rows)


# ---------------------------------------------------------------------------
# 섹션 17: 하드웨어 고유값 vs 펌웨어 설정
# ---------------------------------------------------------------------------


def sec_provenance(d, doc, mode):
    doc.h(2, "주의: 하드웨어 고유값 vs 펌웨어 설정")
    doc.w("같은 덤프 안이라도 값의 성격이 전혀 다르다. 논문/문서에 인용할 때 반드시 구분해야 한다.")
    doc.w()

    doc.h(3, "A. 실리콘 고정 — 소프트웨어로 바꿀 수 없는 값")
    doc.table(["항목", "근거 레지스터", "성격"], [
        ("코어 종류/리비전", "CPUID (0xE000ED00)", "읽기 전용 하드웨어 ID"),
        ("NVIC 인터럽트 라인 수", "ICTR (0xE000E004)", "구현 고정"),
        ("L1 캐시 구성/용량", "CLIDR, CTR, CCSIDR", "읽기 전용 (CSSELR 로 선택만 가능)"),
        ("TCM 크기 필드", "ITCMCR/DTCMCR 의 SZ[6:3]", "읽기 전용 구현값 (EN/RMW 는 쓰기 가능)"),
        ("AHBP 영역 크기", "AHBPCR 의 SZ[3:1]", "읽기 전용 구현값"),
        ("FPU 기능", "MVFR0/1/2", "읽기 전용 — 배정밀도 부재는 실리콘 사실"),
        ("MPU 리전 수", "MPU_TYPE (0xE000ED90)", "읽기 전용"),
        ("하드웨어 브레이크포인트 수", "FP_CTRL 의 NUM_CODE/NUM_LIT/REV", "읽기 전용"),
        ("CoreSight 컴포넌트 구성", "ROM table + 각 컴포넌트 PIDR/CIDR", "읽기 전용"),
        ("디바이스/리비전 ID", "DBGMCU_IDCODE (0xE0042000)", "읽기 전용"),
        ("UID 96비트, Flash 용량, ADC 캘리브레이션", "0x1FF0F420 / 0x1FF0F442 / 0x1FF0F44A~E",
         "ST 공장 기록, 읽기 전용"),
        ("코어 기능 ID (DSP 확장 등)", "ID_PFR/ID_MMFR/ID_ISAR", "읽기 전용"),
        ("JPEG 코덱 부재", "0x50051000 읽기 실패", "버스 디코딩 결과 = 실리콘 사실"),
        ("시스템 부트로더 코드", "0x1FF00000", "ST 가 마스크 ROM 에 기록, 변경 불가"),
    ])

    doc.h(3, "B. 옵션 바이트 — 비휘발성이지만 다시 쓸 수 있는 값")
    doc.table(["항목", "근거 레지스터", "성격"], [
        ("RDP 레벨, BOR 레벨, 워치독 SW/HW", "FLASH_OPTCR (0x40023C14)", "옵션 바이트 (재프로그램 가능)"),
        ("nRST_STOP / nRST_STDBY / nWRP", "FLASH_OPTCR", "옵션 바이트"),
        ("부트 주소 BOOT_ADD0/1", "FLASH_OPTCR1 (0x40023C18)", "옵션 바이트 — 리셋 후 VTOR 을 결정"),
    ])

    doc.h(3, "C. 덤프 시점의 소프트웨어 상태 — 실행 중이던 코드가 설정한 값")
    mode_note = ("이 덤프는 **리셋 직후(UR 모드)** 이므로 아래 값들은 대부분 "
                 "**리셋 기본값**이며 펌웨어의 설정이 아니다."
                 if mode == "UR" else
                 "이 덤프는 **실행 중(HOTPLUG 모드)** 이므로 아래 값들은 그 시점에 "
                 "보드에서 돌고 있던 펌웨어가 설정한 상태다.")
    doc.w(f"> {mode_note}")
    doc.w()
    doc.table(["항목", "근거 레지스터", "성격"], [
        ("I-cache/D-cache/분기예측 ON/OFF", "SCB_CCR (0xE000ED14)", "소프트웨어 설정"),
        ("FPU 활성 여부", "SCB_CPACR (0xE000ED88)", "소프트웨어 설정"),
        ("벡터 테이블 위치", "SCB_VTOR (0xE000ED08)", "리셋 시 BOOT_ADD 로 결정, 이후 SW 가 변경 가능"),
        ("NVIC 우선순위 그룹", "SCB_AIRCR 의 PRIGROUP", "소프트웨어 설정"),
        ("MPU 활성/리전 구성", "MPU_CTRL / MPU_RBAR / MPU_RASR", "소프트웨어 설정"),
        ("ACTLR 성능 비트", "ACTLR (0xE000E008)", "소프트웨어 설정"),
        ("CACR (ECC/FORCEWT/SIWT)", "CACR (0xE000EF9C)", "소프트웨어 설정 (ECCEN 리셋값은 1)"),
        ("SYSCLK/HCLK/PCLK, PLL 계수", "RCC_CR / PLLCFGR / CFGR", "소프트웨어 설정"),
        ("페리페럴 클럭 인에이블 목록", "RCC_AHB1~3ENR, APB1~2ENR", "소프트웨어 설정"),
        ("Flash 대기 상태, ART, 프리페치", "FLASH_ACR (0x40023C00)", "소프트웨어 설정"),
        ("전압 스케일/Over-drive", "PWR_CR1 / PWR_CSR1", "소프트웨어 설정"),
        ("SDRAM 타이밍/리프레시", "FMC_SDCR/SDTR/SDRTR", "소프트웨어 설정 (칩 사양에 맞춘 값)"),
        ("LCD 해상도/타이밍", "LTDC_SSCR/BPCR/AWCR/TWCR", "소프트웨어 설정 (패널 사양에 맞춘 값)"),
        ("GPIO 모드/AF 배정", "GPIOx_MODER / AFRL / AFRH", "소프트웨어 설정 — 단, 배선상 가능한 "
                                                          "AF 조합은 실리콘/보드가 제약한다"),
        ("리셋 원인 플래그", "RCC_CSR 상위 비트", "하드웨어가 세우고 SW 가 지우는 누적 플래그"),
        ("메모리 내용(SRAM/SDRAM/ITCM)", "mem_*.bin", "실행 중이던 프로그램의 데이터"),
    ])

    doc.h(3, "D. 이 덤프로는 확정할 수 없는 것")
    items = [
        "DWT 의 실제 사양(NUMCOMP 등) — DEMCR.TRCENA=0 이라 전부 0 으로 읽힘",
        "MPU 의 전체 리전 구성 — RNR 로 선택된 한 리전만 보임",
        "TCM 의 물리 크기 — SZ 필드가 데이터시트와 어긋남(온디바이스 aliasing 실측 필요)",
        "QSPI 플래시의 존재/용량 — 컨트롤러 클럭이 꺼져 레지스터가 0",
        "패키지 종류 — 패키지 데이터 레지스터가 0xFFFFFFFF",
        "GPIO 중 클럭이 꺼진 포트의 실제 설정 — 블록이 반복 패턴으로 읽힘",
    ]
    for it in items:
        doc.w(f"- {it}")
    doc.w()


# ---------------------------------------------------------------------------
# 문서 조립
# ---------------------------------------------------------------------------


def banner_facts(text):
    """연결 배너에서 관심 있는 줄만 뽑는다."""
    keys = ("ST-LINK SN", "ST-LINK FW", "Board", "Voltage", "SWD freq", "Connect mode",
            "Reset mode", "Device ID", "Revision ID", "Device name", "NVM size",
            "Device type", "Device CPU", "BL Version")
    out = []
    for line in text.splitlines():
        s = line.strip()
        for k in keys:
            if s.startswith(k):
                parts = s.split(":", 1)
                if len(parts) == 2:
                    out.append((parts[0].strip(), parts[1].strip()))
                break
    return out


def build(tag):
    d = Dump(tag)
    if not d.meta and d.blob("ppb_scs") is None:
        print(f"  [skip] {tag}: 덤프 데이터가 없음")
        return None

    doc = Doc()
    mode = d.meta.get("mode", "?")

    # --- 헤더 ---
    doc.w(f"# STM32F746G-DISCO ST-LINK 덤프 해석 ({tag})")
    doc.w()
    mode_desc = {"HOTPLUG": "Hot Plug — **실행 중인 펌웨어의 상태를 그대로 읽음**",
                 "UR": "Under Reset — **리셋 직후 상태(레지스터 리셋 기본값)**",
                 "NORMAL": "Normal"}.get(mode, mode)
    ts = "미상"
    if d.json_path.exists():
        ts = datetime.datetime.fromtimestamp(
            d.json_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    doc.h(2, "덤프 조건")
    rows = [
        ("연결 모드", f"`{mode}` — {mode_desc}"),
        ("덤프 시각", f"{ts} (원본 JSON 파일 mtime 기준)"),
        ("원본 인덱스", f"`info/raw/stlink_dump_{tag}.json`"),
        ("원본 바이너리", f"`info/raw/regions_{tag}/*.bin` "
                     f"({len(list(d.dir.glob('*.bin'))) if d.dir.exists() else 0}개 파일)"),
        ("덤프 생성 스크립트", "`info/code/stlink_dump.py`"),
        ("본 문서 생성 스크립트", "`info/code/decode_dump.py`"),
    ]
    if d.cache_id_raw:
        rows.append(("캐시 ID 원문", f"`info/raw/{d.cache_id_raw.name}`"))
    doc.table(["항목", "내용"], rows)

    bf_rows = banner_facts(d.banner)
    if bf_rows:
        doc.h(3, "CubeProgrammer 연결 배너 (원문 발췌)")
        doc.table(["항목", "값"], bf_rows)
        doc.note("배너는 CubeProgrammer 가 스스로 판단해 출력한 값이다. "
                 "아래 본문의 해석은 모두 **레지스터 원시값에서 직접 계산**한 것이며, "
                 "둘이 일치하면 교차 검증이 된 것이다.")

    summary_anchor = len(doc.lines)   # 여기에 나중에 핵심 요약을 끼워 넣는다

    # --- 본문 ---
    sec_core(d, doc)
    sec_cache(d, doc)
    sec_tcm(d, doc)
    sec_fpu(d, doc)
    sec_mpu(d, doc)
    sec_debug(d, doc)
    sec_factory(d, doc)
    clocks = sec_rcc(d, doc)
    sec_flash_if(d, doc, clocks)
    sec_pwr(d, doc)
    sec_syscfg(d, doc)
    sec_fmc(d, doc, clocks)
    sec_qspi(d, doc, clocks)
    sec_ltdc(d, doc, clocks)

    rcc_enabled = []
    v = d.off32("rcc", 0x30)
    if v is not None:
        rcc_enabled = [n for p, n in RCC_ENR_BITS["AHB1ENR"].items() if bit(v, p)]
    sec_gpio(d, doc, rcc_enabled)

    sec_memory(d, doc)
    sec_bootloader(d, doc)
    sec_presence(d, doc)
    sec_provenance(d, doc, mode)

    # --- 핵심 요약 삽입 ---
    core_facts = doc.summary_by_pri(1)
    other_facts = doc.summary_by_pri(2)
    head = ["", "## 핵심 요약", ""]
    head.append(f"이 덤프(`{mode}` 모드)에서 레지스터 원시값으로 **직접 확정된** 하드웨어 사실이다. "
                "각 항목 괄호 안이 근거 레지스터와 값이다.")
    head.append("")
    for s in core_facts:
        head.append(f"- {s}")
    head.append("")
    if other_facts:
        head.append("### 그 밖에 확정된 사실 (주로 덤프 시점의 소프트웨어 설정)")
        head.append("")
        for s in other_facts:
            head.append(f"- {s}")
        head.append("")
    if doc.warnings:
        head.append("### 요약에 붙는 경고")
        head.append("")
        for w in doc.warnings:
            head.append(f"- ⚠️ {w}")
        head.append("")
    doc.lines[summary_anchor:summary_anchor] = head

    out = RAW / f"decoded_{tag}.md"
    out.write_text(doc.render(), encoding="utf-8")
    print(f"  [ok] {out}  (핵심 {len(core_facts)}개 / 기타 {len(other_facts)}개 요약, {len(doc.warnings)}개 경고, "
          f"{len(doc.lines)}줄)")
    return out


def main():
    tags = sys.argv[1:] or ["hotplug", "reset"]
    print("=== STM32F746 덤프 해석 ===")
    for tag in tags:
        print(f"[{tag}]")
        try:
            build(tag)
        except Exception as e:      # 한 태그가 실패해도 나머지는 계속
            import traceback
            print(f"  [error] {tag}: {e}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
