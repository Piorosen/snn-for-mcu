#!/usr/bin/env python3
"""
호스트 쪽에서 얻을 수 있는 하드웨어 관련 정보를 모은다.

보드에 직접 물어볼 수 없지만 신뢰도가 높은 정보들이다:
  - 컴파일러가 이 타깃에 대해 켜는 아키텍처 기능 매크로
    (__ARM_FEATURE_DSP 같은 매크로는 "이 CPU에 DSP 확장이 있다"를 툴체인이 인정한다는 뜻)
  - ST가 배포한 CMSIS 디바이스 헤더에 박혀 있는 IRQ 번호표와 페리페럴 베이스 주소
    (벤더가 직접 기술한 값이라 데이터시트와 같은 급이다)
  - ST-LINK / USB 장치 정보
  - 이미 빌드된 SNN 펌웨어의 메모리 사용량

출력: info/raw/host_side.json 과 info/raw/host_side.md
"""

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent            # snn-for-mcu
RAW = HERE.parent / "raw"
IC = ROOT / "ImageClassification"

TOOLCHAIN_BIN = Path(
    "/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/"
    "com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.macosaarch64_1.0.0.202602081740"
    "/tools/bin"
)
GCC = TOOLCHAIN_BIN / "arm-none-eabi-gcc"
MCU_FLAGS = ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-sp-d16", "-mfloat-abi=hard"]

HDR = IC / "Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h"


def sh(cmd, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)
        return (p.stdout + p.stderr)
    except Exception as e:
        return f"<실행 실패: {e}>"


def toolchain_info():
    out = {}
    out["gcc_version"] = sh([str(GCC), "--version"]).splitlines()[0] if GCC.exists() else "없음"
    out["target_triplet"] = sh([str(GCC), "-dumpmachine"]).strip()

    # 이 MCU 플래그 조합에서 컴파일러가 켜는 사전정의 매크로 중 아키텍처 관련만 추린다
    dm = sh([str(GCC)] + MCU_FLAGS + ["-dM", "-E", "-x", "c", "/dev/null"])
    keep = {}
    for line in dm.splitlines():
        m = re.match(r"#define\s+(\S+)\s*(.*)", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        if name.startswith(("__ARM_", "__VFP", "__THUMB", "__SOFTFP", "__CORTEX")):
            keep[name] = val
    out["arch_macros"] = dict(sorted(keep.items()))

    # 멀티립: 이 플래그 조합에 실제로 어떤 라이브러리 변형이 선택되는가
    out["multilib_selected"] = sh([str(GCC)] + MCU_FLAGS + ["-print-multi-directory"]).strip()
    libc = sh([str(GCC)] + MCU_FLAGS + ["--specs=nano.specs", "-print-file-name=libc_nano.a"]).strip()
    out["libc_nano_path"] = libc
    return out


def cmsis_header_info():
    """벤더 헤더에서 IRQ 번호표와 페리페럴 베이스 주소를 뽑는다."""
    if not HDR.exists():
        return {"error": f"헤더 없음: {HDR}"}
    text = HDR.read_text(errors="replace")
    info = {}

    # IRQn_Type 열거형: 이 실리콘에 실제로 배선된 인터럽트 목록
    m = re.search(r"typedef enum\s*\{(.*?)\}\s*IRQn_Type;", text, re.S)
    irqs = []
    if m:
        for line in m.group(1).splitlines():
            mm = re.match(r"\s*([A-Za-z0-9_]+)_IRQn\s*=\s*(-?\d+)\s*,?\s*(?:/\*!<(.*?)\*/)?", line)
            if mm:
                irqs.append({
                    "name": mm.group(1),
                    "number": int(mm.group(2)),
                    "desc": (mm.group(3) or "").strip(),
                })
    info["irqs"] = irqs
    info["irq_count_total"] = len(irqs)
    info["irq_count_peripheral"] = sum(1 for i in irqs if i["number"] >= 0)
    info["irq_max_number"] = max((i["number"] for i in irqs), default=None)

    # 페리페럴 베이스 주소
    bases = []
    for mm in re.finditer(r"#define\s+([A-Z0-9_]+)_BASE\s+\(?\s*([^)\n]+?)\s*\)?\s*(?:/\*|$)",
                          text, re.M):
        bases.append({"name": mm.group(1), "expr": mm.group(2).strip()})
    info["base_defines_count"] = len(bases)

    # 숫자 리터럴로 바로 나오는 메모리 영역 정의만 별도로
    mem = {}
    for name in ["FLASH_BASE", "FLASH_END", "FLASH_OTP_BASE", "FLASH_OTP_END",
                 "SRAM1_BASE", "SRAM2_BASE", "DTCMRAMBASE", "DTCMRAM_BASE",
                 "ITCMRAM_BASE", "BKPSRAM_BASE", "PERIPH_BASE", "QSPI_BASE",
                 "FMC_R_BASE", "SRAM1_BB_BASE", "RAMITCM_BASE", "FLASHITCM_BASE"]:
        mm = re.search(rf"#define\s+{name}\s+\(?\s*(0x[0-9A-Fa-f]+)", text)
        if mm:
            mem[name] = mm.group(1)
    info["memory_defines"] = mem

    # 이 헤더가 선언한 페리페럴 인스턴스 (TIM1, USART3 ... 형태)
    insts = sorted(set(re.findall(r"#define\s+([A-Z][A-Z0-9_]*\d*)\s+\(\([A-Za-z_]+_TypeDef\s*\*\)",
                                  text)))
    info["peripheral_instances"] = insts
    info["peripheral_instance_count"] = len(insts)
    return info


def usb_and_stlink():
    out = {}
    sp = sh(["system_profiler", "SPUSBDataType"])
    block = []
    grab = 0
    for line in sp.splitlines():
        if "STM32" in line or "ST-LINK" in line or "STMicro" in line:
            grab = 22
        if grab > 0:
            block.append(line.rstrip())
            grab -= 1
    out["usb_stlink_block"] = "\n".join(block) if block else "USB에서 ST-LINK를 찾지 못함"
    out["st_info_probe"] = sh(["st-info", "--probe"])
    out["serial_ports"] = sh(["bash", "-lc", "ls -la /dev/cu.* /dev/tty.usb* 2>/dev/null"])
    return out


def firmware_footprint():
    """이미 빌드된 SNN 펌웨어의 메모리 사용량 (하드웨어 예산 대비 실사용)."""
    out = {}
    for build in ["build", "build-release", "build-v4"]:
        elf = IC / build / "ImageClassification.elf"
        if not elf.exists():
            continue
        size = sh([str(TOOLCHAIN_BIN / "arm-none-eabi-size"), "-A", str(elf)])
        out[build] = size
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    data = {
        "toolchain": toolchain_info(),
        "cmsis_header": cmsis_header_info(),
        "usb": usb_and_stlink(),
        "firmware_footprint": firmware_footprint(),
    }
    (RAW / "host_side.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # 사람이 읽을 요약
    L = []
    L.append("# 호스트 측에서 수집한 하드웨어 관련 정보\n")
    L.append("데이터 출처는 툴체인·벤더 헤더·OS 장치 정보다. 보드에 직접 물어본 값이 아니므로")
    L.append("실측(신뢰도 A)과는 구분해서 읽어야 한다.\n")

    t = data["toolchain"]
    L.append("\n## 툴체인이 인식하는 타깃 [신뢰도 B: GNU 툴체인 자체 출력]\n")
    L.append(f"- 컴파일러: `{t.get('gcc_version')}`")
    L.append(f"- 타깃 triplet: `{t.get('target_triplet')}`")
    L.append(f"- 이 MCU 플래그에서 선택되는 멀티립: `{t.get('multilib_selected')}`")
    L.append(f"- 빌드 플래그: `{' '.join(MCU_FLAGS)}`\n")
    L.append("\n### 컴파일러가 켜는 아키텍처 기능 매크로\n")
    L.append("이 매크로들은 툴체인이 이 CPU에 어떤 하드웨어 기능이 있다고 보는지를 그대로 드러낸다.\n")
    L.append("| 매크로 | 값 | 의미 |")
    L.append("|---|---|---|")
    MEANING = {
        "__ARM_FEATURE_DSP": "DSP(SIMD32) 확장 명령 사용 가능 — SMLAD/QADD16 등",
        "__ARM_FEATURE_SIMD32": "32비트 SIMD 명령 사용 가능",
        "__ARM_FEATURE_FMA": "융합 곱셈-덧셈(VFMA) 하드웨어 지원",
        "__ARM_FP": "하드웨어 부동소수점 정밀도 비트마스크 (4=단정밀도, 8=배정밀도)",
        "__ARM_ARCH": "ARM 아키텍처 세대",
        "__ARM_ARCH_7EM__": "ARMv7E-M (DSP 확장 포함 M 프로파일)",
        "__ARM_FEATURE_QBIT": "포화 연산 Q 플래그 지원",
        "__ARM_FEATURE_SAT": "포화 명령(SSAT/USAT) 지원",
        "__ARM_FEATURE_CLZ": "CLZ 명령 지원",
        "__ARM_FEATURE_UNALIGNED": "비정렬 접근 하드웨어 지원",
        "__ARM_FEATURE_LDREX": "배타적 로드/스토어 지원 폭",
        "__ARM_FP16_FORMAT_IEEE": "IEEE half-precision 형식",
        "__ARM_PCS_VFP": "VFP 레지스터로 부동소수점 인자 전달 (hard float ABI)",
        "__VFP_FP__": "VFP 부동소수점 형식 사용",
    }
    for k, v in t.get("arch_macros", {}).items():
        L.append(f"| `{k}` | `{v}` | {MEANING.get(k, '')} |")

    c = data["cmsis_header"]
    if "error" not in c:
        L.append("\n\n## 벤더 헤더가 기술한 실리콘 구성 [신뢰도 B: ST 배포 CMSIS 헤더]\n")
        L.append(f"- 인터럽트 벡터: 총 {c['irq_count_total']}개 항목 "
                 f"(그중 페리페럴 IRQ {c['irq_count_peripheral']}개, 최대 번호 {c['irq_max_number']})")
        L.append(f"- 페리페럴 인스턴스: {c['peripheral_instance_count']}개")
        L.append(f"- `_BASE` 정의: {c['base_defines_count']}개\n")
        L.append("\n### 메모리 영역 정의\n")
        L.append("| 심볼 | 주소 |")
        L.append("|---|---|")
        for k, v in c["memory_defines"].items():
            L.append(f"| `{k}` | `{v}` |")
        L.append("\n\n### 페리페럴 인스턴스 목록\n")
        L.append("```")
        insts = c["peripheral_instances"]
        for i in range(0, len(insts), 8):
            L.append("  ".join(f"{x:<14}" for x in insts[i:i + 8]))
        L.append("```")
        L.append("\n\n### 인터럽트 벡터표\n")
        L.append("| # | 이름 | 설명 |")
        L.append("|---|---|---|")
        for irq in c["irqs"]:
            L.append(f"| {irq['number']} | `{irq['name']}_IRQn` | {irq['desc']} |")

    L.append("\n\n## USB / ST-LINK [신뢰도 A: 호스트 OS가 보고한 실제 연결 상태]\n")
    L.append("```")
    L.append(data["usb"]["usb_stlink_block"])
    L.append("```\n")
    L.append("```")
    L.append(data["usb"]["st_info_probe"].strip())
    L.append("```\n")
    L.append("```")
    L.append(data["usb"]["serial_ports"].strip())
    L.append("```")

    if data["firmware_footprint"]:
        L.append("\n\n## 기존 SNN 펌웨어의 메모리 실사용량 [신뢰도 A: 빌드 산출물]\n")
        for k, v in data["firmware_footprint"].items():
            L.append(f"\n### `{k}/`\n")
            L.append("```")
            L.append(v.strip())
            L.append("```")

    (RAW / "host_side.md").write_text("\n".join(L))
    print(f"저장: {RAW / 'host_side.json'}")
    print(f"저장: {RAW / 'host_side.md'}")
    print(f"  아키텍처 매크로 {len(data['toolchain'].get('arch_macros', {}))}개")
    if "error" not in c:
        print(f"  IRQ {c['irq_count_total']}개, 페리페럴 인스턴스 {c['peripheral_instance_count']}개")


if __name__ == "__main__":
    main()
