# 09. ML/DSP 성능 레퍼런스 — STM32F746NGH6 (Cortex-M7 r0p1, Device ID 0x449 Rev Z)

> 대상: STM32F746G-DISCO / STM32F746NGH6, Arm Cortex-M7 (DSP 확장 有, MVE 無), 최대 216 MHz.
> 각 사실 뒤 대괄호는 신뢰도 등급이며, 문서번호·섹션 또는 URL을 함께 표기한다.
> **주의**: 본 프로젝트(`ImageClassification/`)의 실제 클럭 설정은 **200 MHz**(PLLM=25, PLLN=400, HCLK=200 MHz)이므로,
> 이론 피크는 216 MHz와 200 MHz 두 가지 모두 병기했다.

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거: `info/raw/measured_facts.md`, `info/results/probe_output.txt`, `info/raw/decoded_hotplug.md`, `info/raw/host_side.md`

### 실측으로 확인됨

- **칩/코어**: 문서 "Cortex-M7 r0p1 / DEV_ID 0x449 / Rev Z" → CPUID `0x410FC271`(r0p1), DBGMCU_IDCODE `0x10016449`(REV_ID 0x1001 = Rev Z). 일치.
- **Flash 1 MB**: 문서 1 MB → Flash size register `0x0400` = **1024 KB**. 일치.
- **동작 클럭 200 MHz**: 문서 "본 프로젝트 실제 설정은 200 MHz(PLLM=25, PLLN=400)" → RCC_PLLCFGR `0x29406419`(M=25/N=400/P=2), 펌웨어 보고 `sysclk_hz = 200000000`, PCLK1 50 MHz / PCLK2 100 MHz. **216 MHz가 아님이 실측으로 확정**. 일치.
- **L1 캐시 용량·구성**: 문서 "I 4 KB 2-way / D 4 KB 4-way(32 set) / 라인 32 B" → CCSIDR D=`0xF003E019`(32 set × 4 way × 32 B), I=`0xF007E009`(64 set × 2 way × 32 B), CTR=`0x8303C003`. **레지스터와 성능 곡선(4 KB에서 꺾임, 8 KB부터 미스) 양쪽에서 확인**. 완전 일치.
- **"16 KB+16 KB는 F76x/F77x다" 경고(§6.3)**: 실측 4 KB+4 KB로 문서의 경고가 옳다.
- **캐시라인 32 B**: 문서 32 B → 스트라이드 스윕에서 32 B부터 평탄화(SRAM1 9.03 cyc/접근). 일치.
- **TCM은 캐시되지 않음**: 문서 주장 → D-cache OFF/WB/WT 세 조건에서 ITCM·DTCM 대역폭이 완전히 동일(226.17 / 228.26 MB/s). 일치.
- **TCM 저지연**: 문서 "DTCM은 0-wait" → 32 B 스트라이드 읽기 **DTCM 5.16 cyc vs SRAM1/SRAM2 15.66 cyc(3.0배)**. 방향 일치(다만 "0-wait"은 지연 0이 아니라 파이프라인 로드 지연 약 3 cyc를 의미).
- **메모리 맵/용량**: SRAM1 240 KB + SRAM2 16 KB(경계 `0x20050000`부터 폴트), DTCM 64 KB, ITCM 16 KB(`0x00004000`부터 폴트), BKPSRAM 4 KB, 내부 합계 **320 KB**. §6.4 전부 일치.
- **§3.4의 "SRAM 340 kB" 표기**: 320 KB + ITCM 16 KB + 백업 4 KB = **340 KB**로 정합. 모순 아님.
- **DSP 확장 有 / MVE 無 / SMLAD·SMLALD 사용 가능**: ID_ISAR3 SIMD=3, ID_ISAR2 MultS=3(SMLAD/SMUAD), `__ARM_FEATURE_DSP=1`, `__ARM_FEATURE_SIMD32=1`, 빌드 플래그 `-mcpu=cortex-m7 -mfpu=fpv5-sp-d16 -mfloat-abi=hard`. 일치. (FPU는 **FPv5-SP** — `double`은 소프트 에뮬레이션, MVFR0=`0x10110021`)
- **§2.1의 MAC 듀얼이슈 억제 비트(ACTLR bit19/24)**: 실측 `ACTLR = 0x00000000` → 억제 비트가 하나도 켜져 있지 않다(성능 우선 기본 상태). 문서가 말한 감속 요인은 이 보드에서 발동하지 않았다.
- **conv2 가중치 51,200 B = D-cache의 12.5배**: D-cache 4 KB가 실측 확정이므로 이 계산이 성립.
- **SDRAM 구성**: 문서 "MT48LC4M32B2, 하위 16비트만 연결, SDCLK = HCLK/2" → SDCR1 `0x00001954`: MWID=**16 bit**, 4 bank / 12 row / 8 col = **8 MB**, SDCLK=**HCLK/2=100 MHz**, RBURST=1. 부품 구성 일치.
- **에너지 추정에 쓴 3.3 V 가정(§5)**: VDDA 실측 **3.311 V**. 가정이 타당.
- **문서가 "미확인"으로 남긴 캐시 미스 페널티(§2.3, 미확인 #3)** → 실측으로 채워짐: **D-cache 히트 2.0 cyc / SRAM1 미스 8~9 cyc / SDRAM 랜덤 21~26 cyc / Flash AXIM 4 KB 스트라이드 14.34 cyc / Flash ITCM 별칭 9.13 cyc / QSPI 94~198 cyc**.

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

1. **ART Accelerator는 AXIM(0x08000000) 경로를 가속하지 않는다.**
   - 문서: §1 "Flash 실행 + ART Accelerator 0-wait", §6.1 "내장 Flash … AXIM → AXI-to-64-bit-AHB 브리지 + ART Accelerator".
   - 실측: AXIM 순차 읽기 **159.96 MB/s로 ART ON/OFF 완전 동일**. 반대로 **ITCM 별칭(0x00200000)은 336.79 → 492.10 MB/s(×1.46)**로 ART에 반응. AXIM은 D-cache에 반응(159.96 → 304.82, ×1.91), ITCM 별칭은 D-cache에 무반응.
   - 원인: ART는 **ITCM 인터페이스 경유 플래시 전용** 캐시/프리페치다. 문서가 ST 마케팅 문구("0-wait thanks to ART")를 버스 경로 구분 없이 옮겼다.
2. **이 보드의 §5 실측 수치는 ART·프리페치가 꺼진 상태에서 나온 값이다.**
   - 문서: §5 표(4,494 / 9,641 / 2,407 ms)에 플래시 가속 상태에 대한 언급이 없고, §1.1은 "Flash(ART ON, prefetch OFF)"를 전제로 전류를 인용.
   - 실측: `FLASH_ACR = 0x00000006` → **LATENCY=6 WS, PRFTEN=0, ARTEN=0**. 게다가 `BOOT_ADD0=0x80`이라 코드가 **ITCM 별칭에서 실행**되므로 I-cache도 걸리지 않는다(I-cache는 AXIM 접근에만 적용). 원인은 `stm32f7xx_hal_conf.h:150`의 `#define PREFETCH_ENABLE 0U`와 `ART_ACCLERATOR_ENABLE` 매크로 미정의.
   - 함의: 문서의 모든 로컬 실측 수치는 **플래시 페치 경로가 가속되지 않은 하한값**이다. ART만 켜도 그 경로가 1.46배 빨라진다.
3. **§6.1/§6.2 이론 대역폭은 실제 도달 대역폭과 크게 다르다 — 루프라인 결론이 흔들린다.**
   - 문서 이론(@200 MHz) vs 실측 최고치:

     | 메모리 | 문서 이론 | 실측 최고 읽기 | 실측 최고 쓰기/복사 |
     |---|---|---|---|
     | DTCM | 1,600 MB/s (8 B/cyc) | **457.5** (LDM, 2.29 B/cyc) | write 708.0 / copyLDM **1,409.4** (7.05 B/cyc) |
     | ITCM | 1,600 MB/s | **460.0** (2.30 B/cyc) | copyLDM **1,325.6** (6.63 B/cyc) |
     | SRAM1 | 800 MB/s (4 B/cyc) | **389.7** (1.95 B/cyc) | write 709.5 (3.55 B/cyc) |
     | Flash AXIM | 1,600 MB/s | **160.0**(D$ OFF) / **304.8~339.6**(D$ ON) | — |
     | Flash ITCM 별칭 | (문서에 항목 없음) | **492.1**(ART ON) / 336.8(ART OFF) | — |
     | SDRAM | 200 MB/s | **60.3** | write 194.6 |
     | QSPI | (문서에 항목 없음) | **51.7** | — |
   - **실측 쪽 해석의 한계도 명기한다**: 순차 읽기 루프는 ART OFF·6 WS 플래시에서 실행되었으므로 **명령 페치 바운드일 가능성이 크다**(모든 내부 메모리의 `read32`가 205~228 MB/s로 동일하게 눌린 것이 그 정황). 따라서 읽기 실측치는 **하한**이다. 반면 DTCM copyLDM 7.05 B/cyc는 이론 8 B/cyc에 근접하므로 **"DTCM 폭 8 B/cycle"이라는 문서 주장 자체는 반증되지 않았다.** 어긋나는 것은 "이론 폭"이 아니라 **"실제 코드가 그 폭을 쓸 수 있다"는 루프라인 전제**다.
   - 4 B/MAC 기준으로 실측 대역폭을 다시 넣으면: DTCM 114 MMAC/s, SRAM1 97, Flash(AXIM, D$ ON) 76, SDRAM 15 — 문서의 400 / 200 / 400 / 50 MMAC/s와 4~5배 차이.
4. **SDRAM 읽기 대역폭이 문서 이론값의 1/3이다.**
   - 문서: §6.1 "SDRAM 약 200~216 MB/s", §6.2 "SDRAM 최대 12.5 % = 54 MMAC/s".
   - 실측: **read32 60.25 MB/s(0.30 B/cyc)**, write32 194.55 MB/s, 랜덤 접근 21~26 cyc. → 읽기 기준 루프라인 knee는 **15 MMAC/s**.
   - 원인(실측 쪽 조건): 프로브 펌웨어가 **MPU를 설정하지 않아 0xC0000000이 ARMv7-M 기본 메모리맵상 Device 메모리**로 취급된다. 캐시·버스트가 걸리지 않아 읽기가 특히 불리하다. MPU로 Normal cacheable로 재정의하면 값이 올라갈 여지가 있으므로, **"SDRAM은 무조건 60 MB/s"라고 단정하지 말 것**. 다만 프레임버퍼가 SDRAM에 있는 이 보드의 기본 설정에서는 실측값이 현실이다.
5. **플래시 대기 상태: 문서의 "7 WS 추정"이 아니라 실측 6 WS다.**
   - 문서: §6.4 / 미확인 #2 "216 MHz에서 7 WS로 알려져 있으나 원문 미확인 [C]".
   - 실측: 200 MHz·VOS Scale 1에서 `FLASH_ACR.LATENCY = 6`. (216 MHz에서 7 WS인지는 이 보드가 200 MHz로 돌므로 여전히 미검증)
6. **ITCM은 "명령 전용"이 아니다.**
   - 문서: §6.1 "ITCM-RAM(16 KB) 명령 전용", §6.4 "ITCM-RAM 16 KB (CPU 실행 전용)".
   - 실측: ITCM에 대한 **데이터 읽기/쓰기가 정상 동작**한다(read32 226.17 MB/s, write32 263.91 MB/s, copyLDM 1,325.56 MB/s). ITCM은 데이터로도 접근 가능한 RAM이며, "명령 전용"은 용도 관례일 뿐 하드웨어 제약이 아니다.
7. **SDRAM CAS latency는 3이 아니라 2로 동작한다.**
   - 문서: §6.1 "`FMC.CASLatency1 = FMC_SDRAM_CAS_LATENCY_3`(.ioc)", §6.1 각주도 "CAS latency 3을 뺀 이론 상한".
   - 실측: SDCR1 CAS[8:7]=`0b10` → **CL2**. ST BSP `stm32746g_discovery_sdram.c:151`이 `FMC_SDRAM_CAS_LATENCY_2`로 초기화하고, 프로젝트의 `MX_FMC_Init()`은 `main.cc:128,201`에서 **주석 처리되어 있다**.
   - 원인: 문서가 **CubeMX `.ioc` 설정값을 실행 시 값으로 오인**했다. 실제 실행 경로는 BSP다.

### 아직 측정하지 않음

- **CoreMark / DMIPS(1082 / 462)를 이 보드에서 재현**: CoreMark를 ITCM 별칭 실행으로 빌드해 ART ON/OFF 각각 DWT CYCCNT로 측정.
- **SMLAD/SMLALD의 2 MAC/cycle 피크**: ITCM(`.itcm_code`) 상주 언롤 SMLAD 루프를 DWT CYCCNT로 재 IPC 산출.
- **216 MHz vs 200 MHz 비교(§1 병기값)**: PLLN=432로 재빌드해 ART·WS 조건을 고정한 채 동일 워크로드 재측정.
- **ART ON이 SNN 추론 시간에 주는 이득**: `ART_ACCLERATOR_ENABLE 1` / `PREFETCH_ENABLE 1`로 재빌드 후 V1~V4 재측정.
- **코드 배치별 실행 속도(§6.4, AN4667 대체)**: 핫 커널을 `.itcm_code` / AXIM Flash / SRAM1에 각각 배치해 동일 추론 시간 비교.
- **캐시 ON/OFF 실행시간 배수(미확인 #12)**: `SCB_DisableDCache()`/`SCB_DisableICache()` 후 같은 추론을 재측정.
- **D-cache 히트율·CPI 분해(§6.2 결론 4의 "히트율 실측치는 미확인")**: DWT 프로파일 카운터(`NOPRFCNT=0`으로 존재 확인됨 — CPICNT/LSUCNT/FOLDCNT)를 레이어별로 읽기.
- **§1.1 소비 전류(178/165/108 mA)**: VDD 공급 경로에 션트 저항 또는 전류계를 직렬로 넣고 Run 모드 전류 측정.
- **MPU로 SDRAM을 Normal cacheable로 지정했을 때의 대역폭**: MPU 리전 하나를 0xC0000000에 걸고 §6.1 표를 재측정(위 CONFLICT 4의 검증).
- **CMSIS-NN 레이어별 사이클 분해(§4 fast path vs 스칼라 폴백)**: `arm_convolve_fast_s16` / `arm_convolve_s16`을 각각 강제 호출해 DWT CYCCNT로 비교.
- **FC의 512열 하이브리드 폴백 비용(§4.3)**: `rhs_cols`를 511/512/513/1024로 바꿔가며 `arm_nn_vec_mat_mult_t_s16` 사이클 측정.

---

## 요약

- ST 공식 성능 수치: **1082 CoreMark / 462 DMIPS @ 216 MHz**(ART Accelerator + L1 캐시, Flash 실행). DMIPS/MHz = **2.14** (Dhrystone 2.1). [A/B]
- Cortex-M7은 **MAC 파이프라인이 1개**이며 "**one MAC per cycle throughput**, 2-cycle result latency"로 명시되어 있다 → 슈퍼스칼라여도 **MAC은 사이클당 1개**만 발행된다. [A: ARM DDI 0489F §1.2.1]
- `__SMLAD`/`__SMLALD`는 명령 1개당 signed 16×16 MAC **2회** → **이론 피크 2 MACs/cycle**. 216 MHz에서 **432 MMAC/s = 0.864 GOPS**, 200 MHz에서 **400 MMAC/s = 0.800 GOPS**. [D: A급 근거 조합]
- CMSIS-NN 원논문(arXiv 1801.06601)의 CIFAR-10 CNN은 **NUCLEO-F746ZG @216 MHz에서 99.1 ms/이미지(10.1 img/s), 24.7 M Ops(=12.3 M MACs) → 249 MOps/s**. 이를 MAC 기준으로 환산하면 **124 MMAC/s = 0.575 MAC/cycle = 이론 피크의 28.7 %**. [A + D]
- CMSIS-NN이 베이스라인(CMSIS-DSP `arm_conv` 기반) 대비 **runtime/throughput 4.6×, 에너지 효율 4.9×** 개선. 레이어별로는 conv 4.6×, pooling 5.4×, ReLU 2.6×. [A: arXiv 1801.06601 Table 2]
- **s16 DSP fast path 조건은 소스에서 확인됨**: `arm_convolve_wrapper_s16`는 `filter_w × filter_h × input_c < 512` **그리고** `dilation.w == dilation.h == 1`일 때만 `arm_convolve_fast_s16`(SMLAD)로 분기하고, 아니면 스칼라 C인 `arm_convolve_s16`으로 폴백한다. 512는 int32 누산기 오버플로 경계(32768×128×512 = 2³¹)에서 나온 값이다. [A(소스) + D(유도)]
- **본 프로젝트 실측**: conv2의 im2col 열 길이가 5×5×32 = **800 ≥ 512**라 fast path를 못 타서 CMSIS-NN 경로(V2)가 순수 스칼라(V1)보다 **2.1× 느렸다**(9,641 ms vs 4,494 ms/이미지). [D: 로컬 실측 EXPERIMENTS.md]
- 메모리 루프라인: DTCM 8 B/cycle(**1.73 GB/s @216 MHz**) → SRAM1(32-bit AHB) **864 MB/s** → DISCO SDRAM(16-bit @SDCLK=HCLK/2) **약 200~216 MB/s**. 재사용 없는 int16 MAC은 4 B/MAC을 요구하므로 **SRAM1 이상에서 이미 대역폭 바운드**가 된다. [D: A급 폭·클럭 근거 조합]
- L1 캐시는 **I 4 KB / D 4 KB**(F746 계열)로 매우 작다 — 32 B 라인 기준 각각 128 라인뿐이라, 51 KB급 conv2 가중치는 D-캐시의 **12.5배**여서 스트리밍 미스가 불가피하다. [A(용량) + D(계산)]
- STM32F746은 **ST의 MLPerf Tiny 제출 목록에 없다**(ST는 STM32L4R5 / U575 / H7A3만 제출). F746 전용 MLPerf 수치는 존재하지 않는다. [B]

---

## 1. CoreMark / DMIPS 공식 점수

| 항목 | 값 | 조건 | 등급 | 출처 |
|---|---|---|---|---|
| DMIPS | **462 DMIPS** | 216 MHz, Dhrystone 2.1 | A | DS10916(DocID027590) Rev 1, 표지·Features 첫 항목 |
| DMIPS/MHz | **2.14** | 동상 | A | 동상 |
| CoreMark | **1082 CoreMark** | 216 MHz, Flash 실행 + ART Accelerator 0-wait | B | ST STM32F7 시리즈 제품 페이지 / ST 브로슈어 |
| CoreMark/MHz | **5.009** (= 1082 / 216) | 계산값 | D | 위 두 값의 나눗셈 |
| 최대 CPU 주파수 | **216 MHz** (−40 ~ +85 °C), **200 MHz** (−40 ~ +105 °C) | 6 suffix / 7 suffix | A | DS10916 Rev 1, Table 2 각주 3 및 §5 각주 2 |

- ST 문구 원문(제품 페이지/브로슈어): *"1082 CoreMark/462 DMIPS at 216 MHz fCPU executing from embedded Flash memory with 0-wait states thanks to ST's ART Accelerator"* [B]

> **실측 정정**: 이 문구의 "ART Accelerator" 조건은 **ITCM 인터페이스 별칭(0x00200000)에서 실행할 때만** 유효하다.
> 실측에서 AXIM(0x08000000) 경로의 플래시 읽기는 ART ON/OFF가 **159.96 MB/s로 완전히 동일**했고,
> ITCM 별칭 경로만 **336.79 → 492.10 MB/s(×1.46)**로 반응했다. AXIM은 ART가 아니라 D-cache에 반응한다(159.96 → 304.82, ×1.91).
> 더구나 **이 보드의 펌웨어는 ART가 꺼진 채 돌고 있었다**(`FLASH_ACR = 0x00000006` → LATENCY=6 WS, PRFTEN=0, ARTEN=0).
> 즉 "0-wait Flash 실행"은 이 보드의 실측 조건이 **아니다**. [실측: `info/results/probe_output.txt` §6]
- **DS10916 데이터시트 본문에는 CoreMark 수치가 없다.** DMIPS만 표기된다. CoreMark는 ST 마케팅 자료(제품 페이지/브로슈어) 출처다. [A: 데이터시트 부재 확인]
- **EEMBC 공식 등록**: `eembc.org/coremark/scores.php`는 JS로 표를 로드해 본문 추출에 실패했고, STM32F746 항목의 존재 여부를 확인하지 못했다. ST가 EEMBC에 공식 게시한 것을 확인한 STM32는 STM32H5 계열(STM32H563ZIT6: **1023 CoreMark, 4.092 CoreMark/MHz @250 MHz, Arm Clang 6.16, ICache + prefetch ON**)뿐이다. [C: community.st.com 튜토리얼]
- 비교 참고: 1082 CoreMark는 **Flash 실행 시** 수치다. AN4667(STM32F7 System architecture and performance)에는 ITCM/AXI-Flash/SRAM/SDRAM/QuadSPI별 성능 비교 표가 있다고 목차상 확인되나, st.com PDF가 반복 타임아웃되어 **본문 수치는 확보하지 못했다**(§ 미확인 참조).

### 1.1 소비 전류 (에너지 환산용, DS10916 Rev 1)

Run 모드, 데이터 처리 코드, 레귤레이터 ON, TA = 25 °C, Typ 기준.

| 실행 위치 / 캐시 | fHCLK | 주변장치 전부 ON | 주변장치 전부 OFF | 등급 |
|---|---|---|---|---|
| ITCM RAM (Table 25) | 216 MHz | **178 mA** | **102 mA** | A |
| ITCM RAM (Table 25) | 200 MHz | **165 mA** | **95 mA** | A |
| Flash(ART ON, prefetch OFF, L1 ON) 또는 SRAM on AXI (Table 26) | 216 MHz | **186 mA** | **108 mA** | A |
| Flash / SRAM on AXI, **L1-cache disabled** (Table 27) | 216 MHz | **181 mA** | **105 mA** | A |
| Flash / SRAM on AXI, **L1-cache disabled** (Table 27) | 200 MHz | **168 mA** | **98 mA** | A |

> 캐시를 끄면 전류는 **오히려 소폭 감소**(216 MHz OFF-peripheral 기준 108 → 105 mA)하지만 실행 시간이 늘어나므로 에너지/작업은 증가한다. 캐시 OFF 시의 실행시간 배수는 데이터시트에 없다. [A(전류) / 미확인(시간)]

> **실측 정정**: 위 표의 "Flash(ART ON, prefetch OFF, L1 ON)" 행은 **이 보드의 실제 상태가 아니다**.
> 실측 상태는 **ART OFF + prefetch OFF + 6 WS**이며, 코드는 ITCM 별칭에서 실행되므로 I-cache도 걸리지 않는다.
> 전류 자체는 측정하지 않았으므로(전류계 미사용) 위 mA 값들은 여전히 미검증이다.
> 다만 **VDDA는 실측 3.311 V**로, §5의 에너지 환산에 쓴 3.3 V 가정은 타당하다. [실측: `measured_facts.md` §8, §9]

---

## 2. Cortex-M7 마이크로아키텍처 — MAC 처리량의 상한

### 2.1 ARM DDI 0489F 원문 근거 [전부 A]

| 사실 | 원문/수치 | 위치 |
|---|---|---|
| MAC 파이프라인 | "**Single MAC pipeline** capable of 32×32-bit + 64-bit → 64-bit with **two cycle result latency and one MAC per cycle throughput**" | §1.2.1 Data Processing Unit |
| ALU | "Two ALUs, with **one ALU capable of executing SIMD operations**" | §1.2.1 |
| 레지스터 파일 | "six read ports and four write ports for large-scale dual-issue" | §1.2.1 |
| 듀얼 이슈 | "in-order super-scalar pipeline that means many instructions can be dual-issued, **including load/load and load/store instruction pairs**" | §1.1 |
| MAC 듀얼이슈 제어 | ACTLR bit[24] = "Integer MAC and MUL" 은 channel 1 발행 금지, bit[19] = "Integer MAC and MUL"이 channel 0에 있으면 아무것도 듀얼이슈 불가 | §3.3.1 Table 3-3 |
| 명령어 인출 | "**64-bit instruction fetch bandwidth**", "4×64-bit pre-fetch queue" | §1.2.2 Prefetch Unit |
| 인터페이스 폭 | "**ITCM 인터페이스는 64-bit**, DTCM은 **D0TCM/D1TCM 각 32-bit** 두 개로 분할", AXIM은 "**64-bit AXI bus**", AHBP는 "single **32-bit** wide interface" | §5.8 / §5.5 |
| AXI 버스트 | "A burst never transfers more than **32 bytes**" | §5.5 |
| 명령어 사이클 표 | **DDI 0489F에는 명령어별 사이클 타이밍 장(章)이 존재하지 않는다** (목차 확인: Ch.1~12 + Appendix A Revisions뿐) | 목차 |

### 2.2 __SMLAD / __SMLALD 이론 최대 처리량

| 명령 | 동작 | MAC/명령 | 누산기 | 등급 |
|---|---|---|---|---|
| `SMLAD Rd, Rn, Rm, Ra` | signed 16×16 곱 2개를 32-bit 누산기에 더함 | **2** | int32 | A (ARMv7-M ARM / CMSIS SIMD intrinsics 문서) |
| `SMLALD RdLo,RdHi, Rn,Rm` | signed 16×16 곱 2개를 **64-bit** 누산기에 더함 | **2** | int64 | A (CMSIS Core intrinsic 문서: "performs two signed 16-bit multiplications, adding both results to a **64-bit** accumulate operand") |

**이론 피크 계산** [D — 근거: DDI 0489F §1.2.1의 1 MAC/cycle + 명령당 2 MAC]

```
peak_MACs_per_cycle = 1 (MAC 파이프라인 발행폭) × 2 (SMLAD/SMLALD의 16-bit MAC 수) = 2
```

| 클럭 | int16 MAC 피크 | OPS 환산(1 MAC = 2 ops) |
|---|---|---|
| 216 MHz | **432 × 10⁶ MAC/s (0.432 GMAC/s)** | **0.864 GOPS** |
| 200 MHz (본 프로젝트 설정) | **400 × 10⁶ MAC/s (0.400 GMAC/s)** | **0.800 GOPS** |

주의사항:
- Cortex-M7은 슈퍼스칼라지만 **MAC 파이프라인이 1개**이므로 SMLAD 2개를 같은 사이클에 발행할 수 없다. 즉 "2 MACs/cycle"이 **하드 상한**이다. [A + D]
- 단 **load + MAC 듀얼 이슈**는 가능하다(§1.1: load/store pair 듀얼이슈). 따라서 "SMLAD 1개 + LDR 1개 / cycle"이 지속되면 피크에 도달할 수 있다. [A + D]
- ST 제품 페이지는 Cortex-M7 DSP 확장을 "**single cycle 16/32-bit MAC and single cycle dual 16-bit MAC**"으로 표현한다. [B: st.com Arm Cortex-M7 페이지]

> **실측 보강**: 명령 집합 존재 여부는 이 실리콘에서 확인됐다 — ID_ISAR3 SIMD=3(패킹 SIMD/DSP 확장),
> ID_ISAR2 MultS=3(**SMLAD/SMUAD 지원**), 툴체인 `__ARM_FEATURE_DSP=1`·`__ARM_FEATURE_SIMD32=1`,
> 빌드 플래그 `-mcpu=cortex-m7 -mfpu=fpv5-sp-d16 -mfloat-abi=hard`(FPU는 **FPv5-SP** — `double`은 소프트 에뮬레이션).
> 또한 `ACTLR = 0x00000000`이므로 §2.1 표의 **MAC 듀얼이슈 억제 비트(bit19/24)는 켜져 있지 않다**.
> 다만 **2 MAC/cycle이라는 처리량 자체는 아직 실측하지 않았다** — SMLAD 언롤 루프를 ITCM에 올려 DWT CYCCNT로 재야 한다.
> [실측: `measured_facts.md` §4, `info/raw/host_side.md`]

### 2.3 무엇이 피크를 깎는가

| 요인 | 정량 효과 | 등급 |
|---|---|---|
| 오퍼랜드 로드 | SMLAD 1회당 32-bit 워드 2개(=8 B) 필요. 재사용 없으면 **8 B/cycle** 로드 필요 = **1.73 GB/s @216 MHz** → DTCM 대역폭과 정확히 동일하고 SRAM1(4 B/cycle)로는 절반밖에 못 준다 | D |
| 2×2 타일링 | CMSIS-NN `mat_mult`는 2열×2행 타일로 "4 outputs per inner loop"를 계산해 로드 수를 절반으로 줄인다("This enables some data reuse and saves on the total number of load instructions") | A (arXiv 1801.06601 §4.2) |
| q7→q15 확장 | int8 데이터를 SMLAD에 쓰려면 `__SXTB16`으로 16-bit 확장 필요. CMSIS-NN은 재정렬을 생략한 변형(`arm_q7_to_q15` without reordering)을 도입해 이 오버헤드를 줄였다 | A (§4.1, §4.2) |
| D-캐시 미스 | F746의 D-캐시는 **4 KB / 32 B 라인 = 128 라인, 4-way(=32 set)**. conv2 가중치 5×5×32×64 int8 = **51,200 B**는 D-캐시의 **12.5배** → 라인 재사용 없이 스트리밍 미스 | D (용량은 A) |
| 캐시 교체 정책 | "The data cache and Instruction cache implement a **pseudo-random replacement algorithm**" → 워킹셋이 캐시보다 크면 재사용 예측이 어렵다 | A (AN4839 Rev 2 §2) |
| TCM은 캐시 안 됨 | "memories on the Tightly Coupled Memory (TCM) interface … are not cacheable" → DTCM(64 KB)은 항상 0-wait, 캐시 무관 | A (AN4839 §2 / DS10916 §2.5 "accessed at CPU clock speed with 0-wait states") |
| **캐시 미스 페널티(사이클 수)** | **미확인** — ST/ARM 공식 문서에서 STM32F746의 Flash/SRAM/SDRAM 미스 페널티 사이클 수치를 찾지 못함 | 미확인 |

> **실측 정정(보강)**: 마지막 행의 "미확인"은 이제 실측으로 채워졌다. 포인터 체이싱(Sattolo 셔플, 프리페치 개입 불가)과
> 스트라이드 스캔으로 측정한 접근당 사이클:
>
> | 경로 | 실측 |
> |---|---|
> | D-cache 히트 | **2.0 cyc** |
> | D-cache 미스 → SRAM1 | **8~9 cyc** |
> | SDRAM(0xC0000000) 랜덤 | **21~26 cyc** |
> | DTCM / ITCM (32 B 스트라이드, 루프 오버헤드 2.00 포함) | **5.16 cyc** |
> | SRAM1/SRAM2 (동일 조건) | **15.66 cyc** |
> | Flash AXIM 4 KB 스트라이드 | **14.34 cyc** |
> | Flash ITCM 별칭 4 KB 스트라이드 | **9.13 cyc** |
> | QSPI 64 KB 스트라이드 | **94~198 cyc** |
>
> 또한 "TCM은 캐시 안 됨" 행은 실측으로 확인됐다 — D-cache OFF/WB/WT 세 조건에서 ITCM·DTCM 대역폭이 동일했다.
> "캐시 교체 정책이 pseudo-random"이라는 행은 워킹셋 8 KB에서 7.11 cyc(완전 미스 8.0보다 낮음)로 부분 잔존이 관측되어
> 정황상 부합하나, 정책 자체를 직접 확인한 것은 아니다. [실측: `info/results/probe_output.txt` §2·§5·§6]

---

## 3. CMSIS-NN 벤치마크 (Arm 공식 수치)

출처: Liangzhen Lai, Naveen Suda, Vikas Chandra, *"CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs"*, arXiv:1801.06601. 측정 보드 = **NUCLEO-F746ZG (Cortex-M7 @216 MHz)**. [모두 A]

### 3.1 CIFAR-10 CNN 레이어별 (논문 Table 1)

| # | 레이어 | 가중치 | 출력 | Ops | Runtime | MACs(=Ops/2, 계산) | 유효 MAC/s (계산) | MAC/cycle @216 MHz (계산) |
|---|---|---|---|---|---|---|---|---|
| 1 | Convolution | 5×5×3×32 (2.3 KB) | 32×32×32 (32 KB) | 4.9 M | **31.4 ms** | 2.4576 M | 78.3 M | **0.362** |
| 2 | Max Pooling | – | 16×16×32 (8 KB) | 73.7 K | 1.6 ms | – | – | – |
| 3 | Convolution | 5×5×32×32 (25 KB) | 16×16×32 (8 KB) | 13.1 M | **42.8 ms** | 6.5536 M | 153.1 M | **0.709** |
| 4 | Max Pooling | – | 8×8×32 (2 KB) | 18.4 K | 0.4 ms | – | – | – |
| 5 | Convolution | 5×5×32×64 (50 KB) | 8×8×64 (4 KB) | 6.6 M | **22.6 ms** | 3.2768 M | 145.0 M | **0.671** |
| 6 | Max Pooling | – | 4×4×64 (1 KB) | 9.2 K | 0.2 ms | – | – | – |
| 7 | Fully-connected | 4×4×64×10 (10 KB) | 10 | 20 K | **0.1 ms** | 10.24 K | 102.4 M | 0.474 |
| | **합계** | **87 KB weights** | **55 KB activations** | **24.7 M** | **99.1 ms** | **12.298 M** | **124.1 M** | **0.575** |

- 전체 처리량: **249 MOps/s**(논문 명시), **10.1 images/s** [A]
- MAC 기준 환산: **124.1 MMAC/s = 0.575 MAC/cycle @216 MHz = 이론 피크(2 MAC/cycle)의 28.7 %** [D]
- 레이어 1이 유독 낮은 이유(0.362 MAC/cycle): 커널 볼륨이 5×5×3 = **75**로 짧아 im2col/SIMD 정렬 오버헤드 비중이 크기 때문. 커널 볼륨 800인 레이어 3에서 0.709 MAC/cycle로 최고 효율. [D]
- 정확도: 사전 양자화 모델 **80.3 %**, Cortex-M7의 8-bit 양자화 실행 **79.9 %** [A]
- 메모리: partial im2col 사용 시 최대 **~133 KB**, 미사용 시 **~332 KB**(보드에 안 들어감) [A]

### 3.2 CMSIS-NN vs 베이스라인 (논문 Table 2)

| 레이어 종류 | 베이스라인 | CMSIS-NN | Throughput 개선 | 에너지 효율 개선 |
|---|---|---|---|---|
| Convolution | 443.4 ms | **96.4 ms** | **4.6×** | **4.9×** |
| Pooling | 11.83 ms | **2.2 ms** | **5.4×** | **5.2×** |
| ReLU | 1.06 ms | **0.4 ms** | **2.6×** | **2.6×** |
| **Total** | **456.4 ms** | **99.1 ms** | **4.6×** | **4.9×** |

- 베이스라인 = CMSIS-DSP `arm_conv` 기반 1D convolution + Caffe 스타일 pooling/ReLU [A]
- 논문 본문: "The CMSIS-NN kernels achieve **2.6X to 5.4X** improvement in runtime/throughput over the baseline functions." [A]

### 3.3 데이터 레이아웃 (HWC vs CHW)

- Cortex-M7에서 입력 16×16×16 고정, 출력 채널 0~16 스윕 실험: **HWC가 im2col 시간이 더 짧고, matrix-multiplication 시간은 동일**. 최대 runtime은 약 25~30 ms 규모(그래프 Fig.10, 정확한 눈금값은 벡터 그래프라 판독 불가). 이 결과로 CMSIS-NN conv 커널은 전부 **HWC(NHWC)** 가정. [A (결론) / 미확인 (그래프 개별 수치)]

### 3.4 Arm 공식 블로그의 최신 재측정 (동일 CIFAR-10 CNN, Mbed)

| 구성 | 값 | 등급 |
|---|---|---|
| 보드 | STM32F746NG (Flash 1 MB, SRAM 340 kB로 표기) | B |
| CMSIS-NN + GCC | **2.26 inferences/s** | B |
| 레퍼런스 커널 + GCC | **0.72 inferences/s** | B |
| 속도 향상 | **3.1×** (GCC), **2.9×** (armclang) | B |
| 정확도 | **89 %** (CIFAR-10) | B |

> **실측 확인**: 표의 "STM32F746NG (Flash 1 MB, SRAM 340 kB)"는 실측과 모순되지 않는다 —
> Flash size register = 1024 KB, 내부 RAM = DTCM 64 + SRAM1 240 + SRAM2 16 = 320 KB, 여기에 ITCM 16 KB + 백업 SRAM 4 KB를
> 더하면 정확히 **340 KB**다. [실측: `measured_facts.md` §1·§3.2]

출처: developer.arm.com AI blog, "Image recognition on Arm powered microcontrollers".
※ 2.26 img/s는 arXiv 논문의 10.1 img/s와 크게 다르다 — 모델·전처리·Mbed 런타임 차이로 보이나 **원인은 문서에 명시되어 있지 않다**. [C: 해석]

---

## 4. CMSIS-NN s16 커널의 DSP fast path 조건 (소스 근거)

로컬 소스: `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/third_party/cmsis_nn/`
(Copyright 2020-2023 Arm, `arm_convolve_fast_s16` 존재 → **CMSIS-NN v6.0.0 이전 버전**. v6.0.0에서 `arm_convolve_fast_s16`는 제거되고 `arm_convolve_s16`으로 통합됨 [B: CMSIS-NN GitHub Releases v6.0.0, 2024-06-03])

### 4.1 Conv2D int16 — `arm_convolve_wrapper_s16`

`Source/ConvolutionFunctions/arm_convolve_wrapper_s16.c:62-66` [A]

```c
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    if (filter_dims->w * filter_dims->h * input_dims->c < 512 &&
        (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        return arm_convolve_fast_s16(...);   // SMLAD 경로
    }
    else
    {
        return arm_convolve_s16(...);        // 스칼라 C 경로
    }
#else
    return arm_convolve_s16(...);
#endif
```

**fast path 진입 조건 (전부 AND)** [A]

| # | 조건 | 값 |
|---|---|---|
| 1 | 컴파일 플래그 | `ARM_MATH_DSP` 정의 **AND** `ARM_MATH_MVEI` 미정의 |
| 2 | 커널 볼륨 | `filter_w × filter_h × input_ch` **< 512** |
| 3 | dilation | `dilation.w == 1 && dilation.h == 1` |

헤더 문서(`Include/arm_nnfunctions.h`)도 동일하게 명시: *"(filter_dims->w \* filter_dims->h \* input_dims->c) must not exceed 512"*, *"Implementation supports kernel volumes (filter width \* filter height \* input channels) < 512."* [A]

**필요 버퍼 크기** (`arm_convolve_get_buffer_sizes_s16.c:46`) [A]
```
arm_convolve_fast_s16_get_buffer_size = 2 × input_c × filter_w × filter_h × sizeof(int16_t)  [bytes]
arm_convolve_s16_get_buffer_size      = 0
```
즉 fast path는 **커널 볼륨 × 4 바이트**의 im2col 버퍼를 추가로 요구한다(2열 동시 처리 → ×2, int16 → ×2).

### 4.2 Depthwise Conv int16 — `USE_FAST_DW_CONV_S16_FUNCTION`

`Include/arm_nnsupportfunctions.h:43-45` [A]

```c
#define USE_FAST_DW_CONV_S16_FUNCTION(dw_conv_params, filter_dims, input_dims)  \
    (dw_conv_params->ch_mult == 1 && dw_conv_params->dilation.w == 1 &&         \
     dw_conv_params->dilation.h == 1 &&                                         \
     filter_dims->w * filter_dims->h < 512)
```
→ **ch_mult == 1**, dilation 1×1, **filter_w × filter_h < 512** (여기는 채널 수 미포함). [A]
공식 문서 추가 조건: *"Recommended when number of channels is 4 or greater"*, *"for Cortex-M CPUs with DSP extension only"*. [B: arm-software.github.io CMSIS-NN NNConv 문서]

### 4.3 Fully-Connected int16 — `arm_nn_vec_mat_mult_t_s16`

`Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.c` [A]

```c
#define MAX_COL_COUNT (512)
...
int32_t rhs_cols_fast = rhs_cols;
if (rhs_cols > MAX_COL_COUNT) { rhs_cols_fast = MAX_COL_COUNT; }
// ... SMLAD 루프는 rhs_cols_fast 까지만 (int32 acc)
if (rhs_cols > MAX_COL_COUNT) {
    for (int i = MAX_COL_COUNT; i < rhs_cols; i++) { result_64 += *rhs++ * lhs_temp; }  // int64 스칼라 폴백
}
```
→ FC는 **하드 폴백이 아니라 하이브리드**: 앞 512열은 SMLAD, **513열째부터는 int64 스칼라 루프**. 즉 `rhs_cols`가 512를 크게 넘으면 대부분이 스칼라로 처리된다. [A]

### 4.4 512라는 숫자의 근거

Arm 공식 이슈 트래커 원문: *"If rhs_cols is larger than 512, the DSP implementation may **overflow since it is accumulated on 32 bits**."* [A: github.com/ARM-software/CMSIS_5 issue #1380, 2021-12-16]

산술 유도 [D]:
```
int16 활성값 최대 |x| = 32768 = 2^15
int8  가중치 최대 |w| = 128   = 2^7
곱 1개 최대 |x·w|     = 2^22 = 4,194,304
int32 표현 한계        = 2^31
2^31 / 2^22 = 512      → 512개 항을 더하면 정확히 오버플로 경계
```
따라서 `< 512`는 임의 상수가 아니라 **int32 누산기의 정확한 안전 한계**다.

### 4.5 실제 fast path의 내부 (SMLAD, int32 누산)

`Source/NNSupportFunctions/arm_nn_mat_mul_kernel_s16.c` [A]
- 2행×2열 타일(`ch_0_out_0/0_1/1_0/1_1`) → 내부 루프 1회당 `SMLAD` **8회**(4열 처리) → 8 SMLAD로 **16 MAC**
- 누산은 `int32`(`ch_*_out_*`), 최종에만 `int64 acc_64 = ch_0_out_0 + *bias`로 승격
- `Source/ConvolutionFunctions/arm_convolve_fast_s16.c:172,174`도 `SMLAD` 사용, `int64_t acc_64 = sum + bias_data[i]`

> **핵심**: CMSIS-NN의 int16 fast path는 `SMLALD`(64-bit 누산)가 **아니라** `SMLAD`(32-bit 누산)를 쓴다. 그래서 512 제약이 생긴다.
> 본 프로젝트의 퓨전 커널(`ImageClassification/Core/Src/snn_conv_pool_lif_s16.c`)은 `__SMLALD`로 **int64 누산**을 하므로 512 제약이 없다. [A: 로컬 소스]

---

## 5. 본 프로젝트 실측과의 교차 검증 (STM32F746G-DISCO @200 MHz)

출처: 로컬 `EXPERIMENTS.md`, `paper_ieie/06_Envelope.tex`. 등급 D(자체 실측이므로 1차 공식 문서는 아님).

> **실측 정정(측정 조건 보강)**: 아래 표의 수치가 나온 하드웨어 상태를 ST-LINK로 확인한 결과는 다음과 같다.
> - SYSCLK/HCLK = **200 MHz**(PLLM=25/PLLN=400/PLLP=2), PCLK1 50 MHz, PCLK2 100 MHz — 216 MHz가 아님이 확정.
> - `FLASH_ACR = 0x00000006` → **6 wait state, ART OFF, prefetch OFF**.
> - `VTOR = 0x00200000`, 옵션바이트 `BOOT_ADD0 = 0x80` → **ITCM 별칭 플래시에서 실행**. I-cache는 AXIM 접근에만 적용되므로
>   **명령 페치에 아무 캐시도 걸려 있지 않았다**.
> - `SCB_CCR = 0x00070200`(IC/DC/BP ON), `ACTLR = 0x00000000`(감속 비트 없음), MPU 활성(`MPU_CTRL = 0x00000005`).
>
> 즉 **4,494 / 9,641 / 2,407 ms는 플래시 페치 경로가 가속되지 않은 상태의 값**이며, ART를 켜는 것만으로 그 경로가
> 1.46배 빨라진다(492.10 vs 336.79 MB/s 실측). 개선 폭이 전체 추론 시간에 얼마나 반영되는지는 **아직 재측정하지 않았다**.
> [실측: `measured_facts.md` §2.2·§7·§8]

| 변형 | 커널 | total ms/이미지 | NN ms | ms/step (T=30) |
|---|---|---|---|---|
| V1 Scalar | 원본 CHW 스칼라 | 4,494.2 | 4,422.7 | 147.4 |
| V2 TFLM + CMSIS-NN (비퓨전) | 빌트인 CONV_2D int16x8 | **9,641.4** | 9,570.0 | 319.0 |
| V3 TFLM + 퓨전 | `SNN_CONV_POOL_LIF` (6×6/s2, SMLALD) | **2,407.0** | 2,335.7 | **77.9** |
| V4 직접 퓨전 (TFLM 없음) | 동일 커널 | 2,395.5 | 2,324.8 | 77.5 |

**§4의 512 제약이 실측으로 드러난 사례** [D]
- conv2의 im2col 열 길이 = `5 × 5 × 32` = **800** ≥ 512 → `arm_convolve_wrapper_s16`가 `arm_convolve_s16`(스칼라)로 폴백
- 결과: 표준 TinyML 스택(V2)이 순수 스칼라(V1)보다 **2.15× 느림** (9,641 / 4,494)
- 퓨전 커널 도입 시 V2 대비 **4.0×**(NN 기준 4.10×) 개선

**유효 처리량** [D]
```
MACs/step = 884,736 (conv1) + 4,718,592 (conv2) + 40,960 (FC) = 5,644,288
5,644,288 MAC / 77.9 ms = 72.5 × 10^6 MAC/s
@200 MHz → 0.362 MAC/cycle → 이론 피크(2 MAC/cycle)의 18.1 %
@216 MHz 환산 피크 대비 → 16.8 %
```
CMSIS-NN 논문의 CIFAR-10(0.575 MAC/cycle, 28.7 %)보다 낮은데, LIF 상태 갱신·스파이크 인코딩 등 MAC이 아닌 연산이 섞여 있기 때문으로 보인다. [D: 해석]

**에너지 추정** [D — 데이터시트 Typ 전류 × 3.3 V × 실측 시간, 실제 보드는 LCD/SDRAM 포함이라 상한·하한 개념]

| 가정 | 전력 | 이미지당 에너지 (2.407 s) | nJ/MAC (169.3 M MAC/이미지) |
|---|---|---|---|
| 200 MHz, 주변장치 OFF (95 mA) | 313.5 mW | **0.755 J** | 4.46 nJ |
| 200 MHz, 주변장치 ON (165 mA) | 544.5 mW | **1.311 J** | 7.74 nJ |

참고: CMSIS-NN CIFAR-10(216 MHz, 108 mA, 99.1 ms) → 356 mW × 99.1 ms = **35.3 mJ/이미지**, 12.3 M MAC → **2.87 nJ/MAC**. [D]

---

## 6. 메모리 대역폭 / 루프라인 분석

### 6.1 각 메모리 계층의 이론 대역폭

폭 근거는 ARM DDI 0489F(§5.5, §5.8)와 DS10916(§2.5, §2.6). 클럭은 216 MHz / 200 MHz 병기. [D 계산, 폭·클럭은 A]

| 메모리 | 경로 / 폭 | B/cycle | @216 MHz | @200 MHz | 등급 |
|---|---|---|---|---|---|
| **DTCM-RAM (64 KB)** | D0TCM+D1TCM = 2×32-bit, 0-wait | **8** | **1,728 MB/s** | 1,600 MB/s | A(폭)+D(계산) |
| **ITCM-RAM (16 KB)** | ITCM 64-bit, 명령 전용 | 8 | 1,728 MB/s | 1,600 MB/s | A+D |
| **L1 D-cache 히트 (4 KB)** | 코어 내부 | 8 (LDRD 기준) | 1,728 MB/s | 1,600 MB/s | D |
| **SRAM1 (240 KB) / SRAM2 (16 KB)** | AXIM(64-bit) → **AXI-to-32-bit-AHB 브리지** → AHB matrix | **4** | **864 MB/s** | 800 MB/s | A(구조)+D(계산) |
| **내장 Flash (1 MB)** | AXIM → **AXI-to-64-bit-AHB 브리지** + ART Accelerator | 8 (버스트) | 1,728 MB/s | 1,600 MB/s | A(구조)+D(계산) |
| **외부 SDRAM (DISCO, MT48LC4M32B2)** | FMC, **16-bit만 연결**(DQ16~DQ31 미사용), SDCLK = HCLK/2 | 2 B / SDCLK | **216 MB/s** (SDCLK 108) | **200 MB/s** (SDCLK 100) | C(보드)+D(계산) |

보드/설정 근거:
- UM1907: "128-Mbit SDRAM (MT48LC4M32B2B5-6A), **only the lower 16 bits of data are used (64 Mbits accessible)**, DQ16~DQ31은 10 kΩ 풀다운" [C: UM1907 요약/커뮤니티 재인용 — 원문 PDF 직접 확인 실패]
- 본 프로젝트 `ImageClassification.ioc`: `FMC.SDClockPeriod1/2 = FMC_SDRAM_CLOCK_PERIOD_2` (SDCLK = HCLK/2 = **100 MHz**), `FMC.CASLatency1 = FMC_SDRAM_CAS_LATENCY_3`, `ReadBurst = ENABLE`, `HCLKFreq_Value = 200000000`, APB1 = 50 MHz, APB2 = 100 MHz [A: 로컬 프로젝트 설정 파일]
- 위 SDRAM 수치는 **CAS latency 3·리프레시·row activate 오버헤드를 뺀 이론 상한**이다. 실효값은 이보다 낮다. [D]

> **실측 정정 1 — 이론 대역폭 vs 실제 도달 대역폭**: 위 표의 B/cycle은 **버스 폭**이지 코드가 실제로 얻는 값이 아니다.
> 같은 보드에서 인라인 어셈블리로 측정한 값(@200 MHz):
>
> | 메모리 | 문서 이론 | 실측 최고 읽기 | 실측 최고 쓰기/복사 |
> |---|---|---|---|
> | DTCM | 1,600 MB/s | 457.5 (LDM, 2.29 B/cyc) | write 708.0 / copyLDM **1,409.4 (7.05 B/cyc)** |
> | ITCM | 1,600 MB/s | 460.0 (2.30 B/cyc) | copyLDM 1,325.6 (6.63 B/cyc) |
> | SRAM1 | 800 MB/s | 389.7 (1.95 B/cyc) | write 709.5 (3.55 B/cyc) |
> | Flash AXIM | 1,600 MB/s | 160.0 (D$ OFF) / 304.8~339.6 (D$ ON) | — |
> | Flash ITCM 별칭 | (항목 없음) | **492.1 (ART ON)** / 336.8 (ART OFF) | — |
> | SDRAM | 200 MB/s | **60.3** | write 194.6 |
> | QSPI(외부, 16 MB Winbond) | (항목 없음) | 51.7 | — |
>
> 해석 주의: 순차 **읽기** 루프는 ART OFF·6 WS 플래시에서 실행되었으므로 명령 페치 바운드일 가능성이 크다
> (모든 내부 메모리의 `read32`가 205~228 MB/s로 동일하게 눌렸다). 읽기 실측치는 **하한**으로 볼 것.
> 반대로 DTCM copyLDM 7.05 B/cyc는 이론 8 B/cyc에 근접하므로 **"DTCM = 8 B/cycle"이라는 폭 주장 자체는 유효**하다.
>
> **실측 정정 2 — SDRAM**: 실측 읽기 60.3 MB/s는 문서 이론값(200 MB/s)의 30 %다. 랜덤 접근 지연은 21~26 cyc.
> 원인은 프로브 펌웨어가 **MPU를 설정하지 않아 0xC0000000이 ARMv7-M 기본 메모리맵상 Device 메모리**로 취급되기 때문이다
> (D-cache OFF/WB에서 값이 동일한 것이 그 증거). MPU로 Normal cacheable로 재정의하면 개선 여지가 있다.
>
> **실측 정정 3 — CAS latency**: 실행 중 `SDCR1 = 0x00001954`의 CAS[8:7]=`0b10` → **CL2**이지 CL3이 아니다.
> ST BSP(`stm32746g_discovery_sdram.c:151`)가 `FMC_SDRAM_CAS_LATENCY_2`로 초기화하며, 프로젝트의 `MX_FMC_Init()`은
> `main.cc:128,201`에서 **주석 처리**되어 있다. 즉 `.ioc`의 `CASLatency1 = ..._3`은 실행되지 않는 설정값이다.
> 나머지 SDRAM 구성(16-bit, 4 bank / 12 row / 8 col = 8 MB, SDCLK = HCLK/2 = 100 MHz, RBURST=1, 리프레시 15.4 µs)은 문서와 일치.
> [실측: `info/results/probe_output.txt` §3·§4·§7, `measured_facts.md` §11.1]

### 6.2 루프라인 (int16 MAC 기준)

**요구 대역폭**: 재사용이 전혀 없는 int16 MAC은 오퍼랜드 2개 × 2 B = **4 B/MAC**이 필요하다.

| 메모리 | 대역폭 @216 MHz | 이 대역폭으로 가능한 MAC/s (4 B/MAC) | 이론 피크 432 MMAC/s 대비 | 피크 도달에 필요한 산술강도 |
|---|---|---|---|---|
| DTCM | 1,728 MB/s | **432 M** | **100 %** (딱 맞음) | 0.25 MAC/B |
| SRAM1 (32-bit AHB) | 864 MB/s | **216 M** | **50 %** | 0.50 MAC/B (재사용 2×) |
| Flash (ART, 64-bit 브리지) | 1,728 MB/s 버스트 | 432 M | 100 % (지연 무시 시) | 0.25 MAC/B |
| SDRAM (DISCO, 16-bit) | 216 MB/s | **54 M** | **12.5 %** | **2.0 MAC/B (재사용 8×)** |

**결론** [D]
1. **DTCM(64 KB)에 올릴 수 있는 워킹셋만이 이론 피크에 도달할 수 있다.** DTCM은 캐시를 거치지 않고 0-wait이며 폭이 정확히 8 B/cycle이다.
2. SRAM1로 내려가는 순간 이론상 **최대 50 %**로 떨어진다. 재사용률 2× 이상을 만드는 타일링(CMSIS-NN의 2×2 타일 등)이 필수다.
3. SDRAM에 가중치나 활성값을 두면 **최대 12.5 %**(54 MMAC/s)로 제한된다. 본 프로젝트의 실측 72.5 MMAC/s가 이 값을 넘는 것은 파라미터(195 KB)를 Flash에, 활성값(80 KB)을 내부 SRAM/DTCM에 두었기 때문이다. [D]
4. **캐시가 루프라인의 knee를 옮기지 못하는 이유**: F746의 D-캐시는 4 KB뿐이다. conv2 가중치만 **51,200 B = D-캐시의 12.5배**이고, 커널 볼륨 800에 대한 fast_s16 im2col 버퍼도 `2 × 800 × 2 B = 3,200 B`로 D-캐시의 78 %를 차지한다. 워킹셋이 캐시를 초과하고 교체 정책이 pseudo-random이라 히트율이 급락한다. [D — 히트율 실측치는 미확인]

> **실측 정정 — 루프라인을 실측 대역폭으로 다시 그림**: 위 표는 **이론 버스 폭**을 넣은 것이고, 같은 4 B/MAC 가정에
> **실측 대역폭**을 넣으면 결론의 크기가 달라진다(@200 MHz).
>
> | 메모리 | 실측 읽기 대역폭 | 가능한 MAC/s | 문서 표의 값(200 MHz 환산) |
> |---|---|---|---|
> | DTCM | 457.5 MB/s | **114 M** | 400 M |
> | SRAM1 | 389.7 MB/s | **97 M** | 200 M |
> | Flash AXIM (D$ ON) | 304.8 MB/s | **76 M** | 400 M |
> | Flash ITCM 별칭 (ART ON) | 492.1 MB/s | **123 M** | (항목 없음) |
> | SDRAM | 60.3 MB/s | **15 M** | 50 M |
>
> 즉 **"DTCM에 올리면 이론 피크(2 MAC/cycle)에 도달한다"는 결론 1은 이 보드에서 성립하지 않는다** —
> 실측 읽기 대역폭으로는 DTCM조차 이론 피크의 약 29 %(0.57 MAC/cycle 상당)까지만 먹인다.
> 결론 3의 "SDRAM 12.5 %"도 실측 기준으로는 **약 3.7 %**로 더 낮다.
> 다만 위 읽기 실측치는 명령 페치 바운드일 수 있는 하한이므로(§6.1 실측 정정 1 참조),
> 정확한 knee는 ART를 켜고 커널을 ITCM에 올린 뒤 재측정해야 확정된다.
>
> 참고로 §5의 실측 유효 처리량 **72.5 MMAC/s**는 위 표의 "Flash AXIM(D$ ON) 76 MMAC/s"와 근접한다.
> 이 프로젝트가 파라미터를 플래시에 두는 것과 정합적이지만, 우연의 일치일 수 있으므로 단정하지 않는다.
> 결론 4의 "히트율 실측치는 미확인"은 여전히 유효하다(DWT 프로파일 카운터로 측정 가능 — `NOPRFCNT=0` 확인됨).

### 6.3 L1 캐시 파라미터 (F746 계열)

| 항목 | 값 | 등급 | 출처 |
|---|---|---|---|
| I-cache | **4 KB**, **2-way** set associative | A/B | DS10916 §"Harvard caches (4 Kbytes of I-cache and 4 Kbytes of D-cache)" / AN4839 Rev 2 §2 |
| D-cache | **4 KB**, **4-way** set associative | A/B | 동상 |
| 캐시 라인 | **32 bytes** (8 words) | A | AN4839 Rev 2 §2 / DDI 0489F CCSIDR LineSize=0x1 |
| 라인 수 | I: 128, D: 128 (계산 = 4096 / 32) | D | 계산 |
| 세트 수 | I: 64 (128/2), D: 32 (128/4) | D | 계산 |
| 교체 정책 | pseudo-random (I/D 둘 다) | A | AN4839 Rev 2 §2 |
| 쓰기 정책 | write-back / write-through 선택 (MPU 속성) | A | AN4839 Rev 2 §2 |
| TCM 캐시 여부 | **캐시되지 않음** | A | AN4839 Rev 2 §2 |

> ⚠️ 흔한 혼동: "16 KB + 16 KB"는 **STM32F76x/F77x** 사양이다. **STM32F746은 4 KB + 4 KB**다. [A]

> **실측 확인**: 이 표는 실측과 **전 항목 일치**한다. CCSIDR D=`0xF003E019`(32 set × 4 way × 32 B = 4 KB),
> I=`0xF007E009`(64 set × 2 way × 32 B = 4 KB), CTR=`0x8303C003`(라인 32 B), CLIDR=`0x09000003`(L1만, L2 없음).
> 레지스터뿐 아니라 성능 곡선으로도 재확인됐다 — SRAM1 워킹셋 스윕에서 4 KB까지는 235 MB/s를 유지하다
> 8 KB에서 221 MB/s로 꺾이고, 포인터 체이싱은 4 KB까지 2.0 cyc(히트) → 8 KB부터 7.1 cyc(미스)로 급변한다.
> TCM 비캐시도 확인됐다(D-cache OFF/WB/WT에서 ITCM·DTCM 대역폭 동일). [실측: `measured_facts.md` §2·§2.1]

### 6.4 메모리 맵 / 버스 구조 (DS10916 §2.5, §2.6) [A]

- SRAM1: **240 KB** (AHB bus matrix)
- SRAM2: **16 KB** (AHB bus matrix)
- DTCM-RAM: **64 KB** (TCM 인터페이스, CPU 클럭 속도 0-wait)
- ITCM-RAM: **16 KB** (CPU 실행 전용)
- Backup SRAM: 4 KB
- 총 시스템 SRAM: **320 KB** (+ ITCM 16 KB + backup 4 KB)
- 버스: "3× AXI to **32-bit** AHB bridges connected to AHB bus matrix" + "1× AXI to **64-bit** AHB bridge connected to the embedded flash"
- Flash 대기 상태: 216 MHz에서 **7 WS**로 알려져 있으나 RM0385 원문 표를 직접 확인하지 못했다 → [C] (§ 미확인 참조)

> **실측 정정**:
> - **Flash 대기 상태**: 이 보드(200 MHz, VOS Scale 1)에서 `FLASH_ACR.LATENCY = 6` → **6 WS**로 동작 중이다.
>   리셋 직후 기본값은 `FLASH_ACR = 0x00000000`(0 WS·ART OFF)이며, 설정은 전적으로 소프트웨어 책임이다.
>   216 MHz에서 7 WS인지는 이 보드가 200 MHz로 돌기 때문에 **여전히 미검증**이다.
> - **"ITCM-RAM 16 KB (CPU 실행 전용)"**: 실측상 ITCM은 **데이터로도 읽고 쓸 수 있다**
>   (read32 226.17 MB/s, write32 263.91 MB/s, copyLDM 1,325.56 MB/s). "실행 전용"은 관례이지 하드웨어 제약이 아니다.
>   크기 16 KB는 실측 확인됨(`0x00004000`부터 버스 폴트).
> - 나머지 용량(SRAM1 240 KB, SRAM2 16 KB, DTCM 64 KB, Backup 4 KB, 합계 320 KB)은 **주소 스캔·지연 스캔으로 전부 확인**.
>   단 SRAM1과 SRAM2는 접근 지연이 15.66 cyc로 동일해 **성능상 구분되지 않는다**. [실측: `measured_facts.md` §3.2·§8]

---

## 7. X-CUBE-AI / STM32Cube.AI / MLPerf Tiny

### 7.1 STM32F746 전용 수치의 부재

- ST가 MLCommons에 제출한 STM32는 **STM32L4, STM32U5, STM32H7**(모두 X-CUBE-AI v7.3)이며 **STM32F7은 없다**. [B: ST 블로그 / edge-ai-vision 재게시]
- ST 공식 벤치마크 리포지토리 `STMicroelectronics/stm32ai-perf`가 지원하는 보드는 **NUCLEO-H7A3ZI-Q, NUCLEO-U575ZI-Q, NUCLEO-L4R5ZI** 뿐이며 **NUCLEO-F746 / STM32F746은 포함되지 않는다**(요구 도구: STM32Cube.AI v9.0.0). [B: GitHub README]
- ST Edge AI Developer Cloud에서도 STM32F746G 보드 벤치마크가 동작하지 않는다는 커뮤니티 스레드가 존재한다. [C: community.st.com]

**결론: F746에 대한 ST/MLCommons 공식 MLPerf Tiny 수치는 존재하지 않는다.** 아래 값은 다른 STM32에 대한 참고치다.

### 7.2 MLPerf Tiny v1.0 — ST 제출값 (Cortex-M4, STM32L4R5ZIT6U @120 MHz)

| 벤치마크 | STMicroelectronics (X-CUBE-AI) | Plumerai | OctoML (microTVM + CMSIS-NN) | 등급 |
|---|---|---|---|---|
| Visual Wake Words | **230.5 ms** | 208.6 ms | 301.2 ms | C |
| Image Classification | **226.9 ms** | 173.2 ms | 389.5 ms | C |
| Keyword Spotting | **75.1 ms** | 71.7 ms | 99.8 ms | C |
| Anomaly Detection | **7.6 ms** | 5.6 ms | 8.6 ms | C |

출처: blog.plumerai.com "MLPerf Tiny 1.0" (MLCommons 공식 결과 재정리). Cortex-M4 @120 MHz 기준이므로 **Cortex-M7 @216 MHz에 그대로 대입할 수 없다**. [C]

참고 TFLM 대비: Plumerai 194.36 / 170.42 / 66.32 / 5.59 ms vs TFLM 335.97 / 376.08 / 100.72 / 8.45 ms (1.73× / 2.21× / 1.52× / 1.51×). [C]

### 7.3 X-CUBE-AI vs TFLM 일반 주장

- X-CUBE-AI v7.1이 TFLM 대비 추론 시간 **43 % 빠름**, v7.3은 **82 % 빠름**이라는 서술이 2차 자료에 있으나, 어떤 MCU·어떤 모델 기준인지 명시가 없어 **F746에 적용 가능하다고 볼 수 없다**. [C — 신뢰도 낮음]
- ST 위키 `AI:STM32Cube.AI model performances`는 모델별 MACC/추론시간/RAM/Flash 표를 제공하지만 **접속 타임아웃으로 본문 확보 실패**. [미확인]

---

## 8. SNN(스파이킹 신경망)을 MCU에서 돌린 선행 연구

공식 문서가 아닌 학술 문헌이므로 전부 [C], 수치는 논문 원문 인용.

| 연구 | 하드웨어 | 수치 | 등급 |
|---|---|---|---|
| Human Activity Recognition SNN (ACM ICMI/UbiComp 2023, DOI 10.1145/3594738.3611369) | STM32 (Cortex-M7 코어) vs Intel Loihi | STM32: **20.88 ms / 추론, 동적 에너지 8.07 mJ**; Loihi: **4.4 ms, 0.15 mJ** → 지연 4.75×, 에너지 **53.8×** 차이 | C (본문 직접 접근 403, 검색 스니펫 기반) |
| *Compression and Inference of SNNs on Resource-Constrained Hardware* (arXiv 2511.12136) | Arduino Portenta H7 (**STM32H747**, Cortex-M7 **@480 MHz**, SRAM 512 KB) | N-MNIST: 최적화 전 **수 초/샘플**, 최적화 시 **수백 ms/추론** 전망. 모델 풋프린트 가중치 **~50 KB** + 뉴런 상태·버퍼 **~200 KB**. ST-MNIST 피크 메모리 pruned **0.46 MB** / baseline **0.58 MB**. 정확도 N-MNIST **84.20 %**(baseline) → **84.60 %**(pruning 후) | C |
| 동 논문 계열 서술 | Cortex-M4F (**STM32L412KB @80 MHz**, SRAM 40 KB, Flash 128 KB) | 커모디티 MCU 엣지 배포 대표 사례로 사용 | C |
| 동 계열 | Cortex-M7 MCU @600 MHz | 추론 지연 **~350 µs** (경량 SNN, ECG/EEG 4 ms 예산 대비 충분) | C — 모델/조건 불명확 |
| 본 프로젝트 (참고) | STM32F746NGH6 @200 MHz | LeNet-5급 SNN(94,666 param, T=30, rate coding, CIFAR-10) 퓨전 커널 **77.9 ms/step, 2,407 ms/이미지**, 유효 **72.5 MMAC/s** | D (자체 실측) |

**공통 관찰** [C/D]
- SNN을 범용 MCU에서 돌리면 **뉴런 상태(막전위) 저장이 메모리 병목**이 된다(위 arXiv 사례: 가중치 50 KB vs 상태·버퍼 200 KB).
- SNN의 이론적 장점(AC-only, sparsity)은 **범용 Cortex-M에서는 거의 실현되지 않는다** — Cortex-M에는 이벤트 구동 스킵 하드웨어가 없어 dense MAC과 동일한 사이클을 쓴다. Loihi 대비 에너지 53.8× 차이가 이를 뒷받침한다. [C/D]
- 타임스텝 T배만큼 연산이 선형 증가하므로(본 프로젝트 T=30), ANN 대비 MCU에서는 오히려 불리하다. [D]

---

## 출처

### 1차 (ST / ARM 공식 문서 — 직접 본문 확인)

| 문서번호 | 제목 | URL / 경로 | 확보 방식 |
|---|---|---|---|
| **ARM DDI 0489F** | Arm Cortex-M7 Processor Technical Reference Manual r1p2 | https://www.pjrc.com/teensy/DDI0489F_cortex_m7_trm.pdf (원본: https://developer.arm.com/documentation/ddi0489/f) | PDF 전문 추출 |
| **DS10916 (DocID027590 Rev 1)** | STM32F745xx/STM32F746xx 데이터시트 | https://www.waveshare.com/w/upload/9/9b/STM32F745-STM32F746-Datasheet.pdf (원본: https://www.st.com/resource/en/datasheet/stm32f746ng.pdf, 현행 Rev 5) | PDF 전문 추출 |
| **AN4839 Rev 2** | Level 1 cache on STM32F7 Series and STM32H7 Series | https://iq.direct/datasheets/STM32x7%20Cache%20Appnote.pdf (원본: https://www.st.com/resource/en/application_note/an4839-...) | PDF 전문 추출 |
| — | Arm CMSIS-NN 소스 (v6.0.0 이전) | `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/third_party/cmsis_nn/` | 로컬 소스 직접 확인 |

### 1차 (논문 / Arm 공식 채널)

- **arXiv:1801.06601** — Lai, Suda, Chandra, *CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs* — https://arxiv.org/pdf/1801.06601 (PDF 전문 추출, Table 1·2 및 §4.1·4.2·4.3 확인)
- Arm AI Blog — *Image recognition on Arm powered microcontrollers* — https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/image-recognition-on-arm-powered-microcontrollers
- ARM-software/CMSIS_5 Issue #1380 — *CMSIS-NN: arm_nn_vec_mat_mult_t_s16 may overflow* — https://github.com/ARM-software/CMSIS_5/issues/1380
- ARM-software/CMSIS-NN Releases (v6.0.0에서 `arm_convolve_fast_s16` 제거) — https://github.com/ARM-software/CMSIS-NN/releases
- CMSIS-NN v6.0.0 API 문서 (NNConv 그룹) — https://arm-software.github.io/CMSIS-NN/v6.0.0/group__NNConv.html
- CMSIS-Core SIMD intrinsics (`__SMLAD`/`__SMLALD` 정의) — https://arm-software.github.io/CMSIS_6/latest/Core/group__intrinsic__SIMD__gr.html

### 2차

- STMicroelectronics/stm32ai-perf (MLPerf Tiny for STM32, 지원 보드 목록) — https://github.com/STMicroelectronics/stm32ai-perf
- Plumerai Blog — *MLPerf Tiny 1.0 results* (MLCommons 결과 재정리) — https://blog.plumerai.com/2022/11/mlperf-tiny-1.0/
- ST Community — *Latest MLPerf Tiny benchmark results: your STM32 is AI-ready* — https://community.st.com/t5/developer-news/latest-mlperf-tiny-benchmark-results-your-stm32-is-ai-ready/ba-p/842111
- ST Community — *How to run a CoreMark benchmark on an STM32* (STM32H563 = 1023 CoreMark / 4.092 CoreMark/MHz) — https://community.st.com/t5/stm32-mcus/how-to-run-a-coremark-benchmark-on-an-stm32/ta-p/705849
- arXiv:2511.12136 — *Compression and Inference of Spiking Neural Networks on Resource-Constrained Hardware* — https://arxiv.org/html/2511.12136
- ACM DL 10.1145/3594738.3611369 — *Evaluating Spiking Neural Network On Neuromorphic Platform For Human Activity Recognition* — https://dl.acm.org/doi/fullHtml/10.1145/3594738.3611369 (본문 403, 검색 스니펫만)
- EEMBC CoreMark scores — https://www.eembc.org/coremark/scores.php (JS 로딩으로 표 추출 실패)

### 로컬 프로젝트 자료 (자체 실측, 등급 D)

- `/Users/chacha/Desktop/git/snn-for-mcu/EXPERIMENTS.md` — V1~V4 지연/메모리 실측
- `/Users/chacha/Desktop/git/snn-for-mcu/paper_ieie/06_Envelope.tex` — MACs/step, envelope 스케일링
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/ImageClassification.ioc` — 클럭·FMC 설정 (HCLK 200 MHz, SDCLK = HCLK/2, CAS 3)
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Core/Src/snn_conv_pool_lif_s16.c` — `__SMLALD` 퓨전 커널

---

## 미확인 / 추가 조사 필요

| # | 항목 | 상태 | 이유 |
|---|---|---|---|
| 1 | **AN4667 본문 수치** (ITCM/AXI-Flash/SRAM/SDRAM/QuadSPI별 CoreMark 및 실행시간 표) | 미확보 | st.com PDF WebFetch 60 s 타임아웃, curl 120 s 타임아웃(ST 서버가 non-browser 차단), studylib 미러 HTTP 429 |
| 2 | **RM0385 §3.3 Flash 대기 상태 표** (216 MHz에서 정확한 WS 값, VOS/전압별) | 미확인 ([C] 수준으로 "7 WS" 추정만) | RM0385 PDF(20 MB+) 접근 실패 |
| 3 | **캐시 미스 페널티 사이클 수** (STM32F746 Flash / SRAM1 / SDRAM 각각) | 미확인 | ARM DDI 0489F에 명령어·메모리 타이밍 표 자체가 없음; ST 문서에서도 미발견 |
| 4 | **EEMBC 인증 STM32F746 CoreMark 엔트리 존재 여부** | 미확인 | eembc.org/coremark/scores.php가 JS 렌더링 |
| 5 | **UM1907 원문의 SDRAM 상세** (SDCLK 상한, 실측 대역폭) | [C]까지만 | UM1907 PDF 직접 확인 실패, 커뮤니티/미러 요약만 |
| 6 | **ST 위키 `AI:STM32Cube.AI model performances` 표** (모델별 MACC / ms / RAM / Flash) | 미확보 | wiki.st.com WebFetch 타임아웃 |
| 7 | **STM32F746 + X-CUBE-AI의 구체 벤치마크 수치** | 존재하지 않는 것으로 판단 | ST 공식 벤치마크 대상 보드에 F7 미포함 |
| 8 | **CMSIS-NN 논문 Fig.10(CHW vs HWC)의 축 눈금 값** | 미확인 | PDF 벡터 그래프라 텍스트 추출 불가 |
| 9 | **`SMLALD`의 정확한 사이클 처리량** | [D] (1/cycle로 추정) | DDI 0489F에 명령어별 사이클 표가 없어 "single MAC pipeline, 1 MAC/cycle throughput" 서술로부터 유도만 가능 |
| 10 | **로컬 CMSIS-NN의 정확한 릴리스 버전** (v4.1.0 / v5.0.0 중 어느 것인지) | 미확인 | 트리에 버전 매크로·CHANGELOG 없음. `arm_convolve_fast_s16` 존재 → v6.0.0 미만인 것만 확정 |
| 11 | **SNN-on-MCU 벤치마크의 통일된 비교표** | 부분적 | 공개 문헌이 각기 다른 MCU·모델·데이터셋을 쓰고 F746 사례가 거의 없음 |
| 12 | **캐시 ON/OFF 시 동일 워크로드 실행시간 배수** | 미확인 | 데이터시트는 전류만 제공, 시간 비교 없음 (AN4667에 있을 가능성 — 항목 1과 동일 사유) |

> **실측으로 갱신된 항목 (2026-07-27)**
>
> - **#2 (Flash 대기 상태)**: 이 보드에서 **6 WS @200 MHz**로 실측 확정(`FLASH_ACR = 0x00000006`). 216 MHz 값은 여전히 미확인.
> - **#3 (캐시 미스 페널티)**: **해결** — D-cache 히트 2.0 cyc / SRAM1 미스 8~9 cyc / SDRAM 랜덤 21~26 cyc /
>   Flash AXIM 14.34 cyc / Flash ITCM 별칭 9.13 cyc / QSPI 94~198 cyc. (§2.3 실측 정정 표 참조)
> - **#1 (AN4667의 메모리별 성능 비교)**: 문서 확보에는 실패했지만, **같은 성격의 데이터를 이 보드에서 직접 측정**했다 —
>   §6.1 실측 정정 표(ITCM/DTCM/SRAM/Flash AXIM/Flash ITCM 별칭/SDRAM/QSPI 대역폭 + ART ON/OFF 비교).
> - **#12 (캐시 ON/OFF 배수)**: 메모리 접근 수준에서는 측정됨(SRAM1 read32 172.85 → 206.32 MB/s, ×1.19;
>   Flash AXIM 123.02 → 193.86 MB/s, ×1.58). **추론 워크로드 전체의 배수는 여전히 미측정.**
> - **#9 (SMLALD 사이클 처리량)**: 여전히 미측정. 측정 방법은 확보됨(ITCM 상주 루프 + DWT CYCCNT, `NOCYCCNT=0` 확인).
> - **추가로 드러난 항목**: 이 보드는 **ART/프리페치가 꺼진 채**(그리고 ITCM 별칭 부팅으로 I-cache 미적용) 동작 중이었다.
>   문서의 §5 로컬 실측 수치 전부가 그 조건에서 나왔다는 사실은 원래 문서에 없었다.
