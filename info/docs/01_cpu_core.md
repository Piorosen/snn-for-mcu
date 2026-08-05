# STM32F746NGH6 Cortex-M7 (r0p1) 코어 마이크로아키텍처

> 대상: STM32F746G-DISCO / STM32F746NGH6 / Arm Cortex-M7 r0p1 / 216 MHz
> 조사일: 2026-07-27
> 각 사실 뒤 `[A]`~`[D]`는 신뢰도 등급, 괄호는 출처(문서번호 + 섹션).

> **실측 정정**: 이 보드가 **실제로 도는 클럭은 200 MHz**다(216 MHz가 아니다).
> `RCC_PLLCFGR = 0x29406419` → PLLM=25, PLLN=400, PLLP=2, PLLSRC=HSE(25 MHz) → VCO 400 MHz → SYSCLK 200 MHz.
> `RCC_CFGR = 0x0000940A` → HPRE=/1, PPRE1=/4, PPRE2=/2 → **PCLK1 50 MHz, PCLK2 100 MHz**.
> 온디바이스 펌웨어도 `sysclk_hz = 200000000`을 보고했다. 216 MHz는 데이터시트 최댓값일 뿐
> 이 보드의 동작점이 아니므로, 본문의 216 MHz 기준 환산값은 전부 200 MHz로 다시 읽어야 한다.

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거: `info/raw/measured_facts.md`, `info/results/probe_output.txt`,
> `info/raw/decoded_hotplug.md`, `info/raw/decoded_reset.md`, `info/raw/host_side.md`

### 실측으로 확인됨

| 항목 | 문서값 | 실측값 |
|---|---|---|
| CPUID / 코어 리비전 | `0x410FC271` = Cortex-M7 **r0p1** | `0x410FC271` (ST-LINK, 0xE000ED00) — **일치** |
| DBGMCU DEV_ID / REV_ID | `0x449` / Rev Z (등급 [D], 대조 미완) | `DBGMCU_IDCODE = 0x10016449` → DEV_ID `0x449`, **REV_ID `0x1001` = Rev Z** — 확정 |
| Flash 용량 | 1 MB | Flash size register(0x1FF0F442) = `0x0400` → **1024 KB** |
| NVIC 인터럽트 라인 수 | ICTR.INTLINESNUM = `0b0011` (97–128 구간) | `ICTR = 0x00000003` → 128 라인 |
| 우선순위 비트 수 | 4비트(16레벨), `__NVIC_PRIO_BITS = 4` | `NVIC->IP[0]`에 0xFF 기록 후 되읽기 = **0xF0 → 상위 4비트만 구현** (하드웨어로 직접 확인) |
| MPU 리전 수 | 8 (「널리 8로 알려져 있으나 1차 문서 미확인」 [C]) | `MPU_TYPE = 0x00000800` → **DREGION=8, SEPARATE=0** — 확정 |
| FPU 구성 | FPv5-**SP**-D16-M, 배정밀도 없음 | `MVFR0 = 0x10110021` (DP 필드=0), `MVFR1 = 0x11000011`, `MVFR2 = 0x00000040` — 표의 예상값과 **완전 일치** |
| MVFR1 FMA / half-precision 변환 | 지원 [A] | `MVFR1[31:28]=1`(FMA), `[27:24]=1`(VFP HP 변환). 툴체인도 `__ARM_FEATURE_FMA = 1` |
| FPCCR 레이지 스태킹 기본 활성 | 리셋값 `0xC0000000` (ASPEN=1, LSPEN=1) | 실행 중 `FPCCR = 0xC0000058` → **ASPEN=1, LSPEN=1** 확인 |
| CPACR는 소프트웨어가 켜야 함 | FPU 사용 전 CP10/CP11 활성 필수 | 리셋 직후 `CPACR = 0x00000000`, 실행 중 `0x00F00000` |
| CCR.STKALIGN = RO, 값 1 | [A] | 리셋 직후 `SCB_CCR = 0x00040200` → bit9(STKALIGN)=1 |
| CCR.BP는 RAO/WI (분기예측 상시 활성) | [A] | 리셋 직후에도 `CCR` bit18(BP)=1 |
| ACTLR 리셋값 0 (성능 저하 비트 없음) | [A] | 리셋·실행 양쪽 `ACTLR = 0x00000000` |
| DWT 비교기 수 | Full DWT = 4개 (F746 실장 여부는 미확인 [D]) | TRCENA=1 상태에서 `DWT_CTRL = 0x40000001` → **NUMCOMP=4**, `NOCYCCNT=0`, `NOPRFCNT=0` |
| FPB 브레이크포인트 수 | 4개 또는 8개 (미확인) | `FP_CTRL = 0x10000080` → **NUM_CODE=8, NUM_LIT=0, REV=1(FPBv2)** |
| DWT는 TRCENA를 켜야 보인다 | §10.5 절차 [A]+[C] | TRCENA=0인 덤프에서 `DWT_CTRL`이 0으로 읽혔고, 켜자 `0x40000001`이 나왔다 |
| ETM/TPIU 존재 | 구현 옵션, F746 실장 여부 미확인 | **물리적으로 존재**: TPIU(0xE0040000) PIDR part=`0x9A9`, ETM(0xE0041000) part=`0x975`, 제조사 ARM(0x3B). 단 ROM table은 present=0으로 광고하지 않는다 |
| ISA / DSP 확장 | ARMv7E-M + DSP, 하드웨어 나눗셈 | `ID_ISAR` 판독: `SIMD_instrs=3`(패킹 SIMD), `MultS=3`(SMLAD/SMUAD), `Saturate=1`, `Divide=1`, `UMAAL` 지원 |
| L1 캐시 용량 (§11) | I 4 KB + D 4 KB | `CCSIDR(D)=0xF003E019`(32set×4way×32B), `CCSIDR(I)=0xF007E009`(64set×2way×32B) |
| ITCM 16 KB / DTCM 64 KB / 총 SRAM 320 KB (§9.2) | DS10916 [B] | ITCM: `0x00004000`부터 버스 폴트 → **16 KB**. DTCM 64 KB + SRAM1 240 KB + SRAM2 16 KB = **320 KB** |
| AHBP 포트 크기 (§11 「64/128/256/512 MB 선택」) | 미확인 | `AHBPCR = 0x00000009` → **EN=1, SZ=0b100 = 512 MB** |
| CoreSight 컴포넌트 주소 맵 (§10.1) | ITM/DWT/FPB/TPIU/ETM 주소 | 각 주소의 PIDR/CIDR 판독으로 전부 확인 (CIDR 프리앰블 `0xB105...0D` 정상) |

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

**1. 동작 클럭 216 MHz (문서 전반)**
- 문서 주장: 헤더·§4.4(432 MMAC/s)·§7.4(55.6 ns)·§9.2·§9.3 전부 **216 MHz** 기준.
- 실측: **SYSCLK = HCLK = 200 MHz**, PCLK1 50 MHz, PCLK2 100 MHz (`RCC_PLLCFGR = 0x29406419`).
- 원인: 216 MHz는 **데이터시트 최대 주파수**이고, 이 보드의 펌웨어가 실제로 설정한 값은 200 MHz다.
  문서가 "정격 최대 = 동작점"이라고 암묵 가정한 것이 어긋남의 원인이다. 문서 오류라기보다 전제 오류.
- 파급: 사이클→시간 환산, MMAC/s 상한, 인터럽트 레이턴시 ns 값이 모두 **1.08배 길어진다**.

**2. 외부 인터럽트 "98개"** (§7.1)
- 문서 주장: 외부 인터럽트 **98개** (IRQ0 … IRQ97).
- 실측: 벡터 테이블 슬롯은 0~97로 **98칸**이 맞지만, ST CMSIS 헤더가 이름을 붙인 IRQ는 **97개**다
  (**IRQ 79번이 결번**). 총 벡터 엔트리 105개.
- 원인: "번호의 최댓값"과 "실제로 배선된 인터럽트 수"를 같게 본 것. 큰 문제는 아니나
  `NVIC_ISER` 비트를 순회하는 코드를 짤 때 79번이 비어 있음을 알아야 한다.
- 심각도: 낮음.

**3. `0xE0042000` = CTI** (§10.1 디버그 주소 맵)
- 문서 주장: `0xE0042000`–`0xE0042FFF` = **CTI (Cross Trigger Interface)**.
- 실측: 이 주소를 읽으면 **`DBGMCU_IDCODE = 0x10016449`** 가 나온다. ROM table(0xE00FF000)에도 CTI 엔트리가 없다.
- 원인: 문서는 **ARM 코어의 일반 주소 맵**을 옮겨 적었고, ST는 같은 주소에 DBGMCU를 매핑했다.
  문서 §미확인-13이 이미 이 겹침을 의심하고 있었는데, 실측으로 **ST의 DBGMCU가 이긴다**가 확정됐다.
- 심각도: 중간 (CTI를 쓰려고 이 주소를 건드리면 엉뚱한 레지스터를 만진다).

**4. "임베디드 Flash / 내부 RAM / 외부 메모리 어디서 실행해도 동일 CoreMark 스코어"** (§9.2)
- 문서 주장: AN4667 인용 [B].
- 실측(메모리 대역폭 기준): 내부 SRAM/TCM 순차 읽기 **205~220 MB/s**(LDM 390~460)인 데 반해
  **외부 SDRAM은 60.25 MB/s**(1/3.5), **QSPI는 51.7 MB/s**다. 랜덤 접근 레이턴시도
  SRAM1 8 cyc vs SDRAM 21~26 cyc로 3배 이상 차이난다.
- 원인: ST의 주장은 **명령 페치**가 ART/캐시로 가려진다는 뜻이고, 여기 실측은 **데이터 접근**이다.
  둘은 같은 척도가 아니다. 다만 데이터가 외부 메모리에 있는 워크로드(SNN 가중치 등)에는
  "어디서든 동일"이 성립하지 않으므로, 이 문장을 그대로 인용하면 오해를 부른다.
  **CoreMark 자체를 이 보드에서 돌려 본 적은 없다(미측정).**
- 심각도: 낮음(조건부).

### 아직 측정하지 않음

- **명령어별 사이클 표(§4.3–4.5, §5.3)** — LLVM 스케줄 모델 [C] 그대로다. `.itcm_code`에 명령을 언롤한 루프를 올리고 DWT_CYCCNT로 명령당 사이클을 재면 확정된다.
- **듀얼 이슈 실효 폭(§2.5)** — 듀얼 이슈 가능 쌍 / 불가 쌍을 각각 N회 반복한 루프의 CYCCNT를 비교.
- **분기 오예측 페널티·BTAC 엔트리 수·리턴 스택(§3)** — 분기 개수를 늘려 가며 루프당 CYCCNT의 계단을 찾는다.
- **인터럽트 레이턴시 12 / tail-chaining 6 사이클(§7.4)** — SW 트리거 인터럽트(`NVIC_SetPendingIRQ`) 직전 CYCCNT와 ISR 첫 줄 CYCCNT의 차.
- **언얼라인 액세스 페널티(§6.3)** — 프로브 펌웨어가 §8 "정렬 페널티" 측정을 시도했으나 출력 로그가 **4B 정렬 기준선 행(SRAM1 9.03 cyc / SDRAM 19.15 cyc)에서 잘려 있어** 비정렬 행이 없다. 같은 루프의 오프셋만 1/2/3바이트로 바꿔 재실행하면 바로 나온다.
- **`DWT_LAR = 0xC5ACCE55` 언락이 실제로 필요한지(§10.5)** — 프로브 펌웨어가 무조건 써 버려서 판별 불가. LAR 쓰기를 빼고 CYCCNT가 증가하는지 보면 된다.
- **FPU 연산 레이턴시(§5.3)** — `VADD/VMUL/VFMA/VDIV.F32` 의존 사슬을 만들어 CYCCNT로 측정.
- **DWT 프로파일 카운터 5종의 실제 동작(§10.3)** — 존재는 확인됐다(`NOPRFCNT=0`). CPICNT/LSUCNT/FOLDCNT를 켜고 알려진 명령열을 돌려 8비트 오버플로 주기까지 확인.
- **예외 스택 프레임 크기(§7.3)** — FP 컨텍스트 유/무로 ISR 진입 전후 SP 차를 읽는다.
- **ES0290 에라타 항목** — ST 문서 미확보. 실측 대상이 아니라 문서 확보 문제.

---

## 요약

- **CPUID = `0x410FC271`** 는 ARM DDI 0489F §1.7 "Product revisions" 표에서 **r0p1** 의 리셋값으로 명시되어 있다. VARIANT=`0x0`(major r0), REVISION=`0x1`(patch p1). ST의 CMSIS 디바이스 헤더도 `__CM7_REV = 0x0001` 로 일치한다. [A]
- **6단 in-order dual-issue 슈퍼스칼라 파이프라인**. DPU는 정수 레지스터 파일에 **읽기 6포트 / 쓰기 4포트**, **ALU 2개(그중 1개만 SIMD 실행 가능)**, **MAC 파이프 1개(32×32+64→64, 결과 레이턴시 2사이클, 처리율 1 MAC/사이클)**, **디바이더 1개(피연산자 의존 조기 종료)** 를 가진다. [A] (DDI 0489F §1.2.1)
- **PFU: 명령어 페치 대역폭 64비트, 4×64비트 프리페치 큐, BTAC(단일 사이클 턴어라운드) + 정적 예측기 폴백.** BTAC 엔트리 수·리턴 스택 유무는 공개 문서에 없음(미확인). [A]/미확인 (DDI 0489F §1.2.2)
- **LSU: 32비트 로드 채널 2개(합계 64비트 로드 대역폭) + AHBP용 32비트 로드 채널 1개 + 64비트 스토어 채널 1개.** **스토어 버퍼(STB)는 4엔트리, 엔트리당 데이터 64비트 + 주소 32비트**, 같은 캐시 라인이면 머지 가능. [A] (DDI 0489F §1.2.3, §5.9.2)
- **F746의 FPU는 FPv5-SP-D16-M (단정밀도 전용)** — 컴파일 옵션 `-mfpu=fpv5-sp-d16`. 레지스터 뱅크는 S0–S31(32×32비트) = D0–D15(16×64비트) **하나뿐**이며 FPv5에서 배정밀도가 추가돼도 레지스터는 늘지 않는다. 레이지 스태킹은 FPCCR 리셋값 `0xC0000000`(ASPEN=1, LSPEN=1)으로 **기본 활성**. [A]
- **NVIC: F746은 우선순위 4비트(16레벨), 외부 인터럽트 라인 98개(IRQ0…IRQ97, `SPDIF_RX_IRQn = 97`)**. 코어 자체는 최대 240개 / 최대 8비트(256레벨)까지 지원. tail-chaining · late-arriving 지원. [A]/[B]
- **MPU: ARMv7 PMSA, 구현 시 8 또는 16 리전 선택**(MPU_TYPE 리셋값 `0x00000800`=8리전, `0x00001000`=16리전). 리전당 **8개 서브리전(SRD[7:0])**, 속성은 TEX[21:19]/S[18]/C[17]/B[16]/AP[26:24]/XN[28]. [A]
- **성능: ARM 공식 비교표 기준 2.31 DMIPS/MHz, 5.29 CoreMark/MHz.** ST가 STM32F746(216 MHz)에 대해 공표하는 값은 **462 DMIPS(2.14 DMIPS/MHz), 1082 CoreMark(≈5.01 CoreMark/MHz)** — 두 수치가 다른 이유는 컴파일러/구성 차이. [A]/[B]
- **⚠ 중요: Cortex-M7은 명령어별 사이클 수 표가 공식 문서에 존재하지 않는다.** DDI 0489F에는 instruction timing 부록이 없고, DUI 0646B Table 3-1(Cortex-M7 instructions)에는 Cortex-M4 GUG와 달리 **Cycles 열 자체가 없다**. 따라서 아래 사이클 표는 LLVM `ARMScheduleM7.td` 기반 `[C]` 이며, 실측 검증이 필요하다. [A](없음을 확인) / [C](표 값)
- **DWT 카운터 6종(CYCCNT/CPICNT/EXCCNT/SLEEPCNT/LSUCNT/FOLDCNT) 주소는 `0xE0001004`~`0xE0001018` 연속**. CYCCNT만 32비트이고 나머지 5개는 **8비트 카운터**(오버플로 시 이벤트 발생). Cortex-M7에서는 소프트웨어로 쓸 때 `DWT_LAR(0xE0001FB0) = 0xC5ACCE55` 언락이 추가로 필요하다. [A]/[C]

---

## 1. 코어 식별 및 리비전

### 1.1 CPUID Base Register (`0xE000ED00`) = `0x410FC271` 완전 해석

DDI 0489F §3.3.2 Table 3-4 기준. [A]

| 비트 | 필드 | 값 | 의미 |
|---|---|---|---|
| [31:24] | IMPLEMENTER | `0x41` | Arm |
| [23:20] | VARIANT | `0x0` | Revision 0 → 메이저 리비전 **r0** |
| [19:16] | ARCHITECTURE | `0xF` | 항상 `0xF` (CPUID scheme으로 정의) |
| [15:4] | PARTNO | `0xC27` | **Cortex-M7** |
| [3:0] | REVISION | `0x1` | Patch 1 → 마이너 리비전 **p1** |

→ **`0x410FC271` = Cortex-M7 r0p1** [A]

### 1.2 리비전별 CPUID 리셋값 (DDI 0489F §1.7) [A]

| 리비전 | CPUID 리셋값 | 비고 |
|---|---|---|
| r0p0 | (문서에 미기재) | 최초 릴리스 |
| **r0p1** | **`0x410FC271`** | **STM32F746NGH6 해당**. 각종 엔지니어링 에라타 수정 |
| r0p2 | `0x410FC272` | 에라타 수정 |
| r1p0 | `0x411FC270` | `CTLPPBLOCK[3:0]` 추가(ITCMCR/DTCMCR/AHBPCR/VTOR 잠금), ACTLR에 AXI 관련 비트 추가 |
| r1p1 | `0x411FC271` | — |
| r1p2 | `0x411FC272` | speculative access 절 추가 |

> 주의: `0x411FC271`(r1p1)과 `0x410FC271`(r0p1)은 **VARIANT 니블 1자리만 다르다**. 혼동 주의. [A]

### 1.3 ST 측 식별자 [A]/[B]

| 항목 | 값 | 출처 |
|---|---|---|
| `__CM7_REV` | `0x0001` (= Cortex-M7 revision r0p1) | `stm32f746xx.h` L166 [A] |
| `__MPU_PRESENT` | `1` | `stm32f746xx.h` L167 [A] |
| `__NVIC_PRIO_BITS` | `4` (16 우선순위 레벨) | `stm32f746xx.h` L168 [A] |
| `__FPU_PRESENT` | `1` | `stm32f746xx.h` L170 [A] |
| `__ICACHE_PRESENT` / `__DCACHE_PRESENT` | `1` / `1` | `stm32f746xx.h` L171–172 [A] |
| `__FPU_DP` | **정의 없음** → 단정밀도 전용 | `stm32f746xx.h` [A] |
| `__CORTEX_M` | `7` | `core_cm7.h` L71 [A] |
| DBGMCU Device ID | `0x449` (실측), Revision ID = Rev Z | 사용자 실측 (RM0385 대조 미완) [D] |

> **실측 정정(확인)**: `DBGMCU_IDCODE @ 0xE0042000 = 0x10016449` → DEV_ID `0x449`,
> **REV_ID `0x1001`**. CubeProgrammer 배너의 "Rev Z"와 일치하므로 **Rev Z = REV_ID 0x1001**로 확정.
> 문서 §미확인-13의 물음(REV_ID 코드값)이 여기서 해소된다. 등급 [D] → **[A]**.

> 파일 경로: `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h`

---

## 2. 파이프라인 구조

### 2.1 개요

- **6단 in-order dual-issue 슈퍼스칼라 파이프라인 + 동적 분기 예측.** [A](Atmel-44047B §1.1) / DDI 0489F §1.1.1은 "in-order issue, super-scalar pipeline with dynamic branch prediction"으로만 기술하고 **단계 수를 명시하지 않는다.** [A]
- 다중 메모리 인터페이스 덕분에 **load/load, load/store 쌍도 듀얼 이슈 가능**. [A] (DDI 0489F §1.1)
- 부동소수점 파이프가 별도로 있어 **FP 명령과 정수 명령의 듀얼 이슈가 가능**. [C] (Atmel-44047B §1.1)

### 2.2 실행 스테이지 명칭

TRM은 ACTLR.DISDYNADD 설명에서 **EX1 / EX2** 를 직접 언급한다: "Some ADD and SUB instructions are resolved in EX1 / All ADD and SUB instructions are resolved in EX2". [A] (DDI 0489F §3.3.1 Table 3-3 bit[26])

LLVM Cortex-M7 스케줄 모델이 기술하는 스테이지 구성 [C]:

| 스테이지 | 수행 내용 |
|---|---|
| EX1 | 주소 생성(address generation), 시프트 |
| EX2 | 빠른 로드 데이터, ALU, FP 연산 |
| EX3 | 느린 로드 데이터, 정수 라이트백, FP 연산 |
| EX4 | 스토어 데이터, FP 라이트백 |

- EX1과 EX2 **양쪽에 시프터가 있고**, 일부 명령은 두 시프터 사이에서 유연하게 배치된다. [C]
- 스케줄링의 "0점"은 EX2이다 → 단순 ALU 연산 레이턴시 = 1. [C]

### 2.3 실행 유닛 구성

DDI 0489F §1.2.1 (DPU) [A] + LLVM 모델 [C]:

| 유닛 | 개수 | 출처 |
|---|---|---|
| 정수 레지스터 파일 포트 | 읽기 6 / 쓰기 4 (대규모 듀얼 이슈용) | [A] DDI 0489F §1.2.1 |
| ALU | **2개** (그중 **1개만 SIMD 연산 가능**) | [A] DDI 0489F §1.2.1 |
| MAC 파이프라인 | **1개** — 32×32비트 + 64비트 → 64비트, **결과 레이턴시 2사이클, 처리율 1 MAC/사이클** | [A] DDI 0489F §1.2.1 |
| 디바이더 | **1개**, 피연산자 의존 조기 종료(early termination) 지원 | [A] DDI 0489F §1.2.1 |
| 로드 유닛 | 2개(`M7UnitLoadL`, `M7UnitLoadH`) | [C] LLVM |
| 스토어 유닛 | 1개 | [C] LLVM |
| 시프터 | 2개(EX1용, EX2용) | [C] LLVM |
| 분기 유닛 | 1개 | [C] LLVM |
| VFP 유닛 | 1개 + VFP 포트 2개(`VPortL`, `VPortH`) | [C] LLVM |
| SIMD 유닛 | **1개** (ALU 1개를 함께 점유) | [C] LLVM |

> **SNN/CMSIS-NN 관점 시사점**: SIMD 유닛이 1개이고 ALU 하나를 같이 점유하므로, `SMLAD`·`__SXTB16` 같은 DSP 명령을 연속 배치해도 SIMD 처리율은 **최대 1개/사이클**이다. 반면 MAC 파이프는 별도라서 `SMLAD`(MAC 유닛)와 `SXTB16`(SIMD 유닛)의 인터리빙은 유효하다. [D] (LLVM 자원 모델에서 유도)

### 2.4 듀얼 이슈 조건 — ACTLR 로 역추론 (1차 출처)

TRM은 듀얼 이슈 규칙을 직접 나열하지 않지만, **ACTLR(`0xE000E008`)의 디버그/디스에이블 비트가 어떤 명령 타입이 채널별로 취급되는지를 드러낸다.** [A] (DDI 0489F §3.3.1 Table 3-3)

| ACTLR 비트 | 필드 | 의미 (1로 설정 시) |
|---|---|---|
| [25] | DISISSCH1 (VFP) | VFP 명령을 **채널 1**에 발행 금지 |
| [24] | DISISSCH1 (Integer MAC/MUL) | 정수 MAC·MUL을 채널 1에 발행 금지 |
| [23] | DISISSCH1 (Loads to PC) | PC로의 로드를 채널 1에 발행 금지 |
| [22] | DISISSCH1 (Indirect branches) | 간접 분기(PC 로드 제외)를 채널 1에 발행 금지 |
| [21] | DISISSCH1 (Direct branches) | 직접 분기를 채널 1에 발행 금지 |
| [20] | DISDI (VFP) | 이 타입이 **채널 0**에 있을 때 아무것도 듀얼 이슈 불가 |
| [19] | DISDI (Integer MAC/MUL) | 〃 |
| [18] | DISDI (Loads to PC) | 〃 |
| [17] | DISDI (Indirect branches) | 〃 |
| [16] | DISDI (Direct branches) | 〃 |

→ 정상 동작(=비트 0)에서는 **VFP / 정수 MAC·MUL / PC 로드 / 간접 분기 / 직접 분기 모두 채널 0·채널 1 양쪽에 발행 가능하며 듀얼 이슈 대상**이다. 즉 이 5종은 하드웨어가 듀얼 이슈를 지원한다는 것이 ACTLR 정의로부터 확정된다. [A]/[D](역추론)

기타 관련 ACTLR 비트 [A]:

| 비트 | 필드 | 의미 |
|---|---|---|
| [28] | DISFPUISSOPT | 0 = 정상 동작 |
| [27] | DISCRITAXIRUW | AXI critical read-under-write 비활성 |
| [26] | DISDYNADD | 1 = 모든 ADD/SUB를 EX2에서 해결(기본 0은 일부를 EX1에서 해결) |
| [15] | DISCRITAXIRUR | AXI critical read-under-read 비활성. **"최악 인터럽트 레이턴시를 개선할 수 있으나 성능은 저하"** 라고 명시 |
| [14] | DISBTACALLOC | BTAC 신규 엔트리 할당 금지(기존 엔트리 갱신은 허용) |
| [13] | DISBTACREAD | BTAC 사용 금지 → **정적 분기 예측만 동작** |
| [12] | DISITMATBFLUSH | 항상 1, RAO/WI |
| [11] | DISRAMODE | Write-Back Write-Allocate 영역의 동적 read-allocate 모드 비활성 |
| [10] | FPEXCODIS | FPU 예외 출력 비활성 |
| [2] | DISFOLD | 0 = 정상 동작 |

> **실측 정정(확인)**: `ACTLR = 0x00000000` — **리셋 직후와 실행 중 덤프가 동일**하다.
> 즉 DISFOLD·DISRAMODE·DISBTAC*·DISDI/DISISSCH1 계열이 전부 0이며, 듀얼 이슈·분기 예측·
> 동적 read-allocate·IT 폴딩이 모두 성능 우선 기본 상태로 동작한다.
>
> ⚠ 값이 0이라 **비트 위치는 실측으로 구분되지 않는다.** 프로브 로그(`probe_output.txt` §2)는
> DISFOLD를 bit0, FPEXCODIS를 bit2, DISRAMODE를 bit3, DISITMATBFLUSH를 bit4로 주석했는데,
> 이는 DDI 0489F Table 3-3(각각 bit2 / bit10 / bit11 / bit12)과 다르다.
> 어느 쪽도 측정으로 판별할 수 없으므로 **1차 문서인 이 표를 따르는 것이 맞다.**

### 2.5 듀얼 이슈 제약 (LLVM 스케줄 모델 기반) [C]

`IssueWidth = 2`, `MicroOpBufferSize = 0`(in-order), `LoadLatency = 2`(best case), `MispredictPenalty = 4`(모델값; 주석은 "forward branch 실제 오예측 비용은 6").

**듀얼 이슈 불가 (SingleIssue = 1) — 단독 발행:**

| 명령군 | 근거 |
|---|---|
| `SDIV` / `UDIV` | `WriteDIV { SingleIssue = 1 }` |
| `LDM`/`STM`/`PUSH`/`POP` (멀티 레지스터) | `M7SingleIssue` |
| `LDRD` / `STRD` (더블워드) | `M7SingleIssue` |
| `STREX`, `STREXB`, `STREXH` | `M7SingleIssue` |
| `TBB` / `TBH` (테이블 분기) | `M7TableLoad { SingleIssue = 1, NumMicroOps = 2 }` — 발행에 2사이클 |
| `VLDM`/`VSTM` | `M7SingleIssue` |
| `VLDRD`/`VSTRD` (배정밀도 단일 로드/스토어) | `M7SingleIssue` |
| `VMRS`/`VMSR`(FMSTAT) | `M7VMRS`/`M7VMSR { SingleIssue = 1 }` |
| `VMOV` GP↔FP 쌍(`VMOVDRR/RRD/RRS/SRR`) | `M7SingleIssue` |
| `MSR`/`MRS`, `SVC`, `BKPT`, `DMB`/`DSB`/`ISB`, `CLREX`, `HINT`(NOP/WFI 등) | `M7NonGeneralPurpose { Latency=3, SingleIssue=1 }` |

**슬롯 0 전용 (BeginGroup = 1) — 듀얼 이슈 쌍의 "두 번째"가 될 수 없음:**

| 명령군 |
|---|
| 모든 DSP/SIMD 명령(§4.2 표 전체) |
| `CLZ` |
| `BFC` / `BFI` |
| `LDREX` / `LDREXB` / `LDREXH` |
| `VSEL.*` |
| `VMOVD`, `FCONSTD` |
| 배정밀도 VFP 연산 전반(`VADDD`, `VMULD`, `VDIVD`, `VSQRTD`, `VMLAD`, `VCMPD` 등) |

---

## 3. 분기 예측

| 항목 | 값 | 등급/출처 |
|---|---|---|
| 예측기 종류 | **BTAC (Branch Target Address Cache)** — 분기 예측기 상태와 타깃 주소를 **단일 사이클 턴어라운드**로 제공 | [A] DDI 0489F §1.2.2 |
| 폴백 | BTAC 미구현/비활성 시 **정적 분기 예측기** | [A] DDI 0489F §1.2.2, §3.3.1 bit[13] |
| BTAC 예측 대상 | 분기 + **BL(Branch-Link) 포함**, 단순 상태 모델로 분기 흐름 예측 | [C] 2차 자료 |
| 조기 해결 | 디코더 및 첫 실행 스테이지에서 **직접 분기의 조기 해결을 위한 플래그 포워딩** | [A] DDI 0489F §1.2.2 |
| 상시 활성 여부 | **CCR.BP는 RAO/WI** — 분기 예측기는 프로세서에서 항상 활성 | [A] DDI 0489F §3(memory system 절 CCR 언급) |
| 소프트웨어 제어 | ACTLR[14] DISBTACALLOC(할당 금지), ACTLR[13] DISBTACREAD(BTAC 미사용) | [A] DDI 0489F §3.3.1 |
| 오예측 페널티 | 순방향 분기 **6사이클**(LLVM 주석), 모델 채택값 4 | [C] LLVM `ARMScheduleM7.td` |
| **BTAC 엔트리 수** | **미확인** — 공개 TRM/GUG/데이터시트 어디에도 없음. Integration & Implementation Manual(라이선시 전용) 소관으로 추정 | 미확인 |
| **리턴 스택(RAS) 유무/크기** | **미확인** — DDI 0489F §1.2.2 PFU 기능 목록에 리턴 스택 항목이 **없다**. 있다고 단정할 근거 없음 | 미확인 |

### 프리페치 유닛(PFU) 사양 [A] (DDI 0489F §1.2.2)

| 항목 | 값 |
|---|---|
| 명령어 페치 대역폭 | **64비트** (32비트 명령 2개/사이클) |
| 프리페치 큐 | **4 × 64비트** — 명령어 프리페치를 DPU 파이프라인 동작과 디커플링 |

> ITCM 인터페이스가 64비트라서 **32비트 명령 2개를 한 번에 페치**할 수 있고, 이것이 듀얼 이슈의 전제가 된다. [C] (Atmel-44047B §1.2)

---

## 4. 명령어 집합 (ARMv7E-M / Thumb-2 / DSP)

### 4.1 ISA 구성

DDI 0489F §8.1 Table 8-1 [A]:

| 구성 | 지원 ISA |
|---|---|
| FPU 없음 | v7E-M |
| **단정밀도 FPU** | **v7E-M + FPv5-SP-D16-M** ← **STM32F746 해당** |
| 단정밀도 + 배정밀도 FPU | v7E-M + FPv5-DP-D16-M |

- ARMv7-M Thumb 명령 집합 (Arm®v7-M Architecture Reference Manual, DDI 0403 정의) [A]
- **DSP 확장 포함** [A] (DDI 0489F §1.1.1)
- **하드웨어 정수 나눗셈 `SDIV`/`UDIV`** [A]
- 뱅크드 SP(MSP/PSP), Handler/Thread 모드, Thumb/Debug 상태 [A]
- ARMv7-M **big-endian byte-invariant 또는 little-endian** 접근 지원 (STM32는 little-endian) [A]
- 예외 계속 가능(exception-continuable) 명령: `LDM`, `LDMDB`, `STM`, `STMDB`, `PUSH`, `POP` + FPU 있을 때 `VLDM`, `VSTM`, `VPUSH`, `VPOP` [A]

### 4.2 ⚠ 명령어 사이클 수에 대한 중대한 주의

> **Cortex-M7에는 공식 명령어 사이클 표가 존재하지 않는다.**
> - `ARM DDI 0489F` 전체를 텍스트 추출해 확인한 결과 **instruction timing 부록이 없다** (Appendix는 A "Revisions" 하나뿐). [A]
> - `ARM DUI 0646B` §3.1 Table 3-1 "Cortex-M7 instructions" 의 열 구성은 **Mnemonic / Operands / Brief description / Flags / Page** 로, Cortex-M4 Generic User Guide에 있던 **Cycles 열이 없다.** [A]
> - Arm 커뮤니티에서도 "M7은 듀얼 이슈·캐시·분기 예측 때문에 단순 사이클 수를 제시할 수 없다"는 취지의 답변이 반복된다. [C]
>
> 따라서 아래 표의 사이클 수는 **LLVM 컴파일러의 Cortex-M7 스케줄 모델(`ARMScheduleM7.td`)** 에서 가져온 것으로 등급 **[C]** 이며, 실제 코드에서는 DWT_CYCCNT 로 실측 검증할 것.

### 4.3 DSP / SIMD 명령 레이턴시 표 [C] (LLVM `ARMScheduleM7.td`)

레이턴시는 EX2 기준. `BeginGroup=1`은 **듀얼 이슈 쌍의 두 번째 슬롯이 될 수 없음**(= 슬롯 0 전용)을 뜻한다.

| 명령 | 점유 자원 | 레이턴시(사이클) | 발행 제약 |
|---|---|---|---|
| `SXTB`, `SXTH`, `UXTB`, `UXTH` | SIMD + ALU | **1** | 슬롯 0 전용 |
| `SADD16`, `SSUB16`, `SADD8`, `SSUB8`, `SASX`, `SSAX` | SIMD + ALU | **1** | 슬롯 0 전용 |
| `UADD16`, `USUB16`, `UADD8`, `USUB8`, `UASX`, `USAX` | SIMD + ALU | **1** | 슬롯 0 전용 |
| `SHADD16/8`, `SHSUB16/8`, `SHASX`, `SHSAX`, `UHADD*`, `UHSUB*` | SIMD + ALU | **1** | 슬롯 0 전용 |
| **`SEL`** | SIMD + ALU | **1** | 슬롯 0 전용 |
| `QADD16`, `QSUB16`, `QADD8`, `QSUB8`, `QASX`, `QSAX` | SIMD + ALU | **2** | 슬롯 0 전용 |
| `UQADD16`, `UQSUB16`, `UQADD8`, `UQSUB8`, `UQASX`, `UQSAX` | SIMD + ALU | **2** | 슬롯 0 전용 |
| `QADD`, `QSUB` (스칼라 포화) | SIMD + ALU | **2** | 슬롯 0 전용 |
| `QDADD`, `QDSUB` | SIMD + ALU + Shift1 | **2** | 슬롯 0 전용 |
| **`SSAT`, `USAT`** (`SSAT16`/`USAT16` 포함) | SIMD + ALU + Shift1 | **2** | 슬롯 0 전용 |
| **`PKHBT`, `PKHTB`** | SIMD + ALU + Shift1 | **1** | 슬롯 0 전용 |
| `SXTAB`, `SXTAH`, `UXTAB`, `UXTAH`, `SXTAB16`, `UXTAB16` | SIMD + ALU + Shift1 | **1** | 슬롯 0 전용 |
| `SBFX`, `UBFX` | SIMD + ALU + Shift1 | **1** | 슬롯 0 전용 |
| `RBIT`, `REV`, `REV16`, `REVSH` | SIMD + ALU + Shift1 | **0** (EX1에서 바이패스 가능) | 슬롯 0 전용 |
| `USAD8` | SIMD + ALU | **2** | 슬롯 0 전용 |
| `USADA8` | SIMD + ALU | **2** | 슬롯 0 전용 |

### 4.4 곱셈 / MAC / 나눗셈 레이턴시 표 [C] (LLVM) + [A](MAC 파이프 사양)

| 명령 | 점유 자원 | 레이턴시 | 비고 |
|---|---|---|---|
| `MUL`, `MULS` | MAC | **2** | |
| `SMULBB/BT/TB/TT`, `SMULWB/WT` (16비트 곱) | MAC | **2** | |
| `SMULL`, `UMULL` (64비트 결과) | MAC | **2** (Lo), Hi는 추가 uop 0 | |
| `SMMUL`, `SMMULR` | MAC | **2** | |
| `MLA`, `MLS` | MAC | **2** | 누산 입력(`Ra`)은 `ReadMAC` advance 1 → EX3에서 필요 |
| `SMLABB/BT/TB/TT`, `SMLAWB/WT` | MAC | **2** | |
| **`SMLAD`, `SMLADX`** | MAC | **2** | LLVM에서 `t2SML[AS]D` → `WriteMAC32` |
| **`SMLSD`, `SMLSDX`** | MAC | **2** | 〃 |
| `SMUAD`, `SMUADX`, `SMUSD`, `SMUSDX` | MAC | **2** | |
| `SMLAL`, `UMLAL` | MAC | **2** (Lo) | |
| **`SMLALD`, `SMLALDX`**, `SMLSLD`, `SMLSLDX` | MAC | **2** (Lo) | MAC64 기본 스케줄 |
| `SMMLA`, `SMMLS` (+R 변형) | MAC | **2** | |
| **`SDIV`, `UDIV`** | ALU(디바이더) | **7** | **SingleIssue — 듀얼 이슈 불가**, 실제로는 결과 대기 스톨. 디바이더는 조기 종료 지원[A] |

> **처리율**: MAC 파이프는 1개이며 **1 MAC/사이클 처리율**이므로 [A], `SMLAD`(halfword 2쌍 = 2 MAC/명령)를 back-to-back으로 넣으면 이론상 **2 MAC/cycle × 216 MHz = 432 MMAC/s** 가 상한이다. [D] (DDI 0489F §1.2.1 처리율 + SMLAD 의미로부터 계산)

> **실측 정정**: 이 보드의 실제 클럭은 **200 MHz**이므로 같은 계산의 상한은
> **2 MAC/cycle × 200 MHz = 400 MMAC/s**다. (MAC 처리율 자체는 아직 실측하지 않았다 — 여전히 [D] 이론값.)

### 4.5 로드/스토어 레이턴시 [C] (LLVM)

| 명령 | 레이턴시 | 발행 제약 |
|---|---|---|
| `LDR` (워드, 즉시/레지스터 오프셋) | **1** | 듀얼 이슈 가능 |
| `LDRB`, `LDRH`, `LDRSB`, `LDRSH` | **2** (느린 로드) | 듀얼 이슈 가능 |
| `LDREX` | 1 | 슬롯 0 전용 |
| `LDREXB`, `LDREXH` | 2 | 슬롯 0 전용 |
| `STR`, `STRB`, `STRH` | **2** | 듀얼 이슈 가능 |
| `STREX`, `STREXB`, `STREXH` | 2 | **단독 발행** |
| `LDRD`, `STRD` | 1 / 2 | **단독 발행** |
| `LDM`, `STM`, `PUSH`, `POP` | — | **단독 발행** |
| `PLD`, `PLI` | 2 | |
| `TBB`, `TBH` | — | **단독 발행, 발행에 2사이클** |
| `B`, `BL`, `BX`, `BLX`, 테이블 분기 | **2** | 분기 유닛 1개 |

> 모델 전역값 `LoadLatency = 2` (best case for load-use). [C]

---

## 5. FPU (FPv5)

### 5.1 F746의 실제 구성

| 항목 | 값 | 등급/출처 |
|---|---|---|
| 아키텍처 | **FPv5-SP-D16-M** (ARMv7-M Floating-Point Extension) | [A] DDI 0489F §8.1 Table 8-1 + `stm32f746xx.h` |
| 정밀도 | **단정밀도 전용**(SFPU). 배정밀도 미지원 | [A]/[B] ST 제품 페이지: "single floating point unit (SFPU) … all Arm single-precision data-processing instructions and data types" |
| GCC 옵션 | `-mfpu=fpv5-sp-d16 -mfloat-abi=hard` | [B] |
| 레지스터 뱅크 | **32 × 32비트 단정밀도(S0–S31)** = **16 × 64비트 배정밀도(D0–D15)**. **뱅크는 하나뿐**이며, FPv5에서 DP가 추가돼도 "16 double-precision registers. This is the same as for FPv4 and there are no additional registers." | [A] DDI 0489F §8.1, §8.2 |
| IEEE 754-2008 준수 | DN/FZ 모드를 끄면 **하드웨어만으로 완전 준수**, 서포트 코드 불필요 | [A] DDI 0489F §8.2.2 |
| 서브노멀(denormal) | **하드웨어 지원** + 모든 IEEE 754-2008 라운딩 모드 하드웨어 지원 | [A] DDI 0489F §1.2.4 |
| Fused MAC | 지원(정밀도 향상용 combined multiply-accumulate) | [A] DDI 0489F §1.2.4 |
| 예외 트랩 | **미지원**. FPSCR 누적 플래그만 세팅. 출력 핀 `FPIXC/FPUFC/FPOFC/FPDZC/FPIDC/FPIOC` 6개 | [A] DDI 0489F §8.2.3 |
| SP/DP 개별 활성 | 둘 다 구현된 경우에도 **독립적으로 활성화 불가** (CPACR로 동시 제어) | [A] DDI 0489F §8.1 |

### 5.2 FPU 시스템 레지스터 [A] (DDI 0489F §8.3 Table 8-2)

| 주소 | 이름 | 타입 | 리셋값 | 설명 |
|---|---|---|---|---|
| `0xE000EF34` | FPCCR | RW | **`0xC0000000`** | Context Control Register |
| `0xE000EF38` | FPCAR | RW | — | Context Address Register |
| `0xE000EF3C` | FPDSCR | RW | `0x00000000` | Default Status Control Register |
| `0xE000EF40` | MVFR0 | RO | **`0x10110021`** (SP 전용) / `0x10110221` (SP+DP) | Media and VFP Feature Register 0 |
| `0xE000EF44` | MVFR1 | RO | **`0x11000011`** (SP 전용) / `0x12000011` (SP+DP) | Media and VFP Feature Register 1 |
| `0xE000EF48` | MVFR2 | RO | `0x00000040` | Media and VFP Feature Register 2 |
| `0xE000ED88` | CPACR | RW | — | CP10/CP11 활성(FPU 사용 전 필수) |

**MVFR0 = `0x10110021` 해석** (필드 위치는 ARM ARM CPUID scheme 기준, 직접 열람하지 않음 → **[D]**):

| 비트 | 필드 | 값 | 의미 |
|---|---|---|---|
| [31:28] | FP rounding modes | `0x1` | 모든 라운딩 모드 지원 |
| [27:24] | Short vectors | `0x0` | 미지원 |
| [23:20] | Square root | `0x1` | 하드웨어 지원 (`VSQRT`) |
| [19:16] | Divide | `0x1` | 하드웨어 지원 (`VDIV`) |
| [15:12] | FP exception trapping | `0x0` | 미지원 (§8.2.3과 일치) |
| [11:8] | Double-precision | `0x0` | **미지원** (SP 전용 확정) |
| [7:4] | Single-precision | `0x2` | 지원 |
| [3:0] | A_SIMD registers | `0x1` | 16 × 64비트 레지스터 뱅크 |

> **실측 정정(확인)**: 세 레지스터 모두 표의 "SP 전용" 예상값과 **비트 단위로 일치**한다 —
> `MVFR0 = 0x10110021`, `MVFR1 = 0x11000011`, `MVFR2 = 0x00000040`. 등급 [D] → **[A]**.
> `MVFR1`은 `[31:28] = 1`(**VFP FMA 지원**), `[27:24] = 1`(**VFP half-precision 변환 지원**),
> `[7:4] = 1`(D_NaN), `[3:0] = 1`(FtZ)이다. 툴체인 사전정의 매크로도 일치한다
> (`__ARM_FP = 4`, `__ARM_FEATURE_FMA = 1`, `__ARM_FEATURE_DSP = 1`, `__ARM_FEATURE_SIMD32 = 1`).
>
> ⚠ 참고: 프로브 펌웨어의 출력 로그(`info/results/probe_output.txt` §4)는 같은 `MVFR1` 값을 두고
> "FMA 없음 / FP16 변환 없음"이라고 **잘못 주석을 달았다**. 레지스터 원값과 GCC 매크로,
> `info/raw/decoded_reset.md` 해석이 모두 "지원"으로 일치하므로 **이 문서 쪽이 옳고
> 프로브의 텍스트 주석이 버그**다. (실측 원값 > 실측 도구의 해석)

### 5.3 VFP 명령 레이턴시 / 스루풋 표 [C] (LLVM `ARMScheduleM7.td`)

**F32(단정밀도) — F746에서 실제 사용 가능한 것:**

| 명령 | 레이턴시 | 자원 | 비고 |
|---|---|---|---|
| `VADD.F32`, `VSUB.F32` | **3** | VFP + VPort | FP 파이프 기본 레이턴시 = 3 |
| `VMUL.F32`, `VNMUL.F32` | **3** | VFP + VPort | |
| `VMLA.F32`, `VMLS.F32` (chained MAC) | **6** | VFP + VPort | 소스 `ReadFPMAC` advance 3 |
| `VFMA.F32`, `VFMS.F32` (fused MAC) | **5** (모델), advance 2 | VFP + VPort | |
| `VDIV.F32` | **16** | VFP + VPort | 스케줄상 유효 레이턴시는 3으로 오버라이드 |
| `VSQRT.F32` | **16** | VFP + VPort | 〃 |
| `VCVT.*` (F32↔정수/고정소수점) | **3** | VFP + VPort | |
| `VABS.F32`, `VNEG.F32`, `VRINT*.F32`, `VMAXNM/VMINNM.F32` | **3** | VFP + VPort | |
| `VMOV.F32` (레지스터/즉시), `VCONST` | **3** | VPort | |
| `VCMP.F32`, `VCMPE.F32` | **0** | VFP + VPort | 플래그 전달은 `VMRS APSR_nzcv`(FMSTAT) 필요 → 그 명령은 **단독 발행** |
| `VSEL.*.F32` | 3 | VFP + VPort | **슬롯 0 전용** |
| `VLDR.F32` | **1** | Load + VPort | |
| `VSTR.F32` | (스토어 기본) | Store + VPort | |
| `VLDM`/`VSTM` (S) | — | | **단독 발행** |
| `VMOV` GP↔FP (`VMOVRS`/`VMOVSR`) | 3 | | `VMOVRRD/DRR/RRS/SRR`는 **단독 발행** |
| `VMRS`(FMSTAT), `VMSR` | — | VFP + VPort | **단독 발행** |

**F64(배정밀도) — F746에는 없음(F767/F769/H7 참고용):** [C]

| 명령 | 레이턴시 |
|---|---|
| `VADD.F64`, `VSUB.F64` | 4 |
| `VMUL.F64` | 7 |
| `VMLA.F64`, `VMLS.F64` | 11 |
| `VDIV.F64`, `VSQRT.F64` | 30 |
| `VLDR.F64` | 2 (단독 발행) |

> F64 연산은 모두 `BeginGroup=1`(슬롯 0 전용) + VPortL·VPortH **양쪽 점유**. [C]

### 5.4 Lazy stacking (레이지 FP 컨텍스트 저장)

- **동작**: FP 상태 스택 공간만 예약해 두고, **ISR이 실제로 FP 명령을 실행하려 할 때까지 저장을 지연**한다. 이로써 ISR 진입 레이턴시가 줄고, FP를 쓰지 않는 ISR은 FP 컨텍스트 저장 자체를 하지 않는다. [A] (DDI 0489F §1.2.4, §8.2.3)
- **기본값**: FPCCR 리셋값 `0xC0000000` → **ASPEN=1, LSPEN=1** → 자동 FP 상태 보존 + **레이지 상태 보존이 리셋 직후부터 활성**. [A] (DDI 0489F §8.3 Table 8-2 + DUI 0646B §4.7.2 Table 4-60)

**FPCCR 비트 필드** [A] (DUI 0646B Table 4-60):

| 비트 | 필드 | 의미 |
|---|---|---|
| [31] | ASPEN | 1 = FP 명령 실행 시 CONTROL.FPCA 자동 설정 → 예외 진입/복귀 시 FP 컨텍스트 자동 하드웨어 보존/복원 |
| [30] | LSPEN | 1 = **레이지 상태 보존 활성** / 0 = 비활성 |
| [8] | MONRDY | FP 스택 프레임 할당 시 DebugMonitor의 MON_PEND 설정 가능 여부 |
| [6] | BFRDY | 〃 BusFault |
| [5] | MMRDY | 〃 MemManage |
| [4] | HFRDY | 〃 HardFault |
| [3] | THREAD | FP 스택 프레임 할당 시 Thread 모드였는지 |
| [1] | USER | 〃 비특권(user) 레벨이었는지 |
| [0] | LSPACT | 1 = **레이지 보존 진행 중**(프레임은 할당됐지만 저장은 지연됨) |

- `FPCAR`(`0xE000EF38`) [31:3] ADDRESS: 예외 스택 프레임 안의 **아직 채워지지 않은 FP 레지스터 공간 주소**를 보관. [A]
- 관련 FP 폴트 플래그: `MLSPERR`(레이지 FP 상태 보존 중 MemManage), `LSPERR`(레이지 FP 상태 보존 중 BusFault). [A] (DDI 0489F §5.3)

---

## 6. 로드/스토어 유닛, 스토어 버퍼, 언얼라인 액세스

### 6.1 LSU [A] (DDI 0489F §1.2.3)

| 항목 | 값 |
|---|---|
| TCM/D-cache/AXIM 로드 채널 | **32비트 × 2** → **64비트 로드 대역폭**, 듀얼 32비트 로드 가능 |
| AHB(AHBP) 로드 채널 | **32비트 × 1** |
| 스토어 채널 | **64비트 × 1** |
| 스토어 버퍼링 | 스토어 처리율 향상 + 데이터/명령 읽기와의 RAM 경합 최소화 |
| 인터페이스별 버퍼 분리 | **TCM / AHBP / AXIM 각각 별도 스토어 버퍼** (QoS 및 인터페이스별 최적화) |

### 6.2 스토어 버퍼(STB) [A] (DDI 0489F §1.2.8, §5.9.2)

| 항목 | 값 |
|---|---|
| **엔트리 수** | **4** |
| **엔트리당 용량** | **데이터 최대 64비트 + 주소 32비트** |
| 대상 | TCM·AHBP 이외의 모든 data-side 쓰기 요청 |
| 머지 | **Normal 메모리에 한해** 같은 라인의 후속 쓰기를 기존 엔트리에 병합. 같은 캐시 라인이면 **여러 엔트리를 링크**해 병합 가능 |
| 머지 불가 | Strongly-ordered / Device 메모리 (해당 메모리 read 전에는 STB를 자동 드레인) |
| 머지 조건(§1.2.8) | **같은 64비트 정렬 주소**의 여러 스토어를 하나의 트랜잭션으로 병합 |

**엔트리 드레인 조건** [A]:
- 엔트리의 모든 바이트가 채워졌을 때(머지 결과 포함)
- 엔트리를 linefill 버퍼에 병합 가능할 때
- 엔트리가 Device/Strongly-ordered 스토어일 때
- Non-cacheable 또는 Write-Through 엔트리가 머지 데이터를 너무 오래 기다렸을 때

**전체 드레인 조건** [A]:
- 캐시 유지보수 연산 / **`DMB` 또는 `DSB` 명령** / Shared 메모리에 대한 exclusive store
- 스토어 버퍼가 가득 찼거나 곧 찰 것 같을 때
- Strongly-ordered/Device 메모리에서 로드하기 전에 해당 메모리로의 모든 스토어를 드레인

### 6.3 언얼라인(unaligned) 액세스 [A] (DDI 0489F §2.4.2)

- ARMv7 언얼라인 액세스를 지원하며, **모든 액세스를 단일 언얼라인 액세스로 수행**한다. 내부적으로 **2개 이상의 정렬된 액세스로 변환**되어 외부 인터페이스에 나간다.
- **프로세서 외부 액세스는 항상 정렬되어 있다.**
- 언얼라인 지원은 **단일 로드/스토어(`LDR`, `LDRH`, `STR`, `STRH`)에 한정**. `LDRD`/`STRD`는 워드 정렬만 허용하고 그 외 언얼라인은 **폴트 발생**.
- **PPB 공간에는 언얼라인 액세스 미지원.**
- **Device 또는 Strongly-ordered 메모리로의 언얼라인 액세스는 alignment UsageFault 발생.** [A] (DDI 0489F §5.6 관련)
- 메모리 맵 경계를 넘는 언얼라인 액세스는 **아키텍처적으로 UNPREDICTABLE** (동작은 경계 의존).
- **정확한 사이클 페널티 수치는 공개 문서에 없음 → 미확인.** (2개 이상 내부 액세스로 분해되므로 최소 1사이클 이상 추가되는 것은 확실하나, 정량치 없음) [D]

> **실측 정정(여전히 미확인)**: 프로브 펌웨어가 이 항목을 측정하려 시도했으나
> (`probe_output.txt` §8 "정렬 페널티 (32비트 로드)"), 로그가 **기준선 한 행에서 잘려 있다** —
> 4 B 정렬(라인 내부): **SRAM1 9.03 cyc/접근, SDRAM 19.15 cyc/접근** (D-cache WB, 64 B 간격, 워킹셋 64 KB).
> 비정렬(+1/+2/+3 바이트) 행이 출력에 없으므로 페널티는 **아직 미확정**이다.
> 같은 루프의 시작 오프셋만 바꿔 재실행하면 바로 얻을 수 있다.
> 참고로 툴체인은 `__ARM_FEATURE_UNALIGNED = 1`을 정의한다(비정렬 접근 하드웨어 지원).

### 6.4 Exclusive monitor [A] (DDI 0489F §2.5)

- 로컬 exclusive monitor를 구현하되 **물리 주소를 보관하지 않는다**. 모든 액세스를 직전 `LDREX` 주소와 매칭되는 것으로 취급한다.
- → **구현된 ERG(Exclusives Reservation Granule)는 전체 메모리 주소 범위**이다.

### 6.5 Speculative access (r1p2에서 문서화, r0p1에도 하드웨어 동작은 동일 추정) [A]/[D]

**발생 가능** [A] (DDI 0489F §5.2):
- Normal + executable 메모리로의 speculative 명령 페치
- Normal read/write 또는 read-only 메모리로의 speculative 데이터 읽기
- Cacheable 주소로의 speculative 캐시 linefill (드물게 다른 인터페이스로도)
- TCM 영역 대상 speculative 읽기

**발생하지 않음** [A]:
- XN(Execute Never) 영역으로의 speculative 명령 페치
- MPU에서 접근 불가로 표시된 주소로의 speculative 데이터 읽기
- Non-cacheable 주소로의 speculative linefill
- Device/Strongly-ordered 메모리로의 speculative 데이터 읽기/linefill
- **AHBP 인터페이스로의 speculative 읽기**
- **speculative 쓰기는 절대 발생하지 않음**

---

## 7. 인터럽트 / NVIC

### 7.1 코어 및 F746 구성

| 항목 | Cortex-M7 코어 | STM32F746NG 실제 | 등급/출처 |
|---|---|---|---|
| 외부 인터럽트 수 | 1–240 (구현 시 선택) | **98개 (IRQ0 … IRQ97, `SPDIF_RX_IRQn = 97`)** | [A] DDI 0489F §1.1.3 / `stm32f746xx.h` |
| 우선순위 비트 수 | **3–8비트** (8–256 레벨) | **4비트 → 16 레벨** (`__NVIC_PRIO_BITS = 4`) | [A] DDI 0489F Table 1-1 / `stm32f746xx.h` L168 |
| ICTR.INTLINESNUM | 32개 그룹 단위 인코딩 | 98개 → `0b0011` (97–128 구간) | [A]/[D](계산) |

> **실측 정정**: `ICTR = 0x00000003`(= 128 라인)과 우선순위 4비트는 실측으로 확인됐다 —
> `NVIC->IP[0]`에 `0xFF`를 쓰고 되읽으면 **`0xF0`**(상위 4비트만 구현)이 나온다.
> 다만 **"98개"는 슬롯 수**다. ST CMSIS 헤더가 이름을 붙인 페리페럴 IRQ는 **97개이고 79번이 결번**이며,
> 벡터 테이블 전체 엔트리는 105개다. IRQ 번호를 0~97로 순회하는 코드는 79번을 비워 두어야 한다.
| 우선순위 그룹핑 | 선점/비선점 레벨 분리 지원 | AIRCR.PRIGROUP | [A] |
| 동적 재우선순위 | 지원 | | [A] DDI 0489F §1.2.5 |
| tail-chaining | 지원 | | [A] |
| late-arriving | 지원 | | [A] |

### 7.2 NVIC 레지스터 맵 [A] (DDI 0489F §7.3 Table 7-1)

| 주소 | 이름 | 타입 | 리셋 |
|---|---|---|---|
| `0xE000E004` | ICTR (Interrupt Controller Type Register) | RO | — |
| `0xE000E100`–`0xE000E11C` | NVIC_ISER0–ISER7 | RW | `0x00000000` |
| `0xE000E180`–`0xE000E19C` | NVIC_ICER0–ICER7 | RW | `0x00000000` |
| `0xE000E200`–`0xE000E21C` | NVIC_ISPR0–ISPR7 | RW | `0x00000000` |
| `0xE000E280`–`0xE000E29C` | NVIC_ICPR0–ICPR7 | RW | `0x00000000` |
| `0xE000E300`–`0xE000E31C` | NVIC_IABR0–IABR7 | RO | `0x00000000` |
| `0xE000E400`–`0xE000E4EC` | NVIC_IPR0–IPR59 | RW | `0x00000000` |

- **모든 NVIC 레지스터는 워드(32비트) 액세스만 허용.** [A]
- 특권 모드에서만 완전 접근 가능. CCR 설정 시 user 모드에서 pending 설정은 가능하나, 그 외 user 액세스는 **bus fault**. [A]
- **ICTR.INTLINESNUM[3:0]** 인코딩: `0b0000`=0–32, `0b0001`=33–64, `0b0010`=65–96, `0b0011`=97–128, `0b0100`=129–160, `0b0101`=161–192, `0b0110`=193–224, `0b0111`=225–256 (단, 프로세서 최대 외부 인터럽트는 240). [A]

### 7.3 예외 진입/복귀 메커니즘 [A] (DUI 0646B §2.3.7)

| 용어 | 정의 |
|---|---|
| **Preemption** | 실행 중인 예외 핸들러보다 우선순위가 높은 예외가 선점 |
| **Return** | 대기 중인 충분한 우선순위의 예외가 없고, 완료된 핸들러가 late-arriving을 처리한 게 아닐 때 스택 pop + 상태 복원 |
| **Tail-chaining** | 핸들러 완료 시 진입 조건을 만족하는 pending 예외가 있으면 **스택 pop을 생략**하고 곧바로 새 핸들러로 전이 |
| **Late-arriving** | **이전 예외의 상태 저장(state saving) 중** 더 높은 우선순위 예외가 발생하면 그쪽으로 전환하고 해당 벡터 페치를 시작. 저장되는 상태가 동일하므로 state saving은 **중단 없이 계속**. **원래 예외 핸들러의 첫 명령이 execute 스테이지에 들어가기 전까지** late-arriving을 받아들일 수 있음. 복귀 시에는 일반 tail-chaining 규칙 적용 |

**예외 스택 프레임** [A] (DUI 0646B Figure 2-2):

| 구성 | 워드 수 | 바이트 |
|---|---|---|
| FP 상태 없음 | R0, R1, R2, R3, R12, LR, PC, xPSR = **8워드** | **32바이트** |
| FP 상태 포함 | 위 8워드 + S0–S15(16) + FPSCR(1) + Reserved(1) = **26워드** | **104바이트** |

- **CCR.STKALIGN은 Cortex-M7에서 read-only, 값 1** → 예외 스택 프레임 시작 주소는 **항상 8바이트 정렬**. 정렬 패딩으로 최대 4바이트 추가 가능. [A] (DUI 0646B §4.3.6 / §2.3.7)
- 스태킹과 **병렬로** 벡터 테이블에서 핸들러 시작 주소를 벡터 페치한다. 스태킹 완료 시 핸들러 실행 시작, 동시에 LR에 EXC_RETURN 값 기록. [A]

### 7.4 인터럽트 레이턴시 (사이클)

> **DDI 0489F에는 인터럽트 레이턴시 사이클 수가 명시되어 있지 않다.** (Cortex-M4 TRM에는 있음) [A](없음을 확인)

| 항목 | 값 | 등급 |
|---|---|---|
| 최소/일반 인터럽트 레이턴시 (0-wait-state 메모리) | **12 사이클** | [C] 2차 자료 다수 (ARM 마케팅 비교자료 계열) |
| Tail-chaining 레이턴시 | **6 사이클** (레지스터 pop/push 생략으로 총 18사이클 절약) | [C] |
| 최악 레이턴시 | Device/Strongly-ordered 메모리에서의 로드 진행 중이면 **14 사이클**로 증가 | [C] |
| 개선 수단 | ACTLR[15] DISCRITAXIRUR=1 → "최악 인터럽트 레이턴시 개선 가능, 단 성능 저하" | **[A]** DDI 0489F §3.3.1 |

- 216 MHz 기준 12사이클 = **55.6 ns**, 14사이클 = **64.8 ns**. [D] (계산: 12 / 216e6)

> **실측 정정**: 이 보드는 **200 MHz**로 돈다 → 12사이클 = **60.0 ns**, 14사이클 = **70.0 ns**.
> (사이클 수 12/6/14 자체는 여전히 [C]이며 이 보드에서 측정하지 않았다.)
- 대부분의 멀티사이클 명령은 인터럽트 인식 시 중단(abandon) 가능하나, **Device/Strongly-ordered 메모리 로드**와 **AXI에서 이미 시작된 shared store-exclusive**는 예외. [C]

### 7.5 저전력 / WIC [A] (DDI 0489F §7.2.1)

- `WFI`, `WFE`, `SEV` 완전 구현. `SLEEPONEXIT` 지원(핸들러에서 Thread 모드로 복귀 시 슬립 진입).
- WIC 사용 시 System Control Register의 **SLEEPDEEP를 활성화해야 한다.**

### 7.6 레벨 vs 펄스 인터럽트 [A] (DDI 0489F §7.2.2)

- 레벨/펄스 둘 다 지원. 펄스는 **비동기가 아니라 FCLK 상승 에지에 샘플링되어야 한다.**
- 레벨 인터럽트: ISR 복귀 전에 디어서트되지 않으면 다시 pending → 재활성화 (FIFO/버퍼 디바이스에 유용).
- 펄스 인터럽트: ISR 중 재어서트되면 pending과 active가 동시 성립 가능. pending 상태에서 추가 펄스가 와도 ISR은 **한 번만** 실행.

---

## 8. MPU (Memory Protection Unit)

### 8.1 구성 [A] (DDI 0489F §6.1, Table 1-1)

| 항목 | 값 |
|---|---|
| 아키텍처 | **ARMv7 PMSA (Protected Memory System Architecture)** |
| 리전 수 (구현 시 선택) | **0 / 8 / 16** |
| 리전 우선순위 | 겹침 허용, **번호가 클수록 우선순위 높음** (16리전이면 15가 최고, 8리전이면 7이 최고, 0이 최저) |
| 백그라운드 리전 | 활성화 가능(디폴트 메모리 맵 속성 적용) |
| 위반 시 | 프로그래머블 우선순위의 **MemManage fault** 발생 |
| 서브리전 | **8개 (SRD[7:0])**, 128바이트 이하 리전은 서브리전 미지원(SRD=`0x00`으로 쓸 것) |

- **STM32F7의 실제 리전 수는 8로 알려져 있으나 1차 문서로 확인하지 못함 [C].** 런타임에 `MPU->TYPE`(`0xE000ED90`)의 DREGION 필드를 읽어 확인 권장 — 리셋값 `0x00000800`이면 8리전, `0x00001000`이면 16리전. [A]

> **실측 정정(확인)**: `MPU_TYPE = 0x00000800` → **DREGION = 8, IREGION = 0, SEPARATE = 0**.
> **8리전 통합형으로 확정**되었다. 등급 [C] → **[A]**. (프로브 시점 `MPU_CTRL = 0`으로 MPU는 꺼져 있었고,
> 원래 SNN 펌웨어 실행 중 덤프에서는 `MPU_CTRL = 0x00000005` → ENABLE=1, PRIVDEFENA=1이었다.)

### 8.2 MPU 레지스터 맵 [A] (DDI 0489F §6.3 Table 6-1)

| 주소 | 이름 | 타입 | 리셋 |
|---|---|---|---|
| `0xE000ED90` | MPU_TYPE | RO | `0x00000800`(8리전) / `0x00001000`(16리전) / `0x00000000`(0리전) |
| `0xE000ED94` | MPU_CTRL | RW | `0x00000000` |
| `0xE000ED98` | MPU_RNR | RW | Unknown |
| `0xE000ED9C` | MPU_RBAR | RW | Unknown |
| `0xE000EDA0` | MPU_RASR | RW | Unknown |
| `0xE000EDA4` / `0xE000EDA8` | MPU_RBAR_A1 / MPU_RASR_A1 | RW | Unknown |
| `0xE000EDAC` / `0xE000EDB0` | MPU_RBAR_A2 / MPU_RASR_A2 | RW | Unknown |
| `0xE000EDB4` / `0xE000EDB8` | MPU_RBAR_A3 / MPU_RASR_A3 | RW | Unknown |

> **MPU 레지스터는 정렬된 워드 액세스만 지원. 바이트/하프워드 액세스는 UNPREDICTABLE.** [A]

### 8.3 MPU_RASR 비트 필드 [A] (DUI 0646B §4.6.5 Table 4-53)

| 비트 | 필드 | 의미 |
|---|---|---|
| [28] | **XN** | 0 = 명령어 페치 허용, 1 = 금지 |
| [26:24] | **AP** | 접근 권한 필드 |
| [21:19] | **TEX** | Type Extension |
| [18] | **S** | Shareable |
| [17] | **C** | Cacheable |
| [16] | **B** | Bufferable |
| [15:8] | **SRD** | 서브리전 비활성 비트 (0=활성, 1=비활성) |
| [5:1] | **SIZE** | 리전 크기, 최솟값 4(`0b00100`) → 32바이트 |
| [0] | **ENABLE** | 리전 활성 |

### 8.4 TEX / C / B / S 인코딩 [A] (DUI 0646B Table 4-55, 4-56)

| TEX | C | B | S | 메모리 타입 | Shareability | 기타 속성 |
|---|---|---|---|---|---|---|
| `0b000` | 0 | 0 | x | **Strongly Ordered** | Shareable | — |
| `0b000` | 0 | 1 | x | **Device** | Shareable | — |
| `0b000` | 1 | 0 | 0 / 1 | Normal | Not shareable / Shareable | **Outer·inner write-through, No Write-Allocate** |
| `0b000` | 1 | 1 | 0 / 1 | Normal | Not shareable / Shareable | **Outer·inner Write-Back, No Write-Allocate** |
| `0b001` | 0 | 0 | 0 / 1 | Normal | Not shareable / Shareable | **Outer·inner noncacheable** |
| `0b001` | 0 | 1 | x | Reserved | — | — |
| `0b001` | 1 | 0 | x | (ARM 파트너 문의) | — | — |
| `0b001` | 1 | 1 | 0 / 1 | Normal | Not shareable / Shareable | **Outer·inner Write-Back, Write & Read Allocate** |
| `0b010` | 0 | 0 | x | **Device** | Not shareable | Nonshared Device |
| `0b010` | 0 | 1 / 1 x | x | Reserved | — | — |
| `0b1BB` | A | A | 0 / 1 | Normal | Not shareable / Shareable | 캐시 메모리, **BB = outer 정책, AA = inner 정책** |

**AA/BB 캐시 정책 인코딩** (TEX = 4–7일 때) [A]:

| 인코딩 | 캐시 정책 |
|---|---|
| `00` | Non-cacheable |
| `01` | Write-Back, Write and Read Allocate |
| `10` | Write-Through, No Write Allocate |
| `11` | Write-Back, No Write Allocate |

> **SNN 추론 성능 관점**: SDRAM(외부 메모리) 영역을 `TEX=0b001, C=1, B=1`(Write-Back, Write+Read Allocate)로 설정하면 D-cache 할당 정책이 최적화된다. DMA 버퍼는 `TEX=0b000, C=0, B=0`(Strongly Ordered) 또는 `0b001,C=0,B=0`(Normal non-cacheable)로 두는 것이 일반적이다. [D] (위 표에서 유도)

---

## 9. 성능 지표

### 9.1 코어 수준 (Arm 공식 비교표) [A]

출처: *Arm Cortex-M Processor Comparison Table* (documentation-service.arm.com/static/6267de1c7e121f01fd22d677, armkeil.blob 버전 동일)

| 항목 | Cortex-M7 | (참고) Cortex-M4 |
|---|---|---|
| **DMIPS/MHz\*** | **2.31** | 1.26 |
| **CoreMark®/MHz\*** | **5.29** | 3.54 |
| ISA | Armv7-M | Armv7-M |
| DSP Extension | **Yes** | Yes |
| FPU | **SP, DP (option)** | SP (option) |
| Hardware Divide | Yes | Yes |
| 최대 외부 인터럽트 | **240** | 240 |
| 최대 MPU 리전 | **16** | 8 |
| 메인 버스 | **AXI (64-bit)** | AHB Lite (32-bit) |
| Instruction Cache | **0–64 kB** | 없음 |
| Data Cache | **0–64 kB** | 없음 |
| Instruction TCM | **0–16 MB** | 없음 |
| Data TCM | **0–16 MB** | 없음 |
| TrustZone / Helium / Arm Custom Instructions / Coprocessor IF | **모두 No** | 모두 No |
| DCLS 구성 | Yes | No |

\* 각주: "See individual Cortex-M product pages for further information" — 컴파일러/구성에 따라 달라진다는 의미. [A]

### 9.2 STM32F746 실장 수준 (ST 공표) [B]

| 항목 | 값 | 출처 |
|---|---|---|
| 최대 CPU 주파수 | **216 MHz** | ST 제품 페이지 / DS10916 [B] |
| Dhrystone 2.1 | **462 DMIPS = 2.14 DMIPS/MHz** | ST 제품 페이지 (DS10916 §3.1) [B] |
| CoreMark | **1082 CoreMark @ 216 MHz** (≈ **5.009 CoreMark/MHz** [D] 계산) | AN4667 / ST 제품 페이지 [B] |
| 달성 조건 | **임베디드 Flash / 내부 RAM / 외부 메모리 어디서 실행해도 동일 스코어**, ST ART Accelerator™ 로 216 MHz에서 0-wait-state 유사 실행 | AN4667 [B] |
| ART Accelerator 구조 | **ITCM 인터페이스의 Flash 액세스 전용**, 명령/분기 통합 캐시 **256비트 × 64라인** | AN4667 [B] |
| L1 캐시 | **I-cache 4 kB + D-cache 4 kB** | ST 제품 페이지 [B] |
| ITCM-RAM | **16 kB** (critical real-time 루틴용) | DS10916 [B] |
| DTCM-RAM | **64 kB** (critical real-time 데이터용) | DS10916 [B] |
| 총 SRAM | **320 kB** (DTCM 64 kB 포함) + ITCM 16 kB + backup SRAM 4 kB | DS10916 [B] |

> **ARM(2.31 / 5.29)과 ST(2.14 / 5.01)의 수치 차이**는 벤치마크 컴파일러·옵션·메모리 구성 차이 때문이며, 어느 한쪽이 틀린 것이 아니다. 논문 인용 시 **어느 출처인지 반드시 명시**할 것. [D]

> **실측 정정**: 위 표의 값들은 **216 MHz 전제**다. 이 보드는 **200 MHz**로 돌므로
> 같은 효율이라면 462 DMIPS → 약 428 DMIPS, 1082 CoreMark → 약 1002 CoreMark에 해당한다(환산값, 미측정).
> 또한 **"어디서 실행해도 동일 스코어 / ART로 0-wait state 유사 실행"** 은 이 보드의 실제 상태와 다르다:
> 실측 `FLASH_ACR = 0x00000006` → **6 wait state, ARTEN=0, PRFTEN=0** (ART가 꺼진 채로 동작 중이었다).
> 원인은 `stm32f7xx_hal_conf.h:150`의 `PREFETCH_ENABLE 0U` + `ART_ACCLERATOR_ENABLE` 매크로 미정의다.
> 데이터 접근 기준 실측으로도 SDRAM 60 MB/s, QSPI 52 MB/s로 내부 SRAM(205~220 MB/s)과 크게 다르다.
> L1 캐시 4 KB + 4 KB, ITCM 16 KB, DTCM 64 KB, 총 SRAM 320 KB는 실측과 **일치**한다.

### 9.3 사이클 시간 환산 [D] (계산)

| 주파수 | 1 사이클 | 12 사이클(인터럽트 레이턴시) | DWT_CYCCNT 랩어라운드(2³²) |
|---|---|---|---|
| **216 MHz** | **4.630 ns** | **55.6 ns** | **19.884 초** |
| 200 MHz | 5.000 ns | 60.0 ns | 21.475 초 |
| 108 MHz | 9.259 ns | 111.1 ns | 39.769 초 |

> **실측 정정**: 이 보드에 적용되는 행은 **200 MHz 행**이다 (1 cyc = 5.000 ns, CYCCNT 랩어라운드 21.475 s).
> 216 MHz 행은 데이터시트 최댓값용 참고치일 뿐 이 보드의 측정 환산에 쓰면 안 된다.

---

## 10. 디버그 / 트레이스 유닛 (DWT / ITM / ETM / TPIU)

### 10.1 디버그 컴포넌트 주소 맵 [A] (DDI 0489F §2.4 Figure 2-1)

| 주소 범위 | 컴포넌트 |
|---|---|
| `0xE0000000`–`0xE0000FFF` | **ITM** |
| `0xE0001000`–`0xE0001FFF` | **DWT** |
| `0xE0002000`–`0xE0002FFF` | FPB (Flash Patch and Breakpoint) |
| `0xE000E000`–`0xE000EFFF` | SCS (SysTick `+0x010`, NVIC `+0x100`, SCB `+0xD00`, MPU `+0xD90`, FPU `+0xF30`) |
| `0xE000EDF0`–`0xE000EEFF` | CoreDebug (DHCSR/DCRSR/DCRDR/**DEMCR**) |
| `0xE0040000`–`0xE0040FFF` | **TPIU** (TRM 그림 표기: "Reserved (TPIU)") |
| `0xE0041000`–`0xE0041FFF` | **ETM** |
| `0xE0042000`–`0xE0042FFF` | **CTI** (Cross Trigger Interface) |
| `0xE0043000`–`0xE00FDFFF` | Private Peripheral Bus |
| `0xE00FE000`–`0xE00FEFFF` | **Processor ROM table** |
| `0xE00FF000`–`0xE00FFFFF` | **PPB ROM table** |
| `0xE0100000`–`0xFFFFFFFF` | System / External PPB |

> **실측 정정**: `0xE0042000`은 이 칩에서 **CTI가 아니라 ST의 DBGMCU**다.
> 그 주소를 읽으면 `DBGMCU_IDCODE = 0x10016449`가 나오고, CoreSight ROM table(0xE00FF000)에도
> CTI 엔트리가 없다(엔트리는 SCS/DWT/FPB/ITM/TPIU/ETM 6개뿐이고 목록은 `0x00000000`으로 끝난다).
> 위 표는 ARM 코어 일반 주소 맵이며, ST가 같은 주소를 DBGMCU로 덮어썼다.
>
> **TPIU/ETM 실측**: ROM table은 두 엔트리를 **present=0**으로 광고하지만(`0xFFF41002`, `0xFFF42002`),
> `DEMCR.TRCENA=1` 상태에서 해당 주소의 PIDR을 직접 읽으면 성공한다 —
> TPIU part `0x9A9`, ETM part `0x975`, 제조사 ARM(JEP106 0x3B), CIDR class 0x9.
> **두 유닛은 물리적으로 존재하지만 광고되지 않는다.** `DBGMCU_CR = 0x00000000`이라
> `TRACE_IOEN = 0`, 즉 트레이스 출력 핀이 꺼져 있다.

CMSIS 헤더 확인값 (`core_cm7.h` L1762–1785) [A]:
`SCS_BASE = 0xE000E000`, `ITM_BASE = 0xE0000000`, `DWT_BASE = 0xE0001000`, `TPI_BASE = 0xE0040000`, `CoreDebug_BASE = 0xE000EDF0`, `SysTick_BASE = 0xE000E010`, `NVIC_BASE = 0xE000E100`, `SCB_BASE = 0xE000ED00`, `MPU_BASE = 0xE000ED90`, `FPU_BASE = 0xE000EF30`

### 10.2 DWT 기능 [A] (DDI 0489F §11.2)

- **Full DWT: 비교기 4개** — 하드웨어 워치포인트 / ETM 트리거 / PC 샘플러 이벤트 트리거 / 데이터 주소 샘플러 이벤트 트리거로 설정 가능
- **Reduced DWT: 비교기 2개** (워치포인트 또는 트리거, 데이터 매칭 지원)
- **DWT_COMP0**은 **클록 사이클 카운터 CYCCNT와도 비교 가능**
- **DWT_COMP1**은 **데이터 비교기**로도 사용 가능
- 카운터가 오버플로할 때마다 **이벤트 발생**
- 정의된 간격으로 **PC 샘플** 생성, 인터럽트 이벤트 정보 생성 가능
- ITM에 대한 **주기적 프로토콜 동기화 요청** 제공

### 10.3 DWT 성능 카운터 6종 — 주소 및 폭 [A]

| 주소 | 레지스터 | 타입 | 리셋 | 폭 | 세는 대상 |
|---|---|---|---|---|---|
| `0xE0001000` | **DWT_CTRL** | RW | 구현 의존 | 32 | 제어 |
| `0xE0001004` | **DWT_CYCCNT** | RW | `0x00000000` | **32비트** | **프로세서 클록 사이클** |
| `0xE0001008` | **DWT_CPICNT** | RW | — | **8비트** | **CPI** — 첫 사이클을 제외한 모든 명령 사이클(= 멀티사이클 명령의 추가 사이클) |
| `0xE000100C` | **DWT_EXCCNT** | RW | — | **8비트** | **예외 오버헤드** 사이클 |
| `0xE0001010` | **DWT_SLEEPCNT** | RW | — | **8비트** | **슬립 사이클** |
| `0xE0001014` | **DWT_LSUCNT** | RW | — | **8비트** | **LSU(로드/스토어) 연산** 사이클 |
| `0xE0001018` | **DWT_FOLDCNT** | RW | — | **8비트** | **폴딩된(folded) 명령** 수 |
| `0xE000101C` | DWT_PCSR | RO | — | 32 | PC 샘플 |
| `0xE0001020`/`24`/`28` | DWT_COMP0 / MASK0 / FUNCTION0 | RW | FUNCTION `0x0` | | 비교기 0 |
| `0xE0001030`/`34`/`38` | DWT_COMP1 / MASK1 / FUNCTION1 | RW | FUNCTION `0x0` | | 비교기 1 (데이터 비교 가능) |
| `0xE0001040`/`44`/`48` | DWT_COMP2 / MASK2 / FUNCTION2 | RW | FUNCTION `0x0` | | 비교기 2 |
| `0xE0001050`/`54`/`58` | DWT_COMP3 / MASK3 / FUNCTION3 | RW | FUNCTION `0x0` | | 비교기 3 |
| `0xE0001FB0` | **DWT_LAR** (Lock Access Register) | **WO** | — | 32 | **`0xC5ACCE55` 기록으로 언락** |
| `0xE0001FB4` | DWT_LSR (Lock Status Register) | RO | Unknown | 32 | 잠금 상태 |
| `0xE0001FD0`–`0xE0001FEC` | PID4–PID7, PID0–PID3 | RO | PID0=`0x02`, PID1=`0xB0`, PID2=`0x0B`, PID3=`0x00`, PID4=`0x04` | | Peripheral ID |

> ⚠ **5개 성능 카운터(CPICNT/EXCCNT/SLEEPCNT/LSUCNT/FOLDCNT)는 모두 8비트**이므로 최대 255에서 오버플로하며 이벤트를 발생시킨다. 긴 구간 프로파일링에는 **오버플로 이벤트를 세거나 짧은 구간으로 나눠 측정**해야 한다. (`core_cm7.h`의 `DWT_CPICNT_CPICNT_Msk = 0xFF` 등으로 확인) [A]

### 10.4 DWT_CTRL 비트 필드 (카운터 활성) [A] (`core_cm7.h` L1147–1205)

| 비트 | 필드 | 기능 |
|---|---|---|
| [31:28] | NUMCOMP | 구현된 비교기 수 (RO) |
| [27] | NOTRCPKT | 트레이스 샘플/예외 트레이싱 미지원 표시 |
| [26] | NOEXTTRIG | 외부 매치 신호 미지원 표시 |
| [25] | **NOPRFCNT** | **1이면 프로파일링 카운터(CPI/EXC/SLEEP/LSU/FOLD) 미지원** |
| [24] | NOCYCCNT | 1이면 CYCCNT 미지원 |
| [22] | CYCEVTENA | POSTCNT 언더플로 시 이벤트 |
| [21] | **FOLDEVTENA** | FOLDCNT 오버플로 이벤트 활성 |
| [20] | **LSUEVTENA** | LSUCNT 오버플로 이벤트 활성 |
| [19] | **SLEEPEVTENA** | SLEEPCNT 오버플로 이벤트 활성 |
| [18] | **EXCEVTENA** | EXCCNT 오버플로 이벤트 활성 |
| [17] | **CPIEVTENA** | CPICNT 오버플로 이벤트 활성 |
| [16] | EXCTRCENA | 예외 트레이싱 활성 |
| [12] | PCSAMPLENA | PC 샘플링 활성 |
| [11:10] | SYNCTAP | 동기화 패킷 tap 선택 |
| [9] | CYCTAP | POSTCNT tap 선택 (CYCCNT[6] 또는 CYCCNT[10]) |
| [8:5] | POSTINIT | POSTCNT 초기값 |
| [4:1] | POSTPRESET | POSTCNT 리로드값 |
| [0] | **CYCCNTENA** | **CYCCNT 활성** |

### 10.5 CYCCNT 사용 절차 (Cortex-M7 전용 주의) [A]+[C]

```c
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // 0xE000EDFC bit[24]  [A]
DWT->LAR = 0xC5ACCE55;                            // 0xE0001FB0 M7 전용 언락 [C]
DWT->CYCCNT = 0;
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;              // bit[0]  [A]
```

- **`DWT->LAR` 언락은 Cortex-M4(`core_cm4.h`)에는 없고 Cortex-M7(`core_cm7.h`)에만 있다.** 디버거가 붙어 있으면 자동으로 언락되는 경우가 있어, 디버거 없이 스탠드얼론 실행할 때만 문제가 드러나는 함정이 있다. [C]
- 216 MHz 기준 CYCCNT 분해능 = **4.630 ns**, 랩어라운드 주기 = **19.884 초**. [D]

> **실측 정정**: 이 보드는 200 MHz이므로 CYCCNT 분해능 = **5.000 ns**, 랩어라운드 = **21.475 초**.
> 위 절차는 실제로 동작함이 확인됐다(프로브 펌웨어가 그대로 써서 `DWT_CTRL = 0x40000001`,
> `CYCCNTENA=1`, CYCCNT 증가를 확인). 다만 **`DWT->LAR` 언락이 정말 필요한지는 미확인**이다 —
> 프로브가 무조건 써 버려서 필요 여부를 가릴 수 없다. LAR 쓰기를 뺀 빌드로 재확인하면 판별된다.

### 10.6 ITM / TPIU / ETM / CTI

| 유닛 | 상태 | 근거 |
|---|---|---|
| **ITM** | 옵션(구현 시 포함). printf 스타일 디버깅용 instrumentation trace. base `0xE0000000` | [A] DDI 0489F §1.2.13, §12 |
| **DWT** | 옵션. ITM과 함께 "Complete ITM and DWT trace" 또는 "No ITM or DWT trace"로 구현 선택 | [A] Table 1-1 |
| **TPIU** | 옵션. TPA(Trace Port Analyzer)로 온칩 데이터 전달, **SWO 모드 포함**. base `0xE0040000` | [A] §1.2.13, §13 |
| **ETM** | 옵션 3단계: 없음 / **명령 트레이스만** / 명령+데이터 트레이스. base `0xE0041000`. **트레이스 stalling 기능은 Cortex-M7에 미구현** | [A] Table 1-1, §1.2.12 |
| **CTI** | 옵션. base `0xE0042000` | [A] §1.2.11, §10 |
| **FPB** | 고정 구현. 브레이크포인트 비교기 **4개 또는 8개**(구현 선택) | [A] §1.2.13, §9.1 |
| 워치포인트 | **2개(reduced) 또는 4개(full)** | [A] §9.1 |

- **STM32F746의 실제 ETM/ITM/DWT 구현 여부와 비교기 개수는 RM0385에서 확인 필요 → 미확인.** 런타임에 `DWT->CTRL[31:28]`(NUMCOMP)과 `FP_CTRL`의 NUM_CODE/NUM_LIT를 읽으면 확정 가능. [D]

> **실측 정정(확인)**: 제안한 그 방법으로 전부 확정했다.
> - `DWT_CTRL = 0x40000001` → **NUMCOMP = 4 (Full DWT)**, `NOCYCCNT=0`, `NOPRFCNT=0` → CYCCNT와 프로파일 카운터 5종 모두 존재.
> - `FP_CTRL = 0x10000080` → **NUM_CODE = 8, NUM_LIT = 0, REV = 1(FPBv2)** → 하드웨어 브레이크포인트 8개, 리터럴 비교기 없음.
> - ITM(0xE0000000) part `0x001`, TPIU part `0x9A9`, ETM part `0x975` — **전부 실장**.
> - 단, `DEMCR.TRCENA`가 0이면 DWT 레지스터가 전부 0으로 읽힌다. TRCENA를 켜기 전 덤프에서
>   `DWT_CTRL = 0`이 나온 것은 "DWT 없음"이 아니라 트레이스 블록이 꺼져 있었다는 뜻이다.
- STM32F746G-DISCO 보드에서 4비트 병렬 트레이스 포트(TRACECLK/TRACED[0:3])가 커넥터로 노출되는지는 **UM1907/MB1191에서 확인 필요 → 미확인**.

---

## 11. 캐시 및 메모리 시스템 요약 (코어 관점)

| 항목 | 값 | 등급/출처 |
|---|---|---|
| L1 명령 캐시 | 구현 선택 4/8/16/32/64 kB, **F746은 4 kB** | [A] Table 1-1 / [B] ST |
| L1 데이터 캐시 | 구현 선택 4/8/16/32/64 kB, **F746은 4 kB** | [A] / [B] |
| 캐시 ECC | 구현 선택 (없음 / 모든 구현 캐시에 ECC) | [A] Table 1-1 |
| ITCM | 없음 또는 4 kB–16 MB(2의 거듭제곱), **F746은 16 kB** | [A] / [B] |
| DTCM | 없음 또는 4 kB–16 MB. **D0TCM/D1TCM 두 개로 균등 분할**. F746은 총 64 kB | [A] Table 1-1 / [B] |
| ITCM 인터페이스 폭 | **64비트** (32비트 명령 2개 동시 페치 → 듀얼 이슈 전제) | [C] Atmel-44047B |
| DTCM 인터페이스 | **듀얼 32비트 인터리브(D0TCM/D1TCM)** — 코어와 DMA 동시 액세스 최적화 | [C] Atmel-44047B |
| AHBP 크기 | 구현 선택 64/128/256/512 MB | [A] Table 1-1 |
| linefill 버퍼 | **2개** | [A] §5.9.2 |
| 캐시 유지보수 | PPB 내 SCS 영역 레지스터 쓰기로 실행 | [A] §5.9.3 |

> **실측 정정(확인 + 보강)**:
> - L1 I/D 캐시 4 KB씩, 라인 32 B는 `CCSIDR`/`CTR`로 확인 (`0xF003E019` / `0xF007E009` / `0x8303C003`).
>   성능 곡선으로도 이중 확인 — 워킹셋 4 KB까지 D-cache 히트 **2.0 사이클**, 8 KB부터 **7~9 사이클**로 급변한다.
> - ITCM 16 KB(0x00004000부터 버스 폴트), DTCM 64 KB 확인. 다만 `ITCMCR.SZ = 0b1110`은
>   TRM 표대로 읽으면 8 MB가 되어 실제 크기(16 KB)와 맞지 않는다 → `02_cache_tcm.md` §3.4 참조.
> - **AHBP 크기는 실측으로 확정**: `AHBPCR = 0x00000009` → **EN=1, SZ=0b100 = 512 MB**.
> - TCM이 캐시 대상이 아니라는 것도 실측됨: ITCM 경로 순차 읽기가 D-cache ON/OFF에서
>   **492.10 MB/s로 완전히 동일**하다(반면 AXIM 경로는 159.96 → 304.82 MB/s로 ×1.91 반응).
> - TCM 접근 지연은 AXI SRAM의 1/3이다: **DTCM/ITCM 5.16 cyc vs SRAM1/SRAM2 15.66 cyc**
>   (D-cache OFF, 32 B 스트라이드, 루프 오버헤드 2.00 cyc 포함).

---

## 출처

### 1차 출처 (ARM 공식) — 직접 다운로드 후 전문 열람

1. **ARM DDI 0489F** — *Arm Cortex-M7 Processor Technical Reference Manual*, Revision r1p2, ID121118 (2018-11-15)
   https://www.pjrc.com/teensy/DDI0489F_cortex_m7_trm.pdf
   (동일 문서 ARM 호스팅: https://documentation-service.arm.com/static/5e906b038259fe2368e2a7bb , https://developer.arm.com/documentation/ddi0489/f/ )
   열람 섹션: §1.1 About, §1.1.1 Features, §1.1.3 Configuration options(Table 1-1), §1.2.1–1.2.13 Component blocks, §1.7 Product revisions, §2.4 System address map, §2.4.2 Unaligned accesses, §2.5 Exclusive monitor, §3.3.1 ACTLR(Table 3-3), §3.3.2 CPUID(Table 3-4), §5.2 Speculative accesses, §5.9.2 Store buffer, §6.1–6.3 MPU, §7.1–7.3 NVIC(Table 7-1, 7-2), §8.1–8.3 FPU(Table 8-1, 8-2), §9.1 Debug, §11.1–11.3 DWT(Table 11-1)

2. **ARM DUI 0646B** — *Arm Cortex-M7 Devices Generic User Guide*, ID082615 (2015)
   https://documentation-service.arm.com/static/61efd6602dd99944d051417b
   열람 섹션: §2.3.7 Exception entry and return(Figure 2-2), §3.1 Instruction set summary(Table 3-1), §4.6.5 MPU_RASR(Table 4-53), §4.6.6 MPU access permission attributes(Table 4-55, 4-56), §4.7.2 FPCCR(Table 4-60), §4.7.3 FPCAR

3. **Arm Cortex-M Processor Comparison Table** (제품 브리프)
   https://documentation-service.arm.com/static/6267de1c7e121f01fd22d677
   https://armkeil.blob.core.windows.net/developer/Files/pdf/product-brief/arm-cortex-m-processor-comparison-table.pdf

4. **ARM DDI 0403** — *ARMv7-M Architecture Reference Manual* — MVFR0/MVFR1 필드 정의, TEX/C/B/S 세부, 레이지 스태킹 아키텍처 정의의 근거 문서. **직접 열람하지 못함**(참조 표시만)
   https://developer.arm.com/documentation/ddi0403/

### 1차 출처 (ST 공식) — 로컬 CMSIS 소스로 확인

5. **ST CMSIS Device Header** `stm32f746xx.h` (STM32CubeF7 배포본, 로컬 경로)
   `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h`
   확인: L151–156 IRQn 목록(최대 `SPDIF_RX_IRQn = 97`), L166–172 코어 구성 매크로

6. **ARM CMSIS Core** `core_cm7.h`
   `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/CMSIS/Include/core_cm7.h`
   확인: L71 `__CORTEX_M = 7`, L1147–1230 DWT 비트 정의, L1762–1785 페리페럴 베이스 주소

### ST 공식 문서 (검색 스니펫으로만 확인 — 본문 다운로드 실패)

7. **DS10916** — *STM32F745xx/STM32F746xx Datasheet* (DM00166116)
   https://www.st.com/resource/en/datasheet/stm32f746ng.pdf ← **HTTP 403 / 타임아웃으로 본문 열람 실패**
   미러 시도(실패): https://www.mouser.com/pdfdocs/DM00166116.PDF , https://www.waveshare.com/w/upload/9/9b/STM32F745-STM32F746-Datasheet.pdf (403)

8. **AN4667** — *STM32F7 Series system architecture and performance*, DocID027643 Rev 4
   https://www.st.com/resource/en/application_note/an4667-stm32f7-series-system-architecture-and-performance-stmicroelectronics.pdf

9. **PM0253** — *STM32F7/H7 Series Cortex-M7 processor programming manual*, Rev 6
   https://www.st.com/resource/en/programming_manual/pm0253-stm32f7-series-and-stm32h7-series-cortexm7-processor-programming-manual-stmicroelectronics.pdf ← **다운로드 실패**

10. **ES0290** — *STM32F74xxx/STM32F75xxx device errata*, Rev 7 (2019-07)
    https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf ← **다운로드 실패**

11. **STM32F746NG 제품 페이지** — https://www.st.com/en/microcontrollers-microprocessors/stm32f746ng.html
12. **ST Arm Cortex-M7 페이지** — https://www.st.com/content/st_com/en/arm-32-bit-microcontrollers/arm-cortex-m7.html

### 2차 출처 (벤더 앱노트 / 컴파일러 모델)

13. **Atmel-44047B (Microchip)** — *How to Optimize Usage of SAM S70/E70/V7x Architecture*, 2016-03 — §1.1 Six-stage superscalar pipeline, §1.2 TCM, §3.4 Optimize Pipeline Usage
    http://ww1.microchip.com/downloads/en/appnotes/atmel-44047-cortex-m7-microcontroller-optimize-usage-sam-v71-v70-e70-s70-architecture_application-note.pdf

14. **LLVM `ARMScheduleM7.td`** — Cortex-M7 스케줄링 모델 (495줄 전문 열람)
    https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/ARM/ARMScheduleM7.td
    리뷰: https://reviews.llvm.org/D91355

15. **Arm Community** — "Where to find the execution cycles of Cortex m7 instruction"
    https://community.arm.com/support-forums/f/architectures-and-processors-forum/8952/where-to-find-the-execution-cycles-of-cortex-m7-instruction

16. **Arm Developer** — Cortex-M7 TRM 온라인판 §Product revisions, §Store buffer
    https://developer.arm.com/documentation/ddi0489/f/introduction/product-revisions
    https://developer.arm.com/documentation/ddi0489/f/memory-system/l1-caches/store-buffer

17. **MCU on Eclipse** — "Cycle Counting on ARM Cortex-M with DWT" (DWT_LAR 언락 관련)
    https://mcuoneclipse.com/2017/01/30/cycle-counting-on-arm-cortex-m-with-dwt/

---

## 미확인 / 추가 조사 필요

### 하드웨어 마이크로아키텍처 (공개 문서에 없음)

1. **BTAC 엔트리 수 / 연관도 / 인덱싱 방식** — DDI 0489F, DUI 0646B, ARM 비교표 어디에도 없음. Cortex-M7 Integration and Implementation Manual(라이선시 전용) 소관으로 추정. 실측 방법: 서로 다른 개수의 분기를 가진 루프를 만들어 CYCCNT로 오예측 임계점을 탐색.
2. **리턴 스택(Return Address Stack) 유무 및 깊이** — DDI 0489F §1.2.2 PFU 기능 목록에 항목 자체가 **없음**. 있다/없다를 단정할 수 없음. BL/BX LR 시퀀스로 실측 필요.
3. **정적 분기 예측기의 예측 정책** (backward-taken / forward-not-taken 여부) — 문서에 명시 없음.
4. **분기 오예측 페널티 정확한 사이클 수** — LLVM 주석의 "forward branch 6사이클"이 유일한 근거([C]). ARM 1차 문서 확인 불가.
5. **언얼라인 액세스 페널티의 정량적 사이클 수** — "2개 이상의 정렬 액세스로 분해"만 명시되어 있고 수치 없음. 실측 필요(정렬/비정렬 `LDR` 루프를 CYCCNT로 비교).
6. **명령어별 공식 사이클 표** — Cortex-M7에 대해 **존재하지 않음이 확인됨**. 본 문서의 §4.3–4.5, §5.3 표는 모두 LLVM 스케줄 모델 기반 [C]. **논문에 인용하려면 DWT_CYCCNT 실측이 필수.**
7. **인터럽트 레이턴시 12사이클 / tail-chaining 6사이클의 1차 출처** — DDI 0489F에는 없음. ARM 마케팅 자료 계열([C])만 확인됨. 실측 권장(GPIO 토글 + 로직 애널라이저 또는 CYCCNT).
8. **프리페치 큐 4×64비트가 실제 몇 명령어분인지** — 최대 16개 16비트 Thumb 명령 또는 8개 32비트 Thumb-2 명령으로 계산 가능하나 [D], 실효 깊이는 문서 없음.

### STM32F746 특정 (ST 문서 다운로드 실패로 미확인)

9. **STM32F746의 MPU 리전 수 (8인지 16인지)** — ARM 코어는 8/16 선택 가능. 널리 8로 알려져 있으나 [C], RM0385/PM0253 원문 미확인. → **런타임에 `MPU->TYPE >> 8 & 0xFF` 읽어 확정 가능.**
10. **STM32F746의 DWT 비교기 수 (2 또는 4) 및 FPB 브레이크포인트 수 (4 또는 8)** — 미확인. → 런타임에 `DWT->CTRL[31:28]`, `FP_CTRL` 읽어 확정.
11. **STM32F746의 ETM 구현 수준** (없음 / 명령 트레이스만 / 명령+데이터) — 미확인.
12. **STM32F746G-DISCO 보드에서 병렬 트레이스 포트(TRACECLK, TRACED[0:3]) 노출 여부** — UM1907 / MB1191 회로도 확인 필요.
13. **Device ID `0x449` + Rev Z 의 REV_ID 코드값** — DBGMCU_IDCODE(`0xE0042000` — 주의: 코어 CTI 영역과 주소가 겹치는 것처럼 보이나 STM32에서는 DBGMCU가 이 영역에 매핑됨) 의 REV_ID[31:16] 실제 값. RM0385 §DBGMCU 확인 필요.
14. **ES0290 에라타 중 Cortex-M7 코어 관련 항목 전체 목록** — 검색 스니펫으로 다음 2건만 확인:
    - **Arm 1259864**: write-through 메모리에서 `store → store(같은 doubleword) → load` 시퀀스에서 로드가 **최신 스토어가 아닌 이전 스토어의 데이터**를 받는 데이터 손상. [C]
    - **SP(R13)로의 단일 워드 로드 data-phase 중 인터럽트 발생 시**, 로드 명령이 추가로 한 번 더 실행되고 베이스 레지스터가 매 실행마다 잘못 갱신됨. [C]
    → **전체 목록과 정확한 조건/워크어라운드는 ES0290 원문 확인 필수.**
15. **STM32F746 Flash 대기 상태(wait state) 표와 216 MHz에서의 값** — AN4667 스니펫에서 `FLASH_WS`가 0–15 설정 가능하다는 것만 확인. 216 MHz/3.3 V에서의 권장값(통상 7 WS)은 DS10916/RM0385 확인 필요.
16. **런 모드 전류 소비 (216 MHz, mA)** — 데이터시트 다운로드 실패로 미확인.
17. **ARM 2.31 DMIPS/MHz vs ST 2.14 DMIPS/MHz 차이의 정확한 원인** (컴파일러 버전, 캐시 구성, ART 사용 여부) — 미확인.

### 조사 환경 제약

- 본 조사 환경에서 **`st.com` 도메인은 curl / WebFetch 모두 타임아웃 또는 403** 을 반환하여 DS10916, RM0385, PM0253, ES0290, UM1907 의 **본문을 한 줄도 열람하지 못했다.** 해당 문서 기반 사실은 모두 검색 스니펫 수준([B]/[C])이며, 원문 대조가 필요하다.
