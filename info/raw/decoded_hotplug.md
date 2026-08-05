# STM32F746G-DISCO ST-LINK 덤프 해석 (hotplug)


## 덤프 조건

| 항목 | 내용 |
|---|---|
| 연결 모드 | `HOTPLUG` — Hot Plug — **실행 중인 펌웨어의 상태를 그대로 읽음** |
| 덤프 시각 | 2026-07-27 15:43:25 (원본 JSON 파일 mtime 기준) |
| 원본 인덱스 | `info/raw/stlink_dump_hotplug.json` |
| 원본 바이너리 | `info/raw/regions_hotplug/*.bin` (50개 파일) |
| 덤프 생성 스크립트 | `info/code/stlink_dump.py` |
| 본 문서 생성 스크립트 | `info/code/decode_dump.py` |
| 캐시 ID 원문 | `info/raw/cache_id_raw_hotplug.txt` |


### CubeProgrammer 연결 배너 (원문 발췌)

| 항목 | 값 |
|---|---|
| ST-LINK SN | 0668FF39314D423143034442 |
| ST-LINK FW | V2J46M33 |
| Board | STM32F7508-DK |
| Voltage | 3.23V |
| SWD freq | 4000 KHz |
| Connect mode | Hot Plug |
| Reset mode | Software reset |
| Device ID | 0x449 |
| Revision ID | Rev Z |
| Device name | STM32F74x/STM32F75x |
| NVM size | 1 MBytes |
| Device type | MCU |
| Device CPU | Cortex-M7 |
| BL Version | 0x90 |

> 배너는 CubeProgrammer 가 스스로 판단해 출력한 값이다. 아래 본문의 해석은 모두 **레지스터 원시값에서 직접 계산**한 것이며, 둘이 일치하면 교차 검증이 된 것이다.


## 핵심 요약

이 덤프(`HOTPLUG` 모드)에서 레지스터 원시값으로 **직접 확정된** 하드웨어 사실이다. 각 항목 괄호 안이 근거 레지스터와 값이다.

- CPU 코어는 **Cortex-M7 r0p1** 이다 (CPUID=0x410FC271).
- NVIC 는 최대 128개의 외부 인터럽트 라인을 지원한다 (ICTR=0x00000003).
- DSP(SIMD) 명령 확장이 탑재되어 있다 — CMSIS-NN 의 SMLAD/SADD16 계열 사용 근거 (ID_ISAR3=0x01111131, SIMD_instrs=3).
- L1 캐시는 명령/데이터 분리형(Harvard)이며 L1 한 단계뿐이다 (CLIDR=0x09000003, Ctype1=3, LoC=1).
- L1 캐시 실측 용량: **D-cache 4 KB (32 세트 × 4 way × 32 B, CCSIDR=0xF003E019) / I-cache 4 KB (64 세트 × 2 way × 32 B, CCSIDR=0xF007E009)** → 합계 8 KB.
- DTCM 은 활성 상태이며 크기 필드가 64KB 로 STM32F746 사양(64KB)과 일치한다 (DTCMCR=0x0000003B).
- AHBP 페리페럴 포트가 활성이고 512MB 영역을 담당한다 (AHBPCR=0x00000009) → 페리페럴 접근이 AXIM/캐시를 우회한다.
- FPU 는 **FPv5-SP(단정밀도 전용)** 로 배정밀도 하드웨어가 없다 (MVFR0=0x10110021, Double-precision 필드=0) → `double` 은 소프트웨어 에뮬레이션.
- FMA(fused multiply-accumulate)와 반정밀도 변환 명령을 하드웨어로 지원한다 (MVFR1=0x11000011).
- MPU 는 통합형이며 리전 8개를 지원한다 (MPU_TYPE=0x00000800).
- 하드웨어 브레이크포인트 8개, 리터럴 비교기 0개 (FP_CTRL=0x10000080, FPB v2).
- 실리콘은 DEV_ID=0x449(STM32F74x / STM32F75x), REV_ID=0x1001(**Rev Z**) 이다 (DBGMCU_IDCODE=0x10016449).
- CoreSight ROM table 이 present 로 표시한 디버그 컴포넌트: **SCS, DWT, FPB, ITM** (0xE00FF000 엔트리).
- 이 개체의 96비트 고유 UID 는 `330034000850435436303520` 이다 (0x1FF0F420).
- 내장 Flash 용량은 **1024 KB (1 MB)** 로 공장 기록되어 있다 (0x1FF0F442 = 0x0400).
- 메인 PLL 계수: HSE ÷ M=25 × N=400 ÷ P=2 → VCO 400 MHz, PLLP 출력 200 MHz (**실제 SYSCLK 로 사용 중**) (RCC_PLLCFGR=0x29406419, RCC_CR=0x33038283).
- 클럭 트리 실측: **SYSCLK=200 MHz, HCLK=200 MHz, PCLK1=50 MHz, PCLK2=100 MHz** (RCC_CFGR=0x0000940A).
- Flash 대기 상태 = 6 WS, 프리페치 OFF, ART Accelerator **OFF** (FLASH_ACR=0x00000006).
- 부트 주소 옵션: BOOT_ADD0=0x00200000(ITCM 인터페이스의 내장 Flash 별칭), BOOT_ADD1=0x00100000(ITCM 인터페이스의 시스템 메모리(부트로더) 별칭) (FLASH_OPTCR1=0x00400080) → 리셋 직후 VTOR 이 0x00200000 인 이유.
- 전압 스케일 Scale 1, Over-drive ODEN=1/ODSWEN=1 (PWR_CR1=0x0003C100) — Over-drive 가 적용된 상태로, 이것이 200MHz 동작의 전제 조건이다.
- 외부 SDRAM 은 HCLK/2 = **100 MHz**, 16비트 버스, CAS 2, 4 뱅크, 12행/8열 → **8 MB** (FMC_SDCR1=0x00001954).
- LCD 는 **480×272** 활성 해상도로 구동 중이며 픽셀클럭 9.60 MHz, 전체 타이밍 566×286 → **약 59.3 Hz** (LTDC_AWCR=0x0215011B, TWCR=0x0235011D).
- AF 로 설정된 핀은 총 **72개**이며 분포는 AF12 44핀, AF14 27핀, AF9 1핀 이다 — 대부분이 FMC(SDRAM/SDMMC) 와 LTDC 배선이다 (GPIOx_MODER/AFRL/AFRH 실측).
- 0x00200000 과 0x08000000 의 내용이 바이트 단위로 동일하다 → **내장 Flash 는 ITCM/AXIM 두 경로로 동시에 보인다**. 반면 0x00000000 은 내용이 달라 **ITCM RAM 그 자체**임이 확인된다.
- **JPEG 하드웨어 코덱 미탑재**: 0x50051000 읽기가 버스 에러로 실패했다 (STM32F76x/77x 에만 존재하는 블록).

### 그 밖에 확정된 사실 (주로 덤프 시점의 소프트웨어 설정)

- 코어 성능 저하 비트(ACTLR)가 모두 0 → IT 폴딩·D-cache read-allocate·다중사이클 명령 중단이 전부 기본(성능 우선) 상태 (ACTLR=0x00000000).
- 덤프 시점 I-cache=ON, D-cache=ON, 분기예측=ON (SCB_CCR=0x00070200).
- 코어는 리틀엔디언으로 동작한다 (AIRCR.ENDIANNESS=0).
- 벡터 테이블 베이스(VTOR)는 0x00200000 → ITCM 인터페이스를 통한 내장 Flash 별칭(0x00200000) 이다.
- 덤프 시점까지 기록된 폴트가 없다 (CFSR=0x00000000, HFSR=0x00000000).
- FPU 코프로세서 접근은 허용(FPU 활성) 상태였다 (CPACR=0x00F00000).
- 캐시/TCM ECC 는 켜져 있고, write-through 강제는 꺼져 있다 (CACR=0x00000002).
- 비동기 버스 폴트 기록이 없다 — ITCM/DTCM/AHBP/AXIM/EPPB 모두 정상 (ABFSR=0x00000000).
- MPU 는 **활성화** 상태였다 (MPU_CTRL=0x00000005, PRIVDEFENA=1).
- DWT 레지스터가 전부 0 으로 읽힌다 — DEMCR.TRCENA=0 이라 트레이스 블록이 꺼져 있기 때문이며, DWT 부재의 근거가 아니다 (DWT_CTRL=0x00000000).
- ROM table 은 TPIU, ETM 를 present=0 으로 보고하지만 해당 주소의 PIDR 은 ARM 정식 부품 ID 를 반환한다(상충 → 확정 불가).
- 패키지 데이터 레지스터(0x1FF07BF0)가 0xFFFFFFFF → 미프로그램 상태로 패키지 판별 불가.
- LTDC 픽셀 클럭은 PLLSAI(192MHz)/R=5/DIVR=4 = **9.60 MHz** 이다 (PLLSAICFGR=0x54003000, DCKCFGR1=0x00010000).
- 덤프 시점에 클럭이 공급되던 페리페럴은 총 18개다: AHB1ENR=[GPIOC, GPIOD, GPIOE, GPIOF, GPIOG, GPIOH, GPIOI, GPIOJ, GPIOK, DTCMRAM, DMA2, DMA2D]; AHB2ENR=[없음]; AHB3ENR=[FMC]; APB1ENR=[TIM6, PWR]; APB2ENR=[SDMMC1, SYSCFG, LTDC].
- 마지막 리셋 원인 플래그(누적, RMVF 로 지우기 전까지 유지): PORRSTF(POR/PDR), PINRSTF(NRST 핀), BORRSTF(브라운아웃) (RCC_CSR=0x0E000003).
- 읽기 보호는 **RDP Level 0** (RDP=0xAA), 워치독은 둘 다 소프트웨어 시작, BOR 은 BOR off (VBOR0, ~1.8V 리셋 임계) (FLASH_OPTCR=0xC0FFAAFD).
- SDRAM 리프레시 설정은 행당 15.59 µs (전체 63.9 ms / 4096행) 이다 (FMC_SDRTR=0x00000C06).
- QUADSPI 레지스터가 전부 0 — 컨트롤러 클럭이 꺼진 상태라 플래시 용량/클럭 산출 불가 (QUADSPI_CR/DCR = 0x00000000).
- LTDC 레이어1 프레임버퍼는 0xC0000000 (외부 SDRAM(0xC0000000 대)), 픽셀 포맷 ARGB8888 이다.
- LTDC 레이어2 프레임버퍼는 0xC007F800 (외부 SDRAM(0xC0000000 대)), 픽셀 포맷 ARGB8888 이다.
- 두 레이어의 프레임버퍼 간격(522240 B = 0x7F800)이 한 프레임 크기(피치 1920 B × 272 라인 = 522240 B)와 정확히 일치한다 → SDRAM 에 두 프레임을 연속 배치한 더블 버퍼 구성이다.
- 펌웨어 벡터 테이블: 초기 SP = 0x20050000 (SRAM2 최상단 경계(= 이 영역 끝 다음 주소)), Reset_Handler = 0x0802034C (Flash AXIM) → 코드는 AXIM 경로의 Flash 에서 실행된다.
- ITCM RAM(0x00000000) 선두에 비-0 임의값이 남아 있다 → ITCM 은 초기화되지 않은 SRAM 이며 리셋 후에도 이전 내용이 남는다.
- 시스템 부트로더가 실재한다 — 초기 SP=0x20003AF8, 진입점=0x1FF00C88 (0x1FF00000 에서 직접 읽음).

### 요약에 붙는 경고

- ⚠️ **ITCMCR=0x00000073 의 SZ 필드(0b1110)는 Cortex-M7 TRM 표에서 '8MB' 를 뜻하는데, STM32F746 데이터시트의 ITCM 크기(16KB)와 어긋난다.** ST 가 TCM 인터페이스에 할당한 주소공간 크기를 보고하는 것인지, 실제 RAM 크기를 보고하는 것인지 이 덤프만으로는 확정할 수 없다. 온디바이스 aliasing 실측(별칭 주소에 쓰고 되읽어 wrap-around 지점을 찾는 방법)으로 별도 확인 예정.
- ⚠️ **DWT_CTRL 이 0x00000000 으로 읽혔다. 이는 'DWT 가 없다'는 뜻이 아니다.** DEMCR.TRCENA(비트 24) = 0 로 트레이스 블록 전원/클럭이 꺼져 있으면 DWT 레지스터가 전부 0 으로 읽힌다. DWT_CYCCNT 기반 사이클 측정을 하려면 먼저 `CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk` 를 수행해야 한다. 따라서 이 덤프로는 NUMCOMP 등 DWT 실제 사양을 확정할 수 없다(해석 불가).
- ⚠️ ROM table 엔트리에서 **TPIU, ETM** 의 present 비트가 0 이다. 그런데 해당 주소의 PIDR/CIDR 는 ARM 정식 부품 ID 로 읽힌다(위 표). 즉 물리적으로는 존재하나 ROM table 이 '없음'으로 보고하는 상태다. 이 덤프만으로는 미탑재인지, 트레이스 인프라가 꺼져 있어서인지 확정할 수 없다(해석 불가).
- ⚠️ QUADSPI 블록의 클럭(RCC_AHB3ENR.QSPIEN)이 꺼져 있으면 레지스터가 0 으로 읽힌다. 따라서 이 덤프로는 **QSPI 클럭 주파수와 플래시 용량을 계산할 수 없다(해석 불가)**. QSPI 플래시의 존재 자체는 이 덤프로 증명되지도 반증되지도 않는다.
- ⚠️ 다음 포트는 0x400 블록 **전체가 0 이 아닌 하나의 32비트 값으로 반복**되어 있다 — 정상적인 레지스터 이미지가 아니다(덤프 아티팩트). GPIOA(0x40020000, 반복값 0xA8000000, RCC 클럭 OFF), GPIOB(0x40020400, 반복값 0x00000280, RCC 클럭 OFF). 반복된 값이 각 포트 MODER 의 리셋 기본값과 일치하고 해당 포트의 AHB1ENR 클럭이 꺼져 있으므로, **이 포트들의 값은 신뢰할 수 없어 아래 핀 상태표와 AF 목록에서 제외한다(해석 불가).**


## 코어 (Cortex-M7 식별 및 제어 레지스터)

`CPUID @ 0xE000ED00 = 0x410FC271`

- IMPLEMENTER[31:24] = 0x41 → 0x41 = ARM Limited
- VARIANT[23:20] = 0 → 메이저 리비전 r0
- ARCHITECTURE[19:16] = 0xF → ARMv7-M(확장)
- PARTNO[15:4] = 0xC27 → Cortex-M7
- REVISION[3:0] = 1 → 마이너 리비전 p1
- **종합: Cortex-M7 r0p1**

`ICTR @ 0xE000E004 = 0x00000003`

- INTLINESNUM[3:0] = 3 → NVIC 가 지원하는 외부 인터럽트 라인 수 = (3+1)×32 = **128개**

`ACTLR @ 0xE000E008 = 0x00000000`

- DISMCYCINT[0] = 0 → 다중 사이클 명령(LDM/STM/DIV 등)의 인터럽트 중단 허용(기본, 인터럽트 지연 최소)
- DISFOLD[2] = 0 → IT 폴딩 활성화(기본, 성능 유리)
- FPEXCODIS[10] = 0 → FPU 예외 출력 활성화(기본)
- DISRAMODE[11] = 0 → D-cache 의 read-allocate 모드 활성화(기본)
- DISITMATBFLUSH[12] = 0 → ITM/DWT ATB flush 활성화(기본)
- DISDEFWBUF: core_cm7.h 에 정의 없음 (Cortex-M3/M4 전용 비트) → 해석 불가

`SCB_CCR @ 0xE000ED14 = 0x00070200`

- NONBASETHRDENA[0] = 0
- USERSETMPEND[1] = 0
- UNALIGN_TRP[3] = 0 → 비정렬 접근 트랩 해제됨
- DIV_0_TRP[4] = 0 → 0 나눗셈 트랩 해제됨
- BFHFNMIGN[8] = 0
- STKALIGN[9] = 1 → 예외 진입 시 8바이트 스택 정렬 설정됨
- DC[16] = 1 → **D-cache ON**
- IC[17] = 1 → **I-cache ON**
- BP[18] = 1 → 분기 예측 ON

`SCB_AIRCR @ 0xE000ED0C = 0xFA050300`

- VECTKEYSTAT[31:16] = 0xFA05 (읽기 시 항상 0xFA05)
- ENDIANNESS[15] = 0 → **리틀엔디언**
- PRIGROUP[10:8] = 3 (0b011) → STM32F7(우선순위 4비트) 기준 선점 4비트 / 서브 0비트 (HAL 의 NVIC_PRIORITYGROUP_4)
- SYSRESETREQ[2] = 0

`SCB_VTOR @ 0xE000ED08 = 0x00200000`

- TBLOFF[31:7] → 벡터 테이블 베이스 = 0x00200000 → ITCM 인터페이스를 통한 내장 Flash 별칭(0x00200000)

`SCB_SCR @ 0xE000ED10 = 0x00000000`

- SLEEPONEXIT[1] = 0
- SLEEPDEEP[2] = 0 → 일반 sleep
- SEVONPEND[4] = 0

`SCB_SHCSR @ 0xE000ED24 = 0x00010000`

- MEMFAULTACT[0]=0 BUSFAULTACT[1]=0 USGFAULTACT[3]=0
- SVCALLACT[7]=0 MONITORACT[8]=0 PENDSVACT[10]=0 SYSTICKACT[11]=0
- MEMFAULTENA[16] = 1 → MemManage 폴트 핸들러 설정됨
- BUSFAULTENA[17] = 0 → BusFault 핸들러 해제됨
- USGFAULTENA[18] = 0 → UsageFault 핸들러 해제됨

`SCB_CFSR @ 0xE000ED28 = 0x00000000`

| 서브레지스터 | 값 | 세워진 플래그 |
|---|---|---|
| MMFSR (MemManage) | 0x00 | 플래그 없음 |
| BFSR (BusFault) | 0x00 | 플래그 없음 |
| UFSR (UsageFault) | 0x0000 | 플래그 없음 |

`SCB_HFSR @ 0xE000ED2C = 0x00000000`

- VECTTBL[1] = 0 → 벡터 테이블 읽기 실패
- FORCED[30] = 0 → 하위 폴트가 HardFault 로 승격
- DEBUGEVT[31] = 0 → 디버그 이벤트

| 레지스터 | 주소 | 값 | 해석 |
|---|---|---|---|
| SCB_MMFAR | 0xE000ED34 | 0x00000000 | MemManage 폴트 주소 (MMARVALID 일 때만 유효) |
| SCB_BFAR | 0xE000ED38 | 0x00000000 | BusFault 주소 (BFARVALID 일 때만 유효) |
| SCB_AFSR | 0xE000ED3C | 0x00000000 | Auxiliary Fault Status (구현 정의) |

`SCB_CPACR @ 0xE000ED88 = 0x00F00000`

- CP10[21:20] = 0b11 → 전체 접근
- CP11[23:22] = 0b11 → 전체 접근
- **FPU 사용 가능(활성)**

`SYST_CALIB @ 0xE000E01C = 0x4000493E`

- TENMS[23:0] = 18750 (0x493e)
- SKEW[30] = 1 → TENMS 가 정확히 10ms 가 아님
- NOREF[31] = 0 → 외부 레퍼런스 클럭 있음(HCLK/8)


### 코어 기능 ID 레지스터

| 레지스터 | 주소 | 값 | 해석 |
|---|---|---|---|
| ID_PFR0 | 0xE000ED40 | 0x00000030 | State0(ARM ISA)[3:0]=0 → 미지원, State1(Thumb)[7:4]=3 → Thumb-2 지원 |
| ID_PFR1 | 0xE000ED44 | 0x00000200 | 프로그래머 모델[11:8]=2 → ARMv7-M (M-profile) |
| ID_DFR0 | 0xE000ED48 | 0x00100000 | M-profile 디버그 모델[23:20]=1 → 지원 |
| ID_AFR0 | 0xE000ED4C | 0x00000000 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |
| ID_MMFR0 | 0xE000ED50 | 0x00110030 | PMSA[7:4]=3 → PMSAv7 (MPU 있음), TCM 지원[19:16]=1, Aux 레지스터[23:20]=1 |
| ID_MMFR1 | 0xE000ED54 | 0x00000000 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |
| ID_MMFR2 | 0xE000ED58 | 0x01000000 | WFI stalling[27:24]=1 → 지원 |
| ID_MMFR3 | 0xE000ED5C | 0x00000000 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |
| ID_ISAR0 | 0xE000ED60 | 0x01101110 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |
| ID_ISAR1 | 0xE000ED64 | 0x02112000 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |
| ID_ISAR2 | 0xE000ED68 | 0x20232231 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |
| ID_ISAR3 | 0xE000ED6C | 0x01111131 | SIMD_instrs[7:4]=3 → **DSP(SIMD) 확장 탑재** — SADD16/SMLAD 등 사용 가능, Saturate_instrs[3:0]=1 |
| ID_ISAR4 | 0xE000ED70 | 0x01310132 | 상세 필드 해석 생략(ARMv7-M ARM 참조) |


## L1 캐시

`CLIDR @ 0xE000ED78 = 0x09000003`

- Ctype1[2:0] = 3 → L1: **I-cache 와 D-cache 분리(Harvard)**
- LoUIS[23:21] = 0 (Level of Unification, Inner Shareable)
- LoC[26:24] = 1 → **Level of Coherency = L1** (캐시 레벨은 총 1단계)
- LoUU[29:27] = 1 (Level of Unification, Uniprocessor)

`CTR @ 0xE000ED7C = 0x8303C003`

- IminLine[3:0] = 3 → I-cache 최소 라인 8 워드 = **32 B**
- DminLine[19:16] = 3 → D-cache 최소 라인 8 워드 = **32 B**
- ERG[23:20] = 0 → Exclusive Reservation Granule 0 → 미구현/미보고
- CWG[27:24] = 3 → Cache Write-back Granule 8 워드 = **32 B**
- Format[31:29] = 0b100 → **ARMv7 CTR 포맷**


### CCSIDR — D-cache (CSSELR=0)

`CCSIDR @ 0xE000ED80 = 0xF003E019`

- LineSize[2:0] = 1 → 캐시 라인 8 워드 = **32 바이트**
- Associativity[12:3] = 3 → **4-way set associative**
- NumSets[27:13] = 31 → **32 세트**
- WA[28] = 1 → write-allocate 지원
- RA[29] = 1 → read-allocate 지원
- WB[30] = 1 → write-back 지원
- WT[31] = 1 → write-through 지원
- **실제 용량 = 32 세트 × 4 way × 32 B = 4096 B = 4 KB**


### CCSIDR — I-cache (CSSELR=1)

`CCSIDR @ 0xE000ED80 = 0xF007E009`

- LineSize[2:0] = 1 → 캐시 라인 8 워드 = **32 바이트**
- Associativity[12:3] = 1 → **2-way set associative**
- NumSets[27:13] = 63 → **64 세트**
- WA[28] = 1 → write-allocate 지원
- RA[29] = 1 → read-allocate 지원
- WB[30] = 1 → write-back 지원
- WT[31] = 1 → write-through 지원
- **실제 용량 = 64 세트 × 2 way × 32 B = 4096 B = 4 KB**

> 근거 원문: `info/raw/cache_id_raw_hotplug.txt` (CSSELR 에 0/1 을 써 넣고 CCSIDR 을 각각 읽은 CLI 출력)

`CACR @ 0xE000EF9C = 0x00000002`

- SIWT[0] = 0 → Shared cacheable-is-WT 비활성
- ECCEN[1] = 1 → 캐시/TCM **ECC 활성**
- FORCEWT[2] = 0 → 전 영역 write-through 강제 비활성(write-back 허용)


## TCM / 버스 인터페이스

`ITCMCR @ 0xE000EF90 = 0x00000073`

- EN[0] = 1 → TCM **활성** (별칭 주소 0x00000000)
- RMW[1] = 1 → read-modify-write 활성(바이트/하프워드 쓰기 지원)
- RETEN[2] = 0 → 리셋 시 내용 유지 비활성
- SZ[6:3] = 0b1110 → Cortex-M7 TRM 인코딩 표상 **8MB**

> ⚠️ **경고**: **ITCMCR=0x00000073 의 SZ 필드(0b1110)는 Cortex-M7 TRM 표에서 '8MB' 를 뜻하는데, STM32F746 데이터시트의 ITCM 크기(16KB)와 어긋난다.** ST 가 TCM 인터페이스에 할당한 주소공간 크기를 보고하는 것인지, 실제 RAM 크기를 보고하는 것인지 이 덤프만으로는 확정할 수 없다. 온디바이스 aliasing 실측(별칭 주소에 쓰고 되읽어 wrap-around 지점을 찾는 방법)으로 별도 확인 예정.

`DTCMCR @ 0xE000EF94 = 0x0000003B`

- EN[0] = 1 → TCM **활성** (별칭 주소 0x20000000)
- RMW[1] = 1 → read-modify-write 활성(바이트/하프워드 쓰기 지원)
- RETEN[2] = 0 → 리셋 시 내용 유지 비활성
- SZ[6:3] = 0b0111 → Cortex-M7 TRM 인코딩 표상 **64KB**

`AHBPCR @ 0xE000EF98 = 0x00000009`

- EN[0] = 1 → AHBP(페리페럴 포트) **활성**
- SZ[3:1] = 0b100 → AHBP 영역 크기 **512MB** (0x40000000 기준)

`AHBSCR @ 0xE000EFA0 = 0x00000800`

- CTL[1:0] = 0b00 → AHBS(슬레이브 포트) 중재 제어 0b00 = 기본(소프트웨어 접근 우선)
- TPRI[10:2] = 0 → AHBS 임계 우선순위
- INITCOUNT[15:11] = 1 → 페어니스 카운터 초기값

`ABFSR @ 0xE000EFA8 = 0x00000000`

- ITCM[0]=0 DTCM[1]=0 AHBP[2]=0 AXIM[3]=0 EPPB[4]=0
- AXIMTYPE[9:8] = 0b00 → OKAY
- → **기록된 비동기 버스 폴트 없음**


## FPU (MVFR0/1/2)

`MVFR0 @ 0xE000EF40 = 0x10110021`

- A_SIMD registers[3:0] = 1 → 16 × 64비트 레지스터 뱅크(D0-D15 / S0-S31)
- Single-precision[7:4] = 2 → **단정밀도 지원(모든 VFP 연산)**
- Double-precision[11:8] = 0 → **배정밀도 하드웨어 미지원**
- FP exception trapping[15:12] = 0 → 트랩 미지원(예외는 상태 비트로만 보고)
- Divide[19:16] = 1 → VDIV 지원
- Square root[23:20] = 1 → VSQRT 지원
- Short vectors[27:24] = 0 → 미지원
- FP rounding modes[31:28] = 1 → 모든 라운딩 모드 지원

`MVFR1 @ 0xE000EF44 = 0x11000011`

- FtZ mode[3:0] = 1 → **Flush-to-Zero 만 지원** (비정규수는 0으로 플러시)
- D_NaN mode[7:4] = 1 → **Default NaN 만 지원**
- A_SIMD load/store[11:8] = 0, A_SIMD integer[15:12] = 0, A_SIMD single-precision[19:16] = 0, A_SIMD half-precision[23:20] = 0 → Advanced SIMD(NEON) 미탑재
- FP HPFP[27:24] = 1 → **반정밀도(half-precision) 변환 명령 지원(VCVTB/VCVTT)**
- FP fused MAC[31:28] = 1 → **fused multiply-accumulate 지원(VFMA/VFMS)**

`MVFR2 @ 0xE000EF48 = 0x00000040`

- VFP_Misc[7:4] = 4 → **FPv5 부가 명령 지원** — VSEL, VMAXNM/VMINNM, VRINT*, VCVTA/N/P/M

> **결론: 배정밀도(double) 부동소수 하드웨어가 없다 → FPv5-SP (단정밀도 전용) FPU.**
> `double` 연산은 전부 소프트웨어 라이브러리(`__aeabi_d*`)로 처리되므로 추론 커널은 반드시 `float`(단정밀도) 또는 정수 연산으로 작성해야 한다.


## MPU

`MPU_TYPE @ 0xE000ED90 = 0x00000800`

- SEPARATE[0] = 0 → **통합 MPU(명령·데이터 공용)**
- DREGION[15:8] = 8 → **데이터 리전 8개**
- IREGION[23:16] = 0 → 명령 전용 리전 0개

`MPU_CTRL @ 0xE000ED94 = 0x00000005`

- ENABLE[0] = 1 → MPU **활성화됨**
- HFNMIENA[1] = 0 → HardFault/NMI 중에도 MPU 무시
- PRIVDEFENA[2] = 1 → 특권 모드의 기본 메모리 맵 **허용**(리전 미지정 영역도 접근 가능)

`MPU_RNR/RBAR/RASR = 0x00000002 / 0xA0000002 / 0x13050019`

- RNR = 0x00000002 → 마지막으로 선택된 리전 번호 **2**
- RBAR ADDR[31:5] = 0xA0000000 (REGION[3:0]=2, VALID[4]=0)
- RASR ENABLE[0] = 1
- RASR SIZE[5:1] = 12 → 리전 크기 2^(12+1) = **8 KB**
- RASR SRD[15:8] = 0x00 (서브리전 비활성 마스크)
- RASR B[16]=1 C[17]=0 S[18]=1 TEX[21:19]=0 → Device/Shareable (캐시 불가)
- RASR AP[26:24] = 3 → 전체 RW
- RASR XN[28] = 1 → 명령 실행 **금지**

> MPU_RBAR/RASR 는 RNR 로 선택된 **한 리전만** 보인다. 덤프 시점에 마지막으로 선택돼 있던 리전 값이므로 전체 리전 구성은 알 수 없다.


## 디버그 / 트레이스

| 레지스터 | 주소 | 값 | 해석 |
|---|---|---|---|
| DHCSR | 0xE000EDF0 | 0x00010001 | C_DEBUGEN[0]=1, C_HALT[1]=0, S_HALT[17]=0, S_REGRDY[16]=1 → 코어 실행 중 |
| DEMCR | 0xE000EDFC | 0x00000000 | TRCENA[24]=0 → 트레이스/DWT 블록 **비활성**, VC_CORERESET[0]=0, MON_EN[16]=0 |

`DWT_CTRL @ 0xE0001000 = 0x00000000`

- NUMCOMP[31:28] = 0 → DWT 비교기 0개
- NOTRCPKT[27] = 0, NOEXTTRIG[26] = 0, NOCYCCNT[25] = 0, NOPRFCNT[24] = 0
- CYCCNTENA[0] = 0 → 사이클 카운터 해제됨
- PCSAMPLENA[12] = 0, EXCTRCENA[16] = 0, CYCEVTENA[22] = 0

> ⚠️ **경고**: **DWT_CTRL 이 0x00000000 으로 읽혔다. 이는 'DWT 가 없다'는 뜻이 아니다.** DEMCR.TRCENA(비트 24) = 0 로 트레이스 블록 전원/클럭이 꺼져 있으면 DWT 레지스터가 전부 0 으로 읽힌다. DWT_CYCCNT 기반 사이클 측정을 하려면 먼저 `CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk` 를 수행해야 한다. 따라서 이 덤프로는 NUMCOMP 등 DWT 실제 사양을 확정할 수 없다(해석 불가).

`FP_CTRL @ 0xE0002000 = 0x10000080`

- ENABLE[0] = 0 → FPB 비활성
- KEY[1] = 0 (쓰기 시에만 의미)
- NUM_CODE[7:4]+[14:12] = 8 → **하드웨어 명령 브레이크포인트 8개**
- NUM_LIT[11:8] = 0 → 리터럴 비교기 0개
- REV[31:28] = 1 → FPB v2 (Cortex-M7, 32비트 주소 비교, 주소 리맵 없음)

`DBGMCU_IDCODE @ 0xE0042000 = 0x10016449`

- DEV_ID[11:0] = 0x449 → **STM32F74x / STM32F75x**
- REV_ID[31:16] = 0x1001 → **Rev Z**

`DBGMCU_CR @ 0xE0042004 = 0x00000000`

- DBG_SLEEP[0] = 0, DBG_STOP[1] = 0, DBG_STANDBY[2] = 0 → 저전력 모드에서 디버그 유지 전부 비활성
- TRACE_IOEN[5] = 0 → 트레이스 핀 비활성
- TRACE_MODE[7:6] = 0b00


### CoreSight ROM table (0xE00FF000)

| # | 엔트리 값 | present / format | 컴포넌트 베이스 | 식별 |
|---|---|---|---|---|
| 0 | 0xFFF0F003 | 있음 / 32비트 | 0xE000E000 | **SCS** — System Control Space (SCB/NVIC/MPU/FPU/캐시제어) |
| 1 | 0xFFF02003 | 있음 / 32비트 | 0xE0001000 | **DWT** — Data Watchpoint and Trace |
| 2 | 0xFFF03003 | 있음 / 32비트 | 0xE0002000 | **FPB** — Flash Patch and Breakpoint |
| 3 | 0xFFF01003 | 있음 / 32비트 | 0xE0000000 | **ITM** — Instrumentation Trace Macrocell |
| 4 | 0xFFF41002 | **없음** / 32비트 | 0xE0040000 | **TPIU** — Trace Port Interface Unit |
| 5 | 0xFFF42002 | **없음** / 32비트 | 0xE0041000 | **ETM** — Embedded Trace Macrocell |
| 6 | 0x00000000 | - | - | 종료 엔트리(0x00000000) — 이후 엔트리 없음 |


### CoreSight 컴포넌트 ID (PIDR/CIDR)

| 컴포넌트 | 베이스 | PIDR0-3 / PIDR4 | CIDR0-3 | Part number | 제조사·속성 |
|---|---|---|---|---|---|
| SCS | 0xE000E000 | 0C B0 0B 00 / P4=04 | 0D E0 05 B1 | 0x00C = Cortex-M7 SCS (System Control Space) | ARM Ltd. (JEP106 continuation 4, ID 0x3B); rev 0; JEDEC=1; 크기 1×4KB; 클래스 0xE = Generic IP component |
| DWT | 0xE0001000 | 02 B0 0B 00 / P4=04 | 0D E0 05 B1 | 0x002 = Cortex-M7 DWT | ARM Ltd. (JEP106 continuation 4, ID 0x3B); rev 0; JEDEC=1; 크기 1×4KB; 클래스 0xE = Generic IP component |
| FPB | 0xE0002000 | 0E B0 0B 00 / P4=04 | 0D E0 05 B1 | 0x00E = Cortex-M7 FPB | ARM Ltd. (JEP106 continuation 4, ID 0x3B); rev 0; JEDEC=1; 크기 1×4KB; 클래스 0xE = Generic IP component |
| ITM | 0xE0000000 | 01 B0 0B 00 / P4=04 | 0D E0 05 B1 | 0x001 = Cortex-M7 ITM | ARM Ltd. (JEP106 continuation 4, ID 0x3B); rev 0; JEDEC=1; 크기 1×4KB; 클래스 0xE = Generic IP component |
| TPIU | 0xE0040000 | A9 B9 0B 00 / P4=04 | 0D 90 05 B1 | 0x9A9 = CoreSight TPIU-M7 | ARM Ltd. (JEP106 continuation 4, ID 0x3B); rev 0; JEDEC=1; 크기 1×4KB; 클래스 0x9 = CoreSight component |
| ETM | 0xE0041000 | 75 B9 0B 00 / P4=04 | 0D 90 05 B1 | 0x975 = CoreSight ETM-M7 | ARM Ltd. (JEP106 continuation 4, ID 0x3B); rev 0; JEDEC=1; 크기 1×4KB; 클래스 0x9 = CoreSight component |
| ROM table | 0xE00FF000 | C7 B4 0B 00 / P4=04 | 0D 10 05 B1 | 0x4C7 = Cortex-M7 PPB ROM table | 클래스 0x1 = ROM table |

> ⚠️ **경고**: ROM table 엔트리에서 **TPIU, ETM** 의 present 비트가 0 이다. 그런데 해당 주소의 PIDR/CIDR 는 ARM 정식 부품 ID 로 읽힌다(위 표). 즉 물리적으로는 존재하나 ROM table 이 '없음'으로 보고하는 상태다. 이 덤프만으로는 미탑재인지, 트레이스 인프라가 꺼져 있어서인지 확정할 수 없다(해석 불가).


## 팩토리 프로그램 영역 (ST 공장 기록, 읽기 전용)

| 항목 | 주소 | 원시값 | 해석 |
|---|---|---|---|
| UID (96비트) | 0x1FF0F420 | 330034000850435436303520 | 워드: 0x00340033 / 0x54435008 / 0x20353036 — 바이트 ASCII 해석 `3.4..PCT605 ` (ST 는 하위 워드에 웨이퍼 X/Y, 상위에 로트번호 ASCII 를 기록) |
| Flash size | 0x1FF0F442 | 0x0400 | **1024 KB = 1 MB** |
| VREFINT_CAL | 0x1FF0F44A | 0x05D4 | 1492 — 30°C·VDDA=3.3V 에서 측정한 내부 기준전압 ADC 원시값 |
| TS_CAL1 | 0x1FF0F44C | 0x03AE | 942 — 30°C·VDDA=3.3V 온도센서 ADC 원시값 |
| TS_CAL2 | 0x1FF0F44E | 0x04AB | 1195 — 110°C·VDDA=3.3V 온도센서 ADC 원시값 |
| (파생) 온도센서 기울기 | - | - | (110−30)/(TS_CAL2−TS_CAL1) = 80/253 = 0.3162 °C/LSB |

`PACKAGE @ 0x1FF07BF0 = 0xFFFFFFFF`

- 전 비트 1 → 패키지 데이터 레지스터가 프로그램되지 않았거나 이 실리콘 리비전에서 미지원. **해석 불가**


## RCC (클럭 트리 및 페리페럴 클럭)


### RCC_CR — 발진기/PLL 상태

`RCC_CR @ 0x40023800 = 0x33038283`

- HSION[0]=1 HSIRDY[1]=1 → HSI(16MHz) ON/ready
- HSITRIM[7:3]=16, HSICAL[15:8]=0x82 (공장 캘리브레이션)
- HSEON[16]=1 HSERDY[17]=1 → HSE(외부 25MHz) **ON**/ready
- HSEBYP[18]=0 → 크리스털 발진기 모드
- CSSON[19]=0 → 클럭 보안 시스템 해제됨
- PLLON[24]=1 PLLRDY[25]=1 → 메인 PLL **ON**/locked
- PLLI2SON[26]=0 PLLI2SRDY[27]=0 → PLLI2S OFF
- PLLSAION[28]=1 PLLSAIRDY[29]=1 → PLLSAI **ON**


### RCC_PLLCFGR — 메인 PLL 및 SYSCLK 역산

`RCC_PLLCFGR @ 0x40023804 = 0x29406419`

- PLLM[5:0] = 25 → VCO 입력 = HSE = 25 MHz / 25 = **1.000 MHz** (데이터시트 권장 1~2MHz)
- PLLN[14:6] = 400 → VCO 출력 = 1.000 × 400 = **400.0 MHz**
- PLLP[17:16] = 0b00 → 분주 2 → PLLP(SYSCLK 후보) = **200.0 MHz**
- PLLSRC[22] = 1 → PLL 입력 = HSE = 25 MHz
- PLLQ[27:24] = 9 → PLLQ(USB/SDMMC/RNG) = 400.0/9 = **44.444 MHz**  ← 48MHz 가 아니므로 USB FS/HS 를 이 클럭으로 쓸 수 없다
- 비트[30:28] = 0b010 → STM32F746 의 PLLCFGR 에는 PLLR 필드가 정의되어 있지 않다 (stm32f746xx.h 에 RCC_PLLCFGR_PLLR 없음) → 예약 비트, 해석 불가


### RCC_CFGR — SYSCLK 선택 및 버스 프리스케일러

`RCC_CFGR @ 0x40023808 = 0x0000940A`

- SW[1:0] = 2 → 요청된 시스템 클럭 소스 = PLL(PLLP)
- SWS[3:2] = 2 → **실제 사용 중인 시스템 클럭 = PLL(PLLP)** → SYSCLK = **200.0 MHz**
- HPRE[7:4] = 0b0000 → AHB 분주 /1 → **HCLK = 200.0 MHz**
- PPRE1[12:10] = 0b101 → APB1 분주 /4 → **PCLK1 = 50.0 MHz** (APB1 타이머 클럭 100.0 MHz)
- PPRE2[15:13] = 0b100 → APB2 분주 /2 → **PCLK2 = 100.0 MHz** (APB2 타이머 클럭 200.0 MHz)
- RTCPRE[20:16] = 0 → HSE→RTC 분주 (비활성)
- MCO1[22:21]=0 MCO1PRE[26:24]=0 MCO2[31:30]=0 MCO2PRE[29:27]=0

`RCC_CIR @ 0x4002380C = 0x00000000`

- 인터럽트 플래그/인에이블 — 모두 0 (클럭 관련 인터럽트 미사용, 대기 중인 플래그 없음)


### RCC_PLLSAICFGR / PLLI2SCFGR / DCKCFGR — 파생 클럭

`RCC_PLLSAICFGR @ 0x40023888 = 0x54003000`

- RCC_CR.PLLSAION[28] = 1 → PLLSAI **동작 중** — 아래 계수가 실제 클럭을 만든다
- PLLSAIN[14:6] = 192 → PLLSAI VCO = 1.000 × 192 = **192.0 MHz** (VCO 입력은 메인 PLL 의 PLLM 을 공유)
- PLLSAIP[17:16] = 0b00 → 분주 2
- PLLSAIQ[27:24] = 4 → SAI 클럭 48.000 MHz
- PLLSAIR[30:28] = 5 → PLLSAIR = 38.400 MHz

`RCC_DCKCFGR1 @ 0x4002388C = 0x00010000`

- PLLI2SDIVQ[4:0] = 0 (+1 분주)
- PLLSAIDIVQ[12:8] = 0 (+1 분주)
- PLLSAIDIVR[17:16] = 0b01 → LCD-TFT 클럭 분주 /4 → **LCD_CLK = 9.600 MHz**
- SAI1SEL[21:20]=0 SAI2SEL[23:22]=0 TIMPRE[24]=0

`RCC_PLLI2SCFGR @ 0x40023884 = 0x24003000`

- PLLI2SN[14:6]=192 PLLI2SP[17:16]=0 PLLI2SQ[27:24]=4 PLLI2SR[30:28]=2  (PLLI2S 는 RCC_CR.PLLI2SON=0 이므로 미사용)

`RCC_DCKCFGR2 @ 0x40023890 = 0x00000000`

- USART1SEL=PCLK USART2SEL=PCLK USART3SEL=PCLK UART4SEL=PCLK UART5SEL=PCLK USART6SEL=PCLK
- I2C1SEL[17:16]=0 I2C2SEL[19:18]=0 I2C3SEL[21:20]=0 I2C4SEL[23:22]=0
- LPTIM1SEL[25:24]=0 CECSEL[26]=0 CK48MSEL[27]=0 → 48MHz 소스 = PLLQ, SDMMCSEL[28]=0


### 페리페럴 클럭 인에이블 (켜져 있는 것만)

| 레지스터 | 주소 | 값 | 클럭이 켜진 페리페럴 |
|---|---|---|---|
| RCC_AHB1ENR | 0x40023830 | 0x00D007FC | GPIOC, GPIOD, GPIOE, GPIOF, GPIOG, GPIOH, GPIOI, GPIOJ, GPIOK, DTCMRAM, DMA2, DMA2D |
| RCC_AHB2ENR | 0x40023834 | 0x00000000 | (없음) |
| RCC_AHB3ENR | 0x40023838 | 0x00000001 | FMC |
| RCC_APB1ENR | 0x40023840 | 0x10000010 | TIM6, PWR |
| RCC_APB2ENR | 0x40023844 | 0x04004800 | SDMMC1, SYSCFG, LTDC |

| 레지스터 | 주소 | 값 | 해석 |
|---|---|---|---|
| RCC_AHB1LPENR | 0x40023850 | 0x7EF7B7FF | Sleep 모드에서 클럭 유지 마스크 (리셋 기본값은 대부분 1) |
| RCC_AHB2LPENR | 0x40023854 | 0x000000F1 | Sleep 모드에서 클럭 유지 마스크 (리셋 기본값은 대부분 1) |
| RCC_AHB3LPENR | 0x40023858 | 0x00000003 | Sleep 모드에서 클럭 유지 마스크 (리셋 기본값은 대부분 1) |
| RCC_APB1LPENR | 0x40023860 | 0xFFFFCBFF | Sleep 모드에서 클럭 유지 마스크 (리셋 기본값은 대부분 1) |
| RCC_APB2LPENR | 0x40023864 | 0x04F77F33 | Sleep 모드에서 클럭 유지 마스크 (리셋 기본값은 대부분 1) |

`RCC_BDCR @ 0x40023870 = 0x00000000`

- LSEON[0]=0 LSERDY[1]=0 LSEBYP[2]=0 LSEDRV[4:3]=0 → 외부 32.768kHz OFF
- RTCSEL[9:8] = 0b00 → RTC 클럭 소스 = 없음(RTC 클럭 미선택)
- RTCEN[15] = 0 → RTC OFF
- BDRST[16] = 0

`RCC_CSR @ 0x40023874 = 0x0E000003`

- LSION[0] = 1 LSIRDY[1] = 1 → 내부 32kHz LSI **ON**
- RMVF[24] = 0
- **리셋 원인 플래그: PORRSTF(POR/PDR), PINRSTF(NRST 핀), BORRSTF(브라운아웃)**


## FLASH 인터페이스

`FLASH_ACR @ 0x40023C00 = 0x00000006`

- LATENCY[3:0] = 6 → **Flash 대기 상태 6 WS** (HCLK 200 MHz, VDD 2.7~3.6V·VOS scale1+over-drive 기준 RM0385 표에서 6 WS 는 180~210 MHz 구간)
- PRFTEN[8] = 0 → 프리페치 버퍼 **비활성**
- ARTEN[9] = 0 → **ART Accelerator(ITCM 경로 명령 캐시) 비활성**
- ARTRST[11] = 0 → ART 리셋 해제됨

`FLASH_SR @ 0x40023C0C = 0x00000000`

- BSY[16] = 0 → Flash 동작 중 여부
- 세워진 플래그: 없음 (에러 없음)

`FLASH_CR @ 0x40023C10 = 0x80000000`

- PG[0]=0 SER[1]=0 MER[2]=0
- SNB[7:3]=0 PSIZE[9:8]=0b00
- STRT[16]=0, EOPIE[24]=0, ERRIE[25]=0
- LOCK[31] = 1 → Flash 제어 레지스터 **잠김**(쓰기/소거 불가)

`FLASH_OPTCR @ 0x40023C14 = 0xC0FFAAFD`

- OPTLOCK[0] = 1 → 옵션 바이트 레지스터 **잠김**
- OPTSTRT[1] = 0
- BOR_LEV[3:2] = 0b11 → **BOR off (VBOR0, ~1.8V 리셋 임계)**
- WWDG_SW[4] = 1 → 윈도우 워치독 **소프트웨어 제어**
- IWDG_SW[5] = 1 → 독립 워치독 **소프트웨어 제어**
- nRST_STOP[6] = 1 → STOP 진입 시 리셋 없음
- nRST_STDBY[7] = 1 → STANDBY 진입 시 리셋 없음
- RDP[15:8] = 0xAA → **Level 0 — 보호 없음(디버거로 Flash 전체 읽기 가능)**
- nWRP[27:16] = 0x0FF → 쓰기 보호된 섹터: 섹터8, 섹터9, 섹터10, 섹터11  (1MB 단일 뱅크 F746 은 섹터 0~7 만 존재 — 섹터 8~11 비트는 무의미)
- IWDG_STDBY[30] = 1, IWDG_STOP[31] = 1 → 저전력 모드에서 IWDG 동작 여부

`FLASH_OPTCR1 @ 0x40023C18 = 0x00400080`

- BOOT_ADD0[15:0] = 0x0080 → BOOT0=0 일 때 부트 주소 = 0x0080 × 0x4000 = **0x00200000** (ITCM 인터페이스의 내장 Flash 별칭)
- BOOT_ADD1[31:16] = 0x0040 → BOOT0=1 일 때 부트 주소 = 0x0040 × 0x4000 = **0x00100000** (ITCM 인터페이스의 시스템 메모리(부트로더) 별칭)


### 옵션 바이트 원시 덤프 (0x1FFF0000)

```
1FFF0000  FF AA 00 55 FF AA 00 55 FF FF 00 00 FF FF 00 00  |...U...U........|
1FFF0010  80 00 7F FF 80 00 7F FF 40 00 BF FF 40 00 BF FF  |........@...@...|
```

> 옵션 바이트 플래시 셀의 원시 이미지다. 필드 의미는 위의 FLASH_OPTCR/OPTCR1 해석이 정본이며, 여기서는 원문 보존 목적으로만 싣는다.


## PWR (전원/전압 스케일링)

`PWR_CR1 @ 0x40007000 = 0x0003C100`

- LPDS[0] = 0 → STOP 모드에서 레귤레이터 정상
- PDDS[1] = 0, CSBF[3] = 0
- PVDE[4] = 0 → 전원 전압 감지기 해제됨
- PLS[7:5] = 0b000 → PVD 임계 레벨 0 (2.0V 근방; PVDE=0 이면 무의미)
- DBP[8] = 1 → 백업 도메인 쓰기 보호 **해제됨**(RTC/백업 레지스터 쓰기 가능)
- FPDS[9] = 0, LPUDS[10] = 0, MRUDS[11] = 0
- VOS[15:14] = 0b11 → **전압 스케일 **Scale 1** (최대 216MHz, over-drive 시)**
- ODEN[16] = 1 → **Over-drive 활성**
- ODSWEN[17] = 1 → **Over-drive 전환 완료(적용 중)**
- UDEN[19:18] = 0b00 → Under-drive

`PWR_CSR1 @ 0x40007004 = 0x00034000`

- WUIF[0] = 0, SBF[1] = 0, PVDO[2] = 0, BRR[3] = 0
- BRE[9] = 0 → 백업 레귤레이터 해제됨
- VOSRDY[14] = 1 → **전압 스케일 전환 완료**
- ODRDY[16] = 1 → **Over-drive ready**
- ODSWRDY[17] = 1 → **Over-drive 전환 ready(200MHz 동작 가능 상태)**
- UDRDY[19:18] = 0b00


## SYSCFG

`SYSCFG_MEMRMP @ 0x40013800 = 0x00000000`

- MEM_BOOT[2:0] = 0b000 → **메인 Flash** 가 0x00000000 에 매핑
- SWP_FMC[11:10] = 0b00 → FMC 뱅크 스와핑 없음(기본: SDRAM=0xC0000000/0xD0000000)

`SYSCFG_PMC @ 0x40013804 = 0x00000000`

- MII_RMII_SEL[23] = 0 → 이더넷 PHY 인터페이스 = MII
- ADCxDC2[18:16] = 0b000

| 레지스터 | 주소 | 값 | EXTI 소스 포트 |
|---|---|---|---|
| SYSCFG_EXTICR1 | 0x40013808 | 0x00000000 | EXTI0←PA, EXTI1←PA, EXTI2←PA, EXTI3←PA |
| SYSCFG_EXTICR2 | 0x4001380C | 0x00000000 | EXTI4←PA, EXTI5←PA, EXTI6←PA, EXTI7←PA |
| SYSCFG_EXTICR3 | 0x40013810 | 0x00000000 | EXTI8←PA, EXTI9←PA, EXTI10←PA, EXTI11←PA |
| SYSCFG_EXTICR4 | 0x40013814 | 0x00000000 | EXTI12←PA, EXTI13←PA, EXTI14←PA, EXTI15←PA |

> EXTICR 이 전부 0 → 모든 EXTI 라인이 기본값(포트 A)을 가리킨다. 즉 실행 중이던 펌웨어는 GPIO 외부 인터럽트를 하나도 재매핑하지 않았다.


## FMC — SDRAM 컨트롤러 (베이스 0xA0000000)

`FMC_SDCR1 @ 0xA0000140 = 0x00001954`

- NC[1:0] = 0 → 컬럼 주소 **8 비트**
- NR[3:2] = 1 → 로우 주소 **12 비트** (= 4096 행)
- MWID[5:4] = 1 → 메모리 데이터 버스 폭 **16 비트**
- NB[6] = 1 → 내부 뱅크 **4개**
- CAS[8:7] = 2 → **CAS latency 2 사이클**
- WP[9] = 0 → 쓰기 보호 없음
- SDCLK[11:10] = 0b10 → SDRAM 클럭 = HCLK/2 = **100.0 MHz**
- RBURST[12] = 1 → 읽기 버스트 활성
- RPIPE[14:13] = 0 → 읽기 데이터 파이프라인 지연 0 HCLK
- → **주소 지정 가능 용량 = 2^12 행 × 2^8 열 × 4 뱅크 × 2 B = 8388608 B = 8 MB**

`FMC_SDCR2 @ 0xA0000144 = 0x000002D0`

- NC[1:0] = 0 → 컬럼 주소 **8 비트**
- NR[3:2] = 0 → 로우 주소 **11 비트** (= 2048 행)
- MWID[5:4] = 1 → 메모리 데이터 버스 폭 **16 비트**
- NB[6] = 1 → 내부 뱅크 **4개**
- CAS[8:7] = 1 → **CAS latency 1 사이클**
- WP[9] = 1 → 쓰기 보호 **적용**(읽기 전용)
- SDCLK[11:10] = 0b00 → SDRAM 클럭 = **비활성(클럭 없음)**
- RBURST[12] = 0 → 읽기 버스트 비활성
- RPIPE[14:13] = 0 → 읽기 데이터 파이프라인 지연 0 HCLK
- 주의: SDCLK/RBURST/RPIPE 필드는 SDCR1 의 값만 유효하다 (RM0385) → SDCR2 의 해당 필드는 무시된다.

`FMC_SDTR1 @ 0xA0000148 = 0x01116361`

- TMRD[3:0] = 1 → Load Mode Register → Active 지연 **2 사이클** = 20.0 ns
- TXSR[7:4] = 6 → Self-refresh 탈출 지연 **7 사이클** = 70.0 ns
- TRAS[11:8] = 3 → Self-refresh(최소 Active) 시간 **4 사이클** = 40.0 ns
- TRC[15:12] = 6 → Row cycle 지연 **7 사이클** = 70.0 ns
- TWR[19:16] = 1 → Write recovery 시간 **2 사이클** = 20.0 ns
- TRP[23:20] = 1 → Row precharge 지연 **2 사이클** = 20.0 ns
- TRCD[27:24] = 1 → Row→Column 지연 **2 사이클** = 20.0 ns
- (TRC/TRP 는 SDTR1 의 값이 두 뱅크에 공통 적용된다 — RM0385)

`FMC_SDTR2 @ 0xA000014C = 0x0FFFFFFF`

- TMRD[3:0] = 15 → Load Mode Register → Active 지연 **16 사이클** = 160.0 ns
- TXSR[7:4] = 15 → Self-refresh 탈출 지연 **16 사이클** = 160.0 ns
- TRAS[11:8] = 15 → Self-refresh(최소 Active) 시간 **16 사이클** = 160.0 ns
- TRC[15:12] = 15 → Row cycle 지연 **16 사이클** = 160.0 ns
- TWR[19:16] = 15 → Write recovery 시간 **16 사이클** = 160.0 ns
- TRP[23:20] = 15 → Row precharge 지연 **16 사이클** = 160.0 ns
- TRCD[27:24] = 15 → Row→Column 지연 **16 사이클** = 160.0 ns
- (SDTR2 는 뱅크2 용. 전 필드가 최대값이면 뱅크2 미사용을 뜻한다. 단 RM0385 에 따르면 TRC/TRP 는 SDTR1 의 값만 유효하며 두 뱅크에 공통 적용된다.)

`FMC_SDCMR @ 0xA0000150 = 0x00044000`

- MODE[2:0] = 0 → **Normal mode** (초기화 시퀀스 완료)
- CTB2[3] = 0, CTB1[4] = 0 → 명령 대상 뱅크
- NRFS[8:5] = 0 → auto-refresh 연속 횟수 1
- MRD[21:9] = 0x0220 → 마지막으로 SDRAM 모드 레지스터에 쓴 값: Burst Length=1, Burst Type=sequential, CAS Latency=2, Write Burst Mode=single location

`FMC_SDRTR @ 0xA0000154 = 0x00000C06`

- CRE[0] = 0 → refresh error 플래그 클리어
- COUNT[13:1] = 1539 → 리프레시 타이머 카운트
- REIE[14] = 0 → refresh error 인터럽트 해제됨
- → 행 하나당 리프레시 주기 = (COUNT+20)/f_SDRAM = (1539+20)/100MHz = **15.59 µs**
- → 전체 배열 리프레시 주기 = 15.59 µs × 4096 행 = **63.9 ms** (SDRAM 표준 64 ms 사양 대비)

`FMC_SDSR @ 0xA0000158 = 0x00000020`

- RE[0] = 0 → refresh error 플래그
- MODES1[2:1] = 0 → 뱅크1 상태 = Normal
- MODES2[4:3] = 0 → 뱅크2 상태 = Normal
- BUSY[5] = 1 → 컨트롤러가 명령 처리 중 **예**


## QUADSPI (베이스 0xA0001000)

QUADSPI 레지스터가 전부 0x00000000 이다.

> ⚠️ **경고**: QUADSPI 블록의 클럭(RCC_AHB3ENR.QSPIEN)이 꺼져 있으면 레지스터가 0 으로 읽힌다. 따라서 이 덤프로는 **QSPI 클럭 주파수와 플래시 용량을 계산할 수 없다(해석 불가)**. QSPI 플래시의 존재 자체는 이 덤프로 증명되지도 반증되지도 않는다.


## LTDC — LCD-TFT 컨트롤러 (베이스 0x40016800)

`LTDC_SSCR @ 0x40016808 = 0x00280009`

- HSW[27:16] = 40 → 수평 동기 폭 = **41 픽셀클럭**
- VSH[10:0] = 9 → 수직 동기 높이 = **10 라인**

`LTDC_BPCR @ 0x4001680C = 0x0035000B`

- AHBP[27:16] = 53 → 누적 수평 백포치 54 → HBP = 54−41 = **13**
- AVBP[10:0] = 11 → 누적 수직 백포치 12 → VBP = 12−10 = **2**

`LTDC_AWCR @ 0x40016810 = 0x0215011B`

- AAW[27:16] = 533 → 누적 활성 폭 534
- AAH[10:0] = 283 → 누적 활성 높이 284
- → **실제 활성 해상도 = (AAW−AHBP) × (AAH−AVBP) = 480 × 272**

`LTDC_TWCR @ 0x40016814 = 0x0235011D`

- TOTALW[27:16] = 565 → 전체 수평 566 → HFP = 566−534 = **32**
- TOTALH[10:0] = 285 → 전체 수직 286 → VFP = 286−284 = **2**

`LTDC_GCR @ 0x40016818 = 0x00002221`

- LTDCEN[0] = 1 → LTDC **동작 중**
- DBW[6:4]=2 DGW[10:8]=2 DRW[14:12]=2 → 디더링 비트 폭(읽기 전용, 구현 고정값)
- DEN[16] = 0 → 디더링 해제됨
- PCPOL[28]=0 DEPOL[29]=0 VSPOL[30]=0 HSPOL[31]=0 → 신호 극성 (0=기본 active-low/입력 그대로)


### 파생: 활성 해상도와 프레임레이트

- 프레임당 총 픽셀클럭 수 = TOTALW × TOTALH = 566 × 286 = **161,876**
- 픽셀 클럭 = 9.600 MHz (RCC PLLSAI 에서 산출)
- → **추정 프레임레이트 = 9.600MHz / 161,876 = 59.30 Hz**

| 항목 | 값 |
|---|---|
| 활성 해상도 | 480 × 272 |
| HSW / HBP / HFP | 41 / 13 / 32 |
| VSH / VBP / VFP | 10 / 2 / 2 |
| 전체 타이밍 | 566 × 286 |

`LTDC_L1CR @ 0x40016884 = 0x00000001`

- LEN[0] = 1 → 레이어 1 **활성**
- L1PFCR = 0x00000000 → 픽셀 포맷 = ARGB8888
- L1CFBAR (프레임버퍼 주소) = 0xC0000000
- L1CFBLR = 0x07800783 → 라인 길이 1923 B, 피치 1920 B
- L1CFBLNR = 0x00000110 → 프레임버퍼 라인 수 272

`LTDC_L2CR @ 0x40016904 = 0x00000001`

- LEN[0] = 1 → 레이어 2 **활성**
- L2PFCR = 0x00000000 → 픽셀 포맷 = ARGB8888
- L2CFBAR (프레임버퍼 주소) = 0xC007F800
- L2CFBLR = 0x07800783 → 라인 길이 1923 B, 피치 1920 B
- L2CFBLNR = 0x00000110 → 프레임버퍼 라인 수 272


## GPIO — 핀별 설정 (보드 배선 실측 근거)

> ⚠️ **경고**: 다음 포트는 0x400 블록 **전체가 0 이 아닌 하나의 32비트 값으로 반복**되어 있다 — 정상적인 레지스터 이미지가 아니다(덤프 아티팩트). GPIOA(0x40020000, 반복값 0xA8000000, RCC 클럭 OFF), GPIOB(0x40020400, 반복값 0x00000280, RCC 클럭 OFF). 반복된 값이 각 포트 MODER 의 리셋 기본값과 일치하고 해당 포트의 AHB1ENR 클럭이 꺼져 있으므로, **이 포트들의 값은 신뢰할 수 없어 아래 핀 상태표와 AF 목록에서 제외한다(해석 불가).**


### GPIOC @ 0x40020800  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0x02AA0080 |
| OTYPER | 0x00000000 |
| OSPEEDR | 0x03FF00C0 |
| PUPDR | 0x04000000 |
| IDR | 0x00001F0B |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0x0000C000 |
| AFRH | 0x000CCCCC |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PC0 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PC1 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PC2 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PC3 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PC4 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PC5 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PC6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PC7 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PC8 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PC9 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PC10 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PC11 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PC12 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PC13 | 입력 | push-pull | Low | Pull-up | - | 0 | 0 |
| PC14 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PC15 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |


### GPIOD @ 0x40020C00  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0xA02A002A |
| OTYPER | 0x00000000 |
| OSPEEDR | 0xF03F003F |
| PUPDR | 0x00000000 |
| IDR | 0x0000C737 |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0x00000CCC |
| AFRH | 0xCC000CCC |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PD0 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD1 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD2 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD3 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PD4 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PD5 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PD6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PD7 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PD8 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD9 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD10 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD11 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PD12 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PD13 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PD14 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PD15 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |


### GPIOE @ 0x40021000  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0xAAAA820A |
| OTYPER | 0x00000000 |
| OSPEEDR | 0xFFFFC00F |
| PUPDR | 0x00000000 |
| IDR | 0x0000F818 |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0xC00E00CC |
| AFRH | 0xCCCCCCCC |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PE0 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PE1 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PE2 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PE3 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PE4 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PE5 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PE6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PE7 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PE8 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PE9 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PE10 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PE11 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PE12 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PE13 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PE14 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PE15 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |


### GPIOF @ 0x40021400  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0xAA800AAA |
| OTYPER | 0x00000000 |
| OSPEEDR | 0xFFC00FFF |
| PUPDR | 0x00000000 |
| IDR | 0x00002822 |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0x00CCCCCC |
| AFRH | 0xCCCCC000 |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PF0 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PF1 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PF2 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PF3 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PF4 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PF5 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PF6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PF7 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PF8 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PF9 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PF10 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PF11 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PF12 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PF13 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PF14 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PF15 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |


### GPIOG @ 0x40021800  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0x82020A0A |
| OTYPER | 0x00000000 |
| OSPEEDR | 0xC0030F0F |
| PUPDR | 0x00000000 |
| IDR | 0x0000F100 |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0x00CC00CC |
| AFRH | 0xC009000C |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PG0 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PG1 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PG2 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG3 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG4 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PG5 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PG6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG7 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG8 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PG9 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG10 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG11 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PG12 | 대체기능(AF) | push-pull | Low | 없음 | AF9 (CAN1/CAN2/TIM12/TIM13/TIM14/QUADSPI/LTDC) | 1 | 0 |
| PG13 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PG14 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PG15 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |


### GPIOH @ 0x40021C00  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0x00000880 |
| OTYPER | 0x00000000 |
| OSPEEDR | 0x00000CC0 |
| PUPDR | 0x00000000 |
| IDR | 0x000001A0 |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0x00C0C000 |
| AFRH | 0x00000000 |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PH0 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH1 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH2 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH3 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 0 | 0 |
| PH4 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH5 | 대체기능(AF) | push-pull | Very high | 없음 | AF12 (FMC/SDMMC1/OTG_FS) | 1 | 0 |
| PH6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH7 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PH8 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PH9 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH10 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH11 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH12 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH13 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH14 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PH15 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |


### GPIOI @ 0x40022000  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0xA1280000 |
| OTYPER | 0x00000000 |
| OSPEEDR | 0x02000000 |
| PUPDR | 0x00000000 |
| IDR | 0x0000F600 |
| ODR | 0x00001000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0x00000000 |
| AFRH | 0xEE000EE0 |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PI0 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI1 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI2 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI3 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI4 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI5 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI6 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI7 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI8 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI9 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PI10 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PI11 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PI12 | 출력 | push-pull | High | 없음 | - | 1 | 1 |
| PI13 | 입력 | push-pull | Low | 없음 | - | 1 | 0 |
| PI14 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PI15 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |


### GPIOJ @ 0x40022400  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0xA8AAAAAA |
| OTYPER | 0x00000000 |
| OSPEEDR | 0x00000000 |
| PUPDR | 0x00000000 |
| IDR | 0x00000000 |
| ODR | 0x00000000 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0xEEEEEEEE |
| AFRH | 0xEEE0EEEE |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PJ0 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ1 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ2 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ3 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ4 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ5 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ6 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ7 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ8 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ9 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ10 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ11 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ12 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PJ13 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ14 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |
| PJ15 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 0 | 0 |


### GPIOK @ 0x40022800  (RCC 클럭 ON)

| 레지스터 | 값 |
|---|---|
| MODER | 0x0000AA6A |
| OTYPER | 0x00000000 |
| OSPEEDR | 0x00000080 |
| PUPDR | 0x00000000 |
| IDR | 0x000000FF |
| ODR | 0x00000008 |
| BSRR | 0x00000000 |
| LCKR | 0x00000000 |
| AFRL | 0xEEEE0EEE |
| AFRH | 0x00000000 |

| 핀 | 모드 | 출력 타입 | 속도 | 풀업/다운 | 대체기능 | IDR | ODR |
|---|---|---|---|---|---|---|---|
| PK0 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK1 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK2 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK3 | 출력 | push-pull | High | 없음 | - | 1 | 1 |
| PK4 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK5 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK6 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK7 | 대체기능(AF) | push-pull | Low | 없음 | AF14 (LTDC) | 1 | 0 |
| PK8 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK9 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK10 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK11 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK12 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK13 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK14 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |
| PK15 | 입력 | push-pull | Low | 없음 | - | 0 | 0 |


### 대체기능(AF)으로 설정된 핀 전체 목록

보드 배선의 실측 증거다. AF 번호 → 기능 그룹 매핑은 STM32F746 데이터시트의 alternate function 표에서 옮긴 개괄이며, 같은 AF 번호라도 핀마다 실제 신호는 다르다.

| 핀 | AF | AF 그룹(개괄) |
|---|---|---|
| PC3 | AF12 | FMC/SDMMC1/OTG_FS |
| PC8 | AF12 | FMC/SDMMC1/OTG_FS |
| PC9 | AF12 | FMC/SDMMC1/OTG_FS |
| PC10 | AF12 | FMC/SDMMC1/OTG_FS |
| PC11 | AF12 | FMC/SDMMC1/OTG_FS |
| PC12 | AF12 | FMC/SDMMC1/OTG_FS |
| PD0 | AF12 | FMC/SDMMC1/OTG_FS |
| PD1 | AF12 | FMC/SDMMC1/OTG_FS |
| PD2 | AF12 | FMC/SDMMC1/OTG_FS |
| PD8 | AF12 | FMC/SDMMC1/OTG_FS |
| PD9 | AF12 | FMC/SDMMC1/OTG_FS |
| PD10 | AF12 | FMC/SDMMC1/OTG_FS |
| PD14 | AF12 | FMC/SDMMC1/OTG_FS |
| PD15 | AF12 | FMC/SDMMC1/OTG_FS |
| PE0 | AF12 | FMC/SDMMC1/OTG_FS |
| PE1 | AF12 | FMC/SDMMC1/OTG_FS |
| PE4 | AF14 | LTDC |
| PE7 | AF12 | FMC/SDMMC1/OTG_FS |
| PE8 | AF12 | FMC/SDMMC1/OTG_FS |
| PE9 | AF12 | FMC/SDMMC1/OTG_FS |
| PE10 | AF12 | FMC/SDMMC1/OTG_FS |
| PE11 | AF12 | FMC/SDMMC1/OTG_FS |
| PE12 | AF12 | FMC/SDMMC1/OTG_FS |
| PE13 | AF12 | FMC/SDMMC1/OTG_FS |
| PE14 | AF12 | FMC/SDMMC1/OTG_FS |
| PE15 | AF12 | FMC/SDMMC1/OTG_FS |
| PF0 | AF12 | FMC/SDMMC1/OTG_FS |
| PF1 | AF12 | FMC/SDMMC1/OTG_FS |
| PF2 | AF12 | FMC/SDMMC1/OTG_FS |
| PF3 | AF12 | FMC/SDMMC1/OTG_FS |
| PF4 | AF12 | FMC/SDMMC1/OTG_FS |
| PF5 | AF12 | FMC/SDMMC1/OTG_FS |
| PF11 | AF12 | FMC/SDMMC1/OTG_FS |
| PF12 | AF12 | FMC/SDMMC1/OTG_FS |
| PF13 | AF12 | FMC/SDMMC1/OTG_FS |
| PF14 | AF12 | FMC/SDMMC1/OTG_FS |
| PF15 | AF12 | FMC/SDMMC1/OTG_FS |
| PG0 | AF12 | FMC/SDMMC1/OTG_FS |
| PG1 | AF12 | FMC/SDMMC1/OTG_FS |
| PG4 | AF12 | FMC/SDMMC1/OTG_FS |
| PG5 | AF12 | FMC/SDMMC1/OTG_FS |
| PG8 | AF12 | FMC/SDMMC1/OTG_FS |
| PG12 | AF9 | CAN1/CAN2/TIM12/TIM13/TIM14/QUADSPI/LTDC |
| PG15 | AF12 | FMC/SDMMC1/OTG_FS |
| PH3 | AF12 | FMC/SDMMC1/OTG_FS |
| PH5 | AF12 | FMC/SDMMC1/OTG_FS |
| PI9 | AF14 | LTDC |
| PI10 | AF14 | LTDC |
| PI14 | AF14 | LTDC |
| PI15 | AF14 | LTDC |
| PJ0 | AF14 | LTDC |
| PJ1 | AF14 | LTDC |
| PJ2 | AF14 | LTDC |
| PJ3 | AF14 | LTDC |
| PJ4 | AF14 | LTDC |
| PJ5 | AF14 | LTDC |
| PJ6 | AF14 | LTDC |
| PJ7 | AF14 | LTDC |
| PJ8 | AF14 | LTDC |
| PJ9 | AF14 | LTDC |
| PJ10 | AF14 | LTDC |
| PJ11 | AF14 | LTDC |
| PJ13 | AF14 | LTDC |
| PJ14 | AF14 | LTDC |
| PJ15 | AF14 | LTDC |
| PK0 | AF14 | LTDC |
| PK1 | AF14 | LTDC |
| PK2 | AF14 | LTDC |
| PK4 | AF14 | LTDC |
| PK5 | AF14 | LTDC |
| PK6 | AF14 | LTDC |
| PK7 | AF14 | LTDC |


#### AF 번호별 묶음

| AF | 그룹(개괄) | 핀 수 | 핀 목록 |
|---|---|---|---|
| AF9 | CAN1/CAN2/TIM12/TIM13/TIM14/QUADSPI/LTDC | 1 | PG12 |
| AF12 | FMC/SDMMC1/OTG_FS | 44 | PC3, PC8, PC9, PC10, PC11, PC12, PD0, PD1, PD2, PD8, PD9, PD10, PD14, PD15, PE0, PE1, PE7, PE8, PE9, PE10, PE11, PE12, PE13, PE14, PE15, PF0, PF1, PF2, PF3, PF4, PF5, PF11, PF12, PF13, PF14, PF15, PG0, PG1, PG4, PG5, PG8, PG15, PH3, PH5 |
| AF14 | LTDC | 27 | PE4, PI9, PI10, PI14, PI15, PJ0, PJ1, PJ2, PJ3, PJ4, PJ5, PJ6, PJ7, PJ8, PJ9, PJ10, PJ11, PJ13, PJ14, PJ15, PK0, PK1, PK2, PK4, PK5, PK6, PK7 |


## 메모리 영역 샘플 (각 영역 선두 64바이트)


### ITCM RAM 선두 — 0x00000000

```
00000000  0D 31 CD C0 92 A7 68 7C 34 1D 8D 44 C2 09 57 A7  |.1....h|4..D..W.|
00000010  CC B7 AE 24 F4 9F DB D0 CC 98 BF B9 13 33 74 19  |...$.........3t.|
00000020  F4 08 61 41 23 99 39 16 AC 80 98 04 70 C0 33 00  |..aA#.9.....p.3.|
00000030  E9 46 C8 7C 7B 2A 53 14 06 E7 74 CD 7C 83 B6 06  |.F.|{*S...t.|...|
```


### ITCM 인터페이스의 내장 Flash 별칭 — 0x00200000

```
00200000  00 00 05 20 4D 03 02 08 ED 59 00 08 F1 59 00 08  |... M....Y...Y..|
00200010  F5 59 00 08 F9 59 00 08 FD 59 00 08 00 00 00 00  |.Y...Y...Y......|
00200020  00 00 00 00 00 00 00 00 00 00 00 00 9D 03 02 08  |................|
00200030  01 5A 00 08 00 00 00 00 9D 03 02 08 9D 03 02 08  |.Z..............|
```


### AXIM 인터페이스의 내장 Flash = 벡터 테이블 — 0x08000000

```
08000000  00 00 05 20 4D 03 02 08 ED 59 00 08 F1 59 00 08  |... M....Y...Y..|
08000010  F5 59 00 08 F9 59 00 08 FD 59 00 08 00 00 00 00  |.Y...Y...Y......|
08000020  00 00 00 00 00 00 00 00 00 00 00 00 9D 03 02 08  |................|
08000030  01 5A 00 08 00 00 00 00 9D 03 02 08 9D 03 02 08  |.Z..............|
```


### DTCM RAM 선두 — 0x20000000

```
20000000  00 00 00 00 00 00 00 00 00 C2 EB 0B D8 3C 02 08  |.............<..|
20000010  11 00 18 00 00 00 00 00 00 00 00 00 90 57 02 08  |.............W..|
20000020  0B 00 10 00 09 8A 03 2F 00 00 00 00 03 00 00 00  |......./........|
20000030  1C 10 04 20 38 00 00 20 00 00 00 00 1C 10 04 20  |... 8.. ....... |
```


### SRAM1 선두 — 0x20010000

```
20010000  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
20010010  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
20010020  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
20010030  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
```

→ 전 바이트 0x00 (초기화되지 않았거나 지워진 상태).


### SRAM2 선두 — 0x2004C000

```
2004C000  EF E0 41 04 96 67 5E 1A E7 24 C3 F8 7E 80 4D 86  |..A..g^..$..~.M.|
2004C010  B8 18 B2 1E FA 82 F7 C1 6F 5E 21 71 DD 2C FE E3  |........o^!q.,..|
2004C020  6C 3F AF F5 F5 8A 61 8C 99 AA 79 0B 59 F1 51 03  |l?....a...y.Y.Q.|
2004C030  AB 2A FD FA 0B B3 69 3F 45 3F C0 67 0A 10 80 2D  |.*....i?E?.g...-|
```


### 백업 SRAM — 0x40024000

```
40024000  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
40024010  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
40024020  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
40024030  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
```

→ 전 바이트 0x00 (초기화되지 않았거나 지워진 상태).


### 외부 SDRAM(FMC Bank5/6) 선두 — 0xC0000000

```
C0000000  00 00 00 FF 00 00 00 FF 00 00 00 FF 00 00 00 FF  |................|
C0000010  00 00 00 FF 00 00 00 FF 00 00 00 FF 00 00 00 FF  |................|
C0000020  00 00 00 FF 00 00 00 FF 00 00 00 FF 00 00 00 FF  |................|
C0000030  00 00 00 FF 00 00 00 FF 00 00 00 FF 00 00 00 FF  |................|
```


### QSPI 메모리맵 모드 선두 — 0x90000000

```
90000000  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
90000010  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
90000020  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
90000030  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  |................|
```

→ 전 바이트 0x00 (초기화되지 않았거나 지워진 상태).


### 벡터 테이블 해석 (0x08000000)

| # | 엔트리 | 주소 | 값 | 해석 |
|---|---|---|---|---|
| 0 | 초기 SP (MSP) | 0x08000000 | 0x20050000 | 초기 스택 포인터 → SRAM2 최상단 경계(= 이 영역 끝 다음 주소) (스택은 이 주소에서 아래로 자란다) |
| 1 | Reset_Handler | 0x08000004 | 0x0802034D | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x0802034C, Flash AXIM |
| 2 | NMI_Handler | 0x08000008 | 0x080059ED | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x080059EC, Flash AXIM |
| 3 | HardFault_Handler | 0x0800000C | 0x080059F1 | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x080059F0, Flash AXIM |
| 4 | MemManage_Handler | 0x08000010 | 0x080059F5 | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x080059F4, Flash AXIM |
| 5 | BusFault_Handler | 0x08000014 | 0x080059F9 | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x080059F8, Flash AXIM |
| 6 | UsageFault_Handler | 0x08000018 | 0x080059FD | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x080059FC, Flash AXIM |
| 7 | (예약) | 0x0800001C | 0x00000000 |  |
| 8 | (예약) | 0x08000020 | 0x00000000 |  |
| 9 | (예약) | 0x08000024 | 0x00000000 |  |
| 10 | (예약) | 0x08000028 | 0x00000000 |  |
| 11 | SVC_Handler | 0x0800002C | 0x0802039D | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x0802039C, Flash AXIM |
| 12 | DebugMon_Handler | 0x08000030 | 0x08005A01 | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x08005A00, Flash AXIM |
| 13 | (예약) | 0x08000034 | 0x00000000 |  |
| 14 | PendSV_Handler | 0x08000038 | 0x0802039D | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x0802039C, Flash AXIM |
| 15 | SysTick_Handler | 0x0800003C | 0x0802039D | Thumb 함수 주소 (bit0=1) → 실제 명령 주소 0x0802039C, Flash AXIM |


### ITCM 별칭 검증

| 비교 | 결과 | 의미 |
|---|---|---|
| 0x00000000 (ITCM) vs 0x00200000 (Flash ITCM 별칭) | **불일치** | 0x00000000 은 Flash 별칭이 아니라 ITCM RAM 그 자체다 |
| 0x00200000 (Flash ITCM 별칭) vs 0x08000000 (Flash AXIM) | **일치** | 같은 Flash 를 ITCM/AXIM 두 경로로 본 것임이 확인된다 |
| 0x00000000 (ITCM) vs 0x08000000 (Flash AXIM) | **불일치** |  |


## 시스템 메모리 부트로더 (0x1FF00000, ST 내장)

덤프 크기: 32768 바이트 (0x1FF00000 ~ 0x1FF07FFF)


### 부트로더 벡터 테이블

| # | 엔트리 | 주소 | 값 | 해석 |
|---|---|---|---|---|
| 0 | 초기 SP (MSP) | 0x1FF00000 | 0x20003AF8 | 초기 SP → DTCM RAM |
| 1 | Reset_Handler | 0x1FF00004 | 0x1FF00C89 | Thumb 함수 → 0x1FF00C88, 시스템 부트로더 |
| 2 | NMI_Handler | 0x1FF00008 | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 3 | HardFault_Handler | 0x1FF0000C | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 4 | MemManage_Handler | 0x1FF00010 | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 5 | BusFault_Handler | 0x1FF00014 | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 6 | UsageFault_Handler | 0x1FF00018 | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 7 | (예약) | 0x1FF0001C | 0x00000000 |  |
| 8 | (예약) | 0x1FF00020 | 0x00000000 |  |
| 9 | (예약) | 0x1FF00024 | 0x00000000 |  |
| 10 | (예약) | 0x1FF00028 | 0x00000000 |  |
| 11 | SVC_Handler | 0x1FF0002C | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 12 | DebugMon_Handler | 0x1FF00030 | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 13 | (예약) | 0x1FF00034 | 0x00000000 |  |
| 14 | PendSV_Handler | 0x1FF00038 | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |
| 15 | SysTick_Handler | 0x1FF0003C | 0x1FF00211 | Thumb 함수 → 0x1FF00210, 시스템 부트로더 |


### 내장 ASCII 문자열 추출

| 주소 | 문자열 |
|---|---|
| 0x1FF0189E | `(IBFHF` |
| 0x1FF0315A | `sLsM h` |
| 0x1FF031D5 | `pixAph` |
| 0x1FF042D5 | `h7h1iD` |
| 0x1FF043AB | `@pG8H@h` |
| 0x1FF04475 | `H@hAhA` |
| 0x1FF04725 | `a(Fshg` |
| 0x1FF04738 | `qhHaph` |
| 0x1FF04A92 | `pGaHpG` |
| 0x1FF04BFD | `H@hAhA` |
| 0x1FF05A94 | `/IBFHF` |
| 0x1FF05B68 | `(xixH@` |


## 페리페럴 영역 덤프 성공/실패


### 읽기 성공 (50개)

| 영역 | 주소 | 크기 | 설명 |
|---|---|---|---|
| ppb_itm | 0xE0000000 | 4096 B | ITM (Instrumentation Trace Macrocell) |
| ppb_dwt | 0xE0001000 | 4096 B | DWT (Data Watchpoint & Trace) - 사이클 카운터 등 |
| ppb_fpb | 0xE0002000 | 4096 B | FPB (Flash Patch & Breakpoint) |
| ppb_scs | 0xE000E000 | 4096 B | SCS: SysTick/NVIC/SCB/MPU/FPU/캐시제어 전체 |
| ppb_tpiu | 0xE0040000 | 4096 B | TPIU (Trace Port Interface Unit) |
| ppb_etm | 0xE0041000 | 4096 B | ETM (Embedded Trace Macrocell) |
| ppb_dbgmcu | 0xE0042000 | 256 B | DBGMCU: IDCODE(디바이스/리비전 ID), 디버그 프리즈 |
| ppb_romtable | 0xE00FF000 | 4096 B | CoreSight ROM table: 탑재된 디버그 컴포넌트 목록 |
| factory_uid_area | 0x1FF0F000 | 4096 B | UID/Flash size/ADC 캘리브레이션이 있는 팩토리 영역 |
| factory_pkg_area | 0x1FF07800 | 2048 B | 패키지 데이터 레지스터 영역(0x1FF07BF0) |
| sysmem_bootloader | 0x1FF00000 | 32768 B | 시스템 메모리 부트로더 코드 (ST 내장) |
| option_bytes | 0x1FFF0000 | 32 B | 옵션 바이트 (RDP/BOR/WDG/BOOT 설정) |
| rcc | 0x40023800 | 256 B | RCC: 클럭 트리 현재 설정 + 페리페럴 클럭 인에이블 상태 |
| flash_if | 0x40023C00 | 256 B | FLASH 인터페이스: ACR(wait state/ART), OPTCR |
| pwr | 0x40007000 | 32 B | PWR: 전압 스케일링/over-drive 상태 |
| syscfg | 0x40013800 | 64 B | SYSCFG: 메모리 리맵, EXTI 소스 |
| crc | 0x40023000 | 32 B | CRC 계산 유닛 |
| gpio_all | 0x40020000 | 11264 B | GPIOA~GPIOK (포트당 0x400, 11포트) |
| dma1 | 0x40026000 | 256 B | DMA1 컨트롤러 |
| dma2 | 0x40026400 | 256 B | DMA2 컨트롤러 |
| dma2d | 0x4002B000 | 256 B | DMA2D (Chrom-ART 그래픽 가속기) |
| fmc | 0xA0000000 | 512 B | FMC: SDRAM/NOR/NAND 컨트롤러 (SDCR/SDTR 타이밍) |
| quadspi | 0xA0001000 | 256 B | QUADSPI 컨트롤러 |
| ltdc | 0x40016800 | 512 B | LTDC LCD-TFT 컨트롤러 (해상도/타이밍 현재 설정) |
| dcmi | 0x50050000 | 256 B | DCMI 카메라 인터페이스 |
| usart1 | 0x40011000 | 64 B | USART1 (ST-LINK VCP 후보) |
| usart6 | 0x40011400 | 64 B | USART6 |
| uart_bank_apb1 | 0x40004400 | 1024 B | USART2/3, UART4/5 (APB1 연속 영역) |
| i2c_bank | 0x40005400 | 3072 B | I2C1/2/3 (APB1) |
| spi_bank_apb1 | 0x40003800 | 3072 B | SPI2/3, SPDIFRX 영역 |
| adc_bank | 0x40012000 | 1024 B | ADC1/2/3 + 공통 레지스터 |
| sdmmc1 | 0x40012C00 | 256 B | SDMMC1 (microSD) |
| rng | 0x50060800 | 32 B | RNG 난수 발생기 |
| rtc | 0x40002800 | 256 B | RTC + 백업 레지스터 |
| iwdg | 0x40003000 | 32 B | 독립 워치독 |
| wwdg | 0x40002C00 | 32 B | 윈도우 워치독 |
| tim1 | 0x40010000 | 256 B | TIM1 (고급 제어 타이머) |
| tim2 | 0x40000000 | 256 B | TIM2 (32비트 범용 타이머) |
| eth | 0x40028000 | 1024 B | Ethernet MAC |
| usb_hs_core | 0x40040000 | 1024 B | USB OTG HS 코어 레지스터 |
| usb_fs_core | 0x50000000 | 1024 B | USB OTG FS 코어 레지스터 |
| mem_itcm_head | 0x00000000 | 64 B | ITCM 선두 (0x00000000) |
| mem_flash_itcm_alias | 0x00200000 | 64 B | Flash의 ITCM 별칭 (0x00200000) |
| mem_flash_axim | 0x08000000 | 64 B | Flash AXIM 별칭 (0x08000000) = 벡터 테이블 |
| mem_dtcm_head | 0x20000000 | 64 B | DTCM 선두 (0x20000000) |
| mem_sram1_head | 0x20010000 | 64 B | SRAM1 선두 (0x20010000) |
| mem_sram2_head | 0x2004C000 | 64 B | SRAM2 선두 (0x2004C000) |
| mem_bkpsram | 0x40024000 | 64 B | 백업 SRAM (0x40024000) |
| mem_sdram_head | 0xC0000000 | 64 B | 외부 SDRAM 선두 (FMC Bank5/6, 0xC0000000) |
| mem_qspi_head | 0x90000000 | 64 B | QSPI 메모리맵 모드 선두 (0x90000000) |


### 읽기 실패 (1개)

| 영역 | 주소 | 크기 | 설명 | CLI 출력 |
|---|---|---|---|---|
| jpeg | 0x50051000 | 256 B | JPEG 하드웨어 코덱 |   Size          : 256 Bytes /   Address:      : 0x50051000 / Error: Data read failed |

**JPEG 하드웨어 코덱 (0x50051000) — 미탑재 증거**

> STM32F746 계열 데이터시트에는 JPEG 코덱이 없고 STM32F76x/77x 에만 있다. 0x50051000 읽기가 버스 에러로 실패했다는 것은 **이 실리콘에 JPEG 코덱이 물리적으로 탑재되어 있지 않다**는 직접 증거다.

> 주의: 읽기에 '성공'했어도 해당 페리페럴의 RCC 클럭이 꺼져 있으면 레지스터가 전부 0x00000000 으로 읽힌다. 값이 전부 0 인 블록은 '페리페럴 부재'가 아니라 '클럭 미공급'으로 해석해야 한다.


### 전 영역이 0x00 으로 읽힌 블록

| 영역 | 주소 | 크기 | 설명 | 판정 |
|---|---|---|---|---|
| syscfg | 0x40013800 | 64 B | SYSCFG: 메모리 리맵, EXTI 소스 | RCC 클럭 **ON** (SYSCFGEN=1) → 클럭은 살아 있으나 레지스터가 실제로 전부 리셋값(0)임 |
| crc | 0x40023000 | 32 B | CRC 계산 유닛 | RCC 클럭 **OFF** (CRCEN=0) → 클럭 미공급 때문 |
| dma1 | 0x40026000 | 256 B | DMA1 컨트롤러 | RCC 클럭 **OFF** (DMA1EN=0) → 클럭 미공급 때문 |
| quadspi | 0xA0001000 | 256 B | QUADSPI 컨트롤러 | RCC 클럭 **OFF** (QSPIEN=0) → 클럭 미공급 때문 |
| dcmi | 0x50050000 | 256 B | DCMI 카메라 인터페이스 | RCC 클럭 **OFF** (DCMIEN=0) → 클럭 미공급 때문 |
| spi_bank_apb1 | 0x40003800 | 3072 B | SPI2/3, SPDIFRX 영역 | RCC 클럭 **OFF** (SPI2EN=0) → 클럭 미공급 때문 |
| adc_bank | 0x40012000 | 1024 B | ADC1/2/3 + 공통 레지스터 | RCC 클럭 **OFF** (ADC1EN=0) → 클럭 미공급 때문 |
| rng | 0x50060800 | 32 B | RNG 난수 발생기 | RCC 클럭 **OFF** (RNGEN=0) → 클럭 미공급 때문 |
| tim1 | 0x40010000 | 256 B | TIM1 (고급 제어 타이머) | RCC 클럭 **OFF** (TIM1EN=0) → 클럭 미공급 때문 |
| tim2 | 0x40000000 | 256 B | TIM2 (32비트 범용 타이머) | RCC 클럭 **OFF** (TIM2EN=0) → 클럭 미공급 때문 |
| eth | 0x40028000 | 1024 B | Ethernet MAC | RCC 클럭 **OFF** (ETHMACEN=0) → 클럭 미공급 때문 |
| usb_hs_core | 0x40040000 | 1024 B | USB OTG HS 코어 레지스터 | RCC 클럭 **OFF** (OTGHSEN=0) → 클럭 미공급 때문 |
| usb_fs_core | 0x50000000 | 1024 B | USB OTG FS 코어 레지스터 | RCC 클럭 **OFF** (OTGFSEN=0) → 클럭 미공급 때문 |


## 주의: 하드웨어 고유값 vs 펌웨어 설정

같은 덤프 안이라도 값의 성격이 전혀 다르다. 논문/문서에 인용할 때 반드시 구분해야 한다.


### A. 실리콘 고정 — 소프트웨어로 바꿀 수 없는 값

| 항목 | 근거 레지스터 | 성격 |
|---|---|---|
| 코어 종류/리비전 | CPUID (0xE000ED00) | 읽기 전용 하드웨어 ID |
| NVIC 인터럽트 라인 수 | ICTR (0xE000E004) | 구현 고정 |
| L1 캐시 구성/용량 | CLIDR, CTR, CCSIDR | 읽기 전용 (CSSELR 로 선택만 가능) |
| TCM 크기 필드 | ITCMCR/DTCMCR 의 SZ[6:3] | 읽기 전용 구현값 (EN/RMW 는 쓰기 가능) |
| AHBP 영역 크기 | AHBPCR 의 SZ[3:1] | 읽기 전용 구현값 |
| FPU 기능 | MVFR0/1/2 | 읽기 전용 — 배정밀도 부재는 실리콘 사실 |
| MPU 리전 수 | MPU_TYPE (0xE000ED90) | 읽기 전용 |
| 하드웨어 브레이크포인트 수 | FP_CTRL 의 NUM_CODE/NUM_LIT/REV | 읽기 전용 |
| CoreSight 컴포넌트 구성 | ROM table + 각 컴포넌트 PIDR/CIDR | 읽기 전용 |
| 디바이스/리비전 ID | DBGMCU_IDCODE (0xE0042000) | 읽기 전용 |
| UID 96비트, Flash 용량, ADC 캘리브레이션 | 0x1FF0F420 / 0x1FF0F442 / 0x1FF0F44A~E | ST 공장 기록, 읽기 전용 |
| 코어 기능 ID (DSP 확장 등) | ID_PFR/ID_MMFR/ID_ISAR | 읽기 전용 |
| JPEG 코덱 부재 | 0x50051000 읽기 실패 | 버스 디코딩 결과 = 실리콘 사실 |
| 시스템 부트로더 코드 | 0x1FF00000 | ST 가 마스크 ROM 에 기록, 변경 불가 |


### B. 옵션 바이트 — 비휘발성이지만 다시 쓸 수 있는 값

| 항목 | 근거 레지스터 | 성격 |
|---|---|---|
| RDP 레벨, BOR 레벨, 워치독 SW/HW | FLASH_OPTCR (0x40023C14) | 옵션 바이트 (재프로그램 가능) |
| nRST_STOP / nRST_STDBY / nWRP | FLASH_OPTCR | 옵션 바이트 |
| 부트 주소 BOOT_ADD0/1 | FLASH_OPTCR1 (0x40023C18) | 옵션 바이트 — 리셋 후 VTOR 을 결정 |


### C. 덤프 시점의 소프트웨어 상태 — 실행 중이던 코드가 설정한 값

> 이 덤프는 **실행 중(HOTPLUG 모드)** 이므로 아래 값들은 그 시점에 보드에서 돌고 있던 펌웨어가 설정한 상태다.

| 항목 | 근거 레지스터 | 성격 |
|---|---|---|
| I-cache/D-cache/분기예측 ON/OFF | SCB_CCR (0xE000ED14) | 소프트웨어 설정 |
| FPU 활성 여부 | SCB_CPACR (0xE000ED88) | 소프트웨어 설정 |
| 벡터 테이블 위치 | SCB_VTOR (0xE000ED08) | 리셋 시 BOOT_ADD 로 결정, 이후 SW 가 변경 가능 |
| NVIC 우선순위 그룹 | SCB_AIRCR 의 PRIGROUP | 소프트웨어 설정 |
| MPU 활성/리전 구성 | MPU_CTRL / MPU_RBAR / MPU_RASR | 소프트웨어 설정 |
| ACTLR 성능 비트 | ACTLR (0xE000E008) | 소프트웨어 설정 |
| CACR (ECC/FORCEWT/SIWT) | CACR (0xE000EF9C) | 소프트웨어 설정 (ECCEN 리셋값은 1) |
| SYSCLK/HCLK/PCLK, PLL 계수 | RCC_CR / PLLCFGR / CFGR | 소프트웨어 설정 |
| 페리페럴 클럭 인에이블 목록 | RCC_AHB1~3ENR, APB1~2ENR | 소프트웨어 설정 |
| Flash 대기 상태, ART, 프리페치 | FLASH_ACR (0x40023C00) | 소프트웨어 설정 |
| 전압 스케일/Over-drive | PWR_CR1 / PWR_CSR1 | 소프트웨어 설정 |
| SDRAM 타이밍/리프레시 | FMC_SDCR/SDTR/SDRTR | 소프트웨어 설정 (칩 사양에 맞춘 값) |
| LCD 해상도/타이밍 | LTDC_SSCR/BPCR/AWCR/TWCR | 소프트웨어 설정 (패널 사양에 맞춘 값) |
| GPIO 모드/AF 배정 | GPIOx_MODER / AFRL / AFRH | 소프트웨어 설정 — 단, 배선상 가능한 AF 조합은 실리콘/보드가 제약한다 |
| 리셋 원인 플래그 | RCC_CSR 상위 비트 | 하드웨어가 세우고 SW 가 지우는 누적 플래그 |
| 메모리 내용(SRAM/SDRAM/ITCM) | mem_*.bin | 실행 중이던 프로그램의 데이터 |


### D. 이 덤프로는 확정할 수 없는 것

- DWT 의 실제 사양(NUMCOMP 등) — DEMCR.TRCENA=0 이라 전부 0 으로 읽힘
- MPU 의 전체 리전 구성 — RNR 로 선택된 한 리전만 보임
- TCM 의 물리 크기 — SZ 필드가 데이터시트와 어긋남(온디바이스 aliasing 실측 필요)
- QSPI 플래시의 존재/용량 — 컨트롤러 클럭이 꺼져 레지스터가 0
- 패키지 종류 — 패키지 데이터 레지스터가 0xFFFFFFFF
- GPIO 중 클럭이 꺼진 포트의 실제 설정 — 블록이 반복 패턴으로 읽힘

