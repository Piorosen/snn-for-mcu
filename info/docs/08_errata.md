# 08. 에라타 · 알려진 하드웨어 결함 — STM32F746NGH6 (STM32F746G-DISCO)

대상: STM32F746NGH6 (Cortex-M7 r0p1), DEV_ID = 0x449, **Revision ID = Rev Z (REV_ID = 0x1001)**
기준 문서: **ES0290 Rev 7 (18-Jul-2019)**, **ARM SDEN-1068427 v11.0 (28-May-2024)**, RM0385, ARM DDI 0489F, ARM DUI 0646B

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거 자료: `info/raw/measured_facts.md`, `info/results/probe_output.txt`,
> `info/raw/decoded_hotplug.md`, `info/raw/decoded_reset.md`
>
> 에라타 문서의 성격상 대부분의 항목은 "결함이 이 보드에서 재현되는가"를 재야 하는데, 그런 재현 시험은
> 아직 하지 않았다. 아래 "확인됨"은 **에라타의 전제조건·회피책 적용 여부가 레지스터 실측으로 확인된 것**이다.

### 실측으로 확인됨

- **실리콘 식별 (§1)**: 문서 "DEV_ID 0x449 / REV_ID 0x1001 = Rev Z" → 실측 `DBGMCU_IDCODE @0xE0042000 = 0x10016449`.
  **ES0290 Rev 7 Table 3의 36개 항목 전부가 이 보드에 적용된다는 전제가 확정**됐다.
- **코어 리비전 r0p1 (§1.3)**: 문서가 필드 인코딩으로 **계산**한 `0x410FC271` [D] → 실측 `CPUID = 0x410FC271`로 **정확히 일치**.
  ARM SDEN v11.0의 r0p1 적용 25개 항목이 그대로 해당된다는 전제가 확정됐다.
- **§2.3.4 (FMC write FIFO) 회피책이 이미 충족돼 있다**: 실측 `FMC_BCR1 @0xA0000000 = 0x000030DB`(리셋 기본값) →
  **WFDIS(bit 21) = 0 = write FIFO 활성**. 문서가 요구한 조건을 별도 조치 없이 만족한다.
- **§2.4.4 (QUADSPI 타임아웃 카운터) 회피책도 충족**: 실측 `QUADSPI_CR = 0x01400311` → **TCEN(bit 3) = 0**.
  ST BSP가 타임아웃 카운터를 켜지 않으므로 memory-mapped hang 조건에 들어가지 않는다.
- **1259864의 조건 (3)·(4)에 노출돼 있지 않다**: 실측 `CACR = 0x00000002` → **FORCEWT = 0, SIWT = 0**, ECCEN = 1.
  리셋 직후 덤프와 실행 중 덤프가 동일하다. §13.1의 "FORCEWT를 절대 set하지 말 것"은 현재 지켜지고 있다.
- **기본 메모리 맵에서 `0xC0000000`이 Device(캐시 불가)라는 §5.3의 서술이 성능으로 확인됐다**:
  MPU를 설정하지 않은 프로브 펌웨어에서 SDRAM 읽기가 **D-cache OFF 59.70 MB/s vs write-back 59.71 MB/s**로 사실상 동일했다.
  즉 SDRAM은 애초에 캐싱되지 않는다. 반대로 SRAM1(`0x20010000`, 기본맵 WBWA)은 172.85 → 206.32 MB/s로 캐시 효과가 나타난다.
- **`SYSCFG_MEMRMP.SWP_FMC = 0b00`** (실측 `SYSCFG_MEMRMP = 0x00000000`) → §5.3에서 언급한 FMC 뱅크 스왑은 **사용되지 않는 상태**.
  SDRAM은 기본 위치 `0xC0000000`에 있다.
- **"미확인 #6" 해소 — `CM7_AHBSCR`은 STM32F746에서 접근 가능하다**: 실측 `AHBSCR @0xE000EFA0 = 0x00000800`
  → **CTL = 0b00**, INITCOUNT = 1. 즉 **ARM 2328489의 회피책 ①(AHBS priority demotion, CTL=0b00)이 리셋 기본값으로 이미 적용된 상태**다.
  (리셋 직후 덤프와 실행 중 덤프가 동일 → 소프트웨어가 바꾼 값이 아니다.)
- **486321(프로파일링 카운터) 관련 하드웨어 존재 확인**: TRCENA를 켠 뒤 `DWT_CTRL = 0x40000001` →
  NUMCOMP=4, **NOCYCCNT=0, NOPRFCNT=0**. 즉 CPI/EXC/SLEEP/LSU/FOLD 카운터가 실재하므로 이 에라타는 **실제로 적용 대상**이다
  (오계수 자체는 미검증). TRCENA를 켜기 전에는 DWT 레지스터가 전부 0으로 읽히므로 "카운터 없음"으로 오인하기 쉽다.
- **412512 / 505438 / ETM-M7(TM975) 에라타의 적용 대상이 실재한다**: ROM table은 TPIU·ETM을 present=0으로 광고하지만,
  두 주소의 PIDR을 직접 읽으면 **TPIU part=0x9A9, ETM part=0x975 (제조사 ARM 0x3B)** 로 응답한다 → **물리적으로 존재**.
  단 `DBGMCU_CR = 0x00000000`이라 **TRACE_IOEN = 0**(트레이스 핀 비활성)이므로 현재 구성에서는 발현하지 않는다.
- **1518990(big-endian DWT 비교) 해당 없음**: 실측 `AIRCR.ENDIANNESS = 0` → 리틀엔디언 동작.
- **회피책에 필요한 자원 확인**: `MPU_TYPE = 0x00000800` → **MPU 리전 8개**(1259864·1013783·443753의 MPU 회피책에 사용 가능),
  `FP_CTRL = 0x10000080` → 하드웨어 브레이크포인트 8개.

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

1. **§3 1순위 / §5.3의 노출 시나리오 — "MPU를 건드리지 않고 D-Cache만 켜면"이 이 보드의 실제 구성이 아니다**
   - 문서: "MPU 설정 없이 D-Cache만 켜면 Code 영역(0x00000000–0x1FFFFFFF) 전체가 WT라 1259864 조건 영역에 그대로 들어간다."
   - 실측: 원래 SNN 펌웨어가 돌던 상태의 **`MPU_CTRL = 0x00000005`(MPU 활성, PRIVDEFENA=1)**, `SHCSR.MEMFAULTENA=1`.
     즉 이 보드는 **MPU를 켠 채로 동작**하고 있었으므로, 기본 메모리 맵 속성이 그대로 적용된다는 전제가 성립하지 않는다.
     또 옵션바이트 **BOOT_ADD0 = 0x00200000**(ITCM 별칭)에서 부팅하며, **ITCM 인터페이스 경유 접근은 D-cache 대상이 아니다**
     (실측: ITCM 경로 성능이 D-cache ON/OFF에 전혀 반응하지 않음 — 492.10 MB/s로 동일).
   - 원인 추정: 문서가 **아키텍처 일반론**을 기술했고, 실측은 **이 보드의 구체적 구성**이다. 문서 내용 자체가 틀린 것은 아니지만,
     **"이 보드가 그 조건에 있다"는 함의는 사실과 다르다.** 실제 MPU 리전 설정 내용까지는 덤프하지 않았으므로,
     어느 영역이 WT/WB로 잡혀 있는지는 여전히 미확인이다(아래 참조).
   - 실무 결론: 1259864의 노출 여부를 판단하려면 **문서의 기본맵 표가 아니라 이 펌웨어의 실제 MPU 리전 설정(`MPU_RBAR`/`RASR`)을 읽어야 한다.**
2. **§13 표 "Flash 인터페이스 / ART Accelerator / prefetch — 에라타 없음"이 안심의 근거가 될 수 없다**
   - 문서: ES0290 Rev 7에 Flash/ART/prefetch 관련 에라타 항목이 **없다** [A].
   - 실측: 에라타는 없어도 **이 보드는 ART와 prefetch가 꺼진 채로 동작하고 있었다** (`FLASH_ACR = 0x00000006`,
     ARTEN=0 / PRFTEN=0 / 6 wait state). 게다가 **BOOT_ADD0가 ITCM 별칭이라 실행 경로가 ART에 의존**하는데,
     `stm32f7xx_hal_conf.h`의 `PREFETCH_ENABLE 0U` + `ART_ACCLERATOR_ENABLE` 미정의로 **HAL이 ART를 영영 켜지 않는다.**
     실측 영향: ITCM 경로 플래시 접근이 336.79 → 492.10 MB/s(**×1.46**)만큼 손해를 보고 있었다.
   - 원인 추정: 문서는 "결함 목록"만 다루었고, **설정 함정(configuration pitfall)은 에라타 문서에 실리지 않는다.**
   - 실무 결론: "에라타 없음 = 조심할 것 없음"이 아니다. 이 블록은 **에라타가 아니라 기본값 때문에** 성능을 잃는다.
3. **§14 보드 레벨 "No limitation" [C] — 실물과 맞지 않는 부분이 두 군데 있다**
   - 문서: UM1907이 MB1191 전 리비전에 "No limitation"으로 기재 → "보드 고유 하드웨어 결함은 보고된 바 없다".
   - 실측(공식 에라타는 아니지만 실무상 함정):
     - **QSPI NOR이 ST BSP가 전제한 Micron N25Q128A가 아니라 Winbond W25Q128 계열**이다 (Read JEDEC ID = `0xEF4018`,
       Micron이면 `0x20BA18`). 읽기는 표준 명령을 공유해 동작하지만 **소거 시간·페이지 크기·상태 레지스터 비트는 다르다.**
     - **이더넷 PHY가 프로젝트 설정과 다르다**: 실측 PHY는 MDIO **주소 0**의 **LAN8742A**(ID1=0x0007, ID2=0xC131)인데,
       `stm32f7xx_hal_conf.h:220`은 `DP83848_PHY_ADDRESS 0x01`로 정의돼 있다.
   - 원인 추정: **보드 리비전에 따른 부품 교체 + BSP/설정 헤더가 따라오지 못함.** UM1907의 "No limitation"은
     ST가 정의한 "device limitation"의 범위에 부품 대체가 포함되지 않기 때문이다.
4. **§5.2 표의 3092511 적용 표기** — 경미
   - 문서: 3092511의 r0p1 적용을 "✅ (`REL, r0px, r1p0, r1p1`)"로 적었다.
   - 실측으로는 검증 불가한 항목이지만, 표기 자체가 다른 행(✅/❌)과 형식이 달라 **적용 여부가 애매하게 읽힌다.**
     SDEN 원문 표기를 그대로 옮긴 것으로 보이며, 이 보드(r0p1)에 적용된다는 결론은 유지된다. 표기만 정리 필요.

### 아직 측정하지 않음

- **1259864(WT store→load 데이터 손상)의 실제 발현** — MPU로 SRAM 영역을 WT로 지정하거나 `CACR.FORCEWT=1`을 켠 뒤,
  "같은 doubleword에 store → linefill 유발 → 재store → load" 시퀀스를 수백만 회 돌려 불일치를 카운트한다.
  (참고: 프로브 펌웨어는 벤치마크 목적으로 실제로 `FORCEWT=1` 구간을 돌렸으나 **손상 검출 루틴이 없었다.**
  "손상이 보이지 않았다"는 것은 반증이 아니다 — 이 에라타는 타이밍 의존이다.)
- **§2.3.1(동기식 메모리 burst read 끝 dummy read 2회)의 대역폭 손실 정량화** — 현재 SDRAM은 Device 메모리라
  코어가 버스트를 내지 않는다(실측 읽기 60.25 MB/s = 이론 200 MB/s의 30%). **MPU로 SDRAM을 Normal cacheable로 바꿔
  캐시 라인필(32 B 버스트)이 걸리게 한 뒤** 대역폭을 다시 재면, dummy read 2회분(버스트당 약 25%)이 분리돼 보인다.
- **ARM 2328489(TCM 대역폭 공유 stall)** — DMA를 AHBS 경유로 DTCM에 연속 write시키면서 코어가 `STM`/`STRD`로
  같은 TCM에 64비트 store를 반복해 CYCCNT 증가를 비교한다. `AHBSCR.CTL`을 0b00 ↔ 0b01로 바꿔가며 차이를 본다.
- **ARM 486321(프로파일 카운터 오계수)** — LSU가 지배적인 루프(캐시 미스 유발)에서 `DWT_LSUCNT`와 `DWT_CPICNT`를
  동시에 읽어, 메모리 stall이 LSUCNT가 아니라 CPICNT로 잡히는지 확인한다. 카운터 존재는 이미 실측됐다.
- **ARM 440977(BASEPRI 즉시 반영 안 됨)** — `MSR BASEPRI` 직후 1개 명령 창에 타이머 인터럽트를 정렬시켜
  선점 발생 여부를 카운트한다.
- **§2.2.2(ADC 정확도 저하) 회피책의 효과** — 실측 `PWR_CR1 = 0x0003C000` → **ADCDC1(bit 13) = 0**,
  `SYSCFG_PMC = 0x00000000` → **ADCxDC2 = 0**. 즉 **두 회피책 모두 적용돼 있지 않은 상태**에서 VDDA·다이 온도를 측정했다.
  ADCDC1=1 전후로 VREFINT 코드의 표준편차를 비교하면 효과를 정량화할 수 있다.
- **§2.9.x(I2C)** — I2C3에 WM8994·FT5336이 실장돼 있으므로, reload 모드(`NBYTES>1`)와 10비트 주소 NACK 시나리오를
  실제 장치 상대로 재현할 수 있다.
- **§2.12.1(SDMMC1 CCRCFAIL 오탐)** — microSD가 삽입돼 있다(PC13=0). CMD5(`IO_SEND_OP_COND`) 발행 후
  `SDMMC_STA.CCRCFAIL`이 서는지 확인하면 된다.
- **§2.14.x(Ethernet)** — PHY는 LAN8742A로 확인됐으나 링크가 down(BMSR=0x7809)이다. 케이블 연결 후
  `ETH_DMAOMR.TSF` 0/1을 바꿔가며 전송 프레임 CRC 오류율을 비교해야 한다.
- **§2.2.1(TIM TRGO → ADC 트리거 누락)** — TIM 트리거로 ADC를 구동하며 `EXTEN=0x11` 유무에 따른 변환 개수 차이를 센다.
- **MPU 리전 설정 실측** — 위 CONFLICT 1을 마무리하려면 실행 중 `MPU_RBAR`/`MPU_RASR`를 리전 8개 전부 덤프해
  어느 영역이 WT/WB/non-cacheable로 잡혀 있는지 확인해야 한다. 이번 덤프에는 `MPU_CTRL`만 있다.

---

## 요약

- 이 보드의 실리콘은 ES0290 Rev 7 Table 2 기준 **"Rev. Z and 1"** 그룹이며, **REV_ID = 0x1001**, DEV_ID = 0x449 이다. **0x1003은 이 패밀리에서 사용되지 않는다.** Rev Z와 Rev 1이 **동일한 REV_ID(0x1001)** 를 반환하므로 레지스터만으로는 두 리비전을 구분할 수 없다. [A]
- ES0290 Rev 7의 리비전 적용 표(Table 3)는 **단일 컬럼 "Rev. Z and 1"** 뿐이다. 즉 **문서에 실린 36개 한계 전부가 이 보드에 적용된다.** Rev A는 ES0290 Rev 2(2015-05-26)에서 문서상 제거되었다. [A]
- 36개 항목 상태 내역: **A(회피책 있음) 30개 / N(회피책 없음) 5개 / P(부분 회피) 1개**. [A]
- 성능·정확도에 직접 영향을 주는 최상위 항목은 **ARM 에라타 1259864 (ES0290 §2.1.1, Category A)** — D-Cache가 **write-through**로 설정된 영역에서 store→load 데이터 손상. 직접 회피책 없음, MPU로 **write-back 전환**하거나 해당 구간 캐시 비활성화. [A]
- ARMv7-M **기본 메모리 맵**에서 `0x00000000–0x1FFFFFFF`(Code, 내부 Flash 포함)와 `0x80000000–0x9FFFFFFF`가 **WT(write-through)** 다. 즉 MPU를 건드리지 않고 D-Cache만 켜면 1259864의 조건 영역에 그대로 들어간다. [A]
- ARM Cortex-M7 SDEN-1068427 v11.0에 등재된 **26개 에라타 중 25개가 r0p1에 적용**된다(예외: 505438은 r0p2/r1p0 전용). 이 중 **9개는 r1p2까지도 미수정("Open")**, 13개는 r0p2에서 수정, 2개는 r1p1, 1개(1259864)는 r1p2에서 수정. [A]
- ES0290 §2.1 Table 4는 **구(舊) ARM ID 11개만** 나열하며(837070, 834922, 838169, 839269, 839169, 837069, 834971, 834924, 834923, 833872, 830969), 이는 2016년 시점 목록이다. **현행 ARM SDEN v11.0 대비 14개가 누락**되어 있으므로 ES0290만 보면 안 된다. [A]
- FMC 관련: **§2.3.1 동기식 메모리 버스트 읽기 시 끝에 dummy read 2회 삽입 — 회피책 없음(N)**, §2.3.4 **write FIFO 비활성화(FMC_BCR1.WFDIS=1) 시 read 데이터 손상 → write FIFO를 켜라**. SDRAM 대역폭/레이턴시에 직접 영향. [A]
- ES0290 Rev 7에는 **USB OTG, DMA 컨트롤러(DMA1/DMA2), Flash/ART Accelerator, 캐시 일관성 자체에 대한 에라타 항목이 없다.** DMA↔D-Cache 비일관성은 에라타가 아니라 **설계상 제약**이다. [A]
- 보드(MB1191) 레벨: UM1907은 MB1191-F746NGH6-B02/B03/B04/B05/C01 전 리비전에 대해 **"No limitation"** 으로 기재. [C]

---

## 1. 이 보드의 식별값 — 문서 대조

### 1.1 DBGMCU_IDCODE

| 항목 | 값 | 출처 |
|---|---|---|
| 레지스터 주소 | `0xE004 2000` (외부 PPB, 32-bit access only, read-only) | RM0385 §40.6.1 [A] |
| `DEV_ID[11:0]` | **0x449** (STM32F75xxx/F74xxx) | RM0385 §40.6.1 [A] |
| `REV_ID[15:0]` = 0x1000 | Revision A | RM0385 §40.6.1 [A] |
| `REV_ID[15:0]` = **0x1001** | **Revision Z** ← **본 보드 실측값에 해당** | RM0385 §40.6.1 [A] |
| Bits 15:12 | Reserved, reset value 유지 | RM0385 §40.6.1 [A] |

> RM0385 (DocID026670 Rev 5) §40.6.1은 **0x1000과 0x1001 두 값만** 정의한다. **0x1003은 정의되어 있지 않다.** [A]

ES0290 Rev 7 Table 2 (Device variants):

| Reference | Device marking (실리콘 리비전 코드) | REV_ID |
|---|---|---|
| STM32F74xxx / STM32F75xxx | **Z and 1** | **0x1001** |

- 각주 1: 패키지 유형별 마킹 확인 방법은 데이터시트 참조.
- 각주 2: `REV_ID[15:0]` bit field of `DBGMCU_IDCODE`. [A] (ES0290 Rev 7, p.1)

**중요:** ST 문서상 **device marking "Z"와 "1"이 동일하게 REV_ID = 0x1001을 반환**한다. 따라서 ST-LINK/CubeProgrammer가 표시하는 "Rev Z"는 툴 DB가 0x1001 → "Z"로 매핑한 결과이며, 실물 마킹이 "1"인 칩도 같은 값을 읽는다. **두 리비전은 소프트웨어로 구분 불가**이고, ES0290도 둘을 동일 취급한다. [A]

### 1.2 기타 ID 코드

| 항목 | 값 | 출처 |
|---|---|---|
| JTAG ID code (BSC, boundary scan) | `0x0644 9041` | RM0385 §40.6.2 [A] |
| Cortex-M7 w/FPU Debug Port JTAG ID | `0x5BA0 0477` (ARM 기본값, 미변경) | RM0385 §40.6.3 [A] |
| Cortex-M7 JEDEC-106 ID code 위치 | ROM table, `0xE00F F000–0xE00F FFFF` (4KB) | RM0385 §40.6.4 [A] |

### 1.3 코어 리비전 r0p1

ES0290 §2.1: *"All the described limitations are minor and related to the **revision r0p1** of the Cortex-M7 core."* [A]

`CPUID` (`0xE000ED00`) 비트 필드 (ARM DDI 0489F §3.3.2 Table 3-4): [A]

| Bits | Name | 값 → 의미 |
|---|---|---|
| [31:24] | IMPLEMENTER | `0x41` = Arm |
| [23:20] | VARIANT | `0x0` = Revision 0 |
| [19:16] | ARCHITECTURE | `0xF` (고정) |
| [15:4] | PARTNO | `0xC27` = Cortex-M7 |
| [3:0] | REVISION | `0x1` = Patch 1 |

→ r0p1의 CPUID 조합값은 **`0x410F C271`** 이다. [D] (위 필드 인코딩으로부터 계산; ST 커뮤니티에서 STM32F746ZG 실측 `0x410FC271` 보고와 일치 [C])

> **실측 확인**: 이 보드에서 `CPUID @0xE000ED00`을 직접 읽은 값이 **정확히 `0x410FC271`** 이다 (ST-LINK·온디바이스 두 방법 일치).
> 문서가 [D](추론)로 남겨 둔 값이 **[A](실측)로 승격**된다. `DBGMCU_IDCODE = 0x10016449`도 함께 확인 →
> **DEV_ID 0x449 / REV_ID 0x1001 (Rev Z)**. 따라서 이 문서가 전제한 "ES0290 Rev 7 36개 항목 전부 적용 + SDEN r0p1 25개 적용"은
> 실측 근거를 갖는다.

---

## 2. 리비전별 적용 표 (Rev A / Rev Z / Rev 1)

ES0290 Rev 7 Table 3의 상태 컬럼은 **"Rev. Z and 1" 단 하나**이다. Rev A 컬럼은 존재하지 않는다. [A]

| 리비전 | ES0290 Rev 7에서의 취급 | REV_ID | 근거 |
|---|---|---|---|
| **Rev A** | **문서에서 제거됨.** ES0290 Rev 2(26-May-2015) 개정 이력: *"Changing revision A into the revision Z."* | 0x1000 (RM0385 정의) | ES0290 Rev 7 Table 6 [A] |
| **Rev Z** ← 본 보드 | Table 3의 모든 항목 적용 | **0x1001** | ES0290 Rev 7 Table 2/3 [A] |
| **Rev 1** | Rev Z와 **완전히 동일하게** 취급 (동일 컬럼). ES0290 Rev 6(25-Jun-2018)에서 *"Added Revision 1: Updated Table 2, Updated Table 3"* 로 추가 | **0x1001** (Rev Z와 동일) | ES0290 Rev 7 Table 2/6 [A] |

**결론: Rev Z 보드에는 ES0290 Rev 7의 36개 한계 전부가 적용된다. Rev 1로 넘어가도 해소되는 항목은 없다.** [A]

### ES0290 문서 개정 이력 (Table 6) 요약 [A]

| 날짜 | Rev | 주요 변경 |
|---|---|---|
| 19-Jan-2015 | 1 | 최초 릴리스 |
| 26-May-2015 | 2 | **Revision A → Revision Z로 전환**; STM32F745xx RPN 추가; USART 절(2.10.x) 추가; SDMMC 2개 한계 삭제; Standby/LSE/ADC 관련 다수 항목 삭제 |
| 07-Jan-2016 | 3 | **§2.1 Arm Cortex-M7 core 절 신설**; §2.2.3, §2.11.1, §2.8.2, §2.3.2, §2.3.3 추가; "NAND/PCCard transaction and wait timing" 삭제 |
| 24-Nov-2016 | 4 | §2.9.3, §2.4.2, §2.4.3 추가 |
| 18-May-2017 | 5 | §2.3.4 추가; §2.4.2 갱신; 새 에라타 시트 구조 적용 |
| 25-Jun-2018 | 6 | **Revision 1 추가**; STM32F750x8 RPN 추가; §2.4.4, §2.9.4, §2.8.1, §2.7.1 추가; FMC 4개 항목 갱신 |
| **18-Jul-2019** | **7 (현행)** | **§2.1.1 (Cortex-M7 write-through 데이터 손상), §2.2.4 (NJTRST) 추가** |

> Wayback Machine에 보존된 st.com 원본 PDF를 2025-02-21 및 2026-01-08 캡처로 확인한 결과 **둘 다 ES0290 Rev 7 (28페이지, 478,596 bytes)** 로 동일. 2026-07 현재 Rev 7이 최신으로 판단된다. ST 커뮤니티의 "STM32 MCU errata sheets: Expected preliminary updates" 페이지에도 STM32F7 계열 예정 업데이트는 없다. [B]

---

## 3. 성능·정확도에 직접 영향을 주는 항목 (우선순위 정리)

| 순위 | 항목 | 영향 | 회피책 |
|---|---|---|---|
| **1** | **ARM 1259864 / ES0290 §2.1.1** — write-through D-Cache에서 store 후 load가 **옛 데이터 반환** | **연산 결과 오염(정확도)**. 기본 메모리 맵상 Code(0x00000000–0x1FFFFFFF), 0x80000000–0x9FFFFFFF가 WT | **직접 회피책 없음.** MPU로 WT→WB 변경, 또는 해당 구간 D-Cache off |
| **2** | **ES0290 §2.3.4 (FMC)** — write FIFO 비활성화 시 read 데이터 손상 | **SDRAM/외부메모리 read 데이터 손상** | `FMC_BCR1.WFDIS = 0` (write FIFO **활성화**) |
| **3** | **ES0290 §2.3.1 (FMC)** — 동기식 메모리 burst read 끝에 **dummy read 2회** 삽입 | **외부 메모리 대역폭 손실**(기능 오류는 없음) | **없음 (N)** |
| **4** | **ARM 2328489 (Cat B, Open)** — TCM wait state 사용 시 AHBS write가 64-bit software store보다 우선 → **파이프라인 stall** | **순수 성능 저하** | `CM7_AHBSCR.CTL = 0b00` + `INIT = N` 설정(AHBS priority demotion) |
| **5** | **ARM 565285 (Cat B, Open)** — AXI write ID 재정렬 허용 → eviction이 younger store를 덮어씀 | **데이터 손상** (인터커넥트가 재정렬하는 경우) | Write-Back **no** Write-Allocate 사용 회피 등 |
| **6** | **ARM 1313001 (Cat C, Open)** — cache invalidate 직후 배리어 없는 WT store → **stale data** | **DMA 버퍼 무효화 코드에서 발생 가능** | maintenance op와 store 사이에 **DMB** 삽입 |
| **7** | **ES0290 §2.2.2 (System)** — 내부 노이즈가 **ADC 정확도** 저하. Run/Sleep 무관 상시 활성 | **ADC 정확도** | `PWR_CR.ADCDC1` **또는** `SYSCFG_PMC.ADCxDC2` 중 **택일** 설정 |
| **8** | **ES0290 §2.2.1 (System)** — TIM TRGO/TRGO2 → ADC 트리거 **누락** | **샘플링 누락** | TIM1/8: `ADC_CR2.EXTEN/JEXTEN = 0x11` (양 에지) / TIM2·4·5·6: `RCC_APB1ENR`에서 **DAC 클럭 enable** |
| **9** | **ARM 440977 (Cat B, r0p2 수정)** — `MSR BASEPRI` 직후 1개 명령이 **낮은 우선순위 인터럽트에 선점** | **크리티컬 섹션 침해** | `CPSID i` → `MSR BASEPRI` → `CPSIE i` 시퀀스 |
| **10** | **ES0290 §2.14.4 (Ethernet)** — TxFIFO non-empty↔empty 반복 시 **전송 프레임 데이터 손상** | 네트워크 데이터 무결성 | **Store-and-Forward 모드**: `ETH_DMAOMR.TSF = 1` (bit 21) |

> **실측 정정 / 확인 (위 우선순위 표)**
> - **1순위 (1259864)**: 조건 (3) `CACR.FORCEWT`와 (4) `CACR.SIWT`는 **실측 `CACR = 0x00000002`로 둘 다 0** — 노출 없음.
>   그러나 조건 (2)"기본 메모리 맵"은 **이 보드에 그대로 적용되지 않는다**: 실행 중 `MPU_CTRL = 0x00000005`로 **MPU가 켜져 있었다.**
>   실제 노출 여부는 MPU 리전 설정을 읽어야 판정할 수 있다(이번 덤프에 없음).
> - **2순위 (§2.3.4 FMC write FIFO)**: **이미 회피된 상태다.** 실측 `FMC_BCR1 = 0x000030DB`(리셋 기본값) → **WFDIS = 0**.
> - **3순위 (§2.3.1 dummy read)**: 실측 SDRAM 순차 읽기는 **60.25 MB/s = 이론(200 MB/s)의 30%** 로 확실히 낮지만,
>   현재 SDRAM이 **Device 메모리라 버스트가 발생하지 않는** 것이 더 큰 원인일 수 있다. **이 에라타 몫은 아직 분리되지 않았다.**
> - **4순위 (2328489 TCM)**: 회피책 ①이 **리셋 기본값으로 이미 적용돼 있다.** 실측 `CM7_AHBSCR @0xE000EFA0 = 0x00000800`
>   → **CTL = 0b00**, INITCOUNT = 1. 문서의 "미확인 #6"(이 레지스터가 F746에서 접근 가능한가)도 함께 해소됐다 — **접근 가능하다.**
> - **7순위 (§2.2.2 ADC 노이즈)**: **회피책이 적용돼 있지 않다.** 실측 `PWR_CR1 = 0x0003C000` → ADCDC1(bit 13) = 0,
>   `SYSCFG_PMC = 0x00000000` → ADCxDC2 = 0. 이 상태에서 측정한 값이 VDDA 3.311~3.313 V, 다이 온도 35.7~39.2 °C다.
> - **10순위 (§2.14.4 Ethernet)**: 보드 PHY는 **LAN8742A(MDIO 주소 0)** 로 실측 확정. 링크 down 상태라 미검증.

---

## 4. ES0290 Rev 7 전체 요약표 (Table 3, Rev. Z and 1)

상태 표기: **A** = 한계 존재, 회피책 있음 / **N** = 한계 존재, 회피책 없음 / **P** = 부분 회피책만 존재 [A]

| Function | Section | Limitation | Rev. Z and 1 |
|---|---|---|---|
| Arm Cortex-M7 core | 2.1.1 | Cortex-M7 data corruption when using data cache configured in write-through | **A** |
| System | 2.2.1 | Missed ADC triggers from TIM1/TIM8, TIM2/TIM5/TIM4/TIM6/TRGO or TGRO2 event | A |
| System | 2.2.2 | Internal noise impacting the ADC accuracy | A |
| System | 2.2.3 | Wakeup from Standby mode when the back-up SRAM regulator is enabled | A |
| System | 2.2.4 | Full JTAG configuration without NJTRST pin cannot be used | A |
| FMC | 2.3.1 | Dummy read cycles inserted when reading synchronous memories | **N** |
| FMC | 2.3.2 | Wrong data read from a busy NAND memory | A |
| FMC | 2.3.3 | Spurious clock stoppage with continuous clock feature enabled | A |
| FMC | 2.3.4 | Data read might be corrupted when the write FIFO is disabled | A |
| QUADSPI | 2.4.1 | Extra data written in the FIFO at the end of a read transfer | A |
| QUADSPI | 2.4.2 | First nibble of data not written after a dummy phase | A |
| QUADSPI | 2.4.3 | Wrong data from memory-mapped read after an indirect mode operation | A |
| QUADSPI | 2.4.4 | Memory-mapped read operations may fail when timeout counter is enabled | A |
| ADC | 2.5.1 | ADC sequencer modification during conversion | A |
| DAC | 2.6.1 | DMA underrun flag management | A |
| DAC | 2.6.2 | DMA request not automatically cleared by clearing DMAEN | A |
| LPTIM | 2.7.1 | MCU may remain stuck in LPTIM interrupt when entering Stop mode | A |
| RTC | 2.8.1 | RTC calendar registers are not locked properly | **P** |
| RTC | 2.8.2 | Spurious tamper detection when disabling the tamper channel | **N** |
| I2C | 2.9.1 | Wrong data sampling when tSU;DAT is shorter than one I2C kernel clock period | A |
| I2C | 2.9.2 | Spurious bus error detection in master mode | A |
| I2C | 2.9.3 | 10-bit master mode: new transfer cannot be launched if first part of the address is not acknowledged | A |
| I2C | 2.9.4 | Last-received byte loss in reload mode | A |
| USART | 2.10.1 | Start bit detected too soon when sampling for NACK signal from the smartcard | **N** |
| USART | 2.10.2 | Break request can prevent the transmission complete flag (TC) from being set | A |
| USART | 2.10.3 | nRTS is active while RE or UE = 0 | A |
| SPI/I2S | 2.11.1 | I2S slave in PCM short pulse mode sensitive to timing between WS and CK | A |
| SPI/I2S | 2.11.2 | BSY bit may stay high at the end of a data transfer in Slave mode | A |
| SDMMC1 | 2.12.1 | Wrong CCRCFAIL status after a response without CRC is received | A |
| SDMMC1 | 2.12.2 | MMC stream write of less than 8 bytes does not work correctly | A |
| BxCAN | 2.13.1 | BxCAN time triggered mode communication not supported | **N** |
| Ethernet | 2.14.1 | Incorrect L3 checksum inserted in transmitted IPv6 packets without TCP/UDP/ICMP payloads | A |
| Ethernet | 2.14.2 | The Ethernet MAC processes invalid extension headers in the received IPv6 frames | **N** |
| Ethernet | 2.14.3 | MAC stuck in Idle state on TxFIFO flush command exactly 1 clock cycle after transmission completes | A |
| Ethernet | 2.14.4 | Transmit frame data corruption | A |
| Ethernet | 2.14.5 | Successive write operations to the same register might not be fully taken into account | A |

**합계: 36개 항목 = A 30 / N 5 / P 1** [A]

---

## 5. Cortex-M7 코어 에라타 (ES0290 §2.1 + ARM SDEN-1068427)

### 5.1 ES0290 §2.1 Table 4 (구 ARM ID) — 및 현행 ID 매핑

ES0290 §2.1은 ARM 에라타 통지문 2건을 참조하도록 지시한다: [A]
- *Arm processor Cortex-M7 (AT610) and Cortex-M7 with FPU (AT611) software developer errata notice* (= **SDEN-1068427**)
- *Arm embedded trace macrocell CoreSight ETM-M7 (TM975) software developer errata notice*

ES0290 Table 4에 나열된 11개(전부 STM32F74xxx/F75xxx에 "Minor" 영향으로 평가): [A]

| ES0290 Table 4의 구 ARM ID | ARM 카테고리 | 현행 ARM ID | 제목 |
|---|---|---|---|
| 837070 | Cat B | **440977** | Increasing priority using a write to BASEPRI does not take effect immediately |
| 834922 | Cat B | **426115** | Interrupting a FAULTMASK-setting instruction might cause incorrect MPU instruction attributes |
| 838169 | Cat B (rare) | **443753** | A sequence of cacheable stores to memory locations that always return bus faults might cause deadlock |
| 839269 | Cat C | **449383** | Write to FPCCR.ASPEN while a Single-precision FP MAC is completing might corrupt the FP register bank |
| 839169 | Cat C | **408519** | Incorrect GTS packet generation when global timestamps are enabled during debug using the ITM |
| 837069 | Cat C | **431216** | Unimplemented bits of BASEPRI do not read-as-zero |
| 834971 | Cat C | **416915** | HFSR.FORCED bit is not set for configurable priority faults which result in LOCKUP |
| 834924 | Cat C | **423541** | Interrupts on a bus-aborting strongly-ordered or device load to the stack pointer might cause incorrect exception stacking |
| 834923 | Cat C | **422825** | MPU fetch attributes might transiently be incorrect after an exception return |
| 833872 | Cat C | **421025** | Early forwarding from load is incorrectly cancelled inside IT block |
| 830969 | Cat C | **399743** | The Fault Address Register (FAR) might be corrupted when BFHFNMIGN is set |

(구→신 ID 매핑은 SDEN-1068427 v11.0 각 항목 말미의 *"the previous ID … is deprecated"* 문구로 확인) [A]

> **주의:** ES0290 Table 4는 2016년(ES0290 Rev 3) 시점 목록으로, **현행 SDEN v11.0 대비 14개 항목이 빠져 있다**. 아래 5.2가 실제 적용 전체 목록이다. [A]

### 5.2 ARM SDEN-1068427 v11.0 — r0p1 적용 전체 목록 (본 보드에 해당)

SDEN v11.0 등재 **26개 중 25개가 r0p1에 적용**(505438만 제외). [A]

| ARM ID | Cat | 제목 | r0p1 | 수정 버전 |
|---|---|---|---|---|
| **1259864** | **A** | Data corruption in a sequence of Write-Through stores and loads | ✅ | r1p2 |
| 412512 | B | Cortex-M7 TPIU might lose trace data in configurations with an ETM but no ITM | ✅ | r0p2 |
| 426115 | B | Interrupting a FAULTMASK-setting instruction might cause incorrect MPU instruction attributes | ✅ | r0p2 |
| 440977 | B | Increasing priority using a write to BASEPRI does not take effect immediately | ✅ | r0p2 |
| **565285** | **B** | Core can send AXI transactions that permit reordering when it should not | ✅ | **Open** |
| **1013783** | **B** | PLD might perform linefill to address that would generate a MemManage Fault | ✅ | **Open** |
| **2328489** | **B** | TCM bandwidth sharing between AHBS writes and software stores might not function correctly when using TCM wait states | ✅ | **Open** |
| 443753 | B (rare) | A sequence of cacheable stores to memory locations that always return bus faults might cause deadlock | ✅ | r0p2 |
| 399743 | C | The Fault Address Register (FAR) might be corrupted when BFHFNMIGN is set | ✅ | r0p2 |
| 408519 | C | Incorrect GTS packet generation when global timestamps are enabled during debug using the ITM | ✅ | r0p2 |
| 416915 | C | HFSR.FORCED bit is not set for configurable priority faults which result in LOCKUP | ✅ | r0p2 |
| 421025 | C | Early forwarding from load is incorrectly cancelled inside IT block | ✅ | r0p2 |
| 422825 | C | MPU fetch attributes might transiently be incorrect after an exception return | ✅ | r0p2 |
| 423541 | C | Interrupts on a bus-aborting strongly-ordered or device load to the stack pointer might cause incorrect exception stacking | ✅ | r0p2 |
| 431216 | C | Unimplemented bits of BASEPRI do not read-as-zero | ✅ | r0p2 |
| 449383 | C | Write to FPCCR.ASPEN while a Single-precision FP MAC is completing might corrupt the FP register bank | ✅ | r0p2 |
| 486321 | C | Incorrect behavior of profiling counters | ✅ | r1p1 |
| 505438 | C | TPIU cannot be flushed in Debug state if Cortex-M7 TPIU is used | ❌ (r0p2, r1p0만) | r1p1 |
| 513195 | C | Lock Status Indication incorrectly reads as one for debugger reads | ✅ | r1p1 |
| **636315** | **C** | Software programming errors might not be reported for on-line MBIST access to the I-Cache | ✅ | **Open** |
| 702596 | C | Single stepping Cortex-M7 enters pending exception handler | ✅ | r0p2 |
| **1267980** | **C** | ECC error causes data corruption when the data cache error bank registers are locked | ✅ | **Open** |
| **1313001** | **C** | Store after cache invalidate without intervening barrier might cause inconsistent memory view | ✅ | **Open** |
| **1315869** | **C** | Data corruption for load following Store-Exclusive | ✅ | **Open** |
| **1518990** | **C** | Value used for DWT Data Value Comparison is in memory-endianness format, not little-endian | ✅ | **Open** |
| **3092511** | **C** | Cortex-M7 can halt in an incorrect address when breakpoint and exception occurs simultaneously | ✅ (`REL, r0px, r1p0, r1p1`) | **Open** |

집계: r0p1 적용 **25개** = r0p2에서 수정 **13개** + r1p1에서 수정 **2개** + r1p2에서 수정 **1개** + **Open 9개**. [A]

ARM 카테고리 정의 (SDEN §Categorization of errata): [A]
- **Category A**: 치명적. 회피책 없음 또는 회피책의 영향이 큼. 다수 시스템에서 **흔하게** 발생.
- **Category A (Rare)**: 치명적이나 대부분 시스템에서 드묾.
- **Category B**: 중대한 오류, 또는 수용 가능한 회피책이 있는 치명적 오류. 흔함.
- **Category B (Rare)**: 위와 같으나 드묾.
- **Category C**: 사소한 오류.

---

### 5.3 [최우선] ARM 1259864 / ES0290 §2.1.1 — Write-Through 캐시 데이터 손상

**Fault Status:** Present in **r0p1**, r0p2, r1p0, r1p1. Fixed in r1p2. **Category A.** [A]

**증상**
> "If a particular sequence of stores and loads is performed to Write-Through memory, and some timing-based internal conditions are met, then a load might not get the last data stored to that address."
> — load 명령이 **잘못된(오래된) 데이터를 반환**한다. [A]

**영향 구성**: data cache를 가진 **모든 구성**. [A]

**조건 — 메모리가 write-through여야 함 (아래 중 하나)** [A]
1. MPU가 해당 주소를 write-through로 프로그래밍됨
2. **기본 메모리 맵을 사용 중이고 그 맵에서 해당 주소가 write-through**
3. 캐시 가능 메모리이면서 **`CM7_CACR.FORCEWT` = 1**
4. 캐시 가능 + shared 메모리이면서 **`CM7_CACR.SIWT` = 1**

**조건 — 발생 시퀀스** [A]
1. 관심 주소가 D-Cache에 존재
2. 관심 주소와 **같은 doubleword**에 write-through store 수행
3. 다음 중 하나:
   - 관심 주소와 **같은 set/way**에 할당되는 (다른 cacheline 대상) **linefill 시작**
   - D-Cache 내 어디서든 **ECC 오류** 관측
   - **DSB 없는** data cache maintenance operation
4. 관심 주소에 store
5. 관심 주소에서 load

→ 특정 타이밍 조건이 맞으면 load가 **첫 번째 store의 데이터** 또는 **시퀀스 시작 시점의 캐시 내용**을 반환한다.

**회피책** [A]
> "There is no direct workaround for this erratum. Where possible, Arm recommends that you use the MPU to change the attributes on any Write-Through memory to Write-Back memory. If this is not possible, it might be necessary to disable the cache for sections of code that access Write-Through memory."

**이 보드에서의 실제 노출 지점 — ARMv7-M 기본 메모리 맵 캐시 정책 (ARM DUI 0646B Table 2-12)** [A]

| Address range | Memory region | Memory type | Shareability | **Cache policy** |
|---|---|---|---|---|
| `0x00000000–0x1FFFFFFF` | Code | Normal | Non-shareable | **WT** |
| `0x20000000–0x3FFFFFFF` | SRAM | Normal | Non-shareable | WBWA |
| `0x40000000–0x5FFFFFFF` | Peripheral | Device | Non-shareable | – |
| `0x60000000–0x7FFFFFFF` | External RAM | Normal | Non-shareable | WBWA |
| `0x80000000–0x9FFFFFFF` | External RAM | Normal | Non-shareable | **WT** |
| `0xA0000000–0xBFFFFFFF` | External device | Device | Shareable | – |
| `0xC0000000–0xDFFFFFFF` | External device | Device | Non-shareable | – |
| `0xE0000000–0xE00FFFFF` | Private Peripheral Bus | Strongly Ordered | Shareable | – |
| `0xE0100000–0xFFFFFFFF` | Vendor-specific device | Device | Non-shareable | – |

즉 **MPU 설정 없이 D-Cache만 켜면**:
- 내부 Flash(AXIM 경로 `0x08000000`)를 포함한 **Code 영역 전체가 WT** → 1259864 조건 영역. (다만 Flash 영역에 대한 store는 통상 발생하지 않아 실제 발현 확률은 낮음) [D — 표 [A] + 시퀀스 조건 [A] 로부터의 추론]
- `0x80000000–0x9FFFFFFF` (STM32F7에서 FMC NAND bank / QUADSPI memory-mapped 영역대)가 **WT**. [D]
- **SDRAM Bank1 `0xC0000000`은 기본적으로 Device 타입(캐시 불가)** 이다. 캐시를 쓰려면 MPU로 Normal/WB를 지정해야 하며, 그때는 WT를 피해 **WB로 지정**하는 것이 1259864 회피와 일치한다. [A]

> **실측 정정 — 이 보드는 "MPU 설정 없이 D-Cache만 켠" 상태가 아니었다**
>
> - 원래 SNN 펌웨어가 돌던 상태의 실측값은 **`MPU_CTRL = 0x00000005`(MPU **활성**, PRIVDEFENA=1)**, `SHCSR = 0x00010000`(MEMFAULTENA=1)이다.
>   따라서 위 Table 2-12의 기본 속성이 그대로 적용된다는 전제가 이 보드에는 **성립하지 않는다.**
>   다만 이번 덤프에는 `MPU_RBAR`/`MPU_RASR`가 없어 **각 리전이 WT인지 WB인지는 여전히 미확인**이다.
> - **`0xC0000000` = Device(캐시 불가)라는 마지막 줄은 실측으로 확인됐다**: MPU를 설정하지 않은 프로브 펌웨어에서
>   SDRAM 읽기가 **D-cache OFF 59.70 MB/s vs WB 59.71 MB/s**로 사실상 동일했다(캐시가 전혀 개입하지 않음).
>   같은 조건에서 SRAM1은 172.85 → 206.32 MB/s로 캐시 효과가 나타난다.
> - **실행 경로에 대한 중요한 보강**: 이 보드는 옵션바이트 **`BOOT_ADD0 = 0x00200000`(ITCM 인터페이스 별칭)** 에서 부팅한다
>   (실측 `FLASH_OPTCR1 = 0x00400080`, `VTOR = 0x00200000`). **ITCM 인터페이스 경유 접근은 D-cache 대상이 아니다** —
>   실측에서 ITCM 경로 성능이 D-cache ON/OFF에 전혀 반응하지 않았다(492.10 MB/s로 동일).
>   즉 이 경로로 실행되는 코드는 애초에 D-Cache에 들어가지 않으므로 **1259864의 Code 영역 WT 논의 대상이 아니다.**
> - `SYSCFG_MEMRMP = 0x00000000` → **SWP_FMC = 0b00**(스왑 미사용). 아래에서 소개하는 회피 수단은 현재 쓰이지 않고 있다.

**관련 참고 — `SYSCFG_MEMRMP.SWP_FMC[1:0]`** (RM0385 §SYSCFG) [A]
> *"These bits are used to swap the FMC SDRAM banks and FMC NOR/PSRAM **in order to enable the code execution from SDRAM Banks without modifying the default MPU attribute**"*
- `00`: SDRAM bank1/bank2 = `0xC000 0000` / `0xD000 0000` (기본), NOR/RAM = `0x6000 0000`
- `01`: 스왑 — SDRAM bank1/bank2 = **`0x6000 0000` / `0x7000 0000`**, NOR/PSRAM = `0xC000 0000`

→ 스왑을 쓰면 SDRAM이 기본 맵상 **WBWA(Normal)** 인 `0x60000000` 영역으로 이동하므로 MPU 재설정 없이 캐시 가능해지고, **WT가 아니라 WB**이므로 1259864의 조건에서 벗어난다. [D — RM0385 [A] + DUI 0646B Table 2-12 [A] 결합 추론]

---

### 5.4 그 밖의 주요 코어 에라타 상세 (r0p1 적용)

#### ARM 565285 — Core can send AXI transactions that permit reordering when it should not (Cat B, **Open**)
- **증상**: eviction용 ID를 쓴 write에 대해 slave가 `BVALID`를 assert하기 전에, 같은 주소에 대한 **cacheable data ID의 write**를 코어가 내보낼 수 있음. slave/인터커넥트가 재정렬하면 **오래된 eviction이 새 store를 덮어씀 → 데이터 손상**. `WDATA` 자체의 순서는 항상 정확. [A]
- **조건**: eviction 후 그 eviction에 포함된 주소에 대한 cacheable store. 시나리오 (a) Write-Back **no** Write-Allocate 속성 store / dynamic read allocate 모드 진입 / write-after-write hazard로 linefill 미개시, (b) dirty line에 대한 store가 캐시 룩업에서 **ECC 오류**를 만나 전체 라인 eviction 유발. [A]
- **영향 구성**: data cache를 가진 모든 구성. [A]
- **회피책**: cache ECC 미구성/비활성 시스템에서는 **Write-Back no Write-Allocate 사용 회피** 등. [A]

#### ARM 1013783 — PLD might perform linefill to address that would generate a MemManage Fault (Cat B, **Open**)
- **증상**: MPU가 활성이고 해당 주소가 미매핑/권한 부족이라 load 시 MemManage Fault가 나야 할 주소에 대해, **`PLD` 명령이 잘못하여 D-Cache linefill을 유발**. [A]
- **보안 영향 없음**: linefill 데이터는 PLD가 직접 소비하지 않으며, 후속 load는 여전히 권한 검사를 받는다. **권한 없는 데이터에 접근할 수 있게 되지는 않는다.** [A]
- **실제 영향**: **읽기에 부작용이 있는 메모리 영역**이나 **버스 응답을 반환하지 않는 메모리**에 대한 불필요한 접근. 분기 예측 실패 그림자에서 `PLD`가 투기적으로 실행될 수 있음(리터럴 값이 PLD로 디코딩되는 경우 포함). [A]
- **회피책**: MPU 영역 설정으로 미매핑 메모리 접근을 회피. [A]

#### ARM 2328489 — TCM bandwidth sharing (Cat B, **Open**) — *성능 전용*
- **증상**: TCU의 Store Queue(SQ)에서 software store와 AHBS write 사이 round-robin 대역폭 공유가 깨져, **AHBS write가 64-bit software store보다 우선권을 가져 파이프라인 stall** 유발. [A]
- **조건**: `STRD`, `STM`, `PUSH`, `VPUSH`, `VSTR.64`, `VSTM` 중 하나가 **TCM 대상**으로 실행 + **연속 back-to-back AHBS write 스트림** 진행 중 + **TCM wait state 사용**. 64-bit store 직후 load가 이어지면 악화(파이프라인 stall 중에도 read는 진행되나 데이터는 폐기되어 stall 해제까지 재시도). [A]
- **영향**: **성능만** 저하. 기능 오류 없음. [A]
- **회피책** (택일): [A]
  1. AHBS priority demotion — **`CM7_AHBSCR.CTL = 0b00`, `INIT = N`**
  2. 16회의 연속 SEQ/NONSEQ back-to-back write마다 **N회의 연속 BUSY 또는 IDLE 전송** 보장
  - 여기서 **N ≥ (T/2) + 1**, T = TCM 인터페이스에서 read 2회 + write 1회 시퀀스에 걸리는 최대 사이클 수.

> **실측 확인**: `CM7_AHBSCR`은 **STM32F746에서 접근 가능하다**(문서 말미 "미확인 #6" 해소).
> 실측값 `AHBSCR @0xE000EFA0 = 0x00000800` → **CTL = 0b00**, TPRI = 0, INITCOUNT = 1.
> 즉 **회피책 ①(AHBS priority demotion, CTL=0b00)이 리셋 기본값으로 이미 적용된 상태**다.
> 리셋 직후 덤프와 실행 중 덤프가 동일하므로 소프트웨어가 바꾼 값이 아니다.
> 다만 이 에라타의 조건인 "TCM wait state 사용" 여부와 실제 stall 발생은 **아직 측정하지 않았다.**
> 참고로 TCM 접근 지연 실측은 **5.16 cyc**(32 B 스트라이드, 루프 오버헤드 2.00 cyc 포함)로 AXI SRAM(15.66 cyc)의 1/3이다.

#### ARM 1313001 — Store after cache invalidate without intervening barrier (Cat C, **Open**)
- **증상**: cache invalidate → (배리어 없이) WT store → 해당 주소로 linefill 발생 시, **linefill이 store 데이터를 병합하지 않고 할당**되어 캐시에 stale data가 남음. store는 버스로만 나감. [A]
- **영향 구성**: data cache 있고 **ECC 없는** 모든 구성. [A]
- **조건 (해당 cache line 대상)**: `DCCIMVAC`, `DCCISW`, `DCIMVAC`, `DCISW` 중 하나 수행 → WT store → linefill. maintenance op와 store 사이에 **DSB/DMB가 없어야** 함. [A]
- **회피책**: maintenance operation과 store 사이에 **DMB 삽입**. (ARM 주석: cache maintenance와 store 사이에는 암묵적 순서가 없으므로 정상적인 코드라면 이미 DMB/DSB가 있어야 함) [A]

#### ARM 1267980 — ECC error + locked DEBR (Cat C, **Open**)
- **증상**: D-Cache의 error bank register `DEBR0`, `DEBR1`이 **둘 다 lock**된 상태에서 cacheable store가 ECC 오류를 만나면, store 데이터가 **버스로만 기록되고 D-Cache에는 기록되지 않아** 캐시에 stale data가 남음. [A]
- **회피책**: **두 error bank register를 동시에 lock하지 말 것.** [A]

#### ARM 1315869 — Data corruption for load following Store-Exclusive (Cat C, **Open**)
- **증상**: Load-Exclusive와 Store-Exclusive 사이에 끼어든 store가 있을 때, Store-Exclusive 직후의 load가 **Store-Exclusive가 아닌 앞선 store의 데이터**를 forwarding 받을 수 있음. [A]
- **조건**: `STREXB`/`STREXH`/`STREX` 이후 **최대 1개 명령**만 두고 load, 모든 접근이 **Shareable** 메모리. [A]
- **회피책**: **불필요** ("No workaround is necessary") — LDREX/STREX 사이의 store는 실제 코드에서 나타나지 않을 것으로 간주. [A]

#### ARM 443753 — Cacheable stores to always-bus-faulting locations → deadlock (Cat B rare, r0p2 수정)
- **증상**: 항상 bus fault를 반환하는 메모리에 대한 특정 cacheable store 시퀀스 후 **프로세서 deadlock**. [A]
- **회피책**: AXI에 영구 bus-abort 영역이 없는 시스템은 무관. 있다면 **MPU로 해당 영역 접근을 금지**. [A]

#### ARM 440977 — BASEPRI 우선순위 상승이 즉시 반영되지 않음 (Cat B, r0p2 수정)
- **증상**: `MSR BASEPRI`/`BASEPRI_MAX`로 실행 우선순위를 올려도 **직후 1개 명령 동안**은 새 우선순위보다 낮은 예외에 선점될 수 있음. 그 다음 명령부터는 정상. [A]
- **주의**: `PRIMASK`/`FAULTMASK`에 대한 `MSR`/`CPS`는 정상 동작. **BASEPRI 계열에만 해당.** [A]
- **회피책** [A]:
  ```
  CPSID i
  MSR   BASEPRI, <val>
  CPSIE i
  <critical region code>
  ```

#### ARM 426115 — FAULTMASK 설정 명령 인터럽트 시 MPU 명령 속성 오류 (Cat B, r0p2 수정)
- **조건**: MPU 활성 + **`MPU_CTRL.HFNMIENA = 0`** + 우선순위 > 0에서 실행 + `CPS`/`MSR`로 FAULTMASK 설정 시도 중 **NMI가 아닌 비동기 예외**가 인식되어 명령이 kill되고 FAULTMASK가 설정되지 않음. [A]
- **증상**: 해당 인터럽트 핸들러의 **모든 fetch MPU lookup이 HardFault/NMI 우선순위로 잘못 수행**됨 → 기본 메모리 맵 사용. 핸들러 종료 또는 다른 예외 발생 시까지 지속. [A]

#### ARM 423541 — bus-abort SO/Device load to SP + 인터럽트 → 잘못된 예외 스태킹 (Cat C, r0p2 수정)
- **증상**: bus abort를 받는 load-single 명령 실행 중 **다른 비동기 예외가 인식**되면 프로세서 상태가 부정확해질 수 있음. [A]
- **회피책**: 별도 회피책 불필요. 해당 시나리오를 코드에서 회피할 것. [A]

#### ARM 421025 — IT 블록 내 load early forwarding 취소 오류 (Cat C, r0p2 수정)
- **증상**: IT 블록 내 특정 명령 시퀀스가 **정수 레지스터에 잘못된 결과**를 만들 수 있음. [A]
- **회피책**: 중복 load 연산을 제거하도록 코드 시퀀스 수정. [A]

#### ARM 431216 — BASEPRI 미구현 비트가 RAZ 아님 (Cat C, r0p2 수정)
- **증상**: `BASEPRI`/`BASEPRI_MAX`의 미구현 비트가 0으로 읽히지 않고 **쓴 값 그대로** 읽힘. 읽기 값에만 국한되며 예외 우선순위 판정 등 동작 자체는 정상. [A]
- **회피책**: 다른 priority 레지스터로 구현 비트 수를 파악하고, 읽을 때 미구현 비트를 마스킹. [A]

#### ARM 449383 — FPCCR.ASPEN 쓰기 중 SP FP MAC 완료 시 FP 레지스터 뱅크 손상 (Cat C, r0p2 수정)
- **증상**: single-precision FP MAC 실행 중에 `FPCCR.ASPEN`을 변경해 자동 FP 상태 보존을 켜고 **기본 반올림 제어가 현재 설정과 다르면**, 후속 FP 명령이 FP 레지스터 파일에 잘못된 값을 쓸 수 있음. [A]
- **회피책**: 필요 시 `FPCCR` 쓰기 뒤에 **`DSB`, `ISB`** 를 넣고 후속 FP 명령 실행. [A]

#### ARM 486321 — 프로파일링 카운터 오동작 (Cat C, r1p1 수정)
- **증상**: 일부 이벤트가 **잘못된 이벤트 카운터를 증가**시킴. 메모리 시스템 stall 사이클이 `DWT_LSUCNT` 대신 **`DWT_CPICNT`** 를 증가시킴. [A]
- **회피책**: **없음.** → **성능 프로파일링 시 DWT 카운터 해석에 주의 필요.** [A]

> **실측 확인**: 이 에라타가 말하는 프로파일 카운터가 **이 실리콘에 실재한다.** `DEMCR.TRCENA`를 켠 뒤
> `DWT_CTRL = 0x40000001` → NUMCOMP=4, **NOCYCCNT=0**(CYCCNT 있음), **NOPRFCNT=0**(CPI/EXC/SLEEP/LSU/FOLD 있음).
> 따라서 **DWT CYCCNT 기반 사이클 측정은 신뢰할 수 있지만, LSUCNT/CPICNT의 분해는 이 에라타 때문에 신뢰하면 안 된다.**
> (오계수 자체는 아직 재현 측정하지 않았다.)
> 주의: TRCENA를 켜기 전에는 DWT 레지스터가 전부 0으로 읽히므로 "DWT 없음"으로 오인하기 쉽다 — 실제로 첫 ST-LINK 덤프에서 그렇게 보였다.

#### ARM 702596 / 3092511 / 513195 / 1518990 / 636315 (디버그·트레이스 계열)
| ID | 요지 | 회피책 |
|---|---|---|
| 702596 (r0p2 수정) | `DHCSR.C_MASKINTS = 1`로 마스킹해야 할 인터럽트가 Debug state 이탈 시 taken될 수 있음 → **싱글스텝 중 예외 핸들러 진입** | 완전한 회피책 없음. 디버거가 halt 감지 즉시 C_MASKINTS 설정 [A] |
| 3092511 (**Open**) | 비동기 예외와 breakpoint가 동시 발생 시 **breakpoint 주소가 아닌 예외 핸들러 시작 주소에서 halt** | 문서상 회피책 기재 없음 [A] |
| 513195 (r1p1 수정) | 외부 디버거 읽기 시 `FPB_LSR`/`DWT_LSR`/`ITM_LSR`의 LSI/SLK가 잘못 1로 읽힘 | 디버거가 SLI 필드 무시, LAR 쓰기 금지 [A] |
| 1518990 (**Open**) | big-endian 시스템에서 `DWT_COMPn` 데이터 값 비교가 memory-endianness 형식 | big-endian 값으로 프로그래밍 [A] |
| 636315 (**Open**) | on-line MBIST의 I-Cache 접근 시 SW 프로그래밍 오류가 `MBISTERR`로 보고되지 않음 (메모리 결함 검출 자체는 영향 없음) | 없음 [A] |
| 412512 (r0p2 수정) | ETM 있고 ITM 없는 구성에서 TPIU가 **최대 15 bytes** 트레이스 데이터 유실 | 없음 [A] |
| 408519 (r0p2 수정) | ITM GTS 활성 시 GTS1 패킷만 연속 출력되고 GTS2 미출력 | GTS 활성화 직후 `ITM_TCR`에 dummy write (활성화 쓰기를 연속 2회) [A] |
| 399743 (r0p2 수정) | `CCR.BFHFNMIGN` 설정 시 FAR(BFAR/MMFAR 공용 물리 레지스터) 손상 가능 | BFHFNMIGN은 privileged 모드에서만 설정, 해당 상태에서 fault 가능 주소로의 load-to-PC 회피 [A] |
| 416915 (r0p2 수정) | LOCKUP으로 이어지는 configurable priority fault에서 **`HFSR.FORCED`가 set되지 않음** | 없음 [A] |
| 422825 (r0p2 수정) | 예외 복귀 후 일시적으로 MPU fetch 속성이 부정확 | 회피책 불필요 [A] |

> **실측 확인 (트레이스 계열 에라타의 적용 대상이 실재하는가)**
> - **TPIU와 ETM은 이 실리콘에 물리적으로 존재한다**: `0xE0040000`의 PIDR part = **0x9A9 (Cortex-M7 TPIU)**,
>   `0xE0041000`의 PIDR part = **0x975 (Cortex-M7 ETM)**, 둘 다 JEP106 제조사 0x3B(ARM), CIDR 클래스 0x9.
>   따라서 **412512(ETM 있고 ITM 없는 구성에서 트레이스 유실)와 ES0290 §2.1이 참조하는 ETM-M7(TM975) 에라타는 적용 대상이다.**
> - 다만 **CoreSight ROM table은 두 유닛을 present=0으로 광고**하며(`entry[4]=0xFFF41002`, `entry[5]=0xFFF42002`),
>   `DEMCR.TRCENA`를 켠 뒤 다시 읽어도 present 비트는 0 그대로다. 디버거가 ROM table만 훑으면 **"트레이스 없음"으로 판단**한다.
> - 현재 구성에서는 발현하지 않는다: `DBGMCU_CR = 0x00000000` → **TRACE_IOEN = 0**(트레이스 출력 핀 비활성).
>   SWO/ETM 트레이스를 쓰려면 이 비트부터 켜야 하고, 그때 위 에라타들이 실효를 갖는다.
> - **1518990(big-endian DWT 데이터 비교)은 해당 없음**: 실측 `AIRCR.ENDIANNESS = 0`(리틀엔디언).
> - **702596 / 3092511(싱글스텝·브레이크포인트)** 은 디버그 세션 동작이라 이번 측정 범위 밖이다.
>   참고로 하드웨어 브레이크포인트는 실측 **8개**(`FP_CTRL = 0x10000080`, NUM_CODE=8, NUM_LIT=0)다.

---

## 6. System 에라타 (ES0290 §2.2)

### 2.2.1 Missed ADC triggers from TIM1/TIM8, TIM2/TIM5/TIM4/TIM6/TRGO or TGRO2 event — 상태 A
- **증상**: TIM1/TIM8/TIM2/TIM5/TIM4/TIM6의 TRGO/TRGO2 이벤트에 의한 regular/injected 채널 ADC 외부 트리거가 **누락**. [A]
- **조건 (아래 중 하나)** [A]
  - **PCLK2에 prescaler가 enable**된 경우
  - **TIMxCLK = 2 × ADCCLK** 이고 `TIMx_CR2`의 `MMS`/`MMS2`가 reset / update / compare pulse 설정
  - **TIMxCLK = 4 × ADCCLK**
- **회피책** [A]
  - TIM1/TIM8 TRGO/TRGO2: **양 에지 트리거 검출** — `ADC_CR2`의 `EXTEN[1:0]` 또는 `JEXTEN[1:0]`를 **`0x11`** 로 설정
  - TIM2/TIM4/TIM5/TIM6 TRGO/TRGO2: **`RCC_APB1ENR`에서 DAC 페리페럴 클럭 enable**

### 2.2.2 Internal noise impacting the ADC accuracy — 상태 A
- **증상**: **VDD 공급에서 발생한 내부 노이즈**가 내부 전파되어 **ADC 정확도**에 영향. **MCU 전원 모드(Run/Sleep)와 무관하게 항상 활성.** [A]
- **회피책 (둘 중 하나만 설정 가능)** [A]
  - Option1: **`PWR_CR.ADCDC1` = 1**
  - Option2: **`SYSCFG_PMC.ADCxDC2`** (해당 ADC 비트) = 1
  - 상세는 **AN4073** 참조.

> **실측 확인 — 이 보드에는 두 회피책 중 어느 것도 적용돼 있지 않다.**
> `PWR_CR1 = 0x0003C000` → **ADCDC1(bit 13) = 0**, `SYSCFG_PMC = 0x00000000` → **ADCxDC2 = 0**.
> 이 상태에서 온칩 ADC로 측정한 값이 **VDDA 3.311~3.313 V**(VREFINT 코드 1486~1487),
> **다이 온도 35.7~39.2 °C**(온도센서 코드 957~968)이므로, 이 수치들에는 에라타가 말하는 내부 노이즈가 포함돼 있다.
> 64회 평균을 취했으나 회피책 적용 전후 비교는 하지 않았다.

### 2.2.3 Wakeup from Standby mode when the back-up SRAM regulator is enabled — 상태 A
- **증상**: `PWR_CSR1` 쓰기로 back-up SRAM regulator를 enable/disable할 때 **`EIWUP` 비트가 0으로 덮어써지면**, RTC wakeup 이벤트(alarm, RTC Tamper, RTC TimeStamp, RTC wakeup timer)가 **Standby에서 시스템을 깨우지 못함**. [A]
- **회피책**: back-up SRAM regulator 제어를 위한 **`PWR_CSR1` 쓰기마다 `EIWKUP` 비트를 1로 설정**. [A]

### 2.2.4 Full JTAG configuration without NJTRST pin cannot be used — 상태 A
- **증상**: JTAG 디버그 포트 사용 중 **NJTRST 핀(PB4)을 GPIO로 사용하면 디버거 연결이 끊어짐**. **4-wire JTAG 구성에만** 영향. [A]
- **회피책**: **4-wire full JTAG 대신 SWD 디버그 포트 사용.** [A]
- (본 보드는 온보드 ST-LINK가 SWD를 사용하므로 실질 영향 없음. [D])

---

## 7. FMC 에라타 (ES0290 §2.3) — 외부 SDRAM/NOR 관련

### 2.3.1 Dummy read cycles inserted when reading synchronous memories — 상태 **N (회피책 없음)**
- **증상**: 동기식 메모리에서 **burst read** 수행 시, burst 종류와 무관하게 **burst cycle 끝에 dummy read 접근 2회**가 추가로 수행된다. 추가로 읽힌 데이터는 FMC가 사용하지 않으며 **기능적 오류는 없다.** [A]
- **회피책**: **None.** [A]
- **영향**: 순수 **대역폭 손실**. SDRAM 프레임버퍼/가중치 스트리밍처럼 burst read가 지배적인 워크로드에서 유효 대역폭이 감소한다. [D — 문서 기술 [A]로부터의 성능 해석]

> **실측 보강 (아직 이 에라타 몫을 분리하지 못했다)**: 이 보드의 SDRAM 실측 대역폭은
> **순차 읽기 60.25 MB/s / 쓰기 194.6 MB/s / SDRAM→SRAM1 복사 58.4 MB/s**로, 컨트롤러 설정에서 계산한
> 이론 최대 200 MB/s(16비트 × 100 MHz SDR)의 **읽기 30 %** 수준이다.
> 다만 **MPU 미설정 시 `0xC0000000`은 Device 메모리라 코어가 버스트를 내지 않는다**(실측: D-cache ON/OFF 차이 없음).
> 즉 지금 관측되는 읽기 손실은 대부분 "버스트가 없어서"이지 dummy read 2회 때문이라고 단정할 수 없다.
> **에라타 몫을 보려면 MPU로 SDRAM을 Normal cacheable로 지정해 32 B 라인필 버스트가 걸리게 한 뒤 다시 재야 한다.**

### 2.3.2 Wrong data read from a busy NAND memory — 상태 A
- **증상**: NAND에 read 명령 후 chip select de-assert 시점에 R/B 신호가 활성화되는데, read 트랜잭션이 pending이면 NAND 컨트롤러가 이전에 assert된 **R/B(NWAIT 연결) 신호를 감지하지 못하고 잘못된 데이터를 샘플링**. [A]
- **조건**: **`MEMSET` 타이밍 = 0x00**, 또는 **`ATTHOLD` 타이밍 = 0x00 또는 0x01**. [A]
- **회피책**: `MEMSET` > 0x00 **또는** `ATTHOLD` > 0x01 로 설정. [A]

### 2.3.3 Spurious clock stoppage with continuous clock feature enabled — 상태 A
- **증상**: continuous clock 기능 활성 시 **`FMC_CLK`가 갑자기 정지**할 수 있음. [A]
- **조건**: **FMC_CLK 분주비 = 2** AND **32-bit로 설정된 FMC bank에 byte access**. [A]
- **Note**: static memory의 경우, 정지된 클럭은 synchronous transaction 또는 32-bit 데이터 버스 폭에 대한 byte access가 아닌 asynchronous transaction을 발행하면 재시작 가능. [A]
- **회피책**: continuous clock 활성 시, **32-bit asynchronous 메모리를 byte access할 때 FMC_CLK 분주비를 2로 설정하지 말 것.** [A]

### 2.3.4 Data read might be corrupted when the write FIFO is disabled — 상태 A
- **증상**: write FIFO 비활성 시 **모든 write access마다 FIFO empty 이벤트**가 생성된다. write access 중 새 read access가 오면 FMC는 read를 허가하고 FIFO가 빌 때까지 대기하는데, **FIFO empty 이벤트로부터 1 사이클의 매우 짧은 윈도우** 안에 또 다른 read access가 오면 **반환 데이터가 손상**된다. [A]
- **조건**: **`FMC_BCR1` 레지스터의 `WFDIS` 비트가 set** (write FIFO 비활성). [A]
- **회피책**: **write FIFO를 enable** (`WFDIS = 0`). [A]

> **실측 확인**: 이 보드는 **이미 안전한 쪽에 있다.** `FMC_BCR1 @0xA0000000 = 0x000030DB`(리셋 기본값 그대로) →
> **WFDIS(bit 21) = 0 = write FIFO 활성**. ST BSP의 SDRAM 초기화는 `FMC_BCR1`을 건드리지 않는다.
> 이 값을 바꾸는 코드를 추가하지 않는 한 §2.3.4 조건에 들어가지 않는다.

---

## 8. QUADSPI 에라타 (ES0290 §2.4)

| 절 | 증상 | 조건 | 회피책 |
|---|---|---|---|
| 2.4.1 | read transfer 끝에서 FIFO가 가득 차는 순간 데이터를 읽으면 **여분 데이터 1개가 FIFO에 부당하게 기록** | indirect mode **AND** QUADSPI clock = AHB/2 (`QUADSPI_CR.PRESCALER = 0x01`) **AND** quad mode (`QUADSPI_CCR.DMODE = 0b11`) **AND** DDR mode (`QUADSPI_CCR.DDRM = 1`) | ① BUSY 플래그가 low가 될 때까지 여분 데이터를 읽어 폐기, 또는 ② 기대 데이터 수신 후 `QUADSPI_CR.ABORT` set으로 FIFO flush(레지스터 설정은 유지) |
| 2.4.2 | 외부 Flash에 쓸 **데이터의 첫 nibble이 유실** | indirect **write** mode **AND** dummy cycle **1개 이상** 사용 | dummy phase 대신 **alternate byte** 사용(대체할 dummy cycle 수가 8비트 데이터의 배수일 때만 유효). 예: 1 dummy cycle → DDR 4-line에서 alternate byte 1개 / 2 dummy → SDR 4-line 1 byte / 4 dummy → SDR 4-line 2 byte 또는 SDR 2-line 1 byte / 8 dummy → SDR 1-line 1 byte |
| 2.4.3 | memory-mapped 진입 시 `QUADSPI_AR`의 `ADDRESS[1:0]`가 **둘 다 set**이면 **첫 memory-mapped read가 잘못된 데이터** 반환 | 위 조건 | memory-mapped 진입 전 — indirect **read**: `QUADSPI_AR` clear 후 abort 요청으로 읽기 중단 및 BUSY clear / indirect **write**: `QUADSPI_AR` clear. **주의: `QUADSPI_AR` clear 후 `QUADSPI_DR`에 쓰면 안 됨** |
| 2.4.4 | memory-mapped 모드에서 **QUADSPI hang**, read 실패 | `QUADSPI_CR.TCEN = 1`(timeout counter 활성) **AND** `TOF` 플래그 set과 새 memory-mapped read 요청이 **같은 클럭 에지**에 발생 | **timeout counter 비활성화.** chip select를 올리려면 각 memory-mapped read 종료마다 abort 수행 |

(모두 [A], ES0290 Rev 7 §2.4.1–2.4.4)

> **실측 확인 (이 보드의 QUADSPI 설정)**: `QUADSPI_CR = 0x01400311`, `DCR = 0x00170500`, `CCR = 0x0F282DEB`
> - **§2.4.4 조건 미해당**: **TCEN(bit 3) = 0** — 타임아웃 카운터가 꺼져 있다(ST BSP 기본).
> - **§2.4.1 조건 부분 해당 주의**: **PRESCALER = 0x01(= AHB/2)** 이고 **DMODE = 4-line**이다.
>   다만 **DDRM = 0(SDR)** 이고 현재는 **FMODE = 0b11(메모리맵 모드)** 이므로 indirect + DDR 조건은 성립하지 않는다.
>   **indirect 모드로 DDR 읽기를 추가하는 순간 이 에라타의 조건이 전부 갖춰진다** — 주의할 것.
> - 부품 관련 추가 사실: 이 보드의 QSPI NOR은 ST BSP가 전제한 **N25Q128A가 아니라 Winbond W25Q128 계열**이다
>   (Read JEDEC ID 실측 `0xEF4018`; Micron이면 `0x20BA18`). 에라타 회피책은 MCU 쪽이라 영향이 없지만,
>   **쓰기/소거 시퀀스는 Winbond 데이터시트로 재검증해야 한다.**
> - 인터페이스 클럭은 **100 MHz**로, 데이터시트 QUADSPI 최대치 80 MHz를 넘긴 상태로 동작 중이다(기능은 정상).

---

## 9. ADC / DAC / LPTIM / RTC (ES0290 §2.5–2.8)

| 절 | 항목 | 요지 | 회피책 |
|---|---|---|---|
| 2.5.1 | ADC sequencer modification during conversion (A) | **소프트웨어(`SWSTART`)로 시작된** 변환 중 `ADC_SQRx`/`ADC_JSQRx`를 수정하면 현재 변환이 리셋되고 **새 시퀀스를 자동 재시작하지 않음**. **하드웨어 트리거로 시작된 경우에는 해당 없음**(자동 재시작됨) | 소프트웨어로 시작한 경우 `ADC_CR2.SWSTART`를 다시 set해야만 재시작 |
| 2.6.1 | DMA underrun flag management (A) | DMA가 충분히 빠르지 않으면 **같은 디지털 데이터가 2회 변환**되고 `DMAUDR` 플래그가 set. 그런데 **`DMAUDR=1`이 DMA를 비활성화하지 않고** 계속 DAC에 서비스 | DAC DMA 스트림을 끄려면 `DMA_SxCR`의 해당 **`EN` 비트를 직접 reset** |
| 2.6.2 | DMA request not automatically cleared by clearing DMAEN (A) | `DAC_CR.DMAEN` clear 또는 DAC 클럭 비활성으로 **DMA 요청이 자동 clear되지 않음** → DAC 재활성 시 **spurious DMA 요청 서비스 위험** | 순서 준수: ① `DAC_CR.DMAUDR` 확인 → ② DAC 채널 `DMAEN` clear → ③ DAC 클럭 비활성 → ④ DAC/DMA/트리거 재구성 → ⑤ 재시작 |
| 2.7.1 | MCU may remain stuck in LPTIM interrupt when entering Stop mode (A) | LPTIM 인터럽트 발생 근처의 좁은 시간 창에서 `LPTIM_CR.ENABLE`을 clear하면 **wakeup용 LPTIM 인터럽트 신호가 active 상태로 고착** → Stop 진입 실패, ISR에 갇힘. **모든 Stop 모드/모든 LPTIM 인스턴스**에 적용. 발생 확률은 매우 낮음 | `ENABLE` 비트를 clear하지 말고, **RCC의 `LPTIMxRST` 비트(`RCC_APByRSTRz`)를 set/reset하여 페리페럴 전체 리셋** |
| 2.8.1 | RTC calendar registers are not locked properly (**P**) | `BYPSHAD = 0`에서 `RTC_SSR` 읽은 뒤 `RTC_TR`/`RTC_DR`이 lock되지 않을 수 있음(shadow register 갱신 1 APB 클럭 전에 읽기 시작한 경우) → **세 레지스터 간 불일치** | ① **`BYPSHAD = 1`** 사용, 또는 ② `RTC_SSR`, `RTC_TR`, `RTC_DR` 읽은 뒤 `RTC_SSR`을 **재확인**하여 동일하지 않으면 재독 |
| 2.8.2 | Spurious tamper detection when disabling the tamper channel (**N**) | falling edge 검출 설정(`TAMPFLT=00`, `TAMPxTRG=1`)에서 **tamper 핀이 high일 때 tamper 검출을 비활성화하면 가짜 tamper 이벤트** 발생 | **None** |

(모두 [A])

---

## 10. I2C 에라타 (ES0290 §2.9)

### 2.9.1 Wrong data sampling when tSU;DAT is shorter than one I2C kernel clock period — 상태 A
- I2C-bus 규격 최소 데이터 셋업 시간(tSU;DAT): **Standard 250 ns / Fast 100 ns / Fast-mode Plus 50 ns**. [A]
- **증상**: tSU;DAT가 **I2C kernel clock 1주기보다 짧으면** SDA를 잘못 샘플링하여 **직전 SDA 값**을 읽음 → slave address / data byte / ACK 비트 오수신. [A]
- **회피책 — 최소 I2CCLK 주파수** [A]

| 모드 | 송신기 최소 셋업 시간 | **최소 I2CCLK 주파수** |
|---|---|---|
| Standard mode | 250 ns | **≥ 4 MHz** |
| Fast mode | 100 ns | **≥ 10 MHz** |
| Fast-mode Plus | 50 ns | **≥ 20 MHz** |

### 2.9.2 Spurious bus error detection in master mode — 상태 A
- **증상**: master 모드에서 bus error가 **가짜로 검출**되어 `I2C_SR.BERR` 플래그 set 및 (활성 시) 인터럽트 발생. **실제 전송에는 영향 없으며 정상 진행**된다. [A]
- **회피책**: BERR 인터럽트 발생 시 **소프트웨어로 BERR 플래그만 clear**. 다른 조치 불필요, 진행 중 전송은 정상 처리. [A]

### 2.9.3 10-bit master mode: 첫 주소 바이트 NACK 시 새 전송 불가 — 상태 A
- **증상**: 10-bit 주소의 **첫 바이트(5-bit header + 주소 상위 2비트 + direction bit)** 가 NACK되면 MCU는 STOP을 정상 생성하나 **이후 어떤 I2C 전송도 시작할 수 없음**. 이 상태에서 `I2C_ISR.NACKF`와 `I2C_CR2.START`가 **동시에 set**되어 있음(START는 정상적으로 clear되어야 함). [A]
- **회피책** [A]: ① STOP 검출 대기(`I2C_ISR.STOPF = 1`) → ② I2C 페리페럴 disable → ③ **최소 3 APB 사이클 대기** → ④ I2C 페리페럴 재enable

### 2.9.4 Last-received byte loss in reload mode — 상태 A
- **조건 (전부 충족)** [A]: master receiver 또는 `SBC=1` slave receiver + **clock stretching 활성(`NOSTRETCH = 0`)** + `I2C_CR2.RELOAD = 1` + `NBYTES = N > 1` + byte N 수신으로 `TCR` 플래그 상승 + **그 시점에 byte N-1이 아직 data register에서 읽히지 않음**
- **증상**: SCL이 low로 당겨지고(clock stretching), byte N-1을 읽고 NBYTES를 재로드할 때까지 **shift register → data register 전송이 억제**. **독립 클럭을 쓰는 I2C 인스턴스**에서는, byte N의 마지막 데이터 비트 수신 **직전 4 APB 클럭 사이클 이내**에 데이터 N-1을 읽으면 **마지막 수신 데이터가 영구 유실**된다. [A]
- **회피책** [A]
  - master 또는 `SBC=1` slave: reload mode를 **`NBYTES = 1`** 로 사용
  - master receiver에서 255 byte 초과 전송: reload mode 대신 **255 byte 이하 섹션으로 분할 + repeated START**
  - DMA 등으로 **byte N-1을 항상 TCR 상승 전에** 읽도록 보장(독립 클럭 인스턴스는 **마지막 데이터 비트 수신 4 APB 클럭 이전**). 단 이 방법은 버스 속도·인터럽트 관리·SW 지연·DMA 우선순위에 좌우되므로 애플리케이션별로 신중히 검토 필요

---

## 11. USART / SPI·I2S / SDMMC1 / BxCAN (ES0290 §2.10–2.13)

| 절 | 항목 | 요지 / 조건 | 회피책 |
|---|---|---|---|
| 2.10.1 | Start bit detected too soon (smartcard NACK) (**N**) | ISO7816에서 smartcard는 Start bit falling edge 후 **(10.5 ± 0.2) etu**에 NACK 전송, USART는 **(11.0 ± 0.2) etu**에 샘플링해야 함. STM32 USART는 이 타이밍을 지키지 못해 **NACK falling edge가 10.68 etu 이후**에 도달하면 NACK을 올바로 검출하더라도 **Start bit로 오인**할 수 있음 | **None** |
| 2.10.2 | Break request can prevent TC flag from being set (A) | 조건 전부 충족 시 TC 미설정: **CTS 하드웨어 흐름제어 활성** + D1 전송 중 + **D1 전송 종료 전 break 전송 요청** + **D1 전송 종료 전 nCTS de-assert** | 전송 종료 검출이 필요하면 **TC 플래그가 set된 것을 확인한 뒤** break를 요청 |
| 2.10.3 | nRTS is active while RE or UE = 0 (A) | `RTSE` 비트를 set하는 즉시 **nRTS 라인이 low로 구동**됨 — USART 비활성(`UE=0`)이거나 수신기 비활성(`RE=0`)이어서 수신 준비가 안 된 상태에서도 | **`UE`와 `RE`를 set한 뒤에** nRTS용 I/O를 alternate function으로 구성 |
| 2.11.1 | I2S slave PCM short frame WS/CK 타이밍 민감 (A) | `I2SMOD=1`, `I2SCFG[1:0]=00 or 01`, `I2SSTD[1:0]=11`, `PCMSYNC=0`, **`ASTREN=0`(asynchronous start 비활성)** 상태에서 master가 WS hold time을 지키지 않으면 **송수신 데이터가 1비트 shift되어 desync** → 클럭 극성과 무관하게 **1개가 아닌 2개 프레임 폐기** | legacy mode(**`ASTREN=1`**) 사용 + master/slave 간 완전 동기 보장 |
| 2.11.2 | BSY bit may stay high at end of transfer in Slave mode (A) | slave 모드 데이터 전송 종료 시 **BSY 플래그가 산발적으로 high 유지**(내부 CPU 클럭과 외부 SCK의 우연한 일치). **BSY는 전송 종료 검출에 신뢰할 수 없음** | ① NSS 하드웨어 관리 + master가 NSS 제공 시 **NSS 플래그** 사용, ② SPI 수신 모드에서는 **RXNE** 사용, ③ transmit-only 모드에서는 **BSY + timeout 만료** 병용(마지막 데이터 프레임 예상 시간보다 길게, 마지막 프레임 2번째 비트에서 발생하는 TXE 이벤트에 시작). ①②를 우선. 대안 시퀀스: 마지막 데이터 write → TXE high 대기 → **프레임 전송 종료 전에 반드시 `SPE` clear** → BSY low 대기 |
| 2.12.1 | Wrong CCRCFAIL status after a response without CRC (A) | 응답에 CRC 필드가 없어도 CRC가 계산되어, **SDIO `IO_SEND_OP_COND` (CMD5)** 이후 `SDIO_STA.CCRCFAIL`이 set됨 | 소프트웨어가 **CCRCFAIL을 무시**하고, CMD5 응답 수신 후 `SDIO_ICR.CCRCFAILC`로 clear |
| 2.12.2 | MMC stream write of less than 8 bytes (A) | `WRITE_DAT_UNTIL_STOP` (CMD20) 스트림 write에서 `WAITPEND`(`SDMMC_CMD` bit 9) 설정 시 **잔여 50 데이터 비트** 시점까지 CMD12 전송이 pending. **8 byte 미만**이면 CMD12를 데이터 전송 시작 전에 보내야 하나 동시 시작되어, **(8 − DATALENGTH) byte가 0xFF로 기록**됨 | **DATALENGTH < 8 byte인 CMD20 스트림 write를 쓰지 말 것.** `SET_BLOCKLEN`(CMD16) + `WRITE_BLOCK`(CMD24) 사용 |
| 2.13.1 | BxCAN time triggered mode not supported (**N**) | 레퍼런스 매뉴얼에 기술된 **time triggered communication mode 미지원**, time stamp 값 사용 불가 | **`CAN_MCR.TTCM` 비트를 항상 clear로 유지.** 별도 회피책 **None** |

(모두 [A])

---

## 12. Ethernet 에라타 (ES0290 §2.14)

### 2.14.1 IPv6 패킷(TCP/UDP/ICMP payload 없음)에 잘못된 L3 체크섬 삽입 — 상태 A
- **증상**: 자동 체크섬 삽입 활성 상태에서 **TCP/UDP/ICMP payload가 없는 IPv6 패킷**이 입력되면, MAC이 next header(NH) 필드를 extension header로 간주해 계속 파싱하다가 payload 데이터가 우연히 TCP/UDP/ICMP의 NH 값과 일치하면 **체크섬을 삽입**해버림. [A]
- **회피책**: IPv6 패킷이 TCP/UDP/ICMP payload를 가질 때만 송신 체크섬 삽입을 enable하거나, **`TDES0`의 `CIC`(checksum insertion control) 비트 [23:22]** 로 체크섬 삽입을 bypass. [A]

### 2.14.2 수신 IPv6 프레임의 유효하지 않은 extension header 처리 — 상태 **N**
- **증상**: IPv6 extension header 중 **hop-by-hop options header는 main header 직후에만 와야** 하는데, MAC은 첫 extension header 이후에 나타나는 hop-by-hop header까지 **유효/무효 구분 없이 모두 처리**한다. 결과적으로 **임의의 IP payload를 유효한 TCP/UDP/ICMP payload IPv6 프레임으로 인정**하고 해당 프레임의 수신 상태를 잘못 갱신. [A]
- **회피책**: **None.** [A]

### 2.14.3 전송 완료 정확히 1 클럭 사이클 후 TxFIFO flush 명령 → MAC Idle 고착 — 상태 A
- **증상**: TxFIFO read controller가 **MAC으로부터 status를 받은 정확히 1 클럭 사이클 후**에 TxFIFO flush 명령을 받으면 **Idle 상태(`ETH_MACDBGR`의 `TFRS=00`)에 고착**되어 TxFIFO에서 프레임 전송을 중단한다. **리셋(예: 소프트 리셋)으로만 복구 가능.** [A]
- **회피책**: **TxFIFO flush 기능을 사용하지 말 것.** 반드시 필요하면 **TxFIFO가 빈 후에** flush 명령 사용. [A]

### 2.14.4 Transmit frame data corruption — 상태 A
- **증상**: TxFIFO가 **non-empty → empty → non-empty로 매우 짧은 주기로 반복 전이**할 때(underflow는 발생하지 않음) 프레임 데이터가 손상. TxFIFO에 데이터가 쓰이는 속도가 읽히는 속도와 거의 같거나 약간 느릴 때 발생. **MAC이 CRC를 삽입하므로 손상된 데이터로 CRC가 계산되어 수신 측에서 검출 불가.** [A]
- **회피책**: **Store-and-Forward 모드 사용 — `ETH_DMAOMR`의 `TSF` = 1 (bit 21).** 전체 패킷이 TxFIFO에 준비된 후에만 전송. [A]

### 2.14.5 동일 레지스터 연속 쓰기가 완전히 반영되지 않음 — 상태 A
- **증상**: 같은 레지스터에 대한 이전 쓰기로부터 **4 TX_CLK/RX_CLK 클럭 사이클 이내**에 다시 쓰면 완전히 반영되지 않을 수 있음. 이때 **레지스터를 읽으면 최신 쓴 값이 보이지만, Ethernet MAC은 최신 쓰기가 없었던 것처럼 동작**한다. [A]
- **회피책 (2가지)** [A]
  - 동일 레지스터 연속 쓰기 사이에 **4 TX_CLK/RX_CLK 클럭 사이클 지연** 보장
  - 지연 없이 연속 쓰기 후, 모든 작업 완료 시 레지스터를 읽고 **4 클럭 사이클 지연 후 재프로그래밍**

**영향 레지스터/비트 (ES0290 Table 5)** [A]

| 그룹 | 레지스터 | 비트 | 비트명 |
|---|---|---|---|
| DMA | `ETH_DMABMR` | 7 | EDFE |
| DMA | `ETH_DMAOMR` | 26 / 25 / 20 / 7 / 6 / 4:3 | DTCEFD / RSF / FTF / FEF / FUGF / RTC |
| GMAC | `ETH_MACCR` | 25, 23, 22, 19:17, 16, 14, 13, 12, 11, 10, 9, 7, 6:5, 4, 3, 2 | CSTF, WD, JD, IFG, CSD, FES, ROD, LM, DM, IPCO, RD, APCS, BL, DC, TE, RE |
| GMAC | `ETH_MACFFR` | – | MAC frame filter register 전체 |
| GMAC | `ETH_MACHTHR` / `ETH_MACHTLR` | 31:0 | Hash Table High / Low Register 전체 |
| GMAC | `ETH_MACFCR` | 31:16, 7, 5:4, 3, 2, 1, 0 | PT, ZQPD, PLT, UPFD, RFCE, TFCE, FCB/BPA |
| GMAC | `ETH_MACVLANTR` | 16, 15:0 | VLANTC, VLANTI |
| GMAC | `ETH_MACRWUFFR` | – | 모든 remote wakeup 레지스터 |
| GMAC | `ETH_MACPMTCSR` | 31, 9, 2, 1, 0 | WFFRPR, GU, WFE, MPE, PD |
| GMAC | `ETH_MACA0HR/LR` ~ `ETH_MACA3HR/LR` | – | MAC address 0~3 high/low register 전체 |
| IEEE 1588 | `ETH_PTPTSCR` | 18, 17:16, 15, 14, 13, 12, 11, 10, 9, 8, 5, 3, 2, 1, 0 | TSPFFMAE, TSCNT, TSSMRME, TSSEME, TSSIPV4FE, TSSIPV6FE, TSSPTPOEFE, TSPTPPSV2E, TSSSR, TSSARFE, TSARU, TSSTU, TSSTI, TSFCU, TSE |

---

## 13. ES0290에 **없는** 항목 (중요)

ES0290 Rev 7 전체를 확인한 결과, 다음 기능 블록에 대한 **에라타 항목은 존재하지 않는다**. [A]

| 블록 | ES0290 Rev 7 수록 여부 |
|---|---|
| **USB OTG_FS / OTG_HS** | **없음** |
| **DMA1 / DMA2 컨트롤러** | **없음** (DAC의 DMA 사용 관련 §2.6.1/§2.6.2만 존재) |
| **Flash 인터페이스 / ART Accelerator / prefetch** | **없음** |
| **캐시 일관성(D-Cache ↔ DMA)** | **없음** — §2.1.1(write-through)만 캐시 관련 |
| **LTDC / DMA2D (Chrom-ART)** | **없음** |
| **RCC / PLL / 클럭** | **없음** |
| **TIM (타이머 자체)** | **없음** (ADC 트리거 관련 §2.2.1만) |
| **CRC / RNG / HASH / CRYP** | **없음** |

> **실측 정정 — "에라타 없음"이 "조심할 것 없음"은 아니다**
>
> - **Flash 인터페이스 / ART Accelerator 행**: 에라타는 없지만, 이 보드는 **ART와 프리페치가 꺼진 채로 동작하고 있었다**
>   (`FLASH_ACR = 0x00000006` → LATENCY=6, **ARTEN=0, PRFTEN=0**). 게다가 옵션바이트 `BOOT_ADD0 = 0x00200000`으로
>   **ITCM 별칭에서 부팅**하는데, 그 경로의 가속은 전적으로 ART가 담당한다. 원인은 소스에 있다:
>   `stm32f7xx_hal_conf.h`의 `PREFETCH_ENABLE 0U` + `ART_ACCLERATOR_ENABLE` 매크로 **미정의** →
>   `HAL_Init()`의 `#if (ART_ACCLERATOR_ENABLE != 0)` 블록이 컴파일되지 않는다.
>   실측 손실: ITCM 경로 플래시 접근 **336.79 → 492.10 MB/s (×1.46)**. 리셋 직후 `FLASH_ACR = 0x00000000`이므로
>   **ART/프리페치 OFF가 하드웨어 기본값**이고 켜는 것은 전적으로 소프트웨어 책임이다.
> - **RCC / PLL / 클럭 행**: 에라타는 없지만 실측으로 드러난 하드웨어 동작 특성이 있다 —
>   **`PWR_CSR1.VOSRDY`는 SYSCLK이 HSI로 도는 동안 셋되지 않는다.** VOS 설정 후 이 플래그를 기다리도록 코드를 짜면
>   그 자리에서 영구히 멈춘다(PC=0x08000EE8 정지를 ST-LINK로 확인). 올바른 순서:
>   PWR 클럭 → VOS 설정(대기 없음) → HSE → PLL lock → over-drive → wait state → SYSCLK 전환.
> - **HASH / CRYP 행**: "에라타 없음"은 맞지만, 존재 여부는 별개다. 실측에서 두 베이스 주소(`0x50060000`/`0x50060400`)는
>   **BusFault 없이 응답**했다. 반면 **JPEG(`0x50051000`)는 RCC 비트 CLEAR0 + BusFault로 부재가 확정**됐다.
>   (F746에 없는 DFSDM1/DSI 주소도 응답하므로, 버스 응답만으로 존재를 주장할 수는 없다 — **CRYP/HASH는 판정 보류**.)

### 13.1 캐시 일관성은 에라타가 아니라 설계상 제약

- Cortex-M7의 D-Cache는 DMA/버스 마스터와 **하드웨어 일관성(coherency)을 제공하지 않는다.** 이는 결함이 아니라 아키텍처 특성이므로 ES0290/ARM SDEN 어디에도 에라타로 등재되지 않는다. 따라서 DMA 버퍼에 대해서는 **명시적 cache maintenance(`SCB_CleanDCache_by_Addr` / `SCB_InvalidateDCache_by_Addr`)** 또는 MPU로 해당 영역을 non-cacheable 지정이 필요하다. [D — ES0290/SDEN 전수 확인 결과 [A] + Cortex-M7 아키텍처]
- 단, 그 **cache maintenance 코드 자체**가 위의 **ARM 1313001**(invalidate 후 배리어 없는 WT store) 및 **1259864**(maintenance operation without following DSB가 트리거 조건 중 하나)에 걸릴 수 있으므로, **maintenance operation과 후속 store 사이에 DMB/DSB를 반드시 넣어야 한다.** [A]
- `CM7_CACR`의 **`FORCEWT` 비트를 절대 set하지 말 것** — 모든 cacheable 영역이 write-through로 바뀌어 **1259864의 조건 (3)** 에 전면 노출된다. 마찬가지로 **`SIWT`** 도 조건 (4)에 해당한다. [A]

> **실측 확인**: 이 보드의 `CACR = 0x00000002` → **FORCEWT = 0, SIWT = 0, ECCEN = 1**.
> 리셋 직후 덤프와 실행 중 덤프가 동일하므로 **기본값 그대로 안전한 쪽**이다.
>
> 한 가지 기록해 둘 것: **프로브 펌웨어는 캐시 정책 비교 벤치마크를 위해 의도적으로 `CACR.FORCEWT = 1` 구간을 실행했다**
> (SRAM1/SRAM2/FLASH/SDRAM의 OFF/WB/WT 3열 측정). 그 구간에서 눈에 띄는 오동작은 없었지만,
> **손상 검출 루틴이 없었으므로 이것은 1259864의 반증이 아니다.** 이 에라타는 타이밍 의존이며,
> 벤치마크 루프는 "같은 doubleword store → linefill → 재store → load" 패턴을 의도적으로 만들지 않는다.
> 참고로 그 측정에서 WB 열과 WT 열의 수치는 사실상 동일했다(읽기 위주 루프라 정책 차이가 드러나지 않음).
> ECCEN=1이라는 점도 함께 봐야 한다 — **1259864의 트리거 조건 중 하나가 "D-Cache에서 ECC 오류 관측"** 이다.

---

## 14. 보드 레벨 (STM32F746G-DISCO / MB1191)

- UM1907(디스커버리 킷 유저매뉴얼)은 보드 리비전 **MB1191-F746NGH6-B02, B03, B04, B05, C01** 전부에 대해 **"No limitation"** 으로 기재하고 있으며, 디바이스 레벨 한계는 **ES0290을 참조하도록** 안내한다. [C — UM1907 본문 검색 결과 기반, 원문 직접 확인 미완]
- 즉 **보드 고유의 하드웨어 결함은 공식 문서상 보고된 바 없고**, 이 보드에서 겪는 하드웨어 이슈는 전부 MCU 레벨(ES0290) 또는 코어 레벨(ARM SDEN)에서 찾아야 한다. [C]

> **실측 정정 — "No limitation"이라도 실물과 BSP가 어긋나는 지점이 있다**
>
> 공식 에라타는 아니지만, 이 개체에서 **문서·BSP의 전제와 실물이 다른 것**을 두 건 실측했다. 실무에서는 에라타와 똑같이 사람을 잡는다.
>
> | 항목 | 문서/BSP 전제 | 이 보드 실측 |
> |---|---|---|
> | QSPI NOR 부품 | Micron **N25Q128A** (`stm32746g_discovery_qspi.c`) | **Winbond W25Q128 계열** — Read JEDEC ID = `0xEF4018` (Micron이면 `0x20BA18`) |
> | 이더넷 PHY | **DP83848**, 주소 `0x01` (`stm32f7xx_hal_conf.h:220`) | **LAN8742A**, MDIO **주소 0** (ID1=`0x0007`, ID2=`0xC131`, OUI `0x0001F0`) |
>
> - QSPI는 표준 SPI NOR 명령(0x9F/0xEB)을 공유해 초기화·메모리맵 읽기가 정상 동작하지만,
>   **소거 시간·페이지 크기·상태 레지스터 비트는 부품별로 다르다.** 쓰기 코드는 Winbond 기준으로 재검증해야 한다.
> - 이더넷은 PHY 주소가 달라 **설정 그대로는 PHY를 찾지 못한다.**
> - 그 밖에 실측으로 확인된 보드 구성: 32.768 kHz LSE 크리스탈 실장(실측 32,765 Hz), microSD 삽입(PC13=0),
>   WM8994 오디오 코덱(I2C3 0x34), FT5336 터치(0x70), SDRAM 8 MB, LCD 480×272@59.3 Hz.
>   **LSI는 공칭 32 kHz 대비 -16.8 %(26,616~26,652 Hz)** 로, IWDG 타임아웃 계산 시 반드시 고려해야 한다.

---

## 출처

### 1차 출처 (직접 읽음)

| 문서번호 | 제목 / 판 | URL |
|---|---|---|
| **ES0290 Rev 7** (18-Jul-2019, 28 pages) | STM32F74xxx and STM32F75xxx device limitations | https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf |
| ES0290 Rev 7 (아카이브 사본, 실제 본문 취득 경로) | 위 PDF의 Wayback Machine 캡처 (2025-02-21 / 2026-01-08, 두 캡처 동일) | https://web.archive.org/web/20250221152200if_/https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf |
| ES0290 Rev 7 (HTML 부분 미러, Table 1/2/TOC 교차확인) | STMCU 중문 사이트 pdf2htmlEX 변환본 (pp.1–2) | https://www.stmcu.com.cn/upload/pdf_html/b4e2f0c4f7037c59583aa78c13e9c442.html |
| **ARM SDEN-1068427 v11.0** (28-May-2024, 63 pages) | Cortex-M7 (AT610) and Cortex-M7 with FPU (AT611) Software Developer Errata Notice | https://documentation-service.arm.com/static/665dff778ad83c4754308908 (원본 인덱스: https://developer.arm.com/documentation/SDEN-1068427) |
| **RM0385** (DocID026670 Rev 5) | STM32F75xxx / STM32F74xxx Reference manual — §40.6 ID codes, §SYSCFG_MEMRMP | https://www.st.com/resource/en/reference_manual/dm00124865-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf |
| **ARM DDI 0489F** | Cortex-M7 Processor Technical Reference Manual — §3.3.2 CPUID Base Register, Table 3-4 | https://developer.arm.com/documentation/ddi0489/f/system-control/register-descriptions/cpuid-base-register |
| **ARM DUI 0646B** | Cortex-M7 Devices Generic User Guide — Table 2-12 Memory region shareability and cache policies | https://developer.arm.com/documentation/dui0646/ |
| **DS10916** | STM32F745xx / STM32F746xx Datasheet (features/패키지 교차확인) | https://www.st.com/resource/en/datasheet/stm32f746ng.pdf |

### 2차 출처 / 교차확인

| 출처 | 용도 | URL |
|---|---|---|
| ST Community — STM32 MCU errata sheets: Expected preliminary updates | STM32F7 계열 예정 에라타 업데이트 없음 확인 | https://community.st.com/t5/stm32-mcus/stm32-mcu-errata-sheets-expected-preliminary-updates/ta-p/643992 |
| UM1907 Rev 6 (Sep 2025) / Rev 7 (May 2026) | 보드 리비전별 "No limitation" 기재 | https://www.st.com/resource/en/user_manual/um1907-discovery-kit-for-stm32f7-series-with-stm32f746ng-mcu-stmicroelectronics.pdf |
| ES0334 | STM32F76xxx/F77xxx device errata (자매 문서, 항목 비교용 — 본문 미취득) | https://www.st.com/resource/en/errata_sheet/es0334-stm32f76xxx-and-stm32f77xxx-device-errata-stmicroelectronics.pdf |
| PX4-Autopilot PR #11374 / ArduPilot Issue #11029 | 1259864에 대한 실사용 프로젝트 대응(D-Cache 비활성) 사례 | https://github.com/PX4/Firmware/pull/11374 , https://github.com/ArduPilot/ardupilot/issues/11029 |
| state-machine.com 미러 (Cortex-M7 SDEN v8) | ARM SDEN 구판 대조 | https://www.state-machine.com/doc/Cortex-M7_Software_Developers_Errata_Notice_v8.pdf |
| AN4073 (ES0290 §2.2.2에서 참조 지시) | ADC 정확도 개선 (ADCDC1 / ADCxDC2 메커니즘 상세) | https://www.st.com/en/embedded-software/ (문서 본문 미취득) |

### 접근 실패 기록
- `www.st.com` 직접 PDF 다운로드는 세션 중반부터 HTTP/2 `INTERNAL_ERROR` 및 HTTP/1.1 타임아웃으로 차단됨. ES0290 본문은 **Wayback Machine 아카이브 사본**으로 취득하였다(파일 크기 478,596 bytes, 28 pages, 2025-02-21과 2026-01-08 캡처 바이트 동일 → 내용 동일).

---

## 미확인 / 추가 조사 필요

1. **RM0385 Rev 9 (Feb 2025)의 REV_ID 목록** — 직접 확인한 것은 **RM0385 DocID026670 Rev 5 (Apr 2016)** 이며 여기에는 `0x1000 = Rev A`, `0x1001 = Rev Z` 두 값만 정의되어 있다. 최신 Rev 9에서 **`0x1003` 등 추가 값이 정의되었는지 미확인**. (다만 ES0290 Rev 7 Table 2가 "Z and 1 → 0x1001"로 명시하므로, Rev 1도 0x1001일 가능성이 높다.)
2. **Rev Z와 Rev 1의 물리적 구분 방법** — DS10916에서 패키지 마킹상 revision code를 읽는 정확한 위치/포맷을 확인하지 못했다. (본 조사에서 확보한 DS10916 텍스트에 "revision code" 관련 절이 검색되지 않음)
3. **ARM ETM-M7 (TM975) Software Developer Errata Notice** — ES0290 §2.1이 참조하도록 지시하는 두 번째 ARM 문서. **본문 미취득**. 트레이스(ETM) 사용 시 추가 제약이 있을 수 있다.
4. **ES0290 Rev 7이 정말 최신인지** — Wayback 2026-01-08 캡처까지 Rev 7로 확인했으나, 2026-01 이후 개정 여부는 st.com 직접 접근 실패로 미확인.
5. **ES0290 §2.1 Table 4의 "Minor" 평가 근거** — ST는 11개 코어 에라타 전부를 STM32F74xxx/F75xxx에서 "Minor"로 평가했으나, **Open 상태인 565285/1013783/2328489/1313001 등은 Table 4에 아예 없어** ST의 영향도 평가가 존재하지 않는다. 이들에 대한 ST 공식 입장 미확인.
6. ~~**`CM7_AHBSCR` 레지스터가 STM32F746에서 접근 가능한지** (2328489 회피책 1)~~ — **실측으로 해소됨**: 접근 가능하며 `AHBSCR @0xE000EFA0 = 0x00000800`(CTL=0b00, INITCOUNT=1). 회피책 ①이 리셋 기본값으로 이미 적용된 상태다. [A — 실측]
7. **AN4073, AN4839, AN4667, AN4031 본문** — ES0290이 §2.2.2에서 AN4073을 참조하지만 본문을 취득하지 못했다. 캐시/DMA/성능 관련 앱노트(AN4839/AN4667/AN4031)도 이번 조사 범위에서 본문 미취득.
8. **UM1907 원문 직접 확인** — 보드 리비전별 "No limitation" 기재는 검색 결과 기반이며 PDF 원문을 직접 읽지 못했다 [C].
9. **STM32F746G-DISCO의 SDRAM(MT48LC4M32B2 / IS42S32800G) 관련 실사용 이슈** — ST 커뮤니티에 SDRAM 관련 문의가 다수 있으나, ES0290에 등재된 FMC 에라타(§2.3.1~2.3.4) 외에 보드 고유 문제가 있는지는 미확인.
