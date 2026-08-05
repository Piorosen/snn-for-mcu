# STM32F746G-DISCO 외부 메모리 (SDRAM · QSPI Flash · microSD)

대상 보드: STM32F746G-DISCO (MB1191), MCU: STM32F746NGH6 (Cortex-M7 r0p1, Device ID 0x449, Rev Z)
조사 기준일: 2026-07-27

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거: `info/raw/measured_facts.md`, `info/results/probe_output.txt`, `info/raw/decoded_hotplug.md`.

### 실측으로 확인됨

**SDRAM**

- 접근 가능 용량 **8 MB** → 실측 `0xC0000000`부터 8,192 KB, `+0x00800000`에서 버스 폴트. 문서의 "0xC0800000 이상은 Reserved → AHB 에러"가 그대로 관측됐다.
- **FMC_SDCR1 = 0x00001954** (BSP 소스에서 계산) → 실측 레지스터 값 `0x00001954`. NC=0/NR=1/MWID=1/NB=1/CAS=2/WP=0/SDCLK=2/RBURST=1/RPIPE=0까지 비트필드 전부 일치.
- **FMC_SDTR1 = 0x01116361** → 실측 `0x01116361` (TMRD2/TXSR7/TRAS4/TRC7/TWR2/TRP2/TRCD2 사이클).
- **FMC_SDRTR = 0x00000C06 (COUNT=1539)** → 실측 `0x00000C06`.
- **모드 레지스터 0x0220** → 실측 `FMC_SDCMR.MRD = 0x0220` (BL=1 / Sequential / CL=2 / single-location write).
- **SDCLK = HCLK/2** → 실측 `SDCLK=0b10`, 100.00 MHz. (단 HCLK는 200 MHz다 — 아래 CONFLICT 참조)
- **SDRAM은 기본 상태에서 캐시되지 않는다** → 실측으로 확인. MPU를 켜지 않은 프로브 펌웨어에서 SDRAM read32가 D$OFF 59.70 / D$WB 59.71 MB/s, 포인터 체이싱도 OFF 26.06 / WB 26.06 cyc로 **완전히 동일**했다. 캐시가 전혀 개입하지 않는다는 직접 증거다.
- **본 프로젝트 MPU 리전 2** (0xA0000000, 8 KB, bufferable·non-cacheable·shareable, XN, FULL_ACCESS) → 실행 중 덤프의 `MPU_RBAR=0xA0000002`, `MPU_RASR=0x13050019`(SIZE=12→8 KB, B=1 C=0 S=1 TEX=0, AP=3, XN=1)와 **완전 일치**. `MPU_CTRL=0x00000005`로 MPU가 실제로 켜져 있었다.
- **이론 피크 200 MB/s** → 실측 **쓰기** 194.55 MB/s(피크의 97.3%), SRAM1→SDRAM 복사 194.69 MB/s. 쓰기 경로는 문서의 이론 모델과 거의 정확히 맞는다. (읽기는 어긋난다 — CONFLICT 참조)
- **프레임버퍼 522,240 B/레이어 × 2 = 1,044,480 B** → 실측 LTDC L1CFBAR=`0xC0000000`, L2CFBAR=`0xC007F800`, 간격 0x7F800 = 522,240 B, 피치 1920 B × 272 라인. 계산과 바이트 단위로 일치.
- **BSP가 초기화하는 FMC GPIO 38핀** → 실측 AF12 핀 총 44개. PC3(SDCKE0)/PG8(SDCLK)/PH3(SDNE0)/PH5(SDNWE) 등이 모두 AF12 Very-high speed로 확인됐다. (44 = FMC 38 + SDMMC1 6, 아래 CONFLICT 참조)
- **SNN 코드·가중치는 SDRAM이 아닌 내부 Flash/SRAM** → 빌드 산출물의 모든 섹션이 0x08000000(.text/.rodata)과 0x20000000(.data/.bss)에 있다.

**QSPI**

- **용량 16 MB, FSIZE=23** → 실측 `QUADSPI_DCR = 0x00170500`(FSIZE=23, CSHT=5, CKMODE=0), 폴트 경계 `0x90000000 + 0x01000000`.
- **PRESCALER=1 → QSPI_CK = HCLK/2**, SSHIFT=1, FSEL=0, DFM=0 → 실측 `QUADSPI_CR = 0x01400311`. 실측 클럭 100.00 MHz.
- **1-4-4 모드, 0xEB, 24비트 주소, 더미 10, SDR, 메모리맵, SIOO 사용 안 함** → 실측 `QUADSPI_CCR = 0x0F282DEB` (INSTRUCTION=0xEB, IMODE=1선, ADMODE=4선, DMODE=4선, ADSIZE=2, DCYC=10, FMODE=3, DDRM=0, SIOO=0). 문서가 BSP 소스에서 유도한 필드가 하나도 틀리지 않았다.
- **순차 스트리밍 피크 50.0 MB/s** (계산) → 실측 49.97 MB/s(D$OFF/WB 동일), 64 KB 워킹셋 read32 51.69 MB/s. **계산이 거의 그대로 맞았다.**
- **32 B 캐시라인 랜덤 리드 88 QSPI클럭 = 880 ns ≈ 36.4 MB/s** (계산) → 실측 D$WB 스트라이드 197.54 cyc @200 MHz = 987.7 ns(≈32.4 MB/s). AHB 오버헤드만큼 느리지만 같은 자릿수이고, D$OFF 94.22 cyc → D$WB 197.54 cyc로 **캐시라인 채우기가 실제로 일어난다는 것**까지 확인된다(0x90000000이 Normal/WT라는 문서 주장의 방증).
- **본 프로젝트는 QSPI를 초기화조차 하지 않는다** → 실행 중 덤프에서 RCC_AHB3ENR에 QSPIEN 없음, QUADSPI 레지스터 전부 0, QSPI 핀 PD11/PD12/PD13/PE2가 전부 입력 모드로 남아 있었다.

**microSD**

- **카드 검출 PC13** → 실측 PC13 = 풀업 입력, IDR=0 → **microSD 카드가 실제로 삽입되어 있다**.

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

**1. QSPI 플래시 부품이 다르다 (가장 중요)**

- 문서 주장: Micron **N25Q128A13EF840E**. §2 전체(더미 사이클 표, tPP/tSSE/tSE/tBE, ICC, tSHSL)가 이 부품의 데이터시트에 기반한다.
- 실측: 메모리맵 모드 전환 **전에** Read JEDEC ID(0x9F)를 직접 보내 받은 응답이 **`0xEF4018`** = 제조사 0xEF(**Winbond**), 타입 0x40, 용량 0x18(16 MB). Micron이라면 `0x20BA18`이어야 한다.
- 원인: **보드 리비전 차이**. `06_board.md` §1.2가 예측한 대로 이 보드는 **MB1191-F746NGH6-C-03(`$CU4`) 리비전**이고 QSPI가 **Winbond W25Q128JV** 계열로 교체됐다. 문서 오류가 아니라 문서가 참조한 UM1907 회로도(B-02)가 구 리비전인 것이다.
- 파급: §2.1~§2.3의 **부품 고유 수치(더미 사이클 최대 주파수 표, 소거·프로그램 시간, 소비 전류, AC 타이밍)는 이 보드에 적용되지 않는다.** 공통 SPI NOR 명령(0x9F, 0xEB)은 호환이라 ST BSP 초기화·메모리맵 읽기는 정상 동작했다(실측 `BSP_QSPI_Init()` OK).

**2. HCLK 216 MHz 가정으로 병기한 값들**

- 문서 주장: 곳곳에 "HCLK 216 MHz → SDCLK 108 MHz / QSPI_CK 108 MHz / 216 MB/s / 54 MB/s" 병기.
- 실측: 이 보드의 SNN 펌웨어는 PLLM=25, PLLN=400, PLLP=2 → **SYSCLK=HCLK=200 MHz**, SDCLK=QSPI_CK=**100 MHz**. 216 MHz 구성은 이 보드에서 한 번도 돌지 않았다.
- 원인: 문서가 두 구성을 병기한 것이며 200 MHz 쪽을 주값으로 삼았으므로 심각한 오류는 아니다. 다만 "108 MHz 구성이면 규격 초과" 같은 서술은 **이 보드에서는 아예 해당 사항이 없다**.

**3. SDRAM 리프레시 간격 계산이 틀렸다**

- 문서 주장: "BSP 값의 실제 리프레시 간격 = (1539 + 20) × 10 ns = **15.59 µs**", 전체 4096행 = 63.86 ms.
- 실측/정정: RM0385 공식은 `Refresh rate = (COUNT + 1) × SDRAM clock period`이고, `−20`은 **COUNT를 구할 때 빼는 안전 마진**이지 되돌려 더하는 값이 아니다. 실제 간격 = (1539 + 1) × 10 ns = **15.40 µs**, 전체 4096행 = **63.08 ms**.
- 원인: **문서 내부 불일치**(같은 절에서 공식은 `COUNT+1`로 인용해 놓고 계산은 `COUNT+20`으로 했다). 결론(64 ms 이내)은 어느 쪽이든 만족한다.
- 주의: `decoded_hotplug.md`도 같은 `+20` 계산으로 15.59 µs를 적었다. 실측 자료 쪽 해석도 이 지점에서 흔들렸으므로, **RM0385 원문 공식을 따르는 15.40 µs가 맞다**.

**4. SDRAM 읽기 대역폭이 이론의 30%에 불과하다**

- 문서 주장: 이론 피크 **200 MB/s**, 리프레시 차감 후 ≈199.1 MB/s. §4 비교표에도 "이론 피크 대역폭 200 MB/s", "쓰기: 읽기와 동일 속도"로 적혀 있다.
- 실측(D-cache 무관, MPU 미설정 = Device 메모리): **read32 59.70 MB/s / read64 59.56 / readLDM 60.76**, write32 **194.55**, copyLDM **81.46**. SDRAM→SRAM1 복사 58.44 MB/s, SRAM1→SDRAM 194.69 MB/s.
- 즉 **쓰기는 이론값의 97%인데 읽기는 30%다.** "읽기와 쓰기가 같은 속도"라는 문서 서술은 이 보드에서 성립하지 않는다.
- 원인: 쓰기는 스토어 버퍼가 흡수해 파이프라인이 멈추지 않지만, 읽기는 매 로드가 FMC 왕복 레이턴시(아래 5번)를 그대로 맞는다. Device 메모리라 캐시라인 단위 버스트가 만들어지지 않고, CPU 단일 로드로는 FMC 읽기 FIFO의 선인출(CAS+1+RPIPE/2 = 3개)을 다 소화하지 못한다.
- **측정 조건 한계**: 이 수치는 MPU를 설정하지 않은 상태다. 본 프로젝트처럼 MPU로 0xC0000000을 Normal/WT cacheable로 만들면 읽기가 개선될 여지가 있으나 **그 조건은 아직 측정되지 않았다**(아래 참조).

**5. SDRAM 접근 레이턴시가 문서 하한의 2배다**

- 문서 주장: 행 히트 20 ns(2 SDCLK), 행 미스 60 ns(6 SDCLK). 문서 스스로 "FMC 측 하한, AXI/버스 오버헤드 미포함"이라 단서를 달았고 "정확한 CPU 스톨 사이클 수는 미확인, 실측이 필요하다"고 적었다.
- 실측: 포인터 체이싱(Sattolo 셔플, 프리페치 불가) 기준 **21.1~26.1 CPU 사이클 @200 MHz = 105~130 ns**, 워킹셋 1 KB→512 KB에 걸쳐 단조 증가. 32 B 스트라이드 접근은 18.63 cyc/접근.
- 원인: 모순이 아니라 **범위가 다르다**. 문서 값은 SDRAM 핀 레벨 하한, 실측은 CPU가 실제로 멈추는 시간. **코드 성능을 예측할 때 써야 하는 숫자는 105~130 ns다.** 문서의 미확인 항목 #4가 이것으로 해소됐다.

**6. microSD가 "미사용"이라는 서술이 실행 이미지와 다르다**

- 문서 주장(§3.3): `MX_SDMMC1_SD_Init()` 호출이 주석 처리되어 SD카드 미사용.
- 실측(원래 SNN 펌웨어가 돌던 상태의 ST-LINK 덤프): `RCC_APB2ENR`에 **SDMMC1EN=1**(클럭 공급 중), **PC8/PC9/PC10/PC11/PC12 + PD2가 AF12(SDMMC1_D0–D3/CK/CMD)로 설정**되어 있었다. AF12 핀 총계 44개도 정확히 `FMC 38 + SDMMC1 6`으로 떨어진다.
- 원인 추정: **보드에 올라가 있던 플래시 이미지가 현재 소스 트리(HEAD)와 다르다.** 현재 `main.cc:208`은 확실히 주석 처리되어 있으므로, 덤프 시점의 이미지는 SDMMC를 초기화하던 이전 빌드였을 가능성이 높다. `HAL_SD_MspInit()`은 `HAL_SD_Init()`에서만 불리므로 우연한 설정은 아니다.
- 심각도: 중간. "SD 하드웨어를 쓸 수 있다"는 결론에는 영향이 없고, **"현재 프로젝트 상태"라는 서술의 시점만 문제**다.

### 아직 측정하지 않음

- **MPU 리전 1(0xC0000000, 32 MB, WT, 실행 허용)의 실제 RASR** — 덤프는 RNR이 가리키던 리전 2만 잡혔다. `MPU_RNR`에 1을 쓴 뒤 `MPU_RBAR/RASR`을 읽으면 된다.
- **MPU로 SDRAM을 Normal/WT cacheable로 만든 뒤의 대역폭·레이턴시** — 프로브 펌웨어는 MPU를 켜지 않았다. 리전 설정 후 순차 read32와 포인터 체이싱을 재측정해 위 4·5번 수치와 비교.
- **`SWP_FMC = 01` 리맵(0x60000000, WBWA, XIP)** — `SYSCFG_MEMRMP`에 0x400을 쓰고 0x60000000에서 읽기 대역폭과 실제 코드 실행을 확인.
- **행 히트 vs 행 미스 분리** — 한 행 = 256열 × 2 B = 512 B이므로, 512 B 미만 스트라이드(행 내부)와 512 B·뱅크 교차 스트라이드의 접근 사이클을 비교하면 갈라진다.
- **BSP DMA 경로(DMA2 Stream0, FIFO OFF, SINGLE 버스트)의 실효 처리량** — `BSP_SDRAM_ReadData_DMA()`를 DWT CYCCNT로 감싸 CPU 직접 접근과 비교하고, FIFO+INCR8로 바꾼 뒤 재측정.
- **TRAS 40 ns(규격 42 ns) 마진 부족의 실동작 영향** — SDRAM 8 MB 전 영역 walking-1 + 랜덤 패턴을 장시간 반복하며 오류율을 세고, 가능하면 다이 온도를 올려(현재 35~41 °C) 재측정.
- **QSPI `SIOO=1`의 효과** — `QUADSPI_CCR.SIOO`를 세우고 캐시라인 랜덤 리드를 재측정해 오버헤드 24→16 클럭(≈40 MB/s) 예측을 검증.
- **QSPI XIP 실제 코드 실행** — 함수 하나를 0x90000000 영역에 배치해 호출하고 CYCCNT로 실행 사이클을 측정.
- **QSPI 쓰기·소거 시간(Winbond 기준)** — 현재 QSPI에 데이터가 들어 있으므로 백업 후 빈 서브섹터에서 소거·프로그램하며 CYCCNT 측정. Micron 수치(tSSE 0.2 s / tSE 0.7 s)와 대조.
- **microSD 실효 대역폭과 카드 정보** — `BSP_SD_Init()` 후 CID/CSD를 읽어 카드 용량·클래스를 확인하고, 블록 전송을 CYCCNT로 재 4비트 버스 실효 MB/s 산출.
- **ES0290 에라타(FMC 쓰기 FIFO / QUADSPI) 적용 여부** — `FMC_BCR1.WFDIS`를 직접 읽어 리셋 기본값(0)임을 확인하면 문서의 [C] 판정을 [A]로 올릴 수 있다.

---

## 요약

- **SDRAM 부품**: Micron **MT48LC4M32B2B5-6A**. 다이 자체는 128 Mbit(4 Meg × 32 = 1 Meg × 32 × 4 뱅크)이지만, 보드에서 **D0–D15(16비트)만 배선**되어 **실사용 가능 용량은 64 Mbit = 8 MB**. DQ16–DQ31은 미사용 + 10 kΩ 풀다운. [A: UM1907 §6.11]
- **SDRAM 매핑**: FMC SDRAM **Bank 1** (FMC_SDNE0 / FMC_SDCKE0) → 베이스 **0xC0000000**, 유효 범위 **0xC0000000–0xC07FFFFF (0x800000 = 8 MB)**. [A: RM0385 §13.4.3 + BSP]
- **BSP 실제 레지스터 값** (BSP 소스에서 직접 계산): **FMC_SDCR1 = 0x00001954**, **FMC_SDTR1 = 0x01116361**, **FMC_SDRTR COUNT = 0x603 (1539)** → SDRTR 레지스터 기록값 **0x00000C06**, 모드 레지스터 **0x0220** (BL=1, Sequential, CL=2, Single-location write). [A: BSP 소스 + RM0385 비트필드]
- **SDRAM 클럭**: SDCLK = HCLK/2. 본 프로젝트는 HCLK = 200 MHz(PLL: HSE 25 MHz /M25 ×N400 /P2)이므로 **SDCLK = 100 MHz (tCK = 10 ns)**. DS10916은 **3.0–3.6 V에서 FMC_SDCLK 최대 100 MHz**로 규정 → HCLK 216 MHz(SDCLK 108 MHz) 구성은 데이터시트 규격 초과. [A: DS10916 §5.3.x SDRAM waveforms and timings / BSP 주석]
- **SDRAM 이론 대역폭**: 100 MHz × 16 bit = **200 MB/s** 피크, 리프레시 오버헤드 약 0.45% 차감 시 **≈199.1 MB/s**. (108 MHz 구성이면 216 MB/s) [D: 계산]

> **실측 정정**: CPU가 실제로 얻는 값은 **읽기 59.7 MB/s(이론의 30%) / 쓰기 194.6 MB/s(97%)** 다.
> 200 MB/s는 SDRAM 핀 레벨 상한이며 읽기 경로에는 도달하지 않는다. (MPU 미설정 = Device 메모리 조건)
- **SDRAM은 기본적으로 실행 불가 + 캐시 불가**: 0xC0000000–0xDFFFFFFF는 ARMv7-M 기본 주소맵에서 **Device, Non-shareable, XN**(캐시 없음, 명령어 페치 금지). 해결책은 (a) `SYSCFG_MEMRMP.SWP_FMC[1:0]=01`로 0x60000000에 리맵(Normal/WBWA/실행 가능), 또는 (b) MPU 리전으로 속성 변경. 본 프로젝트는 (b)를 사용해 0xC0000000 32 MB를 **Write-Through 캐시 + 실행 허용**으로 설정. [A: ARM DDI 0403E.e Table B3-1 / RM0385 §7.2.1 / main.cc MPU_Config]
- **QSPI Flash 부품**: Micron **N25Q128A13EF840E**, **128 Mbit = 16 MB**, 256 KB 아님 — 64 KB 섹터 × 256개 / 4 KB 서브섹터 × 4096개 / 256 B 페이지 × 65536개. 최대 SCK **108 MHz**(READ 0x03 명령만 54 MHz). [A: N25Q128 데이터시트 / UM1907 §6.12]

> **실측 정정**: 이 보드의 QSPI는 Micron이 **아니다**. Read JEDEC ID(0x9F) 실측값이 **`0xEF4018`**
> = Winbond(0xEF) W25Q128 계열 16 MB다(Micron이면 `0x20BA18`). 보드가 **C-03 / `$CU4` 리비전**이라
> QSPI가 W25Q128JV로 교체된 것이다. 용량 16 MB·페이지 256 B 같은 공통 사양은 그대로지만,
> **아래 §2.2~§2.3의 N25Q 고유 수치(더미 사이클 표, 소거/프로그램 시간, 소비 전류, AC 타이밍)는
> 이 보드에 적용되지 않는다.**
- **QSPI 메모리맵**: **0x90000000** 베이스, FSIZE=23 → 16 MB(0x90000000–0x90FFFFFF). 이 주소대(0x80000000–0x9FFFFFFF)는 ARMv7-M 기본 속성이 **Normal, Write-Through 캐시, 실행 가능**이므로 **XIP(코드 실행)이 기본 설정에서 가능**. [A: RM0385 §14.3.6 / ARM DDI 0403E.e Table B3-1]
- **QSPI 실효 성능**: BSP는 1-4-4 (0xEB QUAD I/O FAST READ, 명령 1-line, 24-bit 주소 4-line, **더미 10 사이클**, 데이터 4-line, SDR) + 프리스케일러 1 → **QSPI_CK = HCLK/2 = 100 MHz**. 순차 스트리밍 피크 **50 MB/s**, 32바이트 캐시라인 랜덤 리드 기준 **≈36.4 MB/s** (트랜잭션당 24 클럭 오버헤드). [D: 계산]
- **microSD**: CN3 슬롯, **SDMMC1**, **4비트 버스**(D0–D3 배선, 8비트 미배선). SDMMC 최대 fPP = 50 MHz. 본 프로젝트에서는 `MX_SDMMC1_SD_Init()`이 주석 처리되어 미사용. [A: UM1907 Table 7 / DS10916 Table 112]

---

## 1. SDRAM (Micron MT48LC4M32B2B5-6A)

### 1.1 부품 식별 및 보드 배선

| 항목 | 값 | 신뢰도 | 출처 |
|---|---|---|---|
| 부품번호 | MT48LC4M32B2B5-6A (MICRON) | [A] | UM1907 §6.11, Figure 26 회로도 |
| 다이 구성 | 128 Mbit = 4 Meg × 32 = **1 Meg × 32 × 4 뱅크** | [A] | Micron 128Mb x32 SDRAM DS, Table 2 |
| 패키지 코드 `B5` | 90-ball VFBGA 8 mm × 13 mm, Pb-free | [A] | Micron DS, Options 표 |
| 속도 등급 `-6A` | tCK(CL=3) = 6 ns → **166/167 MHz** | [A] | Micron DS Table 1 / Farnell 자동차용판 Options |
| 보드 데이터폭 | **16 bit만 사용** (FMC_D[0..15]) | [A] | UM1907 §6.11 |
| 미사용 신호 | **DQ16–DQ31**, 10 kΩ 풀다운 저항 연결 | [A] | UM1907 §6.11 |
| 접근 가능 용량 | **64 Mbit = 8,388,608 바이트 (8 MB)** | [A] | UM1907 §6.11 + `SDRAM_DEVICE_SIZE 0x800000` |
| 전원 | 단일 +3.3 V ±0.3 V | [A] | Micron DS Features |
| 어드레싱 | 뱅크 4개 (BA0, BA1) / **행 4096개 (A0–A11, 12비트)** / **열 256개 (A0–A7, 8비트)** | [A] | Micron DS Table 2 |
| 리프레시 요구 | **4096 행 / 64 ms** → 분산 리프레시 주기 **15.625 µs** | [A] | Micron DS Table 18 (tREF = 64 ms) |
| 지원 CAS latency | 1, 2, 3 | [A] | Micron DS Features |
| 지원 버스트 길이 | 1, 2, 4, 8, full page | [A] | Micron DS Features |

용량 검산: `4 뱅크 × 4096 행 × 256 열 × 2 바이트 = 8,388,608 B = 8 MB` [D: 계산, BSP `SDRAM_DEVICE_SIZE 0x800000`과 일치]

FMC 배선 신호 (UM1907 회로도 넷 이름): `FMC_A[0..11]`, `FMC_D[0..15]`, `FMC_SDNWE`, `FMC_SDNRAS`, `FMC_SDNCAS`, `FMC_SDCLK`, **`FMC_SDNE0`**, **`FMC_SDCKE0`** → SDNE0/SDCKE0 사용은 **SDRAM Bank 1** 확정. [A: UM1907 SDRAM.SchDoc 넷 리스트]

BSP가 초기화하는 GPIO 핀 (`BSP_SDRAM_MspInit`, 모두 AF12_FMC, Pull-up, GPIO_SPEED_FAST):

| 포트 | 핀 |
|---|---|
| GPIOC | PC3 |
| GPIOD | PD0, PD1, PD8, PD9, PD10, PD14, PD15 |
| GPIOE | PE0, PE1, PE7, PE8, PE9, PE10, PE11, PE12, PE13, PE14, PE15 |
| GPIOF | PF0, PF1, PF2, PF3, PF4, PF5, PF11, PF12, PF13, PF14, PF15 |
| GPIOG | PG0, PG1, PG4, PG5, PG8, PG15 |
| GPIOH | PH3, PH5 |

[A: `Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_sdram.c:360-413`]

### 1.2 Micron -6 등급 AC 타이밍 (Table 18)

| 파라미터 | 심볼 | -6 (min/max) | -7 (min/max) | 단위 |
|---|---|---|---|---|
| 클럭 사이클 (CL=3) | tCK(3) | 6 | 7 | ns |
| 클럭 사이클 (CL=2) | tCK(2) | 10 | 10 | ns |
| 클럭 사이클 (CL=1) | tCK(1) | 20 | 20 | ns |
| CLK 상승엣지 기준 액세스 (CL=3) | tAC(3) | max 5.5 | max 5.5 | ns |
| CLK 상승엣지 기준 액세스 (CL=2) | tAC(2) | max 7.5 | max 8 | ns |
| ACTIVE → PRECHARGE | **tRAS** | **42** / 120K | 42 / 120K | ns |
| ACTIVE → ACTIVE (동일 뱅크) | **tRC** | **60** | 70 | ns |
| AUTO REFRESH 주기 | **tRFC** | **60** | 70 | ns |
| ACTIVE → READ/WRITE | **tRCD** | **18** | 20 | ns |
| PRECHARGE 명령 주기 | **tRP** | **18** | 20 | ns |
| ACTIVE bank a → ACTIVE bank b | tRRD | 12 | 14 | ns |
| Write recovery | **tWR** | **1 CLK + 6 ns** | 1 CLK + 7 ns | — |
| Self refresh 탈출 → ACTIVE | **tXSR** | **70** | 70 | ns |
| 리프레시 주기 (4096 행) | tREF | 64 | 64 | ms |
| LOAD MODE REG → ACTIVE/REFRESH | **tMRD** | **2 tCK** | 2 tCK | tCK |
| Data-in → PRECHARGE | tDPL | 2 tCK | 2 tCK | tCK |
| Data-in → ACTIVE (CL=2) | tDAL(2) | 4 tCK | 4 tCK | tCK |

허용 동작 주파수 (Micron DS Table 7): **-6 등급 → CL=1: ≤50 MHz, CL=2: ≤100 MHz, CL=3: ≤166 MHz**. [A: Micron 128Mb x32 SDRAM DS Rev. L, Table 7 / Table 18]

> **중요**: BSP는 CL=2를 쓰는데, **-6 등급의 CL=2 최대 동작 주파수는 100 MHz**다. HCLK 200 MHz(SDCLK 100 MHz)는 정확히 한계값이고, HCLK 216 MHz(SDCLK 108 MHz)는 **CL=2 규격을 초과**한다. [D: Table 7 대조]

### 1.3 STM32 FMC SDRAM 컨트롤러 사양

| 항목 | 값 | 신뢰도 | 출처 |
|---|---|---|---|
| SDRAM 뱅크 수 | 2 (Bank1 = 0xC0000000, Bank2 = 0xD0000000, 각 256 MB 윈도우) | [A] | RM0385 §13.4 Figure 33, §13.4.3 Table 50 |
| 뱅크 선택 비트 | HADDR[28] (0 → Bank1/SDCR1/SDTR1) | [A] | RM0385 Table 50 |
| 데이터 폭 | 8 / 16 / 32 bit | [A] | RM0385 §13.7.1 |
| SDCLK | **HCLK/2 또는 HCLK/3만** 가능 | [A] | RM0385 §13.7.1, SDCR1.SDCLK |
| 읽기 FIFO | **6 라인 × 32비트**, 라인당 14비트 어드레스 태그(열 11비트 + 내부뱅크/행 2비트 + 디바이스 1비트) | [A] | RM0385 §13.7.1, §13.7.3 |
| RBURST 선행 인출 개수 | `CAS latency + 1 + (RPIPE delay)/2` | [A] | RM0385 §13.7.3 |
| CAS latency 지원 | 1, 2, 3 | [A] | RM0385 §13.7.1 |
| FMC_SDCLK 최대 (3.0 V ≤ VDD ≤ 3.6 V) | **100 MHz** @ CL(부하) = 20 pF | [A] | DS10916 "SDRAM waveforms and timings" |
| FMC_SDCLK 최대 (2.7 V ≤ VDD ≤ 3.6 V) | 90 MHz @ CL = 30 pF | [A] | DS10916 동일 절 |
| FMC_SDCLK 최대 (1.71 V ≤ VDD < 1.9 V) | 70 MHz @ CL = 10 pF | [A] | DS10916 동일 절 |
| tw(SDCLK) | 2·THCLK − 0.5 ~ 2·THCLK + 0.5 ns | [A] | DS10916 Table 104 |
| 데이터 입력 셋업/홀드 | tsu = 3.5 ns / th = 1.5 ns | [A] | DS10916 Table 104 |
| FMC 제어 레지스터 베이스 | 0xA0000000–0xA0000FFF (SDCR1 = +0x140, SDTR1 = +0x148, SDCMR = +0x150, SDRTR = +0x154) | [A] | RM0385 Table 1, §13.7.5 |

**16비트 폭 주소 매핑** (RM0385 Table 53, 12-bit row / 8-bit column / 4 banks 구성):

| HADDR 비트 | 역할 |
|---|---|
| HADDR[0] | BM0 (16비트 접근 바이트 마스크) |
| HADDR[8:1] | Column[7:0] |
| HADDR[20:9] | Row[11:0] |
| HADDR[22:21] | Bank[1:0] (= BA[1:0]) |
| HADDR[27:23] | **Reserved — 접근 시 AHB 에러 발생** |

→ 실제 유효 영역은 **0xC0000000–0xC07FFFFF의 8 MB**뿐이며, 그 위(0xC0800000 이상)는 Reserved로 AHB 에러가 난다. Cortex-M7이 접근하면 HardFault, DMA가 접근하면 transfer error. [A: RM0385 Table 53 note 2, §14.3.6 유사 규정 / D: 8 MB 경계 계산]

### 1.4 BSP(stm32746g_discovery_sdram.c)가 설정하는 실제 값

BSP 소스의 `BSP_SDRAM_Init()` 설정값:

```
Timing.LoadToActiveDelay    = 2;   /* TMRD */
Timing.ExitSelfRefreshDelay = 7;   /* TXSR */
Timing.SelfRefreshTime      = 4;   /* TRAS */
Timing.RowCycleDelay        = 7;   /* TRC  */
Timing.WriteRecoveryTime    = 2;   /* TWR  */
Timing.RPDelay              = 2;   /* TRP  */
Timing.RCDDelay             = 2;   /* TRCD */

SDBank             = FMC_SDRAM_BANK1                 (0x00000000)
ColumnBitsNumber   = FMC_SDRAM_COLUMN_BITS_NUM_8     (0x00000000)
RowBitsNumber      = FMC_SDRAM_ROW_BITS_NUM_12       (0x00000004)
MemoryDataWidth    = FMC_SDRAM_MEM_BUS_WIDTH_16      (0x00000010)
InternalBankNumber = FMC_SDRAM_INTERN_BANKS_NUM_4    (0x00000040)
CASLatency         = FMC_SDRAM_CAS_LATENCY_2         (0x00000100)
WriteProtection    = FMC_SDRAM_WRITE_PROTECTION_DISABLE (0x00000000)
SDClockPeriod      = FMC_SDRAM_CLOCK_PERIOD_2        (0x00000800)
ReadBurst          = FMC_SDRAM_RBURST_ENABLE         (0x00001000)
ReadPipeDelay      = FMC_SDRAM_RPIPE_DELAY_0         (0x00000000)
REFRESH_COUNT      = 0x0603
```

[A: `stm32746g_discovery_sdram.c:137-155`, `stm32746g_discovery_sdram.h:56-67`, HAL 상수값은 `stm32f7xx_ll_fmc.h:696-783`]

#### FMC_SDCR1 = **0x00001954**

HAL `FMC_SDRAM_Init()`가 Init 필드를 그대로 OR해서 SDCR[0]에 기록하므로 [A: `stm32f7xx_ll_fmc.c:806-816`], 위 상수들의 합이 곧 레지스터 값이다.

| 비트 | 필드 | 값 | 의미 |
|---|---|---|---|
| 1:0 | NC | 00 | 열 주소 **8 비트** |
| 3:2 | NR | 01 | 행 주소 **12 비트** |
| 5:4 | MWID | 01 | 데이터 폭 **16 비트** |
| 6 | NB | 1 | 내부 뱅크 **4개** |
| 8:7 | CAS | 10 | **CAS latency = 2** |
| 9 | WP | 0 | 쓰기 허용 |
| 11:10 | SDCLK | 10 | **SDCLK = 2 × HCLK 주기 (HCLK/2)** |
| 12 | RBURST | 1 | **버스트 읽기(선행 인출) 활성** |
| 14:13 | RPIPE | 00 | 읽기 파이프 지연 **0 HCLK** |

`0x4 + 0x10 + 0x40 + 0x100 + 0x800 + 0x1000 = 0x1954` [D: 비트필드 합산, RM0385 §13.7.5 인코딩 대조]

#### FMC_SDTR1 = **0x01116361**

HAL이 각 필드에 **−1**을 적용해 기록한다 [A: `stm32f7xx_ll_fmc.c:867-876`].

| 비트 | 필드 | 프로그램값 | 실제 사이클 | @100 MHz 환산 | Micron -6 최소값 | 판정 |
|---|---|---|---|---|---|---|
| 3:0 | TMRD | 0x1 | 2 | 20 ns | 2 tCK = 20 ns | OK |
| 7:4 | TXSR | 0x6 | 7 | 70 ns | 70 ns | OK (딱 맞음) |
| 11:8 | TRAS | 0x3 | 4 | **40 ns** | **42 ns** | **2 ns 부족** |
| 15:12 | TRC | 0x6 | 7 | 70 ns | tRC 60 / tRFC 60 ns | OK |
| 19:16 | TWR | 0x1 | 2 | 20 ns | 1 CLK + 6 = 16 ns | OK |
| 23:20 | TRP | 0x1 | 2 | 20 ns | 18 ns | OK |
| 27:24 | TRCD | 0x1 | 2 | 20 ns | 18 ns | OK |

`0x01000000 | 0x00100000 | 0x00010000 | 0x6000 | 0x300 | 0x60 | 0x1 = 0x01116361` [D: 비트필드 합산]

**타이밍 여유 관련 관찰 (모두 [D], 계산에 의한 지적):**

1. **TRAS 미달**: BSP의 TRAS = 4 사이클 = 40 ns인데 Micron -6의 tRAS(min) = 42 ns. 100 MHz에서 2 ns 부족하다. SDCLK가 108 MHz(HCLK 216 MHz)면 4 × 9.26 = 37.0 ns로 격차가 더 벌어진다. ST BSP 기본값이므로 실사용상 문제는 보고되지 않지만, 규격상 마진이 음수다.
2. **RM0385의 TWR 제약 위반**: RM0385 §13.7.5 TWR 설명은 `TWR ≥ TRAS − TRCD` **그리고** `TWR ≥ TRC − TRCD − TRP`를 요구한다. BSP 값으로는 `TRC − TRCD − TRP = 7 − 2 − 2 = 3`인데 TWR = 2이므로 두 번째 조건을 만족하지 않는다.
3. **SDCLK 108 MHz 시 TXSR 위반**: 7 × 9.26 = 64.8 ns < 70 ns. 셀프 리프레시를 사용하지 않으면 실질 영향은 없다.

#### FMC_SDRTR (리프레시) — COUNT = 0x603 = 1539

HAL은 `MODIFY_REG(SDRTR, FMC_SDRTR_COUNT, RefreshRate << 1)` 로 기록하므로 **레지스터 값 = 0x00000C06** [A: `stm32f7xx_ll_fmc.c:1013`].

RM0385 §13.7.5 공식:

```
Refresh rate = (COUNT + 1) × SDRAM clock period
COUNT = (SDRAM refresh period / Number of rows) − 20
```
(−20은 "읽기 요청이 수락된 상태에서 내부 리프레시 요청이 발생할 때의 안전 마진"이며, COUNT는 최소 41(0x29) 이상이어야 한다.) [A: RM0385 §13.7.5 FMC_SDRTR]

- 공식 그대로 계산: `(64 ms / 4096) × 100 MHz − 20 = 1562.5 − 20 = 1542.5` → 0x606
- BSP 실제값: **1539 (0x603)** → ST가 약 3 사이클 더 보수적으로 잡음
- BSP 값의 실제 리프레시 간격: `(1539 + 20) × 10 ns = 15.59 µs`
- 전체 4096 행 리프레시 소요: `4096 × 15.59 µs = 63.86 ms ≤ 64 ms` → 규격 만족 [D: 계산]

> **실측 정정**: 위 두 줄의 계산이 틀렸다. RM0385 공식은 바로 위에 인용한 대로
> `Refresh rate = (COUNT + 1) × SDRAM clock period`이며, `−20`은 COUNT를 **구할 때** 빼는
> 안전 마진이지 되돌려 더하는 값이 아니다. 실제 간격은 `(1539 + 1) × 10 ns = **15.40 µs**`,
> 전체 4096행 = **63.08 ms**다. (온디바이스 프로브도 15.400 µs로 보고했다.)
> 결론(64 ms 이내 만족)은 어느 쪽이든 같다.
- RM0385 추가 제약 "COUNT는 TWR+TRP+TRC+TRCD+4와 같으면 안 됨" = 2+2+7+2+4 = 17 ≠ 1539 → 만족 [D]

#### SDRAM 모드 레지스터 = **0x0220**

`BSP_SDRAM_Initialization_sequence()`가 LOAD MODE REGISTER로 기록하는 값:

| 필드 | 값 | 의미 |
|---|---|---|
| M2:M0 (BL) | 000 | 버스트 길이 **1** |
| M3 (BT) | 0 | **Sequential** |
| M6:M4 (CL) | 010 | **CAS latency 2** (0x0020) |
| M8:M7 | 00 | Standard operating mode |
| M9 | 1 | **Single-location write** (버스트 쓰기 비활성, 0x0200) |

[A: `stm32746g_discovery_sdram.h:83-93`, `stm32746g_discovery_sdram.c:242-246`]

#### 초기화 시퀀스 (BSP)

1. `FMC_SDRAM_CMD_CLK_ENABLE`, target = Bank1, AutoRefreshNumber = 1
2. `HAL_Delay(1)` — 규격은 최소 100 µs, SysTick 단위가 ms라 1 ms 삽입
3. `FMC_SDRAM_CMD_PALL` (Precharge All)
4. `FMC_SDRAM_CMD_AUTOREFRESH_MODE`, **AutoRefreshNumber = 8**
5. `FMC_SDRAM_CMD_LOAD_MODE`, ModeRegisterDefinition = 0x0220
6. `HAL_SDRAM_ProgramRefreshRate(0x0603)`

[A: `stm32746g_discovery_sdram.c:206-259`]

#### DMA 설정 (BSP_SDRAM_ReadData_DMA / WriteData_DMA)

- **DMA2 Stream0, Channel 0**, MEMORY_TO_MEMORY
- PeriphInc/MemInc 활성, **PDATAALIGN/MDATAALIGN = WORD(32비트)**
- **FIFOMode = DISABLE**, MemBurst/PeriphBurst = **SINGLE (버스트 없음)**
- Priority = HIGH, NVIC 우선순위 0x0F
- 폴링 API(`BSP_SDRAM_ReadData/WriteData`)는 `HAL_SDRAM_Read_32b/Write_32b` → **AHB 32비트 폭 트랜잭션**

[A: `stm32746g_discovery_sdram.c:416-442`, 파일 헤더 주석 45–47행]

> DMA가 FIFO 비활성 + SINGLE 버스트로 설정되어 있으므로, DMA 경로의 SDRAM 처리량은 CPU 직접 접근보다 유리하지 않다. 대역폭이 중요한 전송은 FIFO + INCR4/INCR8 버스트로 재설정해야 한다. [D: 설정값 해석]

### 1.5 대역폭과 레이턴시

**이론 피크 대역폭** [D: 계산]

| SDCLK | 데이터폭 | 피크 | 리프레시 오버헤드 차감 후 |
|---|---|---|---|
| 100 MHz (HCLK 200) | 16 bit = 2 B/clk | **200 MB/s** | ≈ **199.1 MB/s** |
| 108 MHz (HCLK 216) | 16 bit = 2 B/clk | 216 MB/s | ≈ 215.0 MB/s |

리프레시 오버헤드 산출: 1559 SDCLK마다 AUTO REFRESH 1회, 1회당 TRC = 7 사이클 점유 → `7 / 1559 = 0.449%`. [D]

> **실측 정정**: 이 표는 SDRAM 핀 레벨 상한이고 CPU가 볼 수 있는 값이 아니다.
> 실측(HCLK 200 MHz, SDCLK 100 MHz, MPU 미설정 = Device 메모리):
>
> | 연산 | 실측 | 이론 대비 |
> |---|---|---|
> | read32 / read64 / readLDM | 59.70 / 59.56 / 60.76 MB/s | **30%** |
> | write32 / write64 / writeSTM | 194.55 / 194.55 / 194.56 MB/s | **97%** |
> | copyLDM (읽기+쓰기 합산) | 81.46 MB/s | — |
> | SDRAM → SRAM1 | 58.44 MB/s | — |
> | SRAM1 → SDRAM | 194.69 MB/s | — |
>
> **읽기와 쓰기가 같은 속도라는 전제는 이 보드에서 성립하지 않는다.** 쓰기는 스토어 버퍼가
> 흡수해 이론값에 근접하지만, 읽기는 매 로드가 FMC 왕복 레이턴시를 그대로 맞는다.
> (리프레시 오버헤드 산출도 1559가 아니라 1540 SDCLK 주기로 계산해야 한다 → 7/1540 = 0.455%.)

**접근 레이턴시 (FMC 측 하한, AXI/버스 오버헤드 미포함)** [D: RM0385 타이밍 파라미터로부터 유도]

| 시나리오 | SDCLK 사이클 | @100 MHz |
|---|---|---|
| 행 히트(열린 행) 읽기 — CAS만 | 2 (CAS) | 20 ns |
| 행 미스(같은 뱅크, 다른 행) | TRP 2 + TRCD 2 + CAS 2 = 6 | 60 ns |
| 32비트 워드 1개 = SDRAM 16비트 접근 2회 | +1 (연속 열) | +10 ns |

- RBURST=1이므로 CAS latency 기간 동안 `CAS + 1 + RPIPE/2 = 2 + 1 + 0 = 3`개의 데이터가 읽기 FIFO에 선인출된다. 후속 순차 읽기는 FIFO 히트로 처리된다. [A: RM0385 §13.7.3]
- 쓰기 접근 또는 PRECHARGE 명령이 발생하면 읽기 FIFO는 flush된다. [A: RM0385 §13.7.3]
- 행 경계에 도달하며 선인출된 데이터는 FIFO에 저장되지 않는다. [A: RM0385 §13.7.3]

> **정확한 CPU 스톨 사이클 수는 미확인.** RM0385/DS10916은 AXI→FMC 경로의 사이클 단위 레이턴시를 공개하지 않는다. 실측이 필요하다.

> **실측 완료 (이 미확인 항목이 해소됨)**: 포인터 체이싱(32 B 간격 노드를 Sattolo 셔플로 단일
> 순환 연결 → 프리페치 개입 불가)으로 잰 SDRAM 랜덤 접근 레이턴시는 **21.1 ~ 26.1 CPU 사이클
> @200 MHz = 105 ~ 130 ns**다 (워킹셋 1 KB → 512 KB 구간에서 단조 증가, D-cache ON/OFF 무관).
> 32 B 스트라이드 접근은 18.63 cyc/접근.
> 위 표의 20 ns / 60 ns는 SDRAM 핀 레벨 하한이고, **코드 성능 예측에 써야 하는 값은 105~130 ns**다.

### 1.6 캐시 / 실행 가능성 / MPU

**ARMv7-M 기본 주소맵** (ARM DDI 0403E.e Table B3-1) [A]:

| 주소 범위 | 이름 | 메모리 타입 | XN? | 캐시 | 이 보드에서의 의미 |
|---|---|---|---|---|---|
| 0x60000000–0x7FFFFFFF | RAM | Normal | – | **WBWA** | SWP_FMC 리맵 시 SDRAM이 여기로 |
| 0x80000000–0x9FFFFFFF | RAM | Normal | – | **WT** | **QSPI 메모리맵(0x90000000)이 여기** |
| 0xA0000000–0xBFFFFFFF | Device | Device, shareable | **XN** | 없음 | FMC/QSPI 제어 레지스터 |
| 0xC0000000–0xDFFFFFFF | Device | Device, non-shareable | **XN** | 없음 | **SDRAM Bank1/2 기본 위치** |

→ **SDRAM은 기본 상태에서 캐시되지 않고 코드 실행도 불가**하다. AN4667도 "두 SDRAM 뱅크의 기본 영역은 캐시 불가이며, 캐시를 켜도 데이터/명령어가 캐시를 통과하지 않는다"고 명시한다. [B: AN4667]

> **실측 확인 (문서가 맞다)**: MPU를 켜지 않은 프로브 펌웨어에서 SDRAM 순차 읽기가
> D$OFF 59.70 / D$WB 59.71 / D$WT 59.70 MB/s, 포인터 체이싱도 OFF 26.06 / WB 26.06 cyc로
> **세 조건이 완전히 동일**했다. 캐시가 전혀 개입하지 않는다는 직접 증거다.
> 참고로 온디바이스 프로브의 인라인 주석은 "0xC0000000은 기본맵에서 Normal 캐시 가능"이라고
> 잘못 적었는데, **자신이 낸 측정값이 그 주석을 반증한다.** 이 표(ARM DDI 0403E.e Table B3-1)가 맞다.

**해결책 1 — SYSCFG_MEMRMP.SWP_FMC 리맵** [A: RM0385 §7.2.1]

```
SWP_FMC[1:0] (SYSCFG_MEMRMP, offset 0x00, bits 11:10)
  00: 기본. SDRAM Bank1/Bank2 = 0xC0000000 / 0xD0000000, NOR/RAM = 0x60000000
  01: 스왑.  SDRAM Bank1/Bank2 = 0x60000000 / 0x70000000, NOR/PSRAM = 0xC0000000
  10, 11: Reserved
```

RM0385 원문: "These bits are used to swap the FMC SDRAM banks and FMC NOR/PSRAM **in order to enable the code execution from SDRAM Banks without modifying the default MPU attribute**". 0x60000000은 Normal/WBWA/실행 가능이므로 리맵만으로 캐시 + XIP가 모두 해결된다. [A]

**해결책 2 — MPU 리전 (본 프로젝트가 채택)** [A: `Core/Src/main.cc:448-500`]

| 리전 | 베이스 | 크기 | 속성 |
|---|---|---|---|
| 0 | 0x00000000 | 4 GB | NO_ACCESS, non-bufferable, non-cacheable, shareable, TEX0, **SubRegionDisable = 0x87**, XN |
| **1** | **0xC0000000** | **32 MB** | FULL_ACCESS, **non-bufferable + cacheable + non-shareable, TEX level 0 → Write-Through(no write-allocate)**, **실행 허용** |
| 2 | 0xA0000000 | 8 KB | FULL_ACCESS, bufferable, non-cacheable, shareable, XN (FMC 제어 레지스터) |

> **실측 확인**: 실행 중 덤프에서 `MPU_CTRL = 0x00000005`(활성, PRIVDEFENA=1)이고,
> RNR이 가리키던 **리전 2**가 `RBAR = 0xA0000002`, `RASR = 0x13050019`
> (SIZE=12 → 8 KB, B=1/C=0/S=1/TEX=0 → Device·shareable, AP=3 → 전체 RW, XN=1)로
> **위 표와 완전히 일치**했다. 리전 0·1은 RNR이 다른 값을 가리키고 있어 덤프에 잡히지 않았다
> (MPU_RBAR/RASR는 RNR로 선택된 한 리전만 보인다).

> **주의 [D]**: MPU 리전 1의 크기가 **32 MB**로 잡혀 있으나 실제 SDRAM은 **8 MB**뿐이다. 0xC0800000–0xC1FFFFFF 구간에 접근하면 MPU는 통과시키지만 FMC 주소 디코딩상 Reserved라 AHB 에러 → HardFault가 발생한다. 포인터 범위 검증이 없다면 디버깅이 까다로운 결함이 될 수 있다.

> **주의 [D]**: Write-Through 설정이므로 CPU 쓰기는 매번 SDRAM까지 내려간다. LTDC/DMA2D가 같은 SDRAM 프레임버퍼를 읽는 구조에서는 캐시 일관성 문제를 피할 수 있는 대신 쓰기 대역폭이 Write-Back보다 불리하다.

### 1.7 본 프로젝트에서의 SDRAM 사용

- `LCD_FB_START_ADDRESS = 0xC0000000` [A: `stm32746g_discovery_lcd.h:99`]
- `BSP_LCD_LayerDefaultInit(0, LCD_FB_START_ADDRESS)` 및 `LayerDefaultInit(1, LCD_FB_START_ADDRESS + X*Y*4)` → 480 × 272 × 4 B = **522,240 B/레이어**, 2레이어 = **1,044,480 B (약 1.0 MB)** 를 프레임버퍼가 점유 [A: `main.cc:527-528` / D: 계산]
- 나머지 약 **7.3 MB**가 애플리케이션 데이터용으로 남는다 [D: 8,388,608 − 1,044,480 = 7,344,128 B]
- SNN 추론 코드/가중치는 SDRAM이 아닌 내부 Flash/SRAM에 배치되어 있다(링커 스크립트 `STM32F746NGHX_FLASH.ld`에 SDRAM 섹션 없음) [A: 프로젝트 파일 확인]

### 1.8 관련 에라타

| 에라타 | 내용 | 워크어라운드 | 신뢰도 |
|---|---|---|---|
| ES0290 FMC — 쓰기 FIFO 비활성 시 데이터 손상 | 쓰기 접근 중 새 읽기 접근이 오면 FMC가 읽기를 허용하고 FIFO가 빌 때까지 대기하는데, FIFO empty 이벤트 직후 1사이클 이내에 또 다른 읽기가 오면 반환 데이터가 손상된다. **FMC_BCR1의 WFDIS 비트가 set(=쓰기 FIFO 비활성)일 때만 발생.** | **쓰기 FIFO를 활성화(WFDIS = 0, 리셋 기본값)** | [C] |
| ES0290 QUADSPI — 읽기 전송 종료 시 FIFO에 여분 데이터 | 인다이렉트 모드 + AHB/2 클럭 + quad 모드 + DDR 모드 조합에서 발생 | BUSY 플래그가 내려갈 때까지 여분 데이터를 읽어내고 버림 | [C] |

> [C] 사유: ES0290 PDF를 st.com에서 직접 내려받지 못해 검색 요약본으로만 확인. **원문 대조 필요.**
> 본 보드/BSP는 WFDIS를 건드리지 않아 쓰기 FIFO가 활성 상태(기본값)이며, QSPI도 SDR 메모리맵 모드를 쓰므로 두 에라타 모두 해당하지 않는다. [D]

---

## 2. Quad-SPI NOR Flash (Micron N25Q128A13EF840E)

> **실측 정정 — 이 절 전체의 전제가 이 보드에는 맞지 않는다.**
> 실측 JEDEC ID = **`0xEF4018`** → **Winbond W25Q128JV 계열 16 MB**. Micron N25Q128A가 아니다
> (Micron이면 `0x20BA18`). 이 보드는 `06_board.md` §1.2의 **C-03 / `$CU4` 리비전**이다.
> - **그대로 유효한 것**: 용량 16 MB, 4 KB 서브섹터 / 64 KB 섹터 / 256 B 페이지 구조,
>   0x9F·0xEB 등 공통 SPI NOR 명령, STM32 QUADSPI 페리페럴 사양(§2.4), BSP 설정값(§2.5).
>   ST BSP는 N25Q 전제로 작성됐는데도 초기화·메모리맵 읽기가 정상 동작했다(실측 확인).
> - **이 보드에 적용되지 않는 것**: §2.2의 더미 사이클별 최대 주파수 표, §2.3의 소비 전류·
>   tPP/tSSE/tSE/tBE·AC 타이밍, NVCR/VCR 비트 배치, 상태 레지스터 세부 비트.
>   **QSPI에 쓰기를 하는 코드라면 Winbond W25Q128JV 데이터시트를 기준으로 다시 검증해야 한다.**

### 2.1 디바이스 사양

| 항목 | 값 | 신뢰도 | 출처 |
|---|---|---|---|
| 부품번호 | N25Q128A13EF840E (MICRON) | [A] | UM1907 §6.12, Figure 24 |
| 용량 | **128 Mbit = 16 MByte (0x1000000)** | [A] | N25Q128 DS / `n25q128a.h:73` |
| 섹터 | **64 KB × 256개** (0x10000) | [A] | N25Q128 DS / `n25q128a.h:74` |
| 서브섹터 | **4 KB × 4096개** (0x1000) | [A] | N25Q128 DS / `n25q128a.h:75` |
| 페이지 | **256 B × 65536개** (0x100) | [A] | N25Q128 DS / `n25q128a.h:76` |
| 최대 클럭 fC | **108 MHz** (Extended SPI / DIO-SPI / QIO-SPI 전 명령, READ 0x03 제외) | [A] | N25Q128 DS Table 36 |
| 최대 클럭 fR | **54 MHz** (기본 READ 명령) | [A] | N25Q128 DS Table 36 |
| 전원 | 2.7 V ~ 3.6 V 단일 | [A] | N25Q128 DS Features |
| 내구성 | 섹터당 100,000회 이상 program/erase | [A] | N25Q128 DS Features |
| 데이터 보존 | 20년 이상 | [A] | N25Q128 DS Features |
| JEDEC 시그니처 | BA18h (2바이트) | [A] | N25Q128 DS Features |
| XIP | 지원 (NVCR/VCR로 설정, POR 직후 XIP 진입 가능) | [A] | N25Q128 DS §6.2.2 |
| 패키지 코드 `F8` | VDFPN8 8 × 6 mm (MLP8) | [A] | N25Q128 DS Packages |

### 2.2 지원 프로토콜과 더미 사이클

N25Q128 DS **Table 4 — 더미 클럭 사이클별 최대 동작 주파수 (MHz)** [A]:

| 더미 클럭 | FASTREAD (1-1-1) | DOFR (1-1-2) | DIOFR (1-2-2) | QOFR (1-1-4) | **QIOFR (1-4-4)** |
|---|---|---|---|---|---|
| 1 | 54 | 50 | 39 | 43 | 20 |
| 2 | 95 | 85 | 59 | 56 | 39 |
| 3 | 105 | 95 | 75 | 70 | 49 |
| 4 | 108 | 105 | 88 | 83 | 59 |
| 5 | 108 | 108 | 94 | 94 | 69 |
| 6 | 108 | 108 | 105 | 105 | 78 |
| 7 | 108 | 108 | 108 | 108 | 86 |
| 8 | 108 | 108 | 108 | 108 | 95 |
| 9 | 108 | 108 | 108 | 108 | 105 |
| **10** | 108 | 108 | 108 | 108 | **108** |

더미 사이클은 NVCR[15:12](비휘발) / VCR[7:4](휘발)로 **1~15 범위**에서 설정 가능. 기본값은 108 MHz에서 안전하게 동작하도록 설정되어 있으며, 더미 사이클이 부족하면 **잘못된 데이터를 읽는다**. [A: N25Q128 DS §6.2.1]

주요 명령 오피코드 [A: `n25q128a.h:100-139`]:

| 명령 | 오피코드 | BSP 사용 |
|---|---|---|
| FAST READ | 0x0B | – |
| DUAL OUTPUT FAST READ | 0x3B | – |
| DUAL I/O FAST READ | 0xBB | – |
| QUAD OUTPUT FAST READ | 0x6B | – |
| **QUAD I/O FAST READ (QIOFR)** | **0xEB** | **읽기 / 메모리맵** |
| **EXT QUAD IN FAST PROGRAM** | **0x12** | **쓰기** |
| **SUBSECTOR ERASE** | **0x20** | **블록 소거** |
| **BULK ERASE** | **0xC7** | **칩 소거** |

### 2.3 소비 전류 및 동작 시간 (N25Q128 DS Table 35 / Table 36)

| 심볼 | 항목 | 조건 | 최대 | 단위 | 신뢰도 |
|---|---|---|---|---|---|
| ICC1 | Standby current | S = VCC | **100** | µA | [A] |
| ICC3 | Fast Read **Single I/O** | 108 MHz | **15** | mA | [A] |
| ICC3 | Fast Read Single I/O | 54 MHz | 6 | mA | [A] |
| ICC3 | Fast Read **Dual I/O** | 108 MHz | **18** | mA | [A] |
| ICC3 | Fast Read **Quad I/O** | 108 MHz | **20** | mA | [A] |
| ICC4 | Page Program (Single/Dual/Quad) | – | 20 | mA | [A] |
| ICC6 | Sector Erase | – | 20 | mA | [A] |

| 심볼 | 항목 | Typ | Max | 단위 | 신뢰도 |
|---|---|---|---|---|---|
| tPP | Page program (n 바이트) | `int(n/8) × 0.015` | **5** | ms | [A] |
| tSSE | Subsector erase (4 KB) | **0.2** | **2** | s | [A] |
| tSE | Sector erase (64 KB) | **0.7** | **3** | s | [A] |
| tBE | Bulk erase (16 MB) | **170** | **250** | s | [A] |
| tWVCR | Write volatile config register | 40 | – | ns | [A] |
| tWNVCR | Write non-volatile config register | 0.2 | 3 | s | [A] |

AC 타이밍 일부 [A: N25Q128 DS Table 36]: tCH/tCL(min) = 4 ns, tSLCH/tCHSL = 4 ns, tDVCH(setup) = 2 ns, tCHDX(hold) = 3 ns, tSHSL1(정상 read 후 nCS deselect) = **20 ns**, tSHSL2(그 외) = **50 ns**.

BSP는 `ChipSelectHighTime = QSPI_CS_HIGH_TIME_6_CYCLE`로 nCS를 최소 6클럭 유지하는데, 100 MHz에서 60 ns > 50 ns이므로 tSHSL2까지 만족한다. [D: 계산, BSP 주석 "Min 50ns for nonRead"와 일치]

### 2.4 STM32 QUADSPI 페리페럴 사양

| 항목 | 값 | 신뢰도 | 출처 |
|---|---|---|---|
| 메모리맵 베이스 | **0x90000000** (AXI 경유) | [A] | `stm32f746xx.h:1134` (`QSPI_BASE`), RM0385 |
| 제어 레지스터 베이스 | **0xA0001000** | [A] | `stm32f746xx.h:1136` (`QSPI_R_BASE`), RM0385 Table 1 |
| 메모리맵 최대 크기 | **256 MB** (FSIZE가 더 커도 메모리맵은 256 MB 제한) | [A] | RM0385 §14.3.6, §14.3.7 |
| 용량 계산식 | `바이트 수 = 2^(FSIZE+1)` | [A] | RM0385 §14.3.7 |
| 지원 접근 폭 | 8 / 16 / 32 비트 | [A] | RM0385 §14.2, DS10916 §2.9 |
| 코드 실행 | **지원 (XIP)** — 다음 주소를 선인출 | [A] | RM0385 §14.3.6, DS10916 §2.9 |
| FSIZE 범위 밖 접근 | AHB 에러 → CPU면 HardFault, DMA면 transfer error + 채널 자동 비활성 | [A] | RM0385 §14.3.6 |
| 최대 QSPI 클럭 (SDR) | **108 MHz** @ 2.7 V ≤ VDD < 3.6 V, CL = 20 pF | [A] | DS10916 Table 108 |
| 최대 QSPI 클럭 (SDR) | 100 MHz @ 1.71 V < VDD < 3.6 V, CL = 15 pF | [A] | DS10916 Table 108 |
| 최대 QSPI 클럭 (DDR) | 80 MHz | [A] | DS10916 Table 109 |
| 데이터 입력 setup/hold | ts(IN) = 1 ns / th(IN) = 3 ns | [A] | DS10916 Table 108 |
| 데이터 출력 valid/hold | tv(OUT) max 3 ns (2.7–3.6 V) / th(OUT) min 0 ns | [A] | DS10916 Table 108 |
| 프리페치 정지 | 기본적으로 nCS를 낮게 유지한 채 프리페치를 멈추지 않음. `TCEN`(QUADSPI_CR[3]) + `TIMEOUT`(QUADSPI_LPTR)로 nCS 해제 가능 | [A] | RM0385 §14.3.6 |

### 2.5 BSP(stm32746g_discovery_qspi.c)가 설정하는 실제 값

```
QSPIHandle.Init.ClockPrescaler     = 1;   /* QSPI freq = HCLK/(1+1) */
QSPIHandle.Init.FifoThreshold      = 4;
QSPIHandle.Init.SampleShifting     = QSPI_SAMPLE_SHIFTING_HALFCYCLE;
QSPIHandle.Init.FlashSize          = POSITION_VAL(0x1000000) - 1 = 23;
QSPIHandle.Init.ChipSelectHighTime = QSPI_CS_HIGH_TIME_6_CYCLE;  /* CSHT = 5 → 6클럭 */
QSPIHandle.Init.ClockMode          = QSPI_CLOCK_MODE_0;          /* nCS 해제 중 CLK Low */
```
[A: `stm32746g_discovery_qspi.c:123-128`, HAL 상수는 `stm32f7xx_hal_qspi.h:249,262,272`]

**QUADSPI_DCR = 0x00170500** — FSIZE(23) << 16 | CSHT(5) << 8 | CKMODE(0). 16 MB, nCS high 6클럭, 모드 0. [D: HAL `stm32f7xx_hal_qspi.c:367-369`의 조합식으로 계산]

**QUADSPI_CR** — PRESCALER = 1 (bits 31:24) → **QSPI_CK = HCLK/2**, SSHIFT = 1 (bit 4, 반클럭 샘플 시프트), FSEL = 0(Bank1), DFM = 0(단일 플래시). [A: `stm32f7xx_hal_qspi.c:362-364`]

- 본 프로젝트 HCLK = 200 MHz → **QSPI_CK = 100 MHz**
- HCLK 216 MHz 구성이면 **108 MHz** (BSP 주석 "QSPI freq = 216 MHz/(1+1) = 108 MHz")

**메모리맵 모드 커맨드** (`BSP_QSPI_EnableMemoryMappedMode`) [A: `stm32746g_discovery_qspi.c:455-470`]:

| 필드 | 값 |
|---|---|
| InstructionMode | `QSPI_INSTRUCTION_1_LINE` |
| Instruction | **0xEB (QUAD_INOUT_FAST_READ)** |
| AddressMode | `QSPI_ADDRESS_4_LINES` |
| AddressSize | `QSPI_ADDRESS_24_BITS` |
| AlternateByteMode | NONE |
| DataMode | `QSPI_DATA_4_LINES` |
| **DummyCycles** | **10** (`N25Q128A_DUMMY_CYCLES_READ_QUAD`) |
| DdrMode | DISABLE (SDR) |
| SIOOMode | `QSPI_SIOO_INST_EVERY_CMD` (매 커맨드마다 명령어 재전송) |
| TimeOutActivation | `QSPI_TIMEOUT_COUNTER_DISABLE`, TimeOutPeriod = 0 |

→ 즉 **1-4-4 모드**(명령 1선 / 주소 4선 / 데이터 4선). 순수 1-1-1도 아니고 완전한 4-4-4도 아니다. [A]

초기화 시 `QSPI_DummyCyclesCfg()`가 `READ_VOL_CFG_REG_CMD`로 VCR을 읽어 `N25Q128A_VCR_NB_DUMMY(0xF0)` 필드를 **10**으로 갱신 후 `WRITE_VOL_CFG_REG_CMD`로 기록한다. [A: `stm32746g_discovery_qspi.c:631-680`]

**핀 배치** [A: `stm32746g_discovery_qspi.h:71-82` / UM1907 핀 표]:

| 신호 | 핀 | AF |
|---|---|---|
| QUADSPI_CLK | **PB2** (BOOT1 겸용) | AF9 |
| QUADSPI_BK1_NCS | **PB6** | **AF10** |
| QUADSPI_BK1_IO0 | **PD11** | AF9 |
| QUADSPI_BK1_IO1 | **PD12** | AF9 |
| QUADSPI_BK1_IO2 | **PE2** | AF9 |
| QUADSPI_BK1_IO3 | **PD13** | AF9 |

모두 `GPIO_MODE_AF_PP`, `GPIO_SPEED_HIGH`. NCS만 Pull-up, 나머지는 NOPULL. [A: `stm32746g_discovery_qspi.c:515-545`]

### 2.6 QSPI 읽기 대역폭 계산 [D]

**Quad SDR = 4 bit/clock = 0.5 byte/clock**

| QSPI_CK | 순차 스트리밍 피크 | 32 B 캐시라인 랜덤 리드 |
|---|---|---|
| 100 MHz (HCLK 200) | **50.0 MB/s** | **≈36.4 MB/s** |
| 108 MHz (HCLK 216) | 54.0 MB/s | ≈39.3 MB/s |

트랜잭션당 고정 오버헤드:

```
명령 0xEB, 1-line        =  8 클럭
주소 24 bit, 4-line      =  6 클럭
더미                     = 10 클럭
------------------------------------
첫 데이터 전 오버헤드    = 24 클럭  (100 MHz → 240 ns)

32 바이트 데이터, 4-line = 64 클럭
합계 88 클럭 → 32 B / 880 ns = 36.4 MB/s
```

- SIOO가 꺼져 있어(`QSPI_SIOO_INST_EVERY_CMD`) 매 트랜잭션마다 8클럭 명령을 재전송한다. SIOO를 켜면 오버헤드가 24 → 16클럭으로 줄어 캐시라인 대역폭이 **≈40 MB/s**로 개선된다. [D: 계산]
- 메모리맵 XIP 프리페치가 계속 이어지는 완전 순차 접근에서는 50 MB/s에 근접한다. [D]
- 참고: 같은 조건에서 SDRAM(200 MB/s)이 QSPI(50 MB/s)보다 **약 4배** 빠르다. [D]

> **실측 확인 — 이 절의 계산은 거의 그대로 맞았다.**
>
> | 항목 | 문서 계산 | 실측 |
> |---|---|---|
> | 순차 스트리밍 (100 MHz) | 50.0 MB/s | **49.97 MB/s** (64 KB 워킹셋 read32는 51.69) |
> | 32 B 캐시라인 랜덤 리드 | 880 ns / 36.4 MB/s | **987.7 ns**(197.54 cyc @200 MHz) / ≈32.4 MB/s |
> | 캐시 없이 4 B 랜덤 리드 | (24 + 8) 클럭 ≈ 320 ns | **471 ns**(94.22 cyc) |
>
> 캐시라인 랜덤 리드가 D$OFF 94.22 cyc → D$WB 197.54 cyc로 바뀐다는 것은
> **0x90000000이 실제로 Normal/Write-Through로 취급되어 32 B 라인 채우기가 일어난다**는 뜻이다
> (문서 §2.7의 주장이 실측으로 확인됨). 차이분은 전부 AXI/AHB 왕복 오버헤드다.
>
> 마지막 줄의 "SDRAM이 QSPI보다 4배 빠르다"는 **읽기 기준으로는 틀렸다** —
> 실측 SDRAM 읽기 59.7 MB/s vs QSPI 읽기 50.0 MB/s로 **1.2배**밖에 차이 나지 않는다.
> 4배 차이는 쓰기(194.6 vs 사실상 불가) 쪽에서만 유효하다.

### 2.7 QSPI 사용 시 제약

- **쓰기가 극도로 느리다**: 페이지(256 B) 단위 프로그램에 typ `int(256/8) × 0.015 = 0.48 ms`, max 5 ms. 4 KB 서브섹터 소거는 typ 0.2 s / max 2 s. **런타임에 자주 갱신되는 데이터에는 부적합**. [A: N25Q128 DS Table 36 / D: 계산]
- **메모리맵 모드에서는 읽기만 가능**. 쓰기는 인다이렉트 모드로 전환해야 하고, 그동안 XIP 코드 실행이 중단된다. [A: RM0385 §14.3 / BSP 구조]
- **0x90000000 영역은 기본적으로 Normal + Write-Through 캐시 + 실행 가능**이므로 SDRAM과 달리 MPU 조정 없이도 XIP가 동작한다. [A: ARM DDI 0403E.e Table B3-1]
- BSP `BSP_QSPI_Write()`는 페이지 경계를 넘지 않도록 `N25Q128A_PAGE_SIZE - (WriteAddr % N25Q128A_PAGE_SIZE)`로 분할한다. [A: `stm32746g_discovery_qspi.c:231,288`]
- **더미 사이클 10은 108 MHz까지만 보장**된다. QSPI_CK를 더 올릴 수는 없다(HCLK/2가 최소 분주이고 HCLK 최대가 216 MHz). [A: N25Q128 DS Table 4 + RM0385]
- 본 프로젝트에서 QSPI는 **초기화조차 하지 않는다** (`MX_QUADSPI_Init()`이 주석 처리됨). SNN 가중치는 내부 Flash에 있다. [A: `Core/Src/main.cc:132`]

---

## 3. microSD 카드 슬롯 (SDMMC1)

### 3.1 보드 배선

UM1907 §6.9: "The 2-Gbyte (or more) microSD card connected to the **SDMMC1** port of the STM32F746NGH6 is supported by the board." [A]

**커넥터 CN3 핀맵** (UM1907 Table 7) [A]:

| 핀 | 신호 | 핀 | 신호 |
|---|---|---|---|
| 1 | SDMMC_D2 (**PC10**) | 6 | Vss/GND |
| 2 | SDMMC_D3 (**PC11**) | 7 | SDMMC_D0 (**PC8**) |
| 3 | SDMMC_CMD (**PD2**) | 8 | SDMMC_D1 (**PC9**) |
| 5 | SDMMC_CK (**PC12**) | 10 | MicroSDcard_detect (**PC13**) |

→ **D0–D3만 배선 = 4비트 버스 최대**. MMC 8비트 모드(D4–D7)는 보드에 없다. [A: UM1907 Table 7 / D: 8비트 신호 부재 확인]

### 3.2 SDMMC 페리페럴 사양

| 항목 | 값 | 신뢰도 | 출처 |
|---|---|---|---|
| MMC 규격 | MultiMediaCard System Specification **V4.2** 준수, 1/4/8비트 버스 | [A] | RM0385 §35.1, DS10916 §2.30 |
| SD 메모리카드 규격 | **SD Memory Card Specification V2.0** 준수 | [A] | RM0385 §35.1 |
| SD I/O 규격 | SD I/O Card Specification V2.0, 1/4비트 | [A] | RM0385 §35.1 |
| 최대 데이터 전송 | **50 MHz (8비트 모드)** | [A] | RM0385 §35.1 |
| fPP (데이터 전송 모드 클럭) | 0 ~ **50 MHz** | [A] | DS10916 Table 112 |
| SDMMC_CK 생성식 | `SDMMC_CK = SDMMCCLK / (CLKDIV + 2)` | [A] | RM0385 §35.x |
| SDMMC 커널 클럭 | PLL48CLK / PLLSAI48CLK 선택 (`RCC_DCKCFGR2.SDMMC1SEL`), 통상 **48 MHz** | [A] | RM0385 §5 RCC, §35.8 |
| SPI 호환 모드 | **미지원** | [A] | RM0385 §35.1 Note 1 |
| DMA | 지원 | [A] | DS10916 §2.30 |
| 동시 지원 카드 | SD/SDIO/MMC4.2 카드 1장 (MMC4.1 이하는 스택 가능) | [A] | RM0385 §35.1 |

**DS10916 Table 112 (VDD = 2.7~3.6 V) 동적 특성** [A]:

| 심볼 | 파라미터 | 조건 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|---|
| fPP | 데이터 전송 모드 클럭 주파수 | – | 0 | – | **50** | MHz |
| – | SDMMC_CK / fPCLK2 주파수 비 | – | – | – | **8/3** | – |
| tW(CKL) | 클럭 low 시간 | fPP = 50 MHz | 9.5 | 10.5 | – | ns |
| tW(CKH) | 클럭 high 시간 | fPP = 50 MHz | 8.5 | 9.5 | – | ns |
| tISU | 입력 setup (HS) | fPP = 50 MHz | 2.5 | – | – | ns |
| tIH | 입력 hold (HS) | fPP = 50 MHz | 3 | – | – | ns |
| tOV | 출력 valid (HS) | fPP = 50 MHz | – | 11.5 | 12 | ns |
| tOH | 출력 hold (HS) | fPP = 50 MHz | 10.5 | – | – | ns |
| tISUD | 입력 setup (SD default) | fPP = 25 MHz | 2 | – | – | ns |
| tOVD | 출력 valid (SD default) | fPP = 25 MHz | – | 1.5 | 2 | ns |

**대역폭 산출 [D]**: 4비트 버스 × SDMMC_CK. SDMMCCLK = 48 MHz, CLKDIV = 0이면 SDMMC_CK = 24 MHz → `24 MHz × 4 bit / 8 = 12 MB/s` 이론 피크. CLKCR의 BYPASS를 켜면 SDMMC_CK = SDMMCCLK = 48 MHz → 24 MB/s 이론 피크. 실제 SD 카드/파일시스템 오버헤드로 그보다 훨씬 낮다.

### 3.3 본 프로젝트 상태

`MX_SDMMC1_SD_Init()` 호출이 주석 처리되어 **SD카드 미사용**. `Core/Src/sd_diskio.c`는 존재하나 빌드/초기화 경로에서 비활성. [A: `Core/Src/main.cc:135,208`]

> **실측 정정**: 현재 소스는 확실히 주석 처리되어 있으나, **보드에 실제로 올라가 있던 펌웨어는
> SDMMC1을 초기화한 상태**였다. 실행 중 ST-LINK 덤프에서 `RCC_APB2ENR`에 **SDMMC1EN=1**이고
> **PC8/PC9/PC10/PC11/PC12 + PD2가 AF12(SDMMC1_D0–D3 / CK / CMD)** 로 설정되어 있었다.
> AF12 핀 총계 44개도 정확히 `FMC 38 + SDMMC1 6`으로 떨어진다.
> `HAL_SD_MspInit()`은 `HAL_SD_Init()`에서만 불리므로 우연한 설정이 아니다.
> → 플래시 이미지가 현재 소스 트리와 다른 시점의 빌드였을 가능성이 높다.
>
> 또한 **microSD 카드가 실제로 슬롯에 꽂혀 있다** (PC13 = 풀업 입력, IDR = 0).

---

## 4. 세 외부 메모리 비교 요약

| 항목 | SDRAM | QSPI Flash | microSD |
|---|---|---|---|
| 부품 | MT48LC4M32B2B5-6A | N25Q128A13EF840E | (사용자 카드) |
| 용량 | **8 MB** (다이 16 MB 중 절반만 배선) | **16 MB** | 2 GB 이상 |
| 주소 | **0xC0000000** (8 MB 유효) | **0x90000000** (16 MB 유효) | 메모리맵 아님 (SDMMC1 레지스터/DMA) |
| 인터페이스 | FMC SDRAM Bank1, 16비트 병렬 | QUADSPI 1-4-4 SDR | SDMMC1 4비트 |
| 클럭 (본 프로젝트) | **100 MHz** (HCLK/2) | **100 MHz** (HCLK/2) | 미사용 |
| 이론 피크 대역폭 | **200 MB/s** | **50 MB/s** | ~12–24 MB/s |
| 랜덤 접근 레이턴시 | 20 ns(행 히트) ~ 60 ns(행 미스) | ≈240 ns (24클럭 오버헤드) | ms 단위 |
| 휘발성 | 휘발성 | 비휘발성 | 비휘발성 |
| 쓰기 | 읽기와 동일 속도 | 페이지 0.48 ms~5 ms, 소거 필수 | 블록 단위, ms |
| 기본 캐시 속성 | **없음 (Device, XN)** — MPU/SWP_FMC 필요 | **Normal, Write-Through** | 해당 없음 |
| 기본 코드 실행 | **불가 (XN)** | **가능 (XIP)** | 불가 |
| 본 프로젝트 사용 | LCD 프레임버퍼 1.0 MB | **미사용** | **미사용** |

[모든 셀: 위 각 절의 근거를 종합. 계산 항목은 [D]]

> **실측 정정 (이 표에서 고쳐야 할 셀)**
>
> | 항목 | 표의 값 | 실측 |
> |---|---|---|
> | QSPI 부품 | N25Q128A13EF840E (Micron) | **Winbond W25Q128JV 계열** (JEDEC `0xEF4018`) |
> | SDRAM 이론 피크 | 200 MB/s | 상한은 맞으나 **CPU 읽기 실측 59.7 / 쓰기 194.6 MB/s** |
> | QSPI 이론 피크 | 50 MB/s | **실측 49.97 MB/s — 그대로 맞음** |
> | SDRAM 랜덤 레이턴시 | 20 ns(행 히트) ~ 60 ns(행 미스) | **CPU 관측 105 ~ 130 ns** |
> | QSPI 랜덤 레이턴시 | ≈240 ns | **471 ns**(4 B 랜덤) / **988 ns**(32 B 라인 채우기) |
> | SDRAM 쓰기 | "읽기와 동일 속도" | **읽기의 3.3배 빠르다** (194.6 vs 59.7 MB/s) |
> | 클럭 (본 프로젝트) | 100 MHz (HCLK/2) | **맞음. 단 HCLK는 200 MHz이지 216 MHz가 아니다** |
> | microSD 본 프로젝트 사용 | 미사용 | 덤프 시점 이미지는 **SDMMC1 초기화됨**, 카드도 삽입되어 있음 |

---

## 5. 코드/데이터 배치 시 실무 지침 (도출) [D]

1. **SNN 가중치 같은 읽기 전용 대용량 데이터**: QSPI(16 MB)가 용량은 크지만 대역폭이 SDRAM의 1/4이다. 다만 XIP 영역이 기본 Write-Through 캐시라 재사용성이 높은 데이터는 L1 D-Cache(F746는 4 KB+4 KB) 히트로 상당 부분 만회된다. 한 번 읽고 버리는 스트리밍 데이터라면 QSPI가 불리하다.
2. **중간 액티베이션/버퍼**: SDRAM이 적합하나, MPU로 캐시 속성을 반드시 설정해야 성능이 나온다. 설정하지 않으면 Device 메모리로 취급되어 모든 접근이 언캐시드 + 재정렬 금지가 되어 심각하게 느려진다.
3. **SDRAM에서 코드 실행**이 필요하면 `SWP_FMC = 01` 리맵(0x60000000, WBWA 캐시)이 MPU 조작보다 간단하고 캐시 정책도 더 유리하다. 단, 그러면 NOR/PSRAM이 0xC0000000으로 이동한다.
4. **LTDC 프레임버퍼와 CPU가 같은 SDRAM 영역을 공유**할 때는 Write-Through(현 설정) 또는 명시적 `SCB_CleanDCache_by_Addr()`가 필요하다. Write-Back으로 바꾸면 화면 깨짐이 발생할 수 있다.
5. **SDRAM 8 MB 경계**: MPU 리전이 32 MB로 잡혀 있으므로 8 MB 초과 주소 접근은 MPU를 통과한 뒤 FMC에서 AHB 에러 → HardFault가 된다. 리전 크기를 8 MB로 줄이면 MemManage 폴트로 더 명확하게 잡힌다.
6. **DMA로 SDRAM 대량 전송** 시 BSP 기본 DMA 설정(FIFO 비활성, SINGLE 버스트)은 비효율적이다. FIFO + INCR4/INCR8로 바꿔야 SDRAM 페이지 히트를 활용할 수 있다.

---

## 출처

### ST 공식 문서 (직접 본문 확인)

- **UM1907 Rev 3 (DocID027942)** — *Discovery kit for STM32F7 Series with STM32F746NG MCU*. §6.9 microSD card, §6.11 SDRAM memory, §6.12 Quad-SPI Nor Flash memory, §7.7 Table 7 microSD connector CN3, Figure 24/26 회로도.
  https://www.micro-semiconductor.com/datasheet/8c-STM32F746G-DISCO.pdf
  (최신판: https://www.st.com/resource/en/user_manual/um1907-discovery-kit-with-stm32f746ng-mcu-stmicroelectronics.pdf — st.com 직접 접근은 403/타임아웃)
- **RM0385 Rev 5 (DocID026670)** — *STM32F75xxx and STM32F74xxx advanced ARM-based 32-bit MCUs Reference manual*. §7.2.1 SYSCFG_MEMRMP (SWP_FMC), §13.4 External device address mapping, §13.4.3 Table 50/51/53, §13.7.1 SDRAM controller main features, §13.7.3 read FIFO/RBURST, §13.7.5 FMC_SDCR/FMC_SDTR/FMC_SDRTR, §14.2 QUADSPI main features, §14.3.6 memory-mapped mode, §14.3.7 FSIZE/CSHT, §35.1 SDMMC main features.
  https://indico.kit.edu/event/460/contributions/5488/attachments/2862/4152/reference-manual.pdf
  (공식: https://www.st.com/resource/en/reference_manual/rm0385-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf)
- **DS10916 / DocID027590 Rev 1** — *STM32F745xx STM32F746xx Datasheet*. §2.8 FMC (최대 FMC_SDCLK = HCLK/2), §2.9 QUADSPI, §2.30 SDMMC, §5.3 Table 104 SDRAM read timings, Table 106 SDRAM write timings, "SDRAM waveforms and timings" 조건(VDD별 최대 FMC_SDCLK), Table 108/109 Quad SPI characteristics, §5.3.31 Table 112 SD/MMC characteristics.
  https://www.st.com/resource/en/datasheet/stm32f746ng.pdf
- **AN4667 (DocID027643)** — *STM32F7 Series system architecture and performance*. SDRAM 뱅크 기본 비캐시 / 0x60000000·0x70000000 리맵 / MPU 대안. **PDF 원문 직접 열람 실패, 검색 요약본으로만 확인 → [B]**
  https://www.st.com/resource/en/application_note/an4667-stm32f7-series-system-architecture-and-performance-stmicroelectronics.pdf
- **ES0290 Rev 7** — *STM32F74xxx STM32F75xxx Errata sheet*. FMC 쓰기 FIFO 비활성 시 데이터 손상, QUADSPI 읽기 종료 시 FIFO 여분 데이터. **PDF 원문 직접 열람 실패 → [C]**
  https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf

### ARM 공식 문서 (직접 본문 확인)

- **ARM DDI 0403E.e** — *ARMv7-M Architecture Reference Manual*. §B3.1 The system address map, **Table B3-1 Armv7-M address map** (0xC0000000–0xDFFFFFFF = Device / non-shareable / XN / 캐시 없음, 0x80000000–0x9FFFFFFF = Normal / WT).
  https://www.pjrc.com/teensy/DDI0403Ee_arm_v7m_ref_manual.pdf
- **ARM DDI 0489F** — *Cortex-M7 Technical Reference Manual*. §2.4 Figure 2-1 System address map, Table 2-1 Memory regions (External Device는 AXIM 경유).

### 부품 벤더 데이터시트 (직접 본문 확인)

- **Micron 128Mb: x32 SDRAM MT48LC4M32B2 — 1 Meg x 32 x 4 Banks**, Rev. L 1/09 EN (PDF: 09005aef80872800). Table 1 Key Timing Parameters, Table 2 Configurations, Table 3 Part Numbers, Table 7 CAS Latency, Table 18 Electrical Characteristics and Recommended AC Operating Conditions, Table 19 AC Functional Characteristics.
  https://datasheet.octopart.com/MT48LC4M32B2P-7-IT:G-Micron-datasheet-5314496.pdf
- **Micron Automotive SDR SDRAM MT48LC4M32B2** (`-6A3` = 6 ns/167 MHz 표기 확인).
  https://www.farnell.com/datasheets/3151564.pdf
- **Micron N25Q128 — 128Mb 3V, Multiple I/O, 4KB Subsector Erase, XiP Enabled, Serial Flash Memory with 108 MHz SPI Bus Interface**. Features, §6.2.1 Table 4 Maximum operative frequency by dummy clock cycles, Table 35 DC Characteristics, Table 36 AC Characteristics.
  https://docs.rs-online.com/0a63/0900766b80f7f8a8.pdf
  (Mouser 미러: https://www.mouser.com/datasheet/2/671/n25q_128mb_3v_65nm-1283709.pdf)
- Micron 부품 카탈로그 페이지 (사양 미표시): https://www.micron.com/products/dram/sdram/part-catalog/mt48lc4m32b2b5-6a

### 프로젝트 내 ST 공식 BSP/HAL 소스 (레지스터 값 산출 근거)

- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_sdram.c`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_sdram.h`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_qspi.c`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_qspi.h`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/BSP/Components/n25q128a/n25q128a.h`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.c`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/STM32F7xx_HAL_Driver/Inc/stm32f7xx_ll_fmc.h`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_qspi.c`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h`
- `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Core/Src/main.cc` (SystemClock_Config, MPU_Config)

---

## 미확인 / 추가 조사 필요

0. **~~QSPI 부품이 N25Q128A라는 전제~~** → **반증됨(2026-07-27)**: 실측 JEDEC ID `0xEF4018` = Winbond W25Q128JV. 이 보드는 C-03/`$CU4` 리비전이다. **§2의 부품 고유 수치를 Winbond 데이터시트로 다시 써야 한다.**
1. **MT48LC4M32B2B5-6A의 `-6A` 접미사 정확한 정의** — 열람한 Micron 상용 데이터시트(Rev. L)의 Options 표에는 `-6`(6 ns/166 MHz)와 `-7`만 있고 `-6A`는 없다. 자동차용 데이터시트에는 `-6A3`(6 ns/167 MHz)가 있다. `-6A`가 `-6`와 동일 타이밍인지, 개정판 접미사인지 **Micron 원본 카탈로그 확인 필요**. 본 문서는 `-6` 등급 타이밍을 적용했다.
2. **ES0290 원문 미확인** — st.com 직접 다운로드가 403/타임아웃으로 실패했다. FMC 쓰기 FIFO 에라타와 QUADSPI 에라타의 정확한 번호(예: 2.x.y), 영향받는 실리콘 리비전(**Rev Z 해당 여부**), 원문 워크어라운드 문구를 확인해야 한다.
3. **AN4667 원문 미확인** — 캐시 관련 서술은 검색 요약으로만 확인했다. 외부 메모리별 실측 성능 표(CoreMark/실행 사이클/대역폭)가 있는지, 있다면 정확한 수치가 무엇인지 확인 필요.
4. ~~**AXI→FMC SDRAM 접근의 CPU 스톨 사이클 실측값**~~ → **해결됨(2026-07-27)**: 포인터 체이싱 실측 **21.1~26.1 cyc @200 MHz = 105~130 ns**. 다만 행 히트/행 미스/뱅크 미스 **분리**는 아직 안 했다(512 B 행 경계를 넘나드는 스트라이드로 측정 가능).
5. ~~**SDRAM 실측 대역폭**~~ → **해결됨(2026-07-27)**: read32 59.70 / write32 194.55 / copyLDM 81.46 MB/s, SDRAM→SRAM1 58.44, SRAM1→SDRAM 194.69 MB/s. **단 MPU 미설정(Device 메모리) 조건이므로, MPU로 Normal/WT를 건 뒤의 값은 여전히 미측정.**
6. ~~**QSPI 실측 대역폭**~~ → **해결됨(2026-07-27)**: 메모리맵 순차 리드 49.97 MB/s(계산 50.0과 일치), 32 B 캐시라인 랜덤 987.7 ns(계산 880 ns). SIOO=1 조건은 여전히 미측정.
7. **DS10916 Table 112의 "SDMMC_CK / fPCLK2 frequency ratio = 8/3"의 해석** — 표기 방향(SDMMC_CK가 fPCLK2의 8/3배까지 허용인지, 반대로 fPCLK2가 SDMMC_CK의 8/3배 이상이어야 하는지)이 원문상 모호하다. 후자로 해석하는 것이 일반적이나 ST 문의 또는 최신 개정판 확인 필요.
8. **`REFRESH_COUNT = 0x603`을 ST가 선택한 근거** — RM0385 공식 계산값은 0x606(1542)인데 BSP는 0x603(1539)이다. 3 사이클 차이의 근거가 마진 확보인지 다른 계산식인지 미확인.
9. **DS10916 Table 106 (SDRAM write timings) 세부 수치** — PDF xref 손상으로 해당 표 본문을 완전히 추출하지 못했다. Table 104(read)만 확보했다.
10. **MB1191 회로도의 SDRAM 종단/직렬 저항 값** — UM1907 텍스트 추출에서 부품값이 소실되었다. 신호 무결성 검토가 필요하면 원본 PDF의 Figure 26을 직접 봐야 한다.
11. **HCLK 216 MHz 구성에서 SDRAM 실동작 여부** — DS10916상 FMC_SDCLK 100 MHz 초과이고 Micron CL=2 규격(100 MHz)도 초과하나, 다수 사례에서 동작한다고 알려져 있다. 본 보드에서 온도/전압 코너 실측 필요.
