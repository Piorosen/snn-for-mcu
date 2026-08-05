# STM32F746NG 페리페럴 인벤토리 (STM32F746G-DISCO)

> 대상: STM32F746NGH6 (TFBGA216, Cortex-M7 r0p1, Device ID 0x449, Rev Z)
> 1차 출처: DS10916 (= DocID027590 Rev 4, 2016-02), RM0385 (= DocID026670 Rev 5, 2016-04), ST CMSIS `stm32f746xx.h`, ST BSP `STM32746G-Discovery`

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거 자료: `info/raw/measured_facts.md`, `info/results/probe_output.txt`,
> `info/raw/decoded_hotplug.md`, `info/raw/decoded_reset.md`, `info/raw/host_side.md`

### 실측으로 확인됨

- **페리페럴 존재 (본문 §1 전체 표)**: 문서가 나열한 베이스 주소로 61회 프로빙 → **60개 버스 응답 OK**. RCC 인에이블 비트 되읽기와 베이스 주소 `ldr.w` 두 방법이 모두 일치.
- **JPEG 코덱 미탑재**: 문서 "F746 미탑재" → 실측 `0x50051000` **RCC 비트 CLEAR0 + 베이스 읽기 BusFault** (두 방법 일치, 61회 중 유일한 실패) → **확정**.
- **동작 클럭**: 문서 "216 MHz가 아니라 200 MHz, PLLM=25/PLLN=400/PLLP=2, PCLK1 50 / PCLK2 100, 6 WS" → 실측 `RCC_PLLCFGR=0x09406419`, `RCC_CFGR=0x0000940A`, `FLASH_ACR=0x00000006`으로 **전부 일치**.
- **TIMPRE=0 가정 (§2.1)**: 문서 "TIMPRE 미설정" → 실측 `RCC_DCKCFGR1` = `0x00000000`(프로브 펌웨어) / `0x00010000`(SNN 펌웨어) 둘 다 **TIMPRE=0** → 문서가 유도한 **APB1 타이머 100 MHz / APB2 타이머 200 MHz 확정**.
- **PLLQ=9 → 44.44 MHz (§11-5)**: 실측 `PLL Q 출력 = 400 MHz / 9 = 44,444,444 Hz` 그대로 확인.
- **SDRAM 8 MB @0xC0000000, 16비트 버스**: 실측 `FMC_SDCR1=0x00001954` (8열/12행/16비트/4뱅크/CAS2/HCLK÷2=100 MHz) → 8 MB, 폴트 경계도 8,192 KB에서 일치.
- **QSPI 16 MB @0x90000000**: 실측 `QUADSPI_DCR` FSIZE=23 → 16 MB, JEDEC 용량코드 `0x18`, 폴트 경계 16,384 KB → **용량은 문서와 일치**(부품은 아래 CONFLICT).
- **메모리 맵 (§8)**: ITCM 16 KB(0x00004000부터 폴트), DTCM 64 KB, SRAM1 240 KB, SRAM2 16 KB, BKPSRAM 4 KB, Flash 1 MB(AXIM/ITCM 두 별칭 모두), 총 내부 SRAM 320 KB → **전부 폴트 경계 실측으로 확인**. 0x00200000과 0x08000000의 선두 64바이트가 바이트 단위로 동일 = 같은 물리 플래시의 두 경로임도 확인.
- **BKPSRAM 4 KB @0x4002_4000 / AHB1ENR bit 18**, **DTCM 클럭게이트 AHB1ENR bit 20**: 실측 `AHB1ENR=0x00D007FC`에 DTCMRAM 비트가 셋되어 있음으로 확인.
- **LTDC 480×272**: 실측 픽셀클럭 **9.60 MHz**, 전체 타이밍 566×286, **59.3 Hz**, ARGB8888, 프레임버퍼 `0xC0000000`/`0xC007F800` 더블버퍼. 문서의 "LTDC 클럭 출력 최대 45 MHz" 스펙 대비 크게 여유.
- **ID 레지스터 주소**: UID `0x1FF0F420`(= `0x00340033_54435008_20353036`), Flash size `0x1FF0F442`(= `0x0400` → 1024 KB) 확인.
- **IRQ/헤더 사실**: `DMA2D_IRQn = 90`, 페리페럴 IRQ 97개 / 벡터 105개, CMSIS 헤더에 `CRYP_BASE`/`HASH_BASE`/`JPEG_BASE` 정의 없음 → 호스트 측 헤더 파싱으로 확인.
- **RTC 게이트가 `RCC_BDCR` bit 15**: 실측 `BDCR=0x00000003`(LSEON/LSERDY만 1, **RTCEN=0**)로 확인. LSE는 **32,765 Hz 실측 → 32.768 kHz 크리스탈 실장 확정**.
- **§2.2 HAL 타임베이스 = TIM6**: 실행 중 `APB1ENR=0x10000010` → 켜진 APB1 페리페럴이 **TIM6와 PWR뿐**임으로 뒷받침됨.
- **보드 실장 부품(문서가 다루지 않은 부분의 보강)**: WM8994(I2C3 0x34, ID 0x8994), FT5336(0x70, ID 0x51), LAN8742A(MDIO 주소 0, ID 0x0007/0xC131), microSD 삽입(PC13=0), I2C1(Arduino 커넥터)은 빈 버스.

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

1. **QSPI NOR 부품 (요약 §17, §8 메모리맵, §10)** — **[중요]**
   - 문서: "QSPI NOR 16 Mbyte (**N25Q128A**)" (ST BSP `stm32746g_discovery_qspi.h` 기준)
   - 실측: 메모리맵 모드 진입 **전에** Read JEDEC ID(0x9F)를 직접 발행 → **`0xEF4018`**
     = 제조사 `0xEF`(**Winbond**), 타입 `0x40`(W25Q 계열), 용량 `0x18`(16 MB).
     Micron N25Q128A라면 `0x20BA18`이어야 한다.
   - 원인 추정: **보드 리비전에 따른 부품 교체**. ST BSP가 N25Q128A를 전제로 짜였는데도 동작하는 것은
     두 부품이 표준 SPI NOR 명령(0x9F/0xEB 등)을 공유하기 때문이다.
   - 실무 영향: 용량·읽기 명령은 그대로 써도 되지만 **섹터 소거 시간·페이지 크기·상태 레지스터 비트는
     Winbond 데이터시트를 기준으로 검증**해야 한다.
2. **QUADSPI 최대 클럭 80 MHz (§1.3, §6, §10)**
   - 문서: "최대 **80 MHz**" (DS10916 Quad-SPI 특성표)
   - 실측: `QUADSPI_CR=0x01400311` → PRESCALER=1 → **QSPI 클럭 = HCLK/2 = 100 MHz로 실제 동작 중**
     (ST BSP `BSP_QSPI_Init()`이 설정한 값이며, 메모리맵 읽기가 정상 동작한다).
   - 원인 추정: ST BSP가 216 MHz가 아닌 클럭 트리에서 프리스케일러를 1로 두어 **데이터시트 최대치를 25% 초과**한 상태.
     기능은 하지만 스펙 밖 동작이므로, 온도·개체 편차에서 안정성을 보장할 수 없다. (SSHIFT=1이 켜져 있어 마진을 벌고 있다.)
3. **IWDG의 LSI를 "32 kHz"로 기술 (§1.5 표, §7 표)**
   - 문서: "독립 **32 kHz** LSI 구동"
   - 실측: **LSI = 26,616 ~ 26,652 Hz** (TIM5 입력캡처, 100 MHz 타이머 기준). 공칭 대비 **-16.8 %**
   - 원인 추정: 문서가 공칭값을 적었고 실측은 이 개체의 RC 발진기 실제 주파수다(문서 오류가 아니라 정밀도 문제).
   - 실무 영향: **IWDG 타임아웃을 32 kHz로 계산하면 실제로는 약 1.2배 길어진다.** 워치독 설계 시 반드시 보정할 것.
4. **§11 "실제 인에이블되는 페리페럴" 목록 및 관찰 1·2·3**
   - 문서: 소스 `__HAL_RCC_*_CLK_ENABLE()` grep 기준으로 GPIOA~K, CRC, DMA1, DMA2, DMA2D, ETH, DCMI, FMC, QSPI,
     TIM1/2/3/5/6/8/12, I2C1/3, SPDIFRX, SPI2, ADC3, LTDC, SAI2, SDMMC1, USART1/6, PWR, SYSCFG가 켜진다고 기술.
   - 실측(SNN 펌웨어 실행 중 ST-LINK 덤프): 클럭이 실제로 공급되던 페리페럴은 **18개뿐**이다 —
     `AHB1ENR=0x00D007FC`(GPIOC~K, DTCMRAM, DMA2, DMA2D), `AHB2ENR=0x00000000`(**DCMI OFF**),
     `AHB3ENR=0x00000001`(**FMC만, QSPI OFF**), `APB1ENR=0x10000010`(**TIM6, PWR만**),
     `APB2ENR=0x04004800`(SDMMC1, SYSCFG, LTDC).
   - 즉 **ETH·DCMI·SAI2·SPDIFRX·SPI2·I2C1/3·ADC3·USART1/6·TIM1/2/3/5/8/12·CRC·DMA1·QSPI의 클럭은 꺼져 있었다.**
   - 원인 추정: 소스에 호출이 있어도 실행 경로상 도달하지 않거나(조건부 초기화), BSP가 `DeInit`으로 되돌리거나,
     초기화 후 다시 꺼지기 때문. **"소스에 있다"와 "런타임에 켜져 있다"는 다르다.**
   - 파급: 관찰 3 "**CRC 클럭이 이미 켜져 있다**"는 **틀렸다**(CRCEN=0). 관찰 2 "TIM2/TIM5 모두 클럭이 켜져 있다"도 **틀렸다**.
     관찰 1의 "합계 약 40 mA 절감 여지"도 실제로 켜져 있던 것이 DMA2D/LTDC/SDMMC1뿐이므로 **과대 추정**이다.
5. **CRYP / HASH "F746 미탑재 [A]" (§1.2, §4.5, §10)** — 신뢰도 하향 필요
   - 문서: F746 CMSIS 헤더에 정의가 없으므로 **없다**고 [A]로 단정.
   - 실측: `0x50060000`(CRYP)·`0x50060400`(HASH) **베이스 읽기 OK(값 0), RCC 비트 KEEP** — JPEG처럼 BusFault가 나지 않는다.
   - 다만 **실측 쪽 해석도 조심해야 한다**: F746에 없는 `DFSDM1(0x40017400)`·`DSI(0x40016C00)`도 똑같이 OK로 응답했다.
     즉 **주소 디코딩만 살아 있고 기능 블록은 없을 수 있다**. 실측은 문서를 반증하지 못하며, 반대로 헤더 부재도
     실리콘 부재의 증거가 아니다. → **판정 보류**. (JPEG처럼 "확정"이라고 쓰면 안 된다.)
6. **Package 레지스터 주소 (§7 표)**
   - 문서: "Package 레지스터 **0x1FF0_F7E0** [A] CMSIS `PACKAGE_BASE`"
   - 실측: `0x1FF0F7E0 = 0x038DFC72`, `0x1FF07BF0(RM0385가 지정하는 주소) = 0xFFFFFFFF`.
     두 주소가 서로 다른 값을 돌려주며, **어느 쪽으로도 이 다이의 패키지를 판정할 수 없다**(미프로그램).
   - 원인 추정: CMSIS 헤더와 RM0385가 서로 다른 주소를 지정. 문서는 헤더 쪽만 인용해 [A]로 올렸다.
   - → 패키지 종류는 **레지스터가 아니라 마킹으로만** 판정 가능.
7. **OTP 크기 (§7 표)** — 경미
   - 문서: "OTP 메모리 **1024바이트** @0x1FF0_F000"
   - 실측(헤더 파싱): `FLASH_OTP_BASE=0x1FF0F000`, `FLASH_OTP_END=0x1FF0F41F` → **1056 바이트**(데이터 1024 B + 잠금 바이트 32 B).

### 아직 측정하지 않음

- **페리페럴별 소비전류 표(§9, µA/MHz)** — 보드 IDD 점퍼에 전류계를 물리고 각 `RCC_xxxENR` 비트만 토글해 차분 측정.
- **CRC 32비트당 4 HCLK(§7, §10)** — `CRC_DR`에 N회 연속 쓰기 구간을 DWT CYCCNT로 재고 N으로 나눈다.
- **RNG 연속 난수 간 40 RNG_CLK(§1.2, §10)** — `RNG_SR.DRDY` 상승 간격을 CYCCNT로 측정.
- **DMA2 전용 memory-to-memory(§3.1)** — DMA1 스트림에 `DIR=10`(m2m)을 설정해 실제로 전송이 안 되는지 확인.
- **DMA FIFO 4워드 / 4·8·16 beat 버스트(§3.1)** — 버스트 설정별로 SDRAM↔SRAM1 전송 시간을 비교.
- **ADC 2.4 MSPS · tCONV 0.50 µs(§5.1)** — ADC를 연속 변환 모드로 돌려 N회 EOC 사이클 수를 CYCCNT로 측정(온칩 VREFINT 채널로 가능).
- **SDMMC1 8비트 50 MHz(§1.4, §6)** — microSD가 실제로 삽입되어 있으므로 `BSP_SD_ReadBlocks()` 대량 읽기 시간으로 처리량 산출.
- **USART 27 Mbit/s(§6)** — OVER8 + 최소 BRR로 TX/RX를 물리적으로 루프백해 오류율 확인.
- **SPI 50 vs 25 Mbit/s 구분(§6)** — SPI1과 SPI2를 최소 프리스케일러로 돌리고 SCK를 TIM 입력캡처로 실측.
- **LTDC 최대 픽셀클럭 45 MHz(§4.2)** — PLLSAI 분주를 낮춰가며 `LTDC_ISR.FUIF`(FIFO 언더런) 발생 지점을 찾는다.
- **QUADSPI DDR/dual-flash 모드(§1.3)** — `QUADSPI_CCR.DDRM`/`CR.DFM`을 바꿔 메모리맵 읽기 무결성 확인.
- **SDRAM 실효 대역폭 vs 이론값** — 실측 순차 읽기 **60.25 MB/s**(이론 200 MB/s의 30%). MPU로 SDRAM을 Normal cacheable로 바꿔 캐시 라인필 버스트가 걸릴 때의 대역폭을 다시 재야 원인(Device 메모리 · 단일 접근 · FMC 에라타 §2.3.1)을 분리할 수 있다.

## 요약

- 페리페럴 버스 베이스: **APB1 = 0x4000_0000, APB2 = 0x4001_0000, AHB1 = 0x4002_0000, AHB2 = 0x5000_0000, AHB3(FMC/QUADSPI 레지스터) = 0xA000_0000** [A] (RM0385 Rev 5, Table 1, §2.2.2, p.65~69)
- 타이머는 총 **18개**(고급제어 2 + 범용 10 + 기본 2 + LPTIM1 1 + IWDG + WWDG + SysTick). 이 중 **32비트는 TIM2, TIM5 단 2개**이며 나머지 타이머 카운터는 모두 16비트다 [A] (DS10916 Rev 4, Table 6, p.33 / Table 2, p.13)
- APB2 타이머(TIM1/TIM8/TIM9/TIM10/TIM11)는 최대 **216 MHz**, APB1 타이머(TIM2~TIM7/TIM12~TIM14)는 `RCC_DCKCFGR1`의 **TIMPRE 비트**에 따라 **108 MHz 또는 216 MHz** [A] (DS10916 Table 6 note 1; RM0385 §5.2 p.157)
- DMA1/DMA2는 각각 **8 스트림 × 스트림당 최대 8 채널**, 스트림당 **4워드(=16바이트) 깊이 FIFO**, **4/8/16 beat 인크리멘탈 버스트**를 지원. 메모리→메모리 전송은 **DMA2에서만** 가능 [A] (RM0385 §8.2, p.220~221)
- **DMA2D(Chrom-ART), LTDC, DCMI, CRC, RNG는 존재**하지만 **JPEG 코덱과 AES/HASH 암호 가속기는 STM32F746에 없다** — F746 CMSIS 헤더에 `CRYP_BASE`/`HASH_BASE`/`JPEG_BASE` 정의가 아예 없고, 남은 `HASH_RNG_IRQn` 은 `RNG_IRQn` 로의 레거시 별칭일 뿐이다 [A][D]
- ADC는 **12비트 3개(ADC1/2/3)**, F746Nx 패키지에서 **24채널**, 최대 **2.4 MSPS**(삼중 인터리브 시 7.2 MSPS), fADC 최대 **36 MHz**, 12비트 최소 변환시간 **0.50 µs**(fADC=30 MHz) [A] (DS10916 Table 2 p.14, Table 62 p.145)
- 백업 SRAM은 **4 Kbyte @ 0x4002_4000**, 백업 레지스터는 **32 × 32비트 = 128바이트** [A] (DS10916 §2.19 p.28, Table 2 p.13)
- CRC 유닛은 **다항식 프로그래머블(7/8/16/32비트)**, 리셋값 **0x04C11DB7**, 32비트 데이터 1개당 **4 HCLK 사이클**에 계산 완료 → SNN 가중치/입력 버퍼 체크섬에 CPU 부담 거의 없이 사용 가능 [A] (RM0385 §12.2 p.297, §12.4.5)
- **본 프로젝트 실제 클럭은 216 MHz가 아니라 200 MHz**다: HSE 25 MHz, PLLM=25/PLLN=400/PLLP=2 → SYSCLK 200 MHz, HCLK 200 MHz, PCLK1 50 MHz, PCLK2 100 MHz, Flash 6 wait states [A] (프로젝트 `Core/Src/main.cc` `SystemClock_Config`)
- 보드 실장 메모리: **SDRAM 8 Mbyte @ 0xC000_0000 (16비트 버스)**, **QSPI NOR 16 Mbyte (N25Q128A)**, LCD **480×272** [A] (ST BSP `stm32746g_discovery_sdram.h` / `_qspi.h`, `rk043fn48h.h`)

> **실측 정정**: SDRAM 8 MB / LCD 480×272는 실측과 일치한다(`FMC_SDCR1=0x00001954`, LTDC 실측 480×272@59.3 Hz).
> 그러나 **QSPI NOR은 N25Q128A가 아니다.** Read JEDEC ID(0x9F) 실측 응답이 **`0xEF4018` = Winbond W25Q128 계열(16 MB)**
> 이며, Micron N25Q128A라면 `0x20BA18`이어야 한다. ST BSP가 N25Q128A를 전제로 짜였는데도 동작하는 것은
> 두 부품이 표준 SPI NOR 명령을 공유하기 때문이다. 용량(16 MB)만 문서와 같다.

---

## 1. 전체 페리페럴 인벤토리 표

베이스 주소는 RM0385 Rev 5 Table 1 및 ST CMSIS `stm32f746xx.h`(두 출처 일치)에서, RCC 인에이블 비트는 RM0385 §5.3.10~5.3.13 및 동일 헤더에서 확인했다. 두 출처가 교차검증되므로 신뢰도 [A].

### 1.1 AHB1 (베이스 0x4002_0000)

| 페리페럴 | 개수 | 베이스 주소 | 버스 | RCC 인에이블 비트 | 주요 스펙 | 신뢰도 |
|---|---|---|---|---|---|---|
| GPIOA~GPIOK | 11 포트 | 0x4002_0000 + 0x400×n | AHB1 | `RCC_AHB1ENR` bit 0~10 | F746Nx 총 168 I/O, 최대 164개가 108 MHz fast I/O, 166개 5V-tolerant | [A] |
| CRC | 1 | 0x4002_3000 | AHB1 | `RCC_AHB1ENR` bit 12 | 다항식 7/8/16/32비트 프로그래머블, 리셋값 0x04C11DB7, 32비트당 4 HCLK | [A] |
| RCC | 1 | 0x4002_3800 | AHB1 | — (항상 활성) | — | [A] |
| Flash 인터페이스 레지스터 | 1 | 0x4002_3C00 | AHB1 | — | ART Accelerator + L1 캐시 제어 | [A] |
| BKPSRAM (백업 SRAM) | 1 | 0x4002_4000 | AHB1 | `RCC_AHB1ENR` bit 18 | **4 Kbyte**, VBAT 도메인 유지 | [A] |
| DTCM-RAM 클럭 게이트 | — | (0x2000_0000) | AHB1 | `RCC_AHB1ENR` bit 20 | DTCM 64 KB | [A] |
| DMA1 | 1 | 0x4002_6000 | AHB1 | `RCC_AHB1ENR` bit 21 | 8 스트림 × 8 채널, 스트림당 4워드 FIFO | [A] |
| DMA2 | 1 | 0x4002_6400 | AHB1 | `RCC_AHB1ENR` bit 22 | DMA1과 동일 + **memory-to-memory 지원** | [A] |
| DMA2D (Chrom-ART) | 1 | 0x4002_B000 | AHB1 | `RCC_AHB1ENR` bit 23 | 4가지 동작모드, 11개 컬러포맷(4~32bpp), 2소스 블렌딩 | [A] |
| Ethernet MAC | 1 | 0x4002_8000 | AHB1 | `RCC_AHB1ENR` bit 25(MAC), 26(TX), 27(RX), 28(PTP) | 10/100 Mbit/s, MII/RMII, 전용 DMA, TX/RX FIFO 각 2 Kbyte, IEEE 1588 PTP v2 | [A] |
| USB OTG HS | 1 | 0x4004_0000 | AHB1 | `RCC_AHB1ENR` bit 29(OTGHS), 30(ULPI) | FS 내장 PHY / HS는 외부 ULPI PHY 필요(480 Mbit/s), FIFO 4 Kbyte, 8 양방향 EP, 16 host 채널 | [A] |

- ETH 서브블록 오프셋: MAC 0x4002_8000, MMC +0x0100, PTP +0x0700, DMA +0x1000 [A] (CMSIS 헤더)
- DMA 스트림 레지스터 오프셋: Stream0 = base+0x010, 이후 스트림당 +0x018 (Stream7 = base+0x0B8) [A] (CMSIS 헤더)

### 1.2 AHB2 (베이스 0x5000_0000)

| 페리페럴 | 개수 | 베이스 주소 | 버스 | RCC 인에이블 비트 | 주요 스펙 | 신뢰도 |
|---|---|---|---|---|---|---|
| USB OTG FS | 1 | 0x5000_0000 | AHB2 | `RCC_AHB2ENR` bit 7 | 내장 FS PHY, FIFO 합계 1.28 Kbyte, 1 control + 5 IN + 5 OUT EP, 12 host 채널 | [A] |
| DCMI (카메라) | 1 | 0x5005_0000 | AHB2 | `RCC_AHB2ENR` bit 0 | 8/10/12/14비트 병렬, 최대 **54 Mbyte/s @ 54 MHz**, DCMI_PIXCLK ≤ 54 MHz 및 PIXCLK/fHCLK ≤ 0.4 | [A] |
| RNG | 1 | 0x5006_0800 | AHB2 | `RCC_AHB2ENR` bit 6 | 아날로그 32비트 난수, 연속 난수 간 **RNG_CLK 40주기** 소요 | [A] |
| ~~CRYP~~ | **0 (F746 미탑재)** | (0x5006_0000, F756 전용) | AHB2 | — | STM32F746에는 없음 | [A] |
| ~~HASH~~ | **0 (F746 미탑재)** | (0x5006_0400, F756 전용) | AHB2 | — | STM32F746에는 없음 | [A] |

> **실측 정정 (CRYP/HASH의 "0개" 단정)**: `0x50060000`(CRYP)과 `0x50060400`(HASH)의 베이스 주소를 실제로 읽으면
> **BusFault가 나지 않고 값 0으로 정상 응답**하며, RCC 인에이블 비트도 1로 유지(KEEP)된다. JPEG(`0x50051000`)가
> **RCC 비트 CLEAR0 + BusFault**로 명확히 부재를 드러낸 것과 대조된다.
> 다만 **실측 쪽 해석도 과신하면 안 된다** — F746에 존재하지 않는 `DFSDM1(0x40017400)`·`DSI(0x40016C00)`도 똑같이 OK로
> 응답했다. 즉 주소 디코딩만 살아 있고 기능 블록은 없을 수 있다. 결론: **CRYP/HASH는 "확정적 부재"가 아니라 판정 보류**이며,
> 문서의 신뢰도 [A]는 [B]로 낮춰 읽어야 한다. 반대로 JPEG 부재는 두 방법이 일치해 **확정**이다.

> 주의: RM0385는 STM32F75xxx/F74xxx **공통** 매뉴얼이라 Table 1에 CRYP(0x5006_0000)/HASH(0x5006_0400)가 등재되어 있으나, 이는 **STM32F756 전용**이다. F746용 CMSIS 헤더 `stm32f746xx.h`에는 `CRYP_BASE`/`HASH_BASE` 및 관련 레지스터 정의가 전혀 없다 [A][D]. 같은 이유로 RM0385 Table 25(DMA2 요청 매핑)의 `CRYP_OUT`/`CRYP_IN`/`HASH_IN` 항목(Channel 2, Stream 5/6/7)은 F746에서는 **사용되지 않는 빈 슬롯**이다 [D].

### 1.3 AHB3

| 페리페럴 | 개수 | 베이스 주소 | 버스 | RCC 인에이블 비트 | 주요 스펙 | 신뢰도 |
|---|---|---|---|---|---|---|
| FMC 제어 레지스터 | 1 | 0xA000_0000 | AHB3 | `RCC_AHB3ENR` bit 0 | 최대 32비트 데이터 버스, SRAM/PSRAM/SDRAM/NOR/NAND | [A] |
| QUADSPI 제어 레지스터 | 1 | 0xA000_1000 | AHB3 | `RCC_AHB3ENR` bit 1 | 최대 **80 MHz**, SDR/DDR, dual-flash, indirect/status-polling/**memory-mapped** 3모드 | [A] |
| QUADSPI 메모리 맵 영역 | — | 0x9000_0000 | AHB3 | (동일) | 메모리 맵 모드에서 코드/데이터 직접 실행·읽기 | [A] |

> **실측 정정 (QUADSPI 최대 80 MHz)**: 데이터시트 최대치는 80 MHz가 맞지만, **이 보드는 실제로 100 MHz로 돌고 있다.**
> `QUADSPI_CR = 0x01400311` → PRESCALER=1 → QSPI 클럭 = HCLK/2 = **100 MHz** (SSHIFT=1로 홀드 마진 확보).
> ST BSP `BSP_QSPI_Init()`이 설정한 값이며 메모리맵 읽기는 정상 동작한다. 즉 **스펙을 25% 초과한 상태로 동작 중**이므로,
> 온도·개체 편차에 대한 안정성은 보장되지 않는다. 설정 전체: `DCR=0x00170500`(FSIZE=23 → 16 MB, CSHT=5),
> `CCR=0x0F282DEB`(명령 0xEB Quad I/O Fast Read, 3바이트 주소, 더미 10사이클, FMODE=메모리맵).
> 메모리맵 순차 읽기 실측 대역폭은 **약 50 MB/s**, 64 KB 스트라이드 랜덤 접근은 **94~198 cyc/접근**이다.

### 1.4 APB2 (베이스 0x4001_0000, PCLK2 최대 108 MHz)

| 페리페럴 | 개수 | 베이스 주소 | 버스 | RCC 인에이블 비트 | 주요 스펙 | 신뢰도 |
|---|---|---|---|---|---|---|
| TIM1 | 1 | 0x4001_0000 | APB2 | `RCC_APB2ENR` bit 0 | 고급제어, 16비트, 4채널 + 상보출력, 최대 216 MHz | [A] |
| TIM8 | 1 | 0x4001_0400 | APB2 | `RCC_APB2ENR` bit 1 | TIM1과 동일 | [A] |
| USART1 | 1 | 0x4001_1000 | APB2 | `RCC_APB2ENR` bit 4 | 최대 27 Mbit/s (클럭=SYSCLK 216 MHz, OVER8) | [A] |
| USART6 | 1 | 0x4001_1400 | APB2 | `RCC_APB2ENR` bit 5 | 동일 | [A] |
| ADC1 | 1 | 0x4001_2000 | APB2 | `RCC_APB2ENR` bit 8 | 12/10/8/6비트, 2.4 MSPS | [A] |
| ADC2 | 1 | 0x4001_2100 | APB2 | `RCC_APB2ENR` bit 9 | 동일 | [A] |
| ADC3 | 1 | 0x4001_2200 | APB2 | `RCC_APB2ENR` bit 10 | 동일 | [A] |
| ADC 공통 레지스터 | 1 | 0x4001_2300 | APB2 | — | Dual/Triple 모드, 인터리브 지연 설정 | [A] |
| SDMMC1 | 1 | 0x4001_2C00 | APB2 | `RCC_APB2ENR` bit 11 | MMC 4.2 / SD 2.0 / SDIO 2.0, 1/4/8비트 버스, 8비트 모드 **최대 50 MHz**, SPI 모드 미지원 | [A] |
| SPI1 / I2S1 | 1 | 0x4001_3000 | APB2 | `RCC_APB2ENR` bit 12 | 최대 **50 Mbit/s**, 프레임 4~16비트, HW CRC | [A] |
| SPI4 | 1 | 0x4001_3400 | APB2 | `RCC_APB2ENR` bit 13 | 최대 50 Mbit/s | [A] |
| SYSCFG | 1 | 0x4001_3800 | APB2 | `RCC_APB2ENR` bit 14 | EXTI 소스 선택, 메모리 리맵 | [A] |
| EXTI | 1 | 0x4001_3C00 | APB2 | — | — | [A] |
| TIM9 | 1 | 0x4001_4000 | APB2 | `RCC_APB2ENR` bit 16 | 16비트, 업카운터, 2채널, DMA 요청 **없음**, 216 MHz | [A] |
| TIM10 | 1 | 0x4001_4400 | APB2 | `RCC_APB2ENR` bit 17 | 16비트, 1채널, DMA 없음, 216 MHz | [A] |
| TIM11 | 1 | 0x4001_4800 | APB2 | `RCC_APB2ENR` bit 18 | TIM10과 동일 | [A] |
| SPI5 | 1 | 0x4001_5000 | APB2 | `RCC_APB2ENR` bit 20 | 최대 50 Mbit/s | [A] |
| SPI6 | 1 | 0x4001_5400 | APB2 | `RCC_APB2ENR` bit 21 | 최대 50 Mbit/s | [A] |
| SAI1 | 1 (Block A/B) | 0x4001_5800 (A=+0x004, B=+0x024) | APB2 | `RCC_APB2ENR` bit 22 | I2S/TDM/AC'97/PCM/SPDIF-out, 8~192 kHz | [A] |
| SAI2 | 1 (Block A/B) | 0x4001_5C00 (A=+0x004, B=+0x024) | APB2 | `RCC_APB2ENR` bit 23 | 동일 | [A] |
| LTDC | 1 | 0x4001_6800 | APB2 | `RCC_APB2ENR` bit 26 | 2 레이어(각 64×32비트 FIFO), 최대 XGA 1024×768, 8개 입력 컬러포맷, CLUT 256색, AHB 16워드 버스트, **LTDC 클럭 출력 최대 45 MHz** | [A] |

### 1.5 APB1 (베이스 0x4000_0000, PCLK1 최대 54 MHz)

| 페리페럴 | 개수 | 베이스 주소 | 버스 | RCC 인에이블 비트 | 주요 스펙 | 신뢰도 |
|---|---|---|---|---|---|---|
| TIM2 | 1 | 0x4000_0000 | APB1 | `RCC_APB1ENR` bit 0 | **32비트** 카운터, 4채널, up/down/center | [A] |
| TIM3 | 1 | 0x4000_0400 | APB1 | `RCC_APB1ENR` bit 1 | 16비트, 4채널 | [A] |
| TIM4 | 1 | 0x4000_0800 | APB1 | `RCC_APB1ENR` bit 2 | 16비트, 4채널 | [A] |
| TIM5 | 1 | 0x4000_0C00 | APB1 | `RCC_APB1ENR` bit 3 | **32비트** 카운터, 4채널 | [A] |
| TIM6 | 1 | 0x4000_1000 | APB1 | `RCC_APB1ENR` bit 4 | 기본 타이머, 16비트, 채널 0, DAC 트리거용 | [A] |
| TIM7 | 1 | 0x4000_1400 | APB1 | `RCC_APB1ENR` bit 5 | TIM6와 동일 | [A] |
| TIM12 | 1 | 0x4000_1800 | APB1 | `RCC_APB1ENR` bit 6 | 16비트, 2채널, DMA 없음 | [A] |
| TIM13 | 1 | 0x4000_1C00 | APB1 | `RCC_APB1ENR` bit 7 | 16비트, 1채널, DMA 없음 | [A] |
| TIM14 | 1 | 0x4000_2000 | APB1 | `RCC_APB1ENR` bit 8 | 16비트, 1채널, DMA 없음 | [A] |
| LPTIM1 | 1 | 0x4000_2400 | APB1 | `RCC_APB1ENR` bit 9 | 16비트 업카운터, Stop 모드 동작, LSE/LSI/HSI/APB/외부 클럭 | [A] |
| RTC & 백업 레지스터 | 1 | 0x4000_2800 | APB1 | **`RCC_BDCR` bit 15 (RTCEN)** | BCD 캘린더, 알람 2개, 17비트 wakeup 타이머, 0.95 ppm 캘리브레이션, **32 × 32비트 백업 레지스터(128바이트)** | [A] |
| WWDG | 1 | 0x4000_2C00 | APB1 | `RCC_APB1ENR` bit 11 | 7비트 다운카운터, 윈도우 감시 | [A] |
| IWDG | 1 | 0x4000_3000 | APB1 | — (**RCC 게이트 없음**, LSI 구동, `IWDG_KR`로 기동) | 12비트 다운카운터 + 8비트 프리스케일러, 독립 32 kHz LSI | [A] |
| SPI2 / I2S2 | 1 | 0x4000_3800 | APB1 | `RCC_APB1ENR` bit 14 | **최대 25 Mbit/s** (SPI1/4/5/6 대비 절반) | [A] |
| SPI3 / I2S3 | 1 | 0x4000_3C00 | APB1 | `RCC_APB1ENR` bit 15 | 최대 25 Mbit/s | [A] |
| SPDIFRX | 1 (**입력 4개**) | 0x4000_4000 | APB1 | `RCC_APB1ENR` bit 16 | IEC-60958/61937, 최대 심볼레이트 12.288 MHz, 32~192 kHz | [A] |
| USART2 | 1 | 0x4000_4400 | APB1 | `RCC_APB1ENR` bit 17 | 최대 27 Mbit/s | [A] |
| USART3 | 1 | 0x4000_4800 | APB1 | `RCC_APB1ENR` bit 18 | 동일 | [A] |
| UART4 | 1 | 0x4000_4C00 | APB1 | `RCC_APB1ENR` bit 19 | 동일 | [A] |
| UART5 | 1 | 0x4000_5000 | APB1 | `RCC_APB1ENR` bit 20 | 동일 | [A] |
| I2C1 | 1 | 0x4000_5400 | APB1 | `RCC_APB1ENR` bit 21 | Sm 100 kbit/s, Fm 400 kbit/s, SMBus 2.0 / PMBus 1.1, 아날로그+디지털 필터 | [A] |
| I2C2 | 1 | 0x4000_5800 | APB1 | `RCC_APB1ENR` bit 22 | 동일 | [A] |
| I2C3 | 1 | 0x4000_5C00 | APB1 | `RCC_APB1ENR` bit 23 | 동일 | [A] |
| I2C4 | 1 | 0x4000_6000 | APB1 | `RCC_APB1ENR` bit 24 | 동일 | [A] |
| CAN1 (bxCAN) | 1 | 0x4000_6400 | APB1 | `RCC_APB1ENR` bit 25 | 2.0A/2.0B active, 최대 1 Mbit/s, TX 메일박스 3, RX FIFO 2×3단, **28 필터뱅크 공유**, CAN당 SRAM 256바이트 | [A] |
| CAN2 | 1 | 0x4000_6800 | APB1 | `RCC_APB1ENR` bit 26 | 동일 | [A] |
| HDMI-CEC | 1 | 0x4000_6C00 | APB1 | `RCC_APB1ENR` bit 27 | — | [A] |
| PWR | 1 | 0x4000_7000 | APB1 | `RCC_APB1ENR` bit 28 | 오버드라이브, 전압 스케일 1/2/3 | [A] |
| DAC | 1 (**채널 2개**) | 0x4000_7400 | APB1 | `RCC_APB1ENR` bit 29 | 12비트 버퍼드 ×2, 8/12비트 출력, 노이즈/삼각파 생성, 채널별 DMA, **DAC 트리거 입력 8개** | [A] |
| UART7 | 1 | 0x4000_7800 | APB1 | `RCC_APB1ENR` bit 30 | 최대 27 Mbit/s | [A] |
| UART8 | 1 | 0x4000_7C00 | APB1 | `RCC_APB1ENR` bit 31 | 동일 | [A] |

> **실측 정정 (IWDG 행)**: 위 표의 "독립 **32 kHz** LSI"는 공칭값이다. 이 개체의 LSI를 TIM5 입력캡처로 직접 재면
> **26,616 ~ 26,652 Hz (공칭 대비 -16.8 %)** 다. LSI는 RC 발진기라 개체·온도 의존이 크다.
> **IWDG 타임아웃을 32 kHz로 계산하면 실제 타임아웃은 약 1.2배 길어진다.** §7의 IWDG 항목에도 같은 정정이 적용된다.
>
> **실측 확인 (I2C 행)**: I2C3에서 WM8994(0x34)·FT5336(0x70)이 ACK를 반환해 버스가 동작함을 확인했다.
> I2C1(Arduino 확장 커넥터)은 응답 장치가 하나도 없는 빈 버스다.

---

## 2. 타이머 상세 (TIM1~TIM14 + LPTIM1)

DS10916 Rev 4 Table 6 (p.33) 전문 기준 [A].

| 분류 | 타이머 | 카운터 폭 | 카운터 방향 | 프리스케일러 | DMA 요청 | 캡처/비교 채널 | 상보출력 | 최대 인터페이스 클럭 | 최대 타이머 클럭 |
|---|---|---|---|---|---|---|---|---|---|
| 고급제어 | TIM1, TIM8 | **16비트** | Up / Down / Up-down | 1~65536 정수 | Yes | **4** | **Yes** | 108 MHz | **216 MHz** |
| 범용 | **TIM2, TIM5** | **32비트** | Up / Down / Up-down | 1~65536 정수 | Yes | 4 | No | 54 MHz | 108 / 216 MHz |
| 범용 | TIM3, TIM4 | 16비트 | Up / Down / Up-down | 1~65536 정수 | Yes | 4 | No | 54 MHz | 108 / 216 MHz |
| 범용 | TIM9 | 16비트 | Up | 1~65536 정수 | **No** | 2 | No | 108 MHz | 216 MHz |
| 범용 | TIM10, TIM11 | 16비트 | Up | 1~65536 정수 | **No** | 1 | No | 108 MHz | 216 MHz |
| 범용 | TIM12 | 16비트 | Up | 1~65536 정수 | **No** | 2 | No | 54 MHz | 108 / 216 MHz |
| 범용 | TIM13, TIM14 | 16비트 | Up | 1~65536 정수 | **No** | 1 | No | 54 MHz | 108 / 216 MHz |
| 기본 | TIM6, TIM7 | 16비트 | Up | 1~65536 정수 | Yes | **0** | No | 54 MHz | 108 / 216 MHz |
| 저전력 | LPTIM1 | 16비트 | Up | — | — | 1 (compare) | No | — | LSE/LSI/HSI/APB/외부 |

### 2.1 타이머 클럭 결정 규칙 (TIMPRE)

`RCC_DCKCFGR1`의 **TIMPRE** 비트에 따라 결정된다 [A] (RM0385 Rev 5 §5.2, p.157):

- **TIMPRE = 0**: APB 프리스케일러가 1이면 `TIMxCLK = PCLKx`, 그 외에는 **`TIMxCLK = 2 × PCLKx`**
- **TIMPRE = 1**: APB 프리스케일러가 1, 2, 4이면 `TIMxCLK = HCLK`, 그 외에는 `TIMxCLK = 4 × PCLKx`

본 프로젝트 설정(HCLK 200 MHz, APB1 ÷4 → PCLK1 50 MHz, APB2 ÷2 → PCLK2 100 MHz, TIMPRE 미설정) 기준 파생값 [D]:

| 도메인 | PCLK | TIMPRE=0 일 때 TIMxCLK | 1 tick 분해능 |
|---|---|---|---|
| APB1 (TIM2~7, TIM12~14) | 50 MHz | **100 MHz** | 10 ns |
| APB2 (TIM1, TIM8~11) | 100 MHz | **200 MHz** | 5 ns |

> **실측 확인**: "TIMPRE 미설정"이라는 전제가 레지스터로 확인됐다. `RCC_DCKCFGR1`은 프로브 펌웨어에서 `0x00000000`,
> 원래 SNN 펌웨어에서 `0x00010000`(PLLSAIDIVR=/4)로 **양쪽 다 TIMPRE=0**이다. `RCC_CFGR=0x0000940A`로
> PPRE1=/4, PPRE2=/2도 확인됐으므로 위 표의 **APB1 타이머 100 MHz / APB2 타이머 200 MHz는 실측 확정값**이다.

→ **TIM2 또는 TIM5(32비트)를 100 MHz로 프리런시키면 10 ns 분해능으로 42.9초까지 랩어라운드 없이 측정**할 수 있다 (2^32 / 100 MHz = 42.95 s) [D]. 현재 4494 ms / 9641 ms / 2407 ms 급 추론 시간 측정에 16비트 타이머(최대 655 µs @100 MHz)는 부적합하고 32비트 타이머가 적합하다.

### 2.2 현재 프로젝트에서의 타이머 사용

- HAL 타임베이스가 **SysTick이 아니라 TIM6**로 설정되어 있다: `Core/Src/stm32f7xx_hal_timebase_tim.c`에서 TIM6를 프리스케일러로 **1 MHz 카운터**, Period 999 → **1 kHz 인터럽트**로 구동한다 [A]
- 따라서 `HAL_GetTick()` 분해능은 1 ms이며, TIM6는 이미 점유되어 있다 [D]

---

## 3. DMA1 / DMA2

### 3.1 공통 스펙 [A] (RM0385 Rev 5 §8.2, p.220~221)

| 항목 | 값 |
|---|---|
| 컨트롤러 수 | 2 (DMA1, DMA2) |
| 컨트롤러당 스트림 | **8** |
| 스트림당 채널(요청) | **최대 8** |
| FIFO 깊이 | **스트림당 4워드 (32비트 × 4 = 16바이트)** |
| FIFO 임계값 | 1/4, 1/2, 3/4, full (Direct 모드에서는 FIFO 비활성) |
| 버스트 | **인크리멘탈 4 / 8 / 16 beat** (`MBURST`/`PBURST`) |
| 마스터 포트 | 듀얼 AHB 마스터 (메모리 전용 + 페리페럴 전용) |
| 슬레이브 프로그래밍 I/F | **32비트 액세스만** 지원 |
| 전송 데이터 개수 | DMA flow controller 시 **1~65535** |
| 데이터 폭 | source/destination 독립 (byte / half-word / word), 불일치 시 자동 packing/unpacking (**FIFO 모드에서만**) |
| 우선순위 | SW 4단계(very high/high/medium/low) + 동률 시 낮은 스트림 번호 우선 |
| memory-to-memory | **DMA2 전용** (DMA1 불가) |
| 더블 버퍼 | 스트림별 지원 |
| 이벤트 플래그 | 5종 (HT, TC, TE, FIFO error, Direct mode error) → 스트림당 단일 IRQ로 OR |

- 스트림 레지스터 베이스: `DMAx_BASE + 0x010 + 0x018 × n` [A] (CMSIS 헤더)
- 순환(circular) + 버스트 모드 사용 시 제약: `DMA_SxNDTR`은 `(Mburst beat) × (Msize) / (Psize)`의 배수여야 한다 [A] (RM0385 §8.3.12 주석)

### 3.2 요청 매핑 표 위치

- **DMA1 요청 매핑: RM0385 Rev 5 Table 24, p.224 (§8.3.3)** [A]
- **DMA2 요청 매핑: RM0385 Rev 5 Table 25, p.224 (§8.3.3)** [A]

#### Table 24 — DMA1 request mapping [A]

| 채널 | Stream 0 | Stream 1 | Stream 2 | Stream 3 | Stream 4 | Stream 5 | Stream 6 | Stream 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | SPI3_RX | SPDIFRX_DT | SPI3_RX | SPI2_RX | SPI2_TX | SPI3_TX | SPDIFRX_CS | SPI3_TX |
| 1 | I2C1_RX | I2C3_RX | TIM7_UP | — | TIM7_UP | I2C1_RX | I2C1_TX | I2C1_TX |
| 2 | TIM4_CH1 | — | I2C4_RX | TIM4_CH2 | — | I2C4_TX | TIM4_UP | TIM4_CH3 |
| 3 | — | TIM2_UP / TIM2_CH3 | I2C3_RX | — | I2C3_TX | TIM2_CH1 | TIM2_CH2 / TIM2_CH4 | TIM2_UP / TIM2_CH4 |
| 4 | UART5_RX | USART3_RX | UART4_RX | USART3_TX | UART4_TX | USART2_RX | USART2_TX | UART5_TX |
| 5 | UART8_TX | UART7_TX | TIM3_CH4 / TIM3_UP | UART7_RX | TIM3_CH1 / TIM3_TRIG | TIM3_CH2 | UART8_RX | TIM3_CH3 |
| 6 | TIM5_CH3 / TIM5_UP | TIM5_CH4 / TIM5_TRIG | TIM5_CH1 | TIM5_CH4 / TIM5_TRIG | TIM5_CH2 | — | TIM5_UP | — |
| 7 | — | TIM6_UP | I2C2_RX | I2C2_RX | USART3_TX | DAC1 | DAC2 | I2C2_TX |

#### Table 25 — DMA2 request mapping [A]

| 채널 | Stream 0 | Stream 1 | Stream 2 | Stream 3 | Stream 4 | Stream 5 | Stream 6 | Stream 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | ADC1 | SAI1_A | TIM8_CH1/CH2/CH3 | SAI1_A | ADC1 | SAI1_B | TIM1_CH1/CH2/CH3 | SAI2_B |
| 1 | — | DCMI | ADC2 | ADC2 | SAI1_B | SPI6_TX | SPI6_RX | DCMI |
| 2 | ADC3 | ADC3 | — | SPI5_RX | SPI5_TX | *(CRYP_OUT — F746 미탑재)* | *(CRYP_IN — F746 미탑재)* | *(HASH_IN — F746 미탑재)* |
| 3 | SPI1_RX | — | SPI1_RX | SPI1_TX | SAI2_A | SPI1_TX | SAI2_B | QUADSPI |
| 4 | SPI4_RX | SPI4_TX | USART1_RX | SDMMC1 | — | USART1_RX | SDMMC1 | USART1_TX |
| 5 | — | USART6_RX | USART6_RX | SPI4_RX | SPI4_TX | — | USART6_TX | USART6_TX |
| 6 | TIM1_TRIG | TIM1_CH1 | TIM1_CH2 | TIM1_CH1 | TIM1_CH4 / TIM1_TRIG / TIM1_COM | TIM1_UP | TIM1_CH3 | — |
| 7 | — | TIM8_UP | TIM8_CH1 | TIM8_CH2 | TIM8_CH3 | SPI5_RX | SPI5_TX | TIM8_CH4 / TIM8_TRIG / TIM8_COM |

> DMA1은 AHB 버스 매트릭스 특성상 **AHB1/APB1 페리페럴 중심**이고, **ADC/SPI1/SPI4~6/USART1/USART6/SDMMC1/DCMI/QUADSPI/SAI는 DMA2에만** 매핑되어 있다 [D] (Table 24/25에서 유도).

---

## 4. 그래픽 / 이미지 관련 페리페럴

### 4.1 DMA2D (Chrom-ART Accelerator) [A] (RM0385 Rev 5 §9.2, p.254)

| 항목 | 값 |
|---|---|
| 베이스 / RCC | 0x4002_B000 / `RCC_AHB1ENR` bit 23 |
| IRQ | `DMA2D_IRQn` = 90 |
| 마스터 아키텍처 | **단일 AHB 마스터** |
| 슬레이브 액세스 폭 | 8/16/32비트 (CLUT은 32비트만) |
| 동작 모드 | **4가지**: register-to-memory, memory-to-memory, memory-to-memory + PFC, memory-to-memory + PFC + blending |
| 지원 컬러 포맷 | **최대 11종, 4bpp ~ 32bpp** (직접/간접 컬러) |
| 소스 개수 | 최대 2 (blending) |
| 알파 | source 값 / 고정값 / modulated |
| CLUT | 내부 메모리 2개, 자동 로딩 또는 CPU 프로그래밍, 크기 가변 |
| 대역폭 제어 | 내부 AHB dead-time 타이머 |
| 기타 | 워터마크 인터럽트, abort/suspend, 버스 에러 인터럽트 |
| 전류 | **87.1 µA/MHz** (Scale 1) — 전체 페리페럴 중 최대 |

### 4.2 LTDC [A] (RM0385 Rev 5 §18.2, p.507; DS10916 Table 105)

- 24비트 RGB 병렬 출력(RGB888), **레이어 2개**, 레이어당 **64×32비트 FIFO**
- 레이어당 CLUT **최대 256색 (256×24비트)**
- 최대 해상도 **XGA 1024×768**, 입력 컬러 포맷 8종 (ARGB8888 / RGB888 / RGB565 / ARGB1555 / ARGB4444 / L8 / AL44 / AL88)
- AHB 마스터 **16워드 버스트**, 디더링 R/G/B 각 2비트
- **LTDC 클럭 출력 최대 45 MHz** (DS10916 Rev 4, LTDC 특성표)
- 전류 **56.7 µA/MHz** (Scale 1)
- 보드 패널: RK043FN48H **480 × 272** [A] (ST BSP `rk043fn48h.h`)

### 4.3 DCMI [A] (RM0385 Rev 5 §17.2, p.500; DS10916 §2.38)

- 8/10/12/14비트 병렬 인터페이스, 임베디드/외부 라인·프레임 동기
- Continuous / snapshot 모드, **crop** 기능
- 데이터 포맷: 8/10/12/14비트 progressive(모노크롬 또는 raw Bayer), **YCbCr 4:2:2**, **RGB565**, 압축 JPEG 패스스루
- 최대 **54 Mbyte/s @ 54 MHz**, `DCMI_PIXCLK ≤ 54 MHz`, `DCMI_PIXCLK / fHCLK ≤ 0.4`
- 전류 5.1 µA/MHz (Scale 1)

> DCMI의 "Compressed data: JPEG"는 **JPEG 비트스트림을 그대로 통과시키는 기능**이지 하드웨어 JPEG 디코딩이 아니다. `DCMI_CR`의 `JPEG` 비트(bit 3)가 그 제어 비트다 [A] (CMSIS `DCMI_CR_JPEG_Pos = 3`).

### 4.4 JPEG 코덱 — **STM32F746에 없음** [A]

- STM32F746 CMSIS 헤더에 `JPEG_BASE` / `JPEG_TypeDef` 정의가 전혀 없다 (`JPEG` 문자열은 위의 `DCMI_CR_JPEG` 3줄이 전부) [A]
- 하드웨어 JPEG 코덱은 **STM32F767/F777/F769/F779 계열**부터 탑재된다 [B] (ST STM32F7 시리즈 제품 소개 자료)

### 4.5 암호/해시 (CRYP / HASH) — **STM32F746에 없음** [A]

- HW crypto/hash를 가진 STM32F7 파트는 **F730 / F732 / F733 / F750 / F756 / F777 / F779** 이며 **F746은 미포함** [B] (ST STM32F7 시리즈 소개 자료)
- F746 CMSIS 헤더에 `CRYP_BASE` / `HASH_BASE` / `CRYP_TypeDef` / `HASH_TypeDef` 정의 없음. 유일한 잔재는 레거시 IRQ 별칭 `#define HASH_RNG_IRQn RNG_IRQn` [A]

> **실측 정정**: 헤더에 정의가 없는 것은 사실이나(호스트 측 파싱으로 재확인), **실리콘에서 두 베이스 주소는 버스 응답을 반환한다**
> (CRYP `0x50060000`, HASH `0x50060400` → 둘 다 읽기 OK, 값 0). §1.2의 실측 정정과 같은 이유로 **판정 보류**로 두어야 한다.
> 반면 **§4.4의 JPEG 부재는 실측으로 확정**됐다: RCC 비트가 0으로 남고(CLEAR0) 베이스 읽기가 BusFault다.
> 61회 프로빙 중 실패한 유일한 항목이 JPEG였다.

---

## 5. 아날로그

### 5.1 ADC1 / ADC2 / ADC3 [A]

| 항목 | 값 | 출처 |
|---|---|---|
| 개수 | **3** (ADC1/2/3) | DS10916 Table 2 |
| 해상도 | **12 / 10 / 8 / 6비트 설정 가능** | RM0385 §15.2 |
| 채널 수 | **F746Nx(TFBGA216): 24채널** (LQFP100/V 패키지는 16채널) | DS10916 Table 2 p.14 |
| ADC당 외부 채널 | 최대 16 공유 | DS10916 §2.39 |
| 최대 샘플링 속도 | **2.4 MSPS** (VDD 2.4~3.6 V), **삼중 인터리브 7.2 MSPS** | DS10916 표지 features / Table 18 |
| fADC 최대 | **36 MHz** (VDDA 2.4~3.6 V, typ 30 MHz) / 18 MHz (VDDA 1.7~2.4 V) | DS10916 Table 62 p.145 |
| tCONV (12비트, fADC=30 MHz) | **최소 0.50 µs, 최대 16.40 µs** | DS10916 Table 62 |
| tCONV (10 / 8 / 6비트) | 최소 0.43 / 0.37 / 0.30 µs | DS10916 Table 62 |
| 샘플링 시간 tS | 3 ~ 480 × (1/fADC), fADC=30 MHz에서 0.100 ~ 16 µs | DS10916 Table 62 |
| 정규 트리거 지연 tlatr | 2 × (1/fADC) = 0.067 µs @30 MHz | DS10916 Table 62 |
| 외부 입력 임피던스 RAIN | 최대 50 kΩ | DS10916 Table 62 |
| 샘플/홀드 커패시터 CADC | typ 4 pF, max 7 pF | DS10916 Table 62 |
| 트리거 소스 | TIM1, TIM2, TIM3, TIM4, TIM5, TIM8 | DS10916 §2.39 |
| 부가 기능 | Dual/Triple 모드, 스캔, 불연속, 아날로그 워치독, DMA | RM0385 §15.2 |
| 전류 | 4.5 µA/MHz (Scale 1) + **ADON 시 아날로그부 1.73 mA/ADC 추가** | DS10916 Table 35 |
| 온도센서 | ADC1_IN18 (VBAT과 공유; 동시 활성 시 VBAT 우선) | DS10916 §2.40 |

### 5.2 DAC [A]

- **1개 DAC 블록, 12비트 채널 2개** (`DAC_BASE` = 0x4000_7400, `RCC_APB1ENR` bit 29)
- 8비트/12비트 monotonic 출력, 좌/우 정렬, 동기 업데이트, 노이즈파·삼각파 생성, 채널별 DMA
- **DAC 트리거 입력 8개** (타이머 update 이벤트 경유)
- 전류 3 µA/MHz + **EN1/EN2 설정 시 채널당 0.75 mA 추가** (DS10916 Table 35 note 4)

---

## 6. 통신 인터페이스 (총 25개) [A] (DS10916 표지 features, Table 2)

| 인터페이스 | 개수 (F746Nx) | 최대 속도 | 비고 |
|---|---|---|---|
| USART | **4** (USART1, 2, 3, 6) | **27 Mbit/s** (클럭=SYSCLK 216 MHz, OVER8) | ISO7816, LIN, IrDA, modem control, auto baud rate |
| UART | **4** (UART4, 5, 7, 8) | 27 Mbit/s | 동기 모드 없음 |
| SPI | **6** | SPI1/4/5/6 = **50 Mbit/s**, SPI2/3 = **25 Mbit/s** | 3비트 프리스케일러(8단), 프레임 4~16비트, NSS pulse, TI 모드, HW CRC, 전 채널 DMA |
| I2S | **3** (SPI1/2/3과 멀티플렉스, simplex) | 8 kHz ~ 192 kHz | 16/32비트, MCLK = 256 × fs |
| I2C | **4** | Sm 100 kbit/s, Fm 400 kbit/s | SMBus 2.0 / PMBus 1.1, 7/10비트 주소, 아날로그+디지털 노이즈 필터, 독립 클럭 소스 |
| CAN (bxCAN) | **2** | **1 Mbit/s** | 2.0A/2.0B active, TX 메일박스 3, RX FIFO 2개×3단, 필터뱅크 28개 공유, CAN당 SRAM 256바이트 |
| SAI | **2** (각 Block A/B) | 8~192 kHz | I2S, LSB/MSB-justified, PCM/DSP, TDM, AC'97, SPDIF out |
| SPDIFRX | 1 (**입력 4개**) | 심볼레이트 **12.288 MHz** | IEC-60958 / IEC-61937, 32~192 kHz |
| SDMMC | **1** (SDMMC1) | **50 MHz (8비트 모드)** | MMC 4.2 / SD 2.0 / SDIO 2.0, 1/4/8비트 버스, **SPI 호환 모드 없음** |
| QUADSPI | 1 (dual-flash 가능) | **80 MHz** | SDR/DDR, indirect / status-polling / memory-mapped |
| USB OTG FS | 1 | 12 Mbit/s | 내장 PHY, FIFO 1.28 Kbyte, 1 ctrl + 5 IN + 5 OUT EP, 12 host 채널 |
| USB OTG HS | 1 | **480 Mbit/s (외부 ULPI PHY 필요)** | 내장 FS PHY, FIFO **4 Kbyte**, 8 양방향 EP, 16 host 채널, 전용 DMA |
| Ethernet MAC | 1 | **10 / 100 Mbit/s** | MII(17신호) / RMII(9신호), 전용 DMA, TX·RX FIFO 각 2 Kbyte, IEEE 1588-2008 PTP v2 (타임스탬프 비교기는 **TIM2 입력**에 연결) |
| HDMI-CEC | 1 | — | — |

> **실측 정정 / 보강**
> - **QUADSPI "80 MHz"**: 이 보드는 실제로 **100 MHz(HCLK/2)** 로 구동 중이다 (§1.3의 실측 정정 참조). 스펙 초과 동작이다.
> - **Ethernet MAC**: MCU 블록은 존재하나, 보드 PHY는 MDIO 주소 **0**에서 **LAN8742A**(ID1=0x0007, ID2=0xC131, OUI 0x0001F0)로
>   실측 확정됐다. 프로젝트의 `stm32f7xx_hal_conf.h:220`에 있는 `DP83848_PHY_ADDRESS 0x01`은 **이 보드와 맞지 않는다**.
>   측정 시점 BMSR=0x7809 → 링크 down(케이블 미연결).
> - **I2C**: I2C3 버스에 WM8994(0x34)·FT5336(0x70) 실장 확인. 카메라(0x60)·EEPROM(0xA0/0xA6)은 무응답 = 미장착.
> - 위 표의 나머지 최대 속도 수치(USART 27 Mbit/s, SPI 50/25 Mbit/s, SDMMC 50 MHz 등)는 **아직 실측하지 않았다.**

---

## 7. 시스템 / 안전 페리페럴

| 항목 | 값 | 신뢰도 / 출처 |
|---|---|---|
| RTC | BCD 캘린더(서브초·초·분·시·요일·일·월·년), 알람 2개, 17비트 auto-reload wakeup 타이머, 0.95 ppm 디지털 캘리브레이션, 1~32767 RTC 펄스 on-the-fly 보정, 참조클럭 검출(50/60 Hz), 안티탬퍼 핀 3개, 타임스탬프 | [A] DS10916 §2.19 |
| RTC 클럭 게이트 | `RCC_BDCR` bit 15 (**RTCEN**) — APB1ENR 아님 | [A] CMSIS + RM0385 §5.3.20 |
| 백업 레지스터 | **32개 × 32비트 = 128바이트**, 시스템/파워 리셋 및 Standby 웨이크업에도 유지 | [A] DS10916 §2.19 |
| 백업 SRAM | **4 Kbyte @ 0x4002_4000**, `RCC_AHB1ENR` bit 18, 전류 0.8 µA/MHz | [A] DS10916 Table 2, Table 35 |
| IWDG | **12비트 다운카운터 + 8비트 프리스케일러**, 독립 32 kHz LSI 구동, Stop/Standby에서도 동작, 옵션바이트로 HW/SW 선택 | [A] DS10916 §2.22.5 |
| WWDG | **7비트 다운카운터**, 윈도우(상한) 감시, free-running 설정 가능, `RCC_APB1ENR` bit 11, 전류 1.8 µA/MHz | [A] DS10916 §2.22.6 |
| CRC | 다항식 **7/8/16/32비트 프로그래머블**, `CRC_POL` 리셋값 **0x04C11DB7**, 초기값 프로그래머블(`CRC_INIT`), 8/16/32비트 데이터 입력, **32비트당 4 HCLK 사이클**, 입력 버퍼로 버스 스톨 방지, I/O 비트 반전 옵션, 범용 8비트 레지스터 1개 | [A] RM0385 §12.2 p.297, §12.4.5 |
| RNG | 32비트 아날로그 난수, **연속 출력 간 RNG_CLK 40주기**, 엔트로피 모니터링, 전류 2.8 µA/MHz | [A] RM0385 §19.2 p.542 |
| 96비트 Unique ID | 0x1FF0_F420 | [A] CMSIS `UID_BASE` |
| Flash size 레지스터 | 0x1FF0_F442 | [A] CMSIS `FLASHSIZE_BASE` |
| Package 레지스터 | 0x1FF0_F7E0 | [A] CMSIS `PACKAGE_BASE` |
| OTP 메모리 | **1024바이트 @ 0x1FF0_F000** | [A] DS10916 features, CMSIS `FLASH_OTP_BASE` |

> **실측 정정 (위 표 3개 항목)**
> 1. **IWDG의 "32 kHz LSI"** → 이 개체의 LSI 실측은 **26,616 ~ 26,652 Hz (-16.8 %)** 다 (TIM5 입력캡처).
>    IWDG 타임아웃을 공칭값으로 계산하면 실제로는 약 1.2배 길어진다.
> 2. **Package 레지스터 `0x1FF0_F7E0`** → CMSIS `PACKAGE_BASE`는 확실히 이 주소지만, RM0385가 지정하는 주소는
>    **`0x1FF07BF0`** 로 다르다. 실측 결과 `0x1FF0F7E0 = 0x038DFC72`, `0x1FF07BF0 = 0xFFFFFFFF`(미프로그램).
>    **어느 주소로도 이 다이의 패키지를 판정할 수 없다.**
> 3. **OTP 1024 바이트** → CMSIS 헤더는 `FLASH_OTP_BASE=0x1FF0F000`, `FLASH_OTP_END=0x1FF0F41F`로
>    **1056 바이트**(데이터 1024 B + 잠금 바이트 32 B)를 정의한다.
>
> 확인된 항목: **백업 SRAM 4 KB @0x4002_4000**(폴트 경계 실측 일치), **Unique ID 0x1FF0_F420**(= `0x00340033_54435008_20353036`),
> **Flash size 레지스터 0x1FF0_F442**(= `0x0400` → 1024 KB), **RTC 게이트 `RCC_BDCR` bit 15**(실측 RTCEN=0).
> 덧붙여 BKPSRAM은 **읽기 대역폭만 유독 낮다**(read 140 MB/s vs write 470 MB/s) — 문서에 없는 실측 특성이다.

---

## 8. 메모리 맵 요약 (참고)

| 영역 | 주소 | 크기 | 비고 |
|---|---|---|---|
| ITCM-RAM | 0x0000_0000 | **16 KB** | 크리티컬 루틴용 |
| ITCM Flash (ART 경유) | 0x0020_0000 | 1 MB | ART Accelerator |
| AXI Flash | 0x0800_0000 | **1 MB** | L1 캐시 경유 |
| OTP | 0x1FF0_F000 | 1024 B | |
| DTCM-RAM | 0x2000_0000 | **64 KB** | AHB 마스터도 접근 가능(전용 슬레이브 버스) |
| SRAM1 | 0x2001_0000 | **240 KB** | 전 AHB 마스터 접근 |
| SRAM2 | 0x2004_C000 | **16 KB** | 전 AHB 마스터 접근 |
| 백업 SRAM | 0x4002_4000 | 4 KB | VBAT 도메인 |
| QUADSPI 메모리 맵 | 0x9000_0000 | (보드 16 MB) | N25Q128A |
| FMC SDRAM Bank1 | 0xC000_0000 | **보드 8 MB (16비트 버스)** | MT48LC4M32B2 |

- 총 시스템 SRAM = **320 KB (240 + 16 + 64)** + ITCM 16 KB + 백업 4 KB [A] (DS10916 Table 2 p.13, RM0385 §2.3 p.69)
- 베이스 주소는 CMSIS `stm32f746xx.h` 및 RM0385 §2.3에서 일치 확인 [A]
- SDRAM/QSPI 크기는 ST BSP 헤더에서 확인: `SDRAM_DEVICE_SIZE = 0x800000`(8 MB), `SDRAM_MEMORY_WIDTH = FMC_SDRAM_MEM_BUS_WIDTH_16`, `QSPI_FLASH_SIZE = 23`(주소 버스 폭) + N25Q128A(128 Mbit = 16 MB) [A]

> **실측 정정 / 확인 (위 메모리 맵 표 전체)**
> - **크기는 전부 실측 일치**: ITCM 16 KB(0x00004000부터 BusFault), Flash ITCM 별칭 1 MB, Flash AXIM 1 MB,
>   DTCM 64 KB, SRAM1 240 KB, SRAM2 16 KB, BKPSRAM 4 KB, SDRAM 8,192 KB, QSPI 16,384 KB.
>   총 내부 SRAM 320 KB도 확인. `0x00200000`과 `0x08000000`은 선두 64바이트가 바이트 단위로 동일 = 같은 물리 플래시의 두 경로.
> - **QSPI 부품은 N25Q128A가 아니다**: JEDEC ID 실측 **`0xEF4018` = Winbond W25Q128 계열**. 용량 16 MB만 일치한다.
> - **SDRAM 부품(MT48LC4M32B2)은 실측으로 확인할 수 없다** — SDRAM은 읽을 수 있는 ID 레지스터가 없다.
>   확인된 것은 컨트롤러 구성뿐이다: 4뱅크 × 12행 × 8열 × 16비트 = **8 MB**, CAS 2, SDCLK=HCLK/2=100 MHz,
>   리프레시 15.4 µs (`SDCR1=0x00001954`, `SDTR1=0x01116361`, `SDRTR=0x00000C06`).
> - **"AXI Flash는 L1 캐시 경유, ITCM Flash는 ART 경유"** 라는 표의 설명은 실측으로 뒷받침된다:
>   AXIM 경로는 D-cache에 반응(160 → 305 MB/s, ×1.91)하고 ART에 무반응, ITCM 경로는 ART에 반응(337 → 492 MB/s, ×1.46)하고
>   D-cache에 무반응이다. 다만 **이 보드는 옵션바이트 BOOT_ADD0=0x00200000으로 ITCM 별칭에서 부팅하는데 ART가 꺼져 있었다**
>   (`FLASH_ACR=0x00000006`, ARTEN=0/PRFTEN=0).

---

## 9. 페리페럴별 소비전류 (IDD typ, µA/MHz) [A] (DS10916 Rev 4 Table 35, p.118~120)

SNN/ML 워크로드에서 "켜두면 얼마나 손해인가"를 판단하는 근거다.

| 페리페럴 | Scale 1 | Scale 2 | Scale 3 |
|---|---|---|---|
| **DMA2D** | **87.1** | 82.5 | 69.6 |
| **LTDC** | **56.7** | 53.8 | 45.7 |
| OTG_HS (+ULPI) | 57.5 | 54.4 | 47.6 |
| ETH_MAC (전체) | 42.1 | 39.7 | 34.1 |
| USB_OTG_FS | 31.8 | 29.9 | 25.8 |
| TIM8 | 25.3 | 24.0 | 20.4 |
| TIM1 | 25.2 | 23.9 | 20.4 |
| QSPI | 23.2 | 21.8 | 18.5 |
| Bus matrix | 21.06 | 20.3 | 17.2 |
| TIM2 | 19.8 | 18.7 | 16.1 |
| TIM5 | 19.0 | 17.8 | 15.8 |
| FMC | 18.9 | 17.7 | 15.2 |
| DCMI | 5.1 | 4.7 | 4.0 |
| ADC1/2/3 (각) | 4.5 | 4.4 | 3.5 / 3.3 |
| SDMMC1 | 8.5 | 7.9 | 6.7 |
| TIM6 / TIM7 (각) | 3.0 | 2.7 | 2.5 |
| RNG | 2.8 | 2.4 | 2.3 |
| GPIOA~K (각) | 1.9~2.3 | 1.7~2.1 | 1.7~1.9 |
| **CRC** | **1.0** | 0.9 | 0.8 |
| BKPSRAM | 0.8 | 0.7 | 0.6 |
| DMA1 | 2.7 × N + 5.1 | 2.6 × N + 4.7 | 2.2 × N + 4.0 |
| DMA2 | 2.2 × N + 4.9 | 2.6 × N + 4.4 | 2.2 × N + 4.1 |

- DMA 식의 **N은 활성 스트림 수** [A]
- ADC ON(`ADON`) 시 아날로그부 **1.73 mA/ADC** 추가, DAC EN 시 **0.75 mA/채널** 추가 [A] (Table 35 note 4, 5)
- I/O 보상 셀 ON 시 IDD **+0.22 mA** [A] (Table 35 note 1)

> 해석 [D]: 200 MHz 기준 DMA2D 상시 인에이블은 약 **17.4 mA**(87.1 µA/MHz × 200), LTDC는 약 **11.3 mA**에 해당한다. 반면 CRC는 약 **0.2 mA**로 사실상 무료다. 순수 추론 벤치마크 시 LTDC/DMA2D/ETH/USB 클럭을 끄면 수십 mA 절감 여지가 있다.

---

## 10. SNN / ML 워크로드 관점 활용도 코멘트

| 페리페럴 | 활용 가능성 | 근거 / 코멘트 | 신뢰도 |
|---|---|---|---|
| **TIM2 / TIM5 (32비트)** | ★★★ **강력 추천** | 100 MHz 구동 시 10 ns 분해능 · **42.9초** 무랩어라운드. 4494/9641/2407 ms 급 추론 시간을 단일 카운터로 오차 없이 측정 가능. 16비트 타이머는 100 MHz에서 655 µs만에 오버플로우하므로 부적합 | [D] (계산: 2^32/100e6) |
| **DWT CYCCNT (Cortex-M7 코어)** | ★★★ | 페리페럴은 아니지만 사이클 단위 프로파일링의 표준 수단. 현재 프로젝트에서 미사용(`HAL_GetTick()` 1 ms 분해능만 사용 중) | [D] (프로젝트 grep 결과) |
| **CRC** | ★★★ | 1.0 µA/MHz, 32비트당 **4 HCLK**. 가중치 배열/입력 텐서/출력 로짓 무결성 검증에 CPU 부담 거의 없음. 다항식·초기값 프로그래머블이라 임의 체크섬 규격 대응 가능 | [A]+[D] |
| **DMA2 (memory-to-memory)** | ★★★ | **DMA1은 m2m 불가, DMA2만 가능**. QSPI/SDRAM ↔ DTCM 간 가중치 타일 프리페치에 사용해 CPU와 오버랩 가능. 4/8/16 beat 버스트 + 4워드 FIFO로 SDRAM 대역폭 활용 | [A] |
| **DMA2D (Chrom-ART)** | ★★☆ | 이미지 전처리에 유효: **memory-to-memory + PFC 모드**로 RGB565→ARGB8888 등 픽셀 포맷 변환, 영역 복사, crop(오프셋 레지스터)을 CPU 없이 수행. 단 **리사이즈·스케일링 기능은 없고**, 정규화(subtract mean/scale) 같은 산술도 불가 → CIFAR 32×32 입력 정규화는 여전히 CPU 몫. 87.1 µA/MHz로 전류 비용 최대 | [A]+[D] |
| **DCMI** | ★★☆ | 실시간 카메라 추론 데모용. RGB565 직결 + **crop 내장**이라 32×32/96×96 ROI 추출을 하드웨어로 처리 가능. 54 Mbyte/s. 단 보드는 카메라 커넥터만 제공(모듈 별매) | [A] |
| **LTDC** | ★★☆ | 480×272 패널에 추론 결과/입력 이미지 시각화. 레이어 2개라 이미지+오버레이 텍스트 분리 가능. 벤치마크 측정 시에는 AHB 대역폭과 11 mA를 잡아먹으므로 **끄는 것을 권장** | [A]+[D] |
| **QUADSPI (memory-mapped)** | ★★★ | 보드 16 MB NOR. **memory-mapped 모드로 가중치를 XIP 접근** 가능 → 1 MB Flash 한계를 넘는 모델 상주. 80 MHz. 단 랜덤 액세스 지연이 커서 L1 캐시/DTCM 스테이징과 병행 필요 | [A]+[D] |
| **FMC / SDRAM** | ★★★ | 보드 8 MB @ 0xC000_0000. 큰 활성화 맵/프레임버퍼 배치. 16비트 버스라 대역폭은 32비트 대비 절반 | [A] |
| **BKPSRAM (4 KB)** | ★★☆ | 리셋/파워사이클을 넘어 벤치마크 누적 통계·최고 정확도 체크포인트·실행 카운터 보존. 0.8 µA/MHz로 저렴 | [A]+[D] |
| **RTC + 백업 레지스터(128 B)** | ★★☆ | 실험 타임스탬프 기록, 장기 로깅. 백업 레지스터는 리셋에도 유지되므로 "몇 회차 실행인지" 카운터로 적합 | [A] |
| **RNG** | ★★☆ | SNN 확률적 스파이크 생성(Bernoulli/Poisson 인코딩), 드롭아웃, 몬테카를로 추론에 하드웨어 엔트로피 제공. 단 **연속 난수 간 RNG_CLK 40주기** 지연이 있어 고빈도 요구 시 LFSR 소프트웨어 PRNG 대비 병목 가능 | [A]+[D] |
| **ADC (3×12비트)** | ★★☆ | 아날로그 센서 입력 기반 추론(음성·진동·전류). 2.4 MSPS, TIM1~TIM5/TIM8 트리거 + DMA2 연결로 CPU 개입 없는 연속 샘플링 파이프라인 구성 가능. **전력 자가측정**(션트 전압)에도 활용 가능 | [A]+[D] |
| **DAC (2채널)** | ★☆☆ | 추론 결과 아날로그 출력, 또는 테스트 신호 생성. ML 코어 경로와 무관 | [D] |
| **SDMMC1** | ★★☆ | microSD에 데이터셋 배치/가중치/로그 저장. 8비트 50 MHz. DMA2 Stream3/6 Channel4 매핑 | [A] |
| **USART1/6 (27 Mbit/s)** | ★★★ | 호스트 PC로 벤치마크 결과 스트리밍. DMA2 매핑(Stream2/5 RX, Stream7 TX)이라 로그 전송 중 CPU 블로킹 회피 가능 | [A]+[D] |
| **Ethernet MAC** | ★☆☆ | 원격 추론 서버/대량 로그 업로드. 42.1 µA/MHz + PHY 전력. 본 워크로드에는 과함 | [D] |
| **USB OTG FS/HS** | ★☆☆ | CDC로 결과 전송 또는 대용량 데이터 로딩. HS는 외부 ULPI PHY 필요 | [A] |
| **CAN / SPDIFRX / HDMI-CEC / SAI** | ☆☆☆ | SNN 추론 워크로드와 직접 관련 없음 (SAI는 오디오 SNN 시에만) | [D] |
| **IWDG / WWDG** | ★★☆ | 장시간 무인 벤치마크 중 행(hang) 복구. IWDG는 LSI 독립 구동이라 메인 클럭 장애에도 동작 | [A]+[D] |
| **JPEG 코덱** | — | **F746 미탑재.** 필요 시 SW 디코더 또는 F767/F769로 변경 | [A] |
| **CRYP / HASH** | — | **F746 미탑재.** 모델 암호화·무결성 서명이 필요하면 SW 구현 또는 F756/F777 계열 필요. 무결성만이면 CRC 유닛으로 대체 | [A] |

> **실측 정정 / 보강 (§10 활용도 표)**
> - **QUADSPI 행의 "80 MHz"** → 이 보드는 **100 MHz**로 동작 중이다(스펙 초과). "랜덤 액세스 지연이 크다"는 경고는
>   실측으로 강하게 뒷받침된다: 메모리맵 순차 읽기 **약 50 MB/s**, 64 KB 스트라이드 접근 **94~198 cyc/접근**.
>   XIP로 가중치를 두면 캐시/DTCM 스테이징 없이는 매우 느리다.
> - **FMC/SDRAM 행** → 실측 순차 읽기 **60.25 MB/s**(이론 200 MB/s의 30%), 쓰기 194.6 MB/s, SDRAM→SRAM1 복사 58.4 MB/s.
>   읽기가 특히 느린 이유 중 하나는 **MPU를 설정하지 않으면 0xC0000000이 Device 메모리라 D-cache에 캐싱되지 않기 때문**이다
>   (D$OFF와 D$WB 수치가 59.7 MB/s로 동일한 것이 그 증거). 큰 활성화 맵을 SDRAM에 두려면 MPU 설정이 사실상 필수다.
> - **BKPSRAM 행** → 실측 읽기 140 MB/s / 쓰기 470 MB/s로 **읽기만 유독 느리다**. 통계 누적용으로는 문제없다.
> - **DWT CYCCNT 행** → 실측으로 사용 가능 확인(비교기 4개, CYCCNT·프로파일 카운터 모두 존재).
>   **단 `DEMCR.TRCENA`를 켜야 보인다.**
> - **CRC / RNG / DMA2 m2m / ADC 처리량**은 아직 실측하지 않았다(측정 방법은 문서 앞부분 "아직 측정하지 않음" 절 참조).

---

## 11. 본 프로젝트(ImageClassification)의 실제 페리페럴 사용 현황

`__HAL_RCC_*_CLK_ENABLE()` 호출 기준 실제 인에이블되는 페리페럴 [A] (소스 grep):

- **AHB1**: GPIOA~GPIOK(11개 전부), CRC, DMA1, DMA2, DMA2D, ETH
- **AHB2**: DCMI
- **AHB3**: FMC, QSPI
- **APB1**: TIM2, TIM3, TIM5, TIM6, TIM12, I2C1, I2C3, SPDIFRX, SPI2, PWR
- **APB2**: TIM1, TIM8, ADC3, LTDC, SAI2, SDMMC1, USART1, USART6, SYSCFG

> **실측 정정 — 이 목록은 "소스에 호출이 있다"는 뜻이지 "런타임에 켜져 있다"는 뜻이 아니다.**
> SNN 펌웨어가 돌고 있는 상태에서 ST-LINK로 RCC 인에이블 레지스터를 그대로 떠 보면,
> **클럭이 실제로 공급되던 페리페럴은 18개뿐이다**:
>
> | 레지스터 | 실측값 | 켜져 있던 것 |
> |---|---|---|
> | `RCC_AHB1ENR` | `0x00D007FC` | GPIOC~GPIOK, DTCMRAM, DMA2, DMA2D |
> | `RCC_AHB2ENR` | `0x00000000` | (없음 — **DCMI OFF**) |
> | `RCC_AHB3ENR` | `0x00000001` | FMC (**QSPI OFF**) |
> | `RCC_APB1ENR` | `0x10000010` | TIM6, PWR |
> | `RCC_APB2ENR` | `0x04004800` | SDMMC1, SYSCFG, LTDC |
>
> 즉 **ETH·DCMI·SAI2·SPDIFRX·SPI2·I2C1/I2C3·ADC3·USART1/6·TIM1/2/3/5/8/12·CRC·DMA1·QSPI의 클럭은 꺼져 있었다.**
> (GPIOA/GPIOB는 클럭이 꺼져 있어 덤프 아티팩트가 발생했고, 이는 두 포트의 클럭이 실제로 OFF였다는 방증이기도 하다.)
> 원인 추정: 조건부 초기화로 호출에 도달하지 않았거나, BSP `DeInit` 경로에서 다시 꺼졌거나, 초기화 직후 해제된 것이다.

관찰 [D]:
1. **DMA2D, LTDC, ETH, DCMI, SDMMC1, SAI2, SPDIFRX가 모두 인에이블**되어 있다. 순수 추론 시간 측정 시 이들 클럭을 끄면 AHB 매트릭스 경합과 소비전류를 줄일 수 있다 (합계 약 200 µA/MHz ≈ 200 MHz에서 40 mA 규모).
2. **TIM2, TIM5(32비트) 모두 클럭이 켜져 있으나 시간 측정용으로 쓰이지 않는다.** HAL 타임베이스는 TIM6(1 kHz)이고 측정은 `HAL_GetTick()` 기반 1 ms 분해능이다. 4494 ms 급 측정에서 1 ms 분해능은 상대오차 0.02% 수준이라 실용상 문제는 없으나, 레이어별 프로파일링(수백 µs~수 ms)에는 TIM2/TIM5 또는 DWT CYCCNT 전환이 유리하다.
3. **CRC 클럭이 이미 켜져 있다** — 가중치 무결성 검증을 추가 비용 없이 넣을 수 있다.
4. 클럭 설정이 **200 MHz**(PLLM=25, PLLN=400, PLLP=2, Flash 6 WS)이므로, 데이터시트 최대치인 216 MHz 대비 **약 7.4% 성능 여유**가 남아 있다. 216 MHz로 올리려면 PLLN=432, Flash 7 WS + 오버드라이브 ON이 필요하다 [D] (DS10916 Table 18: VDD 2.7~3.6 V에서 216 MHz는 7 WS + over-drive ON).
5. PLLQ=9 → 48 MHz PLL Q 출력이 400/9 ≈ **44.4 MHz**로, USB FS(48 MHz 요구) 및 SDMMC 표준 클럭에는 부정확하다. USB/SDMMC를 실제로 쓸 경우 재검토 필요 [D].

> **실측 정정 — 위 관찰 1·2·3에 대해**
> - **관찰 1 (DMA2D/LTDC/ETH/DCMI/SDMMC1/SAI2/SPDIFRX가 모두 인에이블)**: **부분적으로 틀렸다.**
>   실행 중 실제로 켜져 있던 것은 **DMA2D, LTDC, SDMMC1, DMA2, FMC**뿐이고 ETH·DCMI·SAI2·SPDIFRX는 **꺼져 있었다.**
>   따라서 "합계 약 200 µA/MHz ≈ 40 mA 절감 여지"는 **과대 추정**이다. 실제 절감 대상은 DMA2D(87.1)+LTDC(56.7)+SDMMC1(8.5)
>   ≈ 152 µA/MHz 규모이고, 이마저도 전류값 자체는 아직 실측하지 않았다.
> - **관찰 2 (TIM2/TIM5 클럭이 켜져 있다)**: **틀렸다.** 실측 `APB1ENR=0x10000010` → TIM2EN=0, TIM5EN=0.
>   켜져 있던 APB1 페리페럴은 **TIM6와 PWR뿐**이다. 다만 "HAL 타임베이스가 TIM6"라는 §2.2의 서술은 이 값으로 **뒷받침된다**.
>   TIM2/TIM5를 프로파일링에 쓰려면 **클럭부터 새로 켜야 한다**.
> - **관찰 3 (CRC 클럭이 이미 켜져 있어 추가 비용 없다)**: **틀렸다.** 실측 `AHB1ENR=0x00D007FC` → **CRCEN=0**.
>   CRC를 쓰려면 클럭을 직접 켜야 한다.
> - **관찰 4 (200 MHz 동작)**: **맞다.** `RCC_PLLCFGR=0x09406419`, `RCC_CFGR=0x0000940A`, `FLASH_ACR=0x00000006`으로 확인.
>   전원도 Scale 1 + over-drive 활성(`PWR_CR1=0x0003C000/0x0003C100`, `PWR_CSR1=0x00034000`)이다.
>   216 MHz와의 성능 비교는 **아직 측정하지 않았다.**
> - **관찰 5 (PLLQ=9 → 44.4 MHz)**: **맞다.** 실측 역산값이 정확히 44,444,444 Hz다. 이 보드에서 USB는 사용되지 않았고,
>   SDMMC1은 클럭이 켜져 있었으나 microSD 접근 성능은 측정하지 않았다.
> - 덧붙여 **DWT는 실제로 쓸 수 있다**: TRCENA를 켠 뒤 `DWT_CTRL=0x40000001` → 비교기 4개, CYCCNT 있음,
>   프로파일 카운터(CPI/EXC/SLEEP/LSU/FOLD) 있음. §10에서 "★★★"로 추천한 근거가 실측으로 확인됐다.
>   단 **`DEMCR.TRCENA`를 켜기 전에는 DWT 레지스터가 전부 0으로 읽힌다**(부재로 오인하기 쉽다).

---

## 출처

### 1차 문서 (본문에서 직접 인용)

| 문서번호 | 제목 / 판본 | URL |
|---|---|---|
| **DS10916** (= DocID027590 Rev 4, 2016-02) | STM32F745xx / STM32F746xx 데이터시트 | https://www.st.com/resource/en/datasheet/stm32f746ng.pdf (본 조사에서는 https://indico.kit.edu/event/460/contributions/5488/attachments/2862/4149/Datasheet_STM32F746.pdf 사본 사용) |
| **RM0385** (= DocID026670 Rev 5, 2016-04) | STM32F75xxx and STM32F74xxx advanced ARM-based 32-bit MCUs — Reference manual | https://www.st.com/resource/en/reference_manual/rm0385-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf (본 조사에서는 https://download.kamami.pl/p1188987-STM32F7-Reference-EN.pdf 사본 사용) |
| ST CMSIS Device Header | `stm32f746xx.h` — CMSIS Cortex-M7 Device Peripheral Access Layer Header File | 로컬: `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h` |
| ST BSP (STM32746G-Discovery) | `stm32746g_discovery_sdram.h`, `stm32746g_discovery_qspi.h`, `rk043fn48h.h` | 로컬: `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/BSP/` |

### 참조한 ST 공식 페이지 / 문서 (보조)

- STM32F746NG 제품 페이지 — https://www.st.com/en/microcontrollers-microprocessors/stm32f746ng.html
- 32F746GDISCOVERY 제품 페이지 — https://www.st.com/en/evaluation-tools/32f746gdiscovery.html
- **UM1907** Discovery kit with STM32F746NG MCU — https://www.st.com/resource/en/user_manual/um1907-discovery-kit-for-stm32f7-series-with-stm32f746ng-mcu-stmicroelectronics.pdf
- **MB1191** STM32F746G-DISCO 회로도 — https://www.st.com/resource/en/schematic_pack/mb1191-f746ngh6-c01_schematic.pdf
- **ES0290** Rev 7 STM32F74xxx/STM32F75xxx 에라타 — https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf
- **AN4031** Using the STM32F2, STM32F4 and STM32F7 Series DMA controller — https://www.st.com/resource/en/application_note/an4031-using-the-stm32f2-stm32f4-and-stm32f7-series-dma-controller-stmicroelectronics.pdf
- **AN4861** Introduction to LCD-TFT display controller (LTDC) on STM32 MCUs — https://www.st.com/resource/en/application_note/an4861-introduction-to-lcdtft-display-controller-ltdc-on-stm32-mcus-stmicroelectronics.pdf
- **AN4013** Introduction to timers for STM32 MCUs — https://www.st.com/resource/en/application_note/an4013-introduction-to-timers-for-stm32-mcus-stmicroelectronics.pdf
- STM32F7 Series 제품 소개 자료 (crypto/JPEG 파트별 탑재 여부) — https://www.st.com/content/ccc/resource/training/technical/product_training/group0/37/bf/4c/3b/0d/59/47/11/STM32F7_Introduction_Product_Series_presentation/files/STM32F7_Introduction_Product_Series_presentation.pdf
- STM32F7 시리즈 브로슈어 — https://www.mouser.com/datasheet/2/389/en.brstm32f7-930327.pdf
- STM32F7x7 제품군 페이지 (JPEG 코덱 탑재 라인 확인) — https://www.st.com/en/microcontrollers-microprocessors/stm32f7x7.html

### 2차 자료 (교차확인용, [C] 등급)

- STM32F746G Discovery — Zephyr Project Documentation — https://docs.zephyrproject.org/latest/boards/st/stm32f746g_disco/doc/index.html
- DISCO-F746NG — Mbed 플랫폼 페이지 — https://os.mbed.com/platforms/ST-Discovery-F746NG/
- ST Community: Max internal clock for STM32F7 timers — https://community.st.com/t5/stm32-mcus-products/max-internal-clock-for-stm32f7-timers/td-p/434666

---

## 미확인 / 추가 조사 필요

1. **ES0290 에라타 본문 미확인.** ST 서버 PDF 다운로드가 타임아웃/차단되어 Rev Z 실리콘에 해당하는 개별 device limitation(특히 DMA2D, DMA FIFO, I2C, SDMMC1, ADC 관련)을 원문에서 확인하지 못했다. 검색 결과 수준에서 SDMMC1(`CCRCFAIL` 오탐, 8바이트 미만 MMC stream write 오류), I2C(마스터 모드 spurious bus error, reload 모드 마지막 바이트 손실), SPI/I2S 제한이 존재한다는 것까지만 확인 [C]. **Rev Z에 실제 적용되는 항목의 정확한 번호와 워크어라운드는 미확인.**
2. **UM1907 / MB1191 원문 미확인.** 보드 레벨 배선(각 페리페럴이 실제로 어느 핀·커넥터에 연결되는지, LCD 백라이트 제어 핀, 카메라 커넥터 신호 매핑, Arduino 헤더 핀맵)은 ST BSP 헤더로 일부만 역산했다. 회로도 원문 대조 필요.
3. **LTDC 최대 픽셀 클럭의 조건별 값** — DS10916에서 `fCLK` LTDC clock output frequency 최대 **45 MHz**는 확인했으나, 이것이 어떤 VDD/온도 조건에서의 값인지, 그리고 480×272@60 Hz 요구치(약 9.5 MHz) 대비 마진 계산의 근거 표 번호를 정확히 특정하지 못했다.
4. **QUADSPI 80 MHz의 조건** — DS10916 Quad-SPI 특성표에서 `Fck 최대 80 MHz`는 확인했으나, DDR 모드/SDR 모드별 구분과 부하 커패시턴스 조건을 확인하지 못했다.
5. **RM0385의 각 페리페럴 최대 클럭이 216 MHz 최종 판본(Rev 8/Rev 9)에서 변경되었는지** — 본 조사는 Rev 5(2016-04) 기준이다. 최신 Rev 9(2025-02)와의 차이(특히 타이머/ADC 사양 정정)는 미확인.
6. **DMA1/DMA2 각 스트림의 인터럽트 우선순위 기본값과 NVIC 벡터 번호 전체 표**는 정리하지 않았다. 필요 시 CMSIS 헤더 `IRQn_Type` enum에서 추출 가능.
7. **ADC 24채널의 실제 핀 할당** (F746Nx TFBGA216에서 ADC1/2/3에 각각 어떤 채널이 배정되는지)은 DS10916 Table 10 핀 정의에서 추출해야 하며 본 문서에는 미포함.
8. **PLLQ=9(44.4 MHz)로 USB FS/SDMMC를 구동할 때의 실제 동작 가능 여부** — 계산상 48 MHz를 벗어나지만 프로젝트에서 USB를 실제로 사용하지 않아 검증하지 못했다.

> **실측으로 일부 해소된 항목**
> - 위 3번(LTDC 픽셀클럭 마진): 이 보드는 실제로 **9.60 MHz / 480×272 / 59.3 Hz**로 구동된다.
>   데이터시트 최대 45 MHz 대비 마진이 4.7배이므로 실무상 문제되지 않는다.
> - 위 4번(QUADSPI 80 MHz의 조건): 조건을 특정하지는 못했지만, **이 보드는 SDR·4-line·SSHIFT=1 상태에서 100 MHz로
>   정상 동작한다**는 실측 사실이 확보됐다. 즉 80 MHz는 보수적 보증치로 보인다(권장이 아니라 관측).
> - 위 6번(NVIC 벡터 전체 표): CMSIS 헤더 파싱으로 **페리페럴 IRQ 97개(번호 0~97), 총 105 벡터**,
>   `DMA2D_IRQn = 90`을 확인했다 (`info/raw/host_side.md`).
