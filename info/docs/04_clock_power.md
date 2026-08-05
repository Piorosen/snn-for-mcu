# STM32F746G-DISCO 클럭 트리 · 전원 · 전기적 특성

대상 하드웨어: STM32F746G-DISCO (32F746GDISCOVERY, PCB: MB1191) / MCU: **STM32F746NGH6**
실측 식별자: DEV_ID = `0x449`, REV_ID = `0x1001` (Revision Z)

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거 자료: `info/raw/measured_facts.md`, `info/results/probe_output.txt`,
> `info/raw/decoded_hotplug.md`(실행 중), `info/raw/decoded_reset.md`(리셋 직후).

### 실측으로 확인됨

- **본 프로젝트가 200 MHz로 동작(§6.2)** — 문서 PLLM=25/PLLN=400/PLLP=2/PLLQ=9, APB1 ÷4, APB2 ÷2, 6 WS
  → 실측 레지스터가 한 글자도 어긋나지 않는다: `RCC_PLLCFGR = 0x29406419`(PLLM=25, PLLN=400, PLLP=2,
  PLLQ=9, PLLSRC=HSE), `RCC_CFGR = 0x0000940A`(SW=SWS=PLL, HPRE=÷1, PPRE1=÷4, PPRE2=÷2),
  `FLASH_ACR = 0x00000006`(6 WS). 펌웨어 자가보고 `SYSCLK = 200,000,000 Hz`,
  **PCLK1 = 50 MHz, PCLK2 = 100 MHz**. → 문서 §6.2 표 전체 CONFIRMED.
- **Flash wait state 표(§4.3)** — 문서 "180 < HCLK ≤ 210 MHz → 6 WS" → 실측 LATENCY=6.
  리셋 직후 `FLASH_ACR = 0x00000000`(0 WS)도 문서의 "리셋 직후 0 WS"와 일치.
- **over-drive가 200 MHz의 전제(§7)** — 실측 `PWR_CR1 = 0x0003C100`(VOS=`0b11` Scale 1, ODEN=1, ODSWEN=1),
  `PWR_CSR1 = 0x00034000`(VOSRDY=1, ODRDY=1, ODSWRDY=1). 비트 위치·의미가 문서 §7.1과 정확히 일치.
- **VOS 리셋값 = Scale 1(`0b11`)(§7.1 표)** — 리셋 직후 덤프 `PWR_CR1 = 0x0000C000`
  (VOS=`0b11`, ODEN=0, ODSWEN=0). 즉 **over-drive는 소프트웨어가 켜는 것**이 맞다.
- **"VOS 값은 PLL이 ON 되어야 실제 적용된다"(§7.1)** — 실측으로 그 결과가 확인됐다:
  SYSCLK이 HSI로 도는 동안 `PWR_CSR1.VOSRDY`는 **셋되지 않는다.** VOS 설정 후 이 플래그를 기다리도록
  코드를 짜면 그 자리에서 영구 정지한다(PC=`0x08000EE8`에서 멈춘 것을 ST-LINK로 확인).
  ST HAL도 이 플래그를 기다리지 않는다. → 문서 서술의 실무적 귀결이 실증됨.
- **LSE 32.768 kHz 크리스탈 실장(§5, X3)** — 실측: `BDCR = 0x00000003`(LSEON=1, LSERDY=1, LSEBYP=0,
  LSEDRV=`00` low drive), TIM5 입력 캡처로 **32,743 ~ 32,765 Hz**(공칭 대비 −0.01 ~ −0.08 %).
  기동 전 LSEON=0이었고 켜자 LSERDY가 올라왔다 → **크리스탈이 물리적으로 붙어 있음이 확정**.
- **LSI 규격 17 / 32 / 47 kHz(§2.3)** — 실측 **26,616 ~ 26,652 Hz**. 규격 범위 안이지만 typ 대비 **−16.7 %**.
  문서가 §2.3 말미에 적은 "IWDG 타임아웃은 최악값으로 계산하라"는 경고가 이 개체에서 그대로 현실이다
  (공칭으로 계산하면 실제 타임아웃이 **1.2배 길어진다**).
- **HSE = 25 MHz(§5, X2)** — 직접 주파수를 잰 것은 아니지만 **간접 확인**된다: HSE 기반 PLL에서 만든
  100 MHz 타이머로 잰 LSE가 32,743~32,765 Hz(참값 32,768)이므로, HSE 유래 타임베이스의 오차가
  **0.1 % 이내**다. HSE가 25 MHz가 아니었다면 이 값이 비례해서 어긋났을 것이다.
- **X2가 이더넷 PHY에도 클럭을 공급(§5)** — 실측: MDIO 스캔에서 **주소 0에서만** 응답,
  PHY ID1 `0x0007` / ID2 `0xC131`(OUI `0x0001F0` SMSC/Microchip, 모델 `0x13`) → **LAN8742A 확정**.
  PHY가 클럭을 받아 살아 있다는 뜻이므로 `OSC_25M` 배선 서술과 정합. (BMSR `0x7809` → 케이블 미연결)
- **`RCC_PLLCFGR` 리셋값 `0x2400 3010`(§3.1)** — 리셋 직후 덤프가 정확히 `0x24003010`
  (PLLM=16, PLLN=192, PLLP=2, PLLQ=4, PLLSRC=HSI). `RCC_CFGR` 리셋값 `0x0000 0000`도 일치.
- **메인 PLL에 PLLR이 없다(§3.1)** — 리셋값 비트[30:28]가 `0b010`으로 남고 CMSIS 헤더에
  `RCC_PLLCFGR_PLLR` 정의가 없다. 문서 서술과 정합.
- **PLL 주파수 규격 준수(§3.2)** — 실측 구성의 VCO 입력 1 MHz(규격 0.95~2.10), VCO 출력 400 MHz
  (규격 192~432), PLLP 출력 200 MHz(규격 ≤216). 전부 규격 안.
  PLLQ=9 → **44.44 MHz**로 USB FS 48 MHz가 아니라는 문서 §6.2의 지적도 레지스터로 확인.
- **VBAT 핀이 VDD에 직결(§11.5 주의)** — 실측 VBAT **3.304 ~ 3.306 V**, VDDA **3.311 ~ 3.313 V**.
  백업 배터리가 없어 사실상 같은 전압이다.
- **VREFINT 규격 1.18 / 1.21 / 1.24 V(§11.5)** — 실측 VREFINT 코드 1486~1487
  → 1486 / 4095 × 3.313 V ≈ **1.202 V**. 규격 범위 안.
- **온도센서 2점 캘리브레이션 방식(§11.4 (b)(c))** — 프로브 펌웨어가 문서와 **동일한 식**
  (`T = 80/(TS_CAL2−TS_CAL1) × (TS_DATA×VDDA/3.3 − TS_CAL1) + 30`)으로 다이 온도 **35.7 ~ 39.2 °C**를 산출했다.
  식이 실제로 동작함이 확인됨.
- **Avg_Slope 2.5 mV/°C, V25 0.76 V(§11.2)** — 이 개체의 팩토리 값으로 역산하면
  (1195 − 942) / 80 °C = 3.16 code/°C → × 3.3 V/4095 = **2.55 mV/°C**(typ 2.5와 일치),
  30 °C에서 942 code = 0.759 V → 25 °C 환산 **0.746 V**(typ 0.76과 1.8 % 차, 개체 편차 범위).
- **VDD 3.3 V 동작(§4.4 맨 아래 행)** — 실측 VDDA 3.311~3.313 V, ST-LINK 보고 타깃 전압 3.23 V.
  2.7~3.6 V 행이 적용되는 것이 맞다.

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

1. **ADC 캘리브레이션 값의 주소 (§11.3, 요약 8번째 항목) — 가장 심각한 오류**
   - 문서: `VREFIN_CAL = 0x1FF0 7A4A`, `TS_CAL1 = 0x1FF0 7A4C`, `TS_CAL2 = 0x1FF0 7A4E`.
   - 실측: **`0x1FF0F44A` = 1492(VREFINT_CAL), `0x1FF0F44C` = 942(TS_CAL1), `0x1FF0F44E` = 1195(TS_CAL2)**.
     ST-LINK 덤프와 온디바이스 판독이 일치하며, 이 값으로 계산한 VDDA·다이 온도가 물리적으로 타당하다.
   - 원인: **`0x1FF07A4A`는 STM32F4 계열(F42x/F43x 등)의 주소**다. 문서가 DS10916을 읽으면서
     F4 계열 자료의 주소를 옮겨 적었거나, "0x1FF0**F4**4A"의 숫자를 잘못 옮긴 것으로 보인다.
     `info/docs/03_memory_map.md` §7(같은 조사 결과물)은 `0x1FF0F44A`로 올바르게 적고 있어 문서 간에도 어긋난다.
   - **영향**: 이 주소를 그대로 코드에 넣으면 캘리브레이션 값이 아니라 부트로더 코드 바이트를 읽게 되어
     온도·VDDA 계산이 조용히 틀린다. 반드시 `0x1FF0F44A/44C/44E`를 쓸 것.
2. **"ART/L1-cache 덕분에 wait state가 실행 속도에 사실상 영향을 주지 않는다"(§4.3 말미)**
   - 문서: DS10916 Table 19 note 2를 근거로 "0-wait 동등 성능".
   - 실측: ① 이 보드의 SNN 펌웨어는 **ART가 꺼진 상태**로 돌고 있었다
     (`FLASH_ACR = 0x00000006` → ARTEN=0, PRFTEN=0). 그런데 `VTOR = 0x00200000`,
     즉 **코드가 ITCM 별칭 플래시에서 실행**되므로 L1 I-cache는 그 경로에 적용되지 않는다.
     → **명령어 페치에 아무 가속도 걸려 있지 않았다.**
     ② ART를 켜도 ITCM 경로 플래시 읽기는 **2.46 byte/cycle**이 상한이다(꺼면 1.68 B/cyc, ×1.46 차이).
     0-WS 등가라면 나와야 할 값(≈4.6 B/cyc @6 WS·256 bit 라인)의 절반 수준이다.
   - 원인: 문서가 ST의 마케팅성 각주를 조건 없이 옮겼다. 실제로는 **(a) 소프트웨어가 ART를 켜야 하고
     (b) 켜도 0-WS 등가는 아니다.** 원인 규명: `ImageClassification/Core/Inc/stm32f7xx_hal_conf.h:150`의
     `#define PREFETCH_ENABLE 0U`이고 `ART_ACCLERATOR_ENABLE`는 아예 정의되어 있지 않다.
3. **HSE 모드 — 문서가 맞고 실측 노트의 표현이 오해를 부른다 (경미)**
   - 문서(§5, §5.2): 보드의 X2는 `NZ2520SB-25.00M` **능동 오실레이터**이며, 그럼에도 ST 예제와 본 프로젝트는
     `RCC_HSE_ON`(크리스탈 모드)을 쓴다.
   - 실측 노트(`measured_facts.md` §10): "HSEBYP=0 → **크리스탈 발진 모드**".
   - 판정: **모순이 아니다.** `HSEBYP`는 MCU의 발진기 드라이버 설정일 뿐 보드에 무엇이 붙었는지를 말하지 않는다.
     실측은 문서 §5.2가 지적한 "부품은 오실레이터인데 소프트웨어는 크리스탈 모드"라는 상황을 **확인해 준 것**이다.
     실측 노트의 문장을 "이 보드에 25 MHz 크리스탈이 실장되어 있다"로 읽으면 안 된다.
     (LSE 쪽은 반대로 LSEON을 껐다 켜는 실험까지 해서 크리스탈 실장이 **실증**됐다.)

### 아직 측정하지 않음

- **전류 소비 표 전체(§8.1~§8.7)** — 보드 5 V 입력(또는 USB) 라인에 전류계를 직렬로 넣고
  RUN / Sleep / Stop / Standby를 순회하며 측정. MCU 단독값은 얻을 수 없으므로 **모드 간 상대 비교**로 쓴다.
- **캐시/ART ON·OFF의 전류 차이(§8.2)** — 위와 같은 방법으로 `SCB_CCR.IC/DC`와 `FLASH_ACR.ARTEN`을
  토글하며 전류를 비교. 실행 시간은 이미 실측되어 있으므로 곱하면 §8.2가 요구한 mJ/추론이 나온다.
- **저전력 모드 웨이크업 시간(§8.6)** — WFI 직전에 GPIO를 세우고 웨이크업 직후 다시 세운 뒤
  DWT_CYCCNT(Sleep) 또는 LPTIM/RTC(Stop·Standby)로 간격 측정. Standby는 리셋 경로라
  BKPSRAM에 타임스탬프를 남겨 비교.
- **over-drive 스위칭 시간 `Tod_swen` 45 µs(§7.4)** — `PWR_CR1.ODSW` 세트 직전/`ODSWRDY` 셋 직후의
  DWT_CYCCNT 차이. 단, 이 구간은 클럭이 stall되므로 외부 타이머(LSE 기반)로 재는 편이 정확하다.
- **PLL 락 시간 `tLOCK` 100~300 µs(§3.2)** — `RCC_CR.PLLON` 세트부터 `PLLRDY` 셋까지를
  HSI 기반 DWT_CYCCNT로 카운트.
- **HSI 16 MHz의 실제 오차(§2.3)** — SYSCLK을 HSI로 돌린 뒤 TIM5 입력 캡처로 LSE(32.768 kHz 참값)를
  재면 HSI 오차가 역산된다. LSI/LSE에 쓴 하네스를 그대로 재사용할 수 있다.
- **HSE 주파수 직접 측정** — `MCO1 = HSE`(§4.2)로 내보내 스코프로 확인하거나,
  SYSCLK을 HSI로 고정한 상태에서 TIM 입력 캡처로 HSE/MCO 분주 출력을 캡처.
- **216 MHz 대비 성능/전류(§6.2 각주의 "7.4 % 단축" 추정)** — PLLN을 432로 바꾸고 `FLASH_LATENCY_7`로
  올린 뒤 같은 추론 워크로드의 지연시간을 재측정. **현재는 순수 산술 추정이며 검증되지 않았다.**
- **GPIO 토글 최대 주파수(§10.2)** — `OSPEEDR=11`로 설정한 핀을 BSRR로 토글하는 루프를 ITCM에서 돌리고
  스코프로 파형 확인(또는 다른 핀의 입력 캡처로 계수).
- **I/O 보상 셀(§10.2 각주)** — `SYSCFG_CMPCR.CMP_PD`를 켜고 `READY` 비트가 셋되는지 확인.
  (실측 덤프에서는 SYSCFG 레지스터가 전부 리셋값 0이었다 = 보상 셀 미사용 상태.)
- **온도센서 채널 배정(ADC1_IN18이 VBAT와 공유, §11.1)** — `ADC_CCR.TSVREFE`와 `VBATE`를 **동시에** 켠 뒤
  IN18을 변환해 값이 VBAT 쪽(≈1/4 분압)으로 고정되는지 확인. 프로브 펌웨어는 둘을 따로 켜서 읽었다.
- **BOR/PVD 임계 전압(§9.3), 절대 최대 정격(§9.2)** — 공급 전압을 규격 밖으로 흔들어야 하므로
  이 보드에서는 측정하지 않는다(파괴 위험).

---

## 요약

- **HSE는 크리스탈이 아니라 25 MHz 능동 오실레이터**다. 보드의 X2는 `NZ2520SB-25.00M` 4핀 SMD 오실레이터(VDD/OUT/EN/GND)이며, 출력 네트 `OSC_25M`이 MCU의 `PH0-OSC_IN`과 이더넷 PHY(LAN8742A) `XTAL1/CLKIN`에 동시에 공급된다. [A] UM1907 Rev 1 §1.4 + MB1191 회로도
- **LSE 32.768 kHz는 실장되어 있다.** X3 = `NX3215SA-32.768K` 크리스탈, `PC14-OSC32_IN` / `PC15-OSC32_OUT`에 4.7 pF NPO 커패시터와 함께 연결. [A] MB1191 회로도
- **PLLM=25, PLLN=432, PLLP=2 → 216 MHz가 ST 공식 표준 구성이 맞다.** (VCO_IN = 25 MHz/25 = 1 MHz, VCO_OUT = 432 MHz, SYSCLK = 216 MHz, PLLQ=9, AHB/1, APB1/4=54 MHz, APB2/2=108 MHz, over-drive ON, `FLASH_LATENCY_7`) [B] STM32Cube_FW_F7 STM32746G-Discovery Templates
- **단, 본 프로젝트는 216 MHz가 아니라 200 MHz로 동작한다.** `main.cc`의 `SystemClock_Config()`는 PLLN=400 → SYSCLK 200 MHz, APB1 50 MHz, APB2 100 MHz, `FLASH_LATENCY_6`. [A] 프로젝트 소스
- **216 MHz는 VOS Scale 1 + over-drive ON에서만 가능**하다. over-drive OFF 시 Scale 1의 최대 HCLK는 180 MHz. APB1은 45 → 54 MHz, APB2는 90 → 108 MHz로 함께 확장된다. [A] DS10916 Table 18
- **RUN 216 MHz 전형 전류는 186 mA**(Flash 실행, ART ON/L1-cache ON, 전 주변장치 ON, TA=25 °C). 주변장치를 모두 끄면 108 mA. [A] DS10916 Table 26
- **L1-cache를 끄면 오히려 전류가 약간 줄고(181 mA), ART를 끄면 크게 늘어난다(205 mA).** 캐시/ART의 이득은 전류가 아니라 실행 시간(사이클 수)에서 나온다. [A] DS10916 Table 26/27/28
- 저전력: **Stop 0.45 mA 전형 / Standby 2.3 µA 전형(3.3 V)**. 웨이크업은 Sleep 13 CPU 사이클, Stop 14 µs, Standby 308 µs. [A] DS10916 Table 32/33/37
- 온도센서는 **ADC1_IN18**(VBAT와 공유), VREFINT는 **ADC1_IN17**. 캘리브레이션 값은 `0x1FF0 7A4A`(VREFIN_CAL), `0x1FF0 7A4C`(TS_CAL1@30 °C), `0x1FF0 7A4E`(TS_CAL2@110 °C). [A] DS10916 Table 70/73, RM0385 §15

  > **실측 정정**: 캘리브레이션 주소가 **틀렸다.** 올바른 값은
  > **`0x1FF0F44A`(VREFIN_CAL=1492) / `0x1FF0F44C`(TS_CAL1=942) / `0x1FF0F44E`(TS_CAL2=1195)**이다.
  > `0x1FF07A4A`는 STM32F4 계열의 주소다. 상세는 §11.3의 정정 블록 참조.
- **주의: PLL VCO 출력 하한값이 문서 간 불일치.** RM0385는 "100~432 MHz", DS10916 Table 44는 `fVCO_OUT` min = **192 MHz**. 설계 시 보수적으로 192 MHz 하한을 적용할 것. [A] 두 문서 직접 대조

---

## 1. 사용 문서 버전

| 문서 | 문서번호 | 실제 확인한 개정판 | 비고 |
|---|---|---|---|
| STM32F745xx/F746xx 데이터시트 | DS10916 (DocID027590) | **Rev 1 (2015-05)** | 최신은 Rev 5 (2025-07) — 아래 "미확인" 참조 |
| STM32F75xxx/F74xxx 레퍼런스 매뉴얼 | RM0385 (DocID026670) | **Rev 5 (2016-04)** | 최신은 Rev 9 (2025-02) |
| STM32F746G-DISCO 유저매뉴얼 | UM1907 (DocID027942) | **Rev 1 (2015-06)** | 최신은 Rev 7 (2026-05) |
| 에라타 | ES0290 | **Rev 7 (2019-07)** — 목차만 확보 | 본문 미확보 |
| 보드 회로도 | MB1191 (UM1907 부록) | Rev B-02 | "Change Capacitors value for X3, X4" 주석 존재 |

> 신뢰도 표기: **[A]** ST/ARM 공식 문서 직접 확인 · **[B]** ST 공식 툴/펌웨어/앱노트 · **[C]** 2차 자료 · **[D]** 계산·추론

---

## 2. 클럭 소스

### 2.1 외부 고속 클럭 (HSE)

HSE는 두 가지 모드가 있고, 규격이 서로 다르다.

| 모드 | 파라미터 | Min | Typ | Max | 단위 | 출처 |
|---|---|---|---|---|---|---|
| **Bypass**(외부 클럭 입력, `HSEBYP=1`) | `fHSE_ext` | 1 | – | **50** | MHz | DS10916 Table 38 [A] |
| Bypass | OSC_IN High/Low 최소 폭 `tw(HSE)` | 5 | – | – | ns | 〃 |
| Bypass | OSC_IN 상승/하강 시간 | – | – | 10 | ns | 〃 |
| Bypass | Duty cycle | 45 | – | 55 | % | 〃 |
| Bypass | `VHSEH` / `VHSEL` | 0.7·VDD / VSS | – | VDD / 0.3·VDD | V | 〃 |
| Bypass | OSC_IN 입력 커패시턴스 | – | 5 | – | pF | 〃 |
| **Crystal**(`HSEON=1`) | `fOSC_IN` | **4** | – | **26** | MHz | DS10916 Table 40 [A] |
| Crystal | 피드백 저항 `RF` | – | 200 | – | kΩ | 〃 |
| Crystal | `IDD` (ESR 30 Ω, CL=5 pF @25 MHz) | – | 450 | – | µA | 〃 |
| Crystal | `IDD` (ESR 30 Ω, CL=10 pF @25 MHz) | – | 530 | – | µA | 〃 |
| Crystal | 정확도 `ACCHSE` | −500 | – | +500 | ppm | 〃 |
| Crystal | 임계 gm `Gm_crit_max` | – | – | 1 | mA/V | 〃 |
| Crystal | 기동 시간 `tSU(HSE)` | – | **2** | – | ms | 〃 |

CL1/CL2 권장 범위는 5~25 pF, PCB+핀 커패시턴스는 약 10 pF로 추정해 합산 [A] DS10916 §5.3.9.

### 2.2 외부 저속 클럭 (LSE)

| 모드 | 파라미터 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| Bypass | `fLSE_ext` | – | 32.768 | 1000 | kHz |
| Crystal | `IDD` @ `LSEDRV=00` (Low drive) | – | 250 | – | nA |
| Crystal | `IDD` @ `LSEDRV=10` (Medium low) | – | 300 | – | nA |
| Crystal | `IDD` @ `LSEDRV=01` (Medium high) | – | 370 | – | nA |
| Crystal | `IDD` @ `LSEDRV=11` (High drive) | – | 480 | – | nA |
| Crystal | `Gm_crit_max` (00/10/01/11) | – | – | 0.48 / 0.75 / 1.7 / 2.7 | µA/V |
| Crystal | 기동 시간 `tSU` | – | **2** | – | **초(s)** |

[A] DS10916 Table 39, Table 41. LSE 기동에 초 단위가 걸리는 점에 유의.

### 2.3 내부 RC 오실레이터 (HSI / LSI)

| 심볼 | 파라미터 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `fHSI` | HSI 주파수 | – | **16** | – | MHz |
| `ACCHSI` | 정확도, TA = 25 °C (공장 캘리브레이션) | −1 | – | +1 | % |
| `ACCHSI` | 정확도, TA = −10 ~ 85 °C | −4 | – | +4 | % |
| `ACCHSI` | 정확도, TA = −40 ~ 105 °C | −8 | – | +4.5 | % |
| – | HSI 사용자 트리밍 스텝 (`HSITRIM[4:0]`) | – | – | 1 | % |
| `tsu(HSI)` | HSI 기동 시간 | – | 2.2 | 4 | µs |
| `IDD(HSI)` | HSI 소비 전류 | – | 60 | 80 | µA |
| `fLSI` | LSI 주파수 | **17** | **32** | **47** | kHz |
| `tsu(LSI)` | LSI 기동 시간 | – | 15 | 40 | µs |
| `IDD(LSI)` | LSI 소비 전류 | – | 0.4 | 0.6 | µA |

[A] DS10916 Table 42, Table 43. 공장 캘리브레이션 값은 리셋 시 `RCC_CR`의 `HSICAL[7:0]`에 로드된다 [A] RM0385 §5.2.2.

LSI는 ±47%까지 벌어질 수 있으므로 IWDG 타임아웃 계산 시 최악값(47 kHz / 17 kHz)을 반드시 고려해야 한다 [D] (Min/Max 규격에서 직접 유도).

> **실측 확인**: 이 개체의 LSI는 **26,616 ~ 26,652 Hz**다(TIM5 입력 캡처, 100 MHz 타이머 기준).
> 공칭 32 kHz 대비 **−16.7 %**로 규격(17~47 kHz) 안이지만 편차가 크다.
> 공칭값으로 계산한 IWDG 타임아웃은 이 보드에서 **약 1.2배 길어진다.** 문서의 경고가 실제 사례로 확인됨.
> LSE는 같은 방법으로 **32,743 ~ 32,765 Hz**(공칭 32,768, 오차 −0.01~−0.08 %)로 측정됐다.

---

## 3. PLL 구조

### 3.1 메인 PLL (PLL)

공식 [A] RM0385 §5.3.2:

```
f(VCO clock)              = f(PLL clock input) × (PLLN / PLLM)
f(PLL general clock out)  = f(VCO clock) / PLLP     → SYSCLK 후보
f(USB OTG FS/SDMMC/RNG)   = f(VCO clock) / PLLQ
```

| 필드 | 레지스터 위치 | 허용 범위 | 비고 |
|---|---|---|---|
| `PLLM[5:0]` | `RCC_PLLCFGR[5:0]` | **2 ≤ PLLM ≤ 63** | PLL·PLLI2S·PLLSAI가 **공유**. VCO 입력이 1~2 MHz가 되도록 설정, **2 MHz 권장**(지터 최소화) |
| `PLLN[8:0]` | `RCC_PLLCFGR[14:6]` | **50 ≤ PLLN ≤ 432** | 50~99는 VCO 입력 > 1 MHz일 때만 |
| `PLLP[1:0]` | `RCC_PLLCFGR[17:16]` | **2, 4, 6, 8** | `00`=2, `01`=4, `10`=6, `11`=8 |
| `PLLQ[3:0]` | `RCC_PLLCFGR[27:24]` | **2 ≤ PLLQ ≤ 15** | 0,1은 wrong configuration |
| `PLLSRC` | `RCC_PLLCFGR[22]` | 0=HSI, 1=HSE | PLL·PLLI2S 공통 |

> `RCC_PLLCFGR` 리셋값 = `0x2400 3010` (주소 오프셋 0x04) [A] RM0385 §5.3.2
> **`PLLR`는 메인 PLL에 존재하지 않는다** — STM32F7(RM0385)의 메인 PLL 출력은 P와 Q 두 개뿐이다. R 출력은 PLLI2S/PLLSAI에만 있다. [A] RM0385 §5.3.2 레지스터 비트맵 (bit 15와 21:18은 Reserved)

### 3.2 PLL 주파수 규격

| 심볼 | 파라미터 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `fPLL_IN` | PLL 입력 클럭 (PLLM 나눈 후) | **0.95** | 1 | **2.10** | MHz |
| `fVCO_OUT` | VCO 출력 | **192** | – | **432** | MHz |
| `fPLL_OUT` | PLLP 출력 (시스템 클럭) | 24 | – | **216** | MHz |
| `fPLL48_OUT` | PLLQ 출력 (USB FS 48 MHz) | – | 48 | 75 | MHz |
| `tLOCK` | 락 시간 @ VCO 192 MHz | 75 | – | 200 | µs |
| `tLOCK` | 락 시간 @ VCO 432 MHz | **100** | – | **300** | µs |
| `IDD(PLL)` | VDD 측 소비 @ VCO 432 MHz | 0.45 | – | 0.75 | mA |
| `IDDA(PLL)` | VDDA 측 소비 @ VCO 432 MHz | 0.55 | – | 0.85 | mA |

[A] DS10916 Table 44.

**216 MHz에서의 PLL 지터** [A] DS10916 Table 44:

| 지터 종류 | RMS | Peak-to-peak |
|---|---|---|
| Cycle-to-cycle jitter | 25 ps | ±150 ps |
| Period jitter | 15 ps | ±200 ps |

PLL을 2개 동시 사용하면 지터가 최대 +30% 악화된다 [A] DS10916 Table 44 note 3.

### 3.3 ⚠️ VCO 하한값 문서 불일치

- **RM0385 Rev 5 §5.3.2**: "The software has to set these bits correctly to ensure that the VCO output frequency is **between 100 and 432 MHz**" [A]
- **DS10916 Rev 1 Table 44**: `fVCO_OUT` Min = **192 MHz**, Max = 432 MHz [A]

두 값이 다르다. 전기적 특성표(데이터시트)가 실제 보증값이므로 **192 MHz 하한을 적용**하는 것이 안전하다 [D]. 216 MHz 표준 구성(VCO 432 MHz)은 어느 쪽 기준으로도 안전 범위 안이다.

### 3.4 PLLI2S / PLLSAI

두 보조 PLL은 `PLLM`을 메인 PLL과 **공유**하고, 자체 N/P/Q/R을 갖는다.

| PLL | 필드 | 범위 | 출력 용도 |
|---|---|---|---|
| PLLI2S | `PLLI2SN[8:0]` | 50 ~ 432 | VCO 배수 |
| PLLI2S | `PLLI2SP[1:0]` | 2, 4, 6, 8 | SPDIFRX |
| PLLI2S | `PLLI2SQ[3:0]` | 2 ~ 15 | SAI |
| PLLI2S | `PLLI2SR[2:0]` | 2 ~ 7 | I2S |
| PLLSAI | `PLLSAIN[8:0]` | 50 ~ 432 | VCO 배수 |
| PLLSAI | `PLLSAIP[1:0]` | 2, 4, 6, 8 | 48 MHz (USB/RNG/SDMMC) |
| PLLSAI | `PLLSAIQ[3:0]` | 2 ~ 15 | SAI |
| PLLSAI | `PLLSAIR[2:0]` | 2 ~ 7 | LCD-TFT |

[A] RM0385 §5.3.23 (`RCC_PLLI2SCFGR`, 오프셋 0x84, 리셋값 `0x2400 3000`), §5.3.24 (`RCC_PLLSAICFGR`, 오프셋 0x88, 리셋값 `0x2400 3000`).

주파수 규격 [A] DS10916 Table 45/46: 입력 0.95~2.10 MHz, VCO 192~432 MHz, P/Q/R 출력 모두 최대 216 MHz (PLLSAIP는 48~75 MHz), 락 시간 100~300 µs @432 MHz.

LTDC용 추가 분주기 `PLLSAIDIVR[1:0]`(÷2/4/8/16)이 `RCC_DCKCFGR1`에 존재한다 [A] RM0385 §5.3.25.

---

## 4. 시스템 클럭 트리 · 프리스케일러

### 4.1 최대 주파수 (STM32F746NGH6 = "6" suffix, −40 ~ +85 °C)

| 클럭 | over-drive OFF | over-drive ON | 출처 |
|---|---|---|---|
| SYSCLK / HCLK (`fHCLK`) — VOS Scale 1 | **180 MHz** | **216 MHz** | DS10916 Table 18 [A] |
| SYSCLK / HCLK — VOS Scale 2 | 168 MHz | 180 MHz | 〃 |
| SYSCLK / HCLK — VOS Scale 3 | 144 MHz | (미지원) | 〃 |
| PCLK1 / APB1 (`fPCLK1`) | **45 MHz** | **54 MHz** | 〃 |
| PCLK2 / APB2 (`fPCLK2`) | **90 MHz** | **108 MHz** | 〃 |

> 부품번호 접미사 "6" = −40 ~ +85 °C, 216 MHz. 접미사 "7" = −40 ~ +105 °C, **200 MHz** 상한. STM32F746NG**H6** 이므로 216 MHz 등급이다. [A] DS10916 §7 Ordering information, Table 18 note 2

### 4.2 프리스케일러 범위

| 프리스케일러 | 레지스터 필드 | 분주비 | 비고 |
|---|---|---|---|
| AHB | `RCC_CFGR.HPRE[3:0]` | 1, 2, 4, 8, 16, **64**, 128, 256, 512 | **÷32는 없다** (`1011`=16 다음이 `1100`=64) |
| APB1 (low-speed) | `RCC_CFGR.PPRE1[2:0]` | 1, 2, 4, 8, 16 | `0xx`=1 |
| APB2 (high-speed) | `RCC_CFGR.PPRE2[2:0]` | 1, 2, 4, 8, 16 | `0xx`=1 |
| RTC (HSE 분주) | `RCC_CFGR.RTCPRE[4:0]` | 2 ~ 31 | 결과가 **1 MHz**가 되어야 함. `00000`/`00001` = no clock |
| MCO1 | `RCC_CFGR.MCO1PRE[2:0]` | 1, 2, 3, 4, 5 | `0xx`=분주 없음 |
| MCO2 | `RCC_CFGR.MCO2PRE[2:0]` | 1, 2, 3, 4, 5 | 〃 |

[A] RM0385 §5.3.3 (`RCC_CFGR`, 오프셋 0x08, 리셋값 `0x0000 0000`).

- 프리스케일러 변경은 쓰기 후 **1~16 AHB 사이클** 뒤에 반영된다 [A] RM0385 §5.3.3.
- **이더넷 사용 시 HCLK ≥ 25 MHz** 필수 [A] RM0385 §5.3.3 HPRE Caution.
- MCO1 소스: `00`=HSI, `01`=LSE, `10`=HSE, `11`=PLL. MCO2 소스: `00`=SYSCLK, `01`=PLLI2S, `10`=HSE, `11`=PLL [A] RM0385 §5.3.3.
- 타이머 클럭: `RCC_DCKCFGR1.TIMPRE`=0이면 APBx 프리스케일러가 1일 때 TIMxCLK=PCLKx, 아니면 2×PCLKx. `TIMPRE`=1이면 APBx 프리스케일러 1/2/4에서 TIMxCLK=HCLK, 그 외 4×PCLKx [A] RM0385 §5.2.

### 4.3 Flash 대기 상태 (VDD = 2.7 ~ 3.6 V)

| Wait states (`FLASH_ACR.LATENCY[3:0]`) | HCLK 범위 |
|---|---|
| 0 WS | 0 < HCLK ≤ 30 MHz |
| 1 WS | 30 < HCLK ≤ 60 MHz |
| 2 WS | 60 < HCLK ≤ 90 MHz |
| 3 WS | 90 < HCLK ≤ 120 MHz |
| 4 WS | 120 < HCLK ≤ 150 MHz |
| 5 WS | 150 < HCLK ≤ 180 MHz |
| **6 WS** | **180 < HCLK ≤ 210 MHz** ← 본 프로젝트 200 MHz |
| **7 WS** | **210 < HCLK ≤ 216 MHz** ← ST 표준 216 MHz |

[A] RM0385 Table 4. 리셋 직후는 HCLK 16 MHz / 0 WS. VDD가 낮아지면 표가 달라지며, VDD 2.1~2.4 V에서는 216 MHz에 **9 WS**가 필요하다 [A] RM0385 Table 4.

ART Accelerator(ITCM 인터페이스)와 L1-cache(AXI 인터페이스) 덕분에 위 wait state가 실행 속도에 사실상 영향을 주지 않는다(0-wait 동등 성능) [A] DS10916 Table 19 note 2.

> **실측 정정**: 이 보드에서는 성립하지 않았다.
> ① 실측 `FLASH_ACR = 0x00000006`은 **ARTEN=0, PRFTEN=0** — ART가 꺼진 채로 돌고 있었다.
> 그런데 `VTOR = 0x00200000`(옵션 바이트 `BOOT_ADD0 = 0x80`)이라 코드는 **ITCM 별칭 플래시**에서
> 실행된다. Cortex-M7의 L1 I-cache는 AXIM 접근에만 적용되므로, 이 상태에서는
> **명령어 페치에 ART도 I-cache도 걸리지 않는다**(SCB_CCR.IC=1이어도 마찬가지).
> ② ART를 켜도 0-wait 동등은 아니다. ITCM 경로 플래시 순차 읽기 실측은
> **ART OFF 336.79 MB/s(1.68 B/cyc) → ART ON 492.10 MB/s(2.46 B/cyc), ×1.46**이며,
> 6 WS·256-bit 라인에서 계산되는 0-WS 등가치(약 4.6 B/cyc)의 절반 수준이다.
> ③ 원인은 소프트웨어다: `ImageClassification/Core/Inc/stm32f7xx_hal_conf.h:150`이
> `#define PREFETCH_ENABLE 0U`이고 `ART_ACCLERATOR_ENABLE`가 정의되어 있지 않아
> `HAL_Init()`의 `#if (ART_ACCLERATOR_ENABLE != 0)` 블록이 컴파일되지 않는다.
> → **ART를 켜는 것만으로 플래시 접근이 1.46배 빨라진다.**

### 4.4 VDD 범위별 제약

| VDD 범위 | 무-대기 최대 Flash 주파수 | 최대 HCLK | I/O 보상 셀 | Flash 프로그래밍 폭 |
|---|---|---|---|---|
| 1.7 ~ 2.1 V | 20 MHz | 180 MHz (8 WS, over-drive **OFF**) | 불가 | 8-bit |
| 2.1 ~ 2.4 V | 22 MHz | 216 MHz (9 WS, over-drive ON) | 불가 | 16-bit |
| 2.4 ~ 2.7 V | 24 MHz | 216 MHz (8 WS, over-drive ON) | 동작 | 16-bit |
| **2.7 ~ 3.6 V** | **30 MHz** | **216 MHz (7 WS, over-drive ON)** | 동작 | 32-bit |

[A] DS10916 Table 19. 보드는 3.3 V 동작이므로 맨 아래 행이 적용된다.

---

## 5. STM32F746G-DISCO 보드 실제 클럭 소스

UM1907 §1.4 "Clock sources"는 클럭 소스를 3개로 명시한다 [A]:

| 지정자 | 부품 | 주파수 | 종류 | 공급 대상 |
|---|---|---|---|---|
| **X1** | `NZ2520SB-24.00M` | 24 MHz | **능동 오실레이터**(4핀: VDD/OUT/EN/GND, 3V3) | USB OTG HS PHY(USB3320C-EZK), 카메라 모듈 (`OSC_24M` → `Camera_CLK`) |
| **X2** | `NZ2520SB-25.00M` | **25 MHz** | **능동 오실레이터**(4핀: VDD/OUT/EN/GND, 3V3) | **STM32F746NGH6 `PH0-OSC_IN`** + 이더넷 PHY LAN8742A `XTAL1/CLKIN` (네트명 `OSC_25M`) |
| **X3** | `NX3215SA-32.768K` | **32.768 kHz** | **수동 크리스탈** | STM32F746NGH6 RTC — `PC14-OSC32_IN` / `PC15-OSC32_OUT`, 4.7 pF NPO ±0.25 pF 커패시터, 직렬 R30 = 100 Ω 1% |

[A] UM1907 Rev 1 §1.4 및 부록 MB1191 회로도(IO assignment Table 10: `PH0/OSC_IN → RCC_OSC_IN`, `PH1/OSC_OUT → RCC_OSC_OUT`, `PC14/OSC32_IN → RCC_OSC32_IN`, `PC15/OSC32_OUT → RCC_OSC32_OUT`)

### 5.1 질문에 대한 직접 답변

> **"HSE는 25 MHz 크리스탈인가 오실레이터인가?"**
> → **오실레이터**다. UM1907 본문이 "X2, 25 MHz **oscillator**"로 명시하고, 회로도의 부품이 `NZ2520SB-25.00M`(NDK 2.5×2.0 mm SPXO)이며 EN/GND/OUT/VDD 4핀 심볼로 그려져 있다. 수동 크리스탈이라면 2핀이어야 한다. [A]

> **"LSE 32.768 kHz 유무?"**
> → **있다.** X3 = `NX3215SA-32.768K`, PC14/PC15에 실장. [A]

### 5.2 ⚠️ HSE 모드 관련 주의점

> **실측 확인 (§5.1의 두 답변)**:
> - **LSE 실장** — 확정. `BDCR = 0x00000003`(LSEON=1/LSERDY=1/LSEBYP=0/LSEDRV=`00`)이며,
>   프로브 시작 시점에는 LSEON=0이었다가 켜자 LSERDY가 올라왔다. 실측 주파수 32,743~32,765 Hz.
>   → **32.768 kHz 크리스탈이 물리적으로 실장되어 있음이 실증**됐다.
> - **HSE** — 실측은 `HSEON=1, HSERDY=1, **HSEBYP=0**`이다. 이는 아래 §5.2가 지적한
>   "부품은 능동 오실레이터인데 소프트웨어는 크리스탈 모드"라는 상황을 **확인해 주는** 값이지,
>   보드에 크리스탈이 붙어 있다는 뜻이 **아니다**(`HSEBYP`는 MCU 내부 발진 드라이버 설정일 뿐이다).
>   주파수 자체는 간접 확인됐다 — HSE 기반 PLL로 만든 100 MHz 타이머로 잰 LSE가 참값과 0.1 % 이내로
>   일치하므로 HSE ≈ 25.00 MHz다.
> - **X2가 이더넷 PHY에도 공급된다는 서술** — MDIO 주소 0에서 PHY ID1 `0x0007`/ID2 `0xC131`
>   → **LAN8742A**가 응답했다. PHY가 클럭을 받고 있다는 증거다.

X2가 능동 오실레이터이므로 전기적으로 정확한 설정은 **HSE Bypass**(`RCC_HSE_BYPASS`, `RCC_CR.HSEBYP=1`)다. 그러나 **ST 공식 STM32Cube_FW_F7의 STM32746G-Discovery 예제는 `RCC_OscInitStruct.HSEState = RCC_HSE_ON`(크리스탈 모드)을 사용**하며 [B], 본 프로젝트도 동일하게 `RCC_HSE_ON`을 쓴다 [A]. 실무상 동작하지만 표준 구성은 아니다. 이 불일치의 ST 공식 근거는 확인하지 못했다(→ 미확인 섹션).

보드 전원: 5 V DC 입력, **최대 소비 500 mA**(USB 호스트 급전 시 enumeration에서 500 mA 요청) [A] UM1907 §1.2.

---

## 6. 216 MHz 표준 구성 검증

### 6.1 ST 공식 216 MHz 구성 — **확인됨**

STM32Cube_FW_F7 `Projects/STM32746G-Discovery/Templates/Src/main.c`의 `SystemClock_Config()` [B]:

```c
RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
RCC_OscInitStruct.HSEState       = RCC_HSE_ON;
RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
RCC_OscInitStruct.PLL.PLLM = 25;             /* 25 MHz / 25 = 1 MHz  (VCO 입력) */
RCC_OscInitStruct.PLL.PLLN = 432;            /* 1 MHz × 432 = 432 MHz (VCO 출력) */
RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;  /* 432 / 2 = 216 MHz    (SYSCLK)  */
RCC_OscInitStruct.PLL.PLLQ = 9;              /* 432 / 9 = 48 MHz     (USB FS)  */
HAL_PWREx_EnableOverDrive();                 /* 216 MHz 도달에 필수 */
RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;  /* HCLK  = 216 MHz */
RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;    /* PCLK1 =  54 MHz */
RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;    /* PCLK2 = 108 MHz */
HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7);
```

**결론: PLLM=25 / PLLN=432 / PLLP=2 → 216 MHz는 ST 표준 구성이 맞다.** [B]

검증 사항 [D] (위 수치에서 직접 계산):
- VCO 입력 = 1 MHz — 규격 0.95~2.10 MHz 안이지만 **하한에 가깝다**. RM0385는 지터 최소화를 위해 2 MHz를 권장하므로 이 구성은 권장값이 아닌 최소값 쪽이다.
- VCO 출력 432 MHz = 규격 최대값(432 MHz)에 정확히 일치.
- PCLK1 54 MHz, PCLK2 108 MHz = 각각 over-drive ON 최대값에 정확히 일치. **여유 0**.

### 6.2 본 프로젝트의 실제 구성 — **200 MHz**

`/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Core/Src/main.cc` L390-441 [A]:

| 항목 | 값 | 결과 |
|---|---|---|
| `PWR_REGULATOR_VOLTAGE_SCALE1` | VOS = Scale 1 | — |
| `PLLM` | 25 | VCO 입력 = 1 MHz |
| `PLLN` | **400** | VCO 출력 = **400 MHz** |
| `PLLP` | `RCC_PLLP_DIV2` | **SYSCLK = 200 MHz** |
| `PLLQ` | 9 | 400/9 ≈ 44.44 MHz (48 MHz 아님 — USB FS 부적합) |
| `HAL_PWREx_EnableOverDrive()` | 호출됨 | over-drive ON |
| `AHBCLKDivider` | `RCC_SYSCLK_DIV1` | HCLK = 200 MHz |
| `APB1CLKDivider` | `RCC_HCLK_DIV4` | PCLK1 = **50 MHz** (over-drive 최대 54 MHz 이내 ✓) |
| `APB2CLKDivider` | `RCC_HCLK_DIV2` | PCLK2 = **100 MHz** (over-drive 최대 108 MHz 이내 ✓) |
| Flash latency | `FLASH_LATENCY_6` | 6 WS → 180 < 200 ≤ 210 MHz ✓ |
| 오실레이터 | `HSE` + `LSI` (LSE 미사용) | — |

`ImageClassification.ioc`도 동일하게 `RCC.PLLN=400`, `RCC.HCLKFreq_Value=200000000`이다 [A].

> **실측 확인 (레지스터 레벨)**: 위 표가 실행 중 레지스터 덤프와 완전히 일치한다.
> `RCC_PLLCFGR = 0x29406419`(PLLM=25, PLLN=400, PLLP=2, PLLQ=9, PLLSRC=HSE),
> `RCC_CFGR = 0x0000940A`(SW=SWS=PLL, HPRE=÷1, PPRE1=÷4, PPRE2=÷2),
> `RCC_CR = 0x33038283`(HSI/HSE/PLL/PLLSAI ON), `FLASH_ACR = 0x00000006`(6 WS),
> `PWR_CR1 = 0x0003C100`(Scale 1 + ODEN + ODSWEN), `PWR_CSR1 = 0x00034000`(VOSRDY/ODRDY/ODSWRDY).
> 온디바이스 펌웨어 자가보고도 `SYSCLK = HCLK = 200,000,000 Hz, PCLK1 = 50 MHz, PCLK2 = 100 MHz`다.
> → **§6.2 표 전체 CONFIRMED.**
> 부수적으로 SDRAM 클럭이 HCLK/2 = **100 MHz**(216 MHz였다면 108 MHz)라는 것도 FMC 레지스터로 확인된다.

> **성능 측정 논문에 인용할 때 주의**: 본 프로젝트의 실측 지연시간(4494 / 9641 / 2407 ms 등)은 **216 MHz가 아니라 200 MHz** 기준이다. 216 MHz로 올리면 산술적으로 약 **7.4% 단축**이 기대된다(200→216 MHz, 1 − 200/216 = 7.41%) [D] — 단, 메모리 대기·SDRAM 접근이 지배적인 구간은 그만큼 줄지 않는다.

> **미검증**: 앞 문단의 200 MHz는 실측으로 확정됐지만(레지스터 + 펌웨어 자가보고), **"216 MHz로 올리면 7.4 % 단축"은
> 순수 산술 추정이며 이 보드에서 검증되지 않았다.** 검증하려면 PLLN=432 / `FLASH_LATENCY_7`로 바꾼 뒤
> 같은 워크로드를 재측정해야 한다. 실측된 메모리 특성상 그대로 줄지 않을 근거가 있다 —
> SDRAM 클럭(HCLK/2)과 플래시 wait state는 HCLK에 연동되므로 SDRAM 읽기 60 MB/s·플래시 6→7 WS 구간은
> 이득이 상쇄될 수 있다.

---

## 7. 전압 스케일링(VOS)과 Over-drive

### 7.1 VOS 레벨별 코어 전압과 최대 주파수

| VOS[1:0] (`PWR_CR1[15:14]`) | 모드 | V12 (VCAP) Min/Typ/Max | over-drive OFF 최대 HCLK | over-drive ON 최대 HCLK |
|---|---|---|---|---|
| `01` | Scale 3 | 1.08 / 1.14 / 1.20 V | **144 MHz** | 미지원 |
| `10` | Scale 2 | 1.20 / 1.26 / 1.32 V | **168 MHz** | **180 MHz** |
| `11` | **Scale 1** (리셋값) | **1.26 / 1.32 / 1.40 V** | **180 MHz** | **216 MHz** |
| `00` | Reserved (Scale 3로 동작) | — | — | — |

[A] DS10916 Table 18, RM0385 §4.4.1 `PWR_CR1` VOS 비트 설명.

- `VOS[1:0]`은 **PLL이 OFF일 때만** 변경 가능하고, 새 값은 PLL이 ON 되어야 실제 적용된다. PLL OFF 상태에서는 자동으로 Scale 3이 선택된다. Stop 모드 탈출 후에도 자동으로 Scale 3이 된다. [A] RM0385 §4.4.1

  > **실측 확인 + 실무 경고**: 이 서술의 귀결이 실측으로 확인됐다 —
  > **`PWR_CSR1.VOSRDY`는 SYSCLK이 HSI로 도는 동안(즉 PLL로 전환하기 전에는) 셋되지 않는다.**
  > VOS를 Scale 1로 설정한 뒤 이 플래그를 기다리도록 코드를 짜면 **그 자리에서 영구 정지한다**
  > (PC = `0x08000EE8`에서 멈춘 것을 ST-LINK로 확인). ST HAL도 이 플래그를 기다리지 않는다.
  > 검증된 올바른 순서: PWR 클럭 → VOS 설정(**대기 없음**) → HSE → PLL lock → over-drive
  > → wait state → SYSCLK 전환.
  > 리셋 직후 실측 `PWR_CR1 = 0x0000C000`(VOS=`0b11` Scale 1, ODEN=0, ODSWEN=0)이므로
  > **Scale 1이 리셋값이고 over-drive는 소프트웨어가 켠다**는 위 표의 서술도 확인된다.
- VCAP_1/VCAP_2에는 **2.2 µF (ESR < 2 Ω)** 외부 커패시터가 필요하다. 레귤레이터 바이패스 시에는 100 nF 2개로 대체. [A] DS10916 Table 20
- **over-drive는 VDD = 1.7~2.1 V 구간에서 사용할 수 없다**. 내부 레귤레이터가 OFF일 때도 사용 불가. [A] DS10916 Table 18 note 1/7

### 7.2 Over-drive 진입 절차 (9단계)

[A] RM0385 §4.1.4 "Entering Over-drive mode":

1. 시스템 클럭 소스를 **HSI 또는 HSE**로 선택
2. `RCC_PLLCFGR` 설정 후 `RCC_CR.PLLON` 세트
3. `PWR_CR1.ODEN` = 1 → `PWR_CSR1.ODRDY` 플래그 셋 대기
4. `PWR_CR1.ODSW` = 1 (레귤레이터를 Normal → Over-drive로 전환). **이때 시스템 클럭이 stall되지만 PLL 락 과정은 계속 진행됨**
5. `PWR_CSR1.ODSWRDY` 플래그 셋 대기
6. Flash latency 및 AHB/APB 프리스케일러 설정
7. PLL 락 대기
8. 시스템 클럭을 PLL로 전환
9. System PLL이 아닌 클럭을 쓰는 주변장치(I2S, LCD-TFT, SAI1, USB 48 MHz 등) 활성화

**중요 제약** [A] RM0385 §4.1.4:
- over-drive 스위치 활성화 중에는 **어떤 주변장치 클럭도 켜져 있으면 안 된다**. 주변장치 클럭은 over-drive 활성화 후에 켠다.
- `ODSWEN`(=`ODSW`) 비트를 세트/리셋하려면 **HSI 또는 HSE가 시스템 클럭으로 선택**되어 있어야 한다 [A] RM0385 §4.4.1 bit 17.
- `ODSWEN`은 `ODRDY` 플래그가 셋된 후에만 세트해야 한다.
- **Stop 모드에 진입하면 over-drive와 PLL이 모두 해제된다.** Stop 탈출 후 소프트웨어가 다시 설정해야 한다.

> 레지스터/비트명 표기 혼선: RM0385 Rev 5 본문은 `ODSW`, 레지스터 비트맵(`PWR_CR1` bit 17)은 `ODSWEN`으로 표기한다. 같은 비트다. [A] RM0385 §4.1.4 vs §4.4.1

### 7.3 Over-drive 해제 절차

두 가지 시퀀스가 있다 [A] RM0385 §4.1.4:

- **시퀀스 1**: HSI/HSE로 전환 → 비-System-PLL 주변장치 클럭 끄기 → `ODEN`과 `ODSW`를 **동시에** 리셋 → `PWR_CSR1.ODSWRDY` 클리어 대기
- **시퀀스 2**: HSI/HSE로 전환 → 주변장치 클럭 끄기 → `ODSW`만 리셋(시스템 클럭 stall) → `ODSWRDY` 클리어 대기 → `ODEN` 리셋

시퀀스 2의 3단계에서는 `ODEN`이 셋인 채로 남아 over-drive가 "enabled but not active" 상태가 된다 [A].

### 7.4 Over-drive 스위칭 시간 (Table 24)

| 심볼 | 조건 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `Tod_swen` (활성화) | HSI | – | **45** | – | µs |
| `Tod_swen` | HSE (4 MHz 최대치 / 26 MHz 최소치) | 45 | – | 100 | µs |
| `Tod_swen` | 외부 HSE 50 MHz | – | 40 | – | µs |
| `Tod_swdis` (비활성화) | HSI | – | **20** | – | µs |
| `Tod_swdis` | HSE (4 MHz / 26 MHz) | 20 | – | 80 | µs |
| `Tod_swdis` | 외부 HSE 50 MHz | – | 15 | – | µs |

[A] DS10916 Table 24. 이 시간 동안 시스템 클럭이 stall된다.

---

## 8. 전력 소비

측정 조건 [A] DS10916 §5.3.7: 모든 I/O는 입력·무부하, TA = 25 °C(Typ), 외부 클럭 25 MHz + PLL ON, PCLK1 = HCLK/4, PCLK2 = HCLK/2, CoreMark 상당 부하 코드. **HCLK 216 MHz는 Scale 1 + over-drive ON 조건**.

> **미검증 (§8 전체)**: 이 절의 전류값은 **하나도 실측되지 않았다.** 이번 측정은 전부 온칩 레지스터·타이머
> 기반이라 전류계를 쓰지 않았다. 참고로 이 보드의 실측 조건은 데이터시트 조건과 다르다 —
> PCLK1 = HCLK/4 ✓ / PCLK2 = HCLK/2 ✓는 같지만 HCLK가 **200 MHz**이고 다이 온도가
> **35.7 ~ 39.2 °C**(TA 25 °C가 아님), VDDA는 **3.311 V**다. 200 MHz 행을 쓰되 온도 여유를 감안할 것.
> 측정하려면 보드 5 V 입력에 전류계를 직렬로 넣어야 하며, 그 값은 MCU 단독이 아니라
> LCD·SDRAM·PHY를 포함한 보드 전체 소비다.

### 8.1 RUN 모드 — 216 MHz 및 200 MHz

**Flash 실행, ART ON(prefetch 제외) / L1-cache ON, 레귤레이터 ON** (Table 26) — 가장 일반적인 구성:

| fHCLK | 주변장치 | Typ (25 °C) | Max 25 °C | Max 85 °C | Max 105 °C |
|---|---|---|---|---|---|
| **216 MHz** | 전부 ON | **186 mA** | 213 mA | 234 mA | – |
| **216 MHz** | 전부 OFF | **108 mA** | 124 mA | 146 mA | – |
| **200 MHz** | 전부 ON | **172 mA** | 197 mA | 217 mA | 235 mA |
| **200 MHz** | 전부 OFF | **100 mA** | 115 mA | 135 mA | 154 mA |
| 180 MHz | 전부 ON | 152 mA | 175 mA | 189 mA | 202 mA |
| 168 MHz | 전부 ON | 135 mA | 155 mA | 168 mA | 180 mA |
| 144 MHz | 전부 ON | 104 mA | 119 mA | 130 mA | 140 mA |
| 60 MHz | 전부 ON | 46 mA | 53 mA | 64 mA | 74 mA |
| 25 MHz | 전부 ON | 22 mA | 25 mA | 36 mA | 47 mA |

[A] DS10916 Table 26. ADC를 켜면 ADC당 **+1.73 mA** 추가 [A].

### 8.2 캐시 / ART / 실행 위치별 비교 (216 MHz, TA = 25 °C, Typ)

| 실행 구성 | 주변장치 ON | 주변장치 OFF | 데이터시트 표 |
|---|---|---|---|
| ITCM RAM 실행 | **178 mA** | **102 mA** | Table 25 |
| Flash (ART ON / L1-cache ON) 또는 AXI SRAM (L1-cache ON) | **186 mA** | **108 mA** | Table 26 |
| Flash 또는 AXI SRAM, **L1-cache 비활성** | **181 mA** | **105 mA** | Table 27 |
| Flash on ITCM, **ART 비활성** | **205 mA** | **130 mA** | Table 28 |

[A] DS10916 Table 25~28.

**해석** [D] (위 표에서 직접 유도):
- L1-cache ON → OFF: 186 → 181 mA, 즉 **캐시를 끄면 전류가 5 mA(약 2.7%) 줄어든다**. 캐시 룩업 자체의 동적 전력이 사라지기 때문. 그러나 실행 사이클 수는 크게 늘어나므로 **작업당 에너지는 캐시 ON이 압도적으로 유리**하다.
- ART 비활성: 186 → 205 mA, **+19 mA (약 +10%)**. ART가 없으면 Flash 접근이 반복되어 전류가 늘어난다.
- ITCM RAM 실행이 178 mA로 가장 낮다 (Flash 접근 자체가 없음).
- 즉 **"전류(mA) 관점의 캐시 효과"와 "에너지(mJ/추론) 관점의 캐시 효과"는 방향이 반대**이므로, 전력 논의 시 반드시 실행 시간과 곱해야 한다.

### 8.3 Sleep 모드 (레귤레이터 ON, Table 30)

| fHCLK | 주변장치 | Typ | Max 25 °C | Max 85 °C |
|---|---|---|---|---|
| 216 MHz | 전부 ON | **116 mA** | 137 mA | 159 mA |
| 216 MHz | 전부 OFF | **35 mA** | 46 mA | 71 mA |
| 200 MHz | 전부 ON | 108 mA | 127 mA | 147 mA |
| 200 MHz | 전부 OFF | 32 mA | 43 mA | 66 mA |
| 25 MHz | 전부 OFF | 5 mA | 8 mA | 20 mA |

[A] DS10916 Table 30.

### 8.4 Stop 모드 (Table 32) — 단위 **mA**

| 모드 | 조건 | Typ 25 °C | Max 25 °C | Max 85 °C | Max 105 °C |
|---|---|---|---|---|---|
| `IDD_STOP_NM` (MR, normal) | Flash Stop, 모든 오실레이터 OFF, IWDG 없음 | **0.45 mA** | 2.00 | 14.00 | 22.00 |
| `IDD_STOP_NM` (MR, normal) | Flash Deep power down, 오실레이터 OFF | **0.40 mA** | 2.00 | 14.00 | 22.00 |
| `IDD_STOP_NM` (LPR) | Flash Stop, 오실레이터 OFF, IWDG 없음 | **0.32 mA** | 1.50 | 10.00 | 18.00 |
| `IDD_STOP_NM` (LPR) | Flash Deep power down | **0.27 mA** | 1.50 | 10.00 | 18.00 |
| `IDD_STOP_UDM` (MR, under-drive) | Flash DPD, 오실레이터 OFF | **0.15 mA** | 0.80 | 4.00 | 7.00 |
| `IDD_STOP_UDM` (LPR, under-drive) | Flash DPD, 오실레이터 OFF | **0.10 mA** | 0.70 | 4.00 | 7.00 |

[A] DS10916 Table 32. Stop 모드 최저 전류는 **100 µA (LPR + under-drive + Flash deep power down)**이다.

### 8.5 Standby 모드 (Table 33) — 단위 **µA**

| 조건 | Typ @1.7 V | Typ @2.4 V | **Typ @3.3 V** | Max 25 °C | Max 85 °C | Max 105 °C |
|---|---|---|---|---|---|---|
| Backup SRAM OFF, RTC·LSE OFF | 1.7 | 1.9 | **2.3** | 5 | 15 | 31 |
| Backup SRAM ON, RTC·LSE OFF | 2.4 | 2.6 | 3.0 | 6 | 20 | 40 |
| Backup SRAM OFF, RTC ON, LSE low drive | 2.1 | 2.4 | **2.9** | 6 | 19 | 39 |
| Backup SRAM OFF, RTC ON, LSE high drive | 2.3 | 2.6 | 3.1 | 7 | 20 | 42 |
| Backup SRAM ON, RTC ON, LSE low drive | 2.7 | 3.0 | 3.6 | 8 | 23 | 49 |

[A] DS10916 Table 33. Max 열의 VDD 조건은 3.3 V.

### 8.6 저전력 모드 웨이크업 시간 (Table 37)

측정 기준: 웨이크업 이벤트 발생 → CPU가 첫 명령을 실행할 때까지. Sleep/Stop은 WFE, Standby/Stop/Sleep은 WKUP(PA0) 핀 사용, TA 상온, VDD = 3.3 V [A].

| 심볼 | 조건 | Typ | Max | 단위 |
|---|---|---|---|---|
| `tWUSLEEP` | Sleep 탈출 | **13** | 13 | **CPU 클럭 사이클** |
| `tWUSTOP` | Stop, MR normal | **14** | 14.9 | µs |
| `tWUSTOP` | Stop, MR normal + Flash Deep power down | **104.1** | 107.6 | µs |
| `tWUSTOP` | Stop, LP 레귤레이터 | 21.4 | 24.2 | µs |
| `tWUSTOP` | Stop, LPR + Flash Deep power down | 111.5 | 116.5 | µs |
| `tWUSTOP` | Stop, MR under-drive (Flash DPD) | 107.4 | 113.2 | µs |
| `tWUSTOP` | Stop, LPR under-drive (Flash DPD) | 112.7 | 120 | µs |
| `tWUSTDBY` | Standby 탈출 (rising edge) | **308** | 313 | µs |
| `tWUSTDBY` | Standby 탈출 (falling edge) | 307 | 313 | µs |

[A] DS10916 Table 37.

`tWUSLEEP` 13 사이클을 216 MHz로 환산하면 **약 60.2 ns**, 200 MHz에서는 **65.0 ns**다 [D] (13 / 216e6, 13 / 200e6).

**Flash Deep power down을 쓰면 Stop 웨이크업이 14 µs → 104 µs로 약 7.4배 느려진다**는 점이 트레이드오프의 핵심이다 [D].

### 8.7 오실레이터/PLL 자체 소비 요약

| 블록 | 조건 | 소비 |
|---|---|---|
| HSI | – | 60 µA typ / 80 µA max |
| LSI | – | 0.4 µA typ / 0.6 µA max |
| HSE crystal | ESR 30 Ω, CL 5 pF @25 MHz | 450 µA typ |
| LSE crystal | LSEDRV=00 / 11 | 250 nA / 480 nA typ |
| 메인 PLL | VCO 432 MHz | VDD 0.45~0.75 mA + VDDA 0.55~0.85 mA |
| PLLI2S / PLLSAI (각각) | VCO 432 MHz | VDD 0.45~0.75 mA + VDDA 0.55~0.85 mA |

[A] DS10916 Table 40~46.

---

## 9. 동작 조건 · 최대 정격

### 9.1 일반 동작 조건 (Table 18)

| 심볼 | 파라미터 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `VDD` | 표준 동작 전압 | **1.7**¹ | – | **3.6** | V |
| `VDDA` | 아날로그 전압 (ADC ≤1.2 Msps) | 1.7¹ | – | 2.4 | V |
| `VDDA` | 아날로그 전압 (ADC ≤2.4 Msps) | **2.4** | – | **3.6** | V |
| `VDDUSB` | USB 미사용 | 1.7 | 3.3 | 3.6 | V |
| `VDDUSB` | USB 사용 | **3.0** | – | 3.6 | V |
| `VBAT` | 백업 전압 | **1.65** | – | 3.6 | V |
| `VIN` | RST 및 FT 핀 (2 V ≤ VDD ≤ 3.6 V) | −0.3 | – | **5.5** | V |
| `VIN` | BOOT 핀 | 0 | – | **9** | V |
| `PD` | TFBGA216 전력 소산 (TA=85 °C) | – | – | **690** | mW |
| `TA` | 6-suffix, 최대 전력 소산 | **−40** | – | **+85** | °C |
| `TA` | 6-suffix, 저전력 소산 | −40 | – | +105 | °C |
| `TJ` | 6-suffix 접합 온도 | −40 | – | **+105** | °C |

¹ VDD/VDDA의 1.7 V 하한은 외부 전원 감시 IC를 사용하고 내부 리셋을 OFF했을 때만 유효 [A].

VDD와 VDDA는 같은 소스에서 공급 권장, 파워업/다운 중 최대 300 mV 차이까지 허용 [A] Table 18 note 6.

### 9.2 절대 최대 정격 (Table 15/16/17)

| 심볼 | 항목 | Min | Max | 단위 |
|---|---|---|---|---|
| `VDD–VSS` | 외부 주 전원 (VDDA, VDD, VBAT, VDDUSB 포함) | −0.3 | **4.0** | V |
| `VIN` | FT 핀 입력 | VSS−0.3 | VDD+4.0 | V |
| `VIN` | 그 외 핀 | VSS−0.3 | 4.0 | V |
| `VIN` | BOOT 핀 | VSS | 9.0 | V |
| `\|ΔVDDx\|` | VDD 핀 간 편차 | – | 50 | mV |
| `ΣIVDD` | 전체 VDD_x 유입 전류 | – | **320** | mA |
| `IVDD` | 개별 VDD_x 유입 전류 | – | 100 | mA |
| `IIO` | 임의 I/O 싱크 / 소스 | – | **±25** | mA |
| `ΣIIO` | 전체 I/O 싱크 / 소스 합계 | – | **±120** | mA |
| `ΣIVDDUSB` | VDDUSB 유입 합계 | – | 25 | mA |
| `IINJ(PIN)` | FT/FTf/RST/B 핀 주입 전류 | −5 | +0 | mA |
| `ΣIINJ(PIN)` | 전체 주입 전류 합 | – | ±25 | mA |
| `TSTG` | 보관 온도 | −65 | +150 | °C |
| `TJ` | 최대 접합 온도 | – | **125** | °C |

[A] DS10916 Table 15/16/17.

### 9.3 리셋 · 전원 감시 (Table 23)

| 심볼 | 항목 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `VPOR/PDR` | POR/PDR 임계 (falling) | 1.60 | **1.68** | 1.76 | V |
| `VPOR/PDR` | POR/PDR 임계 (rising) | 1.64 | **1.72** | 1.80 | V |
| `VPDRhyst` | PDR 히스테리시스 | – | 40 | – | mV |
| `VBOR1/2/3` | Brownout 레벨 (falling) | 2.13 / 2.44 / 2.75 | 2.19 / 2.50 / 2.83 | 2.24 / 2.56 / 2.88 | V |
| `VBORhyst` | BOR 히스테리시스 | – | 100 | – | mV |
| `VPVD` | PVD 레벨 `PLS=000` (rising) | 2.09 | 2.14 | 2.19 | V |
| `VPVD` | PVD 레벨 `PLS=111` (rising) | 3.07 | 3.14 | 3.21 | V |
| `VPVDhyst` | PVD 히스테리시스 | – | 100 | – | mV |
| `TRSTTEMPO` | POR 리셋 지연 | 0.5 | **1.5** | 3.0 | ms |
| `IRUSH` | 레귤레이터 파워온 러시 전류 | – | **160** | **250** | mA |
| `ERUSH` | 러시 에너지 (VDD 1.7 V, 105 °C, 171 mA×31 µs) | – | – | 5.4 | µC |

[A] DS10916 Table 23.

전원 상승/하강률 `tVDD` ≥ 20 µs/V [A] DS10916 Table 21.

---

## 10. I/O 특성

### 10.1 출력 전압 / 드라이브 (Table 58)

| 조건 | `VOL` Max | `VOH` Min |
|---|---|---|
| CMOS, `IIO` = ±8 mA, 2.7 V ≤ VDD ≤ 3.6 V | 0.4 V | VDD − 0.4 V |
| TTL, `IIO` = ±8 mA, 2.7 V ≤ VDD ≤ 3.6 V | 0.4 V | 2.4 V |
| `IIO` = ±**20 mA**, 2.7 V ≤ VDD ≤ 3.6 V | **1.3 V** | **VDD − 1.3 V** |
| `IIO` = ±6 mA, 1.8 V ≤ VDD ≤ 3.6 V | 0.4 V | VDD − 0.4 V |
| `IIO` = ±4 mA, 1.7 V ≤ VDD ≤ 3.6 V | 0.4 V | VDD − 0.4 V |
| PC14 (OSC32_OUT), `IIO` = −2 mA | – | VDD − 0.4 V |

[A] DS10916 Table 58. **PC14는 드라이브 능력이 2 mA로 제한**된다는 점에 유의.

### 10.2 GPIO 토글 최대 주파수 (Table 59)

`GPIOx_OSPEEDR`의 `OSPEEDRy[1:0]` 값별 최대 출력 주파수 `fmax(IO)out`:

| OSPEEDR | CL = 50 pF, VDD ≥ 2.7 V | CL = 30 pF, VDD ≥ 2.7 V | CL = 10 pF, VDD ≥ 2.7 V | 상승/하강 시간 (10 pF, ≥2.7 V) |
|---|---|---|---|---|
| `00` (Low) | **4 MHz** | – | **8 MHz** | ≤100 ns (50 pF) |
| `01` (Medium) | **25 MHz** | – | **50 MHz** | ≤6 ns |
| `10` (High) | – (CL=40 pF: **50 MHz**) | – | **100 MHz** | ≤4 ns |
| `11` (Very high) | – | **100 MHz** | **180 MHz** | **≤2.5 ns** |

[A] DS10916 Table 59.

- **최대 GPIO 토글 주파수는 180 MHz** (`OSPEEDR=11`, CL = 10 pF, VDD ≥ 2.7 V) [A].
- **50 MHz를 초과하고 VDD > 2.4 V이면 I/O 보상 셀(compensation cell)을 사용해야 한다** [A] Table 59 note 4. (SYSCFG_CMPCR)
- VDD 1.7 V로 낮추면 `11`에서도 CL=10 pF 기준 72.5 MHz로 떨어진다 [A].
- EXTI가 감지 가능한 외부 신호의 최소 펄스 폭 `tEXTIpw` = **10 ns** [A].

---

## 11. 내장 아날로그 채널: 온도센서 / VREFINT / VBAT

### 11.1 ADC 채널 배정

| 신호 | ADC 채널 | 비고 |
|---|---|---|
| 내부 온도센서 (`VSENSE`) | **ADC1_IN18** | **VBAT와 채널 공유**. 둘 다 활성화하면 **VBAT 변환만 수행**된다 |
| 내부 기준전압 (`VREFINT`) | **ADC1_IN17** | – |
| VBAT (÷4) | **ADC1_IN18** | 내부 50 kΩ 브리지, 비율 Q = 4, 오차 ±1% |

[A] RM0385 §15.10 / §15.11, DS10916 §2.40.

활성화 비트: `ADC_CCR.TSVREFE`(온도센서+VREFINT), `ADC_CCR.VBATE`(VBAT) [A] RM0385 §15.

### 11.2 온도센서 특성 (Table 69)

| 심볼 | 파라미터 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `TL` | 온도 선형성 | – | ±1 | ±2 | °C |
| `Avg_Slope` | 평균 기울기 | – | **2.5** | – | mV/°C |
| `V25` | 25 °C에서의 전압 | – | **0.76** | – | V |
| `tSTART` | 기동 시간 | – | 6 | 10 | µs |
| `TS_temp` | 1 °C 정확도용 ADC 샘플링 시간 | **10** | – | – | µs |

[A] DS10916 Table 69.

### 11.3 캘리브레이션 값 주소

| 심볼 | 내용 | **메모리 주소** | 크기 |
|---|---|---|---|
| `VREFIN_CAL` | VREFINT ADC 원시값 @ TA = 30 °C, VDDA = 3.3 V | **`0x1FF0 7A4A` – `0x1FF0 7A4B`** | 16-bit |
| `TS_CAL1` | 온도센서 ADC 원시값 @ TA = **30 °C**, VDDA = 3.3 V | **`0x1FF0 7A4C` – `0x1FF0 7A4D`** | 16-bit |
| `TS_CAL2` | 온도센서 ADC 원시값 @ TA = **110 °C**, VDDA = 3.3 V | **`0x1FF0 7A4E` – `0x1FF0 7A4F`** | 16-bit |

[A] DS10916 Table 70, Table 73. 세 값 모두 12-bit ADC 우측 정렬 원시값이며 **VDDA = 3.3 V**에서 취득되었다.

> **실측 정정 — 위 표의 주소는 틀렸다. STM32F74x/F75x의 올바른 주소는 다음과 같다.**
>
> | 심볼 | **올바른 주소** | **이 개체의 실측값** |
> |---|---|---|
> | `VREFIN_CAL` | **`0x1FF0F44A`** | **1492** (`0x05D4`) |
> | `TS_CAL1` (30 °C) | **`0x1FF0F44C`** | **942** (`0x03AE`) |
> | `TS_CAL2` (110 °C) | **`0x1FF0F44E`** | **1195** (`0x04AB`) |
>
> ST-LINK 덤프(`regions_*/factory_uid_area.bin`)와 온디바이스 판독이 같은 값을 돌려주며,
> 이 값으로 §11.4의 식을 돌리면 VDDA 3.311~3.313 V, 다이 온도 35.7~39.2 °C라는 타당한 결과가 나온다.
> **`0x1FF07A4A`는 STM32F4 계열(F42x/F43x 등)의 캘리브레이션 주소**이고, STM32F7에서 그 주소는
> 시스템 메모리 부트로더 코드 영역(`0x1FF00000`~) 안이라 아무 의미 없는 바이트가 읽힌다.
> 같은 조사 산출물인 `info/docs/03_memory_map.md` §7은 `0x1FF0F44A`로 올바르게 적고 있다
> (CMSIS `stm32f746xx.h`의 팩토리 영역 배치와도 그쪽이 일치한다).
>
> 참고로 이 개체의 팩토리 값으로 §11.2의 규격을 역산하면
> 기울기 = (1195−942)/80 °C × 3.3 V/4095 = **2.55 mV/°C**(Avg_Slope typ 2.5와 일치),
> 30 °C 전압 = 942 × 3.3/4095 = 0.759 V → 25 °C 환산 **0.746 V**(V25 typ 0.76과 1.8 % 차)다.

### 11.4 계산식

**(a) 비캘리브레이션 방식** (데이터시트 원문) [A] RM0385 §15.10:

```
Temperature (°C) = { (VSENSE − V25) / Avg_Slope } + 25
                   V25 = 0.76 V, Avg_Slope = 2.5 mV/°C
```
칩마다 오프셋 편차가 커서 **절대 온도 측정에는 부적합**하고 변화량 감지용으로 권장된다 [A] DS10916 §2.40.

**(b) 2점 캘리브레이션 방식** (권장) [D] — TS_CAL1(30 °C)·TS_CAL2(110 °C)에 선형 보간을 적용해 유도:

```c
/* TS_DATA: ADC1_IN18 12-bit 원시값 (VDDA = 3.3 V 조건) */
Temperature_C = ( (110.0f - 30.0f)
                  / (float)(*TS_CAL2 - *TS_CAL1) )
                * (float)(TS_DATA - *TS_CAL1) + 30.0f;
```

**(c) VDDA 보정** [D] — VREFINT가 절대 전압이므로 실제 VDDA를 역산 가능:

```c
VDDA_mV = 3300.0f * (float)(*VREFIN_CAL) / (float)VREFINT_DATA;
```
VDDA가 3.3 V가 아니면 `TS_DATA`를 `TS_DATA × VDDA_mV / 3300`으로 스케일한 뒤 (b)에 대입해야 한다 [D].

> (b)와 (c)는 데이터시트가 제공한 캘리브레이션 포인트(30 °C / 110 °C, VDDA=3.3 V)로부터의 선형 유도이며, 수식 자체가 DS10916 Rev 1 본문에 그대로 실려 있지는 않다. 따라서 신뢰도 [D]로 표기한다.

> **실측 확인**: 프로브 펌웨어가 **(b)+(c)를 결합한 동일한 식**
> `T = 80/(TS_CAL2−TS_CAL1) × (TS_DATA × VDDA/3.3 − TS_CAL1) + 30`
> 으로 다이 온도 **35.69 °C / 39.16 °C**(측정 회차별), VDDA **3.311 / 3.313 V**를 산출했다.
> 값이 상온 + 자기발열로 설명되는 범위이므로 **식과 캘리브레이션 값 모두 정상 동작**한다.
> 단, 반드시 §11.3 정정 블록의 주소(`0x1FF0F44A/44C/44E`)를 써야 한다.

### 11.5 VREFINT / VBAT 특성

| 심볼 | 파라미터 | Min | Typ | Max | 단위 |
|---|---|---|---|---|---|
| `VREFINT` | 내부 기준전압, −40 °C < TA < +105 °C | **1.18** | **1.21** | **1.24** | V |
| `VRERINT_s` | 온도 범위 전체 산포 (VDD = 3 V ±10 mV) | – | 3 | 5 | mV |
| `TCoeff` | 온도계수 | – | 30 | 50 | ppm/°C |
| `tSTART` | 기동 시간 | – | 6 | 10 | µs |
| `TS_vrefint` | ADC 샘플링 시간 | **10** | – | – | µs |
| `R` (VBAT) | VBAT 저항 브리지 | – | **50** | – | kΩ |
| `Q` (VBAT) | VBAT 분압비 | – | **4** | – | – |
| `Er` (VBAT) | Q 오차 | −1 | – | +1 | % |
| `TS_vbat` | VBAT 1 mV 정확도 샘플링 시간 | **5** | – | – | µs |

[A] DS10916 Table 71, Table 72.

> 보드 주의: STM32F746G-DISCO는 **VBAT 핀(C1)이 VDD에 직결**되어 있다 (UM1907 Table 10: "C1 VBAT — Connected to VDD"). 별도 백업 배터리가 없다. [A]

> **실측 확인**: 내부 ADC로 잰 값이 이 서술과 정합한다 —
> **VBAT = 3.304 ~ 3.306 V, VDDA = 3.311 ~ 3.313 V**(VREFINT 코드 1486~1487, `VREFIN_CAL` 1492 기준 환산).
> 두 값이 사실상 같다 → 백업 배터리 없이 VDD에 묶여 있음이 확인된다.
> VREFINT 실전압은 약 **1.202 V**로 규격 1.18~1.24 V 안이다.

---

## 12. 에라타 (ES0290)

**적용 대상**: ES0290 Rev 7 (2019-07) Table 1에 `STM32F746NG` 포함. Table 2 실리콘 리비전: 디바이스 마킹 **"Z" 및 "1"** → **`REV_ID = 0x1001`** [A] ES0290 Rev 7 §Applicability.

→ 실측된 REV_ID `0x1001` = **Revision Z**이며 [A] RM0385 §40.6.1 `DBGMCU_IDCODE`, DEV_ID `0x449` [A] 동일 섹션. 사용자 실측값과 완전히 일치한다.

**전원/클럭 관련 항목 (목차 기준)** [A] ES0290 Rev 7 목차:

| 절 | 제목 | 페이지 |
|---|---|---|
| 2.1.1 | Cortex-M7 data corruption when using data cache configured in write-through | 7 |
| 2.2.3 | **Wakeup from Standby mode when the back-up SRAM regulator is enabled** | 8 |
| 2.7.1 | **MCU may remain stuck in LPTIM interrupt when entering Stop mode** | 14 |

2.2.3의 내용은 동일 계열 STM32F7 에라타(ES0334) 기준으로 다음과 같다 [C] (ES0290 본문 미확보, 2차 자료 기반):
> `PWR_CSR1` 레지스터에 백업 SRAM 레귤레이터를 켜거나 끄기 위해 쓰기를 수행할 때 `EIWUP` 비트가 0으로 덮어써지면, RTC 웨이크업 이벤트(alarm / Tamper / TimeStamp / wakeup timer)로 Standby에서 깨어나지 못한다.
> **워크어라운드**: `PWR_CSR1`에 쓸 때마다 `EIWKUP` 비트를 반드시 1로 유지할 것.

> ⚠️ ES0290 Rev 7의 **목차(1~3페이지)만 확보**했고 본문(4~28페이지)은 읽지 못했다. 목차에는 RCC/PWR 전용 절이 보이지 않지만, 목차가 2.8.2(15페이지)까지만 확보되어 **16~27페이지에 추가 절이 존재할 가능성**이 있다. RCC/PWR 에라타 부재를 단정할 수 없다.

---

## 출처

읽은 문서 및 URL 전체:

1. **DS10916** (DocID027590 Rev 1, 2015-05) — *STM32F745xx / STM32F746xx Datasheet*
   미러: https://www.waveshare.com/w/upload/9/9b/STM32F745-STM32F746-Datasheet.pdf
   원본: https://www.st.com/resource/en/datasheet/stm32f746ng.pdf (최신 Rev 5, 2025-07)
   확인 섹션: §2.40, §5.2 (Table 15/16), §5.3.1 (Table 18/19), §5.3.2 (Table 20), §5.3.3–5.3.5 (Table 21/22/23), §5.3.6 (Table 24), §5.3.7 (Table 25–33), §5.3.8 (Table 37), §5.3.9 (Table 38–41), §5.3.10 (Table 42/43), §5.3.11 (Table 44–46), §5.3.16 (Table 58/59), §5.3.22–5.3.24 (Table 69–73), §7 Ordering information

2. **RM0385** (DocID026670 Rev 5, 2016-04) — *STM32F75xxx / STM32F74xxx Reference Manual*
   미러: https://indico.kit.edu/event/460/contributions/5488/attachments/2862/4152/reference-manual.pdf
   원본: https://www.st.com/resource/en/reference_manual/rm0385-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf (최신 Rev 9, 2025-02)
   확인 섹션: §3.3.2 (Table 4 wait states), §4.1.4 (voltage regulator / over-drive), §4.4.1 (`PWR_CR1`), §5.2 (Clocks, Figure 12 clock tree), §5.2.2 (HSI), §5.2.5 (LSI), §5.3.2 (`RCC_PLLCFGR`), §5.3.3 (`RCC_CFGR`), §5.3.23 (`RCC_PLLI2SCFGR`), §5.3.24 (`RCC_PLLSAICFGR`), §5.3.25 (`RCC_DCKCFGR1`), §15.10–15.11 (temp sensor / VBAT), §40.6.1 (`DBGMCU_IDCODE`)

3. **UM1907** (DocID027942 Rev 1, 2015-06) — *Discovery kit for STM32F7 Series with STM32F746NG MCU*
   https://www.st.com/resource/en/user_manual/um1907-discovery-kit-for-stm32f7-series-with-stm32f746ng-mcu-stmicroelectronics.pdf (최신 Rev 7, 2026-05)
   확인 섹션: §1.2 Power supply, §1.4 Clock sources, §1.10 Ethernet, Table 10 IO assignment, 부록 **MB1191 회로도** (Rev B-02)

4. **ES0290** (Rev 7, 2019-07) — *STM32F74xxx and STM32F75xxx device limitations*
   원본: https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf
   확보한 텍스트(목차만): https://www.stmcu.com.cn/upload/pdf_html/b4e2f0c4f7037c59583aa78c13e9c442.html

5. **STM32Cube_FW_F7** — `Projects/STM32746G-Discovery/Templates/Src/main.c` (216 MHz `SystemClock_Config` 원본)
   https://raw.githubusercontent.com/pellepl/verisure1512/master/ext-libs/STM32Cube_FW_F7_V1.2.0/Projects/STM32746G-Discovery/Templates/Src/main.c

6. **프로젝트 소스 (실측 대상 구성)**
   - `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Core/Src/main.cc` (L390–441 `SystemClock_Config`)
   - `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/ImageClassification.ioc` (RCC 섹션)

7. 참고로 검색은 했으나 본문을 읽지 못한 문서: AN4749 (STM32F7 저전력 관리), AN4839 (L1 cache), AN2867 (오실레이터 설계), AN4661 (F7 하드웨어 개발), ARM DDI 0489 (Cortex-M7 TRM)

---

## 미확인 / 추가 조사 필요

1. **데이터시트 최신 개정판(DS10916 Rev 5, 2025-07) 대조 미완료.** 본 문서는 **Rev 1(2015-05)** 기준이다. `st.com`으로의 직접 다운로드가 이 환경에서 차단되어 Waveshare 미러(Rev 1)를 사용했다. Rev 1 → Rev 5 사이에 전류 소비표(Table 25–33), 웨이크업 시간(Table 37), PLL 규격(Table 44) 수치가 갱신되었을 가능성이 있다. **논문 인용 전 Rev 5로 재확인 필요.**

2. **RM0385 최신판(Rev 9, 2025-02) 대조 미완료.** 본 문서는 Rev 5(2016-04) 기준. Rev 5 자체의 개정 이력에 "Updated Section 5.3.2: RCC_PLLCFGR PLLN[8:0]" 항목이 있어, PLLN 관련 서술이 이후에도 수정되었을 수 있다.

3. **PLL VCO 출력 하한 불일치 미해소.** RM0385 "100 MHz" vs DS10916 Table 44 `fVCO_OUT` min "192 MHz". 어느 쪽이 오타/구버전인지 ST 공식 확인 필요. (현재는 보수적으로 192 MHz 권고)

4. **ES0290 본문(4–28페이지) 미확보.** 확보한 목차가 2.8.2(15페이지)까지뿐이라 16–27페이지의 절 목록을 모른다. RCC/PWR 전용 에라타 존재 여부를 단정할 수 없다. 2.2.3의 상세 내용도 동일 계열 문서 기반 [C] 추정이며 ES0290 원문 확인 필요.

5. **X2 주변 수동 소자 값 미확인.** MB1191 회로도 텍스트 추출 시 두 열이 뒤섞여, X2(25 MHz 오실레이터)의 EN 풀업 저항과 OUT 직렬 저항의 지정자/값을 확정하지 못했다. X1은 EN에 10 kΩ 풀업 + OUT에 100 Ω 직렬(R3/R4)로 읽혔으므로 X2도 동일 토폴로지로 추정되나 [D], 회로도 PDF 원본에서 재확인 필요.

6. **`PH1/OSC_OUT`의 실제 결선 미확인.** UM1907 Table 10은 `PH1/OSC_OUT → RCC_OSC_OUT`으로 표기하지만, X2가 능동 오실레이터이므로 OSC_OUT이 실제로 어디에도 연결되지 않았을 가능성이 높다. 회로도 원본 확인 필요.

7. **ST 공식 예제가 `RCC_HSE_BYPASS`가 아닌 `RCC_HSE_ON`을 쓰는 이유 미확인.** 보드에 능동 오실레이터가 실장되어 있음에도 크리스탈 모드로 설정하는 것이 ST의 의도인지, 관성인지, 별도 근거 문서가 있는지 확인하지 못했다. (동작에는 문제가 없으나 표준 구성은 아님)

8. **캐시 ON/OFF의 "에너지" 비교값 미확보.** 데이터시트는 전류(mA)만 제공하고 실행 시간은 제공하지 않는다. mJ/추론 비교는 본 프로젝트의 실측 지연시간과 곱해 직접 산출해야 한다.
   > **부분 해소 (2026-07-27)**: **시간 쪽은 실측됐다.** D-cache는 SRAM1 순차 읽기를 ×1.42(4 KB 이하),
   > AXIM 플래시 읽기를 ×1.91 빠르게 하고, ART는 ITCM 경로 플래시를 ×1.46 빠르게 한다.
   > D-cache 히트 레이턴시는 2 사이클, 미스 시 SRAM1 8~9 사이클이다.
   > **여전히 없는 것은 전류(mA)** 뿐이므로, 전류계만 붙이면 mJ/추론이 바로 나온다.

9. **보드 전체(MCU 외 LCD/SDRAM/PHY 포함) 소비 전류 미확인.** UM1907은 "최대 500 mA"라는 상한만 제시하며, 실제 전형 소비값은 제공하지 않는다. 실측 필요.

10. **AN4749 / AN4839 / AN4031 / AN4667 본문 미확보.** `st.com` 접근 제약으로 저전력 관리·L1 캐시·DMA 앱노트의 상세 수치를 반영하지 못했다.
