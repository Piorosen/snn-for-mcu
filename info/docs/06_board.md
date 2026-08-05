# STM32F746G-DISCO (MB1191) 보드 구성 조사

조사 대상: STM32F746G-DISCO 디스커버리 보드 / 실장 MCU STM32F746NGH6 (Cortex-M7 r0p1, 실측 Device ID 0x449, Revision ID = Rev Z)
조사일: 2026-07-27

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거: `info/raw/measured_facts.md`, `info/results/probe_output.txt`, `info/raw/decoded_hotplug.md`, `info/raw/host_side.md`.

### 실측으로 확인됨

**보드/칩 정체 — 문서가 제시한 판정 절차가 그대로 통했다**

- §2.3의 1번 방법(플래시 크기 레지스터) → 실측 `0x1FF0F442` = **`0x0400` = 1024 KB** → **STM32F746NG 확정**. CubeProgrammer 배너는 예고대로 `Board : STM32F7508-DK`를 표시했고, 문서의 "그 필드를 믿지 말라"는 지침이 정확했다.
- DBGMCU_IDCODE → 실측 `0x10016449` → DEV_ID **0x449**, REV_ID **0x1001 = Rev Z**. 문서의 "DEV_ID로는 F746/F750을 구분할 수 없다"도 그대로 확인됐다.
- 코어 → 실측 CPUID `0x410FC271` = **Cortex-M7 r0p1**.
- 96비트 UID → `0x1FF0F420`에서 정상 판독(`0x00340033_54435008_20353036`).
- 호스트 툴(st-flash)도 독립적으로 `flash: 1048576 / sram: 327680 / chipid: 0x449 / dev-type: STM32F74x_F75x`를 보고.

**§3 메모리 맵 — 전 항목 실측 일치**

| 영역 | 문서 | 실측 |
|---|---|---|
| ITCM-RAM 0x00000000 | 16 KB | **16 KB** (`0x00004000`부터 버스 폴트, 접근 5.16 cyc) |
| Flash(ITCM alias) 0x00200000 | 1 MB | **1,024 KB**, 선두 64 B가 0x08000000과 완전 일치(같은 물리 플래시) |
| Flash(AXIM) 0x08000000 | 1 MB | **1,024 KB** |
| DTCM 0x20000000 | 64 KB | **64 KB** (0x20000000~0x2000FFFF가 5.16 cyc, 이후 15.66 cyc로 계단) |
| SRAM1 0x20010000 + SRAM2 0x2004C000 | 240 + 16 KB | **합계 256 KB** (0x20050000부터 폴트). 둘은 지연이 같아 성능상 구분되지 않는다 |
| Backup SRAM | 4 KB | **4 KB** (0x40024000) |
| QSPI 0x90000000 | 16 MB | **16,384 KB** |
| FMC SDRAM Bank1 0xC0000000 | 8 MB | **8,192 KB** |

→ 내부 RAM 합계 320 KB + ITCM 16 KB + BKPSRAM 4 KB = **340 KB**, 문서 서술과 일치.

**§4 LCD / LTDC — 레지스터 실측과 완전 일치**

- `RK043FN48H_*` 타이밍 **HSYNC 41 / HBP 13 / HFP 32 / VSYNC 10 / VBP 2 / VFP 2** → LTDC SSCR/BPCR/AWCR/TWCR 역산값이 **정확히 동일**.
- 활성 해상도 **480 × 272**, 전체 타이밍 **566 × 286** → 실측 동일.
- 픽셀 클럭 **9.6 MHz**(PLLSAIN 192 / R 5 / DIVR 4) → 실측 PLLSAICFGR=`0x54003000`, DCKCFGR1=`0x00010000` → **9.60 MHz**.
- 프레임레이트 **≈59.3 Hz** → 실측 9.60 MHz / 161,876 = **59.30 Hz**.
- **PG12만 AF9**라는 지적 → 실측: 보드 전체에서 AF9로 설정된 핀은 **PG12 단 1개**, AF14는 **정확히 27개**(문서 §4.4 표의 AF14 행 수와 일치).
- LCD_DISP = **PI12**, LCD_BL_CTRL = **PK3** → 실측 둘 다 **출력 push-pull, High**(디스플레이·백라이트 켜짐).

**§5 터치 / §6 오디오 — I2C 스캔으로 부품 확정**

- FT5336, **I2C3 (PH7/PH8, AF4), 8비트 주소 0x70 / 7비트 0x38** → 실측 ACK 응답. 칩 ID 레지스터 `0xA8` = **0x51**(= `FT5336_ID_VALUE`) → **FT5336 확정**. 펌웨어 버전 0x12, 패널 ID 0x6F.
- WM8994, **8비트 0x34 / 7비트 0x1A** → 실측 ACK 응답. 레지스터 `0x0000` = **0x8994** → **WM8994 확정**.
- **두 칩이 같은 I2C3 버스를 공유**한다는 서술 → 같은 스캔에서 둘만 응답해 확인.
- 카메라 모듈(0x60), EEPROM(0xA0/0xA6)은 **무응답** → 문서가 "미장착"이라 한 것과 일치.

**§7 이더넷 — 문서가 [C]로 유보한 항목이 [A]가 됐다**

- PHY 주소 **0x00** (문서: "CubeMX 기본값 1은 동작하지 않음" [C]) → 실측 MDIO로 0~31을 훑어 **주소 0에서만 응답**. 확정.
- PHY = **LAN8742A** → 실측 ID1=`0x0007`, ID2=`0xC131` → OUI `0x0001F0`(SMSC/Microchip), 모델 `0x13`, 리비전 `0x1`. 확정.
- MDIO=PA2, MDC=PC1(AF11) 배선 → 이 핀 조합으로 실제 통신 성공.
- BMSR=`0x7809` → 링크 down (케이블 미연결 상태와 일치).

**§9 외부 메모리**

- SDRAM 8 MB / Bank1 0xC0000000 / 16비트 버스 / CL2 / 4뱅크 / 12행·8열 → FMC_SDCR1=`0x00001954` 실측 일치.
- SDCLK 핀 **PG8**, SDNE0 **PH3**, SDNWE **PH5**, SDCKE0 **PC3** → 실측 전부 **AF12 / Very high speed**.
- QSPI FSIZE 필드 **23**, 페이지 256 B, CSHT 6클럭, SampleShifting HALFCYCLE → 실측 `QUADSPI_DCR=0x00170500`, `CR=0x01400311` 일치.
- microSD 검출 **PC13 풀업 입력, 삽입 시 Low** → 실측 PC13 = 풀업 입력, IDR **0** → **카드가 실제로 꽂혀 있다**.
- SDMMC1 배선 CK=PC12 / CMD=PD2 / D0–D3=PC8·PC9·PC10·PC11 → 실측 이 6핀이 전부 **AF12(SDMMC1)** 로 설정되어 있었고 SDMMC1 클럭도 켜져 있었다. 보드 배선의 직접 증거다.

**§10 LED·버튼**

- **LD1 = PI1** → 프로브 펌웨어가 이 핀을 진행 표시용으로 구동해 동작 확인.
- **B1 = PI11** → 실측 IDR = 0 (눌리지 않음). 문서의 "외부 풀다운 + 눌림 = High"와 정합.

**§11 ST-LINK / VCP**

- **ST-LINK/V2-1** → 실측 USB VID `0x0483` / PID **`0x374B`**(= ST-LINK/V2-1), Serial `0668FF39314D423143034442`.
- 펌웨어 버전 문자열 형태 `V2Jxx Myy` → 실측 **V2J46M33**(CubeProgrammer) / `V2J46S33`(st-flash).
- **VCP 존재** → 호스트에 `/dev/cu.usbmodem83303`로 열거됨.
- **USB Mass Storage 존재** → 호스트가 3.2 MB 디스크로 인식.

**§12 전원**

- ST-LINK가 열거 시 **500 mA 요청** → 실측 USB 디스크립터 `Current Required (mA): 500`.
- 보드 3.3 V 레일 → 실측 **VDDA 3.311~3.313 V**(VREFINT 환산), ST-LINK 보고 타깃 전압 3.23 V.

**§13 클럭 소스**

- **X3 = 32.768 kHz 크리스털(LSE)** → 실측 LSEON을 켜자 LSERDY=1, **실측 주파수 32,765 Hz**(공칭 대비 −0.009%). **크리스털이 물리적으로 실장되어 있음이 확정**됐다.
- **X2 = 25 MHz → MCU HSE** → 실측 HSEON=1 / HSERDY=1 / **HSEBYP=0 = 크리스털 발진 모드**, PLLM=25로 1 MHz 기준을 만드는 것도 확인.
- 내장 LSI는 실측 **26,652 Hz**(공칭 32 kHz 대비 −16.7%) — 문서에 없던 값이며, IWDG 타임아웃 계산 시 주의가 필요하다.

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

**1. SYSCLK 216 MHz 전제 (§13, §9.1, §9.2)**

- 문서 주장: "HSE 25 MHz + PLLM=25 → 1 MHz 기준, **216 MHz SYSCLK**", SDCLK = **216/2 = 108 MHz**, QSPI 클럭 = **216/(1+1) = 108 MHz**, 리프레시 간격 **≈14.4 µs**.
- 실측: `RCC_PLLCFGR = 0x29406419` → PLLM=25, **PLLN=400**, PLLP=2 → VCO 400 MHz → **SYSCLK = HCLK = 200 MHz**. PCLK1 = 50 MHz, PCLK2 = 100 MHz. 따라서 **SDCLK = QSPI_CK = 100.00 MHz**, 리프레시 간격 = (1539+1) × 10 ns = **15.40 µs**.
- 원인: **문서가 데이터시트 최대치(216 MHz)를 이 프로젝트의 실제 설정으로 착각**했다. 소스에서도 교차 확인된다(`main.cc:413` → `PLLN = 400`), 온디바이스 펌웨어도 `sysclk_hz = 200000000`을 보고했다.
- 파급: SDRAM·QSPI 타이밍 마진 논의가 전부 100 MHz 기준으로 바뀐다(오히려 여유가 늘어난다).

**2. QSPI 실장 부품이 확정됐다 — 이 보드는 C-03 / `$CU4` 리비전이다 (§1.2, §9.2)**

- 문서 상태: §1.2에서 "CU2 = N25Q128A / CU4 = W25Q128JVEIQ" 두 가능성을 병기하고, "PCB 실크 `DK32F746G$CUx`로 확인하라"고 유보했다. §9.2 표도 두 행을 나란히 뒀다.
- 실측: Read JEDEC ID(0x9F) 응답 **`0xEF4018`** = 제조사 0xEF(**Winbond**), 타입 0x40, 용량 0x18(16 MB). Micron N25Q라면 `0x20BA18`이어야 한다.
- 결론: **이 보드는 MB1191-F746NGH6-C-03(`$CU4`)** 이고 QSPI는 **Winbond W25Q128JV 계열**이다. 문서 오류가 아니라 **문서가 열어 둔 분기가 실측으로 닫힌 것**이며, §1.2의 리비전 표가 정확했다는 증거이기도 하다.
- 파급: `05_external_memory.md` §2 전체가 N25Q128A 데이터시트에 기반하므로 **그 절의 부품 고유 수치는 이 보드에 적용되지 않는다**. 또한 §1.2가 "CU4는 LCD 드라이버도 변경됨"이라 적었는데, **실측 LTDC 타이밍은 RK043FN48H 값과 완전히 동일**했다 → 패널이 바뀌었더라도 LTDC 타이밍은 같다.

**3. 프레임버퍼를 RGB565로 상정한 것 (§요약, §4.3, §16-6)**

- 문서 주장: "RGB565 프레임버퍼 1장 = 261,120 B (255.0 KiB)", 실무 체크리스트도 "RGB565 255 KiB/장".
- 실측: 이 보드에서 돌던 펌웨어는 **ARGB8888(32bpp) 더블 버퍼**다. 레이어1 = `0xC0000000`, 레이어2 = `0xC007F800`, 피치 1,920 B × 272 라인 = **522,240 B/장**, 두 장 **1,044,480 B(≈1.0 MB)**. 레이어 PFCR = 0 → ARGB8888.
- 원인: **문서 오류가 아니라 대표값 선택의 문제**다. §4.3의 포맷별 표 자체는 정확하며 ARGB8888 행(522,240 B / 1,044,480 B)이 실측과 바이트 단위로 일치한다. 다만 요약·체크리스트가 실제와 다른 행을 대표로 골랐다.

**4. 패키지 데이터 레지스터 주소가 틀렸다 (§2.3-5, §2.4)**

- 문서 주장: 패키지 데이터 레지스터 = **`0x1FFF7BF0`**, PKG[2:0] = bits[10:8].
- 실측/정정: ST가 배포하는 CMSIS 헤더 `stm32f746xx.h:1230`은 **`PACKAGE_BASE 0x1FF0F7E0`** 으로 정의한다. `0x1FFF7BF0`은 **STM32F4 계열의 주소**다.
  - `0x1FF0F7E0` 실측값 = **`0x038DFC72`** → PKG[2:0] = `0b100` → 문서의 판정 규칙대로 **LQFP208/TFBGA216**. TFBGA216이라는 §1.1 서술과 정합한다.
  - `0x1FF07BF0` 실측값 = `0xFFFFFFFF` (미프로그램).
- 원인: **문서 오류(계열 혼동)**. 동시에 **실측 쪽도 틀렸다** — 프로브 펌웨어와 ST-LINK 덤프가 `0x1FF07BF0`(문서·CMSIS 어느 쪽도 아닌 제3의 주소)을 읽고 "패키지 판별 불가"로 결론지었다. **양쪽 다 정정이 필요하며, 올바른 주소는 CMSIS 기준 `0x1FF0F7E0`이다.**

**5. CN2 pin4 RESET 신호 (§14.1 각주, 미확인 #5) — 실측이 PG3 쪽을 지지한다**

- 문서 상태: UM1907 Table 1은 `RESET(PC10)`, Table 10은 `PG3 = EXT_RST` → 문서 내 불일치로 남겨 뒀다.
- 실측: **PC10은 SDMMC1_D2로 AF12 설정**되어 있었다(microSD 데이터선). PG3은 입력(플로팅, 풀 없음).
- 판단: PC10이 SD카드 데이터선에 물려 있으므로 **Table 1의 `RESET(PC10)`은 오기일 가능성이 높고 Table 10의 PG3=EXT_RST가 맞다**는 쪽을 지지한다. 다만 CN2 pin4가 어느 넷에 붙는지는 회로도로만 최종 확인 가능하므로 **결정적 증거는 아니다.**

### 아직 측정하지 않음

- **PCB 실크 `DK32F746G$CUx` 문자열** — 보드를 뒤집어 육안 확인. (QSPI Winbond 실측으로 `$CU4`가 유력하나 실크로 못 박아야 한다.)
- **CU4 실장 LCD 패널 부품번호** — LCD FPC/모듈 뒷면 라벨 육안 확인. LTDC 타이밍은 이미 RK043FN48H와 동일함이 확인됐다.
- **VCP 실제 통신(USART1 PA9/PB7 AF7, 115200 8N1)** — 펌웨어에서 USART1로 문자열을 송신하고 호스트에서 `/dev/cu.usbmodem83303`을 115200으로 열어 수신 확인. (포트가 열거된다는 것까지는 확인됨.)
- **ST-LINK Mass Storage 볼륨명 `DIS_F746NG`** — 호스트에 이미 3.2 MB 디스크로 열거되어 있으니 마운트해 볼륨명과 `DETAILS.TXT`(펌웨어 버전 문자열)를 읽으면 된다.
- **X1 = 24 MHz 오실레이터** — TIM 입력 캡처로 직접 잴 수 없다. `RCC_AHB1ENR.OTGHSULPIEN`을 켜고 USB3320 PHY가 ULPI로 응답하는지, 또는 DCMI_PIXCK 입력으로 클럭이 살아 있는지로 간접 확인.
- **LD2 / LD3 / LD4 / LD5 / LD6 / LD7 점등 상태** — 육안 확인(전원·USB 상태 LED는 GPIO가 아니라 소프트웨어로 읽을 수 없다).
- **JP1 점퍼 위치와 5 V 소스** — 육안 확인.
- **보드 전체 / MCU 단독 소비 전류** — 문서 §12.3대로 전용 점퍼가 없으므로 VDD 배선 절단 + 션트가 필요하다. 비침습 대안은 CN14 USB 라인에 USB 전류계를 끼워 보드 총 전류를 재는 것.
- **오디오 경로(SAI2: PI4/PI5/PI6/PI7/PG10)** — 실측 덤프에서 이 핀들은 AF 미설정 상태였다. WM8994를 초기화하고 SAI2로 톤을 출력해 CN10(Line Out)에서 확인.
- **MEMS 마이크 MP34DT01 2개** — WM8994의 디지털 마이크 입력으로 PDM을 캡처해 신호 유무 확인.
- **SPDIF 입력(PD7)** — 외부 SPDIF 소스 필요.
- **이더넷 실제 링크·처리량** — RJ45에 케이블을 꽂고 BMSR 링크업 비트와 자동협상 결과를 재판독, MAC 루프백으로 처리량 측정.
- **USB OTG FS(CN13) / HS(CN12) 열거** — 호스트 PC에 연결해 장치 인식 여부 확인. (단 실측 `PLLQ = 44.44 MHz`라 현재 클럭 설정으로는 48 MHz가 나오지 않아 USB가 동작하지 않는다는 점을 먼저 고쳐야 한다.)
- **카메라 커넥터 P1 / 모듈** — I2C3의 0x60이 무응답이라 미장착이 간접 확인됐으나, 모듈을 꽂은 상태의 확인은 미실시.
- **Arduino 커넥터 CN4/CN5/CN7 핀맵** — 각 핀을 GPIO로 토글하며 외부 프로브로 확인 필요. CN7의 I2C1(PB8/PB9)은 이미 **빈 버스**임이 확인됐다.
- **ES0290 에라타의 Rev Z 적용 목록** — 실리콘 리비전은 Rev Z로 확정됐으므로, 에라타 원문만 확보하면 대조 가능.

---

## 요약

- 보드 주문 코드는 **STM32F746G-DISCO**(제품 페이지 명칭 **32F746GDISCOVERY**), 보드 레퍼런스는 **MB1191**, 실장 MCU는 **STM32F746NGH6 / TFBGA216(13×13 mm, 0.8 mm 피치)** 이다. [A]
- **STM32F7508-DK도 동일한 MB1191 보드**다. ST가 공개한 F7508-DK 회로도 파일명 자체가 `mb1191-f750n8h6-c01_schematic.pdf`이며, 시트 구성(14장)도 F746 판과 동일하다. 차이는 **실장 MCU 한 개(STM32F750N8H6 vs STM32F746NGH6)** 뿐이고, **플래시만 64 KB vs 1 MB로 다르며 SRAM은 320+16+4 KB = 340 KB로 동일**하다. [A][B]
- CubeProgrammer가 "Board: STM32F7508-DK"로 보고하는 값은 **MCU에서 읽는 값이 아니라 온보드 ST-LINK/V2-1 펌웨어가 들고 있는 문자열**(`debugConnectParameters.board[100]`)이다. 두 보드가 같은 MB1191이라 ST-LINK 보드 문자열이 구분되지 않는다. [D]
- **실제 칩을 확정하는 유일한 방법은 플래시 크기 레지스터 `0x1FF0F442`(F_SIZE, KB 단위) 읽기**다. `0x0400`(=1024 KB) → F746NG, `0x0040`(=64 KB) → F750N8. CubeProgrammer 내부 DB(`STM32_Prog_DB_0x449.xml`)도 정확히 이 레지스터의 **bit 6(mask 0x40)** 로 두 다이를 구분한다. DBGMCU_IDCODE의 DEV_ID는 F746/F750 모두 **0x449**로 같아서 구분에 쓸 수 없다. [A][B]
- LCD는 ROCKTECH **RK043FN48H-CT672B**(4.3", 480×272, 드라이버 IC **OTA5180A**, 정전식 터치 IC **FT5336GQQ** 내장, 40핀 FPC). LTDC 픽셀 클럭은 ST BSP 기준 **9.6 MHz**, 프레임레이트 약 **59.3 Hz**. RGB565 프레임버퍼 1장 = **261,120 B (255.0 KiB)**. [A][B][D]
  > **실측 정정**: 픽셀 클럭 9.60 MHz·59.30 Hz는 레지스터 실측과 일치하나, 이 보드에서 돌던
  > 펌웨어의 프레임버퍼는 **RGB565가 아니라 ARGB8888 더블버퍼**다 (522,240 B/장, 합 1,044,480 B).
- 터치 컨트롤러 FT5336 I2C 주소 = **0x70 (8-bit) / 0x38 (7-bit)**, 오디오 코덱 WM8994ECS/R I2C 주소 = **0x34 (8-bit) / 0x1A (7-bit)**. 둘 다 **I2C3 (SCL=PH7, SDA=PH8)** 에 공유 연결. [A]
- 사용자 LED **LD1(녹색) = PI1**(Arduino D13 공유), 사용자 버튼 **B1(파랑, USER & WAKE-UP) = PI11**, 리셋 버튼 **B2 = NRST**. LCD의 RESET 핀은 별도 GPIO가 아니라 **보드 NRST에 직결**. [A]
- ST-LINK/V2-1 VCP는 MCU의 **USART1**에 연결: **VCP_TX = PA9 (BGA E15)**, **VCP_RX = PB7 (BGA B5)**, AF7. ST 예제 기본 보레이트 **115200 8N1**. [A][C]
- 전원은 JP1 점퍼로 5V 소스를 선택(5V link / 5V ext / usb_fs / usb_hs), 기본값은 **5V link(CN14 ST-LINK USB mini-B)**. **IDD 전류 측정 전용 점퍼는 이 보드에 없다.** [A][C]

---

## 1. 보드 식별 · 주문 코드 · 리비전

### 1.1 주문 코드 / 명칭

| 항목 | 값 | 신뢰도 | 출처 |
|---|---|---|---|
| 주문 코드(order code) | `STM32F746G-DISCO` | [A] | UM1907 §1, ST 제품 페이지 |
| 제품/키트 명칭 | `32F746GDISCOVERY` | [A] | UM1907 Introduction, st.com/en/evaluation-tools/32f746gdiscovery.html |
| 보드 레퍼런스(PCB 실크/회로도 타이틀블록) | `MB1191` | [A] | UM1907 회로도 타이틀블록 전 시트 "Reference: MB1191" |
| 실장 MCU 주문 코드 | `STM32F746NGH6` | [A] | UM1907 §1 |
| MCU 패키지 | **TFBGA216** (216핀, 13×13 mm, 0.8 mm 볼 피치) | [A] | UM1907 §1 "STM32F746NGH6 (216-pin TFBGA package)"; 패키지 치수는 DS10916 / DigiKey 216-TFBGA(13×13) [B] |
| Digi-Key 파트 | 497-15680-5-ND | [C] | DigiKey |

> **"STM32F746G-DISCO-B" 같은 `-B` 접미 주문 코드는 확인되지 않았다.** ST의 F7 4.3" 디스커버리 계열에서 확인되는 별도 주문 코드는 `STM32F746G-DISCO`, `STM32F7508-DK`, `STM32F7308-DK` 세 가지다. [B] (CubeIDE `stm32targets.xml` 보드 목록)

### 1.2 보드 리비전 (중요 — QSPI/LCD가 바뀐다)

PCB에 인쇄된 `DK32F746G$CUx` 형태의 코드가 하드웨어 리비전을 나타낸다.

| 마킹 | 회로도 리비전 | QSPI NOR 플래시 | LCD 드라이버 | BSP 요구 define | 신뢰도 |
|---|---|---|---|---|---|
| (초기) | MB1191-F746NGH6-**B-02** (2015-05-28) | Micron **N25Q128A13EF840E** (128 Mbit) | — | 없음 | [A] UM1907 회로도 |
| `$CU2` | MB1191-F746NGH6-**C-01** (2019-07-17) | Micron N25Q128A13EF840E | HT043C | 없음 | [C] ST 커뮤니티 |
| `$CU4` | MB1191-F746NGH6-**C-03** | Winbond **W25Q128JVEIQ** (128 Mbit) | 변경됨(부품번호 미공개) | `USE_STM32746G_DISCO_REVC03` | [C] ST 커뮤니티 + [B] ST BSP 소스 |

> **실측 확정**: **이 보드는 `$CU4` (C-03) 행이다.** QSPI에 Read JEDEC ID(0x9F)를 직접 보내
> 받은 응답이 **`0xEF4018`** = Winbond(0xEF) 16 MB다. Micron N25Q128A라면 `0x20BA18`이어야 한다.
> 위 리비전 표는 실측으로 검증됐다.
> 단, **LCD는 "변경됨"인데도 LTDC 타이밍이 RK043FN48H 값(41/13/32/10/2/2)과 완전히 동일**했다
> — 패널 부품이 바뀌었더라도 구동 타이밍은 같다는 뜻이다.
> ST BSP는 `USE_STM32746G_DISCO_REVC03` 없이(=N25Q 전제로) 빌드됐는데도
> `BSP_QSPI_Init()`과 메모리맵 읽기가 정상 동작했다(0x9F/0xEB 명령이 두 부품 공통이라서).

- ST 공식 BSP `stm32746g_discovery_qspi.h`는 `USE_STM32746G_DISCO_REVC03` 정의 시 **W25Q128J**, 그렇지 않으면 **N25Q128A13EF840E** 드라이버를 선택한다. [B]
- C-03 보드에 구 BSP를 그대로 쓰면 QSPI 4비트 모드 초기화가 실패한다(Micron/Winbond 간 명령·상태 레지스터 상이). [C]
- **사용자 보드가 CU2/CU4 중 무엇인지는 PCB 실크의 `DK32F746G$CUx` 문자열로 확인**한다. [C]

---

## 2. STM32F7508-DK와의 차이 — CubeProgrammer가 왜 구분 못 하는가

### 2.1 결론: 같은 PCB, 다른 MCU 한 개

| 항목 | STM32F746G-DISCO | STM32F7508-DK | 신뢰도 |
|---|---|---|---|
| 보드 레퍼런스 | **MB1191** | **MB1191** (회로도 `mb1191-f750n8h6-c01_schematic.pdf`, 타이틀블록 "Reference: MB1191 / Revision: C 01") | [A] |
| 유저매뉴얼 | UM1907 | UM2470 | [B] |
| 실장 MCU | STM32F746NGH6 | STM32F750N8H6 | [A] |
| 내장 Flash | **1 MB** (0x08000000, 1 MB) | **64 KB** (0x08000000, 0x10000) | [A] RM0385 / [B] CubeProgrammer DB |
| 내장 SRAM | 320+16+4 KB = **340 KB** | 320+16+4 KB = **340 KB** (동일) | [A] UM1907 / [B] Zephyr, ST 데이터브리프 |
| DBGMCU DEV_ID | **0x449** | **0x449** (동일!) | [A] RM0385 §40.6.1 |
| 회로도 시트 구성 | 14 시트 | 14 시트, 제목 동일 | [A] |
| LD1 / 사용자 버튼 | PI1 / PI11 | PI1 / PI11 (동일) | [A]/[B] |
| VCP UART | USART1 PA9/PB7 | USART1 PA9/PB7 (동일) | [A]/[B] |
| 제품 상태 | Active | **NRND** (ST가 F7-DK 라인업 정리 목적으로 단종 예고) | [C] ST 커뮤니티 (STOne-32) |

> ST 커뮤니티에서 ST 모더레이터(STOne-32)는 "identical board (MB1911)"이라고 썼으나, **ST가 실제로 배포하는 F7508-DK 회로도 파일명과 타이틀블록은 MB1191**이다. MB1911은 오타로 보인다. [A] > [C]

### 2.2 왜 "Board: STM32F7508-DK"로 표시되는가

1. STM32CubeProgrammer의 Target information 패널 **"Board"** 필드는 **MCU에서 읽어오는 값이 아니다.** CubeProgrammer API 구조체 `debugConnectParameters`의 `char board[100]  /* board Name */` 필드로, `getStLinkList()`가 **연결된 ST-LINK 프로브에서** 채워 넣는다. [C] (CubeProgrammer_API.h)
2. ST-LINK/V2-1(온보드 STM32F103CBT6, U14)은 자체 플래시의 **설정 영역 `0x08003C00`** 에 USB mass-storage 볼륨명(오프셋 0x40+0x02, 태그 `'V'` + 11자)과 **보드 ID(오프셋 0x40+0x0F, 태그 `'B'` + 4자)** 를 저장한다. [C] (st-link-hack upgrade.md 리버스 엔지니어링 문서)
3. **두 보드가 물리적으로 같은 MB1191**이므로 ST 공장에서 동일/유사한 ST-LINK 펌웨어 이미지·보드 문자열이 적용될 수 있고, 그 결과 CubeProgrammer는 MCU와 무관하게 저장된 보드명을 그대로 출력한다. [D] — 근거: (a) MB1191 공유가 [A]로 확인됨, (b) board 필드가 ST-LINK 출처임이 [C]로 확인됨, (c) CubeProgrammer 설치본 어디에도 "STM32F7508-DK" ↔ MCU 매핑 문자열이 없음(로컬 `tools/` 전체 grep 결과 0건).
4. 참고로 **STM32CubeIDE의 보드 DB(`stm32targets.xml`)는 두 보드를 명확히 분리**해 갖고 있다(각각 `parent = stm32f746nghx` / `stm32f750n8hx`). 즉 툴체인 DB의 문제가 아니라 **프로브가 보고하는 문자열**의 문제다. [B]

### 2.3 실제 칩을 확정하는 방법 (권장 순서)

| # | 방법 | 판정 기준 | 신뢰도 |
|---|---|---|---|
| 1 | **플래시 크기 레지스터** `0x1FF0F442` (16-bit, KB 단위) 읽기 | `0x0400` = 1024 KB → **F746NG** / `0x0040` = 64 KB → **F750N8** | [A] RM0385 §41.2 |
| 2 | CubeProgrammer 상단 **"Flash size"** 필드 확인 | `1 MBytes` → F746NG | [B] |
| 3 | CubeProgrammer 메모리 뷰에서 `0x08010000` 읽기 시도 | F750N8은 64 KB뿐이라 접근 실패/미러 | [D] |
| 4 | **PCB 실크 + MCU 상부 마킹 육안 확인** | 칩에 `STM32F746NGH6` 각인 | [D] |
| 5 | 패키지 데이터 레지스터 `0x1FFF7BF0` bit[10:8] | `0b1xx` → LQFP208/TFBGA216 | [A] RM0385 §41.3 |

> **실측 정정 (5번 항목의 주소)**: `0x1FFF7BF0`은 **STM32F4 계열의 주소**다. ST가 배포하는
> CMSIS 헤더 `stm32f746xx.h:1230`은 F746의 패키지 레지스터를 **`PACKAGE_BASE 0x1FF0F7E0`** 으로
> 정의한다. 실측값:
> - `0x1FF0F7E0` = **`0x038DFC72`** → PKG[2:0] = `0b100` → **LQFP208/TFBGA216** (판정 규칙 자체는 맞다)
> - `0x1FF07BF0` = `0xFFFFFFFF` (미프로그램)
>
> 참고로 **실측 자료 쪽도 이 지점에서 틀렸다** — 프로브 펌웨어와 ST-LINK 덤프가
> `0x1FF07BF0`(문서·CMSIS 어느 쪽도 아닌 제3의 주소)을 읽고 "패키지 판별 불가"로 결론지었다.
> **올바른 주소는 `0x1FF0F7E0`이며, 그 값으로는 TFBGA216 판정이 지지된다.**
| 6 | DBGMCU_IDCODE `0xE0042000` | **구분 불가** (양쪽 다 DEV_ID=0x449) — REV_ID만 얻음 | [A] RM0385 §40.6.1 |

CubeProgrammer CLI 예시(주소·값이 핵심, 플래그 표기는 버전에 따라 다를 수 있음 [D]):

```
STM32_Programmer_CLI -c port=SWD -r32 0xE0042000 4      # DBGMCU_IDCODE  -> 0x1001_0449 이면 Rev Z / DEV_ID 0x449
STM32_Programmer_CLI -c port=SWD -r16 0x1FF0F442 2      # F_SIZE         -> 0x0400 = 1024 KB (F746NG)
STM32_Programmer_CLI -c port=SWD -r32 0x1FF0F420 12     # 96-bit UID
STM32_Programmer_CLI -c port=SWD -r16 0x1FFF7BF0 2      # PKG[2:0]
```

### 2.4 관련 레지스터 정리 (RM0385)

| 레지스터 | 주소 | 내용 | 신뢰도 |
|---|---|---|---|
| DBGMCU_IDCODE | `0xE0042000` (32-bit RO) | bits[11:0] DEV_ID = **0x449**; bits[31:16] REV_ID: **0x1000 = Rev A, 0x1001 = Rev Z** | [A] RM0385 §40.6.1 |
| Unique device ID | `0x1FF0F420` (96-bit RO) | 공장 프로그램된 고유 ID | [A] RM0385 §41.1 |
| Flash size (F_SIZE) | `0x1FF0F442` (16-bit RO) | KB 단위 크기. 0x0400 = 1024 KB | [A] RM0385 §41.2 |
| Package data | `0x1FFF7BF0` (16-bit RO) → **실측 정정: `0x1FF0F7E0`** (ST CMSIS `PACKAGE_BASE`) | PKG[2:0] = bits[10:8]. `0b1xx` = LQFP208/TFBGA216. **실측 `0x038DFC72` → `0b100` → TFBGA216** | [A] RM0385 §41.3 / **[A-실측]** |
| Bootloader version | `0x1FF0EDBE` | 시스템 부트로더 버전 | [B] CubeProgrammer DB 0x449 |
| JTAG BSC ID | — | `0x06449041` | [A] RM0385 §40.6.2 |
| Cortex-M7 Debug Port ID | — | `0x5BA00477` | [A] RM0385 §40.6.3 |

> 실측 Revision ID = **Rev Z**(REV_ID = 0x1001)는 STM32F74x/F75x의 **두 번째(최신) 실리콘 리비전**이며, RM0385가 정의한 두 값(A/Z) 중 하나다. [A]

### 2.5 F746NG 플래시 섹터 구성 (CubeProgrammer DB `STM32_Prog_DB_0x449.xml`)

| 섹터 | 시작 주소 | 개수 | 크기 |
|---|---|---|---|
| sector0–3 | `0x08000000` | 4 | 32 KB each |
| sector4 | `0x08020000` | 1 | 128 KB |
| sector5–7 | `0x08040000` | 3 | 256 KB |
| 합계 | — | 8 | **1024 KB** |

- ITCM 별칭(alias): `0x00200000` 에 동일 구성으로 매핑. [B]
- OTP 영역: `0x1FF0F000`, 512 B(0x200). [B]
- 옵션 바이트 FLASH_OPTCR: `0x40023C14` (RDP: 0xAA=Level0, 0xCC=Level2). [B]
- F750N8 구성(config 1): `0x08000000`에 32 KB × 2 = **64 KB**. [B]

---

## 3. MCU 메모리 맵 (보드 관점)

| 영역 | 주소 | 크기 | 비고 | 신뢰도 |
|---|---|---|---|---|
| ITCM-RAM | `0x00000000` | 16 KB | | [A] RM0385 §2.3 |
| Flash (ITCM alias) | `0x00200000` | 1 MB | | [B] |
| Flash (AXIM) | `0x08000000` | 1 MB | | [A] |
| DTCM-RAM | `0x20000000` | 64 KB | | [A] RM0385 §2.3 |
| SRAM1 | `0x20010000` | 240 KB | | [A] |
| SRAM2 | `0x2004C000` | 16 KB | | [A] |
| Backup SRAM | — | 4 KB | 320+16+4 KB 중 4 KB | [A] UM1907 |
| QSPI 메모리맵 | `0x90000000` | 16 MB (128 Mbit) | 외장 NOR | [A] RM0385 |
| FMC SDRAM Bank1 | `0xC0000000` | 8 MB 사용 가능 | 외장 SDRAM | [A] RM0385 §13 + [B] BSP |

> SRAM 합계 320 KB (= DTCM 64 + SRAM1 240 + SRAM2 16), 여기에 ITCM-RAM 16 KB + 백업 SRAM 4 KB를 더해 UM1907이 말하는 **"320+16+4 KB = 340 KB"** 가 된다. [A][D]

---

## 4. LCD (LTDC)

### 4.1 패널 사양 — ROCKTECH RK043FN48H-CT672B

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 부품번호 | **RK043FN48H-CT672B** (ROCKTECH) | [A] UM1907 Figure 31 |
| 크기 / 해상도 | 4.3", **480(RGB) × 272** | [A] |
| 액티브 영역 | 95.04 mm(W) × 53.86 mm(H) | [A] Rocktech DS v1.0 §1 |
| 외형 | 105.5 × 67.2 × 4.3 mm (CTP 포함) | [A] |
| 도트 피치 | 0.198 mm × 0.198 mm | [A] |
| 입력 신호 | **RGB 24-bit 병렬** (R0–R7 / G0–G7 / B0–B7) | [A] |
| 표시 모드 | Normally White transmissive, 12시 방향 | [A] |
| **드라이버 IC** | **OTA5180A** | [A] Rocktech DS §1 |
| **터치 IC (CTP)** | **FT5336GQQ** | [A] Rocktech DS §1 |
| 커넥터 | 40핀 FPC (1=K, 2=A, 3=GND, 4=VDD, 5–12=R0–R7, 13–20=G0–G7, 21–28=B0–B7, 29=GND, 30=DCLK, 31=DISP, 32=HSYNC, 33=VSYNC, 34=DE, 35/36=GND, 37=RESET, 38=INT, 39=SCL, 40=SDA) | [A] Rocktech DS §6.1 |
| 로직 전원 VDD | 3.0 / **3.3** / 3.6 V | [A] |
| 백라이트 | VF 15.0 / **16.0** / 17.0 V, IF **40 mA**, 640 mW, 수명 20,000 h | [A] |
| 휘도 / 명암비 | 360 cd/m² (CTP 부착 후) / C/R 500:1 typ | [A] |
| 응답시간 | Tr+Tf = 20 ms typ | [A] |
| 동작 온도 | −20 ~ +70 °C | [A] |

> Rocktech 데이터시트 §7 "Timing Characteristics"의 수치 표는 PDF 내 이미지로만 존재해 텍스트 추출 불가. DCLK 상세 규격은 "refer to spec of OTA5180A"로 위임되어 있다. [A]

### 4.2 LTDC 타이밍 · 픽셀 클럭 (ST 공식 드라이버 값)

ST 컴포넌트 드라이버 `rk043fn48h.h` 정의값: [B]

| 파라미터 | 값 |
|---|---|
| `RK043FN48H_WIDTH` | 480 |
| `RK043FN48H_HEIGHT` | 272 |
| `RK043FN48H_HSYNC` | **41** |
| `RK043FN48H_HBP` | **13** |
| `RK043FN48H_HFP` | **32** |
| `RK043FN48H_VSYNC` | **10** |
| `RK043FN48H_VBP` | **2** |
| `RK043FN48H_VFP` | **2** |
| `RK043FN48H_FREQUENCY_DIVIDER` | 5 (PLLSAIR) |

`BSP_LCD_ClockConfig()` (stm32746g_discovery_lcd.c): HSE 25 MHz, PLLM=25 → 1 MHz, **PLLSAIN = 192** → VCO 192 MHz, **PLLSAIR = 5** → 38.4 MHz, **PLLSAIDivR = RCC_PLLSAIDIVR_4** → **LTDC 픽셀 클럭 = 9.6 MHz**. [B]

파생 계산 [D]:

| 항목 | 계산 | 값 |
|---|---|---|
| 수평 총 클럭 | 41 + 13 + 480 + 32 | **566** |
| 수직 총 라인 | 10 + 2 + 272 + 2 | **286** |
| 프레임당 픽셀 클럭 | 566 × 286 | **161,876** |
| 프레임레이트 | 9.6 MHz ÷ 161,876 | **≈ 59.3 Hz** |
| 프레임 주기 | 1 / 59.3 | **≈ 16.86 ms** |
| 라인 시간 | 566 / 9.6 MHz | **≈ 58.96 µs** |

### 4.3 프레임버퍼 크기 계산 [D]

480 × 272 = **130,560 픽셀**.

| 픽셀 포맷 | bytes/px | 프레임버퍼 1장 | KiB | 더블버퍼 |
|---|---|---|---|---|
| L8 (8bpp) | 1 | 130,560 B | 127.5 | 261,120 B |
| **RGB565 (16bpp)** | **2** | **261,120 B** | **255.0** | **522,240 B (510 KiB)** |
| RGB888 (24bpp) | 3 | 391,680 B | 382.5 | 783,360 B |
| ARGB8888 (32bpp) | 4 | 522,240 B | 510.0 | 1,044,480 B |

- 기본 프레임버퍼 위치는 **SDRAM `0xC0000000`** (ST BSP `LCD_FB_START_ADDRESS`). 8 MB SDRAM이므로 RGB565 더블버퍼(510 KiB)도 여유. [B]
- **RGB565 단일 버퍼는 내부 SRAM1(240 KB)에 들어가지 않는다** (255 KiB > 240 KiB). SDRAM 필수. [D]
- LTDC DMA 리드 대역폭(RGB565, 액티브 픽셀만): 261,120 B × 59.3 fps ≈ **15.5 MB/s**. [D]

> **실측 정정**: 이 보드에서 실제로 돌던 펌웨어는 **RGB565가 아니라 ARGB8888(32bpp) 더블 버퍼**다.
> 레이어 PFCR = 0(ARGB8888), 레이어1 CFBAR = `0xC0000000`, 레이어2 CFBAR = `0xC007F800`,
> 피치 1,920 B × 272 라인 = **522,240 B/장**, 두 장 합 **1,044,480 B (≈1.0 MB)**.
> 위 표의 **ARGB8888 행이 실측과 바이트 단위로 일치**한다 — 표 자체는 정확하고,
> 요약·체크리스트가 대표값으로 RGB565 행을 고른 것이 실제와 다를 뿐이다.
> ARGB8888 기준 LTDC DMA 리드 대역폭은 522,240 B × 59.3 fps ≈ **31.0 MB/s**다.

### 4.4 LTDC 핀 매핑 (STM32F746NGH6, TFBGA216)

전 항목 UM1907 Table 10 (Appendix A) 기준 [A]. AF는 ST BSP 기준 [B].

| 신호 | GPIO | BGA 볼 | AF |
|---|---|---|---|
| LTDC_R0 | PI15 | G4 | AF14 |
| LTDC_R1 | PJ0 | R6 | AF14 |
| LTDC_R2 | PJ1 | R7 | AF14 |
| LTDC_R3 | PJ2 | P7 | AF14 |
| LTDC_R4 | PJ3 | N8 | AF14 |
| LTDC_R5 | PJ4 | M9 | AF14 |
| LTDC_R6 | PJ5 | M14 | AF14 |
| LTDC_R7 | PJ6 | K12 | AF14 |
| LTDC_G0 | PJ7 | J12 | AF14 |
| LTDC_G1 | PJ8 | H12 | AF14 |
| LTDC_G2 | PJ9 | J13 | AF14 |
| LTDC_G3 | PJ10 | H13 | AF14 |
| LTDC_G4 | PJ11 | G12 | AF14 |
| LTDC_G5 | PK0 | G13 | AF14 |
| LTDC_G6 | PK1 | F12 | AF14 |
| LTDC_G7 | PK2 | F13 | AF14 |
| LTDC_B0 | PE4 | A1 | AF14 |
| LTDC_B1 | PJ13 | B9 | AF14 |
| LTDC_B2 | PJ14 | C9 | AF14 |
| LTDC_B3 | PJ15 | D10 | AF14 |
| **LTDC_B4** | **PG12** | **C7** | **AF9 (예외!)** |
| LTDC_B5 | PK4 | D7 | AF14 |
| LTDC_B6 | PK5 | C6 | AF14 |
| LTDC_B7 | PK6 | C5 | AF14 |
| LTDC_CLK | PI14 | H3 | AF14 |
| LTDC_HSYNC | PI10 | D5 | AF14 |
| LTDC_VSYNC | PI9 | E4 | AF14 |
| LTDC_DE | PK7 | C4 | AF14 |
| LCD_DISP (GPIO out) | PI12 | E3 | GPIO PP |
| LCD_BL_CTRL (GPIO out) | PK3 | D8 | GPIO PP |
| LCD_INT (터치 인터럽트) | PI13 | G3 | GPIO EXTI |
| LCD_SCL (= I2C3_SCL) | PH7 | N13 | AF4 |
| LCD_SDA (= I2C3_SDA) | PH8 | P14 | AF4 |
| LCD_RST | — | — | **보드 NRST에 직결** (전용 GPIO 없음) | 

> **PG12만 AF9**를 쓴다. ST BSP `stm32746g_discovery_lcd.c` MspInit에서 GPIOE/I/J/K는 `GPIO_AF14_LTDC`, GPIOG pin12만 `GPIO_AF9_LTDC`로 설정한다. CubeMX에서 수동 설정 시 흔한 실수 지점. [B]

> 백라이트 드라이버는 **STLD40DPUR** 부스트 컨버터 (BLVDD → VH), `LCD_BL_CTRL`(PK3)이 EN 제어. [A] UM1907 Figure 31

---

## 5. 터치 스크린 (FT5336)

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 컨트롤러 | **FT5336GQQ** (FocalTech), LCD 모듈 내장 | [A] Rocktech DS §1 |
| 방식 | 정전식(capacitive), 멀티터치 | [A] UM1907 |
| **I2C 주소** | **0x70 (8-bit write) / 0x38 (7-bit)** | [A] UM1907 Figure 31 "Default I2C Address : 01110000" |
| 버스 | **I2C3** — SCL = PH7(N13, AF4), SDA = PH8(P14, AF4) | [A] UM1907 Table 10 + [B] BSP `TS_I2C_ADDRESS 0x70`, `TS_I2C = I2C3` |
| 인터럽트 | LCD_INT = **PI13** (BGA G3), EXTI15_10 | [A] |
| 버스 공유 | 동일 I2C3에 **WM8994 오디오 코덱**도 연결 (`AUDIO_SDA/AUDIO_SCL` = 같은 PH7/PH8 네트) | [A] UM1907 회로도 |

---

## 6. 오디오

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 코덱 | **WM8994ECS/R** (Wolfson → Cirrus Logic), 4× DAC + 2× ADC, U11 | [A] UM1907 §1.6 |
| **I2C 주소** | **0x34 (8-bit) / 0x1A (7-bit)** | [A] UM1907 Figure 22 "Default I2C Address : 00110100" + [B] BSP `AUDIO_I2C_ADDRESS 0x34` |
| 오디오 데이터 버스 | **SAI2** — SAI2_MCLK_A=PI4(C3), SAI2_SCK_A=PI5(D3), SAI2_FS_A=PI7(D4), SAI2_SD_A=PI6(D6), SAI2_SD_B=PG10(C8) | [A] UM1907 Table 10 |
| 코덱 인터럽트 | Audio_INT = **PD6** (B11) | [A] Table 10 |
| 코덱 전원 | 1.8 V (LD1117S18TR, U13) + 3.3 V | [A] 회로도 |
| Line In | 파란색 3.5 mm 잭 **CN11** → WM8994 ADC | [A] §1.6 |
| Line Out | 초록색 3.5 mm 잭 **CN10** → WM8994 DAC | [A] §1.6 |
| 스피커 출력 | **JP3**(우) / **JP4**(좌) 2핀 헤더 | [A] §2.9 |
| MEMS 마이크 | **MP34DT01TR** ×2 (ST), PDM 디지털, **WM8994의 디지털 마이크 입력에 연결**(MCU 직결 아님) | [A] §1.6 |
| MP34DT01 스펙 | 감도 −26 dBFS ±3 dB @94 dB SPL, SNR 61 dB, 대역 20 Hz–10 kHz | [C] ST/DigiKey 데이터시트 |
| SPDIF 입력 | RCA 코액시얼 **CN1** → SPDIF_RX0 = **PD7** (A11) | [A] §1.12, Table 10 |

---

## 7. 이더넷

| 항목 | 값 | 신뢰도 |
|---|---|---|
| PHY | **LAN8742A-CZ-TR** (Microchip), U9 | [A] UM1907 §1.10 |
| 인터페이스 | **RMII**, 10/100 Mbps | [A] |
| 커넥터 | RJ45 **CN9** (마그네틱·LED 내장) | [A] §2.5 |
| PHY 25 MHz 클럭 | 오실레이터 **X2 (25 MHz)** — MCU HSE와 공용 | [A] §1.4, §1.10 |
| MCU RMII_REF_CLK | PHY가 생성한 **50 MHz** 를 MCU가 수신 | [A] §1.10 |
| PHY 주소 (SMI) | **0x00** (RXER/PHYAD0 스트랩 풀다운) — CubeMX 기본값 1은 동작하지 않음 | [C] → **[A] 실측 확정** |

> **실측 확인 (신뢰도 [C] → [A])**: MDIO(PA2)/MDC(PC1, AF11)로 PHY 주소 0~31을 전부 훑은 결과
> **주소 0에서만 응답**했다. ID1 = `0x0007`, ID2 = `0xC131` → OUI `0x0001F0`(SMSC/Microchip),
> 모델 `0x13`, 리비전 `0x1` → **LAN8742A 확정**. BMCR = `0x3000`, BMSR = `0x7809`(링크 down,
> 케이블 미연결).
> 참고: 프로젝트의 `stm32f7xx_hal_conf.h:220`에는 `DP83848_PHY_ADDRESS 0x01`로 남아 있는데,
> **부품도 주소도 이 보드와 맞지 않는다.**

RMII 핀 매핑 [A] (UM1907 Table 10):

| 신호 | GPIO | BGA |
|---|---|---|
| ETH_REF_CLK (RMII_REF_CLK) | PA1 | N2 |
| ETH_MDIO | PA2 | P2 |
| ETH_MDC | PC1 | M3 |
| ETH_CRS_DV | PA7 | R3 |
| ETH_RXD0 | PC4 | N5 |
| ETH_RXD1 | PC5 | P5 |
| RMII_RXER | PG2 | M13 |
| ETH_TX_EN | PG11 | B8 |
| ETH_TXD0 | PG13 | B3 |
| ETH_TXD1 | PG14 | A4 |

RJ45 CN9 핀 (front view) [A] §2.5: 1=A(노랑 LED), 2=K(노랑 LED), 3=TD+, 4=TD−, 5=CT 3V3, 6=CT 3V3, 7=RD+, 8=RD−, 9=NC, 10=GND, 11=K(녹색 LED), 12=A(녹색 LED)

---

## 8. USB

### 8.1 USB OTG FS (CN13, micro-AB)

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 커넥터 | **CN13**, micro-AB | [A] §2.6 |
| 핀 | 1=VBUS(PJ12 기본), 2=D−(**PA11**, C15), 3=D+(**PA12**, B15), 4=ID(**PA10**, D15), 5=GND | [A] Table 6 + Table 10 |
| 파워 스위치 | U6, 5 V / 500 mA 제한 | [A] §1.7 |
| PowerSwitchOn | PD5 (C10) | [A] Table 10 |
| OverCurrent | PD4 (D11) | [A] Table 10 |
| 상태 LED | 녹색 **LD5**(VBUS 정상), 적색 **LD6**(과전류) | [A] §1.7 |
| VBUS 감지 주의 | 기본 R64=ON/R63=OFF/R58=ON → **PA9 = VCP_TX**, PJ12 = OTG_FS_VBUS(SW 제어). AF로 OTG_FS_VBUS를 쓰려면 R64=OFF/R63=ON/R58=OFF 로 바꿔야 하며 **그 순간 VCP_TX를 잃는다.** | [A] Table 10 note 1 |

### 8.2 USB OTG HS (CN12, micro-AB)

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 커넥터 | **CN12**, micro-AB. 핀 1=VBUS, 2=D−, 3=D+, 4=ID, 5=GND | [A] §2.4 |
| PHY | **USB3320C-EZK** (Microchip), U15, ULPI 8-bit | [A] §1.8 |
| PHY 클럭 | 오실레이터 **X1 (24 MHz)** (카메라 모듈과 공용) | [A] §1.4 |
| ULPI 핀 | D0=PA3(R2), D1=PB0(R5), D2=PB1(R4), D3=PB10(P12), D4=PB11(R13), D5=PB12(L13), D6=PB13(K14), D7=PB5(A8), CK=PA5(P4), STP=PC0(M2), DIR=PC2(M4), NXT=PH4(H4) | [A] Table 10 |
| 파워 스위치 | U7, 5 V / 500 mA 제한. OverCurrent = PE3(A2) | [A] §1.8, Table 10 |
| 상태 LED | 녹색 **LD4**, 적색 **LD3**(과전류) | [A] §1.8 |

---

## 9. 외부 메모리

### 9.1 SDRAM

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 부품 | **MT48LC4M32B2B5-6A** (Micron), 128 Mbit (4M × 32) | [A] UM1907 §1.11 |
| 실제 사용 | **하위 16비트만 연결 → 64 Mbit = 8 MB 접근 가능**. DQ16–DQ31은 10 kΩ 풀다운 | [A] §1.11 |
| 매핑 | FMC SDRAM **Bank 1 = `0xC0000000`**, 크기 `0x800000` | [B] BSP `SDRAM_DEVICE_ADDR/SIZE` |
| 버스 폭 | `FMC_SDRAM_MEM_BUS_WIDTH_16` | [B] |
| SDCLK | `FMC_SDRAM_CLOCK_PERIOD_2` = HCLK/2 → ~~216/2 = 108 MHz~~ → **실측 200/2 = 100.00 MHz** | [B] → **[A] 실측** |
| Refresh count | `0x0603` = 1539 → ~~(1539+20)/108 MHz ≈ 14.4 µs~~ → **실측 (1539+1)/100 MHz = 15.40 µs** | [B] → **[A] 실측** |

> **실측 정정**: 이 보드의 HCLK는 **216 MHz가 아니라 200 MHz**다
> (`RCC_PLLCFGR = 0x29406419` → PLLM=25 / **PLLN=400** / PLLP=2 → VCO 400 MHz → SYSCLK 200 MHz).
> 따라서 SDCLK = **100.00 MHz**(FMC_SDCR1=`0x00001954`의 SDCLK=0b10로 실측 확인).
> 리프레시 간격도 RM0385 공식 `(COUNT+1) × tCK` = 1540 × 10 ns = **15.40 µs**다
> (`+20`은 COUNT를 구할 때 빼는 마진이지 되돌려 더하는 값이 아니다).
| SDCLK 핀 | FMC_SDCLK = **PG8** (H14) | [A] Table 10 |
| 제어 핀 | SDNE0=PH3(J4), SDCKE0=PC3(L4), SDNWE=PH5(J3), SDNRAS=PF11(P8), SDNCAS=PG15(B7), BA0=PG4(N12), BA1=PG5(N11), NBL0=PE0(A6), NBL1=PE1(A5) | [A] Table 10 |

### 9.2 Quad-SPI NOR Flash

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 부품 (B-02 / C-01, "CU2") | **N25Q128A13EF840E** (Micron), 128 Mbit = **16 MB** | [A] UM1907 §1.12 |
| **부품 (C-03, "CU4") ← 이 보드** | **W25Q128JVEIQ** (Winbond), 128 Mbit = 16 MB | [C] → **[A] 실측: JEDEC `0xEF4018`** |
| 매핑 | 메모리맵 모드 `0x90000000` | [A] RM0385 / **[A] 실측 16,384 KB 경계** |
| QSPI 클럭 | `ClockPrescaler = 1` → ~~216/(1+1) = 108 MHz~~ → **실측 200/2 = 100.00 MHz** | [B] → **[A] 실측 `QUADSPI_CR=0x01400311`** |
| FlashSize 필드 | 23 (→ 2^24 = 16 MB) | [B] `QSPI_FLASH_SIZE 23` |
| 페이지 크기 | **256 B** | [B] `QSPI_PAGE_SIZE 256` |
| 기타 설정 | FifoThreshold 4, SampleShifting = HALFCYCLE, CS High Time = 6 cycles | [B] |
| 핀 | CLK=PB2(M5), NCS=PB6(B6), IO0=PD11(N10), IO1=PD12(M10), IO2=PE2(A3), IO3=PD13(M11) | [A] Table 10 |

### 9.3 microSD

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 커넥터 | **CN3** (PJS008-2003), push-push | [A] §2.7, 회로도 |
| 인터페이스 | **SDMMC1**, 4-bit | [A] §1.9 |
| 핀 | CK=PC12(A12), CMD=PD2(D12), D0=PC8(G14), D1=PC9(F14), D2=PC10(B14), D3=PC11(B13) | [A] Table 10 |
| 카드 검출 | uSD_Detect = **PC13** (D1), 풀업 입력, 카드 삽입 시 Low | [A] Table 10 + 회로도 |
| 지원 | 2 GB 이상 microSD | [A] §1.9 |

---

## 10. LED · 버튼 · 리셋

| 부품 | 색/종류 | 연결 | 신뢰도 |
|---|---|---|---|
| **LD1** | 녹색, **사용자 LED** | **PI1** (BGA D14, Arduino SCK/D13 공유), 직렬 저항 R59 = 510 Ω | [A] UM1907 Table 10 + 회로도 Figure 24, [B] BSP `LED1_GPIO_PORT = GPIOI, LED1_PIN = GPIO_PIN_1` |
| LD2 | 적색 | 5 V 전원 정상 표시 (ST890 스위치 ON) | [A] §1.2 |
| LD3 | 적색 | USB OTG **HS** 과전류 | [A] §1.8 |
| LD4 | 녹색 | USB OTG **HS** VBUS 정상 | [A] §1.8 |
| LD5 | 녹색 | USB OTG **FS** VBUS 정상 | [A] §1.7 |
| LD6 | 적색 | USB OTG **FS** 과전류 | [A] §1.7 |
| LD7 | 적/녹 바이컬러 (`LD_BICOLOR_CMS`) | **ST-LINK COM** 상태 (LED_STLINK, 330 Ω) | [A] 회로도 Figure 20 |
| **B1** | 파란색 푸시버튼, `USER & WAKE-UP Button` | **PI11** (BGA F3), 네트 `B_USER`, R18/R19 10 kΩ + C10 100 nF 디바운스, SB6 | [A] Table 10 + 회로도 Figure 21, [B] BSP `BUTTON_WAKEUP/KEY = GPIOI pin 11, EXTI15_10_IRQn` |
| **B2** | 검은색 푸시버튼 | **RESET** — 보드 NRST(액티브 Low), C25 100 nF | [A] §1.5, 회로도 |

리셋 소스 3가지: ① B2 버튼, ② Arduino CN6 pin3 (NRST), ③ 온보드 ST-LINK/V2-1. [A] §1.5

> LD1이 PI1(= Arduino D13 = SPI2_SCK)에 있다는 점에 주의 — **SPI2를 Arduino 커넥터로 쓰면 LED가 같이 깜빡인다.** [D]

---

## 11. ST-LINK/V2-1 · VCP

| 항목 | 값 | 신뢰도 |
|---|---|---|
| 종류 | **ST-LINK/V2-1** (SWD only, JTAG 없음, SWIM 없음) | [A] UM1907 §1.1, 회로도 Figure 20 제목 |
| ST-LINK MCU | **STM32F103CBT6** (U14) | [A] 회로도 |
| ST-LINK 클럭 | NX3225GD-**8.00 MHz** 크리스털 | [A] 회로도 |
| ST-LINK 전원 | LD3985M33R (U12) 3.3 V | [A] 회로도 |
| USB 커넥터 | **CN14, USB mini-B** (회로도 부품 `USB-MINI-typeB`; UM1907 §2.8 제목은 "USB type B"로 표기하나 §1.2 본문은 "USB type mini-B") | [A] |
| 지원 기능 | USB 소프트웨어 재열거, **VCP(가상 COM 포트)**, USB Mass Storage, 100 mA 초과 전력 요청 | [A] §1.1 |
| 미지원 | SWIM, 3 V 미만 애플리케이션 전압 | [A] §1.1 |
| SWD 신호 | SWCLK=**PA14**(A14), SWDIO=**PA13**(A15), SWO=**PB3**(A10), NRST | [A] Table 10 |
| Mass storage 볼륨명 | `DIS_F746NG` | [C] mbed 플랫폼 페이지 |

### 11.1 VCP ↔ MCU UART 연결 (핵심)

| 방향 | ST-LINK 측 네트 | 솔더브리지 | MCU 핀 | BGA 볼 | 페리페럴 | AF |
|---|---|---|---|---|---|---|
| PC → MCU | `STLINK_TX` | **SB12** | **PB7** (`VCP_RX`) | **B5** | **USART1_RX** | AF7 |
| MCU → PC | `STLINK_RX` | **SB13** | **PA9** (`VCP_TX`) | **E15** | **USART1_TX** | AF7 |

- 출처: UM1907 Table 10 (`E15 / PA9 / VCP_TX`, `B5 / PB7 / VCP_RX`) [A], 회로도 Figure 20의 SB12/SB13 [A], ST BSP `COM1 = USART1, TX = GPIOA PIN_9 AF7_USART1, RX = GPIOB PIN_7 AF7_USART1` [B].
- **기본 보레이트**: ST 예제 및 Zephyr 보드 정의 모두 **115200 8N1** 사용. ST-LINK VCP 자체는 보레이트 무관(호스트가 설정한 값을 전달)이므로 이는 소프트웨어 관례다. [C]
- **주의**: PA9는 OTG_FS_VBUS와 배타적이다(§8.1 참조). R64/R63/R58 저항 구성으로 결정되며 기본은 VCP_TX. [A]

### 11.2 ST-LINK 펌웨어 버전 확인 / 업그레이드

1. **STM32CubeProgrammer** → ST-LINK 연결 패널의 **"Firmware version"** 필드. 버전 문자열은 `V2Jxx Myy`(STM32 + Mass storage) 형태. [C]
2. **STM32CubeIDE** → `Help > ST-LINK Upgrade`, 또는 CubeProgrammer의 **"Firmware upgrade"** 버튼. [C]
3. 독립 툴 **STSW-LINK007** (`STLinkUpgrade`). 릴리스 노트는 **RN0093**. [B]
4. USB Mass Storage(`DIS_F746NG` 드라이브)의 **`DETAILS.TXT`** 파일에도 펌웨어 버전 문자열이 기록된다. [C]
5. ST-LINK 설정 영역은 STM32F103 플래시 **`0x08003C00`** (옵션은 `0x08003C40`부터, 볼륨명 태그 `'V'`+11자, 보드 ID 태그 `'B'`+4자, 버전 영역 `0x08003FF0`). [C] (리버스 엔지니어링 자료)

---

## 12. 전원

### 12.1 5 개의 5 V 소스와 JP1 점퍼

보드는 5 V DC로 동작하며 **JP1 점퍼로 5 V 소스를 선택**한다. [A] UM1907 §1.2

| JP1 실크 위치 | 5 V 공급원 | 전류 제한 | 비고 |
|---|---|---|---|
| **`5V link`** (**기본값**) | **CN14** (ST-LINK USB mini-B) | ST890(U8) 스위치, **Ilim ≈ 510 mA** (612 mA < Isc < 765 mA) | USB 열거 성공 시 `PWR_ENn` 어서트 → LD2 점등 |
| `5V ext` | **JP2**(외부 5 V DC 어댑터) 또는 **CN6 VIN**(7–12 V, Arduino 전원 커넥터) | 외부 소스에 의존 | |
| `usb_fs` | **CN13** (USB OTG FS) | 500 mA | |
| `usb_hs` | **CN12** (USB OTG HS) | 500 mA | |

- USB 열거 과정: 열거 전 호스트가 100 mA만 공급 → 열거 시 **500 mA** 요청 → `SetConfiguration` 성공 시 ST890(U8) ON → **적색 LD2 점등**. 호스트가 500 mA를 못 주면 열거 실패, ST890 OFF, MCU 부분 미급전. [A] §1.2
- **USB 충전기(열거 없음)로 급전할 때만** 저항 **R109**를 실장해야 보드가 켜진다. **R109 실장 상태에서 PC를 CN14에 연결하면 PC 손상 위험** — ST 경고. [A] §1.2
- 외부 전원 사용 시 반드시 다음 순서: ① JP1을 `5V ext`/`usb_hs`/`usb_fs`로 설정 → ② 외부 전원 연결 → ③ **LD2 점등 확인** → ④ PC를 CN14에 연결. [A] §1.3

### 12.2 온보드 레귤레이터

| 부품 | 지정자 | 출력 | 용도 | 신뢰도 |
|---|---|---|---|---|
| LD1117S50TR | U6 | 5 V | VIN(7–12 V) → E5V | [A] 회로도 |
| **LD39050PU33R** | U7 | **3.3 V** | 보드 메인 3V3 | [A] |
| LD1117S18TR | U13 | 1.8 V | 오디오 코덱 1V8 | [A] |
| LD3985M33R | U12 | 3.3 V | ST-LINK 전용 3V3_ST_LINK | [A] |
| **ST890CDR** | U8 | 스위치 | ST-LINK USB 5 V 게이팅, Ilim 510 mA | [A] |

### 12.3 IDD (전류 측정) 점퍼

- **STM32F746G-DISCO(MB1191)에는 IDD 측정 전용 점퍼가 없다.** UM1907이 열거하는 점퍼는 **JP1(전원 소스 선택), JP2(외부 5 V 입력), JP3(우 스피커), JP4(좌 스피커)** 뿐이며, IDD/전류 측정 관련 커넥터·션트는 문서에 언급되지 않는다. [A] UM1907 §1.2/§2.9 (부재 확인) + [C]
- STM32L053-DISCO 등 저전력 보드의 `IDD` 점퍼와 달리, F746G-DISCO에서 MCU 전류를 재려면 **VDD 배선/레귤레이터 출력을 물리적으로 절단하고 션트를 삽입**해야 한다. ST 지원은 `STM32Cube_FW_F7\Projects\STM32746G-Discovery\Examples\PWR\PWR_CurrentConsumption`의 readme를 참조하라고만 답변했다. [C]
- 참고: 보드 전체 전류는 LD2 경로(ST890 출력)에서 측정 가능하나 LCD 백라이트(최대 40 mA @16 V ≈ 640 mW)과 이더넷 PHY가 포함되어 MCU 단독 소비를 나타내지 않는다. [D]

---

## 13. 클럭 소스

| 지정자 | 주파수 | 용도 | 신뢰도 |
|---|---|---|---|
| **X1** | **24 MHz** 오실레이터 | USB OTG HS PHY(USB3320C) + 카메라 모듈(P1 pin21 `Camera_CLK`) | [A] §1.4 |
| **X2** | **25 MHz** 오실레이터 | STM32F746NGH6 **HSE** + 이더넷 PHY | [A] §1.4 |
| **X3** | **32.768 kHz** 크리스털 | STM32 내장 **RTC (LSE)** — RCC_OSC32_IN = PC14(E1), RCC_OSC32_OUT = PC15(F1) | [A] §1.4, Table 10 |

- HSE 25 MHz + PLLM=25 → 1 MHz 기준, 216 MHz SYSCLK 및 LTDC 9.6 MHz(PLLSAI) 도출. [B]
- MCU OSC_IN = PH0 (G1), OSC_OUT = PH1 (H1). [A] Table 10

> **실측 정정**: 이 보드의 SYSCLK은 **216 MHz가 아니라 200 MHz**다.
> `RCC_PLLCFGR = 0x29406419` → PLLM=25 / **PLLN=400** / PLLP=2 → VCO 400 MHz → **SYSCLK = HCLK = 200 MHz**,
> PCLK1 = 50 MHz, PCLK2 = 100 MHz. 216 MHz는 데이터시트 최대치이지 이 프로젝트의 설정이 아니다
> (소스에서도 `main.cc:413` → `PLLN = 400`으로 교차 확인).
> LTDC 9.6 MHz는 별개 PLL(PLLSAI 192 MHz / R=5 / DIVR=4)이라 그대로 **실측 9.60 MHz**로 맞다.
>
> **실측 확인 (X2 / X3)**:
> - X2(25 MHz) → HSEON=1 / HSERDY=1 / **HSEBYP=0 = 크리스털 발진 모드**.
> - X3(32.768 kHz) → LSEON을 켜자 LSERDY=1, TIM5 입력 캡처로 잰 **실측 32,765 Hz**(공칭 대비 −0.009%).
>   **32.768 kHz 크리스털이 물리적으로 실장되어 있음이 확정**됐다.
> - (문서에 없던 값) 내장 **LSI 실측 26,652 Hz** — 공칭 32 kHz 대비 **−16.7%**.
>   IWDG 타임아웃을 공칭값으로 계산하면 실제로는 그만큼 길어진다.
> - X1(24 MHz)은 아직 측정하지 않았다.

---

## 14. 커넥터 전체 목록

| 지정자 | 종류 | 설명 | 신뢰도 |
|---|---|---|---|
| CN1 | RCA 코액시얼 | SPDIF 입력 → PD7 | [A] §2.12 |
| **CN2** | 2×4 헤더 (F206A-2*04MGF-A) | **I2C 확장 / RF-EEPROM** | [A] §2.1 |
| CN3 | microSD (PJS008-2003) | SDMMC1 | [A] §2.7 |
| **CN4** | 8핀 female | Arduino digital D0–D7 | [A] §2.3 |
| **CN5** | 6핀 female | Arduino analog A0–A5 | [A] §2.3 |
| **CN6** | 8핀 female | Arduino power (NC/IOREF/RESET/3V3/5V/GND/GND/VIN) | [A] §2.3 |
| **CN7** | 10핀 female | Arduino digital D8–D15 + AREF/GND | [A] §2.3 |
| CN8 | 2×4 헤더 (TSM-104-01-x-DV) | **Not Fitted** (전원 선택 JP1 관련) | [A] 회로도 |
| CN9 | RJ45 | 이더넷 10/100 | [A] §2.5 |
| CN10 | 3.5 mm 잭 (초록) | Line Out | [A] §2.10 |
| CN11 | 3.5 mm 잭 (파랑) | Line In | [A] §2.11 |
| CN12 | micro-AB | USB OTG HS | [A] §2.4 |
| CN13 | micro-AB | USB OTG FS | [A] §2.6 |
| CN14 | USB mini-B | ST-LINK/V2-1 | [A] §2.8 |
| **P1** | 30핀 | 카메라 모듈 (DCMI) — STM32F4DIS-CAM 호환 | [A] §2.2, §1.13 |
| JP2 | 2핀 | 외부 5 V DC 입력 | [A] §1.2 |
| JP3 / JP4 | 2핀 | 우 / 좌 스피커 출력 | [A] §2.9 |

> **STMod+ 커넥터는 이 보드에 없다.** MB1191의 확장 커넥터는 **Arduino Uno V3 (CN4/CN5/CN6/CN7)**, I2C 확장 CN2(2×4), 카메라 P1(30핀)이 전부다. UM1907 §2 커넥터 목록 및 회로도 전 시트에 STMod+ / MikroBUS 항목 없음. [A] (부재 확인)

### 14.1 I2C 확장 커넥터 CN2 (8핀)

| 핀 | 신호 | 핀 | 신호 |
|---|---|---|---|
| 1 | I2C_SDA (**PB9**) | 5 | +3V3 |
| 2 | NC | 6 | NC |
| 3 | I2C_SCL (**PB8**) | 7 | GND |
| 4 | RESET (**PC10**)¹ | 8 | NC or 5V (SB2) |

[A] UM1907 Table 1.
¹ UM1907 Table 1은 `RESET(PC10)`으로 표기하나, 회로도상 네트명은 `EXT_RST`이고 Table 10은 **PG3 = EXT_RST** (BGA M12)로 기재한다 → **문서 내 불일치, 검증 필요**. [A]/[미확인]

> **실측 참고 (PG3 쪽을 지지)**: 실행 중 덤프에서 **PC10은 AF12 = SDMMC1_D2**(microSD 데이터선)로
> 설정되어 있었고, PG3은 입력(플로팅)이었다. PC10이 SD카드 데이터선에 물려 있으므로
> **Table 1의 `RESET(PC10)`은 오기일 가능성이 높고 Table 10의 PG3 = EXT_RST가 맞다**는 쪽이다.
> 다만 CN2 pin4가 실제로 어느 넷에 붙는지는 회로도로만 최종 확인 가능하므로 결정적 증거는 아니다.

### 14.2 Arduino Uno V3 커넥터 핀맵

**주의: STM32 I/O는 3.3 V이며 5 V 톨러런트 Arduino 규격과 다르다.** [A] §2.3

| CN | 핀 | 라벨 | MCU 핀 | 기능 |
|---|---|---|---|---|
| **CN6 (Power)** | 1 | NC | — | — |
| | 2 | IOREF | — | 3.3 V Ref |
| | 3 | RESET | NRST | RESET |
| | 4 | +3V3 | — | 3.3 V in/out |
| | 5 | +5V | — | 5 V out |
| | 6 | GND | — | |
| | 7 | GND | — | |
| | 8 | VIN | — | 7–12 V 입력 |
| **CN5 (Analog)** | 1 | A0 | **PA0** | ADC3_IN0 |
| | 2 | A1 | **PF10** | ADC3_IN8 |
| | 3 | A2 | **PF9** | ADC3_IN7 |
| | 4 | A3 | **PF8** | ADC3_IN6 |
| | 5 | A4 | **PF7** (또는 PB9, SB 선택) | ADC3_IN5 / I2C1_SDA |
| | 6 | A5 | **PF6** (또는 PB8, SB 선택) | ADC3_IN4 / I2C1_SCL |
| **CN4 (Digital)** | 1 | D0 | **PC7** | USART6_RX |
| | 2 | D1 | **PC6** | USART6_TX |
| | 3 | D2 | **PG6** | — |
| | 4 | D3 | **PB4** | TIM3_CH1 (PWM) |
| | 5 | D4 | **PG7** | — |
| | 6 | D5 | **PA8** | TIM1_CH1 (PWM) |
| | 7 | D6 | **PH6** | TIM12_CH1 (PWM) |
| | 8 | D7 | **PI3** | — |
| **CN7 (Digital)** | 1 | D8 | **PI2** | — |
| | 2 | D9 | **PA15** | TIM2_CH1 (PWM) |
| | 3 | D10 | **PI0** | TIM5_CH4 / SPI2_NSS |
| | 4 | D11 | **PB15** | TIM12_CH2 / SPI2_MOSI |
| | 5 | D12 | **PB14** | SPI2_MISO |
| | 6 | D13 | **PI1** | SPI2_SCK — **LD1 공유** |
| | 7 | GND | — | |
| | 8 | AREF | AVDD | |
| | 9 | D14 | **PB9** | I2C1_SDA |
| | 10 | D15 | **PB8** | I2C1_SCL |

[A] UM1907 Table 3 + Table 10 + 회로도 Figure 24. 솔더브리지 SB1/SB3/SB4/SB5로 A4/A5 ↔ D14/D15 경로가 전환된다. [A]

### 14.3 카메라 커넥터 P1 (30핀) 요약

| 핀 | 신호 | 핀 | 신호 |
|---|---|---|---|
| 1 | GND | 16 | GND |
| 4–11 | DCMI_D0–D7 | 17 | DCMI_HSYNC |
| 15 | DCMI_PIXCK | 19 | DCMI_VSYNC |
| 20, 30 | 3V3 | 21 | Camera_CLK (OSC_24M) |
| 14, 23, 29 | GND | 25 | DCMI_PWR_EN |
| — | — | 26 | DCMI_NRST |
| — | — | 27 / 28 | DCMI_SDA / DCMI_SCL |

[A] UM1907 Table 2. DCMI 신호 GPIO: D0=PH9, D1=PH10, D2=PH11, D3=PH12, D4=PH14, D5=PD3, D6=PE5, D7=PE6, HSYNC=PA4, VSYNC=PG9, PIXCK=PA6, PWR_EN=PH13. [A] Table 10

---

## 15. 회로도 · 문서 URL

### 15.1 회로도 PDF (실제 URL)

| 대상 | URL | 비고 | 신뢰도 |
|---|---|---|---|
| **MB1191 F746NGH6 rev B-02** | `https://www.st.com/resource/en/schematic_pack/mb1191-f746ngh6-b02_schematic.pdf` | 2015-05-28, 13 시트 | [B] |
| **MB1191 F746NGH6 rev C-01** ("CU2") | `https://www.st.com/resource/en/schematic_pack/mb1191-f746ngh6-c01_schematic.pdf` | 2019-07-17, 14 시트 | [B] |
| **MB1191 F750N8H6 rev C-01** (STM32F7508-DK) | `https://www.st.com/resource/en/schematic_pack/mb1191-f750n8h6-c01_schematic.pdf` | 2019-07-17, 14 시트 — **동일 MB1191** | [B] |
| B-02 (구 CCC 경로) | `https://www.st.com/content/ccc/resource/technical/layouts_and_diagrams/schematic_pack/group1/05/61/3e/94/2b/2b/48/5e/mb1191-F746NGH6-B02_schematic/files/mb1191-F746NGH6-B02_schematic.pdf/jcr:content/translations/en.mb1191-F746NGH6-B02_schematic.pdf` | 리다이렉트 원본 | [C] |
| C-01 (구 CCC 경로) | `https://www.st.com/content/ccc/resource/technical/layouts_and_diagrams/schematic_pack/group1/ff/cd/ce/2d/f8/fb/40/69/mb1191-F746NGH6-C01_schematic/files/mb1191-F746NGH6-C01_schematic.pdf/jcr:content/translations/en.mb1191-F746NGH6-C01_schematic.pdf` | 리다이렉트 원본 | [C] |
| 진입점 | `https://www.st.com/en/evaluation-tools/32f746gdiscovery.html` → **CAD Resources** 탭 | 리비전별 전체 목록 | [B] |

> **MB1191-F746NGH6-C-03("CU4") 회로도의 직접 URL은 확인하지 못했다.** ST 제품 페이지 CAD Resources에 게시되었을 가능성이 높으나, 검색으로 `mb1191-f746ngh6-c03_schematic.pdf` 형태의 URL을 확정하지 못했다. [미확인]

### 15.2 문서 PDF URL

| 문서 | URL |
|---|---|
| UM1907 (STM32F746G-DISCO UM) | `https://www.st.com/resource/en/user_manual/um1907-discovery-kit-with-stm32f746ng-mcu-stmicroelectronics.pdf` |
| UM1907 Rev 1 (미러, 실제 읽음) | `https://www.farnell.com/datasheets/1944312.pdf` |
| UM2470 (STM32F7508-DK UM) | `https://www.st.com/resource/en/user_manual/dm00537062-discovery-kit-for-stm32f7-series-with-stm32f750n8-mcu-stmicroelectronics.pdf` |
| RM0385 Rev 9 | `https://www.st.com/resource/en/reference_manual/rm0385-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf` |
| RM0385 Rev 2 (미러, 실제 읽음) | `https://download.kamami.pl/p1188987-STM32F7-Reference-EN.pdf` |
| DS10916 (STM32F746xx DS) | `https://www.st.com/resource/en/datasheet/stm32f746ng.pdf` |
| ES0290 (에라타) | `https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf` |
| RN0093 (ST-LINK 펌웨어 업그레이드 릴리스노트) | `https://www.st.com/resource/en/release_note/rn0093-firmware-upgrade-for-stlink-stlinkv2-stlinkv21-and-stlinkv3-boards-stmicroelectronics.pdf` |
| TN1235 (ST-LINK 파생 제품 개요) | `https://www.st.com/resource/en/technical_note/tn1235-overview-of-stlink-derivatives-stmicroelectronics.pdf` |
| Rocktech RK043FN48H-CT672B DS v1.0 (실제 읽음) | `https://mikrocontroller.bplaced.net/wordpress/wp-content/uploads/2018/01/RK043FN48H-CT672B-V1.0.pdf` |

> **주의**: 이 조사 환경에서 `www.st.com` 은 curl/WebFetch로 직접 접근이 차단(타임아웃)되었다. ST 원본 URL은 검색 결과에서 확인한 정확한 경로이며, 본문 내용은 Farnell·kamami 미러본과 GitHub의 ST 공식 BSP 소스로 검증했다.

---

## 16. 실무 체크리스트 (본 프로젝트 관점)

1. **보드 정체 확인**: CubeProgrammer의 "Board" 필드를 믿지 말고 **"Flash size = 1 MBytes"** 또는 `0x1FF0F442` = `0x0400` 을 확인. → F746NG 확정. [A]
2. **리비전 확인**: PCB 실크의 `DK32F746G$CU2` / `$CU4`. CU4면 QSPI가 Winbond W25Q128J이므로 `USE_STM32746G_DISCO_REVC03` 필요. QSPI를 쓰지 않는다면 무관. [C]
3. **UART 로그**: USART1(PA9 TX / PB7 RX, AF7), 115200 8N1 → ST-LINK VCP로 자동 노출. 추가 케이블 불필요. [A][C]
4. **LED 블링크**: PI1 (GPIOI, pin 1), Push-Pull, High = 점등. [A][B]
5. **버튼**: PI11 (GPIOI, pin 11), 외부 풀다운(R19 10 kΩ) + 100 nF → 눌림 = High. EXTI15_10. [A][B]
6. **프레임버퍼**: SDRAM `0xC0000000`, RGB565 255 KiB/장. 내부 SRAM으로는 단일 버퍼도 불가. [B][D]
   > **실측 정정**: 이 보드에서 돌던 펌웨어는 **ARGB8888 더블버퍼**다 —
   > `0xC0000000`(레이어1) / `0xC007F800`(레이어2), **510 KiB/장, 합 1.0 MB**. [A 실측]
7. **전원**: JP1 기본 `5V link`. ST-LINK USB 하나로 급전 시 보드 총 소비는 **510 mA(ST890 Ilim)** 를 넘을 수 없음 → LCD 백라이트 + 이더넷 + SDRAM 동시 사용 시 여유 확인 필요. [A][D]

---

## 출처

실제로 본문을 읽은 문서·페이지:

1. **UM1907 Rev 1 (DocID027942)** — "Discovery kit for STM32F7 Series with STM32F746NG MCU", 2015-06, 46p. 전문 텍스트 추출·정독 (§1.1–§1.14, §2.1–§2.12, Appendix A Table 10, Figures 19–31 회로도).
   `https://www.farnell.com/datasheets/1944312.pdf`
2. **RM0385 Rev 2 (DocID026670)** — "STM32F75xxx and STM32F74xxx advanced Arm-based 32-bit MCUs", §2.3 메모리맵, §13 FMC, §40.6 ID codes, §41 Device electronic signature.
   `https://download.kamami.pl/p1188987-STM32F7-Reference-EN.pdf`
3. **Rocktech RK043FN48H-CT672B LCD Module Specification v1.0** (2015-03-09) — §1 General Features, §2 Absolute Max, §3 Electrical, §4 Optical, §6 Pin Description, §7 Timing, §9 Reliability.
   `https://mikrocontroller.bplaced.net/wordpress/wp-content/uploads/2018/01/RK043FN48H-CT672B-V1.0.pdf`
4. **ST 공식 BSP — 32f746gdiscovery-bsp**
   - `https://raw.githubusercontent.com/STMicroelectronics/32f746gdiscovery-bsp/main/stm32746g_discovery.h`
   - `https://raw.githubusercontent.com/STMicroelectronics/32f746gdiscovery-bsp/main/stm32746g_discovery_lcd.c`
   - `https://raw.githubusercontent.com/STMicroelectronics/32f746gdiscovery-bsp/main/stm32746g_discovery_qspi.h`
   - `https://raw.githubusercontent.com/STMicroelectronics/32f746gdiscovery-bsp/main/stm32746g_discovery_qspi.c`
   - `https://raw.githubusercontent.com/STMicroelectronics/32f746gdiscovery-bsp/main/stm32746g_discovery_sdram.h`
5. **ST 공식 컴포넌트 드라이버 — stm32-rk043fn48h**
   `https://raw.githubusercontent.com/STMicroelectronics/stm32-rk043fn48h/main/rk043fn48h.h`
6. **ST 공식 BSP — stm32f7508-dk-bsp**
   `https://github.com/STMicroelectronics/stm32f7508-dk-bsp/blob/main/stm32f7508_discovery_lcd.h`
7. **UM2470 회로도 (STM32F7508-DK)** — 타이틀블록 "Reference: MB1191 / Revision: C 01", MCU STM32F750N8H6, 14 시트.
   `https://www.manualslib.com/manual/1728344/St-Stm32f7508-Dk.html`
8. **ST Community — "STM32F7508-DK NRND'd"** (STOne-32, ST Technical Moderator): 두 보드 동일 PCB, F750 64 KB vs F746 1 MB.
   `https://community.st.com/t5/stm32-mcus-boards-and-hardware/stm32f7508-dk-nrnd-d/td-p/75303`
9. **ST Community — "STM32F746G Discovery: What is the difference between CU2 and CU4"**: CU2 = MB1191-F746NGH6-C01, CU4 = C03, LCD 드라이버·QSPI 변경, `USE_STM32746G_DISCO_REVC03`.
   `https://community.st.com/t5/stm32-mcus-touchgfx-and-gui/stm32f746g-discovery-what-is-the-difference-between-cu2-and-cu4/td-p/711279`
10. **ST Community — "[SOLVED] I cannot flash and use the QSPI W25Q128J chip on the new STM32F746G-DISCO version C03"**: B02=N25Q128A13EF840E, C03=W25Q128JVEIQ.
    `https://community.st.com/t5/stm32-mcus-boards-and-hardware/solved-i-cannot-flash-and-use-the-qspi-w25q128j-chip-on-the-new/td-p/735735`
11. **ST Community — "STM32F746-DISCO CU2 & CU4 Display Models"**
    `https://community.st.com/t5/stm32-mcus-boards-and-hardware/stm32f746-disco-cu2-amp-cu4-display-models/td-p/745584`
12. **ST Community — "STM32F746G-DISCO Power Measurement"**: IDD 측정은 PWR_CurrentConsumption 예제 readme 참조 (전용 점퍼 언급 없음).
    `https://community.st.com/t5/stm32-mcus-products/stm32f746g-disco-power-measurement/td-p/417062`
13. **ST Community — "STM32F746G-DISCO CubeMX Ethernet buglet"**: LAN8742A_PHY_ADDRESS = 0.
    `https://community.st.com/t5/stm32cubemx-mcus/stm32f746g-disco-cubemx-ethernet-buglet/td-p/413091`
14. **Zephyr Project Documentation — STM32F7508-DK**: LD1=PI1, 버튼=PI11, UART1 PA9/PB7 115200 8N1, Flash 64 KB, RAM 340 KB.
    `https://docs.zephyrproject.org/latest/boards/st/stm32f7508_dk/doc/index.html`
15. **Zephyr Project Documentation — STM32F746G Discovery**
    `https://docs.zephyrproject.org/latest/boards/st/stm32f746g_disco/doc/index.html`
16. **GMMan/st-link-hack — upgrade.md**: ST-LINK/V2-1 설정 영역 `0x08003C00`, 옵션 시작 `0x08003C40`, 볼륨명 태그 `'V'`(오프셋 0x02, 11자), 보드 ID 태그 `'B'`(오프셋 0x0F, 4자), 버전 영역 `0x03F0`.
    `https://github.com/GMMan/st-link-hack/blob/master/upgrade/upgrade.md`
17. **K-Society/KSociety.SharpCubeProgrammer — CubeProgrammer_API.h**: `debugConnectParameters.board[100] /* board Name */`, `getStLinkList()`가 채움.
    `https://github.com/K-Society/KSociety.SharpCubeProgrammer/blob/master/src/01/Programmer/CubeProgrammer_API.h`
18. **mbed — DISCO-F746NG 플랫폼 페이지**: ST-LINK mass storage 볼륨명 `DIS_F746NG`.
    `https://os.mbed.com/platforms/ST-Discovery-F746NG/`
19. **로컬 검증 (본 머신 설치본)** — STM32CubeIDE 1.x:
    - `/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.macosaarch64_1.0.100.202603051304/tools/Data_Base/STM32_Prog_DB_0x449.xml`
      → DEV_ID 0x449 하나에 두 다이(config 0 = 1 MB, config 1 = 64 KB)가 매핑되고 `ReadRegister address="0x1FF0F442" mask="0x40"` 으로 구분. 플래시 섹터 구성, OTP `0x1FF0F000`, 부트로더 버전 `0x1FF0EDBE`, FLASH_OPTCR `0x40023C14` 확인.
    - `/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.productdb_2.5.200.202606241814/resources/board_def/stm32targets.xml`
      → `STM32F746G-DISCO`(parent `stm32f746nghx`, Active), `STM32F7508-DK`(parent `stm32f750n8hx`, Active), `STM32F7308-DK`(Obsolete) 별도 엔트리. 총 225개 보드. ST-LINK 보드명 매핑 필드 없음.
    - `tools/` 전체 `grep -rl "F7508"` → 0건 (CubeProgrammer 바이너리/DB에 F7508 문자열 부재).

기타 참조(검색 결과에서 확인, 본문 미열람): ST 제품 페이지 `32f746gdiscovery.html` / `stm32f7508-dk.html`, DigiKey/Mouser 제품 페이지, RN0093, TN1235, ES0290, DS10916.

---

## 미확인 / 추가 조사 필요

1. **MB1191-F746NGH6-C-03("CU4") 회로도 PDF의 직접 URL** — 존재는 확인(커뮤니티에서 14 시트 언급)했으나 `st.com/resource/en/schematic_pack/...c03...` 형태 URL을 확정하지 못했다. ST 제품 페이지 CAD Resources 탭에서 직접 내려받아야 한다.
2. **CU4 보드의 LCD 부품번호** — CU2가 HT043C(또는 RK043FN48H-CT672B)라는 언급은 있으나, CU4에 실장되는 패널의 정확한 부품번호를 ST 공식 문서에서 확인하지 못했다. ~~LTDC 타이밍이 동일한지도 미확인.~~
   → **부분 해결(2026-07-27)**: 이 보드가 CU4(C-03)임이 QSPI JEDEC ID로 확정됐고, 그 상태에서 **LTDC 타이밍은 RK043FN48H 값(41/13/32/10/2/2, 9.60 MHz, 59.3 Hz)과 완전히 동일**했다. **부품번호만 여전히 미확인.**
3. **CubeProgrammer "Board" 필드의 정확한 공식 소스** — `debugConnectParameters.board`가 ST-LINK에서 온다는 것까지는 확인했으나, **어떤 ST-LINK USB 커맨드/오프셋**에서 "STM32F7508-DK" 문자열이 읽히는지는 ST 공식 문서로 확정하지 못했다. UM2237(STM32CubeProgrammer UM)과 TN1235 원문 확인 필요. (현재 [D])
4. **UM1907 최신 리비전(Rev 3/4 등)의 변경 내역** — 이 조사에서 읽은 것은 **Rev 1 (2015-06)** 이다. 이후 리비전에서 CU2/CU4 관련 내용, JP 설명, ordering information이 추가·수정되었을 가능성이 있다. `st.com` 접근이 차단되어 최신본을 열람하지 못했다.
5. **CN2 pin 4의 RESET 신호 GPIO 불일치** — UM1907 Table 1은 `RESET(PC10)`, Table 10은 `PG3 = EXT_RST`. 회로도 네트는 `EXT_RST`. 문서 오류로 보이나 실측 확인 권장.
   → **정황 증거 확보(2026-07-27)**: 실측에서 **PC10은 SDMMC1_D2(AF12)** 로 쓰이고 있었고 PG3은 입력이었다. Table 1의 PC10 표기가 오기라는 쪽을 지지하나, CN2 pin4의 실제 넷은 회로도로만 확정 가능하다.
6. ~~**LAN8742A PHY 주소 0의 1차 출처**~~ → **해결됨(2026-07-27)**: MDIO로 PHY 주소 0~31을 훑어 **주소 0에서만 응답**했고, ID1/ID2 = `0x0007`/`0xC131`로 **LAN8742A 확정**. 회로도 스트랩 확인 없이 실측만으로 [A]가 됐다.
7. **ES0290 에라타의 Rev A/Rev Z 별 제한사항 목록** — `st.com` 접근 차단으로 미열람. Cortex-M7 r0p1 관련 에라타(캐시/AXI) 확인 필요.
8. **보드 전체/MCU 단독 전류 실측값** — UM1907에 소비전류 규격 없음. DS10916의 IDD 표(216 MHz 조건)와 보드 실측이 필요.
9. **LTDC 최대 픽셀 클럭 규격 (DS10916)** — 9.6 MHz 사용은 확인했으나 STM32F746의 LTDC_CLK 최대 규격값(MHz)을 데이터시트에서 확인하지 못했다.
10. **ST-LINK/V2-1 펌웨어의 현재 최신 버전 문자열** — 검색 결과가 오래된 값(V2J38M27)이라 2026년 기준 최신 버전은 RN0093 최신본 확인 필요.
    → **이 보드의 실측값(2026-07-27)**: **V2J46M33** (CubeProgrammer 표기) / `V2J46S33` (st-flash 표기), USB VID `0x0483` / PID **`0x374B`**, SN `0668FF39314D423143034442`. 최신본인지는 여전히 미확인.
11. **~~보드 정체가 F746NG인지~~** → **해결됨(2026-07-27)**: `0x1FF0F442` = `0x0400`(1024 KB), DEV_ID `0x449`, REV_ID `0x1001`(Rev Z). **STM32F746NGH6 확정.** 문서 §2.3의 판정 절차가 그대로 통했다.
12. **~~보드 리비전~~** → **QSPI로 확정(2026-07-27)**: JEDEC ID `0xEF4018`(Winbond) → **C-03 / `$CU4`**. PCB 실크 육안 확인만 남았다.
