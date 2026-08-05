# STM32F746NG 메모리 맵 · 버스 매트릭스

대상 디바이스: **STM32F746NGH6** (TFBGA216), STM32F746G-DISCO 탑재.
실측 Device ID `0x449`, Revision ID = Rev Z (`REV_ID = 0x1001`).
기준 동작점: SYSCLK = HCLK = 216 MHz, VDD = 3.3 V.

> **실측 정정**: 이 보드가 실제로 도는 동작점은 **SYSCLK = HCLK = 200 MHz**다
> (`RCC_PLLCFGR = 0x29406419` → PLLM=25/PLLN=400/PLLP=2, `RCC_CFGR = 0x0000940A` → HPRE=/1).
> PCLK1 = 50 MHz, PCLK2 = 100 MHz. 216 MHz는 ST 공식 예제(PLLN=432)의 값이지 이 보드의 값이 아니다.
> 아래 본문에서 "216 MHz 기준"으로 계산된 모든 대역폭·wait state 수치는 그만큼 보정해서 읽어야 한다
> (200/216 = 0.926배). VDD 3.3 V는 실측 VDDA 3.311~3.313 V로 확인됨.

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거 자료: `info/raw/measured_facts.md`, `info/results/probe_output.txt`,
> `info/raw/decoded_hotplug.md`, `info/raw/decoded_reset.md`.

### 실측으로 확인됨

- **내부 RAM 4블록 분할** — 문서 ITCM-RAM 16 KB / DTCM 64 KB / SRAM1 240 KB / SRAM2 16 KB
  → 실측: `0x00000000`~`0x00003FFF`가 5.16 cyc/접근이고 `0x00004000`부터 버스 폴트(**ITCM 16 KB 확정**),
  `0x20000000`~`0x2000FFFF`가 5.16 cyc(**DTCM**), `0x20010000`~`0x2004FFFF`가 15.66 cyc(**SRAM1+SRAM2**),
  `0x20050000`부터 버스 폴트. 총 내부 RAM 320 KB 일치. SRAM2(`0x2004C000`)는 별도 쓰기 시험으로 16 KB 확정.
- **"주소는 연속이지만 성능은 연속이 아니다"(§2 주의)** — 실측: TCM 5.16 cyc vs AXI SRAM 15.66 cyc,
  **약 3배 차이**. 문서의 경고가 수치로 확인됨.
- **Flash 1 MB 이중 매핑** — 문서 `0x00200000`(ITCM 별칭)과 `0x08000000`(AXIM)이 같은 물리 어레이
  → 실측: 두 주소의 선두 64바이트가 **바이트 단위로 완전 일치**, 각각 1024 KB에서 폴트.
  반면 `0x00000000`은 불일치 → 별칭이 아니라 ITCM-RAM 실체임도 확인.
- **"ITCM 별칭은 ART가, AXIM은 L1 캐시가 담당"(§3 표)** — 실측이 가장 선명하게 확인해 준 항목:
  AXIM 경로는 ART ON/OFF에 **전혀 반응하지 않고**(159.96 MB/s 동일) D-cache에만 반응(159.96→304.82, ×1.91),
  ITCM 경로는 D-cache에 **전혀 반응하지 않고**(492.10 동일) ART에만 반응(336.79→492.10, ×1.46).
- **Flash 단일 뱅크 8섹터** — 문서 [B] 추정 → 실측: 옵션 바이트 `nWRP0`~`nWRP7` 8개 존재로 섹터 8개 확인.
- **BOOT_ADDx 인코딩 = 값 × 0x4000** — 문서 [D] 역산 → 실측: `FLASH_OPTCR1 = 0x00400080`,
  `BOOT_ADD0 = 0x0080 → 0x00200000`, `BOOT_ADD1 = 0x0040 → 0x00100000`. 문서의 HAL 상수표와 정확히 일치.
  이 값이 실측 `VTOR = 0x00200000`의 원인이라는 것까지 확인.
- **`FLASH_OPTCR` 비트 필드(§8.3)** — 실측 `FLASH_OPTCR = 0xC0FFAAFD`: RDP[15:8]=0xAA(Level 0),
  nWRP[23:16]=0xFF(보호 없음), IWDG_STOP/STDBY=1. 문서 비트맵대로 디코드된다.
- **`FLASH_ACR` 비트 필드(§8.3)** — 실측 `0x00000006`(LATENCY=6, PRFTEN=0, ARTEN=0),
  리셋 직후는 `0x00000000`. 필드 배치 확인됨.
- **CM7_AHBSCR(`0xE000EFA0`) 비트 정의(§4.4)** — 실측 `0x00000800`
  → CTL[1:0]=`0b00`(AHBS 강등 = **CPU 우선**, 리셋값), TPRI=0, INITCOUNT[15:11]=`0b00001`.
  문서가 적은 리셋값과 정확히 일치.
- **DTCM이 32-bit 포트 2개라는 구조(§4.1)** — 실측: DTCM `copyLDM` = **7.05 byte/cycle**(1409 MB/s).
  32-bit 단일 포트의 상한 4 B/cyc를 넘으므로 **포트가 2개라는 문서 서술이 성능으로 확인**된다.
  같은 측정에서 ITCM(64-bit 단일 포트)은 6.63 B/cyc, SRAM1(32-bit 슬레이브)은 3.36 B/cyc로
  각 포트 폭의 상한 안에 정확히 들어온다.
- **BKPSRAM 4 KB @ `0x40024000`** — 실측: 4 KB에서 폴트. 다만 **읽기가 유독 느리다**(140 MB/s,
  다른 내부 RAM의 2/3). 문서에는 이 특성이 없다.
- **System memory 부트로더 실재** — 문서 `0x1FF00000` → 실측: 초기 SP `0x20003AF8`,
  진입점 `0x1FF00C88`을 직접 읽음. SP가 DTCM 선두 16 KB 안이라는 점도 AN2606 기술과 일치.
- **Device ID / Rev / Flash 용량 / 팩토리 영역 주소** — DEV_ID `0x449`, REV_ID `0x1001`(Rev Z),
  Flash size reg(`0x1FF0F442`) = `0x0400` = 1024 KB, UID `0x1FF0F420`,
  VREFIN_CAL/TS_CAL1/TS_CAL2 = `0x1FF0F44A`/`44C`/`44E`. §7 표와 전부 일치.
- **보드 SDRAM 8 MB / 16-bit 배선** — 문서 "MT48LC4M32B2, 하위 16 bit만 배선 → 8 MB"
  → 실측 `FMC_SDCR1 = 0x00001954`: MWID=16 bit, NR=12행, NC=8열, NB=4뱅크 → **정확히 8 MB**,
  경계 스캔도 8192 KB에서 폴트. 부품 추정의 근거가 레지스터로 뒷받침된다.
- **QSPI 용량 16 MB** — 실측 `QUADSPI_DCR` FSIZE=23 → 2²⁴ = 16 MB, 경계 스캔 16384 KB 일치.
  (**부품명은 다르다 — 아래 참조**)

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

1. **동작 주파수 216 MHz**
   - 문서: 머리말·§4.2·§5 전체가 "216 MHz"를 기준 동작점으로 삼는다.
   - 실측: **200 MHz**(PLLN=400). PCLK1 50 MHz, PCLK2 100 MHz.
   - 원인: 문서가 ST 공식 예제/데이터시트 상한을 기준으로 썼고, 이 프로젝트 펌웨어는 200 MHz로
     설정되어 있다. 문서 오류가 아니라 **기준점 불일치**지만, §5의 모든 절대 대역폭 수치가
     7.4 % 과대평가된다.
2. **Flash 7 wait state**
   - 문서: 요약·§3 "216 MHz / VDD 2.7–3.6 V에서 7 WS".
   - 실측: `FLASH_ACR.LATENCY = 6`(200 MHz 구간의 올바른 값). 리셋 직후는 0 WS.
   - 원인: 동작점 차이. 문서의 표 자체는 옳다.
3. **"ART 또는 L1 캐시로 실효 0-WS에 근접"**
   - 문서: 요약 3번째 항목, §3.
   - 실측: 이 보드는 **ART가 꺼진 채로**(ARTEN=0, PRFTEN=0) ITCM 별칭에서 실행 중이었다.
     즉 실행 경로에 아무 가속도 걸려 있지 않았다. 게다가 ART를 켜도 ITCM 경로 플래시 읽기는
     **2.46 byte/cycle**이 상한이라 0-WS 등가(문서 §5의 4 byte/cycle)에 못 미친다.
   - 원인: 문서는 ST가 광고하는 "0-wait 등가"를 그대로 옮겼고, 실제 실리콘은 그보다 낮다.
     또한 소프트웨어가 ART를 켜지 않으면 그 서술은 아예 성립하지 않는다.
4. **§5 이론 대역폭 표 전반**
   - 문서: AXIM/ITCM 1728 MB/s, DTCM 1728 MB/s, SRAM 864 MB/s, Flash 시퀀셜 864 MB/s(4 B/cyc).
   - 실측(200 MHz, 최댓값): DTCM **1409 MB/s**(7.05 B/cyc), ITCM 1326 MB/s(6.63),
     SRAM1 쓰기 709 MB/s(3.55)·읽기 206 MB/s(1.03), Flash 읽기 최고 **492 MB/s**(2.46 B/cyc, ITCM+ART),
     AXIM+D-cache 305 MB/s(1.52).
   - 원인: 클럭 차이(216→200)만으로는 설명되지 않는다. 문서 수치는 **버스폭×클럭의 순수 상한**이고
     실제로는 로드/스토어 명령 발행률·wait state·중재가 지배한다. 문서도 [D] 이론값이라 명시했으나,
     **읽기 대역폭은 이론값의 1/8까지 떨어질 수 있다**는 점이 새로 확인됐다.
5. **SDRAM 클럭 108 MHz / 대역폭 216 MB/s**
   - 문서: §5 "SDCLK = HCLK/2 = 108 MHz → 216 MB/s".
   - 실측: HCLK가 200 MHz이므로 **SDCLK = 100 MHz**, 이론 상한 200 MB/s.
     실효는 **읽기 60 MB/s / 쓰기 195 MB/s**로 읽기가 이론의 30 %에 불과하다.
     랜덤 접근 레이턴시는 21~26 사이클.
   - 원인: 클럭 차이 + 읽기 경로(CAS 2, RPIPE 0, 16-bit 버스)의 실제 오버헤드.
6. **QSPI 54 MHz DDR / 54 MB/s**
   - 문서: §5 "AN4667 벤치마크 조건 54 MHz DDR → 54 MB/s".
   - 실측: 이 보드는 `QUADSPI_CR.PRESCALER=1` → **100 MHz SDR**, `CCR.DDRM=0`, 명령 0xEB(Quad I/O),
     더미 10사이클. 이론 상한 50 MB/s, 실측 순차 읽기 **49.97 MB/s**.
   - 원인: 문서가 AN4667의 벤치마크 조건을 이 보드의 조건인 양 옮겼다. 숫자가 우연히 비슷할 뿐
     클럭·전송 모드가 다르다.
7. **QSPI 부품이 Micron N25Q128A**
   - 문서: §10 "MICRON N25Q128A13EF840E".
   - 실측: **JEDEC ID `0xEF4018` = Winbond W25Q128 계열**(Micron이면 `0x20BA18`이어야 함).
   - 원인: **보드 리비전에 따른 부품 교체**. 문서 §10 각주가 이미 CubeIDE에 W25Q128JV용 외부 로더가
     함께 존재한다고 적었는데, 이 개체가 바로 그쪽이다. 용량 16 MB는 양쪽 다 같다.
     소거 시간·페이지 크기·상태 레지스터가 다르므로 **QSPI에 쓰는 코드는 Winbond 기준으로 검증할 것**.
8. **패키지 데이터 레지스터 — 이번엔 문서가 맞고 실측 노트가 틀렸다**
   - 문서: §7 "F74x/F75x의 `PACKAGE_BASE`는 `0x1FF0F7E0`이며 `0x1FF07BF0`은 F72x/F73x용".
   - 실측 노트(`measured_facts.md` §1): "패키지 데이터 레지스터 `0x1FF07BF0` = `0xFFFFFFFF`, 해석 불가".
   - **재검증**: 같은 덤프의 `regions_*/factory_uid_area.bin`에서 문서가 지목한 `0x1FF0F7E0`을 읽으면
     **`0xFC72`**(리셋 덤프·핫플러그 덤프 모두 동일)이고, PKG 필드(mask `0x0700`)가 `0x400`이라
     문서 §7의 코드표에서 **LQFP176/LQFP208/TFBGA216 등급**을 가리킨다. 실제 부품은 TFBGA216이다.
   - 결론: **실측이 잘못된 주소를 읽은 것**이다. 문서 §7의 주소가 옳다.
     (`0x1FF07BF0`이 `0xFFFFFFFF`인 것은 그곳이 F74x의 패키지 레지스터가 아니기 때문이다.)

### 아직 측정하지 않음

- **Bit-banding 미구현(§1.1 주의)** — `0x22000000` / `0x42000000`에 읽기를 시도해 BusFault가 나는지 확인.
  (BusFault 프로빙 하네스가 이미 프로브 펌웨어에 있으므로 주소만 추가하면 된다.)
- **DMA의 ITCM 접근 불가 / DTCM 접근은 AHBS 경유(§4.3)** — DMA2 memory-to-memory로
  `0x00000000`(ITCM)과 `0x20000000`(DTCM)에 각각 전송을 걸어 전송 완료/전송 에러 플래그를 비교.
- **AHBS 경유 DMA 처리량과 CM7_AHBSCR.CTL의 영향(§4.4)** — CPU가 DTCM을 연속 접근하는 동안
  DMA2 mem2mem으로 DTCM을 읽으며 CTL을 `0b00`↔`0b01`로 바꿔 전송 시간을 비교.
- **DMA1의 memory-to-memory 불가(§4.3 마지막 항목, [C])** — DMA1 스트림에 M2M 모드를 설정하고
  `HAL_DMA_Init()` 반환값 / DMA_LISR 에러 플래그를 확인.
- **FMC 스왑(`SYSCFG_MEMRMP.SWP_FMC`)** — 스왑 후 `0x60000000`에서 SDRAM이 읽히는지 확인.
  (실측 시점의 `SYSCFG_MEMRMP = 0x00000000`, MEM_BOOT=0으로 스왑 없음만 확인됨.)
- **System memory 부트로더의 실제 끝 주소(`0x1FF0EDBF`)** — 덤프가 `0x1FF07FFF`까지만 떠 있다.
  `0x1FF00000`~`0x1FF0FFFF`를 전부 읽어 코드가 끝나는 지점과 폴트 경계를 확인.
- **Flash 프로그래밍/소거 시간(§9)** — 백업 후 한 섹터를 지우고 다시 쓰며 DWT_CYCCNT로 실측
  (typ 200 ms / 워드 16 µs 검증). 플래시 내구성·데이터 보존(§9.4)은 원리적으로 측정 불가.
- **ITCM-RAM에 코드를 올렸을 때의 실행 속도(§3 AN4667 Table 7)** — 코드 배치별 실행 속도는
  이번 측정 범위 밖이다(데이터 접근만 측정). 같은 커널을 `.itcm_code`/`.data`/플래시에 각각 배치해
  DWT_CYCCNT로 비교하면 된다.

---

## 요약

- 내부 RAM은 **4개의 물리적으로 분리된 블록**이다: ITCM-RAM 16 KB(`0x00000000`), DTCM-RAM 64 KB(`0x20000000`), SRAM1 240 KB(`0x20010000`), SRAM2 16 KB(`0x2004C000`). "320 KB SRAM"은 DTCM+SRAM1+SRAM2 = 64+240+16 KB이며 ITCM-RAM 16 KB는 별도다. [A]
- Flash 1 MB는 **두 주소에 이중 매핑**된다: ITCM 별칭 `0x00200000–0x002FFFFF`(읽기/실행 전용, ART 적용)와 AXIM `0x08000000–0x080FFFFF`(읽기/쓰기, L1 캐시 적용). 물리적으로 동일한 어레이다. [A]
- 216 MHz / VDD 2.7–3.6 V에서 Flash는 **7 wait state**가 필요하다. ART(ITCM 경로) 또는 L1 I/D-cache(AXIM 경로)로 실효 0-WS에 근접시킨다. [A]

  > **실측 정정**: 이 보드는 200 MHz라 `FLASH_ACR.LATENCY = 6`(6 WS)이다. 그리고 "실효 0-WS에 근접"은
  > 이 보드에서 성립하지 않았다 — ① 펌웨어가 **ART를 켜지 않은 채**(ARTEN=0, PRFTEN=0) ITCM 별칭에서
  > 실행 중이었고, ② ART를 켜도 ITCM 경로 플래시 읽기는 **2.46 byte/cycle**이 상한이다
  > (§5가 0-WS 등가로 계산한 4 byte/cycle의 60 %). ART ON/OFF 차이는 ×1.46(336.79 → 492.10 MB/s)이다.
- **DMA는 DTCM에 접근할 수 있다.** 단 Cortex-M7의 전용 **AHBS(32-bit AHB slave) 포트**를 경유해야 하며, **ITCM-RAM에는 DMA 접근이 불가능**하다(AHBS에 ITCM이 노출되지 않음 — ST 구현 제약). [A]
- 버스 매트릭스는 216 MHz, **마스터 12개 × 슬레이브 8개**, round-robin 중재. [A]
- 이론 대역폭(216 MHz): 64-bit 포트(AXIM/ITCM/Flash-AHB) = **1728 MB/s**, 32-bit AHB 레이어·AHBS = **864 MB/s**, DTCM(32-bit×2 포트) = 합계 **1728 MB/s**, APB2(108 MHz) = 432 MB/s, APB1(54 MHz) = 216 MB/s. [D]

  > **실측 정정**: 200 MHz 기준으로 환산해도(×0.926) 실측 최댓값은 한참 아래다.
  > DTCM **1409 MB/s**(7.05 B/cyc), ITCM 1326 MB/s(6.63), SRAM1 쓰기 709 MB/s(3.55)·**읽기 206 MB/s**(1.03),
  > 플래시 읽기 최고 492 MB/s(2.46). **포트 폭 상한(B/cyc)은 넘지 않으므로 §4.1의 버스 구조 서술은 옳고,
  > 절대 대역폭만 과대**하다. 특히 DTCM 7.05 B/cyc는 32-bit 단일 포트 상한(4 B/cyc)을 넘으므로
  > "DTCM = 32-bit × 2 포트"라는 문서 서술을 성능으로 입증한다.
- 부트는 **BOOT 핀 1개 + BOOT_ADD0/BOOT_ADD1 옵션 바이트**로 `0x00000000`–`0x3FFFFFFF` 임의 주소 지정. (F4의 BOOT1 핀은 F7에 없다.) 시스템 메모리 부트로더는 `0x1FF00000–0x1FF0EDBF`(60,864 B ≒ 60 KB). [A]
- Flash는 **단일 뱅크 8섹터**(32 KB×4 + 128 KB×1 + 256 KB×3 = 1024 KB). STM32F746는 dual-bank를 지원하지 않는다. [B]
- Flash 내구성 10 kcycles, 보존 20년(10 kcycles @55 °C) / 30년(1 kcycle @85 °C). 워드 프로그래밍 typ 16 µs, 256 KB 섹터 소거 typ 1 s(PSIZE=x32), mass erase typ 4 s(PSIZE=x32). [A]
- Cortex-M7은 **bit-banding을 지원하지 않는다**(F4에서 이식 시 주의). [A]

---

## 1. 전체 메모리 맵

### 1.1 최상위 블록 맵

| 시작 주소 | 끝 주소 | 크기 | 이름 | 버스 | 비고 | 등급/출처 |
|---|---|---|---|---|---|---|
| `0x00000000` | `0x00003FFF` | 16 KB | ITCM-RAM | ITCM (64-bit) | CPU 전용, 0-latency, DMA 불가 | [A] AN4667 Table 3 |
| `0x00004000` | `0x000FFFFF` | — | Reserved | — | | [D] 계산 |
| `0x00100000` | — | — | System memory 부트 별칭 | ITCM | `BOOT_ADDx=0x0040` 시 부트 진입점 | [B] HAL `OB_BOOTADDR_SYSTEM` |
| `0x00200000` | `0x002FFFFF` | 1 MB | **FLASH-ITCM** (Flash 별칭) | ITCM (64-bit) | 실행/상수 읽기 전용, **ART 적용**, 쓰기 불가 | [A] AN4667 Table 3 |
| `0x00300000` | `0x07FFFFFF` | — | Reserved | — | | [D] |
| `0x08000000` | `0x080FFFFF` | 1 MB | **FLASH-AXIM** | AXI→AHB (64-bit) / DMA는 32-bit AHB | 읽기·쓰기·실행, **L1 캐시 적용** | [A] AN4667 Table 3 |
| `0x08100000` | `0x1FEFFFFF` | — | Reserved | — | | [D] |
| `0x1FF00000` | `0x1FF0EDBF` | 60,864 B | System memory (bootloader) | — | ST 공장 프로그램, 읽기 전용 | [A] AN2606 Rev 61 Table 87 |
| `0x1FF0EDC0` | `0x1FF0EFFF` | 576 B | Reserved | — | | [D] 계산 |
| `0x1FF0F000` | `0x1FF0F41F` | 1056 B | OTP 영역 (1024 B + lock 32 B) | — | `FLASH_OTP_BASE`/`FLASH_OTP_END` | [B] CMSIS `stm32f746xx.h` |
| `0x1FF0F420` | `0x1FF0F42B` | 12 B (96 bit) | Unique device ID | — | `UID_BASE` | [B] CMSIS |
| `0x1FF0F442` | `0x1FF0F443` | 2 B | Flash size register (KB 단위) | — | `FLASHSIZE_BASE` | [B] CMSIS |
| `0x1FF0F44A` | `0x1FF0F44B` | 2 B | VREFIN_CAL | — | 30 °C, VDDA=3.3 V | [A] DS10916 |
| `0x1FF0F44C` | `0x1FF0F44F` | 4 B | TS_CAL1 / TS_CAL2 | — | 30 °C / 110 °C | [A] DS10916 |
| `0x1FF0F7E0` | `0x1FF0F7E1` | 2 B | Package data register | — | `PACKAGE_BASE` (※ `0x1FF07BF0`이 **아님**) | [B] CMSIS + [C] ST 커뮤니티 |
| `0x1FFF0000` | `0x1FFF001F` | 32 B | Option bytes | — | OPTCR/OPTCR1 미러 | [C] DS10916 Figure 19 (폰트 인코딩으로 숫자 일부 추정) |
| `0x20000000` | `0x2000FFFF` | 64 KB | **DTCM-RAM** | DTCM (32-bit ×2) | CPU 0-latency, DMA는 AHBS 경유 | [A] AN4667 Table 3 |
| `0x20010000` | `0x2004BFFF` | 240 KB | **SRAM1** | AHB (32-bit) | 모든 AHB 마스터 접근 가능 | [A] AN4667 Table 3 |
| `0x2004C000` | `0x2004FFFF` | 16 KB | **SRAM2** | AHB (32-bit) | 모든 AHB 마스터 접근 가능 | [A] AN4667 Table 3 |
| `0x20050000` | `0x3FFFFFFF` | — | Reserved | — | | [D] |
| `0x40000000` | `0x40007FFF` | 32 KB | APB1 주변장치 | APB1 (max 54 MHz) | | [A] DS10916 Table 13 |
| `0x40010000` | `0x40016BFF` | — | APB2 주변장치 | APB2 (max 108 MHz) | | [A] DS10916 Table 13 |
| `0x40020000` | `0x4007FFFF` | — | AHB1 주변장치 | AHB1 (216 MHz) | GPIO/DMA/ETH/USB_HS/DMA2D 등 | [A] DS10916 Table 13 |
| `0x40024000` | `0x40024FFF` | 4 KB | BKPSRAM (백업 SRAM) | AHB1 | VBAT 유지, 저전력 모드 보존 | [A] DS10916 Table 13 |
| `0x50000000` | `0x50060BFF` | — | AHB2 주변장치 | AHB2 | USB_OTG_FS / DCMI / RNG | [A] DS10916 Table 13 |
| `0x60000000` | `0x6FFFFFFF` | 256 MB | FMC bank 1 (NOR/PSRAM) | AHB3 | | [A] DS10916 Table 13 |
| `0x70000000` | `0x7FFFFFFF` | 256 MB | FMC bank 2 | AHB3 | | [A] DS10916 Table 13 |
| `0x80000000` | `0x8FFFFFFF` | 256 MB | FMC bank 3 (NAND) | AHB3 | | [A] DS10916 Table 13 |
| `0x90000000` | `0x9FFFFFFF` | 256 MB | **Quad-SPI 메모리 매핑** | AHB3 (AXI 경유) | | [A] DS10916 Table 13 |
| `0xA0000000` | `0xA0000FFF` | 4 KB | FMC 제어 레지스터 | AHB3 | | [A] DS10916 Table 13 |
| `0xA0001000` | `0xA0001FFF` | 4 KB | QUADSPI 제어 레지스터 | AHB3 | | [A] DS10916 Table 13 |
| `0xA0002000` | `0xBFFFFFFF` | — | Reserved | — | | [A] DS10916 Table 13 |
| `0xC0000000` | `0xCFFFFFFF` | 256 MB | FMC bank 5 (SDRAM bank1) | AHB3 | DISCO 보드 SDRAM 위치 | [A] DS10916 Table 13 |
| `0xD0000000` | `0xDFFFFFFF` | 256 MB | FMC bank 6 (SDRAM bank2) | AHB3 | | [A] DS10916 Table 13 |
| `0xE0000000` | `0xE00FFFFF` | 1 MB | Cortex-M7 내부 주변장치 (PPB) | Cortex-M7 전용 | NVIC/SCB/DWT/ITM/MPU/CM7_AHBSCR | [A] DS10916 Table 13 |
| `0xE0100000` | `0xFFFFFFFF` | — | Reserved | — | | [A] DS10916 Table 13 |

> **주의 (bit-banding):** Cortex-M7은 bit-band 별칭 영역(`0x22000000`, `0x42000000`)을 구현하지 않는다. STM32F4 코드 이식 시 bit-band 의존 코드는 동작하지 않는다. [A] ARM DDI 0489F §1.4 "Code designed for the Cortex-M3 and Cortex-M4 processors is compatible with the Cortex-M7 processor as long as it does not rely on bit-banding."

> **FMC 스왑:** `SYSCFG_MEMRMP.SWP_FMC[1:0]`로 FMC NOR/PSRAM(`0x60000000`)과 SDRAM(`0xC0000000`) 영역을 서로 교환할 수 있다. 기본값은 SDRAM = `0xC0000000`. [B] `stm32f7xx_hal.h` `__HAL_SYSCFG_REMAPMEMORY_FMC()` / `_FMC_SDRAM()`

### 1.2 주변장치 레지스터 경계 주소 (버스별, 전체) [A] DS10916 Table 13

**AHB3**

| 주소 범위 | 주변장치 |
|---|---|
| `0xD0000000 – 0xDFFFFFFF` | FMC bank 6 |
| `0xC0000000 – 0xCFFFFFFF` | FMC bank 5 |
| `0xA0001000 – 0xA0001FFF` | Quad-SPI 제어 레지스터 |
| `0xA0000000 – 0xA0000FFF` | FMC 제어 레지스터 |
| `0x90000000 – 0x9FFFFFFF` | Quad-SPI |
| `0x80000000 – 0x8FFFFFFF` | FMC bank 3 |
| `0x70000000 – 0x7FFFFFFF` | FMC bank 2 |
| `0x60000000 – 0x6FFFFFFF` | FMC bank 1 |

**AHB2**

| 주소 범위 | 주변장치 |
|---|---|
| `0x50060800 – 0x50060BFF` | RNG |
| `0x50050000 – 0x500503FF` | DCMI |
| `0x50000000 – 0x5003FFFF` | USB OTG FS |

**AHB1**

| 주소 범위 | 주변장치 |
|---|---|
| `0x40040000 – 0x4007FFFF` | USB OTG HS |
| `0x4002B000 – 0x4002BBFF` | Chrom-ART (DMA2D) |
| `0x40028000 – 0x400293FF` | ETHERNET MAC (5개 서브 블록) |
| `0x40026400 – 0x400267FF` | DMA2 |
| `0x40026000 – 0x400263FF` | DMA1 |
| `0x40024000 – 0x40024FFF` | BKPSRAM |
| `0x40023C00 – 0x40023FFF` | Flash interface register (FLASH_ACR/KEYR/OPTCR…) |
| `0x40023800 – 0x40023BFF` | RCC |
| `0x40023000 – 0x400233FF` | CRC |
| `0x40020000 – 0x40022BFF` | GPIOA…GPIOK (각 1 KB) |

**APB2** (`0x40010000 – 0x40016BFF`): TIM1, TIM8, USART1, USART6, ADC1/2/3, SDMMC, SPI1/I2S1, SPI4, SYSCFG, EXTI, TIM9/10/11, SPI5, SPI6, SAI1, SAI2, LCD-TFT(`0x40016800`)

**APB1** (`0x40000000 – 0x40007FFF`): TIM2…TIM7, TIM12/13/14, LPTIM1, RTC&BKP, WWDG, IWDG, SPI2/I2S2, SPI3/I2S3, SPDIFRX, USART2/3, UART4/5/7/8, I2C1…I2C4, CAN1/2, HDMI-CEC, PWR, DAC

---

## 2. 내부 SRAM 분할 상세

| 블록 | 시작 | 끝 | 크기 | 붙는 버스 | 접근 가능 마스터 | 접근 폭 | 등급 |
|---|---|---|---|---|---|---|---|
| ITCM-RAM | `0x00000000` | `0x00003FFF` | 16 KB (`0x4000`) | ITCM (64-bit) | **CPU만** | 8/16/32/64-bit | [A] |
| DTCM-RAM | `0x20000000` | `0x2000FFFF` | 64 KB (`0x10000`) | DTCM (32-bit ×2: D0TCM/D1TCM) | CPU(DTCM 포트) + 모든 AHB 마스터(**AHBS 경유**) | 8/16/32/64-bit | [A] |
| SRAM1 | `0x20010000` | `0x2004BFFF` | 240 KB (`0x3C000`) | 32-bit AHB (버스 매트릭스 슬레이브) | 모든 AHB 마스터 | 8/16/32-bit | [A] |
| SRAM2 | `0x2004C000` | `0x2004FFFF` | 16 KB (`0x4000`) | 32-bit AHB (별도 슬레이브 포트) | 모든 AHB 마스터 | 8/16/32-bit | [A] |
| BKPSRAM | `0x40024000` | `0x40024FFF` | 4 KB | AHB1 | AHB 마스터 | 8/16/32-bit | [A] |

출처: AN4667 Rev 4 Table 3(`internal memory summary of the STM32F74xxx/STM32F75xxx devices`), §1.5.2 Embedded SRAM; DS10916 Rev 4 Table 13 / Figure 19. CMSIS `stm32f746xx.h`의 `SRAM1_BASE = 0x20010000UL /* 240KB */`, `SRAM2_BASE = 0x2004C000UL /* 16KB */`, `RAMITCM_BASE = 0x00000000UL /* 16KB */`와 일치. [A][B]

**연속성 검증(계산 [D]):** `0x20000000 + 0x10000 = 0x20010000`(DTCM→SRAM1), `0x20010000 + 0x3C000 = 0x2004C000`(SRAM1→SRAM2), `0x2004C000 + 0x4000 = 0x20050000`(SRAM2 끝). 따라서 `0x20000000–0x2004FFFF`가 물리적으로 빈틈 없는 320 KB 연속 영역이다. 프로젝트 링커 스크립트(`ImageClassification/STM32F746NGHX_FLASH.ld`)도 `RAM (xrw) : ORIGIN = 0x20000000, LENGTH = 320K`로 하나의 영역으로 선언한다. [B]

> **실측 확인**: 이 "주의"는 실측으로 정확히 뒷받침된다. D-cache OFF·32 B 스트라이드 읽기의 접근당 사이클이
> `0x20000000`~`0x2000FFFF`(DTCM) **5.16 cyc**, `0x20010000`~`0x2004FFFF`(SRAM1+SRAM2) **15.66 cyc**,
> `0x20050000`부터 버스 폴트다. **경계가 성능 곡선에 그대로 나타난다.** SRAM1과 SRAM2는 지연이 동일해
> 성능상 구분되지 않는다. 다만 표의 "ITCM/DTCM 0-latency"는 문자 그대로는 아니다 —
> 루프 오버헤드 2.00 cyc을 빼도 TCM 접근에 약 3 cyc이 든다(로드-사용 지연). AXI SRAM 대비 3배 빠를 뿐이다.

> 주의: 주소는 연속이지만 **성능·접근 특성은 연속이 아니다.** `0x20000000–0x2000FFFF`(DTCM 64 KB)는 CPU에서 0-wait, 캐시 비대상(TCM은 L1 캐시를 거치지 않음)이며, `0x20010000` 이상(SRAM1/2)은 AXI/AHB를 거치고 D-cache 대상이다. 링커가 320 KB를 하나로 보고 배치하면 버퍼가 이 경계를 넘어 걸칠 수 있으므로, DMA 버퍼/캐시 유지보수가 필요한 데이터는 명시적으로 섹션 분리하는 것이 안전하다. [D] (근거: AN4667 §1.5.2 + AN4839 캐시 정책 설명)

---

## 3. ITCM-Flash 별칭(`0x00200000`) vs AXIM-Flash(`0x08000000`)

| 항목 | FLASH-ITCM `0x00200000` | FLASH-AXIM `0x08000000` | 등급 |
|---|---|---|---|
| 물리 어레이 | 동일한 1 MB Flash | 동일한 1 MB Flash | [A] |
| 버스 | 64-bit ITCM | AXI→AHB 64-bit (CPU), 32-bit AHB (DMA) | [A] |
| 용도 | 코드 실행 + 상수 읽기 | 코드 실행 + 읽기 + **쓰기/프로그래밍** | [A] |
| 쓰기 | **불가** ("The write access to the Flash memory is not permitted via this bus") | 가능 | [A] |
| 가속 | **ART Accelerator** (`FLASH_ACR.ARTEN`, bit 9) + ART-Prefetch (`FLASH_ACR.PRFTEN`, bit 8) | **L1 I-cache / D-cache** (`SCB_CCR`, CMSIS `SCB_EnableICache()/EnableDCache()`) | [A] |
| ART 구조 (F74x/F75x) | 256-bit × 64 line 통합 명령/분기 캐시 | 해당 없음 | [A] |
| DMA 접근 | 불가(모든 전송은 AHB로 강제) — DMA는 `0x08000000`에서 읽어야 함 | 가능 (32-bit AHB) | [A] |
| 캐시 대상 | ART가 담당 (L1 캐시 미경유) | L1 I/D-cache 대상 | [A] |
| Flash 어레이 읽기 폭 | 256-bit (F74x/F75x) | 256-bit | [A] |

> **실측 확인 (이 표의 "가속" 행)**: 위 표에서 가장 중요한 주장 —
> "ITCM 별칭은 ART가, AXIM은 L1 캐시가 담당하며 서로 관여하지 않는다" — 가 실측으로 완전히 확인됐다.
> 측정 루프를 `.itcm_code`에 올려 코드 페치 경로를 고정한 뒤 데이터 읽기만 잰 결과(워킹셋 64 KB):
>
> | 경로 | D-cache | ART | 순차 읽기 |
> |---|---|---|---|
> | AXIM `0x08000000` | OFF | ON | 159.96 MB/s |
> | AXIM | OFF | OFF | 159.96 MB/s ← **ART 무반응** |
> | AXIM | WB | ON | 304.82 MB/s ← **D-cache ×1.91** |
> | ITCM `0x00200000` | OFF | ON | 492.10 MB/s ← **ART ×1.46** |
> | ITCM | OFF | OFF | 336.79 MB/s |
> | ITCM | WB | ON | 492.10 MB/s ← **D-cache 무반응** |
>
> 또한 이 보드의 SNN 펌웨어는 **ART가 꺼진 채 ITCM 별칭에서 실행**되고 있었다
> (`FLASH_ACR = 0x00000006`, `VTOR = 0x00200000`). 원인은 `stm32f7xx_hal_conf.h`의
> `PREFETCH_ENABLE 0U` + `ART_ACCLERATOR_ENABLE` 매크로 미정의다.
> **ART를 켜는 것만으로 플래시 접근이 1.46배 빨라진다.**

**성능 실측 (AN4667 Rev 4 Table 7, MDK-ARM, FFT 데모, CPU 216 MHz, VDD 3.3 V, Flash 7 WS):** [A]

| 구성 | 활성 기능 | CPU 사이클 |
|---|---|---|
| 1-FlashITCM_rwRAM-DTCM | ART + ART-PF ON | **118,577** |
| 2-FlashITCM_rwSRAM1 | ART + ART-PF + D-cache ON | 135,296 |
| 3-FlashAXI_rwRAM-DTCM | I-cache + D-cache ON | **118,653** |
| 4-FlashAXI_rwSRAM1 | I-cache + D-cache ON | 145,917 |
| 5-RAMITCM_rwRAM-DTCM | (가속 불필요) | **112,428** (최속) |
| 6_1-QuadSPI_rwRAM-DTCM | I-cache + D-cache (상수도 QSPI) | 176,441 |
| 6_2-QuadSPI_rwRAM-DTCM | I-cache + ART + ART-PF (상수는 Flash-ITCM) | 126,900 |
| 7-ExtSDRAM-Swapped_rwDTCM | I-cache + D-cache (HCLK 200 MHz) | 128,398 |

해석 [D]:
- ITCM+ART(118,577)와 AXIM+I-cache(118,653) 차이는 **0.06 %**로 사실상 동등하다. "ITCM이 항상 빠르다"는 통념은 이 워크로드에서 성립하지 않는다.
- **데이터 배치가 코드 배치보다 영향이 크다.** 같은 ITCM 실행에서 데이터를 DTCM(118,577) 대신 SRAM1(135,296)에 두면 **+14.1 %** 느려진다. AXIM 실행에서는 118,653 → 145,917로 **+23.0 %**.
- 이론 최속은 ITCM-RAM 실행 + DTCM 데이터(112,428)로, ITCM+ART 대비 5.5 % 빠르다. 다만 ITCM-RAM은 16 KB뿐이라 SNN 커널 같은 소형 핫루프에만 적용 가능하다.

> **실측 상태(미검증)**: 위 표는 AN4667이 216 MHz에서 잰 **코드 배치별 실행 사이클**이다.
> 이번 측정은 데이터 접근 경로만 다뤘으므로 이 표의 값 자체는 **이 보드에서 검증되지 않았다**.
> 반증도 확증도 아니며, 검증하려면 같은 커널을 ITCM-RAM/Flash-ITCM/Flash-AXIM에 각각 배치해
> DWT_CYCCNT로 비교해야 한다.

**Wait state (VDD 2.7–3.6 V):** 216 MHz + over-drive ON에서 **7 WS**. VDD 2.4–2.7 V는 8 WS, VDD 2.1–2.4 V는 9 WS, VDD 1.7–2.1 V는 최대 180 MHz/8 WS. `FLASH_ACR.LATENCY[3:0]`(0–15)로 설정. [A] DS10916 Rev 4 Table 15 각주 2 / CMSIS `FLASH_ACR_LATENCY_Msk = 0xF`

---

## 4. 시스템 버스 매트릭스

### 4.1 Cortex-M7 (r0p1) 버스 인터페이스 5종 [A] AN4667 §1.3, ARM DDI 0489F

| 인터페이스 | 방향 | 폭 | 연결 대상 | 비고 |
|---|---|---|---|---|
| **AXIM** | 마스터 | **64-bit** (AMBA4 AXI) | AXI→Multi-AHB 브리지 → 버스 매트릭스 | L1 캐시 리필 경로. TCM/AHBP가 아닌 모든 접근 |
| **ITCM** | 마스터 | **64-bit** | ITCM-RAM 16 KB, FLASH-ITCM | 1 포트 |
| **DTCM** | 마스터 | **32-bit × 2** (D0TCM, D1TCM) | DTCM-RAM 64 KB | Harvard 구조의 데이터 측 |
| **AHBS** | **슬레이브** | **32-bit** (AMBA3 AHB-Lite) | 외부 마스터(DMA)가 TCM에 접근하는 통로 | ST 구현에서는 **DTCM만** 노출 |
| **AHBP** | 마스터 | **32-bit** | AHB1 주변장치 경로 | 주변장치 레지스터 접근 전용 |

AXIM에서 나오는 버스 구성: **1× 64-bit AHB(내장 Flash 전용) + 3× 32-bit AHB(버스 매트릭스)**. [A] AN4667 §1.3.1

### 4.2 마스터 12 / 슬레이브 8 [A] AN4667 §1.4

**마스터 (12)**
1. AXI→AHB 브리지 32-bit AHB #1
2. AXI→AHB 브리지 32-bit AHB #2
3. AXI→AHB 브리지 32-bit AHB #3
4. AXI→AHB 브리지 64-bit AHB (내장 Flash 전용)
5. Cortex-M7 AHBP (주변장치 버스)
6. DMA1 memory 버스
7. DMA2 memory 버스
8. DMA2 peripheral 버스
9. Ethernet DMA 버스
10. USB OTG HS DMA 버스
11. LCD-TFT (LTDC) DMA 버스
12. Chrom-ART (DMA2D) memory 버스

**슬레이브 (8)**
1. 내장 Flash (AHB 경로; 읽기/쓰기/실행/데이터)
2. **Cortex-M7 AHBS** (= DMA가 DTCM-RAM에 접근하는 유일한 경로)
3. SRAM1 (240 KB)
4. SRAM2 (16 KB)
5. AHB1 주변장치 (AHB→APB 브리지 포함, APB1/APB2 포함)
6. AHB2 주변장치
7. FMC 메모리 인터페이스
8. Quad-SPI 메모리 인터페이스

중재: **round-robin** 알고리즘, 버스 매트릭스 클럭 216 MHz (CPU와 동일). [A] AN4667 §1.4

### 4.3 마스터 × 슬레이브 접근 매트릭스

`O` = 접근 가능, `X` = 불가, `AHBS` = Cortex-M7 AHBS 슬레이브 포트 경유.

| 마스터 \ 슬레이브 | ITCM-RAM | DTCM-RAM | SRAM1 | SRAM2 | FLASH (AHB, `0x08000000`) | FLASH-ITCM (`0x00200000`) | AHB1/APB | AHB2 | FMC | QUADSPI | 등급 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cortex-M7 ITCM 포트 | **O** (직결) | X | X | X | X | **O** (직결) | X | X | X | X | [A] |
| Cortex-M7 DTCM 포트 | X | **O** (직결) | X | X | X | X | X | X | X | X | [A] |
| Cortex-M7 AXIM | X | X | O | O | O | X | O | O | O | O | [A] |
| Cortex-M7 AHBP | X | X | X | X | X | X | O | X | X | X | [A] |
| DMA1 (memory 버스) | **X** | O (AHBS) | O | O | O | X | O | O | O | O | [C]/[D] |
| DMA2 (memory + peripheral 버스) | **X** | O (AHBS) | O | O | O | X | O | O | O | O | [A] 마스터 목록 / [D] 조합 |
| Ethernet DMA | **X** | O (AHBS) | O | O | O | X | O | O | O | O | [A]/[D] |
| USB OTG HS DMA | **X** | O (AHBS) | O | O | O | X | O | O | O | O | [A]/[D] |
| LTDC DMA | **X** | O (AHBS) | O | O | O | X | O | O | O | O | [A]/[D] |
| DMA2D (Chrom-ART) | **X** | O (AHBS) | O | O | O | X | O | O | O | O | [A]/[D] |

근거 [A]: AN4667 §1.3.3 — "the AHBS allows only data transfer from/to DTCM-RAM. The ITCM bus is not accessible on AHBS, so **the DMA data transfer to/from ITCM RAM is not supported**. For DMA transfer to/from the Flash memory on ITCM interface, all the transfers are forced through AHB bus."
근거 [A]: AN4667 §1.5.2 — "The DTCM-RAM ... accessible by all AHB masters from the AHB bus Matrix: by the CPU through the DTCM bus and **by DMAs through the specific AHBS bus of the core**."
근거 [A]: AN4667 §1.5.2 — "The SRAM1/SRAM2 is accessible by all the AHB masters from the AHB bus Matrix."

**주의(정확한 표현):** "DMA는 DTCM에 접근 불가"는 STM32F7에서는 **틀린 서술**이다. 접근 가능하되 AHBS(32-bit) 경유이며, ITCM-RAM만 진짜로 불가하다. 다만 실무에서 문제가 되는 조건은 다음과 같다 [D]:
- AHBS는 32-bit 단일 포트이므로 DTCM에 대한 DMA 처리량이 CPU 경로(64-bit급)보다 낮다.
- Cortex-M7 TRM: "**Loopback of transactions from the AXIM or AHBP interface onto the AHBS is not supported because it might cause deadlock**" — 즉 CPU가 트리거한 트랜잭션이 다시 AHBS로 되돌아오는 구성은 금지된다. [A] ARM DDI 0489F §5.7.2
- AHBS 접근은 **MPU 검사를 수행하지 않으며**, privileged/unprivileged 구분도 하지 않는다. DTCM에 대한 DMA 접근에는 MPU 보호가 적용되지 않는다. [A] ARM DDI 0489F §5.7.2
- AHBS는 exclusive/locked 접근을 지원하지 않으므로 DTCM을 CPU-DMA 간 락 기반 공유 자료구조로 쓸 수 없다. [A] ARM DDI 0489F §5.7.2
- memory-to-memory DMA는 **DMA2에서만** 가능하다(DMA1 불가). [C] ST 커뮤니티/RM0385 DMA 장 — RM0385 원문 재확인 필요.

### 4.4 AHBS 메모리 맵과 CPU/DMA 우선순위 제어

**AHBS 메모리 맵** (ARM DDI 0489F Table 5-37) [A]

| 시작 | 끝 | HADDRS[2] | 접근되는 TCM |
|---|---|---|---|
| `0x00000000` | `0x00000000 + <ITCM size>` | — | ITCM |
| `0x20000000` | `0x20000000 + <DTCM size>` | 0 | D0TCM |
| `0x20000000` | `0x20000000 + <DTCM size>` | 1 | D1TCM |

이 범위 밖의 주소는 AHBS에서 에러 응답을 반환한다. STM32F7에서는 ITCM 구간이 AHBS에 연결되어 있지 않다(ST 구현). [A]

**CM7_AHBSCR (`0xE000EFA0`)** — DTCM에 대한 CPU vs DMA 우선순위 조정 [A] ARM DDI 0489F Table 3-15

| 비트 | 이름 | 설명 |
|---|---|---|
| [31:16] | — | Reserved |
| [15:11] | INITCOUNT | Fairness counter 초기값. 리셋값 `0b01`. `0b00` 설정 금지(livelock 위험). CTL=`0b11`이면 미사용 |
| [10:2] | TPRI | AHBS 트래픽 강등 임계 실행 우선순위. NVIC 인코딩과 동일. `0b11111111`=우선순위 −1(HardFault), `0b11111110`=−2(NMI) |
| [1:0] | CTL | `00`=AHBS 우선순위 강등(리셋값), `01`=소프트웨어(CPU) 우선순위 강등, `10`=실행 우선순위가 TPRI 이상일 때 AHBS 강등, `11`=AHBSPRI 신호가 제어 |

리셋 기본값 CTL=`0b00`이므로 **기본적으로 CPU가 DTCM 접근에서 DMA보다 우선**한다. 즉 CPU가 DTCM을 바쁘게 쓰는 동안 DMA의 DTCM 접근은 지연될 수 있다. [A]+[D]

또한 AHBS 쓰기는 Store Queue에 버퍼링되고 항상 즉시 OK 응답을 반환하며, 이후 중단되면 `WABORTS` 비동기 abort가 발생한다. 읽기는 버퍼된 쓰기와 데이터 의존성이 있으면 SQ가 드레인될 때까지 스톨한다(바이트 단위 해저드 검사). [A] ARM DDI 0489F §5.7

---

## 5. 이론 대역폭 (216 MHz 기준)

> **실측 정정 (이 절 전체에 적용)**: ① 이 보드의 HCLK는 **200 MHz**이므로 아래 모든 값에 0.926을 곱해야 한다.
> ② 그렇게 보정해도 실측 최댓값은 이론값보다 훨씬 낮다. 200 MHz 기준 실측(전부 D-cache ON, 인라인 asm):
>
> | 경로 | 문서 이론(216 MHz) | 200 MHz 환산 | **실측 최댓값** | 실측 B/cyc |
> |---|---|---|---|---|
> | DTCM | 1728 MB/s | 1600 | **1409 MB/s** (copyLDM) | 7.05 |
> | ITCM | 1728 | 1600 | **1326** (copyLDM) | 6.63 |
> | SRAM1 | 864 | 800 | **709** 쓰기 / **206** 읽기 | 3.55 / 1.03 |
> | SRAM2 | 864 | 800 | 694 쓰기 / 205 읽기 | 3.47 / 1.03 |
> | Flash (ITCM+ART) | 1728 | 1600 | **492** | 2.46 |
> | Flash (AXIM+D-cache) | 1728 | 1600 | **305** | 1.52 |
> | BKPSRAM | — | — | 140 읽기 / 470 쓰기 | 0.70 / 2.35 |
>
> **B/cyc 열은 어느 것도 포트 폭 상한을 넘지 않는다** → §4.1의 버스 구조 서술은 옳다.
> 어긋나는 것은 절대 대역폭뿐이며, 특히 **읽기가 쓰기보다 3배 이상 느리다**(스토어 버퍼가 있고
> 로드는 지연을 숨기지 못한다). 단순 `LDR/STR` 루프는 `LDM/STM` 대비 절반 이하의 대역폭만 낸다.

모두 `대역폭 = 버스폭(byte) × 클럭(Hz)` 로 계산한 **이론 상한**이며, MB = 10⁶ byte 기준이다. 실제로는 중재·wait state·burst 효율로 낮아진다. 등급 [D](AN4667의 버스폭/클럭 [A] + 산술 계산).

| 경로 | 폭 | 클럭 | 이론 대역폭 | 계산 |
|---|---|---|---|---|
| Cortex-M7 AXIM | 64-bit | 216 MHz | **1728 MB/s** | 8 × 216×10⁶ |
| Cortex-M7 ITCM (ITCM-RAM / FLASH-ITCM) | 64-bit | 216 MHz | **1728 MB/s** | 8 × 216×10⁶ |
| Cortex-M7 DTCM (D0TCM + D1TCM) | 32-bit × 2 | 216 MHz | **1728 MB/s** (포트당 864) | 2 × 4 × 216×10⁶ |
| Cortex-M7 AHBS (DMA → DTCM) | 32-bit | 216 MHz | **864 MB/s** | 4 × 216×10⁶ |
| Cortex-M7 AHBP (주변장치) | 32-bit | 216 MHz | **864 MB/s** | 4 × 216×10⁶ |
| AXI→AHB 64-bit Flash 레인 | 64-bit | 216 MHz | **1728 MB/s** | 8 × 216×10⁶ |
| AXI→AHB 32-bit 레인 (3개 각각) | 32-bit | 216 MHz | **864 MB/s** (합 2592) | 4 × 216×10⁶ |
| SRAM1 / SRAM2 슬레이브 포트 (각각) | 32-bit | 216 MHz | **864 MB/s** | 4 × 216×10⁶ |
| APB2 (SDMMC, USART1/6, SPI1/4/5/6, TIM1/8 …) | 32-bit | 108 MHz | **432 MB/s** | 4 × 108×10⁶ |
| APB1 (USART2/3, I2C, TIM2-7 …) | 32-bit | 54 MHz | **216 MB/s** | 4 × 54×10⁶ |

**Flash 실효 시퀀셜 읽기 대역폭 [D]:** Flash 어레이는 256-bit(32 byte) 폭이고 216 MHz에서 7 WS이므로 한 라인 읽기에 1+7 = 8 사이클. ART/캐시 히트 시 후속 접근은 0 WS이지만, 미스 시 시퀀셜 상한은 `32 byte / 8 cycle × 216 MHz = 864 MB/s` = 4 byte/cycle. Thumb-2 평균 명령 길이를 3 byte로 잡으면 미스 상태에서도 약 1.33 명령/사이클에 해당하여, 순차 코드에서 ART가 "0-WS 등가"를 낼 수 있는 이유가 된다.

**DTCM/ITCM 실효 [D]:** TCM은 wait state가 없어(AN4667 §1.5.2 "accessible at a maximum CPU clock speed without latency") 위 수치가 곧 실효 상한이다.

**보드 외부 메모리 [D]:**
- SDRAM (FMC bank5 `0xC0000000`, 16-bit 데이터 폭, SDCLK = HCLK/2 = 108 MHz): `2 byte × 108×10⁶ = 216 MB/s`
- Quad-SPI (`0x90000000`, 4-bit ×DDR, AN4667 벤치마크 조건 54 MHz DDR): `4 bit × 2 edge × 54×10⁶ / 8 = 54 MB/s`

> **실측 정정**: 두 줄 다 이 보드의 조건이 아니다.
> - **SDRAM**: HCLK 200 MHz → **SDCLK = 100 MHz**(`SDCR1.SDCLK=0b10`), 이론 상한 **200 MB/s**.
>   실측은 **읽기 60 MB/s / 쓰기 195 MB/s / copyLDM 81 MB/s**로, 쓰기는 이론에 근접하지만
>   **읽기는 이론의 30 %**다. 랜덤 접근 레이턴시 21~26 사이클.
>   컨트롤러 실측 조건: 16-bit / 4뱅크 / 12행 / 8열 / **CAS 2** / RBURST=1 / RPIPE=0, 리프레시 15.4 µs.
> - **QSPI**: DDR이 아니다. `CR.PRESCALER=1` → **100 MHz SDR**, `CCR = 0x0F282DEB`
>   (명령 0xEB Quad I/O, 주소 4선, 데이터 4선, 더미 10사이클, 메모리맵 모드).
>   이론 상한 `4 bit × 100×10⁶ / 8 = 50 MB/s`, **실측 순차 읽기 49.97 MB/s**(상한에 도달).
>   문서의 54 MB/s와 숫자는 비슷하지만 도출 조건이 완전히 다르다.
>   64 KB 스트라이드 랜덤 접근은 **94~198 사이클/접근**으로 매우 비싸다.

즉 **DTCM(1728 MB/s) : SRAM1(864 MB/s) : SDRAM(216 MB/s) : QSPI(54 MB/s) ≈ 32 : 16 : 4 : 1** 의 계층이 된다. [D]

> **실측 정정**: 실측 읽기 대역폭으로 다시 세우면 계층비가 달라진다 —
> **DTCM(1409) : SRAM1 읽기(206) : Flash ITCM+ART(492) : SDRAM 읽기(60) : QSPI(50) ≈ 28 : 4 : 10 : 1.2 : 1**.
> 즉 **ART를 켠 플래시가 SRAM1 읽기보다 2배 이상 빠르고, SDRAM 읽기는 QSPI와 큰 차이가 없다.**
> 이론 계층비(32:16:4:1)와는 순서 자체가 다르므로, 버퍼 배치 판단은 실측 쪽을 근거로 해야 한다.

---

## 6. 부트 구성

### 6.1 BOOT 핀과 BOOT_ADDx 옵션 바이트

STM32F7은 STM32F4와 달리 **BOOT 핀이 1개(BOOT0)** 이고, BOOT1 핀은 없다. 대신 **BOOT_ADD0 / BOOT_ADD1 옵션 바이트**로 부트 주소를 자유롭게 지정한다. [A] DS10916 Rev 4 §2.15

> "At startup, the boot memory space is selected by the BOOT pin and BOOT_ADDx option bytes, allowing to program any boot memory address from `0x00000000` to `0x3FFFFFFF` which includes: all Flash address space mapped on ITCM or AXIM interface; all RAM address space: ITCM, DTCM RAMs and SRAMs mapped on AXIM interface; the System memory bootloader."

| BOOT 핀 | 사용되는 옵션 바이트 | 등급 |
|---|---|---|
| BOOT0 = 0 | **BOOT_ADD0** (`FLASH_OPTCR1[15:0]`) | [B] |
| BOOT0 = 1 | **BOOT_ADD1** (`FLASH_OPTCR1[31:16]`) | [B] |

**부트 주소 인코딩:** `부트 주소 = BOOT_ADDx × 0x4000` (즉 `BOOT_ADDx << 14`), 16 KB 단위. [D] (아래 HAL 상수들로부터 역산; 모든 값이 일치)

| HAL 상수 | BOOT_ADDx 값 | 실제 부트 주소 | 대상 | 등급 |
|---|---|---|---|---|
| `OB_BOOTADDR_ITCM_RAM` | `0x0000` | `0x00000000` | ITCM-RAM | [B] |
| `OB_BOOTADDR_SYSTEM` | `0x0040` | `0x00100000` | 시스템 메모리 부트로더 (ITCM 별칭) | [B] |
| `OB_BOOTADDR_ITCM_FLASH` | `0x0080` | `0x00200000` | Flash on ITCM | [B] |
| `OB_BOOTADDR_AXIM_FLASH` | `0x2000` | `0x08000000` | Flash on AXIM | [B] |
| `OB_BOOTADDR_DTCM_RAM` | `0x8000` | `0x20000000` | DTCM-RAM | [B] |
| `OB_BOOTADDR_SRAM1` | `0x8004` | `0x20010000` | SRAM1 | [B] |
| `OB_BOOTADDR_SRAM2` | `0x8013` | `0x2004C000` | SRAM2 (F746 기준) | [B] |

출처: `Drivers/STM32F7xx_HAL_Driver/Inc/stm32f7xx_hal_flash_ex.h` L264–273 (ST 공식 HAL). 현재 어떤 BOOT_ADDx가 선택되었는지는 `SYSCFG_MEMRMP.MEM_BOOT` 비트로 읽을 수 있다(`__HAL_SYSCFG_GET_BOOT_MODE()`). [B]

### 6.2 시스템 메모리 부트로더

| 항목 | 값 | 등급/출처 |
|---|---|---|
| 위치 | `0x1FF00000 – 0x1FF0EDBF` | [A] AN2606 Rev 61 Table 87 |
| 크기 | **60,864 byte (0xEDC0)**, AN2606 본문 표기 "60 Kbytes" | [A] AN2606 Rev 61 Table 85 |
| 부트로더가 사용하는 RAM | `0x20000000`부터 16 KB (AN2606 Table 85) / 실사용 표기 `0x20004000 – 0x2004FFFF` (Table 87) | [A] |
| Device ID | **`0x449`** (STM32F74xxx/75xxx) | [A] AN2606 Rev 61 Table 87 |
| Bootloader ID | `0x70` = V7.x (rev A 디바이스), **`0x90` = V9.x (rev Z 이상)** | [A] AN2606 Rev 61 §39 |
| 부트로더 클럭 | HSI 기반 60 MHz (CAN/DFU 선택 시 HSE 기반 60 MHz) | [A] AN2606 Rev 61 Table 85 |
| 부트로더 Flash 설정 | 3 wait state, ART Accelerator ON, byte 단위 write | [A] AN2606 Rev 61 Table 85 |
| 지원 인터페이스 | USART1(PA9/PA10), USART3(PB10/PB11 또는 PC10/PC11), CAN, USB DFU(FS), I2C, SPI | [A] AN2606 Rev 61 §39 |
| 진입 패턴 | Pattern 8 | [A] AN2606 Rev 61 §39 |

> 본 보드는 실측 REV_ID = Rev Z이므로 **V9.x 부트로더(ID `0x90`)** 가 탑재되어 있다. [D] (AN2606 §39 "V9.x ... embedded in STM32F74xxx/75xxx rev. Z and above" + 실측 Rev Z)

> RDP Level 2가 활성화된 경우에도 STM32F74xxx/75xxx는 시스템 메모리로 부트할 수 있다(AN2606 §39 Note). [A]

---

## 7. 팩토리 프로그램 영역 (Device electronic signature / OTP)

| 이름 | 주소 | 크기 | 내용 | 등급 |
|---|---|---|---|---|
| OTP 영역 | `0x1FF0F000 – 0x1FF0F41F` | 1056 byte (데이터 1024 B + lock 32 B) | 1회 기록 영역 | [B] CMSIS `FLASH_OTP_BASE/END`; DS10916 "1024 bytes of OTP memory" [A] |
| **Unique device ID** | `0x1FF0F420` | **96 bit (12 byte)** | 웨이퍼 X/Y 좌표, 로트 번호 등 | [B] CMSIS `UID_BASE`; DS10916 "96-bit unique ID" [A] |
| **Flash size register** | `0x1FF0F442` | 16 bit | Flash 용량(KB). F746NG = `0x0400` (=1024 KB) | [B] CMSIS `FLASHSIZE_BASE` / [D] 값 유도 |
| VREFIN_CAL | `0x1FF0F44A – 0x1FF0F44B` | 16 bit | 30 °C, VDDA=3.3 V 기준 raw | [A] DS10916 |
| TS_CAL1 | `0x1FF0F44C – 0x1FF0F44D` | 16 bit | 30 °C, VDDA=3.3 V | [A] DS10916 |
| TS_CAL2 | `0x1FF0F44E – 0x1FF0F44F` | 16 bit | 110 °C, VDDA=3.3 V | [A] DS10916 |
| **Package data register** | **`0x1FF0F7E0`** | 16 bit | 패키지 코드 | [B] CMSIS `PACKAGE_BASE`; [C] ST 커뮤니티 |

> **작업 지시서의 `0x1FF07BF0`은 STM32F72xxx/F73xxx용 주소이며 STM32F74xxx/F75xxx에는 해당하지 않는다.** ST 공식 CMSIS 헤더 `stm32f746xx.h` L1230이 `#define PACKAGE_BASE 0x1FF0F7E0UL`로 정의하고, ST 커뮤니티 스레드에서도 "STM32F74xx/STM32F75xx devices: 0x1FF0F7E0, STM32F72xx/STM32F73xx devices: 0x1FF07BF0"으로 확인된다. 또한 문서 원본에 `0x1FFF7BF0`으로 오기된 사례가 있으니 주의. [B]/[C]

**패키지 코드 표:** RM0385 원문 미확보. ST 커뮤니티에서 인용된 (F76x/F77x 기준으로 보이는) 값은 다음과 같으며 **F746에 그대로 적용된다고 확인되지 않았다**. [C]

| 값 (mask `0x0700`) | 패키지 |
|---|---|
| `0x0100` | LQFP100 |
| `0x0200` | LQFP144 또는 WLCSP143 |
| `0x0300` | WLCSP180, LQFP176 또는 UFBGA176 |
| `0x0400`–`0x0700` | LQFP176, LQFP208 또는 TFBGA216 |

본 디바이스 STM32F746**N**G**H**6의 실제 패키지는 **TFBGA216 (13 × 13 × 0.8 mm)** 이다. [A] DS10916 Rev 4 §6.8 / UM1907

> **실측 확인 — 이 항목은 문서가 맞았다**: ST-LINK 덤프에서 `0x1FF07BF0`(F72x/F73x용 주소)은
> `0xFFFFFFFF`로 읽혀 "패키지 판별 불가"로 기록되었으나, 문서가 지목한 **`0x1FF0F7E0`을 같은 덤프에서
> 읽으면 `0xFC72`**(리셋 덤프·실행 중 덤프 모두 동일)이다. PKG 필드(mask `0x0700`)가 `0x400`이므로
> 위 코드표의 **LQFP176/LQFP208/TFBGA216 등급**을 가리키며, 실제 부품 TFBGA216과 모순되지 않는다.
> → **주소는 문서 쪽(`0x1FF0F7E0`)이 옳다.** 실측 노트의 "패키지 판별 불가"는 주소를 잘못 읽은 결과다.
> (근거 파일: `info/raw/regions_reset/factory_uid_area.bin` 오프셋 `0x7E0`)

**DBGMCU_IDCODE (`0xE0042000`)**: `DEV_ID[11:0] = 0x449`, `REV_ID[31:16]`. Rev Z = `0x1001`. [A] AN2606 Rev 61 Table 87 (`0x449`) / [C] ES0290 (REV_ID 0x1001) — RM0385 §DBGMCU 원문 재확인 필요.

---

## 8. Flash 섹터 구성 · 옵션 바이트 · 보호

### 8.1 섹터 구성 (단일 뱅크 1 MB)

ST 공식 플래시 로더 `0x449.stldr`(STM32CubeIDE 내장 STM32CubeProgrammer, `FlashLoader/0x449.stldr`)의 `StorageInfo` 구조체를 직접 파싱한 결과: [B]

```
DeviceStartAddress = 0x08000000
DeviceSize         = 0x00100000   (1 MB)
PageSize           = 0x00008000   (32 KB)
EraseValue         = 0xFF
Sectors: (count=4, size=0x08000)  (count=1, size=0x20000)  (count=3, size=0x40000)
```

| 섹터 | AXIM 주소 (`0x08000000` 기준) | ITCM 별칭 (`0x00200000` 기준) | 크기 | 등급 |
|---|---|---|---|---|
| Sector 0 | `0x08000000 – 0x08007FFF` | `0x00200000 – 0x00207FFF` | 32 KB | [B]/[D] |
| Sector 1 | `0x08008000 – 0x0800FFFF` | `0x00208000 – 0x0020FFFF` | 32 KB | [B]/[D] |
| Sector 2 | `0x08010000 – 0x08017FFF` | `0x00210000 – 0x00217FFF` | 32 KB | [B]/[D] |
| Sector 3 | `0x08018000 – 0x0801FFFF` | `0x00218000 – 0x0021FFFF` | 32 KB | [B]/[D] |
| Sector 4 | `0x08020000 – 0x0803FFFF` | `0x00220000 – 0x0023FFFF` | 128 KB | [B]/[D] |
| Sector 5 | `0x08040000 – 0x0807FFFF` | `0x00240000 – 0x0027FFFF` | 256 KB | [B]/[D] |
| Sector 6 | `0x08080000 – 0x080BFFFF` | `0x00280000 – 0x002BFFFF` | 256 KB | [B]/[D] |
| Sector 7 | `0x080C0000 – 0x080FFFFF` | `0x002C0000 – 0x002FFFFF` | 256 KB | [B]/[D] |

합계 = 4×32 + 128 + 3×256 = **1024 KB** ✔. CMSIS 헤더의 `#define FLASH_SECTOR_TOTAL 8`, `FLASH_END = 0x080FFFFF`와 일치. 주소 자체는 섹터 크기 누적으로 계산했으므로 [D], 섹터 개수/크기는 ST 툴 데이터이므로 [B].

### 8.2 이중 뱅크 여부

**STM32F745xx/F746xx/F756xx(1 MB)는 단일 뱅크 전용이며 dual-bank 모드를 지원하지 않는다.** [C]
근거: CMSIS `stm32f746xx.h`에 `FLASH_OPTCR_nDBANK` / `FLASH_OPTCR_nDBOOT` / `FLASH_OPTCR2_PCROP` 비트 정의가 **존재하지 않는다**(HAL 소스가 `#if defined (FLASH_OPTCR_nDBANK)`로 감싸 조건부 컴파일). Dual-bank는 2 MB급 STM32F76xxx/F77xxx의 기능이다(AN4826). [B]/[D]

### 8.3 옵션 바이트

| 레지스터 | 주소 | 주요 필드 | 등급 |
|---|---|---|---|
| `FLASH_OPTCR` | `0x40023C14` (= `FLASH_R_BASE + 0x14`, `FLASH_R_BASE = 0x40023C00`) | 아래 표 | [B] CMSIS |
| `FLASH_OPTCR1` | `0x40023C18` | `BOOT_ADD0[15:0]`, `BOOT_ADD1[31:16]` | [B] CMSIS |
| 옵션 바이트 영역 (Flash 매핑) | `0x1FFF0000 – 0x1FFF001F` | OPTCR/OPTCR1 미러 | [C] DS10916 Figure 19 |

`FLASH_OPTCR` 비트 필드 [B] CMSIS `stm32f746xx.h` L7316–7365:

| 비트 | 이름 | 설명 |
|---|---|---|
| [0] | OPTLOCK | 옵션 바이트 잠금 |
| [1] | OPTSTRT | 옵션 바이트 수정 시작 |
| [3:2] | BOR_LEV | Brown-out 레벨 |
| [4] | WWDG_SW | WWDG 소프트웨어/하드웨어 선택 |
| [5] | IWDG_SW | IWDG 소프트웨어/하드웨어 선택 |
| [6] | nRST_STOP | Stop 진입 시 리셋 생성 여부 |
| [7] | nRST_STDBY | Standby 진입 시 리셋 생성 여부 |
| [15:8] | **RDP[7:0]** | 읽기 보호 레벨 |
| [23:16] | **nWRP[7:0]** | 섹터 0–7 쓰기 보호 (비트당 1섹터, 0 = 보호) |
| [30] | IWDG_STDBY | Standby에서 IWDG 정지 여부 |
| [31] | IWDG_STOP | Stop에서 IWDG 정지 여부 |

`FLASH_ACR` (`0x40023C00`) 관련 비트 [B] CMSIS:

| 비트 | 이름 | 설명 |
|---|---|---|
| [3:0] | LATENCY | wait state 0–15 (216 MHz@3.3 V → 7) |
| [8] | PRFTEN | ART-Prefetch enable |
| [9] | ARTEN | ART Accelerator enable |
| [11] | ARTRST | ART reset |

### 8.4 읽기 보호(RDP) 레벨 [B] HAL `stm32f7xx_hal_flash_ex.h` L164–166 / [C] ST 문서

| 레벨 | RDP 값 | 동작 |
|---|---|---|
| Level 0 | `0xAA` | 보호 없음(공장 출하 기본값) |
| Level 1 | `0xAA`/`0xCC` 이외 값 (HAL 상수는 `0x55`) | 디버그 연결/RAM 부트/부트로더에서 Flash 접근 차단. Level 0으로 되돌리면 **mass erase** 발생 |
| Level 2 | `0xCC` | **영구 보호**. 디버그 포트 완전 차단, 되돌릴 수 없음 |

> Level 2 상태에서도 STM32F74xxx/75xxx는 시스템 메모리 부트가 가능하다. [A] AN2606 §39

---

## 9. Flash 프로그래밍/소거 시간, 내구성, 데이터 보존

전부 [A] DS10916 Rev 4 (DocID027590 Rev 4) Table 47/48/49/50. 조건: TA = −40 ~ 105 °C 별도 명시 없는 한.

### 9.1 프로그래밍/소거 시간 (VPP 없음, Table 48)

| 파라미터 | PSIZE | Typ | Max | 단위 |
|---|---|---|---|---|
| tprog 워드 프로그래밍 | x8/x16/x32 | **16** | 100 ⁽¹⁾ | µs |
| tERASE32KB (32 KB 섹터) | x8 | 400 | 800 | ms |
| tERASE32KB | x16 | 250 | 600 | ms |
| tERASE32KB | **x32** | **200** | 500 | ms |
| tERASE128KB | x8 | 1100 | 2400 | ms |
| tERASE128KB | x16 | 800 | 1400 | ms |
| tERASE128KB | **x32** | **500** | 1100 | ms |
| tERASE256KB | x8 | 2.1 | 4 | s |
| tERASE256KB | x16 | 1.5 | 2.6 | s |
| tERASE256KB | **x32** | **1** | 2 | s |
| tME (mass erase, 1 MB 전체) | x8 | 8 | 16 | s |
| tME | x16 | 5.6 | 11.2 | s |
| tME | **x32** | **4** | 8 | s |

⁽¹⁾ 최대 프로그래밍 시간은 100K erase 동작 이후 측정값.

프로그래밍 전압: x32 모드 2.7–3 V, x16 모드 2.1–3.6 V, x8 모드 1.7–3.6 V. **VDD 3.3 V의 DISCO 보드는 x32(word) 병렬성 사용 가능.** [A]

전체 1 MB를 섹터 단위로 지우고 다시 쓰는 데 걸리는 대략적 시간 [D]: 섹터 소거 typ 합계 = 4×200 ms + 500 ms + 3×1 s = 0.8 + 0.5 + 3.0 = **4.3 s**(mass erase typ 4 s와 정합). 프로그래밍은 1 MB = 262,144 워드 × 16 µs = **약 4.2 s**.

### 9.2 VPP 사용 시 (Table 49, TA = 0~+40 °C, VDD = 3.3 V, VPP = 8.5 V)

| 파라미터 | Typ |
|---|---|
| tprog (double word) | 16 µs |
| tERASE32KB | 180 ms |
| tERASE128KB | 450 ms |
| tERASE256KB | 900 ms |
| tME | 6.9 s |

VPP 범위 7–9 V, VPP 핀 최소 sink 전류 10 mA, **VPP 누적 인가 시간 최대 1시간**. [A]

### 9.3 소비 전류 (Table 47)

| 조건 | IDD (Typ) |
|---|---|
| Write/Erase 8-bit mode, VDD = 1.7 V | 14 mA |
| Write/Erase 16-bit mode, VDD = 2.1 V | 17 mA |
| Write/Erase 32-bit mode, VDD = 3.3 V | **24 mA** |

### 9.4 내구성 / 데이터 보존 (Table 50)

| 파라미터 | 조건 | Min |
|---|---|---|
| NEND 내구성 | TA = −40~+85 °C (suffix 6) / −40~+105 °C (suffix 7) | **10 kcycles** |
| tRET 보존 | 1 kcycle @ TA = 85 °C | **30 년** |
| tRET 보존 | 1 kcycle @ TA = 105 °C | **10 년** |
| tRET 보존 | 10 kcycles @ TA = 55 °C | **20 년** |

디바이스는 Flash가 소거된 상태로 출하된다. [A]

---

## 10. STM32F746G-DISCO 보드 외부 메모리 (참고)

| 메모리 | 부품 | 주소 | 용량 | 비고 | 등급 |
|---|---|---|---|---|---|
| SDRAM | MICRON **MT48LC4M32B2B5-6A** (128 Mbit) | `0xC0000000` (FMC bank 5) | **8 MB 접근 가능** (64 Mbit) | **하위 16-bit만 배선**, DQ16–DQ31은 10 kΩ 풀다운 | [A] UM1907 §6.11 |
| Quad-SPI NOR | MICRON **N25Q128A13EF840E** (128 Mbit) | `0x90000000` | **16 MB** | 메모리 매핑 모드 지원 | [A] UM1907 §6.12 |

프로젝트 `ImageClassification.ioc`에서 LTDC 프레임버퍼가 `LTDC.FBStartAdress_L0 = 0xC0000000`으로 SDRAM에 배치되어 있다. [B] (로컬 프로젝트 파일)

CubeIDE 내장 외부 로더도 `N25Q128A_STM32F746G-DISCO.stldr`, `W25Q128JVEIQ_STM32F746G-DISCO.stldr` 두 종이 존재한다(보드 리비전에 따라 QSPI 칩이 N25Q128A 또는 W25Q128JV). [B]

> **실측 정정 (QSPI 부품)**: 이 개체는 **N25Q128A가 아니라 Winbond W25Q128 계열**이다.
> 메모리맵 모드로 전환하기 전에 Read JEDEC ID(0x9F)를 직접 보낸 응답이 **`0xEF4018`**이다
> (제조사 `0xEF` = Winbond, 타입 `0x40` = W25Q, 용량 `0x18` = 2²⁴ = 16 MB).
> Micron N25Q128A라면 `0x20BA18`이어야 한다. 위 각주가 언급한 두 리비전 중 **W25Q128JV 쪽**이다.
> ST BSP(`stm32746g_discovery_qspi.c`)가 N25Q128A 전제로 쓰였는데도 동작하는 것은 두 부품이
> 표준 SPI NOR 명령(0x9F, 0xEB 등)을 공유하기 때문이며, **소거 시간·페이지 크기·상태 레지스터 비트는
> 다르므로 QSPI에 쓰기를 하는 코드는 Winbond 데이터시트로 검증해야 한다.**
>
> **SDRAM 쪽은 문서가 맞다**: `FMC_SDCR1 = 0x00001954` 디코드가 16-bit 버스 / 4뱅크 / 12행 / 8열 →
> 정확히 **8 MB**로, "128 Mbit 부품의 하위 16-bit만 배선 → 8 MB 접근 가능"이라는 서술과 일치한다.
> 실제 접근 경계 스캔도 `0xC0000000`+8192 KB에서 폴트한다. SDRAM 클럭은 HCLK/2 = **100 MHz**(200 MHz 기준).

---

## 출처

### 1차 출처 (실제로 본문을 읽은 문서)

| 문서 | 버전/ID | URL | 사용 내용 |
|---|---|---|---|
| **DS10916** STM32F745xx/STM32F746xx 데이터시트 | DocID027590 Rev 4 | https://www.st.com/resource/en/datasheet/stm32f746ng.pdf | Table 13(레지스터 경계 주소 전체), Figure 19(메모리 맵), Table 15(wait state), Table 47–50(Flash 특성), §2.15(부트 모드), §6.8(TFBGA216) |
| **AN4667** STM32F7 Series system architecture and performance | DocID027643 Rev 4 | https://www.st.com/resource/en/application_note/an4667-stm32f7-series-system-architecture-and-performance-stmicroelectronics.pdf (미러: https://kolegite.com/EE_library/application_notes/MCU/STM32/an4667-stm32f7-series-system-architecture-and-performance-stmicroelectronics.pdf) | §1.3(CM7 인터페이스), §1.4(버스 매트릭스 마스터/슬레이브), §1.5.1(Flash 인터페이스/ART), §1.5.2(Embedded SRAM), Table 3(내부 메모리 맵), Table 7(F74x/F75x 성능) |
| **AN2606** STM32 microcontroller system memory boot mode | AN2606 Rev 61 | https://kolegite.com/EE_library/application_notes/MCU/STM32/STM32%20microcontroller%20system%20memory%20boot%20mode.pdf | §39, Table 85(부트로더 리소스), Table 87(Device ID 0x449 / 시스템 메모리 0x1FF00000–0x1FF0EDBF / 부트로더 ID 0x70·0x90) |
| **ARM DDI 0489F** Cortex-M7 Technical Reference Manual | r1p2, ID121118 | (로컬 추출본) | §1.3.2·§5.7(AHBS), Table 5-37(AHBS 메모리 맵), §5.7.2(제약), Table 3-15(CM7_AHBSCR `0xE000EFA0`), bit-banding 비호환 |
| **UM1907** STM32F746G-DISCO 유저매뉴얼 | DocID027942 Rev 3 | https://www.st.com/resource/en/user_manual/um1907-discovery-kit-for-stm32f7-series-with-stm32f746ng-mcu-stmicroelectronics.pdf | §6.11 SDRAM, §6.12 Quad-SPI |
| **AN4839** Level 1 cache on STM32F7/STM32H7 | — | https://www.st.com/resource/en/application_note/an4839-level-1-cache-on-stm32f7-series-and-stm32h7-series-stmicroelectronics.pdf | 캐시 라인 32 byte, D-cache 4-way / I-cache 2-way, write-back/write-through |
| **ES0290** STM32F74xxx/F75xxx Errata sheet | ES0290 Rev 7 (2019-07) | https://www.stmcu.com.cn/upload/pdf_html/b4e2f0c4f7037c59583aa78c13e9c442.html | write-through 모드 D-cache 데이터 손상 제한(제목만 확인) |

### ST 공식 툴/소스 (로컬)

| 항목 | 경로 | 사용 내용 |
|---|---|---|
| CMSIS Device 헤더 | `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/CMSIS/Device/ST/STM32F7xx/Include/stm32f746xx.h` | 모든 베이스 주소(L1128–1150, L1227–1232), `FLASH_ACR`/`FLASH_OPTCR`/`FLASH_OPTCR1` 비트 정의(L7228–7373), `FLASH_SECTOR_TOTAL 8` |
| HAL Flash Ex 헤더 | `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/Drivers/STM32F7xx_HAL_Driver/Inc/stm32f7xx_hal_flash_ex.h` | `OB_BOOTADDR_*`(L264–273), `OB_RDP_LEVEL_*`(L164–166) |
| HAL 공통 헤더 | `.../Inc/stm32f7xx_hal.h` | `__HAL_SYSCFG_REMAPMEMORY_FMC()` (FMC/SDRAM 스왑), `__HAL_SYSCFG_GET_BOOT_MODE()` |
| ST 플래시 로더 | `/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.macosaarch64_1.0.100.202603051304/tools/bin/FlashLoader/0x449.stldr` | `StorageInfo` 파싱 → 섹터 구성 (4×32 KB, 1×128 KB, 3×256 KB), PageSize 0x8000, EraseValue 0xFF |
| 프로젝트 링커 스크립트 | `/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/STM32F746NGHX_FLASH.ld` | `RAM 0x20000000 LENGTH 320K`, `FLASH 0x08000000 LENGTH 1024K` |

### 2차 출처

- ST Community, "Package data register on STM32F779" — https://community.st.com/stm32-mcus-products-25/package-data-register-on-stm32f779-83488 (패키지 레지스터 주소 F74x/F75x = `0x1FF0F7E0`, F72x/F73x = `0x1FF07BF0`, 문서 오기 `0x1FFF7BF0` 정정, 패키지 코드 표)
- ST Community, "How to change the Read Out Protection on STM32F7" — https://community.st.com/t5/stm32-mcus/how-to-change-the-read-out-protection-on-stm32f7/ta-p/49414 (RDP 0xAA/0xCC 정의)
- ST Community, "STM32F7 HAL DMA cache coherence and DTCM RAM differentiation" — https://community.st.com/t5/stm32-mcus-products/stm32f7-hal-dma-cache-coherence-and-dtcm-ram-differentiation/td-p/435201
- AN4826 STM32F7 Series Flash memory dual bank mode — https://www.st.com/resource/en/application_note/an4826-stm32f7-series-flash-memory-dual-bank-mode-stmicroelectronics.pdf (본문 미열람, dual-bank 대상 디바이스 확인용)

### 접근 실패한 문서

- **RM0385 Rev 9** (https://www.st.com/resource/en/reference_manual/rm0385-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf) — PDF 46 MB로 웹 페치 크기 상한(10 MB) 초과, 미러(kamami, amperka, sekorm, usermanual.wiki, stmcu.com.cn) 모두 실패. 따라서 RM0385 Table 1(메모리 맵), Table 3(Flash 섹터), §2.1(버스 매트릭스 그림), §41(전자 서명 레지스터) 원문은 **직접 확인하지 못했다.** 위 내용은 DS10916 Table 13 + AN4667 Table 3 + ST 공식 툴 데이터로 교차 검증했다.

---

## 미확인 / 추가 조사 필요

1. **RM0385 원문 미확보.** 아래 항목은 RM0385로 재검증 필요:
   - Table 1 "Memory map and register boundary addresses"의 Reserved 구간 정확한 경계
   - Table 3 "Flash memory organization" 섹터 표 (현재는 ST 플래시 로더 데이터로 [B] 재구성)
   - §2.1 Figure 1 시스템 아키텍처의 마스터×슬레이브 연결선 (현재는 AN4667 Figure 1 기반, 개별 셀은 [D])
2. **옵션 바이트 영역 주소 `0x1FFF0000 – 0x1FFF001F`** — DS10916 Figure 19에서 폰트 인코딩 때문에 숫자 글리프가 소실되어 letter 복원으로 추정했다. 신뢰도 [C]. RM0385 Table 8로 확인 필요.
3. **패키지 데이터 레지스터의 PKG 비트 인코딩** — F74x/F75x 전용 코드 표 미확인. 현재 표는 F76x/F77x 사용자 코드에서 유래한 [C] 자료다.
4. **DMA1의 접근 가능 슬레이브 범위** — "DMA1은 memory-to-memory 불가, DMA1 memory 포트가 SRAM에만 연결" 이라는 STM32F4 계열의 제약이 STM32F7에도 동일하게 적용되는지 RM0385 DMA 장으로 확인 필요. AN4667은 DMA1_MEM을 버스 매트릭스 마스터로 나열할 뿐 슬레이브 범위를 명시하지 않는다.
5. **각 마스터별 접근 가능 슬레이브 상세** — §4.3 표에서 DMA/ETH/USB/LTDC/DMA2D 행의 개별 셀은 AN4667 Figure 1(경로 1~10) 해석 기반 [D]다. RM0385 Figure 1의 실제 연결선으로 확인 필요.
6. **버스 매트릭스 실측 대역폭** — §5는 순수 이론값 [D]다. 중재 오버헤드, DMA burst 길이, SDRAM CAS latency, QSPI 명령 오버헤드를 반영한 실측이 없다. STM32F746G-DISCO에서 DWT_CYCCNT 기반 memcpy 벤치마크로 DTCM/SRAM1/SRAM2/SDRAM 각각 측정 권장.
   > **해소됨 (2026-07-27)**: 온디바이스 프로브 펌웨어가 DWT_CYCCNT로 전부 측정했다.
   > 결과는 §5의 실측 정정 표와 §10 각주에 반영했다. 요약: 실측 최댓값은 이론값의 20~90 %이며
   > **읽기 대역폭이 병목**이다. (원문: `info/results/probe_output.txt` §3~§8)
7. **ES0290 에라타 전문 미열람** — "Cortex-M7 data corruption when using data cache configured in write-through" 외의 메모리/캐시/DMA 관련 제한 사항과 Rev A vs Rev Z 영향 범위 미확인. 본 보드가 Rev Z이므로 Rev Z 적용 항목 전수 확인 필요.
8. **ARM Cortex-M7 r0p1 고유 에라타** — STM32F746의 코어는 r0p1이나, 로컬 확보한 TRM은 r1p2다. r0p1에만 존재하는 TCM/AHBS 관련 에라타(ARM SDEN-1068427) 확인 필요.
9. **DTCM 접근 시 DMA 지연 실측** — CM7_AHBSCR.CTL 기본값(`0b00`, AHBS 강등)이 실제 DMA 처리량에 미치는 영향 수치 미확인.
10. **ITCM-RAM 16 KB의 로딩 방법** — 부팅 시 ITCM-RAM에 코드를 복사하는 경로(AXIM→SRAM→? 또는 CPU 직접 복사)와 소요 시간 미확인. DMA로는 ITCM에 쓸 수 없으므로 CPU memcpy가 유일한 수단으로 보이나 [D], 확인 필요.
