# STM32F746NGH6 — L1 캐시 · TCM · 메모리 계층 심층 조사

대상: STM32F746G-DISCO (STM32F746NGH6, Cortex-M7 r0p1, Device ID 0x449, Rev Z)
조사일: 2026-07-27

---

## 이 보드 실측과의 대조 (2026-07-27)

> 아래 내용은 이 문서를 쓴 뒤, 같은 보드에서 실제로 측정한 값과 대조한 결과다.
> **실측(신뢰도 A)과 문서(신뢰도 B)가 어긋나면 실측이 맞다.**
> 근거: `info/raw/measured_facts.md`, `info/results/probe_output.txt`,
> `info/raw/decoded_hotplug.md`, `info/raw/decoded_reset.md`
>
> **가장 먼저 알아야 할 것: 이 보드는 216 MHz가 아니라 200 MHz로 돈다.**
> 본문의 216 MHz 기준 대역폭·wait state·전류 수치는 전부 그 전제로 다시 읽어야 한다.

### 실측으로 확인됨

| 항목 | 문서값 | 실측값 |
|---|---|---|
| L1 D-cache | 4 KB, 4-way, 32 set, 32 B line, `CCSIDR = 0xF003E019` (예상) | `CCSIDR(CSSELR=0) = 0xF003E019` — **비트 단위 일치** |
| L1 I-cache | 4 KB, 2-way, 64 set, 32 B line, `CCSIDR = 0xF007E009` (예상) | `CCSIDR(CSSELR=1) = 0xF007E009` — **일치** |
| CTR | `0x8303C003` (PIPT, 모든 라인 32 B) | `CTR = 0x8303C003` — 일치 (L1Ip=3=PIPT, I/DminLine 32 B, CWG 32 B) |
| CLIDR | `0x09000003` (계산에 의한 **예측** [D]) | `CLIDR = 0x09000003` — **예측이 맞았다.** Ctype1=3(하버드), LoC=1(L2 없음). [D] → **[A]** |
| D-cache 라인 32 B | [A] | 스트라이드 스윕에서 **32 B부터 평탄**(SRAM1 16 B 6.51 → 32 B 9.03 → 64 B 9.06 cyc) |
| D-cache 4 KB 용량 | [A] | 워킹셋 스윕에서 **4 KB→8 KB 사이에 계단**(WB 235.67 → 221.94 MB/s), 포인터 체이싱은 **4 KB 2.09 cyc → 8 KB 7.11 cyc** |
| way 1개 = 1 KB → 1 KB 간격 충돌 | [D, 계산] | 프로브도 동일 계산(충돌 간격 1024 B). 스래싱 자체는 미측정 |
| SDRAM(0xC0000000)은 기본 맵에서 **Device → 캐시 불가** | [A] | D-cache OFF와 WB의 SDRAM 성능이 **완전히 동일**(read32 59.70 vs 59.71 MB/s, 포인터 체이싱 26.06 vs 26.06 cyc) → 캐싱되지 않음 확인 |
| QSPI(0x90000000)는 기본 맵에서 캐시 가능(WT) | [A] | D-cache를 켜면 64 KB 스트라이드 접근이 **94.22 → 197.54 cyc/접근**으로 나빠진다 → 라인필이 발생 = 캐시 대상임을 시사 |
| Flash on AXIM은 L1 D-cache 대상 | [A] | AXIM 순차 읽기 **159.96 → 304.82 MB/s (×1.91)** |
| TCM은 캐시 대상이 아님 (§3.3) | "MPU 설정과 무관하게 Non-cacheable" [A] | ITCM 경로 읽기가 D-cache ON/OFF에서 **492.10 MB/s로 동일**, ITCM/DTCM의 OFF/WB/WT 3열이 전부 같은 값 |
| ITCM 16 KB @ 0x00000000 | DS10916 [A] | `0x00000000`~`0x00003FFF` 정상(5.16 cyc), **`0x00004000`부터 버스 폴트** |
| DTCM 64 KB @ 0x20000000 | DS10916 [A] | `0x20000000`~`0x2000FFFF`가 5.16 cyc, `0x20010000`부터 15.66 cyc로 갈림 |
| SRAM1 240 KB + SRAM2 16 KB | DS10916 [A] | `0x20050000`부터 폴트 → DTCM 64 K + 240 K + 16 K = **320 KB** 확인 |
| Flash on ITCM = 0x00200000 별칭 | 커뮤니티/AN4667 인용 [C] | `0x00200000`과 `0x08000000`의 선두 64바이트가 **완전 일치**, 둘 다 1024 KB. [C] → **[A]** |
| DTCMCR SZ = 0b0111 (64 KB) | 예상 [D] | `DTCMCR = 0x0000003B` → **SZ=0b0111 일치** (EN=1) |
| AHBSCR INITCOUNT 리셋값 0b01, CTL 리셋값 0b00 | [A] | `AHBSCR = 0x00000800` → **INITCOUNT=1, TPRI=0, CTL=0** — 일치 |
| CACR FORCEWT 리셋값 0 / SIWT 리셋값 0 | [A] | `CACR = 0x00000002` → **FORCEWT=0, SIWT=0** (bit[1]은 아래 「어긋남」 참조) |
| ACTLR에 DISDEFWBUF/DISMCYCINT 없음 (§2.6 "중요 정정") | [A, 부재로 확인] | `ACTLR = 0x00000000` (리셋·실행 동일). 값으로는 판별 불가하나 반증도 없음 |
| FLASH_ACR 비트 배치 (LATENCY[3:0], PRFTEN b8, ARTEN b9) | RM0385 색인 + libopencm3 [B]/[C] | 그 비트로 ART를 껐다 켰다 하니 **실제로 성능이 변했다**(ITCM 경로 336.79 ↔ 492.10 MB/s) → 비트 배치 기능적으로 확인 |
| wait state 유도표의 180–210 MHz 구간 = 6 WS | [D, 30 MHz 등간격 가정] | 200 MHz에서 `FLASH_ACR = 0x00000006` → **LATENCY = 6**. 유도표에 독립적인 데이터 포인트 하나가 생겼다 |
| 외부 SDRAM 8 MB (16비트만 사용) | UM1907 [A] | `SDCR1 = 0x00001954` 디코드 → 4뱅크 × 2¹² 행 × 2⁸ 열 × 2 B = **8,388,608 B**. 크기 스캔도 8192 KB에서 폴트 |
| AHBP 포트 (§3.7 SZ 미확인) | 미확인 | `AHBPCR = 0x00000009` → **EN=1, SZ=0b100 = 512 MB** |
| MPU 리전 8개 (§7.4 각주) | 8 또는 16 [A] | `MPU_TYPE = 0x00000800` → **8리전 확정** |

### 실측과 어긋남 — 문서 쪽을 신뢰하지 말 것

**1. `ITCMCR` 예상값 `0x29`(SZ=0b0101=16 KB) — §3.4** ★ 가장 큰 어긋남
- 문서 주장: SZ 인코딩표(0b0101 = 16 KB)에 따라 `ITCMCR = 0x29`, `DTCMCR = 0x39`를 예상.
- 실측: **`ITCMCR = 0x00000073`** (EN=1, **RMW=1**, RETEN=0, **SZ=0b1110**),
  **`DTCMCR = 0x0000003B`** (EN=1, **RMW=1**, RETEN=0, SZ=0b0111).
  DTCM은 SZ가 맞지만, **ITCM의 SZ=0b1110은 이 표대로면 8 MB**다.
  그런데 온디바이스 폴트 경계 실측은 **ITCM 실제 크기 = 16 KB**(`0x00004000`부터 버스 폴트)라고 답한다.
- 원인 추정 (두 가설, 어느 쪽이든 **문서의 "0x29 예상"은 틀렸다**):
  1. ST가 SZ 필드를 TRM 인코딩과 다르게 하드와이어했거나 미구현으로 두었다.
  2. **더 설득력 있는 쪽** — SZ는 "붙어 있는 RAM 용량"이 아니라 **코어가 ITCM 인터페이스에 할당한
     주소 공간 크기**를 보고한다. 0b1110 = 8 MB 창 안에 ST가 `0x00000000`~`0x00003FFF`에는 ITCM SRAM을,
     `0x00200000`~`0x002FFFFF`에는 Flash 별칭을 배치했다고 보면 8 MB 창과 16 KB RAM이 모순 없이 공존한다.
     `0x00200000`(= 2 MB)이 8 MB 창 안에 들어간다는 점이 이 가설을 지지한다.
     프로브 펌웨어 자신도 같은 취지의 주석을 달고 있다.
- **실측 해석 재검토**: `measured_facts.md` §3.1은 "TRM 표 해석이 틀렸다"고 단정하지만, 가설 2가 맞다면
  **TRM 표도 SZ 값도 틀리지 않았고 "SZ = RAM 용량"이라는 문서의 전제만 틀린 것**이다.
  확정하려면 `0x00004000`~`0x001FFFFF` 구간의 폴트 종류(BusFault 정밀/부정밀)를 봐야 한다 — 미측정.
- 심각도: 높음 (SZ로 TCM 용량을 계산하는 코드는 이 칩에서 8 MB를 얻는다).

**2. `RMW`/`RETEN`은 0일 것 — §3.4**
- 문서 주장: "STM32F7의 TCM에는 ECC가 없으므로 RMW/RETEN은 0일 것으로 예상".
- 실측: 두 TCM 모두 **RMW = 1**(RETEN은 예상대로 0). 리셋 직후 덤프와 실행 중 덤프가 동일하므로
  소프트웨어가 켠 것이 아니라 **실리콘 스트랩(`INITRMWEN`)이 1**이다.
- 원인: RMW는 ECC 전용이 아니라 **TCM 폭(64비트)보다 좁은 쓰기를 read-modify-write로 처리**하는 기능이다.
  ECC 유무와 RMW 필요성을 같은 것으로 본 추론이 어긋남의 원인.
- 심각도: 중간 (바이트/하프워드 쓰기가 TCM에서 RMW 시퀀스를 타므로 성능 모델에 영향).

**3. `CACR` bit[1]의 이름과 리셋값 — §2.5**
- 문서 주장: bit[1] = `ECCDIS`, **"0 = 캐시 ECC 활성(기본)"**.
- 실측: **리셋 직후에도 `CACR = 0x00000002`, 즉 bit[1] = 1**이다(실행 중 덤프도 동일).
- 원인: 문서가 §2.5에서 이미 지적한 TRM 내부 표기 불일치(§3.3.8 `ECCDIS` vs §4.1.4 `ECCEN`)가 여기서 갈린다.
  ARM은 "ECC는 리셋 시 활성"이라고 기술하므로, **리셋값 1 = ECC 활성**이라면 올바른 이름은 `ECCEN`이다.
  `ECCDIS`로 읽으면 "리셋 시 ECC 꺼짐"이 되어 ARM 기술과 모순된다. → **`ECCEN` 쪽이 맞다.**
- **실측 해석 재검토**: 프로브 로그는 `ECCEN=ON`으로 출력하면서 바로 아래 줄에
  "STM32F7은 ECC 로직 자체가 없어 항상 0으로 읽힌다"는 **자기모순 주석**을 붙이고 있다(값은 1인데).
  ECC 로직의 실재 여부는 이 레지스터만으로는 판정 불가 — 미확정.
- 심각도: 중간(비트 이름·기본값 해석). 비트 **위치**가 [1]인 것은 양쪽 다 동일.

**4. "216 MHz이므로 LATENCY = 7" — §5.2**
- 문서 주장: STM32F746G-DISCO는 VDD 3.3 V / 216 MHz이므로 **7 wait state**.
- 실측: 이 보드는 **200 MHz**로 돌고 `FLASH_ACR = 0x00000006` → **6 wait state**.
- 원인: 클럭 전제 오류. 216 MHz는 데이터시트 최댓값이고 실제 설정은 PLLN=400 → 200 MHz다.
  다만 문서 §5.2의 **[D] 유도표(180 < HCLK ≤ 210 → 6 WS)와는 정확히 일치**하므로, 유도표 자체는 이 데이터 포인트로 보강된다.
- 심각도: 높음 (클럭을 216으로 올리려면 WS도 7로 올려야 한다는 결론 자체는 유효).

**5. "ART = 216 MHz까지 0-wait state 등가" — §5.3 / §6**
- 문서 주장: ART 히트 시 **1 cycle 등가**, 즉 6~7 WS가 사실상 사라진다.
- 실측(데이터 읽기 기준): ITCM 경로 순차 읽기가 ART OFF **336.79 MB/s** → ART ON **492.10 MB/s**,
  즉 **×1.46**에 그친다. 4 KB 스트라이드(랜덤 근사)는 ART ON/OFF 모두 **9.13 cyc/접근으로 변화 없음**.
- 원인: ART는 F74x에서 **명령/분기 통합 캐시**다(데이터·리터럴 캐시가 아니다).
  위 측정은 **데이터 읽기**이므로 ART의 본령인 명령 페치 가속을 측정한 것이 아니다.
  "1 cycle 등가"는 명령 페치에 대한 주장이므로 이 측정으로 반증되지 않는다 —
  **다만 "ART를 켜면 플래시 접근이 전부 0 WS가 된다"고 일반화하면 틀린다.**
  (데이터 읽기에서도 ×1.46이 나온 것은 프리페치(PRFTEN)를 함께 켰기 때문으로 보인다.)
- 부수 사실: **이 보드에서 돌던 SNN 펌웨어는 ART/프리페치가 꺼진 상태였다**
  (`FLASH_ACR = 0x00000006`, ARTEN=0, PRFTEN=0). 원인은 `stm32f7xx_hal_conf.h:150`의
  `PREFETCH_ENABLE 0U` + `ART_ACCLERATOR_ENABLE` 매크로 미정의.
  게다가 옵션바이트 `BOOT_ADD0 = 0x80` → **0x00200000(ITCM 별칭)에서 부팅**하므로
  L1 I-cache가 걸리지 않는 경로다. 즉 **명령 페치에 아무 캐시도 없었다.**
- 심각도: 중간~높음 (이 프로젝트의 성능 수치 해석에 직결).

**6. "ITCM/DTCM = 1 cycle (0 WS)" — §6 표**
- 문서 주장: TCM 접근은 캐시 ON/OFF 무관하게 **1 cycle**.
- 실측: 32 B 스트라이드 읽기의 **접근당 5.16 cyc**(루프 오버헤드 2.00 cyc 포함 → 순 3.16 cyc).
  SRAM1/SRAM2는 15.66 cyc이므로 **TCM이 3배 빠른 것은 맞다.**
- 원인: **측정 조건 차이**. 문서의 "1 cycle"은 파이프라인 처리율이고, 실측값은 주소 생성 + 로드-사용
  의존 + 루프 오버헤드가 포함된 **실효 레이턴시**다(문서 §6 각주도 이 구분을 이미 언급한다).
  순차 대역폭 쪽 실측은 문서와 정합적이다 — DTCM `readLDM 457.49 MB/s`(2.29 B/cyc).
- 심각도: 낮음 (모순이 아니라 지표가 다름). 다만 §6 표의 "1 cycle"을 그대로 성능 모델에 넣으면 과대평가된다.

**7. QSPI NOR = Micron N25Q128A — §9**
- 문서 주장: 디스커버리 보드의 QSPI는 **128-Mbit N25Q128A** (UM1907 [A]).
- 실측: 메모리맵 모드 전환 **전**에 Read JEDEC ID(0x9F)를 직접 보낸 응답이
  **`0xEF4018`** = 제조사 **Winbond**(0xEF), W25Q 계열, 용량 2²⁴ = **16 MB**.
  Micron이라면 `0x20BA18`이어야 한다. 크기 스캔도 16,384 KB에서 폴트했다(문서의 128 Mbit = 16 MB와 용량은 동일).
- 원인: **보드 리비전에 따른 부품 교체**. ST BSP(`stm32746g_discovery_qspi.c`)는 N25Q128A 전제로 짜여 있는데도
  동작하는 이유는 두 부품이 표준 SPI NOR 명령(0x9F, 0xEB 등)을 공유하기 때문이다.
  **쓰기/소거를 하는 코드는 Winbond 데이터시트 기준으로 검증해야 한다**(페이지 크기·소거 시간·상태 비트가 다르다).
- 심각도: 높음 (QSPI에 쓰기를 하는 경우).

**8. 216 MHz 기준 이론 대역폭 표 — §3.2**
- 문서 주장: ITCM/DTCM/AXIM 각 **1.728 GB/s**, AHBS 864 MB/s (= 8 B × 216 MHz).
- 실측: 클럭이 200 MHz이므로 같은 계산의 상한은 **1.6 GB/s / 800 MB/s**다.
- 원인: 클럭 전제 오류.
- **실측 해석 주의**: 실제 도달 대역폭은 프로브 로그 안에서도 두 절이 서로 다르다 —
  §3 순차 대역폭 표는 DTCM `copyLDM 473.59 MB/s`, §4 D-cache 비교 표는 같은 항목을 `1409.37 MB/s`로 낸다.
  워킹셋·바이트 계수 방식이 다른 것으로 보이며, **어느 쪽이 이론 상한 대비 실효율인지는 확정 불가**다.
- 심각도: 중간(이론값), 낮음(측정 자체는 미확정으로 남김).

**9. APB 최대 주파수 — §9**
- 문서 주장: CPU/AXI/AHB 216 MHz, **APB2 108 MHz, APB1 54 MHz**.
- 실측: `RCC_CFGR = 0x0000940A` → PPRE1=/4, PPRE2=/2 → **PCLK1 = 50 MHz, PCLK2 = 100 MHz**.
- 원인: 문서 값은 **정격 최대치**, 실측은 **현재 설정값**이다. 200 MHz 기준으로 /4, /2를 쓰면 각각 50/100 MHz로
  최대치 이내다. 모순은 아니지만 페리페럴 타이밍(UART 보드레이트, 타이머 틱)을 108/54로 계산하면 틀린다.
- 심각도: 중간.

### 아직 측정하지 않음

- **캐시 유지관리 연산 비용(§4.3)** — `SCB_InvalidateDCache()`(DCISW 128회), `SCB_CleanDCache()`, `ICIALLU` 앞뒤로 DWT_CYCCNT를 읽어 dirty 라인 수를 바꿔 가며 측정.
- **pseudo-random 교체 정책(§1.1)** — way 수 + 1개의 라인을 같은 set에 순환 접근했을 때의 미스율을 LRU 예측치와 비교하면 갈린다.
- **dynamic read-allocate mode(§1.2)** — 큰 `memset` 루프를 `ACTLR.DISRAMODE = 0`과 `1`에서 각각 재고 대역폭 차이를 본다.
- **store buffer 4엔트리·머지 동작(§1.2)** — 같은 라인에 연속 store와 서로 다른 라인에 흩어진 store의 사이클을 비교(엔트리 수만큼 계단이 생긴다).
- **에라타 1259864 재현(§8.1)** — 기본 맵 WT 영역(0x00000000대)에서 store→linefill→store→load 시퀀스를 만들어 load 값이 어긋나는지 확인. 재현되면 Cat A 확정.
- **에라타 2328489(§8.2)** — TCM wait state 구성 자체를 만들 수 없으면 원리적으로 확인 불가. `RMW=1`이 확인됐으므로 64비트 store + AHBS(DMA) 동시 스트림으로 stall 여부만은 관측 가능.
- **DMA ↔ D-cache 일관성 시나리오(§7.1)** — AN4839의 재현 절차(SRAM1 WBWA에 CPU write → DMA read)를 D-cache ON/OFF로 각각 돌려 값 불일치를 직접 확인.
- **AHBS 경유 DMA → DTCM 대역폭과 ITCM DMA 불가(§3.5)** — DMA2 mem-to-mem으로 DTCM/ITCM을 목적지로 전송해 전송률과 오류 발생 여부를 본다.
- **소비 전류 178 mA / 205 mA(§5.4)** — 보드 IDD 측정 점퍼에 전류계를 물리고 ITCM 실행 vs Flash+ART OFF 실행을 비교. (200 MHz이므로 값 자체도 다를 것)
- **ART 연관도·교체 정책(§5.3)** — 2 KB(64라인 × 256비트)를 넘나드는 크기의 코드 루프를 만들어 ART ON에서 실행 시간의 계단을 찾는다.
- **I-cache 실효 동작(§5.3의 "둘 중 하나" 명제)** — 같은 코드를 0x08000000(AXIM+I-cache)과 0x00200000(ITCM+ART)에 각각 두고 실행 시간을 비교. 지금까지의 실측은 **데이터 접근 경로만** 다뤘다.
- **MPU로 SDRAM을 Normal Cacheable로 재정의했을 때의 이득(§1.3)** — 지금 실측은 MPU가 꺼진 기본 맵(=Device) 상태다. MPU 리전 하나만 설정하면 바로 잰다.

---

## 요약

- STM32F746xx의 L1 캐시는 **I-cache 4 KB + D-cache 4 KB**다. (F76x/F77x의 16 KB+16 KB와 다르므로 혼동 주의) [A]
- **D-cache = 4-way, 32 sets, 32-byte line** / **I-cache = 2-way, 64 sets, 32-byte line**. 두 캐시 모두 **pseudo-random 교체 알고리즘**을 쓴다. [A]
- `CCSIDR` 실측 예상값: D-cache(CSSELR=0) = **0xF003E019**, I-cache(CSSELR=1) = **0xF007E009**. `CTR` = **0x8303C003** (PIPT, 모든 라인 32 B). [A]
- **ITCM 16 KB @ 0x00000000**, **DTCM 64 KB @ 0x20000000**, 둘 다 CPU 클럭 **0 wait state**. TCM은 MPU 설정과 무관하게 **항상 Non-cacheable Normal**로 동작하므로 L1 D-cache 일관성 문제가 원천적으로 없다. [A]
- DTCM은 CPU의 **AHBS(32-bit AHB-Lite 슬레이브)** 포트를 통해 DMA가 접근 가능하지만, **ITCM은 ST 구현상 CPU 전용**이다. [A]
- 기본 메모리 맵에서 **Code 영역(0x00000000–0x1FFFFFFF)은 Write-Through**, **SRAM 영역(0x20000000–0x3FFFFFFF)은 Write-Back Write-Allocate**다. MPU를 켜지 않으면 이 기본값이 적용된다. [A]
- **ART Accelerator는 ITCM 인터페이스(0x00200000~)로 접근하는 Flash에만 적용**되며, F74x/F75x에서는 **256 bit × 64 line(= 2 KB)** 통합 명령/분기 캐시다. AXIM 경로(0x08000000)의 Flash는 ART가 아니라 L1 캐시가 담당한다. [A]/[B]
- VDD 2.7–3.6 V, 216 MHz(over-drive ON)에서 Flash **7 wait state**(= 액세스당 8 HCLK ≈ 37.0 ns)가 필요하다. 무대기 최대 Flash 주파수는 30 MHz다. [A]

> **실측 정정**: 이 보드는 216 MHz가 아니라 **200 MHz**로 돌고 있고,
> 실제 `FLASH_ACR = 0x00000006` → **6 wait state**다(§5.2 유도표의 180–210 MHz 구간과 일치).
> over-drive는 켜져 있다(`PWR_CR1 = 0x0003C100`, `PWR_CSR1 = 0x00034000` → ODRDY/ODSWRDY=1).
- **ARM 에라타 1259864 (Category A)** — Write-Through 메모리에 대한 store/load 시퀀스에서 데이터 손상. **r0p1 영향(r1p2에서 수정)** → STM32F746은 영향권. 기본 맵의 Code 영역이 WT이므로 주의가 필요하다. [A]
- 디스커버리 보드의 외부 SDRAM(0xC0000000, FMC bank 5)은 기본 맵에서 **Device/Non-cacheable**이다. 캐시 이득을 보려면 MPU로 Normal Cacheable 재정의가 필수다. [A]

---

## 1. L1 캐시 구조

### 1.1 용량 · 연관도 · 라인

| 항목 | I-cache | D-cache | 출처 |
|---|---|---|---|
| 용량 (STM32F746) | 4 KB | 4 KB | DS10916(DocID027590) Rev 4, 표지 feature list [A] |
| 연관도 (way) | 2-way set-associative | 4-way set-associative | ARM DDI 0489B §5.8 L1 caches [A] / AN4839 Rev 2 §2 [A] |
| 캐시 라인 | 32 byte (8 word) | 32 byte (8 word) | DDI 0489B §5.8 [A] |
| set 수 | 64 | 32 | CCSIDR.NumSets = 0x3F / 0x1F, DDI 0489B Table 3-7 [A] |
| way 하나의 크기 | 2048 byte | 1024 byte | 4096 / ways [D, 계산] |
| set index 비트 | addr[10:5] | addr[9:5] | set 수와 라인 크기로부터 유도 [D] |
| tag 비트 | addr[31:11] | addr[31:10] | 위와 동일 [D] |
| 교체 정책 | pseudo-random | pseudo-random | AN4839 Rev 2 §2: "The data cache and Instruction cache implement a pseudo-random replacement algorithm." [A] |
| 인덱싱/태깅 | PIPT | PIPT | CTR.L1Ip = 0b11 (= PIPT), DDI 0489B Table 3-1 CTR=0x8303C003 [A] |

> Cortex-M7에는 MMU가 없어 VA = PA이므로 PIPT/VIPT 구분은 실질적 의미가 없다. 다만 CTR이 L1Ip=0b11(PIPT)로 보고한다. [D, 해석]

**실무 함의(캐시 스래싱):** D-cache의 way 하나가 1 KB이므로, **1 KB 간격으로 떨어진 서로 다른 5개 이상의 주소를 반복 접근하면 같은 set에서 충돌**한다. SNN 가중치/멤브레인 전위 배열을 1 KB 정렬 배수 간격으로 배치하면 4-way가 금방 소진된다. [D, 구조로부터 유도]

### 1.2 write policy / allocate 정책

- **두 캐시 모두 read-allocate가 항상 적용**된다. D-cache는 해당 영역이 Write-Allocate로 표시된 경우에만 write miss 시 할당한다.
  - 원문(DDI 0489B §5.8): "Both caches allocate a memory location to a cache line on a cache miss because of a read, that is, all cacheable locations are Read-Allocate." [A]
- Write-Through 영역: 캐시에 쓰면서 동시에 AXIM으로도 write를 수행. Write-Back 영역: 라인을 dirty로 표시하고 eviction 시점에만 AXIM write. [A, DDI 0489B §5.8]
- **Dynamic read allocate mode**: WBWA 영역에서 "linefill 완료 전에 코어가 라인 전체를 덮어쓰는" 상황이 **연속 3회** 감지되면 자동으로 read-allocate 모드로 전환된다(= write miss 시 linefill 없이 외부로 직접 write). `memset()` 같은 대량 쓰기에서 캐시 오염을 막는 용도. `ACTLR.DISRAMODE=1`로 비활성화 가능. [A, DDI 0489B §5.8.1]
- **Store buffer**: 4 엔트리, 엔트리당 최대 64-bit 데이터 + 32-bit 주소. Normal 메모리에 한해 같은 라인끼리 병합(merge)된다. Strongly-ordered/Device는 병합하지 않는다. [A, DDI 0489B §5.8.2]
- **Linefill 동작**: critical word first + non-blocking + streaming. AXI 버스트는 **WRAP4 64-bit**(4 beat × 8 B = 32 B). linefill buffer는 2개. [A, DDI 0489B §5.4/§5.8]

### 1.3 MPU를 끈 상태(기본 메모리 맵)의 캐시 속성 — **매우 중요**

AN4839 Rev 2 Table 2 (STM32F7/H7 기본 설정) [A]:

| 주소 범위 | 영역 | 메모리 타입 | Shareability | 캐시 정책 |
|---|---|---|---|---|
| 0x00000000–0x1FFFFFFF | Code | Normal | Non-shareable | **WT** |
| 0x20000000–0x3FFFFFFF | SRAM | Normal | Non-shareable | **WBWA** |
| 0x40000000–0x5FFFFFFF | Peripheral | Device | Non-shareable | – |
| 0x60000000–0x7FFFFFFF | External RAM | Normal | Non-shareable | **WBWA** |
| 0x80000000–0x9FFFFFFF | External RAM | Normal | Non-shareable | **WT** |
| 0xA0000000–0xBFFFFFFF | External Device | Device | Shareable | – |
| 0xC0000000–0xDFFFFFFF | External Device | Device | Non-shareable | – |
| 0xE0000000–0xE00FFFFF | PPB | Strongly-ordered | Non-shareable | – |
| 0xE0100000–0xFFFFFFFF | Vendor-specific | Device | Non-shareable | – |

STM32F746 관점 해석 [D, DS10916 Table 13 register boundary와 대조]:
- Flash on AXIM `0x08000000` → Code 영역 → **Write-Through** 캐시
- DTCM `0x20000000–0x2000FFFF` → SRAM 영역이지만 **TCM이 우선하므로 캐시되지 않음** (§3.3 참고)
- SRAM1 `0x20010000–0x2004BFFF`(240 KB), SRAM2 `0x2004C000–0x2004FFFF`(16 KB) → **WBWA**
- QSPI `0x90000000` (FMC bank3/QSPI 영역) → **WT**
- **SDRAM `0xC0000000` (FMC bank 5) → Device, 캐시 불가**. 디스커버리 보드의 MT48LC4M32B2 SDRAM이 여기 매핑되므로, 캐시를 쓰려면 MPU로 Normal Cacheable로 바꿔야 한다. [A, DS10916 Table 13 + AN4839 Table 2]

> **실측 정정(확인)**: 이 절의 핵심 주장이 실측으로 확인됐다. MPU를 켜지 않은 상태에서
> SDRAM 접근은 **D-cache OFF와 WB가 완전히 동일**하다 —
> read32 59.70 vs 59.71 MB/s, 포인터 체이싱 26.06 vs 26.06 cyc, copyLDM 81.45 vs 81.46 MB/s.
> **0xC0000000은 캐싱되지 않는다**는 뜻이며, 캐시 정책 차이 때문이 아니라 Device 메모리이기 때문이다.
> QSPI(0x90000000, WT) 쪽은 반대로 D-cache를 켜면 64 KB 스트라이드 비용이 94.22 → 197.54 cyc로
> 나빠진다 — 라인필이 일어난다는 뜻이므로 **캐시 대상임을 시사**한다.
>
> ⚠ 프로브 로그(`probe_output.txt` §5 MPU)에는 "SDRAM(0xC0000000)은 기본맵에서 Normal 비공유
> 캐시 가능 영역"이라는 **잘못된 주석**이 있다. 위 측정 결과와 ARMv7-M 기본 맵(0xA0000000–0xDFFFFFFF =
> External device) 양쪽이 이 문서 쪽 서술을 지지한다. **실측 수치 > 실측 도구의 주석.**

---

## 2. 캐시 식별/제어 레지스터

### 2.1 주소와 리셋값 (ARM DDI 0489B/F Table 3-1) [A]

| 주소 | 이름 | 타입 | 리셋값 | 비고 |
|---|---|---|---|---|
| 0xE000E008 | ACTLR | RW | 0x00000000 | Auxiliary Control |
| 0xE000ED14 | CCR | RW | – | bit16=DC, bit17=IC |
| 0xE000ED78 | CLIDR | RO | 구현 의존 | Cache Level ID |
| 0xE000ED7C | CTR | RO | **0x8303C003** | Cache Type |
| 0xE000ED80 | CCSIDR | RO | 구현 의존 | Cache Size ID |
| 0xE000ED84 | CSSELR | RW | UNPREDICTABLE | 0=D-cache, 1=I-cache |
| 0xE000EF90 | CM7_ITCMCR | RW | 핀 의존 | ITCM 제어 |
| 0xE000EF94 | CM7_DTCMCR | RW | 핀 의존 | DTCM 제어 |
| 0xE000EF98 | CM7_AHBPCR | RW | 핀 의존 | AHBP 제어 |
| 0xE000EF9C | CM7_CACR | RW | – | L1 Cache Control |
| **0xE000EFA0** | CM7_AHBSCR | RW | – | AHB **Slave** Control (※ 0xE000EF98이 아님) |
| 0xE000EFA8 | CM7_ABFSR | RW | 0x00000000 | Auxiliary Bus Fault Status |

### 2.2 CTR = 0x8303C003 해석 [A + D]

| 비트 | 필드 | 값 | 의미 |
|---|---|---|---|
| [31:29] | Format | 0b100 | ARMv7 CTR 포맷 |
| [27:24] | CWG | 0x3 | Cache Write-back Granule = 2^3 word = **32 B** |
| [23:20] | ERG | 0x0 | Exclusives Reservation Granule 미보고 |
| [19:16] | DminLine | 0x3 | D-cache 최소 라인 = 2^3 word = **32 B** |
| [15:14] | L1Ip | 0b11 | L1 명령 캐시 정책 = **PIPT** |
| [3:0] | IminLine | 0x3 | I-cache 최소 라인 = **32 B** |

(필드 분해는 ARMv7-M CTR 레이아웃에 따른 해석 [D]. 리셋값 자체는 [A].)

### 2.3 CCSIDR — F746 예상 실측값 (DDI 0489B/F Table 3-7) [A]

| CSSELR | 캐시 | 크기 | **CCSIDR 전체값** | WT | WB | RA | WA | NumSets | Associativity | LineSize |
|---|---|---|---|---|---|---|---|---|---|---|
| 0x0 | Data | 4 KB | **0xF003E019** | 1 | 1 | 1 | 1 | 0x001F(=32 set) | 0x3(=4 way) | 0x1(=32 B) |
| 0x1 | Instruction | 4 KB | **0xF007E009** | 1 | 1 | 1 | 1 | 0x003F(=64 set) | 0x1(=2 way) | 0x1(=32 B) |

비트 필드: `[31]WT [30]WB [29]RA [28]WA [27:13]NumSets(=sets-1) [12:3]Associativity(=ways-1) [2:0]LineSize(=log2(words)-2)` [A]

읽는 절차: `CSSELR ← 0 또는 1` → `DSB` → `CCSIDR` 읽기. [A, DDI 0489B §4.1.3 예제 코드]

> **실측 정정(확인)**: 두 예상값이 **비트 하나 틀리지 않고 그대로 나왔다** —
> `CCSIDR(CSSELR=0) = 0xF003E019`, `CCSIDR(CSSELR=1) = 0xF007E009`.
> 레지스터뿐 아니라 **성능 곡선으로도 이중 확인**했다: 워킹셋을 키우며 SRAM1 순차 읽기를 재면
> D-cache WB에서 4 KB 235.67 → 8 KB 221.94 MB/s로 꺾이고, 포인터 체이싱 레이턴시는
> **4 KB 2.09 cyc(히트) → 8 KB 7.11 cyc(미스)** 로 급변한다. 스트라이드 스윕은 32 B에서 평탄해진다.
> 즉 **D-cache 히트 = 2 사이클, 미스 시 SRAM1 = 8~9 사이클**이다.

### 2.4 CLIDR — F746 예상값 = **0x09000003** [D, 계산]

DDI 0489B/F Table 3-5의 필드 정의 [A]로부터 유도:
- `[29:27] LoUU = 0b001` (캐시가 하나라도 구현되어 있으므로)
- `[26:24] LoC  = 0b001`
- `[2:0] Ctype1 = 0b011` (명령 + 데이터 캐시 모두 구현)
→ (1<<27) | (1<<24) | 0x3 = **0x09000003**
※ TRM은 CLIDR 리셋값을 "구현 의존"으로만 표기하므로 이 값은 계산에 의한 예측이다. 실칩에서 반드시 읽어 확인할 것.

> **실측 정정(확인)**: 실칩 판독값 `CLIDR = 0x09000003` — **계산 예측이 맞았다.** [D] → **[A]**
> (Ctype1=3 → I/D 분리 하버드, LoC=1 → 코히런시 유지가 필요한 캐시 레벨은 L1뿐, L2 없음)

### 2.5 CM7_CACR (0xE000EF9C) [A, DDI 0489B §3.3.8]

| 비트 | 이름 | 타입 | 설명 |
|---|---|---|---|
| [31:3] | – | – | Reserved, RAZ/WI |
| [2] | **FORCEWT** | RW | 1 = 모든 Cacheable 영역을 Write-Through로 강제. **리셋값 0**. D-cache 미구현 시 RAZ/WI |
| [1] | **ECCDIS** | RW | 0 = 캐시 ECC 활성(기본), 1 = 비활성. D-cache 미구현 시 RAZ/WI |
| [0] | **SIWT** | RW | 0 = Normal Cacheable **Shared** 영역을 Non-cacheable 취급(기본). 1 = Write-Through 취급 |

주의:
- ECC 필드는 **양쪽 캐시를 모두 끈 상태에서만** 변경해야 하며, 변경 후 전체 캐시를 invalidate 해야 한다. 그러지 않으면 ECC 정보 불일치로 데이터 손실이 발생한다. [A, DDI 0489B §4.1.4]
- TRM 내부에 표기 불일치가 있다(§3.3.8은 `ECCDIS`, §4.1.4 예제 코드는 같은 bit[1]을 `ECCEN`으로 부름). 비트 위치는 [1]로 동일. [A, 원문 대조]
- **FORCEWT=1은 ARM 에라타 1259864(Cat A)의 트리거 조건 중 하나**다(§7 참고). r0p1인 STM32F746에서는 사용을 피하는 편이 안전하다. [A]

> **실측 정정**: 실칩 `CACR = 0x00000002` — **리셋 직후와 실행 중이 동일**하다.
> FORCEWT=0, SIWT=0은 표와 일치하지만, **bit[1]의 리셋값이 1**이라는 점이 표의 "0 = ECC 활성(기본)"과 어긋난다.
> ARM이 "ECC는 리셋 시 활성"이라고 기술하므로, 리셋값 1이 ECC 활성을 뜻하려면 이 비트의 이름은
> `ECCDIS`가 아니라 **`ECCEN`(DDI 0489F §4.1.4의 표기)** 이어야 한다. 본문이 지적한 TRM 내부 표기 불일치는
> **실측 리셋값에 의해 `ECCEN` 쪽으로 기운다.** (비트 위치가 [1]인 것은 양쪽 동일)
> 다만 STM32F7에 캐시 ECC 로직이 실제로 있는지는 이 레지스터만으로 판정할 수 없다 — 미확정.

### 2.6 ACTLR (0xE000E008) — Cortex-M7 전용 비트 [A, DDI 0489B/F Table 3-3]

| 비트 | 이름 | 의미 |
|---|---|---|
| [27] | DISCRITAXIRUW | critical AXI read-under-write 비활성 (DDI 0489F에서 추가 문서화) |
| [26] | DISDYNADD | ADD/SUB의 EX1 조기 해결 비활성 |
| [25:21] | DISISSCH1 | 채널1 이슈 금지 마스크 ([25]VFP [24]MAC/MUL [23]LDR to PC [22]indirect br [21]direct br) |
| [20:16] | DISDI | 채널0에 해당 명령이 있을 때 dual-issue 금지 (비트 배치는 DISISSCH1과 동일) |
| [15] | DISCRITAXIRUR | critical AXI Read-Under-Read 비활성. 최악 인터럽트 지연 개선, 성능은 하락 |
| [14] | DISBTACALLOC | BTAC 신규 엔트리 할당 금지 |
| [13] | DISBTACREAD | BTAC 미사용, static branch prediction만 사용 |
| [12] | DISITMATBFLUSH | ITM/DWT ATB flush 비활성 |
| [11] | **DISRAMODE** | **WBWA 영역의 dynamic read allocate mode 비활성** |
| [10] | FPEXCODIS | FPU 예외 출력 비활성 |
| [2] | **DISFOLD** | **dual-issue 기능 비활성** (성능 하락) |
| [1:0] | – | Reserved |

> **중요 정정**: `DISDEFWBUF`, `DISMCYCINT`는 **Cortex-M3/M4의 ACTLR 비트이며 Cortex-M7에는 존재하지 않는다.** DDI 0489B/F Table 3-3에 해당 필드가 없다(bit[9:3]은 Reserved). Cortex-M7의 write buffer는 store buffer로 대체되었고 소프트웨어로 끌 수 없다. [A, 부재로 확인]

---

## 3. TCM (ITCM / DTCM)

### 3.1 STM32F746 구성

| 항목 | 값 | 출처 |
|---|---|---|
| ITCM-RAM | **16 KB @ 0x00000000** | DS10916 §2.5 (크기) [A] / 프로젝트 링커스크립트 `info/code/fw_hwprobe/stm32f746_probe.ld` L23 (주소) [C] |
| DTCM-RAM | **64 KB @ 0x20000000** | DS10916 §2.5 [A] / 같은 링커스크립트 L24 [C] |
| SRAM1 | 240 KB @ 0x20010000 | DS10916 §2.5 [A] / 링커스크립트 [C] |
| SRAM2 | 16 KB @ 0x2004C000 | DS10916 §2.5 [A] / 링커스크립트 [C] |
| Flash on AXIM | 1 MB @ 0x08000000 | 링커스크립트 [C] |
| Flash on ITCM | 0x00200000~ | 커뮤니티/AN4667 인용 [C] |
| 대기 상태 | **0 wait state, CPU 클럭 속도** | DS10916 §2.5: "It is accessed at CPU clock speed with 0-wait states." [A] |

### 3.2 TCM 인터페이스 물리 구조 [A, DDI 0489B §5.7]

- **ITCM: 64-bit 포트 1개** — dual-issue 명령 페치를 지원
- **DTCM: 32-bit 포트 2개 (D0TCM, D1TCM)** — 요청 주소의 **bit[2]** 로 선택. 즉 DTCM RAM은 64-bit 워드가 두 뱅크에 인터리브되어 있고, 두 개의 32-bit 접근을 동시에 처리할 수 있다.
  - 원문: "the DTCM has two 32-bit memory interfaces, D0TCM and D1TCM, selected on bit[2] of the request address"
- TCM 접근자는 LSU, PFU, AHBS 세 종류이며 **중재 우선순위는 LSU > PFU > AHBS**. 상위 우선순위가 접근 중이면 하위는 stall된다. [A, DDI 0489B §5.7.3]
- 프로세서는 TCM에 **투기적(speculative) read**를 발행하며, store 명령이 일부 write만 수행한 상태에서 인터럽트될 수 있다. → **TCM에 FIFO 같은 read/write-sensitive 장치를 붙이면 안 된다.** [A]

**대역폭 (216 MHz 기준, 계산값 [D]):**

| 포트 | 폭 | 이론 최대 대역폭 |
|---|---|---|
| ITCM | 64-bit | 8 B × 216 MHz = **1.728 GB/s** |
| DTCM (D0+D1) | 2 × 32-bit | 8 B × 216 MHz = **1.728 GB/s** |
| AHBS (DMA용) | 32-bit AHB-Lite | 4 B × 216 MHz = **864 MB/s** (이론 상한. single outstanding이라 실제는 훨씬 낮음) |
| AXIM | 64-bit AXI | 8 B × 216 MHz = **1.728 GB/s** |

> **실측 정정**: 클럭이 **200 MHz**이므로 같은 계산의 상한은 ITCM/DTCM/AXIM **1.6 GB/s**, AHBS **800 MB/s**다.
> 실제 도달값(실측, D-cache ON):
>
> | 영역 | read32 | readLDM | write32 | writeSTM |
> |---|---|---|---|---|
> | ITCM | 219.86 MB/s | **459.96** | 211.84 | **470.80** |
> | DTCM | 220.58 | **457.49** | 213.14 | **473.66** |
> | SRAM1 | 206.91 | 389.74 | 213.20 | 473.61 |
>
> **단순 `LDR`/`STR` 루프로는 상한의 1/7밖에 못 쓰고, `LDM`/`STM`(8레지스터)을 쓰면 2배로 뛴다.**
> ⚠ 프로브 로그 §4(D-cache 상태별 비교)는 같은 DTCM copyLDM을 `1409.37 MB/s`로 보고해
> §3의 `473.59 MB/s`와 3배 차이가 난다(워킹셋·바이트 계수 방식이 다른 것으로 보인다).
> **이론 상한 대비 실효율은 아직 확정되지 않았다.**

### 3.3 TCM과 캐시/MPU의 관계 — **핵심**

> "TCMs always behave as Non-cacheable Non-shared Normal memory, **irrespective of the memory type attributes defined in the MPU** for a memory region containing addresses held in the TCM." (DDI 0489B §5.7.1) [A]

즉:
1. **ITCM/DTCM은 L1 캐시에 절대 들어가지 않는다.** MPU로 Cacheable을 지정해도 무시된다.
2. 따라서 **DTCM에 배치한 DMA 버퍼는 D-cache clean/invalidate가 전혀 필요 없다.** DMA는 AHBS로, CPU는 DTCM 포트로 같은 SRAM에 직접 접근하므로 일관성 문제가 원천적으로 없다. [D, 위 사실 + AHBS 경로로부터 유도]
3. MPU의 **접근 권한(AP)** 은 TCM 영역에도 정상 적용된다(권한/정렬 위반 시 MemManage/UsageFault 발생). [A]
4. AN4839도 같은 취지: "This must not be confused with memories on the Tightly Couple Memory (TCM) interface, **which are not cacheable**." [A]

> **실측 정정(확인)**: TCM이 캐시 대상이 아니라는 것이 성능으로 직접 확인됐다.
> D-cache **OFF / WB / (CACR.FORCEWT=1)WT** 세 상태에서 ITCM·DTCM의 값이 **소수점까지 동일**하다 —
> ITCM read32 226.17 / write32 263.91 / copyLDM 1325.56 MB/s, DTCM read32 228.26 / write32 707.96 / copyLDM 1409.37 MB/s.
> Flash의 ITCM 별칭(0x00200000)도 마찬가지로 D-cache에 반응하지 않는다(492.10 MB/s로 동일).
> 반면 같은 물리 플래시라도 AXIM 별칭(0x08000000)은 D-cache에 ×1.91로 반응한다.
> **즉 "캐시 대상 여부는 주소가 아니라 어느 인터페이스로 접근하느냐"가 결정한다.**

### 3.4 CM7_ITCMCR / CM7_DTCMCR (0xE000EF90 / 0xE000EF94) [A, DDI 0489B §3.3.6]

| 비트 | 이름 | 타입 | 설명 |
|---|---|---|---|
| [31:7] | – | – | Reserved |
| [6:3] | **SZ** | **RO** | TCM 크기. 0b0000=없음, 0b0011=4 KB, 0b0100=8 KB, **0b0101=16 KB**, 0b0110=32 KB, **0b0111=64 KB**, 0b1000=128 KB … 0b1111=16 MB |
| [2] | RETEN | RW | Retry phase enable (외부 ECC 정정 지원용) |
| [1] | RMW | RW | Read-Modify-Write enable (TCM 폭 미만 쓰기 시 RMW 시퀀스) |
| [0] | **EN** | RW | TCM enable. **0이면 해당 주소 접근이 전부 AXIM으로 나간다** |

**F746 예상값 [D, 계산]:**
- `ITCMCR` : SZ = 0b0101(16 KB) → bits[6:3] = 0x28. TCM 활성 시 **0x29** (RMW/RETEN = 0 가정)
- `DTCMCR` : SZ = 0b0111(64 KB) → bits[6:3] = 0x38. TCM 활성 시 **0x39**
- STM32F7의 TCM에는 ECC가 없으므로 RMW/RETEN은 0일 것으로 예상하나, 리셋값은 실리콘 핀 스트랩(`INITTCMEN`/`INITRMWEN`/`INITRETRYEN`)이 결정하므로 **실칩에서 읽어 확인 필요**.

> **실측 정정 — 이 절의 예상값은 둘 다 틀렸다.**
>
> | 레지스터 | 문서 예상 | **실측** | 디코드 |
> |---|---|---|---|
> | `ITCMCR` | `0x29` | **`0x00000073`** | EN=1, **RMW=1**, RETEN=0, **SZ=0b1110** |
> | `DTCMCR` | `0x39` | **`0x0000003B`** | EN=1, **RMW=1**, RETEN=0, SZ=0b0111 (**16 KB→64 KB 예상 적중**) |
>
> 리셋 직후(UR)와 실행 중(HOTPLUG) 덤프가 **완전히 동일**하므로 소프트웨어가 바꾼 값이 아니라 스트랩 값이다.
>
> 1. **RMW = 1** — "ECC가 없으니 0"이라는 추론이 어긋났다. RMW는 ECC용이 아니라
>    **TCM 폭보다 좁은 쓰기(바이트/하프워드)를 read-modify-write로 처리**하는 기능이다.
> 2. **ITCM SZ = 0b1110** — 위 표대로 읽으면 **8 MB**인데, 온디바이스 실측은
>    **ITCM 실제 크기 = 16 KB**라고 답한다(`0x00004000`부터 버스 폴트, `0x00000000`~`0x00003FFF`는 5.16 cyc로 정상).
>    가장 그럴듯한 설명은 **SZ가 "붙어 있는 RAM 용량"이 아니라 "코어가 ITCM 인터페이스에 할당한 주소 공간 크기"**
>    라는 것이다 — 8 MB 창 안에서 ST가 `0x00000000~0x00003FFF`에는 ITCM SRAM을,
>    `0x00200000~0x002FFFFF`에는 Flash 별칭(= 옵션바이트 `BOOT_ADD0 = 0x80`의 부팅 주소)을 배치했다고 보면
>    8 MB 창과 16 KB RAM이 모순 없이 공존한다. DTCM은 창과 RAM이 우연히 같아서 예상이 맞았을 뿐일 수 있다.
>    (대안 가설: ST가 이 필드를 TRM 인코딩과 무관하게 하드와이어했다. 어느 쪽이든 **SZ로 TCM 용량을 계산하면 안 된다.**)
>
> **결론: TCM 실제 용량은 SZ 필드가 아니라 폴트 경계 실측으로 판정하라.**

TCM 활성화 코드(TRM §4.1.5) [A]:
```
CM7_ITCMCR EQU 0xE000EF90
CM7_DTCMCR EQU 0xE000EF94
    ; 각각 bit0(EN) = 1로 설정
    DSB
    ISB
```

### 3.5 AHBS 슬레이브 포트 (DMA → TCM) [A, DDI 0489B §5.6]

- **32-bit AHB-Lite 슬레이브**로, 시스템(DMA)이 ITCM / D0TCM / D1TCM에 접근할 수 있게 한다.
- 쓰기는 프로세서 내부 **Store Queue(SQ)** 에 버퍼링된다. 소프트웨어 store도 같은 SQ를 쓰며, SQ는 **in-order, 병합 없음**.
- **single outstanding transaction만 지원.** 지연을 줄이기 위해 read가 버퍼된 write를 추월할 수 있고, 의존성이 있으면 hazard 로직이 read를 stall시키고 SQ를 드레인한다. hazard 판정은 **byte 단위**.
- **MPU 조회를 하지 않는다.** privileged/unprivileged 구분도 내부적으로 하지 않는다(HPROTS 무시). TCM 보호는 시스템(외부 로직)이 책임진다.
- **exclusive/locked 액세스 미지원**, AHBS store는 내부 exclusive monitor 상태를 바꾸지 않는다.
- 프로세서가 sleep 상태여도 AHBS는 동작한다.
- ARM 주석: "The processor has **higher bandwidth to TCM than the AHBS interface**." → CPU가 직접 옮기는 편이 DMA보다 빠를 수 있다.

**STM32F746에서의 실제 제약 [A, DS10916 §2.5]:**
> "The Data TCM RAM is accessible by the GP-DMAs and peripherals DMAs through specific AHB slave of the CPU. **The TCM RAM instruction is reserved only for CPU.**"

→ 코어의 AHBS는 ITCM도 지원하지만, **ST의 STM32F7 구현에서는 DTCM만 DMA 접근 대상**이다. **ITCM으로의 DMA 전송은 지원되지 않는다.**

### 3.6 CM7_AHBSCR (0xE000EFA0) — AHBS 우선순위 제어 [A, DDI 0489B §3.3.12]

| 비트 | 이름 | 설명 |
|---|---|---|
| [31:16] | – | Reserved |
| [15:11] | **INITCOUNT** | fairness counter 초기값. **리셋값 0b01**. **0b00 금지**(livelock 위험). CTL=0b11이면 미사용 |
| [10:2] | **TPRI** | AHBS 트래픽 강등 임계 실행 우선순위. NVIC 인코딩과 동일. 0b11111111 = -1(HardFault), 0b11111110 = -2(NMI) |
| [1:0] | **CTL** | 0b00 = AHBS 우선순위 강등(**리셋값**) / 0b01 = 소프트웨어 우선순위 강등 / 0b10 = 실행 우선순위가 TPRI 이상일 때만 AHBS 강등 / 0b11 = 외부 AHBSPRI 신호가 제어 |

- 라운드로빈은 모든 모드의 기반이며, 경합이 없으면 요청자 누구나 즉시 승인된다.
- 주의: "CM7_AHBSCR settings can be **overruled** and should only be considered as a **hint** to the processor." [A]
- 변경이 즉시 반영되지 않을 수 있다(진행 중인 AHBS 트래픽 완료 후 적용).

### 3.7 CM7_AHBPCR (0xE000EF98) [A, DDI 0489B §3.3.7]

| 비트 | 이름 | 타입 | 설명 |
|---|---|---|---|
| [31:4] | – | – | Reserved |
| [3:1] | SZ | RO | AHBP 크기: 0b000=0 MB(비활성), 0b001=64 MB, 0b010=128 MB, 0b011=256 MB, 0b100=512 MB |
| [0] | EN | RW | AHBP enable. 0이면 모든 접근이 AXIM으로 나간다 |

STM32F7은 AHBP(peripheral) 포트를 사용한다(AN4839 Figure 1 "CPU AHBP bus"). F746의 SZ 실측값은 **미확인**.

> **실측 정정(확인)**: `AHBPCR = 0x00000009` → **EN = 1, SZ = 0b100 = 512 MB**.
> 리셋 직후와 실행 중 값이 같으므로 스트랩 고정값이다. 즉 페리페럴 접근(0x40000000~)은
> AXIM/캐시를 거치지 않고 전용 포트로 나가며, 코드 페치와 버스를 다투지 않는다.

---

## 4. 캐시 유지관리 연산

### 4.1 레지스터 주소 (모두 WO, PPB/SCS 영역) [A, DDI 0489B Table 3-1]

| 주소 | 이름 | 인자 | 동작 |
|---|---|---|---|
| 0xE000EF50 | **ICIALLU** | 무시 | I-cache 전체 invalidate to PoU |
| 0xE000EF54 | – | – | Reserved |
| 0xE000EF58 | **ICIMVAU** | 주소(MVA) | I-cache 주소별 invalidate to PoU |
| 0xE000EF5C | **DCIMVAC** | 주소(MVA) | D-cache 주소별 invalidate to PoC |
| 0xE000EF60 | **DCISW** | set/way | D-cache set/way invalidate |
| 0xE000EF64 | **DCCMVAU** | 주소(MVA) | D-cache 주소별 clean to PoU |
| 0xE000EF68 | **DCCMVAC** | 주소(MVA) | D-cache 주소별 clean to PoC |
| 0xE000EF6C | **DCCSW** | set/way | D-cache set/way clean |
| 0xE000EF70 | **DCCIMVAC** | 주소(MVA) | D-cache 주소별 clean+invalidate to PoC |
| 0xE000EF74 | **DCCISW** | set/way | D-cache set/way clean+invalidate |
| 0xE000EF78 | BPIALL | – | **미구현 (RAZ/WI)** |

지원 연산 요약(DDI 0489B §5.8.3) [A]:
- D-cache: invalidate/clean/clean+invalidate 각각 **by address** 와 **by set/way**
- I-cache: **invalidate all** 과 **invalidate by address** 만 (clean 없음 — 명령 캐시는 dirty가 없으므로)

### 4.2 set/way 인자 인코딩 (F746 4 KB D-cache 기준) [D, TRM §4.1.3 예제 코드 + CCSIDR로부터 유도]

| 필드 | 비트 | F746 D-cache 값 |
|---|---|---|
| Way | [31:32-log2(ways)] = **[31:30]** | 0~3 |
| Set | [log2(line)+log2(sets)-1 : log2(line)] = **[9:5]** | 0~31 |
| Level | [3:1] | 0 (L1) |

TRM 예제 코드가 계산하는 방식 [A]:
- `r6 = CLZ(ways-1) = CLZ(3) = 30` → way를 왼쪽으로 30비트 시프트 → bits[31:30]
- `r7 = LineSize + 4 = 1 + 4 = 5` → set을 왼쪽으로 5비트 시프트 → bits[9:5]

주소 기반 연산(`DCIMVAC`/`DCCMVAC`/`DCCIMVAC`/`ICIMVAU`)은 **32 byte 라인 정렬**이 전제이며, 주소 하위 5비트는 무시된다. CMSIS의 `SCB_CleanDCache_by_Addr(addr, size)`는 내부에서 32 B 단위로 반복한다. [D]

### 4.3 소요 사이클

| 연산 | 반복 횟수 (F746) | 사이클 |
|---|---|---|
| `ICIALLU` (I-cache 전체 무효화) | **1회 쓰기** | **미확인** (문서화된 사이클 수 없음) |
| D-cache 전체 invalidate (`DCISW`) | 32 set × 4 way = **128회 쓰기** | **미확인** |
| D-cache 전체 clean (`DCCSW`) | **128회 쓰기** | **미확인** |
| D-cache 전체 clean+invalidate (`DCCISW`) | **128회 쓰기** | **미확인** |
| 주소별 연산 | ⌈size/32⌉회 | **미확인** |

- 반복 횟수는 CCSIDR(32 set × 4 way)로부터 계산한 값 [D]. **ARM/ST 문서에 이 연산들의 사이클 수는 공개되어 있지 않다.**
- 단, 캐시 유지관리 연산은 store buffer를 완전히 드레인시킨다("The store buffer is completely drained when: an explicit drain request is done for cache maintenance operations"). 따라서 실제 비용은 그 시점의 dirty 라인 수와 AXI/Flash 지연에 크게 좌우된다. [A, DDI 0489B §5.8.2]
- I-cache는 4 KB / 32 B = 128 라인이지만 `ICIALLU` 한 번으로 전체 무효화되므로 D-cache 대비 훨씬 저렴하다. [D]

### 4.4 활성화/비활성화 절차 (반드시 지킬 것) [A, DDI 0489B §5.8.4, §4.1.3; AN4839 §4]

```
[활성화]
1. (D-cache) DCISW 루프로 전체 invalidate → DSB → ISB
   (I-cache) ICIALLU 쓰기 → DSB → ISB
2. CCR(0xE000ED14).DC = bit16, .IC = bit17 설정
3. DSB → ISB

[비활성화]
- D-cache: 반드시 전체 clean(DCCISW) 후 비활성화 (dirty 데이터 flush)
- I-cache: 비활성화 후 ISB로 파이프라인 flush
```
- 리셋 직후 캐시는 **비활성 상태**이며, **활성화 전 반드시 invalidate 해야 한다**(UNPREDICTABLE 동작 방지). [A]
- CMSIS 함수(AN4839 Table 1) [A]: `SCB_EnableICache()`, `SCB_DisableICache()`, `SCB_InvalidateICache()`, `SCB_EnableDCache()`, `SCB_DisableDCache()`, `SCB_InvalidateDCache()`, `SCB_CleanDCache()`, `SCB_CleanInvalidateDCache()`

---

## 5. Flash · ART Accelerator · Wait State

### 5.1 FLASH_ACR 레지스터 비트

| 비트 | 이름 | 설명 | 신뢰도 |
|---|---|---|---|
| [3:0] | **LATENCY** | Flash wait state 수 (0~15) | [B] RM0385 §3.7.1 색인 |
| [8] | **PRFTEN** | Prefetch enable (ART-prefetch) | [B]/[C] RM0385 §3.7.1 + libopencm3 `(1<<8)` |
| [9] | **ARTEN** | ART Accelerator enable | [B]/[C] 위와 동일, `(1<<9)` |
| [11] | **ARTRST** | ART cache reset. **ARTEN=0(비활성) 상태에서만 효과** | [C] libopencm3 `(1<<11)` + ST 커뮤니티 |

STM32Cube HAL 매크로: `__HAL_FLASH_ART_ENABLE()`, `__HAL_FLASH_ART_RESET()`, `__HAL_FLASH_PREFETCH_BUFFER_ENABLE()` [C]

> ⚠ RM0385 PDF 본문을 직접 열람하지 못해(ST 서버 타임아웃) 비트 위치는 RM0385 목차/색인 + libopencm3 헤더 값으로 교차 확인했다. 실칩 검증 권장.

> **실측 정정(확인)**: 실칩에서 검증됐다. 이 보드의 `FLASH_ACR = 0x00000006` → LATENCY=6, PRFTEN=0, ARTEN=0으로
> 디코드되고, 프로브 펌웨어가 **bit8/bit9를 토글하자 실제로 성능이 변했다**
> (ITCM 경로 순차 읽기 336.79 ↔ 492.10 MB/s). **LATENCY[3:0] / PRFTEN(b8) / ARTEN(b9) 비트 배치는 기능적으로 확정.**
> (ARTRST b11은 만지지 않았으므로 여전히 [C].)

### 5.2 Wait state 표

**직접 확인된 값 (DS10916 / DocID027590 Rev 4, Table 18 "Limitations depending on the operating power supply range") [A]:**

| VDD 범위 | 무대기 최대 Flash 주파수 (f_Flashmax) | 최대 HCLK vs wait state | over-drive |
|---|---|---|---|
| 1.7 – 2.1 V | 20 MHz | **180 MHz @ 8 WS** | OFF |
| 2.1 – 2.4 V | 22 MHz | **216 MHz @ 9 WS** | ON |
| 2.4 – 2.7 V | 24 MHz | **216 MHz @ 8 WS** | ON |
| **2.7 – 3.6 V** | **30 MHz** | **216 MHz @ 7 WS** | **ON** |

→ **STM32F746G-DISCO는 VDD = 3.3 V, 216 MHz 이므로 LATENCY = 7 (7 wait state)** [A]

> **실측 정정**: 이 보드는 **200 MHz**로 돌고, 실제 `FLASH_ACR = 0x00000006` → **LATENCY = 6**이다.
> 216 MHz는 데이터시트 최댓값일 뿐 이 보드의 동작점이 아니다(`RCC_PLLCFGR = 0x29406419` → PLLN=400 → 200 MHz).
> 다만 아래 **[D] 유도표의 "180 < HCLK ≤ 210 → 6 WS" 행과 정확히 일치**하므로,
> 유도표에 실측 데이터 포인트가 하나 추가된 셈이다.
> 참고: 하드웨어 리셋 직후 `FLASH_ACR = 0x00000000`(0 WS, ART/프리페치 OFF)이며,
> wait state와 ART를 올리는 것은 전적으로 소프트웨어 책임이다.

**전체 구간 표 (2.7–3.6 V) — f_Flashmax = 30 MHz로부터 유도 [D]:**

| WS (LATENCY) | HCLK 범위 |
|---|---|
| 0 | 0 < HCLK ≤ 30 MHz |
| 1 | 30 < HCLK ≤ 60 MHz |
| 2 | 60 < HCLK ≤ 90 MHz |
| 3 | 90 < HCLK ≤ 120 MHz |
| 4 | 120 < HCLK ≤ 150 MHz |
| 5 | 150 < HCLK ≤ 180 MHz |
| 6 | 180 < HCLK ≤ 210 MHz |
| **7** | **210 < HCLK ≤ 216 MHz** |

⚠ 이 표는 RM0385 Table 7의 원문을 직접 읽지 못하고 DS10916 Table 18의 f_Flashmax(30 MHz)와 216 MHz/7 WS로부터 30 MHz 등간격을 가정해 유도한 것이다 **[D]**. 0 WS = 30 MHz, 216 MHz = 7 WS 두 끝점은 [A]로 확인됨.

**타이밍 환산 [D]:**
- 7 WS = 액세스당 **8 HCLK 사이클** = 8 / 216 MHz = **약 37.0 ns**

> **실측 정정**: 이 보드 기준은 **6 WS = 7 HCLK = 7 / 200 MHz = 35.0 ns**다.
> 실측 랜덤 접근(4 KB 스트라이드, ART 무관)은 **ITCM 경로 9.13 cyc, AXIM 경로 14.34 cyc**로
> (루프 오버헤드 약 2 cyc 포함) 위 계산과 같은 크기 대역이다.
- ART가 없는 Flash 순차 접근의 실효 대역폭(라인당 32 B 가정): 32 B / 37.0 ns ≈ **864 MB/s** (라인폭이 256-bit일 때)

### 5.3 ART Accelerator

| 항목 | 내용 | 신뢰도 |
|---|---|---|
| 적용 범위 | **ITCM 인터페이스로 접근하는 Flash에만** (0x00200000~). AXIM 경로(0x08000000)에는 적용 안 됨 | [B] RM0385 §3.3.3 인용 |
| 캐시 구성 (F74x/F75x) | **통합 명령 캐시 + 분기 캐시, 256 bit × 64 line** | [B] RM0385 §3.3.3 인용 |
| 총 용량 | 64 × 32 B = **2048 byte (2 KB)** | [D] 계산 |
| (참고) F72x/F73x | 128 bit × 64 line | [B] |
| (참고) F76x/F77x | bank 모드에 따라 128 또는 256 bit × 64 line | [B] |
| 효과 | 216 MHz까지 Flash에서 **0-wait state 등가 실행** | [A] DS10916 표지 |
| 활성화 | FLASH_ACR.ARTEN(bit9) = 1, prefetch는 PRFTEN(bit8) = 1 | [B]/[C] |
| 캐시 리셋 | FLASH_ACR.ARTRST(bit11). ART를 먼저 끈 뒤 수행 | [C] |
| 연관도 | **미확인** (ST 미공개) | – |
| 교체 정책 | **미확인** (ST 미공개) | – |

**ART vs L1 I-cache 관계 — 중요** [A, AN4839 §4 "Mistakes to avoid and tips"]:
> "When using the ART accelerator, the CPU can read an instruction in just 1 clock from the internal Flash memory (like 0-wait state). **So I-cache cannot be used for the internal Flash memory.**"

즉 STM32F746에서 코드 배치는 **둘 중 하나**를 고르는 문제다:
1. **Flash on ITCM (0x00200000) + ART ON** — ART 2 KB 캐시가 처리, L1 I-cache는 관여하지 않음
2. **Flash on AXIM (0x08000000) + L1 I-cache ON** — L1 I-cache 4 KB가 처리, ART 관여 안 함

ART(2 KB, 256-bit 라인)와 L1 I-cache(4 KB, 2-way, 32 B 라인)는 서로 다른 캐시이며, 같은 코드에 동시에 적용되지 않는다. [D, 위 두 사실로부터 유도]

> **실측 정정 / 보강**: 두 경로가 서로 다른 가속기에 반응한다는 명제는 **데이터 접근 측면에서 실측으로 확인**됐다.
> (측정 루프를 `.itcm_code`에 올려 ITCM에서 실행 → 코드 페치 경로를 고정한 뒤 데이터 접근만 측정, 워킹셋 64 KB)
>
> | 경로 | D-cache | ART | 순차 읽기 | 4 KB 스트라이드 |
> |---|---|---|---|---|
> | AXIM (0x08000000) | OFF | ON | 159.96 MB/s | 14.34 cyc |
> | AXIM | OFF | OFF | **159.96** (변화 없음) | 14.34 cyc |
> | AXIM | WB | ON | **304.82 (×1.91)** | 14.34 cyc |
> | ITCM (0x00200000) | OFF | ON | **492.10** | 9.13 cyc |
> | ITCM | OFF | OFF | **336.79 (×1.46 차)** | 9.13 cyc |
> | ITCM | WB | ON | **492.10** (변화 없음) | 9.13 cyc |
>
> → **AXIM은 ART에 전혀 반응하지 않고 D-cache에만 반응**, **ITCM은 ART에만 반응하고 D-cache에 반응하지 않는다.**
> 문서의 "둘 중 하나" 명제와 정합적이다.
>
> 다만 **"0-wait state 등가"는 과장으로 읽지 말 것**: 데이터 읽기 기준 ART의 이득은 **×1.46**에 그치고,
> 랜덤 접근(4 KB 스트라이드)은 ART ON/OFF가 **9.13 cyc로 동일**하다. ART는 명령/분기 캐시이므로
> 이 측정이 "1 cycle 등가"(명령 페치에 대한 주장)를 반증하는 것은 아니지만,
> **플래시 접근 일반에 0 WS가 적용된다고 일반화하면 틀린다.** 명령 페치 쪽은 아직 미측정이다.
>
> **이 보드의 실제 상태**: `FLASH_ACR = 0x00000006` → **ARTEN = 0, PRFTEN = 0**.
> 옵션바이트 `BOOT_ADD0 = 0x80` → 부팅·실행 주소가 **0x00200000(ITCM 별칭)** 이므로 L1 I-cache도 걸리지 않는다.
> 즉 SNN 펌웨어는 **명령 페치에 아무 캐시도 없는 상태**로 돌고 있었다.
> 원인은 `ImageClassification/Core/Inc/stm32f7xx_hal_conf.h:150`의 `PREFETCH_ENABLE 0U`와
> `ART_ACCLERATOR_ENABLE` 매크로 미정의다(`HAL_Init()`이 `#if (ART_ACCLERATOR_ENABLE != 0)` 안에서만 ART를 켠다).

### 5.4 소비 전류로 본 ART 효과 (DS10916 Rev 4, 216 MHz, TA=25 °C) [A]

| 실행 위치 | 전 주변장치 ON (typ) | 전 주변장치 OFF (typ) | 출처 |
|---|---|---|---|
| **ITCM RAM** 실행 | **178 mA** | **102 mA** | Table 24 |
| **Flash on ITCM, ART 비활성** | **205 mA** | **130 mA** | Table 27 |

→ ART를 끄고 Flash에서 실행하면 ITCM RAM 실행 대비 typ **+27 mA / +28 mA** (216 MHz) [D, 두 표의 차]

참고 성능 지표: 216 MHz에서 **1082 CoreMark / 462 DMIPS** (ART + L1 캐시 사용) [B, ST STM32F7 제품 소개 자료]

> **실측 정정(미측정 항목 주의)**: 위 전류·성능 값은 모두 **216 MHz 전제**이며, 이 보드는 **200 MHz**로 돈다.
> 전류는 이 조사에서 **측정하지 않았다**(ST-LINK로는 잴 수 없다 — 보드의 IDD 측정 지점에 전류계를 물려야 한다).
> CoreMark/DMIPS도 이 보드에서 돌린 적이 없다. 클럭만 환산하면 462 DMIPS → 약 428, 1082 CoreMark → 약 1002이지만
> **이 보드의 펌웨어는 ART가 꺼진 상태로 동작했으므로 "ART + L1 캐시 사용" 조건 자체가 성립하지 않았다.**

---

## 6. 메모리 영역별 예상 접근 사이클

| 영역 | 주소 | 캐시 OFF | 캐시/가속기 ON |
|---|---|---|---|
| **ITCM-RAM** | 0x00000000 (16 KB) | **1 cycle (0 WS)** [A] | 동일 — TCM은 캐시 대상이 아님 [A] |
| **DTCM-RAM** | 0x20000000 (64 KB) | **1 cycle (0 WS)** [A] | 동일 [A] |
| **Flash on ITCM** | 0x00200000 | 8 HCLK (7 WS) [D] | ART hit 시 **1 cycle 등가** [A] |
| **Flash on AXIM** | 0x08000000 | 8 HCLK + AXI 오버헤드 [D] | L1 hit 시 1~2 cycle [D] / miss 시 32 B linefill = WRAP4 64-bit 버스트 [A] |
| **SRAM1 / SRAM2** | 0x20010000 / 0x2004C000 | AXI→AHB 브리지 경유, **정확한 사이클 미확인** | L1 hit 시 1~2 cycle [D], 기본 WBWA [A] |
| **외부 SDRAM** | 0xC0000000 | FMC/SDRAM 타이밍 의존, **미확인** | **기본 맵에서 Device → 캐시 안 됨**. MPU로 Normal Cacheable 지정 필요 [A] |
| **QSPI** | 0x90000000 | QSPI 타이밍 의존, **미확인** | 기본 맵 **WT** [A] |

> **실측 정정 — 이 표의 "미확인" 칸을 실측값으로 채운다.**
> (D-cache OFF, 32 B 스트라이드 읽기, 접근당 사이클. **스트라이드 루프 자체 오버헤드 2.00 cyc/회가 포함**된 값이다.)
>
> | 영역 | 문서 | **실측 (D$ OFF)** | **실측 (D$ ON, 히트)** |
> |---|---|---|---|
> | ITCM-RAM (0x00000000) | 1 cycle (0 WS) | **5.16 cyc** | 동일 (캐시 대상 아님) |
> | DTCM-RAM (0x20000000) | 1 cycle (0 WS) | **5.16 cyc** | 동일 (캐시 대상 아님) |
> | SRAM1 / SRAM2 | "정확한 사이클 미확인" | **15.66 cyc** | **2.0 cyc**(히트) / 8~9 cyc(미스) |
> | Flash on ITCM (0x00200000) | 8 HCLK (7 WS) | **9.13 cyc**(4 KB 스트라이드, ART 무관) | — |
> | Flash on AXIM (0x08000000) | 8 HCLK + AXI 오버헤드 | **14.34 cyc**(4 KB 스트라이드) | 순차 ×1.91 |
> | 외부 SDRAM (0xC0000000) | "미확인" | **21~26 cyc**(랜덤), 순차 60.25 MB/s | **캐시되지 않음**(기본 맵 Device) |
> | QSPI (0x90000000) | "미확인" | 순차 51.69 MB/s, 64 KB 스트라이드 **94.22 cyc** | 스트라이드가 **197.54 cyc로 악화**(라인필 발생) |
>
> 읽는 법:
> - **TCM은 AXI SRAM보다 접근 지연이 3배 낮다**(5.16 vs 15.66 cyc). 문서의 "1 cycle"과 모순이 아니라
>   **지표가 다르다** — 1 cycle은 처리율, 위 값은 주소 생성 + 로드-사용 의존 + 루프 오버헤드를 포함한 실효 레이턴시다.
>   순차 대역폭으로 보면 TCM과 SRAM 차이는 훨씬 작다(DTCM readLDM 457.49 vs SRAM1 389.74 MB/s).
> - **SRAM1과 SRAM2는 지연이 동일**해 성능상 구분되지 않는다.
> - **BKPSRAM은 읽기만 유독 느리다**(140.49 MB/s, 쓰기는 정상 211.84).
> - SDRAM에서 D$ OFF와 ON이 같은 이유는 정책 차이가 아니라 **기본 맵에서 0xC0000000이 Device**라서다
>   (§1.3의 주장이 실측으로 확인된 것). MPU로 Normal Cacheable로 바꾸면 달라진다 — 미측정.

- "1 cycle"은 파이프라인 throughput 기준이며 **load-use latency**(로드 결과를 즉시 쓰는 명령의 지연)는 이보다 크다. Cortex-M7 TRM(DDI 0489B/F)에는 명령별 사이클 표가 없어 **정확한 load-use latency는 미확인**.
- Cortex-M7 TCM은 "single-cycle random TCM access running at the full processor speed"로 설계되었다. [C, Atmel/Microchip Atmel-45151 아티클]
- 캐시 라인 32 B는 float32 8개 / int8 32개에 해당한다. SNN 가중치를 32 B 정렬 + 연속 배치하면 라인당 유효 데이터가 최대화된다. [D]

---

## 7. DMA ↔ D-cache 일관성과 MPU 권장 설정

### 7.1 문제 재현 (AN4839 §3.2) [A]

CPU가 SRAM1(기본 **WBWA**)에 데이터를 쓴 뒤 DMA로 그 영역을 읽어 DTCM으로 옮기는 시나리오에서:
- D-cache **비활성** 시: 정상 동작
- D-cache **활성** 시: DMA가 읽은 데이터가 원본과 **불일치** (CPU의 write가 캐시에만 남아 있고 SRAM1에 반영되지 않음)

### 7.2 AN4839가 제시하는 4가지 해법 [A]

1. **CPU→DMA 방향**: DMA 시작 **전** `SCB_CleanDCache()` (dirty 라인을 SRAM에 flush)
2. MPU로 해당 SRAM 영역을 **Write-Through**로 변경
3. MPU로 해당 영역을 **Shareable**로 변경 → 기본(CACR.SIWT=0)에서 D-cache가 이를 **Non-cacheable로 취급**하여 자동 우회
4. `CACR.FORCEWT = 1` 로 모든 Cacheable 영역을 Write-Through 강제

**DMA→CPU 방향**: DMA 완료 후 데이터를 읽기 **전** 반드시 `SCB_InvalidateDCache_by_Addr()` (invalidate). [A]

AN4839의 추가 권고 [A]:
- "**Always better to use non-cacheable regions for DMA buffers.** The software can use the MPU to set up a non-cacheable memory block to use as a shared memory between the CPU and DMA."
- "Do not enable cache for the memory that is being used extensively for a DMA operation."

### 7.3 STM32F746에서의 실전 권장 순위 [D, 위 사실들로부터 종합]

| 순위 | 방법 | 장점 | 단점 |
|---|---|---|---|
| 1 | **DMA 버퍼를 DTCM(0x20000000, 64 KB)에 배치** | 캐시 유지관리 **완전 불필요**, 0 WS, 결정적 | 64 KB 제한, AHBS 경합 시 CPU stall 가능 |
| 2 | MPU로 DMA 버퍼 영역을 **Normal Non-cacheable**로 지정 | 유지관리 불필요, 크기 자유 | 해당 영역 CPU 접근 성능 하락 |
| 3 | WBWA 유지 + 명시적 clean/invalidate | 성능 최적 | 실수 시 버그, 32 B 정렬·크기 관리 필요 |
| 4 | `CACR.FORCEWT=1` | 간단 | **에라타 1259864(Cat A) 트리거 조건** — r0p1인 F746에서 위험 |

### 7.4 ARMv7-M MPU 속성 인코딩 (TEX / C / B / S) [C]

| 메모리 타입 | TEX | C | B | S | 용도 |
|---|---|---|---|---|---|
| Strongly-ordered | 0b000 | 0 | 0 | – | – |
| Device (shareable) | 0b000 | 0 | 1 | – | 주변장치 |
| Normal, WT, no WA | 0b000 | 1 | 0 | 0/1 | 코드/읽기 위주 |
| Normal, WB, no WA | 0b000 | 1 | 1 | 0/1 | – |
| **Normal, Non-cacheable** | **0b001** | **0** | **0** | **1** | **DMA 공유 버퍼 (권장)** |
| Normal, WB, WA | 0b001 | 1 | 1 | 0/1 | 일반 SRAM (기본) |
| Device (non-shareable) | 0b010 | 0 | 0 | – | – |

⚠ 이 표는 ARMv7-M 아키텍처의 표준 인코딩이며, PM0253(STM32F7 Cortex-M7 프로그래밍 매뉴얼) 원문 표를 직접 확인하지 못했다. [C] — 실사용 전 PM0253 §4.2.x 확인 권장.

주의: **MPU 하위 영역(subregion)이 아니라 region 번호가 클수록 우선순위가 높다** (Cortex-M7은 8 또는 16 region, 15/7이 최고 우선순위). [A, DDI 0489B §6.1]

### 7.5 CACR.SIWT의 실제 의미 [A, DDI 0489B §3.3.8]

- `SIWT = 0` (기본): Normal Cacheable **Shared** 영역을 **Non-cacheable로 취급**한다. MPU에 지정한 inner cacheability는 무시된다. → **소프트웨어 캐시 유지관리가 전혀 필요 없다.**
- `SIWT = 1`: Shared 영역을 **Write-Through**로 취급. 모든 write는 전역 가시적이 되지만, **다른 마스터(DMA)의 갱신은 캐시 유지관리 없이는 CPU에 보이지 않는다.**

→ AN4839 해법 3(Shareable 지정)이 동작하는 이유가 바로 `SIWT=0` 기본값이다. [D]

---

## 8. 관련 에라타 (Cortex-M7 r0p1 = STM32F746)

ARM SDEN-1068427 v11.0 (2024-05-28), *Cortex-M7 (AT610) / Cortex-M7 with FPU (AT611) Software Developer Errata Notice* [A]

| ID | 카테고리 | 제목 | r0p1 영향 | 요약 |
|---|---|---|---|---|
| **1259864** | **A** | Data corruption in a sequence of Write-Through stores and loads | **예** (r0p1/r0p2/r1p0/r1p1 영향, **r1p2에서 수정**) | 아래 상세 |
| 1013783 | B | PLD might perform linefill to address that would generate a MemManage Fault | 예 (전 리비전, Open) | MPU가 fault를 낼 주소에 `PLD`를 하면 D-cache linefill이 잘못 발생 |
| 2328489 | B | TCM bandwidth sharing between AHBS writes and software stores might not function correctly when using TCM wait states | 예 (전 리비전, Open) | 아래 상세 |
| 1313001 | C | Store after cache invalidate without intervening barrier might cause inconsistent state | 예 | 캐시 invalidate 직후 배리어 없이 store 시 불일치 |
| 1267980 | C | ECC error causes data corruption when the data cache error bank registers are … | 예 | ECC 관련 |

### 8.1 에라타 1259864 (Category A) — 가장 중요 [A]

- **조건**: 다음 중 하나로 인해 해당 주소가 **Write-Through**일 때
  1. MPU가 그 주소를 WT로 설정
  2. **기본 메모리 맵을 쓰고 있고 그 주소가 기본 맵에서 WT** ← **STM32F746의 Code 영역(0x00000000–0x1FFFFFFF)과 0x80000000–0x9FFFFFFF가 해당**
  3. 메모리가 cacheable이고 `CM7_CACR.FORCEWT = 1`
  4. 메모리가 cacheable + shared이고 `CM7_CACR.SIWT = 1`
- **시퀀스**: (1) 대상 주소가 D-cache에 존재 → (2) 같은 double-word에 WT store → (3) 같은 set/way에 다른 라인의 linefill 시작 / 또는 D-cache 어디서든 ECC 에러 / 또는 **DSB 없는 캐시 유지관리 연산** → (4) 대상 주소에 store → (5) 대상 주소 load
- **결과**: load가 두 번째 store가 아니라 첫 store 또는 시퀀스 시작 시점의 캐시 값을 읽는다.
- **워크어라운드**: 직접적 해결책 **없음**. ARM 권고 = "MPU로 Write-Through 메모리를 **Write-Back으로 바꿔라**". 불가능하면 해당 코드 구간에서 캐시를 꺼라.
- **실무 함의**: STM32F746에서 **`CACR.FORCEWT=1` 및 WT 정책 사용을 피하고**, DMA 일관성은 (a) DTCM 배치 또는 (b) Non-cacheable MPU 영역 또는 (c) WBWA + 명시적 clean/invalidate로 해결하는 것이 안전하다. [D, 에라타로부터 유도]

### 8.2 에라타 2328489 (Category B) [A]

- **조건**: **TCM wait state를 쓰는 구성**에서 `STRD`/`STM`/`PUSH`/`VPUSH`/`VSTR.64`/`VSTM`(64-bit store)이 TCM을 대상으로 실행되는 동안, AHBS의 back-to-back write 스트림이 진행 중일 때
- **결과**: AHBS write가 64-bit 소프트웨어 store보다 우선권을 가져 **파이프라인 stall** (기능 오류 아님, 성능만 영향)
- **F746 적용 여부**: STM32F746의 TCM은 0 wait state로 문서화되어 있으므로 [A, DS10916 §2.5], 조건("when using TCM wait states")에 해당하지 않을 가능성이 높다. **단정 불가 — 미확인**.

### 8.3 ST 디바이스 에라타 ES0290

STM32F74xxx/F75xxx 디바이스 에라타(ES0290 Rev 7, 2019-07)의 캐시/TCM 관련 항목은 **ST 서버 다운로드 실패로 확인하지 못했다.** → §10 참고

---

## 9. STM32F746 시스템 아키텍처 요약 (참고)

DS10916 §2.6 + AN4839 Figure 1 [A]:
- **AXI → multi-AHB 브리지**: AXI4 → AHB-Lite 변환
  - 3× AXI→32-bit AHB 브리지 (AHB 버스 매트릭스 연결)
  - 1× AXI→**64-bit** AHB 브리지 (내장 Flash 연결)
- **8-layer 32-bit multi-AHB BusMatrix**: 마스터(CPU, DMA1/2, Ethernet, USB HS, LCD-TFT, DMA2D) ↔ 슬레이브(Flash, RAM, FMC, QuadSPI, AHB/APB 주변장치)
- CPU 측 포트: **ITCM / DTCM / AXIM / AHBP / AHBS**
- 최대 주파수: CPU/AXI/AHB **216 MHz**, APB2 108 MHz, APB1 54 MHz [A, DS10916]
- 디스커버리 보드 외부 메모리: 128-Mbit SDRAM (MT48LC4M32B2B5-6A, **16-bit 데이터만 사용 → 64 Mbit = 8 MB 접근 가능**), 128-Mbit QSPI NOR (N25Q128A) [A, UM1907 §6.11/§6.12]

> **실측 정정**:
> - **주파수**: 위 값은 *정격 최대치*다. 이 보드의 **현재 설정은 SYSCLK/HCLK 200 MHz, PCLK2 100 MHz, PCLK1 50 MHz**
>   (`RCC_CFGR = 0x0000940A` → HPRE=/1, PPRE2=/2, PPRE1=/4). 페리페럴 타이밍을 108/54 MHz로 계산하면 틀린다.
> - **SDRAM 크기·타이밍은 확인**: `SDCR1 = 0x00001954` → 16비트 버스 / 4뱅크 / 행 12비트 / 열 8비트 /
>   **CL2** / SDCLK = HCLK÷2 = **100 MHz** → 용량 **8 MB**. `SDRTR = 0x00000C06` → 리프레시 간격 **15.4 µs**.
>   크기 스캔도 8192 KB에서 폴트해 일치한다.
> - ⚠ **QSPI 부품은 문서와 다르다.** 메모리맵 모드로 넘어가기 전에 Read JEDEC ID(0x9F)를 직접 보낸 응답이
>   **`0xEF4018`** = **Winbond**(0xEF) W25Q 계열 **16 MB**다. N25Q128A라면 `0x20BA18`(Micron 0x20)이어야 한다.
>   **보드 리비전에 따라 QSPI 부품이 교체된 것**으로 보인다. 용량(16 MB)과 표준 명령(0x9F/0xEB)이 같아
>   ST BSP가 그대로 동작하지만, **소거 시간·페이지 크기·상태 레지스터 비트는 부품마다 다르므로
>   QSPI에 쓰기를 하는 코드는 Winbond 데이터시트로 검증해야 한다.**
>   실측 QUADSPI 설정: `CR = 0x01400311`(PRESCALER=1 → 100 MHz), `DCR = 0x00170500`(FSIZE=23 → 16 MB),
>   `CCR = 0x0F282DEB`(0xEB Quad I/O Fast Read, 메모리맵 모드).

---

## 출처

### 직접 본문을 읽은 1차 문서
1. **ARM DDI 0489B** — *ARM Cortex-M7 Processor Technical Reference Manual* (r0p2, 2014-12, 145 p). §3.3.1 ACTLR / §3.3.3 CLIDR / §3.3.4 CCSIDR / §3.3.6 ITCMCR·DTCMCR / §3.3.7 AHBPCR / §3.3.8 CACR / §3.3.12 AHBSCR / §4.1.3–4.1.8 초기화 / §5.6 AHBS / §5.7 TCM / §5.8 L1 caches / §6 MPU
   - HTML 미러: https://usermanual.wiki/Document/ARM20CortexM720Processor2020Technical20Reference20Manual.840381815/html
2. **ARM DDI 0489F** — *Arm Cortex-M7 Processor Technical Reference Manual* (172 p) — https://www.pjrc.com/teensy/DDI0489F_cortex_m7_trm.pdf (CCSIDR Table 3-7, CLIDR Table 3-5, ACTLR bit[27] 확인)
3. **ARM SDEN-1068427 v11.0** (2024-05-28) — *Cortex-M7 (AT610) and Cortex-M7 with FPU (AT611) Software Developer Errata Notice*. 에라타 1259864 / 1013783 / 2328489 / 1313001
4. **ST AN4839 Rev 2** (2018-03) — *Level 1 cache on STM32F7 Series and STM32H7 Series* — https://www.st.com/resource/en/application_note/an4839-level-1-cache-on-stm32f7-series-and-stm32h7-series-stmicroelectronics.pdf (미러: https://iq.direct/datasheets/STM32x7%20Cache%20Appnote.pdf)
5. **ST DS10916 / DocID027590 Rev 4** (2016-02) — *STM32F745xx STM32F746xx Datasheet*. 표지 feature list, §2.5 Embedded SRAM, §2.6 AXI-AHB bus matrix, Table 13 register boundary addresses, Table 18 operating power supply range, Table 24/27 current consumption — https://www.st.com/resource/en/datasheet/stm32f746ng.pdf
6. **ST UM1907 / DocID027942 Rev 3** — *Discovery kit for STM32F7 Series with STM32F746NG MCU*. §6.11 SDRAM, §6.12 QSPI
7. Atmel(현 Microchip) **Atmel-45151A** — *Run Blazingly Fast Algorithms with Cortex-M7 Tightly Coupled Memories* — https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-45151-Fast-Algorithms-with-Cortex-M7_Article.pdf

### 검색/2차 확인에 사용한 자료
8. **RM0385** (STM32F75xxx/F74xxx Reference Manual) — https://www.st.com/resource/en/reference_manual/rm0385-stm32f75xxx-and-stm32f74xxx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf — **PDF 본문 직접 열람 실패**(ST 서버 타임아웃 / 10 MB 초과). 목차·색인은 HTML 미러 https://usermanual.wiki/Document/stm32f7referencemanual.1548380844/help 로 확인 (§3.3.2 Read access latency, §3.3.3 ART Accelerator, §3.7.1 FLASH_ACR)
9. **AN4667** (STM32F7 Series system architecture and performance, DocID027643 Rev 4) — https://www.st.com/resource/en/application_note/an4667-stm32f7-series-system-architecture-and-performance-stmicroelectronics.pdf — **열람 실패**
10. **ES0290 Rev 7** (STM32F74xxx/F75xxx errata) — https://www.st.com/resource/en/errata_sheet/es0290-stm32f74xxx-and-stm32f75xxx-device-limitations-stmicroelectronics.pdf — **열람 실패**
11. libopencm3 STM32F7 flash 헤더 — https://libopencm3.org/docs/latest/stm32f7/html/f7_2flash_8h_source.html (FLASH_ACR_ARTRST=1<<11, ARTEN=1<<9, PRFTEN=1<<8)
12. Feabhas, *Introduction to the ARM Cortex-M7 Cache – Part 2 Cache Replacement Policy* — https://blog.feabhas.com/2020/10/introduction-to-the-arm-cortex-m7-cache-part-2-cache-replacement-policy/
13. ST STM32F7 시리즈 제품 소개 자료 (stmcu.jp) — https://www.stmcu.jp/wp/wp-content/uploads/files/presentation-en/STM32F7/02_STM32F7-Introduction_Product_Serie_presentation.pdf (F7x6 = L1 4K+4K, F7x7 = 16K+16K, 1082 CoreMark / 462 DMIPS)
14. ST 커뮤니티 — *HOW to use ART Accelerator in STM32F7xx* — https://community.st.com/t5/stm32-mcus-products/how-to-use-art-accelerator-in-stm32f7xx/td-p/760025
15. 로컬 프로젝트 링커 스크립트 `/Users/chacha/Desktop/git/snn-for-mcu/info/code/fw_hwprobe/stm32f746_probe.ld` (ITCM/DTCM/SRAM1/SRAM2/FLASH 주소·크기 교차 확인)

---

## 미확인 / 추가 조사 필요

1. **RM0385 Table 7 "Number of wait states according to CPU clock (HCLK) frequency" 원문** — 2.7–3.6 V 구간의 0~6 WS 경계값을 직접 확인하지 못했다. 본문 §5.2의 유도표는 [D]다. (양 끝점 0 WS≤30 MHz, 216 MHz=7 WS 는 [A])
2. **RM0385 §3.3.3 ART Accelerator 원문** — "256 bit × 64 line" 은 검색 결과 인용이며 PDF 본문 미확인. ART의 **연관도, 교체 정책, 분기 캐시 엔트리 수, 데이터(리터럴) 캐시 유무**는 ST가 공개하지 않아 미확인.
3. **ES0290 (STM32F74xxx/F75xxx 디바이스 에라타)** 의 캐시/TCM/DMA 관련 항목 — Rev Z 실리콘에 적용되는 ST 고유 제한사항 전체를 확인하지 못했다.
4. **AN4667** 의 메모리 구성별 성능 비교 표(Flash-ITCM ART ON/OFF, Flash-AXIM 캐시 ON/OFF, ITCM-RAM, DTCM-RAM, SRAM1, 외부 메모리) — 미확인. X-CUBE-32F7PERF 패키지에 원자료가 있다.
5. **캐시 유지관리 연산의 사이클 수** — ICIALLU, DCISW/DCCSW/DCCISW 1회당 사이클, 전체 무효화 총 사이клы. ARM/ST 문서에 없음. **DWT_CYCCNT(0xE0001004)로 실측 필요.**
6. **Cortex-M7의 load-use latency** (D-cache hit / DTCM hit 시) — DDI 0489B/F에 명령별 사이클 표가 없음. 실측 필요.
7. **F746의 실제 리셋 시 ITCMCR/DTCMCR/AHBPCR 값** — SZ 필드와 EN/RMW/RETEN 리셋값은 실리콘 스트랩 핀이 결정. §3.4의 0x29/0x39는 계산값이므로 실칩 읽기로 확인 필요.
8. **CLIDR 실측값** — 0x09000003은 계산값. 실칩 확인 필요.
9. **SRAM1/SRAM2/SDRAM/QSPI의 캐시 miss 시 실제 접근 사이클** — AXI→AHB 브리지 및 FMC 타이밍 의존. 실측 필요.
10. **AHBP 포트의 F746 SZ 값** (CM7_AHBPCR[3:1]) — 미확인.
11. **STM32F7 Flash의 read 폭** (256-bit 여부) — ART 라인 폭이 256-bit인 점에서 추정되나 RM0385 원문 미확인.
12. **PM0253 (STM32F7 Cortex-M7 프로그래밍 매뉴얼)** 의 MPU TEX/C/B/S 인코딩 표 원문 — §7.4 표는 ARMv7-M 표준 인코딩 기반 [C].

### 실칩에서 바로 확인할 수 있는 항목 (권장 프로브 코드)

```c
// 모두 privileged 모드에서 읽기
uint32_t clidr  = SCB->CLIDR;    // 0xE000ED78  예상 0x09000003
uint32_t ctr    = SCB->CTR;      // 0xE000ED7C  예상 0x8303C003
SCB->CSSELR = 0; __DSB();
uint32_t ccsidr_d = SCB->CCSIDR; // 예상 0xF003E019 (4KB, 4-way, 32set, 32B)
SCB->CSSELR = 1; __DSB();
uint32_t ccsidr_i = SCB->CCSIDR; // 예상 0xF007E009 (4KB, 2-way, 64set, 32B)
uint32_t itcmcr = SCB->ITCMCR;   // 0xE000EF90  예상 0x29 (SZ=0b0101=16KB, EN=1)
uint32_t dtcmcr = SCB->DTCMCR;   // 0xE000EF94  예상 0x39 (SZ=0b0111=64KB, EN=1)
uint32_t ahbpcr = SCB->AHBPCR;   // 0xE000EF98
uint32_t cacr   = SCB->CACR;     // 0xE000EF9C  FORCEWT/ECCDIS/SIWT
uint32_t ahbscr = SCB->AHBSCR;   // 0xE000EFA0  CTL/TPRI/INITCOUNT
uint32_t actlr  = SCnSCB->ACTLR; // 0xE000E008
uint32_t cpuid  = SCB->CPUID;    // 코어 리비전 (VARIANT[23:20], REVISION[3:0])
uint32_t facr   = FLASH->ACR;    // LATENCY[3:0]=7, PRFTEN(b8), ARTEN(b9), ARTRST(b11)
```
