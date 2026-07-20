# 실험 기록 (SNN on STM32F746)

2026-07-19~20에 수행한 최적화·검증·실측 실험 전체 기록.
논문(`paper_ieie/main.pdf`)의 표·그림은 모두 이 문서의 실험에서 나왔으며,
원시 데이터는 `experiments/results.json`, 재현 코드는 `experiments/` 아래에 있다.

## 0. 공통 환경

| 항목 | 값 |
|---|---|
| 보드 | STM32F746G-DISCO (Cortex-M7 @ 200MHz Over-Drive, 단정밀 FPU, I/D 캐시 활성) |
| 메모리 | SRAM 320KB, Flash 1MB |
| 툴체인 | STM32CubeIDE 내장 GNU Tools for STM32 14.3.rel1 (arm-none-eabi-gcc 14.3.1) |
| 빌드 | CMake+Ninja, Release `-O3 -funroll-loops -fomit-frame-pointer` + LTO, `ARM_MATH_DSP` |
| 모델 | LeNet-5급 SNN (Conv 3→32 5×5 → AvgPool → LIF → Conv 32→64 5×5 → AvgPool → LIF → FC 4096→10 → LIF), 94,666 파라미터, T=30, rate coding, CIFAR-10 |
| 양자화 | Brevitas QAT 가중치 int8(per-tensor) + Q15(int16) 활성값, LIF 파라미터 Q15 (β=0.9→29491, θ=1.0→32767) |
| 학습 체크포인트 | `resources/0096_acc50_snn_cifar10.pth` (에폭 96, CIFAR-10 테스트 50.72%; 학습 로그 `txt`, 최고 51.02%@에폭79) |

### 보드 시간 측정 방법 (모든 온보드 실험 공통)

1. `main.cc`에 volatile 계측 구조체를 임시 삽입 (측정 후 제거됨):
   ```c
   typedef struct { uint32_t magic, images, last_total_ms, last_nn_ms,
                    accum_total_ms, accum_nn_ms; int32_t preds[10]; } perf_stats_t;
   volatile perf_stats_t g_perf = { 0x50455246u, 0, ... };
   ```
   - `total`: 이미지 1장의 30타임스텝 루프 전체(LCD 출력 포함), `HAL_GetTick()`(1ms 해상도)
   - `nn`: `spiking_rate()+snn_forward_step()` 순수 합
   - `preds[i]`: 데모 이미지 i(/Media/000i.jpg, 정답=i)의 최종 예측
2. 주소 확인: `arm-none-eabi-nm build-release/ImageClassification.elf | grep g_perf`
3. 실행 중 판독 (보드 무정지):
   ```
   STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -q -r32 0x<addr> 64
   ```
4. 평균 = accum/images. 입력 인코딩이 시드 고정 xorshift32라 부팅마다 결정적.

---

## 1. 온보드 지연시간: Scalar / CMSIS-NN / Fusion (논문 표 1, 그림 3·4)

**질문**: 표준 TinyML 스택(TFLM+CMSIS-NN) 이식과 Conv+AvgPool+LIF 퓨전이 각각 얼마나 빠른가.

**변형** (동일 -O3+LTO, 동일 데모 워크로드):

| 변형 | 커널 구성 | 인코딩 |
|---|---|---|
| V1 Scalar | 원본 CHW 스칼라 conv/pool/LIF | 원본 `rand()` |
| V2 TFLM+CMSIS-NN(비퓨전) | 빌트인 CONV_2D/AVG_POOL_2D/FC(int16x8) + LIF 커스텀(SIMD), 아레나 128KB | LUT+xorshift32 |
| V3 TFLM+퓨전(제안) | `SNN_CONV_POOL_LIF` 커스텀 오퍼(6×6/s2 int16, SMLALD) ×2 + FC + LIF_out, 아레나 256KB | LUT+xorshift32 |
| V4 직접 퓨전 | V3와 동일 퓨전 커널을 TFLM 없이 직접 호출 | LUT+xorshift32 |

**결과** (이미지당, T=30):

| 변형 | total(ms) | NN(ms) | ms/step | 평균 이미지 수 | 데모 예측(10장) |
|---|---|---|---|---|---|
| V1 | 4,494.2 | 4,422.7 | 147.4 | 18 | (레이아웃 상이로 비교 제외) |
| V2 | 9,641.4 | 9,570.0 | 319.0 | 16 | [2,1,2,3,6,2,6,7,0,2] → 5/10 |
| V3 | **2,407.0** | 2,335.7 | 77.9 | 40 | [2,1,2,3,6,2,6,7,8,1] → 6/10 |
| V4 | 2,395.5 | 2,324.8 | 77.5 | 41 | V3와 10/10 동일 |

**해석**:
- **V2 성능 역전**: conv2(im2col 열 길이 5×5×32=800)가 CMSIS-NN s16 DSP fast 경로 적용 조건(<512)을 위반해 일반 C 커널로 폴백 → 스칼라보다 2.1배 느림. 표준 스택 이식만으로는 SNN에서 감속될 수 있음을 보여주는 핵심 발견.
- **퓨전 효과(동일 조건 비교)**: V3/V2 = **4.0×** (NN 기준 4.10×). V3/V1 = 1.87×이나 여기엔 인코딩 개선분 포함(공정 비교는 V2 기준).
- **TFLM 런타임 오버헤드** = V3−V4 = 11.5ms/이미지 ≈ **0.5%** → 인터프리터 채택 비용은 무시 가능.
- V3·V4 예측 완전 일치(동일 커널·동일 시드). V2↔V3의 2장 차이는 반올림 경로 차이(아래 실험 H4의 0.75% 드리프트)로 설명.

**주의/한계**: HAL_GetTick 1ms 해상도(이미지당 2.4~9.6초라 상대 오차 무시 가능), 변형별 평균 이미지 수 상이(측정 세션 길이 차이), 분산 미저장(평균만 보고).

**재현**: V3·V4는 HEAD 소스로 빌드(`-DSNN_USE_TFLM=ON/OFF`). V2는 `experiments/v2_unfused/`의 두 파일을 각각 `tools/`·`Core/Src/snn_tflm.cc`에 배치하고 생성기를 실행해 비퓨전 모델을 재생성한 뒤 빌드. V1은 커밋 `b10712b`의 `snn.c`+`snn_accel_init(){}` 스텁.

## 2. 온보드 메모리 (논문 표 2, 그림 5)

링커 맵(`--print-memory-usage`) + TFLM `arena_used_bytes()` 실측:

| 변형 | RAM(B) | Flash(B) | 아레나 설정 | 아레나 실사용 |
|---|---|---|---|---|
| V1 | 251,256 | 223,556 | – | – |
| V2 | 186,296 | 296,052 | 131,072 | 84,544 |
| V3 | 268,200 | 373,860 | 262,144 | **80,288** |
| V4 | 85,512 | 325,168 | – | – |

- V3 RAM은 여유 있게 설정한 256KB 아레나 포함 — SNN 실소요는 아레나 실사용 80.3KB(+코드 정적분)이며 V4의 85.5KB가 하한 근거.
- V3 Flash 분해: 가중치 195,328B(conv1 6,912 + conv2 147,456 + FC 40,960) + 코드·런타임 178,532B.
- 퓨전 후 아레나에는 LIF 상태(퍼시스턴트)·im2col 스크래치까지 포함됨(전부 아레나 일원화).

## 3. 배포 가능 영역(envelope) 추정 (논문 표 3, 그림 6)

측정된 V3 처리량(77.9ms/step에서 유효 약 72.5M MAC/s)과 층별 스케일링(conv1 2×, conv2 4×, FC 2×)으로 외삽:

| 구성 | 퓨전 가중치 Flash | MACs/step | ms/step | 판정 |
|---|---|---|---|---|
| 현 모델(32/64ch) | 195.3KB | 5.64M | 77.9 (실측) | 배포·실측 완료 |
| 채널 2×(64/128ch) | 약 685.6KB | 20.7M | 약 286 (추정) | 가능(추정) — 예산 870KB 내, 여유 184KB |
| 채널 4×(128/256ch) | 약 2,550.8KB | 79.2M | 약 1,093 (추정) | **불가**: Flash 1MB 초과 |
| VGG급 / 입력 224² | 수십 MB | – | – | 불가 |

가중치 예산 = 1,048,576 − 코드·런타임 178,532 ≈ 870KB. 실측점은 1개뿐이므로 추정치는 전부 '추정' 표기(캐시 적중률 변화 미반영 1차 근사).

## 4. 정확도 관련 측정

| 측정 | 값 | 성격 |
|---|---|---|
| QAT 학습(float 시뮬) CIFAR-10 테스트 | 50.72% (배포 체크포인트) / 최고 51.02% | 학습 로그 `txt` |
| 정수(Q15) 스칼라 호스트 하니스, 학습 분할 100장 | 44/100 | 보조 근거 (정식 테스트셋 아님; n=100 이항 95% CI ±9.7%p에서 50.72%와 모순 없음) |
| 보드 데모 10장(클래스당 1장) | V2 5/10, V3 6/10, V4=V3 | 데모 지표 (정확도 지표로 부적합) |

**갱신(2026-07-20)**: 정수 구현의 전체 테스트셋 정확도는 추가 실험 E1에서 **52.38%(10,000장)**로
실측 완료 — 6절 참조. (논문 V.5의 "미실측" 한계 서술은 갱신 필요.)

## 5. 호스트 수치 검증 (H1~H5) — `experiments/host_tests/`

실행: `experiments/host_tests/build_and_run_all.sh` (전부 PASS 상태로 저장됨).
실제 펌웨어 소스 파일을 그대로 포함(include)해 검증하며, SIMD 경로는
`simd_mock/`(QADD16/SSUB16(GE)/SEL/PKHBT/SMLALD의 ARM ARM 시맨틱 시뮬레이션)으로 호스트에서 실행한다.

| ID | 파일 | 검증 대상 | 방법 | 결과 |
|---|---|---|---|---|
| H1 | `test_mapping.c` | CMSIS requantize(mult/shift/bias64) 매핑 vs 원본 Q15 수식 | 실제 스케일(231/349, 534/697, 639/256), 랜덤 스파이크, 스파이크 크기 보상(16384) 포함 | conv ≤2 LSB, FC 0 LSB, int32 누산 여유 ≥20× |
| H2 | `test_lif.c` | LIF SIMD 커널(`snn_lif_s16.c` 실소스) vs 스칼라 시맨틱 | mem 전 범위 65,536 × 경계 입력 × {β,θ,spike} 32조합 = **18,874,368 전수** + 실호출 크기 벡터(8192/4096/홀수 꼬리) 50스텝 | 비트 일치 0 fail |
| H3 | `test_encode.c` | LUT+xorshift32 인코더(`snn_encode.c` 실소스) | 경계(p=0/1), gain/offset 클램프, 픽셀값 256종 × 4.9만 표본 발화율 vs 이론 확률 | 최악 3.28σ (512검정 최댓값 기대 범위) |
| H4 | `test_fused.c` | 퓨전 항등식 + 퓨전 커널(`snn_conv_pool_lif_s16.c` 실소스) | ① 6×6 가중치 raw 누산 == 5×5 conv 4위치 합(전 픽셀·채널) ② 커널 vs 나이브 레퍼런스 30스텝×3발화율 ③ 비퓨전 경로 대비 드리프트 ④ leftover 경로(Wo=3, Co=5) | ①②④ 비트 일치, ③ 스파이크 불일치 0.753%(반올림 1회 vs 2회, 임계 근처 타이밍 플립) |
| H5 | `test_tflm.cc` | TFLM 전체 파이프라인(실제 인터프리터+모델+커스텀 오퍼) | 독립 나이브 레퍼런스와 3발화율×30스텝 최종 출력 비교 + `arena_used_bytes()` | **900개 출력 스파이크·막전위 비트 일치**, 예측 3/3, 아레나 80,288B |

## 6. 추가 실험 (2026-07-20, E1~E5) — `experiments/eval_results.json`

기존 한계였던 "정수 구현 전체 테스트셋 정확도 미실측"과 "T 축소 트레이드오프 미검증"을 채우는 실험.
E1~E3·E5는 배포 정수 파이프라인(실제 퓨전 커널·인코더, 디바이스 비트일치 검증 경로)을
호스트에서 실행(`experiments/host_tests/eval_testset.c` + `experiments/run_eval.py`, 8프로세스 병렬).
데이터: CIFAR-10 **테스트셋** 10,000장 (`experiments/export_testset.py`로 준비).

### E1. 정수 파이프라인 전체 테스트셋 정확도
- **52.38% (5,238/10,000, T=30)** — QAT float 시뮬레이션(50.72%)과 동급(차이는 rate coding의
  확률적 재추출 변동 범위). **양자화·퓨전으로 인한 정확도 손실 없음이 전량으로 확인됨.**

### E2. 타임스텝 수 T 스케일링 (accuracy vs T, 단일 패스 누적투표)
| T | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| 정확도 | 23.1% | 45.2% | 48.9% | 50.8% | 51.6% | 52.4% |
- 지연은 T에 선형(76.7ms/step 실측)이므로: **T=20이면 지연 33% 절감에 −1.6%p**, T=10이면 3× 빠르고 −7.1%p.
- 그림: `paper_ieie/figures/fig_acc_vs_T.pdf`

### E3. 레이어별 평균 스파이크 발화율 (10,000장 × 30스텝)
| 위치 | 입력(인코딩) | LIF1(conv2 입력) | LIF2(FC 입력) |
|---|---|---|---|
| 발화율 | 47.7% | **4.1%** | **3.4%** |
- 은닉층 스파이크의 ~96%가 0 → conv2에 zero-skipping(이벤트 드리븐) 적용 시 큰 이득 잠재력의 정량 근거.
- 그림: `paper_ieie/figures/fig_spike_density.pdf`

### E4. 보드 지연 분해·분산 (V3, 53장 평균)
- 전체 2,404.8ms/이미지 (기존 실측 2,407.0 대비 0.1% 이내 재현)
- 분해: **인코딩 34.3ms(1.4%) + 순수 추론 2,299.5ms(95.6%) + LCD·기타 71ms(3.0%)** → 스텝당 인코딩 1.14ms / 추론 76.7ms
- 이미지별 시간(링버퍼 16개): 2,404~2,406ms, **표준편차 ≈0.5ms(0.02%)** — 워크로드 결정적, 평균값 보고의 타당성 근거

### E5. 퓨전 vs 비퓨전 정수 시맨틱 — 예측 일치율과 정확도 (n=1,000, 동일 스파이크 입력)
- 최종 예측 일치율 **75.8%** (보드 데모 10장에서의 8/10 일치와 부합)
- 정확도: **퓨전 53.5% vs 비퓨전 46.8%** — 두 정수 시맨틱은 "통계적 등가"가 아니라 **퓨전이 유의하게 유리**
- 원인 분석: 반올림 횟수 차이(1회 vs 2회)에 더해, 비퓨전은 **풀링 전 픽셀별 int16 포화 클리핑**이
  먼저 일어나는 반면 퓨전은 풀링 합산 후 1회만 클리핑 → 높은 발화율(입력 47.7%)에서 conv1 출력이
  자주 포화하므로 조기 클리핑이 정보를 더 깎는다. **퓨전은 속도(4.0×)뿐 아니라 정확도에서도 이득.**
- 주의: 이 결과는 기존 서술("두 경로는 통계적으로 등가") 대비 새로운 발견으로, 논문 5.5절의 해당
  문구는 "퓨전 시맨틱이 유리(+6.7%p, n=1,000)"로 갱신할 필요가 있음.

## 7. 부수 실험·측정 (세션 중 수행, 논문 미수록 포함)

- **컴파일러 플래그**: `-O2`→`-O3+LTO` 전환 시 Flash 21.3%, RAM 76.7%(당시 구성)로 빌드 확인. LTO가 ST BSP의 사전 존재 타입 불일치(wm8994/ov9655)를 노출 → 미사용 BSP 컴파일 제외로 해소.
- **워크버퍼/가중치 배치**: 퓨전 가중치는 int16 147.5KB(conv2)로 SRAM 배치 불가 → 빌드타임 생성 후 Flash 상주 결정(`tools/gen_fused_weights.c`).
- **아레나 극대화**: 256KB 설정 시 RAM 81.8%. 잔여 ~59KB는 스택+libjpeg malloc 힙 몫으로 남김(이보다 키우면 JPEG 디코드 힙 고갈 위험).
- **conv2 fast 경로 512 제약 우회 검토**: LIF 스파이크를 Q14(16384)로 낮춰 int32 누산 안전화(직접 CMSIS 경로에서 사용) — 이후 SMLALD(int64) 퓨전 커널로 대체되어 제약 자체가 소멸.
- **TFLM 트림판 flatbuffers 결함**: null allocator 폴백 제거로 빌더 사용 시 segfault → 명시 allocator 전달로 해결(모델 생성기에 반영).

## 8. 파일 맵

```
EXPERIMENTS.md                  ← 이 문서
experiments/
  results.json                  ← 온보드 실측 원시 데이터 (논문 표 1·2의 근거)
  make_figures.py               ← 그림 6종 재생성 (→ paper_ieie/figures/)
  host_tests/
    build_and_run_all.sh        ← H1~H5 일괄 빌드·실행
    test_mapping.c ... test_tflm.cc
    simd_mock/                  ← Cortex-M SIMD 인트린식 호스트 시뮬레이션
  v2_unfused/                   ← 실험 1의 V2(비퓨전 TFLM) 재구성용 소스
paper_ieie/                     ← 논문 (main.pdf), 심사 핑퐁 기록(review_log.md)
ImageClassification/tools/      ← 빌드타임 생성기 (퓨전 가중치, TFLite 모델)
txt                             ← QAT 학습 로그 (에폭별 정확도)
```
