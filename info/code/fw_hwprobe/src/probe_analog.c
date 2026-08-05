/*
 * 칩 내부 아날로그 실측: 공급 전압, 다이 온도, 저속 발진기.
 *
 * 여기서 재는 값들은 데이터시트를 아무리 읽어도 알 수 없는 것들이다.
 * "이 개체가 지금 몇 볼트로, 몇 도에서 돌고 있는가"는 이 칩에 직접 물어야만 나온다.
 * LSI는 RC 발진기라 개체마다 ±50%까지 흔들리므로 공칭값(32kHz)은 사실상 의미가 없고,
 * LSE는 보드에 크리스탈이 실장되어 있는지 자체가 확인 대상이다.
 *
 * ADC는 HAL을 쓰지 않고 레지스터를 직접 다룬다. 변환 몇 번이 전부라 HAL을 끌어올
 * 이유가 없고, 샘플링 시간 같은 결정적 요소를 눈에 보이게 두는 편이 낫다.
 */
#include "probe.h"
#include "stm32f7xx.h"

/* 팩토리 캘리브레이션 (ST가 공장에서 3.3V 기준으로 기록) */
#define VREFINT_CAL_ADDR  ((volatile uint16_t *)0x1FF0F44Au)
#define TS_CAL1_ADDR      ((volatile uint16_t *)0x1FF0F44Cu)   /* 30 °C */
#define TS_CAL2_ADDR      ((volatile uint16_t *)0x1FF0F44Eu)   /* 110 °C */
#define VREFINT_CAL_VDD_MV 3300u

/* 내부 채널 (RM0385 13.10): VREFINT=IN17, 온도센서/VBAT=IN18 (동시 선택 불가) */
#define CH_VREFINT 17u
#define CH_TS_VBAT 18u

static void adc_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
    (void)RCC->APB2ENR;

    /* ADCCLK = PCLK2 / 8. 200MHz에서 PCLK2=100MHz → 12.5MHz.
     * F746의 ADC 최대 클럭은 36MHz이므로 여유가 있다. 내부 채널은 소스 임피던스가
     * 높아 긴 샘플링이 필요하므로 클럭을 낮춰 잡는 편이 안전하다. */
    ADC->CCR = (ADC->CCR & ~ADC_CCR_ADCPRE) | (3u << ADC_CCR_ADCPRE_Pos);

    ADC1->CR1 = 0;                      /* 12비트 해상도, 스캔 없음 */
    ADC1->CR2 = 0;
    ADC1->SQR1 = 0;                     /* 변환 1개 */

    /* 내부 채널 샘플링 시간을 최대(480 사이클)로. 데이터시트가 요구하는
     * VREFINT 최소 샘플링 시간(10µs)을 12.5MHz에서 넉넉히 넘긴다. */
    ADC1->SMPR1 = 0x07FFFFFFu;          /* 채널 10~18 모두 111 */
    ADC1->SMPR2 = 0x3FFFFFFFu;          /* 채널 0~9 모두 111 */

    ADC1->CR2 |= ADC_CR2_ADON;
    /* 안정화 대기 (tSTAB). 넉넉히 돌린다. */
    for (volatile int i = 0; i < 100000; i++) { }
}

static uint32_t adc_read(uint32_t ch, int samples)
{
    ADC1->SQR3 = ch;
    uint32_t acc = 0;
    for (int i = 0; i < samples; i++) {
        ADC1->SR = 0;
        ADC1->CR2 |= ADC_CR2_SWSTART;
        while (!(ADC1->SR & ADC_SR_EOC)) { }
        acc += ADC1->DR & 0xFFFFu;
    }
    return acc / (uint32_t)samples;
}

/* LSI/LSE 주파수 실측.
 * TIM5의 입력 캡처 채널 4는 GPIO 대신 LSI/LSE/RTC 웨이크업으로 리맵할 수 있다
 * (RM0385 TIM5_OR.TI4_RMP). 저속 클럭의 상승엣지 간격을 TIM5(APB1 타이머 클럭)로
 * 세면, 알고 있는 고속 클럭을 자로 삼아 모르는 저속 클럭을 재는 셈이 된다. */
static uint32_t measure_lowspeed_clock(uint32_t ti4_rmp, uint32_t timer_clk_hz)
{
    RCC->APB1ENR |= RCC_APB1ENR_TIM5EN;
    (void)RCC->APB1ENR;

    TIM5->CR1  = 0;
    TIM5->PSC  = 0;
    TIM5->ARR  = 0xFFFFFFFFu;           /* TIM5는 32비트 카운터 */
    TIM5->OR   = ti4_rmp << TIM_OR_TI4_RMP_Pos;
    TIM5->CCMR2 = (1u << TIM_CCMR2_CC4S_Pos);   /* CC4를 입력으로, TI4에 매핑 */
    TIM5->CCER = TIM_CCER_CC4E;                 /* 상승엣지 캡처 */
    TIM5->EGR  = TIM_EGR_UG;
    TIM5->SR   = 0;
    TIM5->CR1  = TIM_CR1_CEN;

    /* 첫 캡처 두 번은 버린다 (리맵 직후 과도 상태) */
    uint32_t cap[6];
    int n = 0;
    uint32_t guard = 0;
    while (n < 6) {
        if (TIM5->SR & TIM_SR_CC4IF) {
            cap[n++] = TIM5->CCR4;
            TIM5->SR = ~TIM_SR_CC4IF;
            guard = 0;
        }
        /* 클럭이 죽어 있으면 영원히 안 온다. 대략 0.5초쯤에서 포기한다. */
        if (++guard > 40000000u) { TIM5->CR1 = 0; return 0; }
    }
    TIM5->CR1 = 0;

    /* 마지막 3주기의 평균 주기 → 주파수 */
    uint32_t period = (cap[5] - cap[2]) / 3u;
    if (period == 0) return 0;
    return timer_clk_hz / period;
}

void probe_run_analog(void)
{
    probe_section("칩 내부 아날로그 실측 (이 개체 고유값)");

    const uint16_t vrefint_cal = *VREFINT_CAL_ADDR;
    const uint16_t ts_cal1     = *TS_CAL1_ADDR;
    const uint16_t ts_cal2     = *TS_CAL2_ADDR;

    probe_printf("팩토리 캘리브레이션 값:\n");
    probe_printf("  VREFINT_CAL @0x1FF0F44A = %u  (3.3V, 30'C 에서 측정된 VREFINT ADC 코드)\n", vrefint_cal);
    probe_printf("  TS_CAL1     @0x1FF0F44C = %u  (30'C)\n", ts_cal1);
    probe_printf("  TS_CAL2     @0x1FF0F44E = %u  (110'C)\n", ts_cal2);

    if (vrefint_cal == 0xFFFFu || vrefint_cal == 0) {
        probe_printf("  !! 캘리브레이션 값이 비어 있다. 전압/온도 환산을 신뢰할 수 없다.\n");
    }

    adc_init();

    /* --- 공급 전압 (VDDA) --- */
    /* ADC->CCR.TSVREFE 가 VREFINT와 온도센서를 함께 켠다 */
    ADC->CCR |= ADC_CCR_TSVREFE;
    ADC->CCR &= ~ADC_CCR_VBATE;          /* VBAT와 온도센서는 같은 채널을 공유한다 */
    for (volatile int i = 0; i < 100000; i++) { }

    uint32_t vref_raw = adc_read(CH_VREFINT, 64);
    uint32_t vdda_mv = 0;
    if (vref_raw) {
        /* VDDA = 3.3V * VREFINT_CAL / VREFINT_측정값 */
        vdda_mv = (VREFINT_CAL_VDD_MV * (uint32_t)vrefint_cal) / vref_raw;
    }
    probe_printf("\nVREFINT 측정 코드 = %u (12비트, 64회 평균)\n", vref_raw);
    probe_printf("→ VDDA = %u.%03u V   ; 이 보드에 지금 실제로 걸린 아날로그 공급 전압\n",
                 vdda_mv / 1000u, vdda_mv % 1000u);

    /* --- 다이 온도 --- */
    uint32_t ts_raw = adc_read(CH_TS_VBAT, 64);
    probe_printf("\n온도센서 측정 코드 = %u (12비트, 64회 평균)\n", ts_raw);
    if (ts_cal2 > ts_cal1 && vdda_mv) {
        /* 캘리브레이션은 3.3V 기준이므로 측정값을 3.3V 스케일로 환산한 뒤 대입한다 */
        uint32_t ts_scaled = (ts_raw * vdda_mv) / VREFINT_CAL_VDD_MV;
        /* T = (110-30)/(CAL2-CAL1) * (TS - CAL1) + 30, 소수 2자리까지 정수 연산 */
        int32_t num = (int32_t)(80 * 100) * ((int32_t)ts_scaled - (int32_t)ts_cal1);
        int32_t den = (int32_t)ts_cal2 - (int32_t)ts_cal1;
        int32_t t_c100 = num / den + 3000;
        probe_printf("→ 다이 온도 = %d.%02u 'C  (VDDA 보정 후 코드 %u)\n",
                     t_c100 / 100, (unsigned)((t_c100 < 0 ? -t_c100 : t_c100) % 100), ts_scaled);
        probe_printf("   공식: T = 80/(TS_CAL2-TS_CAL1) * (TS_DATA*VDDA/3.3 - TS_CAL1) + 30\n");
    } else {
        probe_printf("→ 온도 환산 불가 (캘리브레이션 값 이상)\n");
    }

    /* --- VBAT --- */
    ADC->CCR &= ~ADC_CCR_TSVREFE;
    ADC->CCR |= ADC_CCR_VBATE;
    for (volatile int i = 0; i < 100000; i++) { }
    uint32_t vbat_raw = adc_read(CH_TS_VBAT, 64);
    ADC->CCR &= ~ADC_CCR_VBATE;
    if (vdda_mv) {
        /* VBAT 채널은 내부적으로 1/4 분압되어 들어온다 (RM0385) */
        uint32_t vbat_mv = (vbat_raw * vdda_mv * 4u) / 4096u;
        probe_printf("\nVBAT 측정 코드 = %u → VBAT = %u.%03u V (내부 1/4 분압 보정 후)\n",
                     vbat_raw, vbat_mv / 1000u, vbat_mv % 1000u);
        probe_printf("   주: 이 보드에 백업 배터리가 없으면 VBAT은 VDD에 묶여 있다.\n");
    }

    /* --- 저속 발진기 --- */
    probe_section("저속 발진기 실측 (LSI / LSE)");

    /* LSI 기동 */
    RCC->CSR |= RCC_CSR_LSION;
    uint32_t guard = 0;
    while (!(RCC->CSR & RCC_CSR_LSIRDY) && ++guard < 10000000u) { }
    int lsi_ready = (RCC->CSR & RCC_CSR_LSIRDY) ? 1 : 0;
    probe_printf("LSI: LSIRDY=%d\n", lsi_ready);

    if (lsi_ready) {
        /* TIM5는 APB1 타이머. APB1 프리스케일러가 1이 아니면 타이머 클럭은 PCLK1의 2배다. */
        uint32_t tim_clk = (g_pclk1_hz == g_hclk_hz) ? g_pclk1_hz : g_pclk1_hz * 2u;
        uint32_t lsi_hz = measure_lowspeed_clock(1u, tim_clk);   /* TI4_RMP=01 → LSI */
        if (lsi_hz) {
            probe_printf("→ LSI 실측 = %u Hz  (공칭 32000 Hz, TIM5 타이머클럭 %u Hz 기준)\n",
                         lsi_hz, tim_clk);
            int32_t dev = ((int32_t)lsi_hz - 32000) * 1000 / 32000;
            probe_printf("   공칭 대비 편차 %d.%u %%  ; RC 발진기라 개체·온도별로 크게 흔들린다\n",
                         dev / 10, (unsigned)((dev < 0 ? -dev : dev) % 10));
        } else {
            probe_printf("→ LSI 캡처 실패 (엣지가 오지 않음)\n");
        }
    }

    /* LSE 존재 여부: 백업 도메인 쓰기를 열고 LSEON을 세운 뒤 LSERDY를 기다린다.
     * 크리스탈이 실장되어 있지 않으면 영원히 준비되지 않으므로 타임아웃으로 판정한다. */
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    (void)RCC->APB1ENR;
    PWR->CR1 |= PWR_CR1_DBP;
    while (!(PWR->CR1 & PWR_CR1_DBP)) { }

    int lse_was_on = (RCC->BDCR & RCC_BDCR_LSEON) ? 1 : 0;
    RCC->BDCR |= RCC_BDCR_LSEON;
    guard = 0;
    /* LSE 기동은 느리다(수백 ms). 200MHz에서 넉넉히 1초 정도 기다린다. */
    while (!(RCC->BDCR & RCC_BDCR_LSERDY) && ++guard < 200000000u) { }
    int lse_ready = (RCC->BDCR & RCC_BDCR_LSERDY) ? 1 : 0;

    probe_printf("\nLSE: 시도 전 LSEON=%d, 기동 후 LSERDY=%d (BDCR=0x%08X)\n",
                 lse_was_on, lse_ready, RCC->BDCR);
    if (lse_ready) {
        uint32_t tim_clk = (g_pclk1_hz == g_hclk_hz) ? g_pclk1_hz : g_pclk1_hz * 2u;
        uint32_t lse_hz = measure_lowspeed_clock(2u, tim_clk);   /* TI4_RMP=10 → LSE */
        if (lse_hz)
            probe_printf("→ LSE 실측 = %u Hz (공칭 32768 Hz) ; 32.768kHz 크리스탈이 실장되어 있다\n", lse_hz);
        else
            probe_printf("→ LSE는 준비됐으나 캡처 실패\n");
    } else {
        probe_printf("→ LSE 미기동. 이 보드에 32.768kHz 크리스탈이 실장되어 있지 않거나 연결되지 않았다.\n");
        RCC->BDCR &= ~RCC_BDCR_LSEON;    /* 켜 둔 채로 두지 않는다 */
    }

    /* --- HSE 기동 시간 --- */
    probe_printf("\nHSE: 현재 HSEON=%u HSERDY=%u HSEBYP=%u\n",
                 (RCC->CR & RCC_CR_HSEON) ? 1u : 0u,
                 (RCC->CR & RCC_CR_HSERDY) ? 1u : 0u,
                 (RCC->CR & RCC_CR_HSEBYP) ? 1u : 0u);
    probe_printf("   HSEBYP=0 이면 크리스탈 발진 모드, 1이면 외부 클럭 입력 모드다.\n");
    probe_printf("   주: SYSCLK이 HSE 기반 PLL로 돌고 있으므로 HSE를 껐다 켜는 실험은 하지 않는다.\n");
}
