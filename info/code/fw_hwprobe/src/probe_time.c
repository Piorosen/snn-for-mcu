/*
 * 타이밍 및 시스템 상태 조작 헬퍼.
 *
 * 사이클 측정은 DWT_CYCCNT를 쓴다. 이 카운터는 DEMCR.TRCENA가 0이면 아예 동작하지
 * 않고, ST-LINK로 읽은 리셋 직후 상태에서도 DWT_CTRL이 0으로 보인다. 그래서
 * probe_time_init()에서 TRCENA를 먼저 켠 뒤 카운터를 활성화한다.
 */
#include "probe.h"
#include "stm32f7xx.h"

uint32_t g_sysclk_hz;
uint32_t g_hclk_hz;
uint32_t g_pclk1_hz;
uint32_t g_pclk2_hz;

void probe_time_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    /* 일부 코어는 TRCENA 직후 카운터 레지스터에 한 번 써 줘야 활성화된다 */
    DWT->LAR = 0xC5ACCE55;            /* 락 해제 (Cortex-M7에서 필요할 수 있음) */
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    PROBE_BARRIER();
}

uint32_t probe_cycles_now(void)
{
    return DWT->CYCCNT;
}

void probe_dwt_counters_reset(void)
{
    /* 이벤트 카운터는 8비트이며 오버플로 시 이벤트를 낸다. 긴 구간에서는
     * 오버플로가 나므로, 짧은 구간에서만 의미가 있다. */
    DWT->CTRL &= ~(DWT_CTRL_CPIEVTENA_Msk | DWT_CTRL_EXCEVTENA_Msk |
                   DWT_CTRL_SLEEPEVTENA_Msk | DWT_CTRL_LSUEVTENA_Msk |
                   DWT_CTRL_FOLDEVTENA_Msk);
    DWT->CPICNT   = 0;
    DWT->EXCCNT   = 0;
    DWT->SLEEPCNT = 0;
    DWT->LSUCNT   = 0;
    DWT->FOLDCNT  = 0;
    DWT->CYCCNT   = 0;
    DWT->CTRL |= DWT_CTRL_CPIEVTENA_Msk | DWT_CTRL_EXCEVTENA_Msk |
                 DWT_CTRL_SLEEPEVTENA_Msk | DWT_CTRL_LSUEVTENA_Msk |
                 DWT_CTRL_FOLDEVTENA_Msk;
    PROBE_BARRIER();
}

void probe_dwt_counters_read(probe_dwt_counters_t *out)
{
    PROBE_BARRIER();
    out->cyc   = DWT->CYCCNT;
    out->cpi   = DWT->CPICNT   & 0xFFu;
    out->exc   = DWT->EXCCNT   & 0xFFu;
    out->sleep = DWT->SLEEPCNT & 0xFFu;
    out->lsu   = DWT->LSUCNT   & 0xFFu;
    out->fold  = DWT->FOLDCNT  & 0xFFu;
}

/* ---- 캐시 / ART 제어 ----------------------------------------------------- */

void probe_icache_enable(void)  { SCB_EnableICache(); }
void probe_icache_disable(void) { SCB_DisableICache(); }

void probe_dcache_enable(void)
{
    /* 켜기 전에 무효화하지 않으면 예전 라인이 살아나 잘못된 데이터를 읽는다.
     * SCB_EnableDCache()가 내부적으로 무효화를 수행한다. */
    SCB_EnableDCache();
}

void probe_dcache_disable(void)
{
    /* 끄기 전에 클린해야 write-back 상태의 더티 라인이 메모리에 반영된다.
     * SCB_DisableDCache()가 클린+무효화를 수행한다. */
    SCB_DisableDCache();
}

void probe_art_enable(void)
{
    FLASH->ACR |= FLASH_ACR_PRFTEN;
    FLASH->ACR |= FLASH_ACR_ARTEN;
    __DSB(); __ISB();
}

void probe_art_disable(void)
{
    FLASH->ACR &= ~FLASH_ACR_ARTEN;
    FLASH->ACR &= ~FLASH_ACR_PRFTEN;
    __DSB(); __ISB();
}

void probe_art_reset(void)
{
    /* ARTRST는 ARTEN=0 상태에서만 의미가 있다 (RM0385 FLASH_ACR 설명) */
    uint32_t acr = FLASH->ACR;
    FLASH->ACR &= ~FLASH_ACR_ARTEN;
    FLASH->ACR |= FLASH_ACR_ARTRST;
    __DSB(); __ISB();
    FLASH->ACR &= ~FLASH_ACR_ARTRST;
    FLASH->ACR = acr;
    __DSB(); __ISB();
}

/* ---- 클럭 재설정 --------------------------------------------------------- */
/*
 * 부팅 진행 표식. probe_io_init()이 돌기 전에는 UART도 결과 버퍼도 없으므로,
 * 어디까지 갔는지를 이 고정 주소에 남긴다. 펌웨어가 멈추면 ST-LINK로 이 주소를
 * 읽어 어느 단계에서 걸렸는지 바로 알 수 있다. (전용 DIAG 영역이라 결과 버퍼와 겹치지 않는다)
 */
#define BOOT_STEP (*(volatile uint32_t *)PROBE_DIAG_BOOT_STEP)
#define STEP(n)   do { BOOT_STEP = (n); __DSB(); } while (0)

/* 무한 대기를 만들지 않는다. 플래그가 안 서면 표식을 남기고 진행하거나 실패를 돌려준다. */
#define WAIT_FLAG(cond, step_on_fail)                                   \
    do {                                                                \
        uint32_t _g = 0;                                                \
        while (!(cond)) {                                               \
            if (++_g > 50000000u) { STEP(step_on_fail); return -2; }     \
        }                                                               \
    } while (0)

/*
 * HSE=25MHz(보드 크리스탈) 기준.
 *   200MHz: PLLM=25, PLLN=400, PLLP=2  → VCO 400MHz, 6 wait state
 *   216MHz: PLLM=25, PLLN=432, PLLP=2  → VCO 432MHz, 7 wait state
 * 두 경우 모두 over-drive가 필요하다. APB1 ≤54MHz, APB2 ≤108MHz 제약 때문에 /4, /2로 나눈다.
 *
 * 순서가 중요하다. ST의 HAL이 쓰는 순서를 그대로 따른다:
 *   PWR 클럭 → VOS 설정 → HSE 기동 → PLL 기동 → over-drive → wait state → SYSCLK 전환
 * 특히 VOS를 설정한 뒤 PWR_CSR1.VOSRDY를 기다리면 안 된다. 이 칩에서 HSI로 도는
 * 동안에는 그 플래그가 서지 않아 영원히 멈춘다 (실측으로 확인함 — HAL도 기다리지 않는다).
 * over-drive는 PLL이 lock된 뒤에 켜야 ODRDY/ODSWRDY가 정상적으로 올라온다.
 */
int probe_set_sysclk(uint32_t target_mhz)
{
    uint32_t plln, latency;
    if (target_mhz == 216)      { plln = 432; latency = FLASH_ACR_LATENCY_7WS; }
    else if (target_mhz == 200) { plln = 400; latency = FLASH_ACR_LATENCY_6WS; }
    else return -1;

    STEP(0x10);
    /* 1) 안전한 클럭(HSI)으로 먼저 옮겨야 PLL을 만질 수 있다 */
    RCC->CR |= RCC_CR_HSION;
    WAIT_FLAG(RCC->CR & RCC_CR_HSIRDY, 0x11);
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_HSI;
    WAIT_FLAG((RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_HSI, 0x12);

    /* 2) 전원 스케일링. VOSRDY는 기다리지 않는다 (위 주석 참조). */
    STEP(0x20);
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    (void)RCC->APB1ENR;
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS) | PWR_CR1_VOS;   /* Scale 1 = 0b11 */
    (void)PWR->CR1;

    /* 3) HSE 기동 */
    STEP(0x30);
    RCC->CR |= RCC_CR_HSEON;
    WAIT_FLAG(RCC->CR & RCC_CR_HSERDY, 0x31);

    /* 4) PLL 재구성 */
    STEP(0x40);
    RCC->CR &= ~RCC_CR_PLLON;
    WAIT_FLAG(!(RCC->CR & RCC_CR_PLLRDY), 0x41);
    RCC->PLLCFGR = (25u << RCC_PLLCFGR_PLLM_Pos)
                 | (plln << RCC_PLLCFGR_PLLN_Pos)
                 | (0u   << RCC_PLLCFGR_PLLP_Pos)     /* 00 = /2 */
                 | RCC_PLLCFGR_PLLSRC_HSE
                 | (9u   << RCC_PLLCFGR_PLLQ_Pos);
    RCC->CR |= RCC_CR_PLLON;
    WAIT_FLAG(RCC->CR & RCC_CR_PLLRDY, 0x42);

    /* 5) over-drive: PLL이 lock된 뒤에 켠다 */
    STEP(0x50);
    PWR->CR1 |= PWR_CR1_ODEN;
    WAIT_FLAG(PWR->CSR1 & PWR_CSR1_ODRDY, 0x51);
    PWR->CR1 |= PWR_CR1_ODSWEN;
    WAIT_FLAG(PWR->CSR1 & PWR_CSR1_ODSWRDY, 0x52);

    /* 6) wait state를 올린 뒤 (전환 전에 반드시) 프리스케일러 설정 */
    STEP(0x60);
    FLASH->ACR = (FLASH->ACR & ~FLASH_ACR_LATENCY) | latency;
    WAIT_FLAG((FLASH->ACR & FLASH_ACR_LATENCY) == latency, 0x61);

    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
              | RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2;

    /* 7) SYSCLK을 PLL로 전환 */
    STEP(0x70);
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    WAIT_FLAG((RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL, 0x71);

    g_sysclk_hz = target_mhz * 1000000u;
    g_hclk_hz   = g_sysclk_hz;
    g_pclk1_hz  = g_sysclk_hz / 4u;
    g_pclk2_hz  = g_sysclk_hz / 2u;
    SystemCoreClock = g_hclk_hz;
    STEP(0x7F);
    return 0;
}
