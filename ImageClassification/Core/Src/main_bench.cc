/*
 * SNN 추론 벤치마크 하네스 (CMake 옵션 SNN_BENCH=ON 일 때 main.cc 대신 빌드된다)
 *
 * 목적: 최적화의 효과를 흔들림 없이 재는 것. 그래서 애플리케이션 main.cc가 하는 일 중
 * 추론과 무관한 것을 전부 뺐다 — SD카드 읽기, JPEG 디코딩, LCD 출력이 없다.
 * 입력은 플래시에 박아 둔 고정 이미지 8장이고, 인코딩 RNG는 시드 고정 xorshift32라
 * 같은 바이너리를 다시 돌리면 사이클 수까지 거의 그대로 재현된다.
 *
 * 시간은 HAL_GetTick(1 ms 해상도) 대신 DWT_CYCCNT(1 사이클 해상도)로 잰다.
 * 200 MHz에서 1 ms = 200,000 사이클이므로, 기존 방식으로는 보이지 않던
 * 레이어 단위 차이가 드러난다.
 *
 * 결과는 g_snn_bench 전역 구조체에 남는다. 호스트에서 회수하는 방법:
 *   arm-none-eabi-nm build-bench/ImageClassification.elf | grep g_snn_bench
 *   STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -q -r32 0x<addr> 0x100
 * (info/code/run_snn_bench.py 가 이 과정을 자동화한다)
 */

#include "main.h"
#include "snn.h"

#include <string.h>

extern "C" {

/* stm32f7xx_it.c의 IRQ 핸들러가 참조하는 핸들. 벤치는 LCD/DMA2D를 초기화하지
 * 않으므로 해당 IRQ는 발생하지 않지만, 링크를 위해 정의는 있어야 한다. */
LTDC_HandleTypeDef hltdc;
DMA2D_HandleTypeDef hdma2d;

/* 벤치 입력 (snn_bench_images.c, 자동 생성) */
#define SNN_BENCH_IMAGE_COUNT 8
extern const uint8_t g_bench_images[SNN_BENCH_IMAGE_COUNT][32 * 32 * 3];
extern const int32_t g_bench_labels[SNN_BENCH_IMAGE_COUNT];

/* 레이어별 계측 카운터. TFLM 경로(snn_tflm.cc)가 강한 정의로 대체해 누적한다.
 * 직접 퓨전 경로(SNN_USE_TFLM=OFF)에는 계측점이 없으므로 이 weak 정의가
 * 남아 0으로 기록된다. */
#if defined(SNN_BENCH)
volatile uint32_t g_snn_cyc_cpl1 __attribute__((weak)) = 0;
volatile uint32_t g_snn_cyc_cpl2 __attribute__((weak)) = 0;
volatile uint32_t g_snn_cyc_lifout __attribute__((weak)) = 0;
volatile uint32_t g_snn_cyc_memcpy __attribute__((weak)) = 0;
#endif

/* TFLM 경로에서만 존재한다 (snn_tflm.cc). SNN_USE_TFLM=OFF(직접 퓨전 경로)
 * 빌드에서도 링크되도록 weak 선언하고, 부재 시 0을 기록한다. */
size_t snn_tflm_arena_used(void) __attribute__((weak));

/* ------------------------------------------------------------------ */
/* 결과 구조체                                                          */
/* ------------------------------------------------------------------ */
#define SNN_BENCH_MAGIC   0x424E4E53u   /* 'SNNB' */
#define SNN_BENCH_T       30

typedef struct {
    uint32_t magic;
    uint32_t done;            /* 1 = 측정 완료 */
    uint32_t opt_flags;       /* 어떤 최적화가 켜진 빌드인지 (아래 OPT_* 비트) */
    uint32_t sysclk_hz;

    uint32_t images;          /* 측정에 쓴 이미지 수 (워밍업 제외) */
    uint32_t timesteps;       /* 이미지당 타임스텝 */

    /* 이미지 1장당 사이클 (측정 이미지 평균) */
    uint32_t cyc_per_image;
    uint32_t cyc_encode;      /* 그중 인코딩 */
    uint32_t cyc_invoke;      /* 그중 추론(Invoke) */

    /* 레이어별 (이미지 1장, T스텝 합) */
    uint32_t cyc_cpl1;
    uint32_t cyc_cpl2;
    uint32_t cyc_lifout;
    uint32_t cyc_memcpy;      /* 입출력 텐서 복사 */

    /* 회귀 검증: 최적화가 계산 결과를 바꾸지 않았는지 */
    int32_t  preds[SNN_BENCH_IMAGE_COUNT];
    int32_t  labels[SNN_BENCH_IMAGE_COUNT];
    uint32_t correct;
    uint32_t spike_checksum;  /* 전 이미지·전 스텝 출력 스파이크 누적 (비트 단위 회귀 검출) */

    /* 측정 당시 하드웨어 상태 (라벨이 실제 조건과 맞는지 증명) */
    uint32_t flash_acr;       /* ART/프리페치/wait state */
    uint32_t scb_ccr;         /* I-cache/D-cache */
    uint32_t arena_used;
    uint32_t itcm_code_bytes; /* ITCM에 올린 코드 크기 */
} snn_bench_t;

/* 최적화 플래그 비트 */
#define OPT_ART    (1u << 0)
#define OPT_ITCM   (1u << 1)
#define OPT_DTCM   (1u << 2)
#define OPT_KERNEL (1u << 3)

volatile snn_bench_t g_snn_bench;

/* ------------------------------------------------------------------ */
/* 사이클 카운터                                                        */
/* ------------------------------------------------------------------ */
static inline void cyccnt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55;          /* Cortex-M7에서 필요할 수 있는 락 해제 */
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB(); __ISB();
}
static inline uint32_t cyc(void) { return DWT->CYCCNT; }

/* ------------------------------------------------------------------ */
/* 시스템 초기화 (main.cc와 같은 조건을 만든다)                          */
/* ------------------------------------------------------------------ */

void Error_Handler(void)
{
    __disable_irq();
    for (;;) { }
}

/* main.cc의 SystemClock_Config와 같은 값을 쓴다. 벤치가 애플리케이션과 다른
 * 클럭에서 돌면 비교가 성립하지 않으므로 PLL 계수까지 그대로 옮겼다.
 * HSE 25 MHz / M=25 / N=400 / P=2 → SYSCLK 200 MHz, APB1 /4, APB2 /2, 6 wait state. */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    HAL_PWR_EnableBkUpAccess();
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    osc.OscillatorType = RCC_OSCILLATORTYPE_LSI | RCC_OSCILLATORTYPE_HSE;
    osc.HSEState = RCC_HSE_ON;
    osc.LSIState = RCC_LSI_ON;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = 25;
    osc.PLL.PLLN = 400;
    osc.PLL.PLLP = RCC_PLLP_DIV2;
    osc.PLL.PLLQ = 9;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) Error_Handler();

    if (HAL_PWREx_EnableOverDrive() != HAL_OK) Error_Handler();

    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV4;
    clk.APB2CLKDivider = RCC_HCLK_DIV2;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_6) != HAL_OK) Error_Handler();
}

static void MPU_Config_Bench(void)
{
    /* main.cc의 MPU_Config와 동일한 설정. 조건을 맞춰야 비교가 성립한다. */
    MPU_Region_InitTypeDef r;
    HAL_MPU_Disable();

    r.Enable = MPU_REGION_ENABLE;
    r.BaseAddress = 0x00;
    r.Size = MPU_REGION_SIZE_4GB;
    r.AccessPermission = MPU_REGION_NO_ACCESS;
    r.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
    r.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
    r.IsShareable = MPU_ACCESS_SHAREABLE;
    r.Number = MPU_REGION_NUMBER0;
    r.TypeExtField = MPU_TEX_LEVEL0;
    r.SubRegionDisable = 0x87;
    r.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
    HAL_MPU_ConfigRegion(&r);

    r.BaseAddress = 0xC0000000;      /* SDRAM: write-through cacheable */
    r.Size = MPU_REGION_SIZE_32MB;
    r.AccessPermission = MPU_REGION_FULL_ACCESS;
    r.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
    r.IsCacheable = MPU_ACCESS_CACHEABLE;
    r.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
    r.Number = MPU_REGION_NUMBER1;
    r.SubRegionDisable = 0x00;
    r.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
    HAL_MPU_ConfigRegion(&r);

    r.BaseAddress = 0xA0000000;      /* FMC 제어 레지스터: device */
    r.Size = MPU_REGION_SIZE_8KB;
    r.IsBufferable = MPU_ACCESS_BUFFERABLE;
    r.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
    r.IsShareable = MPU_ACCESS_SHAREABLE;
    r.Number = MPU_REGION_NUMBER2;
    r.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
    HAL_MPU_ConfigRegion(&r);

    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/*
 * ART Accelerator + 프리페치.
 *
 * 이 보드는 옵션 바이트 BOOT_ADD0=0x0080 때문에 플래시의 ITCM 별칭(0x00200000)에서
 * 실행된다. Cortex-M7의 L1 I-cache는 AXIM 접근에만 적용되므로, 그 경로의 명령 페치를
 * 가속하는 것은 ART뿐이다. 그런데 stm32f7xx_hal_conf.h에 ART_ACCLERATOR_ENABLE이
 * 정의되어 있지 않아(#if에서 0으로 평가) HAL_Init()이 ART를 켜지 않는다.
 *
 * 실측(info/raw/measured_facts.md): 캐시에 안 들어가는 크기의 루프에서
 * ART OFF 1.26 cyc/insn → ON 0.89 cyc/insn (1.42배).
 */
static void art_configure(void)
{
#if defined(SNN_OPT_ART) && (SNN_OPT_ART != 0)
    __HAL_FLASH_ART_ENABLE();
    __HAL_FLASH_PREFETCH_BUFFER_ENABLE();
#else
    __HAL_FLASH_ART_DISABLE();
    __HAL_FLASH_PREFETCH_BUFFER_DISABLE();
#endif
    __DSB(); __ISB();
}

/* .itcm_code 섹션 적재는 공용 구현 snn_itcm_init() (snn_tcm.c) 사용 */

/* ------------------------------------------------------------------ */
/* 벤치마크 본체                                                        */
/* ------------------------------------------------------------------ */
static int16_t s_spikes[32 * 32 * 3];
static int16_t s_spk_out[10];
static int16_t s_mem_out[10];

/* 이미지 한 장을 T스텝 추론하고 예측 클래스를 돌려준다.
 * enc/inv에 인코딩·추론 사이클을 누적한다. */
static int run_one_image(const uint8_t *img,
                         uint32_t *enc, uint32_t *inv, uint32_t *chk)
{
    int32_t calc[10];
    memset(calc, 0, sizeof(calc));

    snn_reset_state();

    for (int t = 0; t < SNN_BENCH_T; ++t) {
        uint32_t t0 = cyc();
        spiking_rate(img, s_spikes, SNN_BENCH_T, 1, 3, 32, 32, 1.0f, 0.0f);
        uint32_t t1 = cyc();
        snn_forward_step(s_spikes, s_spk_out, s_mem_out);
        uint32_t t2 = cyc();

        *enc += t1 - t0;
        *inv += t2 - t1;

        for (int i = 0; i < 10; ++i) {
            /* main.cc와 동일한 판정식 */
            calc[i] += (s_spk_out[i] > 16000);
            *chk += (uint32_t)(uint16_t)s_spk_out[i];
        }
    }

    int best = 0;
    for (int i = 1; i < 10; ++i) if (calc[i] > calc[best]) best = i;
    return best;
}

int main(void)
{
    /* 1) main.cc와 같은 순서로 시스템을 세운다 */
    MPU_Config_Bench();
    SCB_EnableICache();
    SCB_EnableDCache();

    HAL_Init();
    SystemClock_Config();
    art_configure();

    uint32_t itcm_bytes = snn_itcm_init();

    cyccnt_init();

    /* 틱 인터럽트는 측정 노이즈다. HAL_Delay를 쓰지 않으므로 꺼도 된다.
     * 이 프로젝트의 HAL 타임베이스는 SysTick이 아니라 TIM6이다
     * (stm32f7xx_hal_timebase_tim.c) — TIM6 IRQ를 꺼야 실제로 조용해진다. */
    HAL_NVIC_DisableIRQ(TIM6_DAC_IRQn);
    SysTick->CTRL &= ~SysTick_CTRL_TICKINT_Msk;

    snn_accel_init();

    /* 2) 워밍업 — 첫 이미지는 캐시/분기예측이 차갑고 lazy 초기화가 섞인다.
     *    측정에서 제외하되 결과 검증에는 포함한다. */
    {
        uint32_t e = 0, i = 0, c = 0;
        (void)run_one_image(g_bench_images[0], &e, &i, &c);
    }

    /* 3) 측정 */
    volatile snn_bench_t *B = &g_snn_bench;
    B->magic = SNN_BENCH_MAGIC;
    B->done = 0;
    B->timesteps = SNN_BENCH_T;
    B->sysclk_hz = HAL_RCC_GetHCLKFreq();
    B->opt_flags = 0
#if defined(SNN_OPT_ART) && (SNN_OPT_ART != 0)
        | OPT_ART
#endif
#if defined(SNN_OPT_ITCM) && (SNN_OPT_ITCM != 0)
        | OPT_ITCM
#endif
#if defined(SNN_OPT_DTCM) && (SNN_OPT_DTCM != 0)
        | OPT_DTCM
#endif
#if defined(SNN_OPT_KERNEL) && (SNN_OPT_KERNEL != 0)
        | OPT_KERNEL
#endif
        ;

    /* 64비트 누산: v0 기준 8장 합이 uint32의 87%에 달해, 더 느린 설정
     * (Debug 빌드, T 증가)에서 조용히 랩되는 것을 막는다. 평균은 uint32에 든다. */
    uint64_t tot_all = 0, enc_all = 0, inv_all = 0;
    uint32_t chk = 0, correct = 0;
#if defined(SNN_BENCH)
    uint64_t l1_all = 0, l2_all = 0, lo_all = 0, mc_all = 0;
#endif

    for (int k = 0; k < SNN_BENCH_IMAGE_COUNT; ++k) {
        uint32_t enc = 0, inv = 0;
#if defined(SNN_BENCH)
        g_snn_cyc_cpl1 = 0; g_snn_cyc_cpl2 = 0;
        g_snn_cyc_lifout = 0; g_snn_cyc_memcpy = 0;
#endif
        uint32_t t0 = cyc();
        int pred = run_one_image(g_bench_images[k], &enc, &inv, &chk);
        uint32_t t1 = cyc();

        tot_all += t1 - t0;
        enc_all += enc;
        inv_all += inv;
#if defined(SNN_BENCH)
        l1_all += g_snn_cyc_cpl1;
        l2_all += g_snn_cyc_cpl2;
        lo_all += g_snn_cyc_lifout;
        mc_all += g_snn_cyc_memcpy;
#endif
        B->preds[k] = pred;
        B->labels[k] = g_bench_labels[k];
        if (pred == g_bench_labels[k]) correct++;
    }

    const uint32_t N = SNN_BENCH_IMAGE_COUNT;
    B->images = N;
    B->cyc_per_image = tot_all / N;
    B->cyc_encode = enc_all / N;
    B->cyc_invoke = inv_all / N;
#if defined(SNN_BENCH)
    B->cyc_cpl1 = l1_all / N;
    B->cyc_cpl2 = l2_all / N;
    B->cyc_lifout = lo_all / N;
    B->cyc_memcpy = mc_all / N;
#endif
    B->correct = correct;
    B->spike_checksum = chk;
    B->flash_acr = FLASH->ACR;
    B->scb_ccr = SCB->CCR;
    B->arena_used = snn_tflm_arena_used ? (uint32_t)snn_tflm_arena_used() : 0;
    B->itcm_code_bytes = itcm_bytes;

    __DSB();
    B->done = 1;
    __DSB();
    /* 결과 구조체는 캐시 대상 SRAM에 있다. ST-LINK가 보는 물리 메모리에
     * 반영되도록 클린한다. 이걸 빼면 호스트가 옛 값을 읽는다. */
    SCB_CleanDCache();

    /* WFI 금지: DBGMCU_CR=0(실측)이라 DBG_SLEEP=0이고, 슬립 중에는 SWD가
     * 붙지 않아 호스트가 결과를 회수할 수 없다. busy 스핀으로 대기한다. */
    for (;;) { __NOP(); }
}

} /* extern "C" */
