/*
 * STM32F746G-DISCO 하드웨어 프로브 펌웨어 - 진입점.
 *
 * 이 펌웨어의 목적은 애플리케이션을 돌리는 것이 아니라, 보드에 실제로 올라간
 * 실리콘이 어떤 물건인지를 "칩 자신에게 물어보고" 그 답을 남기는 것이다.
 * 데이터시트에 뭐라고 쓰여 있든, 여기 찍히는 값이 이 보드의 사실이다.
 *
 * 결과 회수 경로
 *   - USART1(115200 8N1) → ST-LINK VCP → /dev/cu.usbmodem*
 *   - 0x20035000 결과 버퍼 → ST-LINK 메모리 판독 (시리얼을 놓쳐도 남는다)
 *
 * 기준 클럭은 200MHz로 잡는다. 이 저장소의 SNN 실험 펌웨어가 PLLN=400 →
 * SYSCLK 200MHz로 동작하므로, 기존 측정치와 직접 비교하려면 조건을 맞춰야 한다.
 * (216MHz 조건도 probe_cpu.c에서 별도로 재측정한다.)
 */
#include "probe.h"
#include "stm32f7xx.h"

#define PROBE_BASE_MHZ 200u

/* 진행 상황 표시용 LED: STM32F746G-DISCO의 LD1 = PI1 */
static void led_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOIEN;
    (void)RCC->AHB1ENR;
    GPIOI->MODER = (GPIOI->MODER & ~(3u << (1 * 2))) | (1u << (1 * 2));  /* 출력 */
    GPIOI->OSPEEDR |= (3u << (1 * 2));
}
static void led_set(int on)
{
    if (on) GPIOI->BSRR = (1u << 1);
    else    GPIOI->BSRR = (1u << (1 + 16));
}

/* .itcm_code 섹션(플래시에 적재)을 실제 ITCM으로 복사한다.
 * probe_cpu.c가 "ITCM에서 실행할 때의 속도"를 재려면 이 복사가 선행되어야 한다. */
extern uint32_t _sitcm_code, _eitcm_code, _sitcm_code_load;
static void itcm_code_load(void)
{
    uint32_t *dst = &_sitcm_code;
    const uint32_t *src = &_sitcm_code_load;
    while (dst < &_eitcm_code) *dst++ = *src++;
    __DSB();
    __ISB();
}

static void banner(int sdram_ok, int qspi_ok)
{
    probe_section("STM32F746G-DISCO 하드웨어 프로브");
    probe_printf("펌웨어 빌드: " __DATE__ " " __TIME__ "\n");
    probe_printf("SYSCLK=%u Hz  HCLK=%u Hz  PCLK1=%u Hz  PCLK2=%u Hz\n",
                 g_sysclk_hz, g_hclk_hz, g_pclk1_hz, g_pclk2_hz);
    probe_printf("외부 메모리 초기화: SDRAM=%s  QSPI=%s\n",
                 sdram_ok == 0 ? "OK" : "FAIL",
                 qspi_ok == 0 ? "OK" : "FAIL");
    probe_printf("결과 버퍼: 0x%08X (%u bytes)\n", PROBE_RESULT_ADDR, PROBE_RESULT_SIZE);
    if (qspi_ok == 0) {
        extern void probe_report_qspi_id(void);
        probe_report_qspi_id();
    }
    probe_puts("\n주: 이 파일의 모든 수치는 이 보드에서 직접 측정한 값이다 (신뢰도 A).\n");
}

/* probe_time.c와 같은 진행 표식. 초기화 도중 멈추면 ST-LINK로 이 주소를 읽어
 * 어느 단계까지 갔는지 알아낸다. 0x10~0x7F는 probe_set_sysclk이 쓴다. */
#define BOOT_STEP (*(volatile uint32_t *)PROBE_DIAG_BOOT_STEP)
#define STEP(n)   do { BOOT_STEP = (n); __DSB(); } while (0)

int main(void)
{
    STEP(0x01);
    /* 1) 클럭을 기준 조건으로 고정 */
    int clk_rc = probe_set_sysclk(PROBE_BASE_MHZ);

    /* 2) HAL 기동 (BSP의 SDRAM/QSPI 초기화가 HAL_Delay에 의존한다) */
    STEP(0x80);
    HAL_Init();

    STEP(0x81);
    led_init();
    led_set(1);

    /* 3) 출력 경로와 사이클 카운터 */
    STEP(0x82);
    probe_io_init(g_sysclk_hz, g_pclk2_hz);
    STEP(0x83);
    probe_time_init();

    STEP(0x84);
    itcm_code_load();

    probe_printf("\n[boot] probe_set_sysclk(%u) 반환 %d\n", PROBE_BASE_MHZ, clk_rc);

    /* 4) 외부 메모리를 켠다. 실패해도 나머지 측정은 진행한다. */
    STEP(0x85);
    int sdram_ok = probe_init_sdram();
    STEP(0x86);
    int qspi_ok  = probe_init_qspi();
    STEP(0x87);

    banner(sdram_ok, qspi_ok);

    /* 5) 측정. 순서에 의미가 있다: 레지스터를 먼저 떠 두어야 이후 벤치마크가
     *    캐시/ART 상태를 바꾸기 전의 "부팅 직후 상태"가 기록으로 남는다. */
    probe_run_registers();
    led_set(0);

    /* 칩 내부 아날로그(전압/온도/저속 발진기)와 I2C 버스 스캔.
     * probe.h에 선언을 늘리지 않기 위해 여기서만 extern으로 끌어온다. */
    extern void probe_run_analog(void);
    extern void probe_run_i2c(void);
    extern void probe_run_board(void);
    probe_run_analog();
    probe_run_i2c();
    probe_run_board();

    probe_run_memory();
    led_set(1);

    probe_run_cpu();

    probe_section("측정 종료");
    probe_printf("총 로그 %u bytes, 버퍼 초과로 버려진 %u bytes\n",
                 g_probe_result->length, g_probe_result->overflow);
    probe_finish();

    /* 완료 신호: LED를 빠르게 깜빡인다 */
    for (;;) {
        led_set(1);
        for (volatile int i = 0; i < 2000000; i++) { }
        led_set(0);
        for (volatile int i = 0; i < 2000000; i++) { }
    }
}
