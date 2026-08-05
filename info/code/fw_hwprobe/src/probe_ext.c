/*
 * 외부 메모리(SDRAM, QSPI NOR) 초기화.
 *
 * 보드 BSP(stm32746g_discovery_sdram.c / _qspi.c)를 그대로 쓴다. BSP가 설정한
 * FMC/QUADSPI 타이밍 레지스터 값 자체가 "이 보드의 외부 메모리가 어떤 조건으로
 * 동작하는가"에 대한 1차 증거이므로, probe_regs.c에서 그 값을 그대로 덤프한다.
 */
#include "probe.h"
#include "stm32f7xx.h"
#include "stm32746g_discovery_sdram.h"
#include "stm32746g_discovery_qspi.h"

int probe_init_sdram(void)
{
    uint8_t r = BSP_SDRAM_Init();
    return (r == SDRAM_OK) ? 0 : -1;
}

/*
 * QSPI NOR 플래시의 JEDEC ID를 읽어 부품을 확정한다.
 *
 * 반드시 메모리맵 모드로 넘어가기 **전에** 해야 한다. 메모리맵 모드에서는
 * 컨트롤러가 CPU의 읽기 요청을 자동으로 플래시 명령으로 바꿔 보내므로,
 * 이쪽에서 임의 명령(0x9F)을 끼워 넣을 수 없다.
 *
 * BSP가 내부 QSPI_HandleTypeDef를 static으로 감춰 두었기 때문에 HAL을 통하지 않고
 * QUADSPI 레지스터를 직접 다룬다. indirect read 모드로 명령만 보내고 데이터를 받는다.
 */
static uint32_t s_qspi_jedec_id;

static int qspi_read_jedec_id(uint8_t out[3])
{
    /* 진행 중인 동작이 있으면 정리 */
    QUADSPI->FCR = QUADSPI_FCR_CTOF | QUADSPI_FCR_CSMF | QUADSPI_FCR_CTCF | QUADSPI_FCR_CTEF;

    uint32_t guard = 0;
    while (QUADSPI->SR & QUADSPI_SR_BUSY) {
        if (++guard > 5000000u) return -1;
    }

    QUADSPI->DLR = 2;            /* 3바이트 수신 (제조사, 메모리 타입, 용량) */
    /* FMODE=01(indirect read), DMODE=01(1-line 데이터), ADMODE=00(주소 없음),
     * IMODE=01(1-line 명령), INSTRUCTION=0x9F (Read JEDEC ID) */
    QUADSPI->CCR = (1u << QUADSPI_CCR_FMODE_Pos)
                 | (1u << QUADSPI_CCR_DMODE_Pos)
                 | (1u << QUADSPI_CCR_IMODE_Pos)
                 | 0x9Fu;

    int n = 0;
    guard = 0;
    while (n < 3) {
        if (QUADSPI->SR & (QUADSPI_SR_FTF | QUADSPI_SR_TCF)) {
            out[n++] = *(volatile uint8_t *)&QUADSPI->DR;
        }
        if (++guard > 5000000u) return -2;
    }

    guard = 0;
    while (!(QUADSPI->SR & QUADSPI_SR_TCF)) {
        if (++guard > 5000000u) break;
    }
    QUADSPI->FCR = QUADSPI_FCR_CTCF;
    return 0;
}

int probe_init_qspi(void)
{
    uint8_t r = BSP_QSPI_Init();
    if (r != QSPI_OK) return -1;

    uint8_t id[3] = {0, 0, 0};
    if (qspi_read_jedec_id(id) == 0)
        s_qspi_jedec_id = ((uint32_t)id[0] << 16) | ((uint32_t)id[1] << 8) | id[2];
    else
        s_qspi_jedec_id = 0xFFFFFFFFu;   /* 읽기 실패 표시 */

    /* 메모리맵 모드로 넘겨야 0x90000000에서 CPU가 직접 읽을 수 있다 */
    r = BSP_QSPI_EnableMemoryMappedMode();
    return (r == QSPI_OK) ? 0 : -2;
}

/* main/probe 모듈이 결과를 출력할 때 쓴다 */
void probe_report_qspi_id(void)
{
    uint32_t v = s_qspi_jedec_id;
    if (v == 0xFFFFFFFFu || v == 0) {
        probe_printf("QSPI JEDEC ID: 읽기 실패 (0x%06X)\n", v);
        return;
    }
    uint8_t mfr = (uint8_t)(v >> 16), type = (uint8_t)(v >> 8), cap = (uint8_t)v;
    probe_printf("QSPI JEDEC ID: 0x%06X  (제조사 0x%02X, 타입 0x%02X, 용량코드 0x%02X)\n",
                 v, mfr, type, cap);
    /* JEP106 제조사 코드 */
    const char *mname =
        (mfr == 0x20) ? "Micron / Numonyx (구 ST)" :
        (mfr == 0xC2) ? "Macronix" :
        (mfr == 0xEF) ? "Winbond" :
        (mfr == 0x01) ? "Spansion / Cypress" :
        (mfr == 0x9D) ? "ISSI" : "미상";
    probe_printf("  제조사: %s\n", mname);
    /* 용량 코드는 보통 2^cap 바이트를 뜻한다 */
    if (cap >= 0x10 && cap <= 0x20) {
        uint32_t bytes_mb = 1u << (cap - 20);
        probe_printf("  용량 코드 0x%02X → 2^%u 바이트 = %u MB\n", cap, cap, bytes_mb);
    }
    if (mfr == 0x20 && type == 0xBA)
        probe_printf("  → Micron N25Q / MT25QL 계열(3V)로 식별된다.\n");
    else if (mfr == 0x20 && type == 0xBB)
        probe_printf("  → Micron N25Q / MT25QU 계열(1.8V)로 식별된다.\n");
}
