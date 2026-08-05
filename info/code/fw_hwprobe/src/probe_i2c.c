/*
 * I2C 버스 스캔: 이 보드에 실제로 어떤 칩이 붙어 있는지 실측한다.
 *
 * 회로도나 유저매뉴얼을 읽으면 "무엇이 붙어 있어야 하는지"는 알 수 있지만,
 * 내 손에 있는 이 개체에 그것이 실제로 응답하는지는 버스를 두드려 봐야 안다.
 * 주소 하나하나에 대해 START + 주소 바이트를 보내고 ACK가 돌아오는지 확인한다.
 *
 * STM32F746G-DISCO의 I2C 배선 (BSP 헤더 기준):
 *   I2C3 : PH7=SCL, PH8=SDA (AF4) — 오디오 코덱 WM8994, 터치 컨트롤러 FT5336
 *   I2C1 : PB8=SCL, PB9=SDA (AF4) — Arduino 확장 커넥터
 */
#include "probe.h"
#include "stm32f7xx.h"

/* BSP가 쓰는 타이밍 값 (stm32746g_discovery.h의 DISCOVERY_I2Cx_TIMING) */
#define I2C_TIMING_100K 0x40912732u

/* BSP 헤더에 적힌, 이 보드에 있어야 할 칩들의 7비트 주소(<<1 된 8비트 표기) */
typedef struct { uint16_t addr8; const char *name; } known_dev_t;
static const known_dev_t KNOWN[] = {
    { 0x34, "WM8994 오디오 코덱 (AUDIO_I2C_ADDRESS)" },
    { 0x70, "FT5336 정전식 터치 컨트롤러 (TS_I2C_ADDRESS)" },
    { 0x60, "카메라 모듈 (CAMERA_I2C_ADDRESS, 보통 미장착)" },
    { 0xA0, "EEPROM A01 (EEPROM_I2C_ADDRESS_A01)" },
    { 0xA6, "EEPROM A02 (EEPROM_I2C_ADDRESS_A02)" },
};

static I2C_HandleTypeDef s_hi2c;

static int i2c_setup(I2C_TypeDef *inst)
{
    GPIO_InitTypeDef g = {0};

    if (inst == I2C3) {
        __HAL_RCC_GPIOH_CLK_ENABLE();
        __HAL_RCC_I2C3_CLK_ENABLE();
        __HAL_RCC_I2C3_FORCE_RESET();
        __HAL_RCC_I2C3_RELEASE_RESET();
        g.Pin       = GPIO_PIN_7 | GPIO_PIN_8;
        g.Mode      = GPIO_MODE_AF_OD;
        g.Pull      = GPIO_PULLUP;
        g.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
        g.Alternate = GPIO_AF4_I2C3;
        HAL_GPIO_Init(GPIOH, &g);
    } else if (inst == I2C1) {
        __HAL_RCC_GPIOB_CLK_ENABLE();
        __HAL_RCC_I2C1_CLK_ENABLE();
        __HAL_RCC_I2C1_FORCE_RESET();
        __HAL_RCC_I2C1_RELEASE_RESET();
        g.Pin       = GPIO_PIN_8 | GPIO_PIN_9;
        g.Mode      = GPIO_MODE_AF_OD;
        g.Pull      = GPIO_PULLUP;
        g.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
        g.Alternate = GPIO_AF4_I2C1;
        HAL_GPIO_Init(GPIOB, &g);
    } else {
        return -1;
    }

    s_hi2c.Instance              = inst;
    s_hi2c.Init.Timing           = I2C_TIMING_100K;
    s_hi2c.Init.OwnAddress1      = 0;
    s_hi2c.Init.AddressingMode   = I2C_ADDRESSINGMODE_7BIT;
    s_hi2c.Init.DualAddressMode  = I2C_DUALADDRESS_DISABLE;
    s_hi2c.Init.OwnAddress2      = 0;
    s_hi2c.Init.GeneralCallMode  = I2C_GENERALCALL_DISABLE;
    s_hi2c.Init.NoStretchMode    = I2C_NOSTRETCH_DISABLE;
    return (HAL_I2C_Init(&s_hi2c) == HAL_OK) ? 0 : -2;
}

static const char *lookup(uint16_t addr8)
{
    for (unsigned i = 0; i < sizeof(KNOWN) / sizeof(KNOWN[0]); i++)
        if (KNOWN[i].addr8 == addr8) return KNOWN[i].name;
    return NULL;
}

static void scan_bus(I2C_TypeDef *inst, const char *label, const char *pins)
{
    probe_printf("\n--- %s (%s) ---\n", label, pins);
    if (i2c_setup(inst) != 0) {
        probe_printf("  초기화 실패. 이 버스는 건너뛴다.\n");
        return;
    }

    int found = 0;
    /* 7비트 주소 0x08~0x77 만 훑는다. 0x00~0x07과 0x78~0x7F는 예약 주소다.
     * HAL은 8비트 표기(주소<<1)를 받으므로 0x10~0xEE 범위가 된다. */
    for (uint16_t a7 = 0x08; a7 <= 0x77; a7++) {
        uint16_t a8 = (uint16_t)(a7 << 1);
        if (HAL_I2C_IsDeviceReady(&s_hi2c, a8, 2, 10) == HAL_OK) {
            const char *name = lookup(a8);
            probe_printf("  응답: 7비트 0x%02X (8비트 0x%02X)", a7, a8);
            if (name) probe_printf("  → %s", name);
            probe_puts("\n");
            found++;
        }
    }
    if (!found)
        probe_printf("  응답한 장치 없음.\n");
    else
        probe_printf("  총 %d개 장치가 응답했다.\n", found);

    /* 알려진 주소 중 응답하지 않은 것도 명시해 둔다 — "없다"는 것도 정보다 */
    probe_printf("  BSP 헤더에 정의된 주소별 응답 여부:\n");
    for (unsigned i = 0; i < sizeof(KNOWN) / sizeof(KNOWN[0]); i++) {
        HAL_StatusTypeDef st = HAL_I2C_IsDeviceReady(&s_hi2c, KNOWN[i].addr8, 2, 10);
        probe_printf("    0x%02X %-6s %s\n", KNOWN[i].addr8,
                     (st == HAL_OK) ? "응답" : "무응답", KNOWN[i].name);
    }

    HAL_I2C_DeInit(&s_hi2c);
}

/* 터치 컨트롤러가 응답하면 칩 ID 레지스터를 읽어 실제 부품을 확정한다.
 * 기대값의 근거는 이 저장소에 있는 ST 드라이버다:
 *   Drivers/BSP/Components/ft5336/ft5336.h
 *     FT5336_CHIP_ID_REG = 0xA8, FT5336_ID_VALUE = 0x51
 *   Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_ts.c 의 BSP_TS_Init()도
 *   ReadID() 결과를 0x51과 비교해 드라이버를 붙일지 결정한다.
 * 따라서 0xA8에서 0x51이 나오면 FT5336으로 확정할 수 있다. */
static void identify_touch(void)
{
    if (i2c_setup(I2C3) != 0) return;
    if (HAL_I2C_IsDeviceReady(&s_hi2c, 0x70, 2, 10) != HAL_OK) {
        HAL_I2C_DeInit(&s_hi2c);
        return;
    }
    probe_printf("\n터치 컨트롤러 식별 (I2C3 0x70):\n");
    struct { uint8_t reg; const char *what; } regs[] = {
        { 0xA8, "칩 ID (FT5336_CHIP_ID_REG) — FT5336이면 0x51" },
        { 0xA6, "펌웨어 버전 (FT5336_FIRMID_REG)" },
        { 0xA3, "패널 ID" },
        { 0xAF, "릴리스 코드" },
        { 0xA1, "라이브러리 버전 하위 바이트" },
    };
    uint8_t chip_id = 0;
    for (unsigned i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
        uint8_t v = 0;
        if (HAL_I2C_Mem_Read(&s_hi2c, 0x70, regs[i].reg, I2C_MEMADD_SIZE_8BIT,
                             &v, 1, 20) == HAL_OK) {
            probe_printf("  reg 0x%02X = 0x%02X   ; %s\n", regs[i].reg, v, regs[i].what);
            if (regs[i].reg == 0xA8) chip_id = v;
        } else {
            probe_printf("  reg 0x%02X 읽기 실패   ; %s\n", regs[i].reg, regs[i].what);
        }
    }
    if (chip_id == 0x51)
        probe_printf("  → 칩 ID가 0x51이므로 이 보드의 터치 컨트롤러는 FT5336으로 확정된다.\n");
    else
        probe_printf("  → 칩 ID 0x%02X는 FT5336 기대값(0x51)과 다르다. 다른 부품일 수 있다.\n", chip_id);
    HAL_I2C_DeInit(&s_hi2c);
}

void probe_run_i2c(void)
{
    probe_section("I2C 버스 스캔 (보드에 실장된 칩 실측)");
    probe_puts("각 주소에 START+주소를 보내고 ACK가 돌아오는지로 판정한다.\n");
    probe_puts("주소는 7비트/8비트 표기를 함께 적는다 (8비트 = 7비트 << 1).\n");

    scan_bus(I2C3, "I2C3", "PH7=SCL, PH8=SDA, AF4 — 오디오 코덱·터치 컨트롤러 계열");
    scan_bus(I2C1, "I2C1", "PB8=SCL, PB9=SDA, AF4 — Arduino 확장 커넥터");

    identify_touch();
}
