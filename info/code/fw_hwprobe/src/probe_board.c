/*
 * 보드 레벨 실측: 주변 칩을 ID 레지스터로 확정하고, 물리 입력 핀 상태를 읽는다.
 *
 * I2C 스캔(probe_i2c.c)이 "그 주소에 무언가 있다"까지 알려준다면, 여기서는 한 걸음
 * 더 들어가 그 무언가가 정확히 어떤 부품인지를 칩 자신의 ID 레지스터로 확정한다.
 * 이더넷 PHY는 I2C가 아니라 MDIO(관리 인터페이스)로 물어봐야 한다.
 */
#include "probe.h"
#include "stm32f7xx.h"

/* ---- WM8994 오디오 코덱 ---------------------------------------------------
 * I2C 주소 0x34. 레지스터 주소도 데이터도 16비트이고 빅엔디안이다.
 * 레지스터 0x0000이 칩 ID이며, 드라이버(Components/wm8994/wm8994.h)의
 * WM8994_ID가 0x8994로 정의되어 있다. */
static void identify_audio_codec(I2C_HandleTypeDef *h)
{
    probe_printf("\n오디오 코덱 식별 (I2C3 0x34):\n");
    uint8_t buf[2] = {0, 0};
    if (HAL_I2C_Mem_Read(h, 0x34, 0x0000, I2C_MEMADD_SIZE_16BIT, buf, 2, 50) != HAL_OK) {
        probe_printf("  칩 ID 레지스터 읽기 실패\n");
        return;
    }
    uint16_t id = (uint16_t)((buf[0] << 8) | buf[1]);
    probe_printf("  reg 0x0000 (칩 ID) = 0x%04X\n", id);
    if (id == 0x8994)
        probe_printf("  → 0x8994이므로 오디오 코덱은 WM8994로 확정된다.\n");
    else
        probe_printf("  → 드라이버 기대값 0x8994와 다르다. 다른 부품일 수 있다.\n");
}

/* ---- 이더넷 PHY (MDIO) -----------------------------------------------------
 * MDIO는 MAC이 HCLK에서 만들어 내는 MDC로 동작하므로, RMII 데이터 경로가 살아
 * 있지 않아도 PHY의 레지스터는 읽히는 경우가 많다. 다만 확신할 수 없으므로
 * 모든 대기에 타임아웃을 건다. PHY 주소도 모르므로 0~31을 전부 훑는다.
 *
 * IEEE 802.3 표준 레지스터:
 *   2 = PHY Identifier 1 (OUI 상위 16비트)
 *   3 = PHY Identifier 2 (OUI 하위 6비트 + 모델 번호 6비트 + 리비전 4비트)
 */
static int mdio_read(uint32_t phy_addr, uint32_t reg, uint16_t *out)
{
    /* MACMIIAR.CR: HCLK 150~216MHz 구간은 0b100 (MDC = HCLK/102) */
    uint32_t v = (phy_addr << 11) | (reg << 6) | (4u << 2) | ETH_MACMIIAR_MB;
    ETH->MACMIIAR = v;
    uint32_t guard = 0;
    while (ETH->MACMIIAR & ETH_MACMIIAR_MB) {
        if (++guard > 2000000u) return -1;
    }
    *out = (uint16_t)(ETH->MACMIIDR & 0xFFFFu);
    return 0;
}

static void identify_eth_phy(void)
{
    probe_printf("\n이더넷 PHY 식별 (MDIO: PA2=MDIO, PC1=MDC):\n");

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_ETH_CLK_ENABLE();

    GPIO_InitTypeDef g = {0};
    g.Mode = GPIO_MODE_AF_PP;
    g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF11_ETH;
    g.Pin = GPIO_PIN_2;                 /* PA2 = ETH_MDIO */
    HAL_GPIO_Init(GPIOA, &g);
    g.Pin = GPIO_PIN_1;                 /* PC1 = ETH_MDC */
    HAL_GPIO_Init(GPIOC, &g);

    int found = 0;
    for (uint32_t addr = 0; addr < 32; addr++) {
        uint16_t id1 = 0, id2 = 0;
        if (mdio_read(addr, 2, &id1) != 0) {
            probe_printf("  PHY 주소 %u: MDIO 타임아웃 — 이후 주소는 건너뛴다.\n", addr);
            probe_printf("  (MDIO가 응답하지 않는다. RMII REF_CLK가 없거나 PHY가 리셋 상태일 수 있다.)\n");
            return;
        }
        if (mdio_read(addr, 3, &id2) != 0) return;

        /* 0x0000 / 0xFFFF는 "그 주소에 아무도 없다"는 뜻이다 */
        if ((id1 == 0x0000 && id2 == 0x0000) || (id1 == 0xFFFF && id2 == 0xFFFF))
            continue;

        found++;
        uint32_t oui   = ((uint32_t)id1 << 6) | ((id2 >> 10) & 0x3F);
        uint32_t model = (id2 >> 4) & 0x3F;
        uint32_t rev   = id2 & 0x0F;
        probe_printf("  PHY 주소 %u: ID1=0x%04X ID2=0x%04X\n", addr, id1, id2);
        probe_printf("      OUI=0x%06X  모델=0x%02X  리비전=0x%X\n", oui, model, rev);
        /* Microchip/SMSC LAN8742A: ID1=0x0007, ID2 상위가 0xC13x */
        if (id1 == 0x0007 && (id2 & 0xFFF0) == 0xC130)
            probe_printf("      → SMSC/Microchip LAN8742A 계열로 식별된다.\n");
        else if (id1 == 0x0007)
            probe_printf("      → SMSC/Microchip 계열이다 (정확한 모델은 모델번호로 대조 필요).\n");

        uint16_t bmcr = 0, bmsr = 0;
        if (mdio_read(addr, 0, &bmcr) == 0 && mdio_read(addr, 1, &bmsr) == 0) {
            probe_printf("      BMCR=0x%04X BMSR=0x%04X  링크=%s, 자동협상=%s\n",
                         bmcr, bmsr,
                         (bmsr & 0x0004) ? "up" : "down",
                         (bmsr & 0x0020) ? "완료" : "미완료");
        }
    }
    if (!found)
        probe_printf("  응답한 PHY 없음 (주소 0~31 전부 조회).\n");
}

/* ---- 물리 입력 핀 상태 -----------------------------------------------------
 * 보드에 무엇이 꽂혀 있고 어떤 버튼이 눌려 있는지는 핀을 읽어야만 안다. */
static void read_board_pins(void)
{
    probe_printf("\n보드 입력 핀 상태:\n");

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOI_CLK_ENABLE();

    GPIO_InitTypeDef g = {0};
    g.Mode  = GPIO_MODE_INPUT;
    g.Speed = GPIO_SPEED_FREQ_LOW;

    /* SD 카드 검출: PC13. 카드가 들어가면 스위치가 GND로 당긴다(풀업 필요). */
    g.Pin = GPIO_PIN_13;
    g.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOC, &g);

    /* 사용자 버튼 B1(파랑): PI11. 눌리면 High. */
    g.Pin = GPIO_PIN_11;
    g.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(GPIOI, &g);

    for (volatile int i = 0; i < 100000; i++) { }   /* 풀 저항 안정화 */

    int sd_detect = HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13);
    int btn       = HAL_GPIO_ReadPin(GPIOI, GPIO_PIN_11);

    probe_printf("  PC13 (SD_DETECT) = %d  → microSD 카드 %s\n",
                 sd_detect, sd_detect ? "없음" : "삽입됨");
    probe_printf("  PI11 (사용자 버튼 B1) = %d  → %s\n",
                 btn, btn ? "눌린 상태" : "눌리지 않음");
    probe_printf("  PI1  (LED1/LD1) 은 이 펌웨어가 진행 표시용으로 출력 구동 중\n");
    probe_printf("  PI12 (LCD_DISP), PK3 (LCD 백라이트) 는 LCD를 초기화하지 않아 건드리지 않았다\n");
}

/* ---- LTDC 현재 설정 --------------------------------------------------------
 * 이 펌웨어는 LCD를 초기화하지 않는다. 그러나 직전에 돌던 펌웨어가 남긴 설정이
 * 그대로 남아 있을 수 있으므로, 값이 유효해 보이면 해상도를 역산해 둔다. */
static void read_ltdc(void)
{
    probe_printf("\nLTDC (LCD-TFT 컨트롤러) 현재 레지스터:\n");
    __HAL_RCC_LTDC_CLK_ENABLE();
    uint32_t sscr = LTDC->SSCR, bpcr = LTDC->BPCR, awcr = LTDC->AWCR, twcr = LTDC->TWCR;
    probe_printf("  SSCR=0x%08X BPCR=0x%08X AWCR=0x%08X TWCR=0x%08X GCR=0x%08X\n",
                 sscr, bpcr, awcr, twcr, LTDC->GCR);
    if (awcr == 0 || bpcr == 0) {
        probe_printf("  → 타이밍 레지스터가 비어 있다. 이 펌웨어는 LCD를 초기화하지 않는다.\n");
        return;
    }
    uint32_t hsync = (sscr >> 16) + 1, vsync = (sscr & 0x7FF) + 1;
    uint32_t hbp   = (bpcr >> 16) + 1, vbp   = (bpcr & 0x7FF) + 1;
    uint32_t haw   = (awcr >> 16) + 1, vah   = (awcr & 0x7FF) + 1;
    uint32_t htot  = (twcr >> 16) + 1, vtot  = (twcr & 0x7FF) + 1;
    probe_printf("  → 활성 영역 %u x %u 픽셀\n", haw - hbp, vah - vbp);
    probe_printf("     HSYNC=%u HBP=%u 전체가로=%u / VSYNC=%u VBP=%u 전체세로=%u\n",
                 hsync, hbp, htot, vsync, vbp, vtot);
    probe_printf("     한 프레임에 필요한 픽셀 클럭 = %u\n", htot * vtot);
}

void probe_run_board(void)
{
    probe_section("보드 주변 부품 실측 (ID 레지스터·핀 상태)");

    /* 오디오 코덱은 I2C3에 있다. probe_i2c.c가 쓰던 핸들을 재사용하지 않고
     * 여기서 독립적으로 버스를 세운다 (모듈 간 상태 의존을 만들지 않는다). */
    static I2C_HandleTypeDef h;
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_I2C3_CLK_ENABLE();
    GPIO_InitTypeDef g = {0};
    g.Pin       = GPIO_PIN_7 | GPIO_PIN_8;
    g.Mode      = GPIO_MODE_AF_OD;
    g.Pull      = GPIO_PULLUP;
    g.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF4_I2C3;
    HAL_GPIO_Init(GPIOH, &g);

    h.Instance             = I2C3;
    h.Init.Timing          = 0x40912732u;
    h.Init.OwnAddress1     = 0;
    h.Init.AddressingMode  = I2C_ADDRESSINGMODE_7BIT;
    h.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    h.Init.OwnAddress2     = 0;
    h.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    h.Init.NoStretchMode   = I2C_NOSTRETCH_DISABLE;
    if (HAL_I2C_Init(&h) == HAL_OK) {
        identify_audio_codec(&h);
        HAL_I2C_DeInit(&h);
    } else {
        probe_printf("\n오디오 코덱: I2C3 초기화 실패\n");
    }

    identify_eth_phy();
    read_board_pins();
    read_ltdc();
}
