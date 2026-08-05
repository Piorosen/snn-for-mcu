/*
 * SystemInit / 인터럽트 핸들러 최소 구현.
 *
 * CubeMX가 만드는 system_stm32f7xx.c를 쓰지 않고 직접 두는 이유:
 * 그 파일은 VECT_TAB_OFFSET 매크로에 따라 VTOR을 ITCM 별칭(0x00200000)으로
 * 잡을 수 있는데, 이 프로브 펌웨어는 코드 페치 경로를 스스로 통제해야 하므로
 * VTOR을 AXIM 플래시(0x08000000)로 명시적으로 고정한다.
 */
#include "stm32f7xx.h"

uint32_t SystemCoreClock = 16000000u;   /* 리셋 직후 HSI */
const uint8_t AHBPrescTable[16] = {0,0,0,0,0,0,0,0,1,2,3,4,6,7,8,9};
const uint8_t APBPrescTable[8]  = {0,0,0,0,1,2,3,4};

void SystemInit(void)
{
    /* FPU 활성화: CP10/CP11 full access. -mfloat-abi=hard로 빌드하므로 필수. */
#if (__FPU_PRESENT == 1) && (__FPU_USED == 1)
    SCB->CPACR |= ((3UL << 10 * 2) | (3UL << 11 * 2));
#endif

    /* RCC를 리셋 기본값으로 되돌린다. ST-LINK 핫플러그로 이전 펌웨어가 남긴
     * 클럭 설정 위에서 부팅하는 상황을 배제하기 위함. */
    RCC->CR |= RCC_CR_HSION;
    RCC->CFGR = 0x00000000u;
    RCC->CR &= ~(RCC_CR_HSEON | RCC_CR_CSSON | RCC_CR_PLLON);
    RCC->PLLCFGR = 0x24003010u;      /* 리셋값 */
    RCC->CR &= ~RCC_CR_HSEBYP;
    RCC->CIR = 0x00000000u;

    /* 벡터 테이블은 AXIM 플래시 기준으로 고정 */
    SCB->VTOR = FLASH_BASE;
}

void SystemCoreClockUpdate(void)
{
    /* 실제 갱신은 probe_set_sysclk()이 담당한다. HAL이 호출해도 문제없도록 둔다. */
}

/* ---- 예외 핸들러 --------------------------------------------------------- */
/* 프로브 펌웨어에서 폴트가 나면 원인을 남기고 멈춰야 진단이 된다.
 * 폴트 상태 레지스터를 결과 버퍼 뒤쪽 고정 위치에 적어 두고 무한 루프. */
#define FAULT_LOG ((volatile uint32_t *)0x20034010u)  /* PROBE_DIAG_FAULT_LOG */

static void fault_record(uint32_t kind)
{
    FAULT_LOG[0] = 0xDEADBEEFu;
    FAULT_LOG[1] = kind;
    FAULT_LOG[2] = SCB->CFSR;
    FAULT_LOG[3] = SCB->HFSR;
    FAULT_LOG[4] = SCB->MMFAR;
    FAULT_LOG[5] = SCB->BFAR;
    FAULT_LOG[6] = SCB->SHCSR;
    FAULT_LOG[7] = *(volatile uint32_t *)0xE000EFA8u;   /* ABFSR */
    __DSB();
    for (;;) { }
}

void NMI_Handler(void)        { fault_record(1); }
void HardFault_Handler(void)  { fault_record(2); }
void MemManage_Handler(void)  { fault_record(3); }
void BusFault_Handler(void)   { fault_record(4); }
void UsageFault_Handler(void) { fault_record(5); }
void SVC_Handler(void)        { }
void DebugMon_Handler(void)   { }
void PendSV_Handler(void)     { }

void SysTick_Handler(void)
{
    HAL_IncTick();
}
