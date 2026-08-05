# 호스트 측에서 수집한 하드웨어 관련 정보

데이터 출처는 툴체인·벤더 헤더·OS 장치 정보다. 보드에 직접 물어본 값이 아니므로
실측(신뢰도 A)과는 구분해서 읽어야 한다.


## 툴체인이 인식하는 타깃 [신뢰도 B: GNU 툴체인 자체 출력]

- 컴파일러: `arm-none-eabi-gcc (GNU Tools for STM32 14.3.rel1.20251027-0700) 14.3.1 20250623`
- 타깃 triplet: `arm-none-eabi`
- 이 MCU 플래그에서 선택되는 멀티립: `thumb/v7e-m+fp/hard`
- 빌드 플래그: `-mcpu=cortex-m7 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard`


### 컴파일러가 켜는 아키텍처 기능 매크로

이 매크로들은 툴체인이 이 CPU에 어떤 하드웨어 기능이 있다고 보는지를 그대로 드러낸다.

| 매크로 | 값 | 의미 |
|---|---|---|
| `__ARM_32BIT_STATE` | `1` |  |
| `__ARM_ARCH` | `7` | ARM 아키텍처 세대 |
| `__ARM_ARCH_7EM__` | `1` | ARMv7E-M (DSP 확장 포함 M 프로파일) |
| `__ARM_ARCH_EXT_IDIV__` | `1` |  |
| `__ARM_ARCH_ISA_THUMB` | `2` |  |
| `__ARM_ARCH_PROFILE` | `77` |  |
| `__ARM_ASM_SYNTAX_UNIFIED__` | `1` |  |
| `__ARM_EABI__` | `1` |  |
| `__ARM_FEATURE_CLZ` | `1` | CLZ 명령 지원 |
| `__ARM_FEATURE_COPROC` | `15` |  |
| `__ARM_FEATURE_DSP` | `1` | DSP(SIMD32) 확장 명령 사용 가능 — SMLAD/QADD16 등 |
| `__ARM_FEATURE_FMA` | `1` | 융합 곱셈-덧셈(VFMA) 하드웨어 지원 |
| `__ARM_FEATURE_IDIV` | `1` |  |
| `__ARM_FEATURE_LDREX` | `7` | 배타적 로드/스토어 지원 폭 |
| `__ARM_FEATURE_QBIT` | `1` | 포화 연산 Q 플래그 지원 |
| `__ARM_FEATURE_SAT` | `1` | 포화 명령(SSAT/USAT) 지원 |
| `__ARM_FEATURE_SIMD32` | `1` | 32비트 SIMD 명령 사용 가능 |
| `__ARM_FEATURE_UNALIGNED` | `1` | 비정렬 접근 하드웨어 지원 |
| `__ARM_FP` | `4` | 하드웨어 부동소수점 정밀도 비트마스크 (4=단정밀도, 8=배정밀도) |
| `__ARM_PCS_VFP` | `1` | VFP 레지스터로 부동소수점 인자 전달 (hard float ABI) |
| `__ARM_SIZEOF_MINIMAL_ENUM` | `1` |  |
| `__ARM_SIZEOF_WCHAR_T` | `4` |  |
| `__THUMBEL__` | `1` |  |
| `__THUMB_INTERWORK__` | `1` |  |
| `__VFP_FP__` | `1` | VFP 부동소수점 형식 사용 |


## 벤더 헤더가 기술한 실리콘 구성 [신뢰도 B: ST 배포 CMSIS 헤더]

- 인터럽트 벡터: 총 105개 항목 (그중 페리페럴 IRQ 97개, 최대 번호 97)
- 페리페럴 인스턴스: 71개
- `_BASE` 정의: 109개


### 메모리 영역 정의

| 심볼 | 주소 |
|---|---|
| `FLASH_END` | `0x080FFFFF` |
| `FLASH_OTP_BASE` | `0x1FF0F000` |
| `FLASH_OTP_END` | `0x1FF0F41F` |
| `SRAM1_BASE` | `0x20010000` |
| `SRAM2_BASE` | `0x2004C000` |
| `BKPSRAM_BASE` | `0x40024000` |
| `PERIPH_BASE` | `0x40000000` |
| `QSPI_BASE` | `0x90000000` |
| `FMC_R_BASE` | `0xA0000000` |
| `RAMITCM_BASE` | `0x00000000` |
| `FLASHITCM_BASE` | `0x00200000` |


### 페리페럴 인스턴스 목록

```
ADC             ADC1            ADC123_COMMON   ADC2            ADC3            CAN1            CAN2            CEC           
CRC             DAC             DAC1            DBGMCU          DCMI            DMA1            DMA2            ETH           
EXTI            FLASH           GPIOA           GPIOB           GPIOC           GPIOD           GPIOE           GPIOF         
GPIOG           GPIOH           GPIOI           GPIOJ           GPIOK           IWDG            LPTIM1          LTDC          
PWR             QUADSPI         RCC             RNG             RTC             SAI1            SAI2            SDMMC1        
SPDIFRX         SPI1            SPI2            SPI3            SPI4            SPI5            SPI6            SYSCFG        
TIM1            TIM10           TIM11           TIM12           TIM13           TIM14           TIM2            TIM3          
TIM4            TIM5            TIM6            TIM7            TIM8            TIM9            UART4           UART5         
UART7           UART8           USART1          USART2          USART3          USART6          WWDG          
```


### 인터럽트 벡터표

| # | 이름 | 설명 |
|---|---|---|
| -14 | `NonMaskableInt_IRQn` | 2 Non Maskable Interrupt |
| -12 | `MemoryManagement_IRQn` | 4 Cortex-M7 Memory Management Interrupt |
| -11 | `BusFault_IRQn` | 5 Cortex-M7 Bus Fault Interrupt |
| -10 | `UsageFault_IRQn` | 6 Cortex-M7 Usage Fault Interrupt |
| -5 | `SVCall_IRQn` | 11 Cortex-M7 SV Call Interrupt |
| -4 | `DebugMonitor_IRQn` | 12 Cortex-M7 Debug Monitor Interrupt |
| -2 | `PendSV_IRQn` | 14 Cortex-M7 Pend SV Interrupt |
| -1 | `SysTick_IRQn` | 15 Cortex-M7 System Tick Interrupt |
| 0 | `WWDG_IRQn` | Window WatchDog Interrupt |
| 1 | `PVD_IRQn` | PVD through EXTI Line detection Interrupt |
| 2 | `TAMP_STAMP_IRQn` | Tamper and TimeStamp interrupts through the EXTI line |
| 3 | `RTC_WKUP_IRQn` | RTC Wakeup interrupt through the EXTI line |
| 4 | `FLASH_IRQn` | FLASH global Interrupt |
| 5 | `RCC_IRQn` | RCC global Interrupt |
| 6 | `EXTI0_IRQn` | EXTI Line0 Interrupt |
| 7 | `EXTI1_IRQn` | EXTI Line1 Interrupt |
| 8 | `EXTI2_IRQn` | EXTI Line2 Interrupt |
| 9 | `EXTI3_IRQn` | EXTI Line3 Interrupt |
| 10 | `EXTI4_IRQn` | EXTI Line4 Interrupt |
| 11 | `DMA1_Stream0_IRQn` | DMA1 Stream 0 global Interrupt |
| 12 | `DMA1_Stream1_IRQn` | DMA1 Stream 1 global Interrupt |
| 13 | `DMA1_Stream2_IRQn` | DMA1 Stream 2 global Interrupt |
| 14 | `DMA1_Stream3_IRQn` | DMA1 Stream 3 global Interrupt |
| 15 | `DMA1_Stream4_IRQn` | DMA1 Stream 4 global Interrupt |
| 16 | `DMA1_Stream5_IRQn` | DMA1 Stream 5 global Interrupt |
| 17 | `DMA1_Stream6_IRQn` | DMA1 Stream 6 global Interrupt |
| 18 | `ADC_IRQn` | ADC1, ADC2 and ADC3 global Interrupts |
| 19 | `CAN1_TX_IRQn` | CAN1 TX Interrupt |
| 20 | `CAN1_RX0_IRQn` | CAN1 RX0 Interrupt |
| 21 | `CAN1_RX1_IRQn` | CAN1 RX1 Interrupt |
| 22 | `CAN1_SCE_IRQn` | CAN1 SCE Interrupt |
| 23 | `EXTI9_5_IRQn` | External Line[9:5] Interrupts |
| 24 | `TIM1_BRK_TIM9_IRQn` | TIM1 Break interrupt and TIM9 global interrupt |
| 25 | `TIM1_UP_TIM10_IRQn` | TIM1 Update Interrupt and TIM10 global interrupt |
| 26 | `TIM1_TRG_COM_TIM11_IRQn` | TIM1 Trigger and Commutation Interrupt and TIM11 global interrupt |
| 27 | `TIM1_CC_IRQn` | TIM1 Capture Compare Interrupt |
| 28 | `TIM2_IRQn` | TIM2 global Interrupt |
| 29 | `TIM3_IRQn` | TIM3 global Interrupt |
| 30 | `TIM4_IRQn` | TIM4 global Interrupt |
| 31 | `I2C1_EV_IRQn` | I2C1 Event Interrupt |
| 32 | `I2C1_ER_IRQn` | I2C1 Error Interrupt |
| 33 | `I2C2_EV_IRQn` | I2C2 Event Interrupt |
| 34 | `I2C2_ER_IRQn` | I2C2 Error Interrupt |
| 35 | `SPI1_IRQn` | SPI1 global Interrupt |
| 36 | `SPI2_IRQn` | SPI2 global Interrupt |
| 37 | `USART1_IRQn` | USART1 global Interrupt |
| 38 | `USART2_IRQn` | USART2 global Interrupt |
| 39 | `USART3_IRQn` | USART3 global Interrupt |
| 40 | `EXTI15_10_IRQn` | External Line[15:10] Interrupts |
| 41 | `RTC_Alarm_IRQn` | RTC Alarm (A and B) through EXTI Line Interrupt |
| 42 | `OTG_FS_WKUP_IRQn` | USB OTG FS Wakeup through EXTI line interrupt |
| 43 | `TIM8_BRK_TIM12_IRQn` | TIM8 Break Interrupt and TIM12 global interrupt |
| 44 | `TIM8_UP_TIM13_IRQn` | TIM8 Update Interrupt and TIM13 global interrupt |
| 45 | `TIM8_TRG_COM_TIM14_IRQn` | TIM8 Trigger and Commutation Interrupt and TIM14 global interrupt |
| 46 | `TIM8_CC_IRQn` | TIM8 Capture Compare Interrupt |
| 47 | `DMA1_Stream7_IRQn` | DMA1 Stream7 Interrupt |
| 48 | `FMC_IRQn` | FMC global Interrupt |
| 49 | `SDMMC1_IRQn` | SDMMC1 global Interrupt |
| 50 | `TIM5_IRQn` | TIM5 global Interrupt |
| 51 | `SPI3_IRQn` | SPI3 global Interrupt |
| 52 | `UART4_IRQn` | UART4 global Interrupt |
| 53 | `UART5_IRQn` | UART5 global Interrupt |
| 54 | `TIM6_DAC_IRQn` | TIM6 global and DAC1&2 underrun error  interrupts |
| 55 | `TIM7_IRQn` | TIM7 global interrupt |
| 56 | `DMA2_Stream0_IRQn` | DMA2 Stream 0 global Interrupt |
| 57 | `DMA2_Stream1_IRQn` | DMA2 Stream 1 global Interrupt |
| 58 | `DMA2_Stream2_IRQn` | DMA2 Stream 2 global Interrupt |
| 59 | `DMA2_Stream3_IRQn` | DMA2 Stream 3 global Interrupt |
| 60 | `DMA2_Stream4_IRQn` | DMA2 Stream 4 global Interrupt |
| 61 | `ETH_IRQn` | Ethernet global Interrupt |
| 62 | `ETH_WKUP_IRQn` | Ethernet Wakeup through EXTI line Interrupt |
| 63 | `CAN2_TX_IRQn` | CAN2 TX Interrupt |
| 64 | `CAN2_RX0_IRQn` | CAN2 RX0 Interrupt |
| 65 | `CAN2_RX1_IRQn` | CAN2 RX1 Interrupt |
| 66 | `CAN2_SCE_IRQn` | CAN2 SCE Interrupt |
| 67 | `OTG_FS_IRQn` | USB OTG FS global Interrupt |
| 68 | `DMA2_Stream5_IRQn` | DMA2 Stream 5 global interrupt |
| 69 | `DMA2_Stream6_IRQn` | DMA2 Stream 6 global interrupt |
| 70 | `DMA2_Stream7_IRQn` | DMA2 Stream 7 global interrupt |
| 71 | `USART6_IRQn` | USART6 global interrupt |
| 72 | `I2C3_EV_IRQn` | I2C3 event interrupt |
| 73 | `I2C3_ER_IRQn` | I2C3 error interrupt |
| 74 | `OTG_HS_EP1_OUT_IRQn` | USB OTG HS End Point 1 Out global interrupt |
| 75 | `OTG_HS_EP1_IN_IRQn` | USB OTG HS End Point 1 In global interrupt |
| 76 | `OTG_HS_WKUP_IRQn` | USB OTG HS Wakeup through EXTI interrupt |
| 77 | `OTG_HS_IRQn` | USB OTG HS global interrupt |
| 78 | `DCMI_IRQn` | DCMI global interrupt |
| 80 | `RNG_IRQn` | RNG global interrupt |
| 81 | `FPU_IRQn` | FPU global interrupt |
| 82 | `UART7_IRQn` | UART7 global interrupt |
| 83 | `UART8_IRQn` | UART8 global interrupt |
| 84 | `SPI4_IRQn` | SPI4 global Interrupt |
| 85 | `SPI5_IRQn` | SPI5 global Interrupt |
| 86 | `SPI6_IRQn` | SPI6 global Interrupt |
| 87 | `SAI1_IRQn` | SAI1 global Interrupt |
| 88 | `LTDC_IRQn` | LTDC global Interrupt |
| 89 | `LTDC_ER_IRQn` | LTDC Error global Interrupt |
| 90 | `DMA2D_IRQn` | DMA2D global Interrupt |
| 91 | `SAI2_IRQn` | SAI2 global Interrupt |
| 92 | `QUADSPI_IRQn` | Quad SPI global interrupt |
| 93 | `LPTIM1_IRQn` | LP TIM1 interrupt |
| 94 | `CEC_IRQn` | HDMI-CEC global Interrupt |
| 95 | `I2C4_EV_IRQn` | I2C4 Event Interrupt |
| 96 | `I2C4_ER_IRQn` | I2C4 Error Interrupt |
| 97 | `SPDIF_RX_IRQn` | SPDIF-RX global Interrupt |


## USB / ST-LINK [신뢰도 A: 호스트 OS가 보고한 실제 연결 상태]

```
            STM32 STLink:

              Product ID: 0x374b
              Vendor ID: 0x0483  (STMicroelectronics)
              Version: 1.00
              Serial Number: 0668FF39314D423143034442
              Speed: Up to 12 Mb/s
              Manufacturer: STMicroelectronics
              Location ID: 0x08330000 / 117
              Current Available (mA): 500
              Current Required (mA): 500
              Extra Operating Current (mA): 0
              Media:
                microcontroller:
                  Capacity: 3.2 MB (3,182,592 bytes)
                  Removable Media: Yes
                  BSD Name: disk5
                  Logical Unit: 0
                  Partition Map Type: Unknown
                  S.M.A.R.T. status: Verified
                  USB Interface: 1

            AirPods Charging Case:

              Product ID: 0x13a1
              Vendor ID: 0x05ac (Apple Inc.)
              Version: 51.0b
              Serial Number: LNX23003GX
              Speed: Up to 12 Mb/s
```

```
Found 1 stlink programmers
  version:    V2J46S33
  serial:     0668FF39314D423143034442
  flash:      1048576 (pagesize: 2048)
  sram:       327680
  chipid:     0x449
  dev-type:   STM32F74x_F75x
```

```
crw-rw-rw-  1 root  wheel  0x9000003 Jul  3 02:00 /dev/cu.Bluetooth-Incoming-Port
crw-rw-rw-  1 root  wheel  0x9000001 Jul  3 00:05 /dev/cu.debug-console
crw-rw-rw-  1 root  wheel  0x9000005 Jul 27 15:17 /dev/cu.usbmodem83303
crw-rw-rw-  1 root  wheel  0x9000004 Jul 27 15:17 /dev/tty.usbmodem83303
```


## 기존 SNN 펌웨어의 메모리 실사용량 [신뢰도 A: 빌드 산출물]


### `build/`

```
/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/build/ImageClassification.elf  :
section                size        addr
.isr_vector             456   134217728
.text                104056   134218192
.rodata              232456   134322256
.ARM.extab                0   134554712
.ARM                      8   134554712
.preinit_array            0   134554720
.init_array               8   134554720
.fini_array               4   134554728
.data                   500   536870912
.bss                 266088   536871424
._user_heap_stack      1536   537137512
.ARM.attributes          48           0
.debug_info         4071549           0
.debug_abbrev        127617           0
.debug_loclists      284528           0
.debug_aranges        18008           0
.debug_rnglists       18982           0
.debug_macro         539489           0
.debug_line          677659           0
.debug_str          2112525           0
.comment                 67           0
.debug_frame          57528           0
.debug_line_str         159           0
Total               8513271
```

### `build-release/`

```
/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/build-release/ImageClassification.elf  :
section               size        addr
.isr_vector            456   134217728
.text               143672   134218192
.rodata             228992   134361872
.ARM.extab               0   134590864
.ARM                     8   134590864
.preinit_array           0   134590872
.init_array              8   134590872
.fini_array              4   134590880
.data                  496   536870912
.bss                266104   536871408
._user_heap_stack     1536   537137512
.ARM.attributes         48           0
.comment                67           0
.debug_frame          5316           0
Total               646707
```

### `build-v4/`

```
/Users/chacha/Desktop/git/snn-for-mcu/ImageClassification/build-v4/ImageClassification.elf  :
section               size        addr
.isr_vector            456   134217728
.text               107996   134218192
.rodata             216132   134326192
.ARM.extab               0   134542324
.ARM                     8   134542324
.preinit_array           0   134542332
.init_array              4   134542332
.fini_array              4   134542336
.data                  556   536870912
.bss                 83412   536871472
._user_heap_stack     1540   536954884
.ARM.attributes         48           0
.comment                67           0
.debug_frame          5052           0
Total               415275
```