cmake_minimum_required(VERSION 3.20)
project(hwprobe C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# HAL/BSP/CMSIS는 기존 ImageClassification 프로젝트 트리를 그대로 참조한다
# (드라이버를 복제하면 버전이 갈라져 측정 조건이 달라질 수 있다).
set(IC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ImageClassification)

set(MCU_FLAGS -mcpu=cortex-m7 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard)
set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/stm32f746_probe.ld)

file(GLOB PROBE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

# 필요한 HAL 모듈만: 클럭/전원/GPIO/FMC(SDRAM)/QSPI/DMA
set(HAL_SOURCES
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_cortex.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc_ex.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr_ex.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_gpio.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_i2c.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_i2c_ex.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_qspi.c
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.c
)

set(BSP_SOURCES
    ${IC_DIR}/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_sdram.c
    ${IC_DIR}/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_qspi.c
)

set(STARTUP ${IC_DIR}/Core/Startup/startup_stm32f746nghx.s)

add_executable(${PROJECT_NAME}.elf
    ${PROBE_SOURCES}
    ${HAL_SOURCES}
    ${BSP_SOURCES}
    ${STARTUP}
)

target_include_directories(${PROJECT_NAME}.elf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${IC_DIR}/Core/Inc                                    # stm32f7xx_hal_conf.h
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Inc
    ${IC_DIR}/Drivers/STM32F7xx_HAL_Driver/Inc/Legacy
    ${IC_DIR}/Drivers/CMSIS/Device/ST/STM32F7xx/Include
    ${IC_DIR}/Drivers/CMSIS/Include
    ${IC_DIR}/Drivers/BSP/STM32746G-Discovery
    ${IC_DIR}/Drivers/BSP/Components/Common
)

target_compile_definitions(${PROJECT_NAME}.elf PRIVATE
    USE_HAL_DRIVER
    STM32F746xx
    ARM_MATH_DSP
)

target_compile_options(${PROJECT_NAME}.elf PRIVATE
    ${MCU_FLAGS}
    -ffunction-sections -fdata-sections
    -Wall -Wextra -Wno-unused-parameter
    $<$<CONFIG:Debug>:-Og -g3>
    # 벤치마크 커널이 통째로 사라지지 않도록 LTO는 쓰지 않는다.
    $<$<CONFIG:Release>:-O2 -g>
)

target_link_options(${PROJECT_NAME}.elf PRIVATE
    ${MCU_FLAGS}
    -T${LINKER_SCRIPT}
    --specs=nano.specs
    --specs=nosys.specs
    -Wl,-Map=${PROJECT_NAME}.map,--cref
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    -static
)

set_target_properties(${PROJECT_NAME}.elf PROPERTIES LINK_DEPENDS ${LINKER_SCRIPT})

add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>
    COMMENT "hwprobe.bin 생성"
)
