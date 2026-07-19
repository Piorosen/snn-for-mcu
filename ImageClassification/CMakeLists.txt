cmake_minimum_required(VERSION 3.20)

project(ImageClassification C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# LTO: Release에서 전체 프로그램 최적화 (인라이닝/데드코드 제거 극대화)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(MCU_FLAGS -mcpu=cortex-m7 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard)
set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/STM32F746NGHX_FLASH.ld)

# --- 소스 목록 ---------------------------------------------------------------
file(GLOB HAL_SOURCES Drivers/STM32F7xx_HAL_Driver/Src/*.c)
list(FILTER HAL_SOURCES EXCLUDE REGEX ".*_template\\.c$")

file(GLOB BSP_SOURCES Drivers/BSP/STM32746G-Discovery/*.c)
file(GLOB_RECURSE BSP_COMPONENT_SOURCES Drivers/BSP/Components/*.c)

set(FATFS_SOURCES
    Middlewares/Third_Party/FatFs/src/ff.c
    Middlewares/Third_Party/FatFs/src/diskio.c
    Middlewares/Third_Party/FatFs/src/ff_gen_drv.c
    Middlewares/Third_Party/FatFs/src/option/syscall.c
    Middlewares/Third_Party/FatFs/src/option/ccsbcs.c
)

# Fonts는 stm32746g_discovery_lcd.c가 .c째로 include하므로 별도 컴파일하지 않음
file(GLOB LIBJPEG_SOURCES Middlewares/Third_Party/LibJPEG/source/*.c)

set(CORE_SOURCES
    Core/Src/main.cc
    Core/Src/snn.c
    Core/Src/snn_weights.c
    Core/Src/decode.c
    Core/Src/jdata_conf.c
    Core/Src/sd_diskio.c
    Core/Src/stm32f7xx_hal_msp.c
    Core/Src/stm32f7xx_hal_timebase_tim.c
    Core/Src/stm32f7xx_it.c
    Core/Src/syscalls.c
    Core/Src/sysmem.c
    Core/Src/system_stm32f7xx.c
    Core/Startup/startup_stm32f746nghx.s
)

add_executable(${PROJECT_NAME}.elf
    ${CORE_SOURCES}
    ${HAL_SOURCES}
    ${BSP_SOURCES}
    ${BSP_COMPONENT_SOURCES}
    ${FATFS_SOURCES}
    ${LIBJPEG_SOURCES}
)

target_include_directories(${PROJECT_NAME}.elf PRIVATE
    Core/Inc
    Drivers/STM32F7xx_HAL_Driver/Inc
    Drivers/STM32F7xx_HAL_Driver/Inc/Legacy
    Drivers/CMSIS/Device/ST/STM32F7xx/Include
    Drivers/CMSIS/Include
    Drivers/BSP/STM32746G-Discovery
    Drivers/BSP/Components/Common
    Middlewares/Third_Party/FatFs/src
    Middlewares/Third_Party/LibJPEG/include
    Utilities/Fonts
)

target_compile_definitions(${PROJECT_NAME}.elf PRIVATE
    USE_HAL_DRIVER
    STM32F746xx
)

target_compile_options(${PROJECT_NAME}.elf PRIVATE
    ${MCU_FLAGS}
    -ffunction-sections -fdata-sections
    -Wall
    $<$<CONFIG:Debug>:-Og -g3>
    $<$<CONFIG:Release>:-O3 -funroll-loops -fomit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>
)

# SNN 핫패스: FPU가 단정밀도(fpv5-sp)이므로 double 승격을 막고 fast-math 적용
set_source_files_properties(Core/Src/snn.c Core/Src/main.cc PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Release>:-ffast-math;-fsingle-precision-constant>"
)

target_link_options(${PROJECT_NAME}.elf PRIVATE
    ${MCU_FLAGS}
    -T${LINKER_SCRIPT}
    --specs=nano.specs
    -u _printf_float
    -Wl,--gc-sections
    -Wl,-Map=${PROJECT_NAME}.map
    -Wl,--print-memory-usage
    -static
    -Wl,--start-group -lc -lm -Wl,--end-group
)

set_target_properties(${PROJECT_NAME}.elf PROPERTIES LINK_DEPENDS ${LINKER_SCRIPT})

# --- bin/hex 생성 및 사이즈 출력 ---------------------------------------------
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>
)

# --- 보드 플래싱: cmake --build build --target flash --------------------------
# st-flash 1.8.0은 F74x 플래시 로더 오류로 실패하므로 CubeIDE 내장 CLI 사용
set(STM32_PROGRAMMER "/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.macosaarch64_1.0.100.202603051304/tools/bin/STM32_Programmer_CLI")
add_custom_target(flash
    COMMAND ${STM32_PROGRAMMER} -c port=SWD -w ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.bin 0x08000000 -v -rst
    DEPENDS ${PROJECT_NAME}.elf
    COMMENT "ST-LINK로 펌웨어 플래싱"
)
