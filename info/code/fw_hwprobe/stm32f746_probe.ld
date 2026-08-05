/*
 * STM32F746NGH6 하드웨어 프로브 펌웨어 링커 스크립트.
 *
 * 목적상 일반 애플리케이션과 다른 점:
 *  - 내부 메모리를 실제 물리 구획(ITCM / DTCM / SRAM1 / SRAM2)대로 쪼개서
 *    영역별 벤치마크 버퍼를 원하는 메모리에 정확히 배치할 수 있게 한다.
 *  - 측정 결과 텍스트를 담을 RESULT 영역을 SRAM1 끝의 고정 주소(0x20040000)에
 *    둔다. 펌웨어가 죽든 말든 ST-LINK로 이 주소를 그대로 읽어 결과를 회수한다.
 *  - 코드 페치 위치가 성능에 미치는 영향을 재기 위해 ITCM 실행용 섹션을 둔다.
 *
 * 메모리 경계는 RM0385 기준:
 *   ITCM   0x00000000 +16K
 *   DTCM   0x20000000 +64K
 *   SRAM1  0x20010000 +240K   (여기서 192K RAM + 48K RESULT 로 분할)
 *   SRAM2  0x2004C000 +16K
 *   FLASH  0x08000000 +1024K  (AXIM 별칭)
 */

ENTRY(Reset_Handler)

/*
 * SRAM1(240K)을 셋으로 쪼갠다:
 *   RAM    144K — .data/.bss/힙
 *   DIAG     4K — 부팅 진행 표식과 폴트 기록. 결과 버퍼와 반드시 분리해야 한다.
 *                 (예전에 진단 주소를 결과 버퍼 안에 두었다가, 로그가 버퍼를 채우면서
 *                  진단 기록을 덮어써 폴트 원인을 잃은 적이 있다.)
 *   RESULT  92K — 측정 로그 본문. 전체 측정 출력이 48K를 넘어 확대했다.
 */
MEMORY
{
  ITCM   (xrw) : ORIGIN = 0x00000000, LENGTH = 16K
  DTCM   (xrw) : ORIGIN = 0x20000000, LENGTH = 64K
  RAM    (xrw) : ORIGIN = 0x20010000, LENGTH = 144K
  DIAG   (rw)  : ORIGIN = 0x20034000, LENGTH = 4K
  RESULT (rw)  : ORIGIN = 0x20035000, LENGTH = 92K
  SRAM2  (xrw) : ORIGIN = 0x2004C000, LENGTH = 16K
  FLASH  (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
}

/* 스택은 DTCM 최상단에 둔다 (0 wait state, 캐시 무관하게 결정적) */
_estack = ORIGIN(DTCM) + LENGTH(DTCM);
_Min_Heap_Size  = 0x400;
_Min_Stack_Size = 0x2000;

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  /* 코드 페치 경로 비교용: FLASH에 원본을 두고 런타임에 ITCM으로 복사해 실행 */
  .itcm_code :
  {
    . = ALIGN(8);
    _sitcm_code = .;
    *(.itcm_code)
    *(.itcm_code*)
    . = ALIGN(8);
    _eitcm_code = .;
  } >ITCM AT> FLASH
  _sitcm_code_load = LOADADDR(.itcm_code);

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)); PROVIDE_HIDDEN(__preinit_array_end = .); } >FLASH
  .init_array    : { PROVIDE_HIDDEN(__init_array_start = .);    KEEP(*(SORT(.init_array.*))); KEEP(*(.init_array*)); PROVIDE_HIDDEN(__init_array_end = .); } >FLASH
  .fini_array    : { PROVIDE_HIDDEN(__fini_array_start = .);    KEEP(*(SORT(.fini_array.*))); KEEP(*(.fini_array*)); PROVIDE_HIDDEN(__fini_array_end = .); } >FLASH

  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >RAM AT> FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } >RAM

  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE(end = .);
    PROVIDE(_end = .);
    . = . + _Min_Heap_Size;
    . = ALIGN(8);
  } >RAM

  /* DTCM 벤치마크 버퍼 (NOLOAD: 초기화하지 않음) */
  .dtcm_bss (NOLOAD) :
  {
    . = ALIGN(32);
    _sdtcm = .;
    *(.dtcm_bss)
    *(.dtcm_bss*)
    . = ALIGN(32);
    _edtcm = .;
  } >DTCM

  /* SRAM2 벤치마크 버퍼 */
  .sram2_bss (NOLOAD) :
  {
    . = ALIGN(32);
    _ssram2 = .;
    *(.sram2_bss)
    *(.sram2_bss*)
    . = ALIGN(32);
    _esram2 = .;
  } >SRAM2

  /* ITCM 데이터 버퍼 (ITCM을 데이터로 접근할 때의 특성 측정용) */
  .itcm_bss (NOLOAD) :
  {
    . = ALIGN(32);
    _sitcm_bss = .;
    *(.itcm_bss)
    *(.itcm_bss*)
    . = ALIGN(32);
    _eitcm_bss = .;
  } >ITCM

  /* 결과 텍스트 버퍼: 고정 주소 0x20040000. ST-LINK로 그대로 읽어간다. */
  .probe_result (NOLOAD) :
  {
    . = ALIGN(32);
    _sprobe_result = .;
    KEEP(*(.probe_result))
    . = ALIGN(4);
    _eprobe_result = .;
  } >RESULT

  /DISCARD/ : { libc.a(*) libm.a(*) libgcc.a(*) }
  .ARM.attributes 0 : { *(.ARM.attributes) }
}
