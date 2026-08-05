/*
 * SNN 핫 코드/데이터 배치 매크로.
 *
 * 근거 (info/raw/measured_facts.md 실측):
 *  - 이 보드의 I-cache는 4KB뿐이고 추론 핫 코드는 ~6KB라 플래시(AXIM) 실행 시
 *    스래싱한다. ITCM RAM(16KB, 0x00000000)에서 실행하면 0.50 cyc/insn(이론 상한).
 *  - DTCM(64KB, 0x20000000)은 코어 직결이라 캐시 유지보수가 필요 없고
 *    로드 대역폭이 SRAM1의 2배다. SMLALD 내적은 로드가 병목이므로 효과가 크다.
 *
 * SNN_OPT_* 매크로가 꺼진 빌드에서는 빈 정의가 되어 기존 배치와 완전히 같다.
 * (.itcm_code/.dtcm_bss 섹션은 STM32F746NGHX_FLASH.ld 참조. .itcm_code는
 *  snn_itcm_init()이 부팅 시 플래시에서 ITCM RAM으로 복사한다.)
 */
#ifndef SNN_HOT_H
#define SNN_HOT_H

#if defined(SNN_OPT_ITCM) && (SNN_OPT_ITCM != 0)
/* noinline: 마킹 안 된 (플래시 상주) 호출자에 인라인되어 ITCM 배치가
 * 무효화되는 것을 막는다. 마킹된 함수끼리는 어차피 같은 섹션이다. */
#define SNN_ITCM_CODE __attribute__((section(".itcm_code"), noinline))
#else
#define SNN_ITCM_CODE
#endif

#if defined(SNN_OPT_DTCM) && (SNN_OPT_DTCM != 0)
#define SNN_DTCM_BSS __attribute__((section(".dtcm_bss"), aligned(4)))
#else
#define SNN_DTCM_BSS __attribute__((aligned(4)))
#endif

#endif /* SNN_HOT_H */
