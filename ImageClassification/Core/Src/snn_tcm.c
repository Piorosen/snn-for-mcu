/*
 * .itcm_code 섹션 적재기.
 *
 * 링커(STM32F746NGHX_FLASH.ld)가 핫 커널 코드를 LMA=플래시 / VMA=ITCM RAM
 * (0x00000000, 16KB)으로 배치한다. 부팅 후 이 함수가 플래시 적재분을 ITCM으로
 * 복사해야 해당 코드가 실행 가능해진다.
 *
 * 링커 심볼은 char 배열로 선언한다 — uint32_t 스칼라로 선언하면 LTO가
 * "서로 다른 객체의 주소 비교"를 UB로 보고 바이트 수 계산을 0으로 접는다
 * (실측: 복사는 수행되는데 반환값만 0이 되는 오컴파일이 실제로 발생했다).
 *
 * ITCM RAM은 캐시 대상이 아니고(TCM은 코어 직결) ART는 플래시 전용이므로,
 * 복사 후 캐시 유지보수는 필요 없다 — DSB/ISB로 충분하다
 * (info/docs/02_cache_tcm.md §3.3, §5.3).
 */
#include "snn.h"
#include <string.h>
#include "stm32f7xx.h"

#if defined(SNN_OPT_ITCM) && (SNN_OPT_ITCM != 0)

extern char _sitcm_code[], _eitcm_code[], _sitcm_code_load[];

uint32_t snn_itcm_init(void)
{
    const uint32_t n = (uint32_t)(_eitcm_code - _sitcm_code);
    memcpy(_sitcm_code, _sitcm_code_load, n);
    __DSB();
    __ISB();
    return n;
}

#else

uint32_t snn_itcm_init(void) { return 0; }

#endif
