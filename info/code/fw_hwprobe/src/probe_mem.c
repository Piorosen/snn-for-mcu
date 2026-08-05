/*
 * probe_mem.c - 메모리 크기 / 대역폭 / 레이턴시 / 캐시 특성 실측.
 *
 * 이 파일의 모든 수치는 STM32F746NGH6 실물에서 CPU 로드/스토어로 직접 잰 값이다.
 * 데이터시트 값은 "비교 대상"으로만 쓰고, 판정은 실측이 한다.
 *
 * 핵심 설계
 *  - 폴트 내성 프로브: 존재하지 않는 주소를 찔러 봐야 경계를 알 수 있는데, 그러면
 *    버스 폴트가 난다. probe_system.c의 폴트 핸들러는 원인을 기록하고 멈추도록
 *    되어 있어 그대로 두면 측정이 거기서 끝난다. 그래서 이 모듈이 도는 동안에는
 *    벡터 테이블을 DTCM으로 옮기고(VTOR) 자체 폴트 핸들러를 걸어, setjmp/longjmp
 *    유사 방식으로 폴트에서 빠져나온다. 모듈이 끝나면 VTOR을 원복한다.
 *  - 측정 커널은 전부 인라인 어셈블리다. -O2에서 로드/스토어가 소거되지 않게 하고,
 *    LDR / LDRD / LDM 중 무엇이 실행되는지를 컴파일러 재량에 맡기지 않기 위함이다.
 *  - 플래시 접근 특성 측정 루프는 .itcm_code(ITCM)에 둔다. ART/캐시를 끄면 플래시에
 *    있는 측정 루프 자체가 느려져 데이터 접근 측정이 오염되기 때문이다.
 *  - 나머지 측정 커널은 플래시(.text)에 있다. I-cache가 꺼져 있으면 6 wait state
 *    플래시에서 루프 본체를 매 반복 재페치하게 되어 명령 페치 비용이 데이터 측정에
 *    섞인다. 그래서 이 모듈은 시작할 때 I-cache를 켜고(그 사실을 출력하고) 끝나면
 *    원래 상태로 되돌린다.
 *  - 이 펌웨어는 MPU를 설정하지 않는다. 그래서 0xC0000000(SDRAM)은 ARMv7-M 기본
 *    메모리맵의 Device 영역이고, D-cache에 캐싱되지 않으며 비정렬 접근은 항상
 *    UsageFault다. SDRAM의 캐시 효과/정렬 페널티를 실제로 재기 위해 이 모듈은
 *    측정 구간에서만 MPU 리전 0으로 SDRAM을 Normal WBWA로 재정의하고, 기본맵
 *    조건과 나란히 비교한 뒤 MPU를 다시 끈다.
 */
#include "probe.h"
#include "stm32f7xx.h"

/* ------------------------------------------------------------------------- */
/* 영역 테이블                                                                */
/* ------------------------------------------------------------------------- */
enum {
    R_ITCM = 0, R_DTCM, R_SRAM1, R_SRAM2,
    R_FAXI, R_FITCM, R_BKP, R_SDRAM, R_QSPI, R_COUNT
};

static probe_region_t s_regions[] = {
    { "ITCM",        0x00000000u,  16u * 1024u,         MEMKIND_TCM,   1, 1 },
    { "DTCM",        0x20000000u,  64u * 1024u,         MEMKIND_TCM,   1, 1 },
    { "SRAM1",       0x20010000u, 240u * 1024u,         MEMKIND_SRAM,  1, 1 },
    { "SRAM2",       0x2004C000u,  16u * 1024u,         MEMKIND_SRAM,  1, 1 },
    { "FLASH_AXIM",  0x08000000u, 1024u * 1024u,        MEMKIND_FLASH, 0, 1 },
    { "FLASH_ITCM",  0x00200000u, 1024u * 1024u,        MEMKIND_FLASH, 0, 1 },
    { "BKPSRAM",     0x40024000u,   4u * 1024u,         MEMKIND_BKP,   1, 0 },
    { "SDRAM",       0xC0000000u,   8u * 1024u * 1024u, MEMKIND_EXT,   1, 0 },
    { "QSPI",        0x90000000u,  16u * 1024u * 1024u, MEMKIND_EXT,   0, 0 },
};

const probe_region_t *probe_regions(void) { return s_regions; }
int probe_region_count(void) { return (int)(sizeof(s_regions) / sizeof(s_regions[0])); }

/* ------------------------------------------------------------------------- */
/* 벤치마크 버퍼 (링커 스크립트의 물리 메모리별 섹션에 정확히 배치)             */
/* ------------------------------------------------------------------------- */
#define SRAM1_WORDS   (16u * 1024u)                    /*  64 KB - SRAM1(.bss) */
#define DTCM_WORDS    ( 8u * 1024u)                    /*  32 KB - DTCM        */
#define SRAM2_WORDS   ( 2u * 1024u)                    /*   8 KB - SRAM2       */
#define ITCM_WORDS    ( 1u * 1024u)                    /*   4 KB - ITCM        */

static uint32_t s_sram1_buf[SRAM1_WORDS] __attribute__((aligned(32)));
static uint32_t s_dtcm_buf[DTCM_WORDS]   __attribute__((aligned(32), section(".dtcm_bss")));
static uint32_t s_sram2_buf[SRAM2_WORDS] __attribute__((aligned(32), section(".sram2_bss")));
static uint32_t s_itcm_buf[ITCM_WORDS]   __attribute__((aligned(32), section(".itcm_bss")));

#define SDRAM_ADDR 0xC0000000u
#define QSPI_ADDR  0x90000000u
#define BKP_ADDR   0x40024000u
#define FAXI_ADDR  0x08000000u
#define FITCM_ADDR 0x00200000u

/* 외부 메모리 워킹셋 (실측 후 available 이 서면 쓴다) */
#define EXT_WS_BYTES (64u * 1024u)

static volatile uint32_t s_sink;

/* 외부/백업 메모리 실측 판정 결과 */
static int s_sdram_ok, s_qspi_ok, s_bkp_ok;

/* ------------------------------------------------------------------------- */
/* 폴트 내성 프로브 (setjmp/longjmp 유사)                                     */
/* ------------------------------------------------------------------------- */
/*
 * pm_try_begin()은 콜리세이브 레지스터 + SP + 복귀주소를 s_ctx에 적고 0을 돌려준다.
 * 가드 구간에서 폴트가 나면 폴트 핸들러가 예외 프레임의 PC를 pm_try_resume으로
 * 바꾸고 r0에 s_ctx를 실어 예외 복귀시킨다. pm_try_resume은 저장해 둔 문맥을
 * 되살리고 1을 돌려주며 pm_try_begin의 호출 지점으로 돌아간다.
 */
static uint32_t s_ctx[10];               /* r4-r11, sp, 복귀주소 */
static volatile uint32_t s_guard_active;
static volatile uint32_t s_fault_hit;
static volatile uint32_t s_fault_cfsr;
static volatile uint32_t s_fault_count;

/* 자체 벡터 테이블: DTCM(캐시/ART 무관, 0 wait) */
#define VTAB_ENTRIES 128u
static uint32_t s_vtab[VTAB_ENTRIES] __attribute__((aligned(512), section(".dtcm_bss")));
static uint32_t s_vtor_saved;
static int      s_guard_installed;

__attribute__((naked, noinline, returns_twice))
static int pm_try_begin(uint32_t *ctx)
{
    __asm volatile(
        "stmia r0!, {r4-r11}\n\t"
        "mov   r1, sp\n\t"
        "stmia r0!, {r1, lr}\n\t"
        "movs  r0, #0\n\t"
        "bx    lr\n\t");
}

__attribute__((naked, noinline, used))
static void pm_try_resume(void)
{
    /* r0 = s_ctx (폴트 핸들러가 예외 프레임의 r0에 심어 둔다) */
    __asm volatile(
        "ldmia r0!, {r4-r11}\n\t"
        "ldmia r0, {r1, r2}\n\t"
        "mov   sp, r1\n\t"
        "movs  r0, #1\n\t"
        "bx    r2\n\t");
}

/* probe_system.c의 FAULT_LOG와 같은 자리에 기록한다 (결과 버퍼 밖) */
#define PM_FAULT_LOG ((volatile uint32_t *)PROBE_DIAG_MEM_LOG)

__attribute__((used))
static void pm_fault_fix(uint32_t *frame)
{
    uint32_t cfsr = SCB->CFSR;
    uint32_t abfsr = *(volatile uint32_t *)0xE000EFA8u;

    if (!s_guard_active) {
        /* 가드 밖에서 난 폴트는 복구할 근거가 없다. 원래 핸들러와 같은 자리에
         * 기록하고 멈춘다 (진단 가능성을 유지). */
        PM_FAULT_LOG[0] = 0xDEADBEEFu;
        PM_FAULT_LOG[1] = 0x4D454Du;      /* 'MEM' - probe_mem 구간에서 발생 */
        PM_FAULT_LOG[2] = cfsr;
        PM_FAULT_LOG[3] = SCB->HFSR;
        PM_FAULT_LOG[4] = SCB->MMFAR;
        PM_FAULT_LOG[5] = SCB->BFAR;
        PM_FAULT_LOG[6] = frame[6];       /* 폴트 PC */
        PM_FAULT_LOG[7] = abfsr;
        __DSB();
        for (;;) { }
    }

    s_fault_hit  = 1u;
    s_fault_cfsr = cfsr;
    s_fault_count++;

    SCB->CFSR = cfsr;                                  /* write-1-to-clear */
    SCB->HFSR = SCB->HFSR;
    *(volatile uint32_t *)0xE000EFA8u = abfsr;         /* ABFSR write-1-to-clear */

    frame[0] = (uint32_t)(uintptr_t)s_ctx;             /* r0  */
    frame[6] = ((uint32_t)(uintptr_t)&pm_try_resume) | 1u;  /* PC */
    frame[7] = 0x01000000u;                            /* xPSR: Thumb 만 */
    __DSB();
    __ISB();
}

__attribute__((naked, used))
static void pm_fault_entry(void)
{
    /* 이 펌웨어는 스레드 모드에서도 MSP를 쓴다 → 예외 프레임은 현재 SP 위치 */
    __asm volatile(
        "mov   r0, sp\n\t"
        "push  {r4, lr}\n\t"       /* EXC_RETURN 보존 (8바이트 정렬 유지) */
        "bl    pm_fault_fix\n\t"
        "pop   {r4, pc}\n\t");     /* pc <- EXC_RETURN → 예외 복귀 */
}

static void guard_install(void)
{
    if (s_guard_installed) return;
    s_vtor_saved = SCB->VTOR;
    const uint32_t *src = (const uint32_t *)(uintptr_t)s_vtor_saved;
    for (uint32_t i = 0; i < VTAB_ENTRIES; i++) s_vtab[i] = src[i];
    s_vtab[3] = ((uint32_t)(uintptr_t)&pm_fault_entry) | 1u;   /* HardFault  */
    s_vtab[4] = s_vtab[3];                                     /* MemManage  */
    s_vtab[5] = s_vtab[3];                                     /* BusFault   */
    s_vtab[6] = s_vtab[3];                                     /* UsageFault */
    __DSB();
    SCB->VTOR = (uint32_t)(uintptr_t)s_vtab;
    __DSB();
    __ISB();
    s_guard_installed = 1;
}

static void guard_remove(void)
{
    if (!s_guard_installed) return;
    __DSB();
    SCB->VTOR = s_vtor_saved;
    __DSB();
    __ISB();
    s_guard_installed = 0;
}

/* ------------------------------------------------------------------------- */
/* 주소 안전성 판정                                                            */
/* ------------------------------------------------------------------------- */
/* 0 = 건드리면 안 됨(측정 중단), 1 = 시험 가능, 2 = 실재가 확실하나 건드리면 안 됨 */
extern uint32_t _ebss;

static int addr_class(uint32_t a)
{
    uint32_t sp = __get_MSP();

    /* 스택: 지금 이 코드가 쓰고 있다. 실재하는 RAM 이므로 '있다'로 간주하고 통과 */
    if (a >= (sp - 4096u) && a < 0x20010000u) return 2;
    /* 결과 버퍼(로그 본문). 실재하는 SRAM1 이므로 '있다'로 간주 */
    if (a >= PROBE_RESULT_ADDR && a < PROBE_RESULT_ADDR + PROBE_RESULT_SIZE) return 2;
    /* 진단 영역(부팅 표식·폴트 기록). 덮으면 사고 원인을 잃는다 */
    if (a >= PROBE_DIAG_ADDR && a < PROBE_DIAG_ADDR + PROBE_DIAG_SIZE) return 2;
    /* 폴트 복구 장치 자신(벡터 테이블/문맥 저장소)은 절대 건드리지 않는다 */
    if (a >= (uint32_t)(uintptr_t)s_vtab &&
        a <  (uint32_t)(uintptr_t)s_vtab + sizeof(s_vtab)) return 2;
    if (a >= (uint32_t)(uintptr_t)s_ctx &&
        a <  (uint32_t)(uintptr_t)s_ctx + sizeof(s_ctx)) return 2;

    /* 페리페럴 레지스터 공간: 임의 쓰기는 위험하다. BKPSRAM과 그 직후 예약구간만 허용 */
    if (a >= 0x40000000u && a < 0x60000000u) {
        if (a >= BKP_ADDR && a < BKP_ADDR + 0x2000u) return 1;
        return 0;
    }
    if (a >= 0xA0000000u && a < 0xB0000000u) return 0;   /* FMC/QUADSPI 제어 레지스터 */
    if (a >= 0xE0000000u) return 0;                      /* PPB */
    return 1;
}

/* ------------------------------------------------------------------------- */
/* 가드 걸린 단위 접근                                                        */
/* ------------------------------------------------------------------------- */
#define MARK_A 0x5A5A0F0Fu

static volatile uint32_t s_t_orig, s_t_back, s_b_back;

/* 반환: 0=정상(실재/되감김 없음) 1=되감김 2=쓰기 유지 실패 3=폴트 */
static int cell_test(volatile uint32_t *base, uint32_t addr, int writable, uint32_t mark_b)
{
    volatile uint32_t *t = (volatile uint32_t *)(uintptr_t)addr;
    int rc;

    s_fault_hit = 0;
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) != 0) {
        s_guard_active = 0;
        return 3;
    }

    if (!writable) {
        s_t_back = *t;
        __DSB();
        s_guard_active = 0;
        return 0;
    }

    s_t_orig = *t;
    __DSB();
    *t = mark_b;
    __DSB();
    s_b_back = *base;
    s_t_back = *t;
    *t = s_t_orig;                /* 원상 복구 (되감김이면 base=MARK_A 가 복원된다) */
    __DSB();
    s_guard_active = 0;

    if (s_b_back != MARK_A) rc = 1;
    else if (s_t_back != mark_b) rc = 2;
    else rc = 0;
    return rc;
}

/* 가드 걸고 한 워드를 쓴 뒤 되읽어 확인한다. 1=쓰기 유지, 0=폴트/유지 실패 */
static int guarded_write_check(uint32_t addr, uint32_t val)
{
    s_fault_hit = 0;
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) != 0) { s_guard_active = 0; return 0; }
    *(volatile uint32_t *)(uintptr_t)addr = val;
    __DSB();
    s_b_back = *(volatile uint32_t *)(uintptr_t)addr;
    s_guard_active = 0;
    return (s_b_back == val) ? 1 : 0;
}

/* 가드 걸고 한 워드를 쓴다 (원상 복구용, 결과는 보지 않는다) */
static void guarded_store(uint32_t addr, uint32_t val)
{
    s_fault_hit = 0;
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) == 0) {
        *(volatile uint32_t *)(uintptr_t)addr = val;
        __DSB();
    }
    s_guard_active = 0;
}

/* base 한 워드를 가드 걸고 읽는다. 0=폴트 */
static int guarded_read(uint32_t addr, uint32_t *out)
{
    s_fault_hit = 0;
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) != 0) { s_guard_active = 0; return 0; }
    s_t_back = *(volatile uint32_t *)(uintptr_t)addr;
    __DSB();
    s_guard_active = 0;
    *out = s_t_back;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* 측정 커널 (전부 인라인 어셈블리)                                           */
/* ------------------------------------------------------------------------- */
typedef void (*mem_kern_t)(void *dst, const void *src, uint32_t n, uint32_t p);

/* 오버헤드 전용 커널. 다른 커널과 프롤로그 형태를 똑같이 맞춘다 (n=0으로 부르면
 * 루프에 들어가지 않고 바로 복귀한다). noinline: 오버헤드 측정이 run_min의 간접
 * 호출 경로를 그대로 타야 하므로 절대 인라인되면 안 된다. */
__attribute__((noinline))
static void k_nop(void *dst, const void *src, uint32_t n, uint32_t p)
{
    uint32_t c = n >> 3;
    if (!c) return;
    __asm volatile("" : "+r"(c) :: "memory");
}

/* 순차 읽기: LDR(32비트) 8개 언롤. n = 워드 수 (8의 배수) */
static void k_read32(void *dst, const void *src, uint32_t n, uint32_t p)
{
    const uint32_t *q = (const uint32_t *)src;
    uint32_t c = n >> 3, t0, t1, t2, t3;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldr %[t0], [%[q], #0]\n\t"
        "ldr %[t1], [%[q], #4]\n\t"
        "ldr %[t2], [%[q], #8]\n\t"
        "ldr %[t3], [%[q], #12]\n\t"
        "ldr %[t0], [%[q], #16]\n\t"
        "ldr %[t1], [%[q], #20]\n\t"
        "ldr %[t2], [%[q], #24]\n\t"
        "ldr %[t3], [%[q], #28]\n\t"
        "adds %[q], %[q], #32\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [t0] "=&r"(t0), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3),
          [q] "+r"(q), [c] "+r"(c)
        :
        : "cc", "memory");
    (void)t0; (void)t1; (void)t2; (void)t3;
}

/* 순차 읽기: LDRD(64비트) 4개 언롤. n = 워드 수 (8의 배수) */
static void k_read64(void *dst, const void *src, uint32_t n, uint32_t p)
{
    const uint32_t *q = (const uint32_t *)src;
    uint32_t c = n >> 3, t0, t1, t2, t3;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldrd %[t0], %[t1], [%[q], #0]\n\t"
        "ldrd %[t2], %[t3], [%[q], #8]\n\t"
        "ldrd %[t0], %[t1], [%[q], #16]\n\t"
        "ldrd %[t2], %[t3], [%[q], #24]\n\t"
        "adds %[q], %[q], #32\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [t0] "=&r"(t0), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3),
          [q] "+r"(q), [c] "+r"(c)
        :
        : "cc", "memory");
    (void)t0; (void)t1; (void)t2; (void)t3;
}

/* 순차 읽기: LDM 8레지스터 x2 (한 번에 32바이트). n = 워드 수 (16의 배수) */
static void k_readldm(void *dst, const void *src, uint32_t n, uint32_t p)
{
    const uint32_t *q = (const uint32_t *)src;
    uint32_t c = n >> 4;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldmia %[q]!, {r4-r6, r8-r12}\n\t"
        "ldmia %[q]!, {r4-r6, r8-r12}\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [q] "+r"(q), [c] "+r"(c)
        :
        : "cc", "memory", "r4", "r5", "r6", "r8", "r9", "r10", "r11", "r12");
}

/* 순차 쓰기: STR(32비트) 8개 언롤 */
static void k_write32(void *dst, const void *src, uint32_t n, uint32_t p)
{
    uint32_t *q = (uint32_t *)dst;
    uint32_t c = n >> 3;
    uint32_t v = 0xA5A5A5A5u;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "str %[v], [%[q], #0]\n\t"
        "str %[v], [%[q], #4]\n\t"
        "str %[v], [%[q], #8]\n\t"
        "str %[v], [%[q], #12]\n\t"
        "str %[v], [%[q], #16]\n\t"
        "str %[v], [%[q], #20]\n\t"
        "str %[v], [%[q], #24]\n\t"
        "str %[v], [%[q], #28]\n\t"
        "adds %[q], %[q], #32\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [q] "+r"(q), [c] "+r"(c)
        : [v] "r"(v)
        : "cc", "memory");
}

/* 순차 쓰기: STRD(64비트) 4개 언롤 */
static void k_write64(void *dst, const void *src, uint32_t n, uint32_t p)
{
    uint32_t *q = (uint32_t *)dst;
    uint32_t c = n >> 3;
    uint32_t v0 = 0xA5A5A5A5u, v1 = 0x5A5A5A5Au;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "strd %[v0], %[v1], [%[q], #0]\n\t"
        "strd %[v0], %[v1], [%[q], #8]\n\t"
        "strd %[v0], %[v1], [%[q], #16]\n\t"
        "strd %[v0], %[v1], [%[q], #24]\n\t"
        "adds %[q], %[q], #32\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [q] "+r"(q), [c] "+r"(c)
        : [v0] "r"(v0), [v1] "r"(v1)
        : "cc", "memory");
}

/* 순차 쓰기: STM 8레지스터 x2 */
static void k_writestm(void *dst, const void *src, uint32_t n, uint32_t p)
{
    uint32_t *q = (uint32_t *)dst;
    uint32_t c = n >> 4;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "stmia %[q]!, {r4-r6, r8-r12}\n\t"
        "stmia %[q]!, {r4-r6, r8-r12}\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [q] "+r"(q), [c] "+r"(c)
        :
        : "cc", "memory", "r4", "r5", "r6", "r8", "r9", "r10", "r11", "r12");
}

/* 복사: LDR/STR 4쌍 (32비트) */
static void k_copy32(void *dst, const void *src, uint32_t n, uint32_t p)
{
    uint32_t *d = (uint32_t *)dst;
    const uint32_t *s = (const uint32_t *)src;
    uint32_t c = n >> 2, t0, t1, t2, t3;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldr %[t0], [%[s], #0]\n\t"
        "ldr %[t1], [%[s], #4]\n\t"
        "ldr %[t2], [%[s], #8]\n\t"
        "ldr %[t3], [%[s], #12]\n\t"
        "str %[t0], [%[d], #0]\n\t"
        "str %[t1], [%[d], #4]\n\t"
        "str %[t2], [%[d], #8]\n\t"
        "str %[t3], [%[d], #12]\n\t"
        "adds %[s], %[s], #16\n\t"
        "adds %[d], %[d], #16\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [t0] "=&r"(t0), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3),
          [s] "+r"(s), [d] "+r"(d), [c] "+r"(c)
        :
        : "cc", "memory");
    (void)t0; (void)t1; (void)t2; (void)t3;
}

/* 복사: LDM/STM 8레지스터 (memcpy 상당의 최적 형태) */
static void k_copyldm(void *dst, const void *src, uint32_t n, uint32_t p)
{
    uint32_t *d = (uint32_t *)dst;
    const uint32_t *s = (const uint32_t *)src;
    uint32_t c = n >> 3;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldmia %[s]!, {r4-r6, r8-r12}\n\t"
        "stmia %[d]!, {r4-r6, r8-r12}\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [s] "+r"(s), [d] "+r"(d), [c] "+r"(c)
        :
        : "cc", "memory", "r4", "r5", "r6", "r8", "r9", "r10", "r11", "r12");
}

/* 스트라이드 읽기: n회 접근, p바이트 간격 (정렬 페널티/캐시라인 측정용) */
static void k_stride(void *dst, const void *src, uint32_t n, uint32_t p)
{
    const uint8_t *q = (const uint8_t *)src;
    uint32_t c = n, t0;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldr %[t0], [%[q]]\n\t"
        "add %[q], %[q], %[s]\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [t0] "=&r"(t0), [q] "+r"(q), [c] "+r"(c)
        : [s] "r"(p)
        : "cc", "memory");
    (void)t0;
}

/* 스트라이드 루프에서 로드만 뺀 것 (루프 오버헤드 측정용) */
static void k_stride_empty(void *dst, const void *src, uint32_t n, uint32_t p)
{
    const uint8_t *q = (const uint8_t *)src;
    uint32_t c = n;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "add %[q], %[q], %[s]\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [q] "+r"(q), [c] "+r"(c)
        : [s] "r"(p)
        : "cc", "memory");
}

/* 포인터 체이싱: p = *(void**)p 를 n회. 의존 사슬이라 실효 레이턴시가 그대로 나온다 */
static void k_chase(void *dst, const void *src, uint32_t n, uint32_t p)
{
    const void *q = src;
    uint32_t c = n;
    if (!c) return;
    __asm volatile(
        "1:\n\t"
        "ldr %[q], [%[q]]\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [q] "+r"(q), [c] "+r"(c)
        :
        : "cc", "memory");
    s_sink = (uint32_t)(uintptr_t)q;
}

/* ------------------------------------------------------------------------- */
/* ITCM 상주 측정 루프 (플래시/ART 측정 전용)                                 */
/* ------------------------------------------------------------------------- */
/* main.c의 itcm_code_load()가 이 코드를 ITCM으로 복사한 뒤에 호출되어야 한다.  */

/* itcm_timed_* 와 완전히 같은 계측 프레임에서 루프만 뺀 것.
 * 여기서 나온 값을 빼야 6절의 기준선이 run_min(s_ovh를 뺀다)과 맞춰진다. */
__attribute__((section(".itcm_code"), noinline))
static uint32_t itcm_timed_ovh(void)
{
    uint32_t start, end, pm;
    pm = __get_PRIMASK();
    __disable_irq();
    __asm volatile("dsb 0xF" ::: "memory");
    __asm volatile("isb 0xF" ::: "memory");
    start = DWT->CYCCNT;
    __asm volatile("dsb 0xF" ::: "memory");
    __asm volatile("isb 0xF" ::: "memory");
    end = DWT->CYCCNT;
    __set_PRIMASK(pm);
    return end - start;
}

__attribute__((section(".itcm_code"), noinline))
static uint32_t itcm_timed_read32(const void *src, uint32_t nwords)
{
    const uint32_t *q = (const uint32_t *)src;
    uint32_t c = nwords >> 3, t0, t1, t2, t3;
    uint32_t start, end, pm;
    if (!c) return 0;
    pm = __get_PRIMASK();
    __disable_irq();
    __asm volatile("dsb 0xF" ::: "memory");
    __asm volatile("isb 0xF" ::: "memory");
    start = DWT->CYCCNT;
    __asm volatile(
        "1:\n\t"
        "ldr %[t0], [%[q], #0]\n\t"
        "ldr %[t1], [%[q], #4]\n\t"
        "ldr %[t2], [%[q], #8]\n\t"
        "ldr %[t3], [%[q], #12]\n\t"
        "ldr %[t0], [%[q], #16]\n\t"
        "ldr %[t1], [%[q], #20]\n\t"
        "ldr %[t2], [%[q], #24]\n\t"
        "ldr %[t3], [%[q], #28]\n\t"
        "adds %[q], %[q], #32\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [t0] "=&r"(t0), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3),
          [q] "+r"(q), [c] "+r"(c)
        :
        : "cc", "memory");
    __asm volatile("dsb 0xF" ::: "memory");
    __asm volatile("isb 0xF" ::: "memory");
    end = DWT->CYCCNT;
    __set_PRIMASK(pm);
    (void)t0; (void)t1; (void)t2; (void)t3;
    return end - start;
}

__attribute__((section(".itcm_code"), noinline))
static uint32_t itcm_timed_stride(const void *src, uint32_t n, uint32_t stride)
{
    const uint8_t *q = (const uint8_t *)src;
    uint32_t c = n, t0;
    uint32_t start, end, pm;
    if (!c) return 0;
    pm = __get_PRIMASK();
    __disable_irq();
    __asm volatile("dsb 0xF" ::: "memory");
    __asm volatile("isb 0xF" ::: "memory");
    start = DWT->CYCCNT;
    __asm volatile(
        "1:\n\t"
        "ldr %[t0], [%[q]]\n\t"
        "add %[q], %[q], %[s]\n\t"
        "subs %[c], %[c], #1\n\t"
        "bne 1b\n\t"
        : [t0] "=&r"(t0), [q] "+r"(q), [c] "+r"(c)
        : [s] "r"(stride)
        : "cc", "memory");
    __asm volatile("dsb 0xF" ::: "memory");
    __asm volatile("isb 0xF" ::: "memory");
    end = DWT->CYCCNT;
    __set_PRIMASK(pm);
    (void)t0;
    return end - start;
}

/* ------------------------------------------------------------------------- */
/* 측정 하네스                                                                */
/* ------------------------------------------------------------------------- */
static uint32_t s_ovh;                /* run_min 경로의 빈 측정 오버헤드 (사이클) */
static uint32_t s_flash_frame_ovh;    /* 6절 ITCM 계측 프레임의 오버헤드 (사이클) */

#define MODE_WARM 0             /* 한 번 데운 뒤 측정 (정상 상태) */
#define MODE_COLD 1             /* 매 반복 전에 D-cache clean+invalidate (콜드) */

/* noinline: 이 함수가 호출 지점마다 인라인/특수화되면 오버헤드 측정(measure_overhead)이
 * 실제 측정과 다른 코드로 컴파일된다. 반드시 같은 기계어를 타야 한다. */
__attribute__((noinline))
static uint32_t run_min(mem_kern_t k, void *dst, const void *src,
                        uint32_t n, uint32_t p, int reps, int mode)
{
    uint32_t best = 0xFFFFFFFFu;

    if (mode == MODE_WARM) k(dst, src, n, p);       /* 워밍업 */

    for (int i = 0; i < reps; i++) {
        uint32_t pm, t0, d;
        if (mode == MODE_COLD) {
            SCB_CleanInvalidateDCache();            /* 클린 없이 무효화하면 로그가 날아간다 */
            __DSB(); __ISB();
        }
        pm = __get_PRIMASK();
        __disable_irq();
        PROBE_BARRIER();
        t0 = DWT->CYCCNT;
        k(dst, src, n, p);
        PROBE_BARRIER();
        d = DWT->CYCCNT - t0;
        __set_PRIMASK(pm);
        if (d < best) best = d;
    }
    return (best > s_ovh) ? (best - s_ovh) : 0u;
}

/* volatile 함수 포인터: 컴파일러가 k_nop을 상수 전파해 직접 호출/인라인으로
 * 바꾸지 못하게 막는다. 오버헤드는 반드시 실제 측정과 같은 blx 경로로 재야 한다. */
static mem_kern_t volatile s_knop = k_nop;

static void measure_overhead(void)
{
    /* run_min 자신을 그대로 쓴다 → 인자 셋업 + blx + 커널 프롤로그/에필로그 +
     * dsb/isb + CYCCNT 2회 읽기까지 실제 측정과 완전히 동일한 경로가 포함된다. */
    s_ovh = 0;                       /* run_min이 빼기 전에 0으로 둔다 */
    s_ovh = run_min(s_knop, 0, 0, 0, 0, 8, MODE_WARM);
}

/* run_min을 폴트 가드 안에서 수행한다. 폴트가 나면 *ok=0 으로 알리고 0을 돌려준다.
 * (가드 밖에서 폴트가 나면 pm_fault_fix가 기록만 남기고 무한 루프에 빠지므로,
 *  실재 여부가 확실하지 않은 주소나 비정렬 접근은 반드시 이쪽을 써야 한다.) */
static volatile uint32_t s_saved_pm;

static uint32_t run_min_guarded(mem_kern_t k, void *dst, const void *src,
                                uint32_t n, uint32_t p, int reps, int mode, int *ok)
{
    uint32_t r;

    s_fault_hit   = 0;
    s_saved_pm    = __get_PRIMASK();
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) != 0) {
        /* run_min은 PRIMASK=1 상태에서 폴트를 맞았을 수 있다 → 반드시 되돌린다 */
        __set_PRIMASK(s_saved_pm);
        s_guard_active = 0;
        if (ok) *ok = 0;
        return 0u;
    }
    r = run_min(k, dst, src, n, p, reps, mode);
    s_guard_active = 0;
    if (ok) *ok = 1;
    return r;
}

/* ------------------------------------------------------------------------- */
/* 수치 출력 (64비트 나눗셈 금지: 링커가 libgcc를 버린다)                     */
/* ------------------------------------------------------------------------- */
static uint32_t bpc100(uint32_t bytes, uint32_t cycles)
{
    if (!cycles) return 0;
    return (bytes * 100u + cycles / 2u) / cycles;      /* bytes<=512KB → 오버플로 없음 */
}

static uint32_t mbps100(uint32_t bytes, uint32_t cycles)
{
    uint32_t k, a, q, r, res;
    if (!cycles) return 0;
    k = g_sysclk_hz / 1000000u;                       /* MHz */
    a = bytes * 100u;
    q = a / cycles;
    r = a % cycles;
    res = q * k;
    if (cycles < 20000000u) res += (r * k) / cycles;
    return res;                                        /* MB/s x100 */
}

static void pr_pad(const char *s, int w)
{
    int n = 0;
    while (s[n]) n++;
    probe_puts(s);
    while (n < w) { probe_puts(" "); n++; }
}

static void pr_bw(uint32_t bytes, uint32_t cycles)
{
    uint32_t m = mbps100(bytes, cycles), b = bpc100(bytes, cycles);
    probe_printf("%6u.%02u MB/s  %4u.%02u B/cyc",
                 m / 100u, m % 100u, b / 100u, b % 100u);
}

/* 접근 1회당 사이클 x100 */
static uint32_t cpa100(uint32_t cycles, uint32_t accesses)
{
    if (!accesses) return 0;
    return (cycles * 100u + accesses / 2u) / accesses;
}

/* ------------------------------------------------------------------------- */
/* D-cache 상태 제어                                                          */
/* ------------------------------------------------------------------------- */
#define CM_OFF 0
#define CM_WB  1
#define CM_WT  2

static const char *cm_name(int m)
{
    return (m == CM_OFF) ? "OFF" : (m == CM_WB) ? "WB" : "WT";
}

static void dcache_set(int on)
{
    int cur = (SCB->CCR & SCB_CCR_DC_Msk) ? 1 : 0;
    /* CMSIS의 SCB_EnableDCache()는 켜기 전에 '클린 없이' 무효화한다. 이미 켜져
     * 있는 상태에서 다시 부르면 더티 라인(로그 버퍼 포함)이 사라진다. */
    if (on && !cur) probe_dcache_enable();
    else if (!on && cur) probe_dcache_disable();
}

static void cache_mode(int m)
{
    dcache_set(0);                       /* 항상 먼저 끈다 (clean+invalidate 동반) */
    if (m == CM_WT) SCB->CACR |= SCB_CACR_FORCEWT_Msk;
    else            SCB->CACR &= ~SCB_CACR_FORCEWT_Msk;
    __DSB(); __ISB();
    if (m != CM_OFF) dcache_set(1);
}

/* 라벨이 실제 조건과 맞는지 증명하기 위해 함께 찍는다 */
static uint32_t dc_bit(void)  { return (SCB->CCR  & SCB_CCR_DC_Msk)        ? 1u : 0u; }
static uint32_t fwt_bit(void) { return (SCB->CACR & SCB_CACR_FORCEWT_Msk)  ? 1u : 0u; }

/* ------------------------------------------------------------------------- */
/* 메모리 속성 / MPU                                                          */
/* ------------------------------------------------------------------------- */
/*
 * MPU를 설정하지 않으면 ARMv7-M 기본 메모리맵(ARMv7-M ARM Table B3-1)이 적용된다.
 * 이 보드에서 중요한 결과는 0xC0000000(SDRAM)이 Device 라는 점이다. Device 메모리는
 *   - CCR.DC 값과 무관하게 D-cache에 절대 캐싱되지 않고,
 *   - CCR.UNALIGN_TRP 값과 무관하게 비정렬 접근이 항상 UsageFault다.
 * 따라서 기본맵 상태로는 SDRAM의 "캐시 효과"도 "정렬 페널티"도 측정이 성립하지 않는다.
 */
static const char *default_attr(uint32_t a)
{
    if (a < 0x20000000u) return "Normal WT (Code)";
    if (a < 0x40000000u) return "Normal WBWA (SRAM)";
    if (a < 0x60000000u) return "Device (Peripheral)";
    if (a < 0x80000000u) return "Normal WBWA (RAM)";
    if (a < 0xA0000000u) return "Normal WT (RAM)";
    if (a < 0xC0000000u) return "Device, shareable";
    if (a < 0xE0000000u) return "Device, non-shareable";
    return "Strongly-ordered (PPB)";
}

/*
 * 리전 0 하나로 SDRAM 8MB를 Normal / non-shareable / write-back write-allocate로
 * 재정의한다. PRIVDEFENA=1이라 나머지 주소는 기본맵 그대로다.
 * 속성이 바뀌는 시점 전후로 D-cache를 clean+invalidate 해야 한다.
 *   - Device→Normal: 이전 상태의 라인이 남아 있으면 안 된다.
 *   - Normal→Device: SDRAM 더티 라인을 내리지 않으면 그 데이터가 유실된다.
 */
#define MPU_SDRAM_REGION 0u
static int s_mpu_sdram_normal;

static void mpu_sdram_set(int on)
{
    if (on == s_mpu_sdram_normal) return;
    __DMB();
    SCB_CleanInvalidateDCache();
    ARM_MPU_Disable();
    if (on) {
        /* XN=1, AP=full, TEX=1 C=1 B=1 → Normal WBWA, S=0 → non-shareable */
        ARM_MPU_SetRegionEx(MPU_SDRAM_REGION,
                            ARM_MPU_RBAR(MPU_SDRAM_REGION, SDRAM_ADDR),
                            ARM_MPU_RASR(1u, ARM_MPU_AP_FULL, 1u, 0u, 1u, 1u, 0u,
                                         ARM_MPU_REGION_SIZE_8MB));
        ARM_MPU_Enable(MPU_CTRL_PRIVDEFENA_Msk);
    } else {
        ARM_MPU_ClrRegion(MPU_SDRAM_REGION);
    }
    __DSB();
    __ISB();
    s_mpu_sdram_normal = on;
}

/* 현재 SDRAM에 실제로 적용 중인 속성 (라벨 증명용) */
static const char *sdram_attr_now(void)
{
    return s_mpu_sdram_normal ? "MPU r0: Normal WBWA, non-shareable"
                              : "default map: Device, non-shareable";
}

/* ------------------------------------------------------------------------- */
/* 1. 영역 실제 크기 실측                                                     */
/* ------------------------------------------------------------------------- */
static uint32_t s_meas_size[R_COUNT];
static const char *s_meas_note[R_COUNT];

static const char *rc_name(int rc)
{
    switch (rc) {
    case 0: return "OK";
    case 1: return "WRAP";
    case 2: return "DEAD";
    default: return "FAULT";
    }
}

/*
 * base(영역 내부의 안전한 시작 주소)에 표식을 두고, 2의 거듭제곱 오프셋마다
 * 시험한다. 첫 실패 오프셋을 찾으면 1KB 단위 이분 탐색으로 경계를 좁힌다.
 * 반환값은 영역 시작(rbase) 기준 실측 크기(바이트).
 */
static uint32_t probe_region_extent(uint32_t rbase, uint32_t base, int writable,
                                    uint32_t max_off, int *out_rc, uint32_t *out_badoff)
{
    volatile uint32_t *b = (volatile uint32_t *)(uintptr_t)base;
    uint32_t orig = 0, off, lo = 0, hi = 0;
    int rc = 0, ok;

    *out_rc = 0;
    *out_badoff = 0;

    if (!guarded_read(base, &orig)) { *out_rc = 3; return 0; }
    if (writable && !guarded_write_check(base, MARK_A)) { *out_rc = 2; return 0; }

    for (off = 1024u; off <= max_off; off <<= 1) {
        uint32_t a = base + off;
        int cls = addr_class(a);
        if (cls == 0) { rc = 0; break; }            /* 더 볼 수 없다 → 여기서 중단 */
        if (cls == 2) { lo = off; continue; }        /* 실재가 확실한 구간 */
        rc = cell_test(b, a, writable, 0xF0F00000u ^ off);
        if (rc != 0) break;
        lo = off;
    }
    hi = off;

    if (rc != 0 && hi > lo) {
        /* 1KB 단위 이분 탐색으로 경계 확정 */
        uint32_t l = lo, h = hi;
        while (h - l > 1024u) {
            uint32_t mid = l + ((h - l) >> 1);
            uint32_t a;
            mid &= ~1023u;
            if (mid <= l) break;
            a = base + mid;
            if (addr_class(a) != 1) { l = mid; continue; }
            ok = (cell_test(b, a, writable, 0xF0F00000u ^ mid) == 0);
            if (ok) l = mid; else h = mid;
        }
        lo = l;
        *out_badoff = h;
    }

    if (writable) guarded_store(base, orig);

    *out_rc = rc;
    if (rc == 0) *out_badoff = lo;      /* 경계 미검출: 여기까지는 확인했다 */
    /* rc==1(되감김)이면 되감김 주기 자체가 크기. 그 외에는 base 오프셋 + 유효 범위 */
    if (rc == 1) return *out_badoff ? *out_badoff : hi;
    return (base - rbase) + lo + 4u;
}

static void round_kb(uint32_t bytes, uint32_t *kb)
{
    *kb = (bytes + 1023u) / 1024u;
}

static void run_size_probe(void)
{
    struct { int idx; uint32_t base; uint32_t maxoff; } tab[] = {
        { R_ITCM,  (uint32_t)(uintptr_t)s_itcm_buf,  32u * 1024u * 1024u },
        { R_DTCM,  (uint32_t)(uintptr_t)s_dtcm_buf,  32u * 1024u * 1024u },
        { R_SRAM1, (uint32_t)(uintptr_t)s_sram1_buf, 32u * 1024u * 1024u },
        { R_SRAM2, (uint32_t)(uintptr_t)s_sram2_buf, 32u * 1024u * 1024u },
        { R_BKP,   BKP_ADDR,                          8u * 1024u },
        { R_SDRAM, SDRAM_ADDR,                       32u * 1024u * 1024u },
        { R_FAXI,  FAXI_ADDR,                        32u * 1024u * 1024u },
        { R_FITCM, FITCM_ADDR,                       32u * 1024u * 1024u },
        { R_QSPI,  QSPI_ADDR,                        32u * 1024u * 1024u },
    };
    unsigned i;

    probe_section("1. 메모리 영역 실제 크기 실측 (aliasing / 폴트 경계 탐색)");
    probe_puts("방법: 기준 주소에 표식을 쓰고 2^k 오프셋마다 다른 값을 써 본 뒤 기준을 다시 읽는다.\n"
               "      기준값이 바뀌면 되감김(WRAP), 쓴 값이 유지되지 않으면 없는 메모리(DEAD),\n"
               "      버스 폴트면 FAULT. 첫 실패 지점은 1KB 단위 이분 탐색으로 좁혔다.\n"
               "      읽기 전용 영역(FLASH/QSPI)은 쓰기 시험이 불가하므로 '읽어서 폴트가 나는\n"
               "      첫 주소'만으로 상한을 잡는다 (되감김은 이 방법으로 구분되지 않는다).\n"
               "      D-cache는 이 시험 동안 OFF다 (캐시가 되감김을 가려 버린다).\n\n");

    cache_mode(CM_OFF);

    probe_puts("region       base        DS크기     실측크기   판정   경계에서\n");
    probe_puts("---------------------------------------------------------------\n");

    for (i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
        int idx = tab[i].idx;
        int rc = 0;
        uint32_t badoff = 0, sz, kb, dskb;

        sz = probe_region_extent(s_regions[idx].base, tab[i].base,
                                 s_regions[idx].writable, tab[i].maxoff, &rc, &badoff);
        s_meas_size[idx] = sz;
        s_meas_note[idx] = rc_name(rc);
        round_kb(sz, &kb);
        dskb = s_regions[idx].size / 1024u;

        pr_pad(s_regions[idx].name, 13);
        probe_printf("0x%08X  %5u KB  %5u KB  ", tab[i].base, dskb, kb);
        pr_pad(rc_name(rc), 6);
        if (rc == 0) probe_printf("(%u KB 까지 시험, 경계 미검출)", badoff / 1024u);
        else         probe_printf("+0x%08X 에서 %s", badoff, rc_name(rc));
        probe_puts(kb == dskb ? "  [일치]\n" : "  [불일치]\n");

        s_regions[idx].available = (sz > 0) ? 1 : 0;
    }

    /* QSPI 경계 탐색에서 FSIZE 밖을 찔렀다면 QUADSPI에 에러 플래그가 남는다.
     * 지우지 않으면 이후 메모리맵 읽기가 계속 실패할 수 있다. */
    QUADSPI->FCR = QUADSPI_FCR_CTEF | QUADSPI_FCR_CTCF | QUADSPI_FCR_CSMF | QUADSPI_FCR_CTOF;
    __DSB();
    {
        uint32_t v = 0;
        s_regions[R_QSPI].available = guarded_read(QSPI_ADDR, &v);
        s_qspi_ok = s_regions[R_QSPI].available;
    }

    probe_puts("\n주의: DTCM(0x20000000+64K) - SRAM1(0x20010000) - SRAM2(0x2004C000)은 주소가 연속이라\n"
               "      쓰기 시험만으로는 서로의 경계가 드러나지 않는다 (되감김이 없다). 각 영역의 실측값은\n"
               "      '그 주소에서 위로 얼마나 더 실재하는 메모리가 이어지는가'로 읽어야 하며,\n"
               "      물리적 경계는 아래 2번의 접근 지연 스캔이 갈라 준다.\n");
}

/* ------------------------------------------------------------------------- */
/* 2. 접근 지연 기반 경계 스캔 (연속 주소 영역을 물리적으로 갈라낸다)          */
/* ------------------------------------------------------------------------- */
#define SCAN_MAX 96
static uint16_t s_scan[SCAN_MAX];

static void run_boundary_scan(const char *title, uint32_t start, uint32_t end, uint32_t step)
{
    uint32_t a, n = 0, i;
    uint32_t accesses = 32u;

    probe_printf("\n[%s] %u KB 간격 접근 지연 스캔 (D-cache OFF, 32B 스트라이드 읽기)\n",
                 title, step / 1024u);

    for (a = start; a < end && n < SCAN_MAX; a += step, n++) {
        uint32_t v = 0, c, span = accesses * 32u;   /* 스캔이 실제로 읽는 구간 */
        int ok = 0;
        /* 첫 워드만 확인하고 나머지 1KB를 가드 없이 읽으면, 구간 안에 접근 불가
         * 경계가 있을 때 펌웨어가 멈춘다. 구간의 첫/마지막 워드를 모두 확인하고
         * 측정 자체도 가드 안에서 돌린다. */
        if (!guarded_read(a, &v) ||
            !guarded_read(a + span - 4u, &v)) { s_scan[n] = 0xFFFFu; continue; }
        c = run_min_guarded(k_stride, 0, (const void *)(uintptr_t)a,
                            accesses, 32u, 3, MODE_WARM, &ok);
        s_scan[n] = ok ? (uint16_t)cpa100(c, accesses) : 0xFFFFu;
    }

    /* 값이 비슷한 구간끼리 묶어서 출력한다 (한 줄에 하나의 평탄 구간) */
    i = 0;
    while (i < n) {
        uint32_t j = i + 1, base = s_scan[i];
        while (j < n) {
            uint32_t v = s_scan[j];
            uint32_t d = (v > base) ? (v - base) : (base - v);
            if (base == 0xFFFFu || v == 0xFFFFu) { if (v != base) break; }
            else if (d * 4u > base) break;          /* 25% 이상 차이나면 다른 구간 */
            j++;
        }
        probe_printf("  0x%08X - 0x%08X : ", start + i * step, start + j * step - 1u);
        if (base == 0xFFFFu) probe_puts("접근 불가 (버스 폴트)\n");
        else probe_printf("%3u.%02u cyc/접근\n", base / 100u, base % 100u);
        i = j;
    }
}

/* ------------------------------------------------------------------------- */
/* 3. 대역폭                                                                  */
/* ------------------------------------------------------------------------- */
typedef struct {
    const char *name;
    void       *buf;        /* 쓰기 가능한 워킹셋 (읽기 전용이면 소스로만 사용) */
    uint32_t    words;
    int         writable;
    const char *attr;       /* 이 주소에 실제로 적용 중인 메모리 속성 = 캐시 대상 여부 */
    int         avail;
} bwtgt_t;

static void bw_block(const bwtgt_t *t, int mode)
{
    uint32_t bytes = t->words * 4u;
    uint32_t c;

    if (!t->avail) {
        probe_printf("[%s] 사용 불가 - 건너뜀\n", t->name);
        return;
    }
    probe_printf("[%s] 워킹셋 %u KB  속성: %s\n", t->name, bytes / 1024u, t->attr);

    c = run_min(k_read32, 0, t->buf, t->words, 0, 3, mode);
    probe_puts("  read32   "); pr_bw(bytes, c); probe_puts("\n");
    c = run_min(k_read64, 0, t->buf, t->words, 0, 3, mode);
    probe_puts("  read64   "); pr_bw(bytes, c); probe_puts("\n");
    c = run_min(k_readldm, 0, t->buf, t->words, 0, 3, mode);
    probe_puts("  readLDM  "); pr_bw(bytes, c); probe_puts("\n");

    if (t->writable) {
        c = run_min(k_write32, t->buf, 0, t->words, 0, 3, mode);
        probe_puts("  write32  "); pr_bw(bytes, c); probe_puts("\n");
        c = run_min(k_write64, t->buf, 0, t->words, 0, 3, mode);
        probe_puts("  write64  "); pr_bw(bytes, c); probe_puts("\n");
        c = run_min(k_writestm, t->buf, 0, t->words, 0, 3, mode);
        probe_puts("  writeSTM "); pr_bw(bytes, c); probe_puts("\n");
        /* 같은 영역 안에서의 복사: 앞 절반 → 뒤 절반 */
        c = run_min(k_copyldm, (uint8_t *)t->buf + bytes / 2u, t->buf,
                    t->words / 2u, 0, 3, mode);
        probe_puts("  copyLDM  "); pr_bw(bytes, c);   /* 읽기+쓰기 합쳐 bytes 만큼 이동 */
        probe_puts("  (읽기+쓰기 합산)\n");
    }
}

/* ------------------------------------------------------------------------- */
/* 포인터 체이싱용 순환 리스트 (셔플: 프리페치/스트라이드 예측 차단)           */
/* ------------------------------------------------------------------------- */
static uint32_t s_rng = 0x12345678u;
static uint32_t rnd32(void)
{
    s_rng = s_rng * 1664525u + 1013904223u;
    return s_rng;
}

/* 32바이트(캐시라인) 간격 노드. 워드0=next, 워드1=셔플용 인덱스 */
static void *build_chase(void *buf, uint32_t bytes)
{
    volatile uint32_t *b = (volatile uint32_t *)buf;
    uint32_t n = bytes / 32u, i, k;
    if (n < 2u) return 0;

    for (i = 0; i < n; i++) b[i * 8u + 1u] = i;
    for (i = n - 1u; i > 0u; i--) {              /* Sattolo: 단일 순환 보장 */
        uint32_t j = rnd32() % i;
        uint32_t tmp = b[i * 8u + 1u];
        b[i * 8u + 1u] = b[j * 8u + 1u];
        b[j * 8u + 1u] = tmp;
    }
    for (k = 0; k < n; k++) {
        uint32_t a = b[k * 8u + 1u];
        uint32_t c = b[((k + 1u) % n) * 8u + 1u];
        b[a * 8u] = (uint32_t)(uintptr_t)(&((uint32_t *)buf)[c * 8u]);
    }
    return (void *)&((uint32_t *)buf)[b[1u] * 8u];
}

/* ------------------------------------------------------------------------- */
/* probe_run_memory                                                           */
/* ------------------------------------------------------------------------- */
static void detect_external(void)
{
    uint32_t v = 0;
    volatile uint32_t *p;

    /* --- BKPSRAM: 클럭 + 백업 도메인 쓰기 보호 해제 --- */
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    (void)RCC->APB1ENR;
    PWR->CR1 |= PWR_CR1_DBP;
    RCC->AHB1ENR |= RCC_AHB1ENR_BKPSRAMEN;
    (void)RCC->AHB1ENR;
    __DSB();

    s_bkp_ok = 0;
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) == 0) {
        p = (volatile uint32_t *)BKP_ADDR;
        uint32_t o = *p;
        *p = 0xBEEF0001u;
        __DSB();
        if (*p == 0xBEEF0001u) s_bkp_ok = 1;
        *p = o;
        __DSB();
    }
    s_guard_active = 0;

    /* --- SDRAM: 3개 지점 쓰기/되읽기 --- */
    s_sdram_ok = 0;
    s_guard_active = 1;
    if (pm_try_begin(s_ctx) == 0) {
        int good = 1;
        uint32_t offs[3] = { 0u, 1024u * 1024u, 7u * 1024u * 1024u };
        for (int i = 0; i < 3; i++) {
            p = (volatile uint32_t *)(uintptr_t)(SDRAM_ADDR + offs[i]);
            *p = 0xC0DE0000u + (uint32_t)i;
            __DSB();
        }
        for (int i = 0; i < 3; i++) {
            p = (volatile uint32_t *)(uintptr_t)(SDRAM_ADDR + offs[i]);
            if (*p != 0xC0DE0000u + (uint32_t)i) good = 0;
        }
        s_sdram_ok = good;
    }
    s_guard_active = 0;

    /* --- QSPI: 메모리맵 모드가 아니면 읽기 자체가 버스 폴트 --- */
    s_qspi_ok = guarded_read(QSPI_ADDR, &v) &&
                guarded_read(QSPI_ADDR + 0x100000u, &v);

    s_regions[R_BKP].available   = s_bkp_ok;
    s_regions[R_SDRAM].available = s_sdram_ok;
    s_regions[R_QSPI].available  = s_qspi_ok;
}

void probe_run_memory(void)
{
    int cm_saved = (SCB->CCR & SCB_CCR_DC_Msk)
                 ? ((SCB->CACR & SCB_CACR_FORCEWT_Msk) ? CM_WT : CM_WB) : CM_OFF;
    uint32_t ccr_saved = SCB->CCR;
    int      ic_saved  = (SCB->CCR & SCB_CCR_IC_Msk) ? 1 : 0;
    uint32_t mpu_type_at_entry = MPU->TYPE;
    uint32_t mpu_ctrl_at_entry = MPU->CTRL;
    uint32_t c, c2, c3;

    guard_install();

    /* 측정 커널(k_read32 등)은 .text = 플래시 AXIM에 있다. I-cache가 꺼져 있으면
     * 6 wait state 플래시에서 루프 본체를 매 반복 다시 페치하게 되어, 명령 페치
     * 비용이 데이터 대역폭 수치에 그대로 섞인다(특히 TCM/SRAM이 크게 저평가된다).
     * 그래서 이 모듈이 도는 동안에는 I-cache를 켜고, 끝나면 원래 상태로 되돌린다.
     * 오버헤드 측정도 반드시 이 조건이 잡힌 뒤에 해야 한다. */
    if (!ic_saved) probe_icache_enable();
    measure_overhead();

    probe_section("메모리 서브시스템 실측");
    probe_printf("SYSCLK %u Hz 기준. 반복 3회의 최솟값을 쓴다 (최솟값이 노이즈가 가장 적다).\n",
                 g_sysclk_hz);
    probe_printf("측정 하네스 오버헤드 %u cyc. run_min()과 완전히 같은 경로(인자 셋업 + 함수 포인터\n"
                 "  간접 호출 blx + 커널 프롤로그/에필로그 + dsb/isb + CYCCNT 2회 읽기)로 쟀고,\n"
                 "  아래 3~5·7·8절의 모든 수치에서 뺐다. 6절은 ITCM 상주 계측 프레임을 쓰므로\n"
                 "  그 절에서 같은 방식으로 프레임 오버헤드를 따로 재서 뺐다 (그 절 머리말 참조).\n",
                 s_ovh);
    probe_printf("I-cache: 진입 시 %s → 이 모듈이 도는 동안 강제로 ON (CCR.IC=%u), 종료 시 원복.\n",
                 ic_saved ? "ON" : "OFF",
                 (SCB->CCR & SCB_CCR_IC_Msk) ? 1u : 0u);
    probe_printf("측정 중 인터럽트는 PRIMASK로 차단한다. 폴트 복구용 벡터테이블을 0x%08X에 임시 설치했다.\n",
                 (uint32_t)(uintptr_t)s_vtab);
    probe_printf("버퍼 배치: ITCM 0x%08X(%u KB) DTCM 0x%08X(%u KB) SRAM1 0x%08X(%u KB) SRAM2 0x%08X(%u KB)\n",
                 (uint32_t)(uintptr_t)s_itcm_buf,  ITCM_WORDS  * 4u / 1024u,
                 (uint32_t)(uintptr_t)s_dtcm_buf,  DTCM_WORDS  * 4u / 1024u,
                 (uint32_t)(uintptr_t)s_sram1_buf, SRAM1_WORDS * 4u / 1024u,
                 (uint32_t)(uintptr_t)s_sram2_buf, SRAM2_WORDS * 4u / 1024u);

    /* ---- 측정 시작 시점의 MPU 상태와 각 주소의 기본 메모리맵 속성 ---- */
    probe_puts("\n[메모리 속성 / MPU 상태 - 측정 시작 시점]\n");
    probe_printf("  MPU_TYPE = 0x%08X  (DREGION = %u개 지원, IREGION = %u)\n",
                 mpu_type_at_entry,
                 (mpu_type_at_entry >> 8) & 0xFFu, (mpu_type_at_entry >> 16) & 0xFFu);
    probe_printf("  MPU_CTRL = 0x%08X  → MPU %s\n", mpu_ctrl_at_entry,
                 (mpu_ctrl_at_entry & MPU_CTRL_ENABLE_Msk)
                     ? "켜져 있음" : "꺼져 있음 → ARMv7-M 기본 메모리맵이 그대로 적용된다");
    probe_puts("  기본 메모리맵 속성 (ARMv7-M ARM Table B3-1):\n");
    {
        static const struct { const char *nm; uint32_t a; } am[] = {
            { "ITCM / FLASH_ITCM alias", 0x00000000u },
            { "FLASH_AXIM             ", 0x08000000u },
            { "DTCM / SRAM1 / SRAM2   ", 0x20000000u },
            { "BKPSRAM                ", 0x40024000u },
            { "QSPI memory-mapped     ", 0x90000000u },
            { "SDRAM                  ", 0xC0000000u },
        };
        unsigned i;
        for (i = 0; i < sizeof(am) / sizeof(am[0]); i++)
            probe_printf("    0x%08X  %s : %s\n", am[i].a, am[i].nm, default_attr(am[i].a));
    }
    probe_puts("  단, ITCM(0x00000000~0x00003FFF)과 DTCM(0x20000000~0x2000FFFF)은 TCM 인터페이스\n"
               "  직결이라 위 속성과 무관하게 애초에 D-cache 대상이 아니다.\n"
               "  → SDRAM(0xC0000000)은 기본맵에서 Device다. Device 메모리는 CCR.DC와 무관하게\n"
               "     D-cache에 캐싱되지 않고, CCR.UNALIGN_TRP와 무관하게 비정렬 접근이 항상\n"
               "     UsageFault다. 그래서 4·5·7·8절은 SDRAM에 한해 MPU 리전 0으로 8MB를\n"
               "     Normal WBWA(non-shareable)로 재정의한 조건을 함께 재서 나란히 비교한다.\n"
               "     측정이 끝나면 MPU를 다시 끈다 (마지막 절에서 확인 출력).\n");

    detect_external();
    probe_printf("외부/백업 메모리 실측 판정: SDRAM=%s  QSPI=%s  BKPSRAM=%s\n",
                 s_sdram_ok ? "사용 가능" : "사용 불가",
                 s_qspi_ok ? "사용 가능" : "사용 불가",
                 s_bkp_ok ? "사용 가능" : "사용 불가");

    /* ---------------- 1. 크기 ---------------- */
    run_size_probe();

    /* ---------------- 2. 경계 스캔 ---------------- */
    probe_section("2. 접근 지연으로 본 물리 경계 (주소가 연속인 영역 구분)");
    probe_puts("같은 주소 공간이라도 물리적으로 다른 메모리면 접근 비용이 다르다.\n"
               "아래는 D-cache OFF 상태에서 32B 간격 읽기의 접근당 사이클이다 (루프 오버헤드 포함).\n");
    cache_mode(CM_OFF);
    run_boundary_scan("0x20000000 대역", 0x20000000u, 0x20054000u, 4096u);
    run_boundary_scan("0x00000000 대역", 0x00000000u, 0x00006000u, 1024u);
    c = cpa100(run_min(k_stride_empty, 0, s_sram1_buf, 256u, 32u, 3, MODE_WARM), 256u);
    probe_printf("\n참고: 스트라이드 루프 자체의 오버헤드는 %u.%02u cyc/회 다 (위 값에 포함되어 있다).\n",
                 c / 100u, c % 100u);

    /* ---------------- 3. 대역폭 ---------------- */
    {
        bwtgt_t tg[] = {
            { "ITCM",       s_itcm_buf,        ITCM_WORDS,        1,
              "TCM (코어 직결, 캐시 대상 아님)",                              1 },
            { "DTCM",       s_dtcm_buf,        DTCM_WORDS,        1,
              "TCM (코어 직결, 캐시 대상 아님)",                              1 },
            { "SRAM1",      s_sram1_buf,       SRAM1_WORDS,       1,
              "Normal WBWA (캐시 대상)",                                      1 },
            { "SRAM2",      s_sram2_buf,       SRAM2_WORDS,       1,
              "Normal WBWA (캐시 대상)",                                      1 },
            { "BKPSRAM",    (void *)BKP_ADDR,  1024u,             1,
              "Device (기본맵 Peripheral 영역, 캐시 대상 아님)",              s_bkp_ok },
            { "SDRAM",      (void *)SDRAM_ADDR, EXT_WS_BYTES / 4u, 1,
              "Device (기본맵, 캐시 대상 아님 - 4/7절에서 MPU로 바꿔 재측정)", s_sdram_ok },
            { "FLASH_AXIM", (void *)FAXI_ADDR, EXT_WS_BYTES / 4u, 0,
              "Normal WT (캐시 대상)",                                        1 },
            { "QSPI",       (void *)QSPI_ADDR, EXT_WS_BYTES / 4u, 0,
              "Normal WT (캐시 대상)",                                        s_qspi_ok },
        };
        unsigned i;

        mpu_sdram_set(0);           /* 3절은 기본 메모리맵 조건에서 잰다 */
        probe_section("3. 순차 대역폭 (D-cache ON / write-back, 정상 상태, 기본 메모리맵)");
        probe_puts("read32 = LDR x8 언롤, read64 = LDRD x4, readLDM = LDM 8레지스터 x2,\n"
                   "write32 = STR x8, write64 = STRD x4, writeSTM = STM 8레지스터 x2,\n"
                   "copyLDM = LDM/STM 8레지스터 (memcpy 상당). 전부 인라인 어셈블리다.\n"
                   "각 행의 '속성'은 그 주소에 실제로 적용되는 메모리 속성이다. 캐시 대상(Normal)\n"
                   "영역은 워킹셋을 D-cache(4KB)보다 크게 잡았고, TCM/Device 영역은 캐시와 무관하므로\n"
                   "영역 크기에 맞췄다 (ITCM/BKPSRAM은 4KB로 D-cache와 같은 크기이지만, 이 둘은\n"
                   "애초에 캐싱되지 않으므로 워킹셋 크기가 결과에 영향을 주지 않는다).\n"
                   "복사(copyLDM)의 대역폭 규약은 '이동 바이트 = 읽기 + 쓰기 합산'으로 통일했다.\n\n");
        cache_mode(CM_WB);
        probe_printf("현재 캐시 상태 증명: CCR.DC=%u, CACR.FORCEWT=%u, CCR.IC=%u\n\n",
                     dc_bit(), fwt_bit(), (SCB->CCR & SCB_CCR_IC_Msk) ? 1u : 0u);
        for (i = 0; i < sizeof(tg) / sizeof(tg[0]); i++) bw_block(&tg[i], MODE_WARM);

        /* 영역 간 복사 - 규약은 위와 같다 (읽기 + 쓰기 합산) */
        probe_puts("\n[영역 간 복사] LDM/STM 8레지스터. 아래 모든 행의 대역폭은 '읽기+쓰기 합산' 규약이다\n"
                   "  (payload 32 KB를 옮기면 이동 바이트는 64 KB로 센다 - 위 영역 내 복사와 같은 규약).\n");
        {
            uint32_t words = 8u * 1024u;               /* payload 32 KB */
            uint32_t moved = words * 4u * 2u;          /* 읽기 + 쓰기 = 64 KB */
            uint32_t dmoved = DTCM_WORDS * 4u * 2u;
            if (s_sdram_ok) {
                c = run_min(k_copyldm, s_sram1_buf, (void *)SDRAM_ADDR, words, 0, 3, MODE_WARM);
                probe_puts("  SDRAM -> SRAM1  "); pr_bw(moved, c); probe_puts("  (읽기+쓰기 합산)\n");
                c = run_min(k_copyldm, (void *)SDRAM_ADDR, s_sram1_buf, words, 0, 3, MODE_WARM);
                probe_puts("  SRAM1 -> SDRAM  "); pr_bw(moved, c); probe_puts("  (읽기+쓰기 합산)\n");
            }
            c = run_min(k_copyldm, s_sram1_buf, (void *)FAXI_ADDR, words, 0, 3, MODE_WARM);
            probe_puts("  FLASH -> SRAM1  "); pr_bw(moved, c); probe_puts("  (읽기+쓰기 합산)\n");
            c = run_min(k_copyldm, s_dtcm_buf, s_sram1_buf, DTCM_WORDS, 0, 3, MODE_WARM);
            probe_puts("  SRAM1 -> DTCM   "); pr_bw(dmoved, c); probe_puts("  (읽기+쓰기 합산)\n");
            c = run_min(k_copyldm, s_sram1_buf, s_dtcm_buf, DTCM_WORDS, 0, 3, MODE_WARM);
            probe_puts("  DTCM  -> SRAM1  "); pr_bw(dmoved, c); probe_puts("  (읽기+쓰기 합산)\n");
            c = run_min(k_copy32, s_sram1_buf, s_dtcm_buf, DTCM_WORDS, 0, 3, MODE_WARM);
            probe_puts("  DTCM  -> SRAM1  "); pr_bw(dmoved, c);
            probe_puts("  (읽기+쓰기 합산, LDR/STR 32비트판)\n");
        }
    }

    /* ---------------- 4. 캐시 상태별 비교 ---------------- */
    probe_section("4. D-cache 상태별 비교 (콜드 상태: 매 측정 전 clean+invalidate)");
    probe_puts("OFF = D-cache 끔, WB = write-back, WT = CACR.FORCEWT=1 강제 write-through.\n"
               "각 행의 [ ] 안이 그 측정 동안 실제로 적용된 메모리 속성이다. 캐시 대상이 아닌\n"
               "속성(TCM / Device)에서는 세 열이 같게 나오는 것이 정상이며, 그 차이 없음은\n"
               "'캐시 정책의 차이가 작다'는 뜻이 아니라 '애초에 캐싱되지 않는다'는 뜻이다.\n"
               "SDRAM은 기본맵(Device) 조건과 MPU로 Normal WBWA로 바꾼 조건을 둘 다 실었다.\n"
               "각 행 끝의 DC/FWT는 세 열을 잴 때 실제로 읽은 CCR.DC와 CACR.FORCEWT 비트다\n"
               "(라벨이 조건과 일치함을 증명한다).\n");
    probe_printf("워킹셋: SRAM1 %u KB, SRAM2 %u KB, FLASH_AXIM %u KB, SDRAM %u KB\n\n",
                 SRAM1_WORDS * 4u / 1024u, SRAM2_WORDS * 4u / 1024u,
                 EXT_WS_BYTES / 1024u, EXT_WS_BYTES / 1024u);
    probe_puts("대상 [속성]               연산     OFF                        WB                         WT\n");
    probe_puts("----------------------------------------------------------------------------------------------\n");
    {
        struct { const char *nm; void *buf; uint32_t words; int wr; int mpu; int avail; } cs[] = {
            { "SRAM1 [Normal WBWA]",      s_sram1_buf,        SRAM1_WORDS,       1, 0, 1 },
            { "SRAM2 [Normal WBWA]",      s_sram2_buf,        SRAM2_WORDS,       1, 0, 1 },
            { "FLASH_AXIM [Normal WT]",   (void *)FAXI_ADDR,  EXT_WS_BYTES / 4u, 0, 0, 1 },
            { "SDRAM [map:Device]",       (void *)SDRAM_ADDR, EXT_WS_BYTES / 4u, 1, 0, s_sdram_ok },
            { "SDRAM [MPU:Normal WBWA]",  (void *)SDRAM_ADDR, EXT_WS_BYTES / 4u, 1, 1, s_sdram_ok },
            { "ITCM [TCM]",               s_itcm_buf,         ITCM_WORDS,        1, 0, 1 },
            { "DTCM [TCM]",               s_dtcm_buf,         DTCM_WORDS,        1, 0, 1 },
        };
        unsigned i;
        for (i = 0; i < sizeof(cs) / sizeof(cs[0]); i++) {
            uint32_t bytes = cs[i].words * 4u;
            uint32_t d1, d2, d3, w1, w2, w3;
            if (!cs[i].avail) { probe_printf("%s: 사용 불가 - 건너뜀\n", cs[i].nm); continue; }

            mpu_sdram_set(cs[i].mpu);

            cache_mode(CM_OFF); d1 = dc_bit(); w1 = fwt_bit();
            c  = run_min(k_read32, 0, cs[i].buf, cs[i].words, 0, 3, MODE_COLD);
            cache_mode(CM_WB);  d2 = dc_bit(); w2 = fwt_bit();
            c2 = run_min(k_read32, 0, cs[i].buf, cs[i].words, 0, 3, MODE_COLD);
            cache_mode(CM_WT);  d3 = dc_bit(); w3 = fwt_bit();
            c3 = run_min(k_read32, 0, cs[i].buf, cs[i].words, 0, 3, MODE_COLD);
            pr_pad(cs[i].nm, 25); pr_pad("read32", 9);
            pr_bw(bytes, c);  probe_puts("  ");
            pr_bw(bytes, c2); probe_puts("  ");
            pr_bw(bytes, c3);
            probe_printf("   DC=%u%u%u FWT=%u%u%u\n", d1, d2, d3, w1, w2, w3);

            if (cs[i].wr) {
                cache_mode(CM_OFF); c  = run_min(k_write32, cs[i].buf, 0, cs[i].words, 0, 3, MODE_COLD);
                cache_mode(CM_WB);  c2 = run_min(k_write32, cs[i].buf, 0, cs[i].words, 0, 3, MODE_COLD);
                cache_mode(CM_WT);  c3 = run_min(k_write32, cs[i].buf, 0, cs[i].words, 0, 3, MODE_COLD);
                pr_pad(cs[i].nm, 25); pr_pad("write32", 9);
                pr_bw(bytes, c);  probe_puts("  ");
                pr_bw(bytes, c2); probe_puts("  ");
                pr_bw(bytes, c3); probe_puts("\n");

                cache_mode(CM_OFF); c  = run_min(k_copyldm, (uint8_t *)cs[i].buf + bytes / 2u, cs[i].buf, cs[i].words / 2u, 0, 3, MODE_COLD);
                cache_mode(CM_WB);  c2 = run_min(k_copyldm, (uint8_t *)cs[i].buf + bytes / 2u, cs[i].buf, cs[i].words / 2u, 0, 3, MODE_COLD);
                cache_mode(CM_WT);  c3 = run_min(k_copyldm, (uint8_t *)cs[i].buf + bytes / 2u, cs[i].buf, cs[i].words / 2u, 0, 3, MODE_COLD);
                pr_pad(cs[i].nm, 25); pr_pad("copyLDM", 9);
                pr_bw(bytes, c);  probe_puts("  ");
                pr_bw(bytes, c2); probe_puts("  ");
                pr_bw(bytes, c3); probe_puts("  (읽기+쓰기 합산)\n");
            }
        }
        mpu_sdram_set(0);
    }
    probe_puts("\n읽는 법:\n"
               "  - ITCM/DTCM은 코어 직결(TCM 인터페이스)이라 D-cache 대상이 아니다. 세 열이 같아야 정상이고,\n"
               "    같게 나왔다면 그 사실이 실측으로 확인된 것이다.\n"
               "  - 'SDRAM [map:Device]' 행의 세 열이 같은 이유는 캐시 정책 차이가 없어서가 아니라\n"
               "    Device 메모리라 CCR.DC=1이어도 D-cache에 한 줄도 올라가지 않기 때문이다.\n"
               "    (DC 열이 0/1/1로 찍혀 있는데도 값이 같다는 점이 바로 그 증거다.)\n"
               "  - 'SDRAM [MPU:Normal WBWA]' 행이 이 칩에서 SDRAM 캐시 효과의 진짜 값이다.\n"
               "    두 SDRAM 행의 차이가 곧 'MPU 설정 유무가 외부 SDRAM 성능에 미치는 영향'이다.\n");

    /* ---------------- 5. 캐시 크기/계층 ---------------- */
    probe_section("5. 캐시 크기·계층 실측 (워킹셋 스윕)");
    probe_puts("SRAM1에서 워킹셋을 2배씩 키우며 순차 읽기 대역폭을 잰다 (정상 상태, 워밍업 후).\n"
               "D-cache가 4KB라면 4KB를 넘는 순간 계단이 나타난다. OFF 열은 대조군이라 평탄해야 한다.\n\n");
    probe_puts("워킹셋      D-cache OFF                D-cache WB                 WB/OFF 배율\n");
    probe_puts("--------------------------------------------------------------------------------\n");
    {
        uint32_t kb;
        for (kb = 1u; kb <= SRAM1_WORDS * 4u / 1024u; kb <<= 1) {
            uint32_t words = kb * 256u;
            uint32_t bytes = words * 4u;
            uint32_t r;
            cache_mode(CM_OFF); c  = run_min(k_read32, 0, s_sram1_buf, words, 0, 3, MODE_WARM);
            cache_mode(CM_WB);  c2 = run_min(k_read32, 0, s_sram1_buf, words, 0, 3, MODE_WARM);
            r = c2 ? (c * 100u / c2) : 0u;
            probe_printf("%4u KB    ", kb);
            pr_bw(bytes, c);  probe_puts("  ");
            pr_bw(bytes, c2); probe_printf("   x%u.%02u\n", r / 100u, r % 100u);
        }
    }

    /* 포인터 체이싱 */
    probe_puts("\n[포인터 체이싱 레이턴시] 32B(캐시라인) 간격 노드를 Sattolo 셔플로 단일 순환 연결하고\n"
               "p = *(void**)p 를 반복한다. 순서를 섞었으므로 스트라이드 예측/프리페치가 개입하지 못한다.\n"
               "값은 접근 1회당 사이클 (로드 의존 사슬의 실효 레이턴시)이다.\n"
               "SDRAM 두 열은 MPU 리전 0으로 SDRAM을 Normal WBWA로 재정의한 상태에서 잰 값이다.\n"
               "(기본맵 Device 상태에서는 SDRAM이 캐싱되지 않으므로 OFF/WB 두 열이 같은 값이 되고,\n"
               " 그 조건의 대비는 4절과 7절의 SDRAM 두 블록에 실려 있다.)\n\n");
    probe_puts("워킹셋      SRAM1 OFF    SRAM1 WB     SDRAM OFF    SDRAM WB   (SDRAM = MPU:Normal WBWA)\n");
    probe_puts("--------------------------------------------------------------------------------------\n");
    {
        uint32_t kb;
        mpu_sdram_set(1);
        for (kb = 1u; kb <= 512u; kb <<= 1) {
            uint32_t bytes = kb * 1024u;
            uint32_t acc = 4096u;
            uint32_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
            void *head;

            if (bytes <= SRAM1_WORDS * 4u) {
                cache_mode(CM_WB);
                head = build_chase(s_sram1_buf, bytes);
                if (head) {
                    cache_mode(CM_OFF);
                    v1 = cpa100(run_min(k_chase, 0, head, acc, 0, 3, MODE_WARM), acc);
                    cache_mode(CM_WB);
                    v2 = cpa100(run_min(k_chase, 0, head, acc, 0, 3, MODE_WARM), acc);
                }
            }
            if (s_sdram_ok) {
                cache_mode(CM_WB);
                head = build_chase((void *)SDRAM_ADDR, bytes);
                if (head) {
                    cache_mode(CM_OFF);
                    v3 = cpa100(run_min(k_chase, 0, head, acc, 0, 3, MODE_WARM), acc);
                    cache_mode(CM_WB);
                    v4 = cpa100(run_min(k_chase, 0, head, acc, 0, 3, MODE_WARM), acc);
                }
            }
            probe_printf("%4u KB    ", kb);
            if (v1) probe_printf("%4u.%02u     %4u.%02u     ", v1 / 100u, v1 % 100u, v2 / 100u, v2 % 100u);
            else    probe_puts("     -            -        ");
            if (v3) probe_printf("%4u.%02u     %4u.%02u\n", v3 / 100u, v3 % 100u, v4 / 100u, v4 % 100u);
            else    probe_puts("     -            -\n");
        }
    }

    /* 캐시라인 크기 */
    probe_puts("\n[캐시라인 크기 실측] 스트라이드를 키우며 접근당 사이클을 잰다 (D-cache WB, 워킹셋 32KB).\n"
               "스트라이드가 라인 크기에 도달하면 접근마다 새 라인을 끌어오므로 값이 평탄해진다.\n"
               "SRAM1은 Normal WBWA, SDRAM은 MPU 리전 0으로 Normal WBWA로 재정의한 상태다.\n"
               "SDRAM 열에는 캐시라인뿐 아니라 SDRAM 행(page)/버스트 특성도 함께 섞여 나타난다.\n\n");
    probe_puts("stride     SRAM1 [Normal WBWA]     SDRAM [MPU:Normal WBWA]\n");
    probe_puts("------------------------------------------------------------\n");
    {
        uint32_t st;
        mpu_sdram_set(1);
        cache_mode(CM_WB);
        probe_printf("(측정 조건 증명: CCR.DC=%u, CACR.FORCEWT=%u, SDRAM 속성=%s)\n",
                     dc_bit(), fwt_bit(), sdram_attr_now());
        for (st = 4u; st <= 128u; st <<= 1) {
            uint32_t acc = (32u * 1024u) / st;
            uint32_t a, b = 0;
            if (acc > 2048u) acc = 2048u;
            a = cpa100(run_min(k_stride, 0, s_sram1_buf, acc, st, 3, MODE_COLD), acc);
            if (s_sdram_ok) b = cpa100(run_min(k_stride, 0, (void *)SDRAM_ADDR, acc, st, 3, MODE_COLD), acc);
            probe_printf("%4u B     %4u.%02u cyc/접근        ", st, a / 100u, a % 100u);
            if (b) probe_printf("%4u.%02u cyc/접근\n", b / 100u, b % 100u);
            else   probe_puts("   -\n");
        }
        mpu_sdram_set(0);           /* 다음 절을 위해 기본맵으로 되돌린다 */
    }

    /* ---------------- 6. 플래시 접근 특성 ---------------- */
    probe_section("6. 플래시 접근 특성 (AXIM 별칭 vs ITCM 별칭, ART ON/OFF)");
    probe_puts("측정 루프는 .itcm_code 섹션에 올려 ITCM에서 실행한다. 그래야 ART/캐시를 꺼도\n"
               "코드 페치 경로가 흔들리지 않고 데이터 접근만 순수하게 측정된다.\n");
    probe_printf("워킹셋 %u KB. FLASH_AXIM=0x%08X, FLASH_ITCM=0x%08X (같은 물리 플래시의 다른 경로)\n",
                 EXT_WS_BYTES / 1024u, FAXI_ADDR, FITCM_ADDR);
    {
        uint32_t fovh = 0xFFFFFFFFu;
        int j;
        for (j = 0; j < 8; j++) {                    /* 계측 프레임 자체의 비용 */
            uint32_t v = itcm_timed_ovh();
            if (v < fovh) fovh = v;
        }
        probe_printf("기준선: 이 표의 값에서는 ITCM 계측 프레임 오버헤드 %u cyc(dsb/isb + CYCCNT 2회 읽기)를\n"
                     "  뺐다. 위 3~5절의 run_min 기준선과 같은 방식이다. 다만 두 기준선 모두 루프 제어\n"
                     "  (포인터 증가 + subs + bne) 비용은 포함한 채로 남아 있으므로, 접근당 사이클을\n"
                     "  절대값으로 읽을 때는 2절 끝에 찍힌 루프 오버헤드를 함께 보아야 한다.\n\n",
                     fovh);
        s_flash_frame_ovh = fovh;
    }
    probe_puts("경로         D-cache  ART    순차읽기                              4KB 스트라이드\n");
    probe_puts("--------------------------------------------------------------------------------\n");
    {
        struct { const char *nm; uint32_t base; int cm; int art; } fl[] = {
            { "AXIM", FAXI_ADDR,  CM_OFF, 1 },
            { "AXIM", FAXI_ADDR,  CM_OFF, 0 },
            { "AXIM", FAXI_ADDR,  CM_WB,  1 },
            { "ITCM", FITCM_ADDR, CM_OFF, 1 },
            { "ITCM", FITCM_ADDR, CM_OFF, 0 },
            { "ITCM", FITCM_ADDR, CM_WB,  1 },
        };
        unsigned i;
        uint32_t words = EXT_WS_BYTES / 4u;
        for (i = 0; i < sizeof(fl) / sizeof(fl[0]); i++) {
            uint32_t best = 0xFFFFFFFFu, bs = 0xFFFFFFFFu, acc = 128u;
            int j;
            cache_mode(fl[i].cm);
            if (fl[i].art) { probe_art_enable(); } else { probe_art_disable(); }
            probe_art_reset();
            for (j = 0; j < 3; j++) {
                uint32_t v;
                SCB_CleanInvalidateDCache();
                probe_art_reset();
                v = itcm_timed_read32((const void *)(uintptr_t)fl[i].base, words);
                if (v < best) best = v;
                SCB_CleanInvalidateDCache();
                probe_art_reset();
                v = itcm_timed_stride((const void *)(uintptr_t)fl[i].base, acc, 4096u);
                if (v < bs) bs = v;
            }
            /* ART를 다시 켜 두어야 이후 출력 코드(플래시 상주)가 정상 속도로 돈다 */
            probe_art_enable();
            probe_art_reset();
            cache_mode(CM_WB);

            /* 위 3~5절과 기준선을 맞춘다 (계측 프레임 오버헤드 제거) */
            best = (best > s_flash_frame_ovh) ? (best - s_flash_frame_ovh) : 0u;
            bs   = (bs   > s_flash_frame_ovh) ? (bs   - s_flash_frame_ovh) : 0u;

            pr_pad(fl[i].nm, 13);
            pr_pad(fl[i].cm == CM_OFF ? "OFF" : cm_name(fl[i].cm), 9);
            pr_pad(fl[i].art ? "ON" : "OFF", 7);
            pr_bw(words * 4u, best);
            probe_printf("   %4u.%02u cyc/접근\n",
                         cpa100(bs, acc) / 100u, cpa100(bs, acc) % 100u);
        }
    }
    probe_puts("\nART(ARTEN+PRFTEN)는 ITCM 경로의 명령/데이터 페치를 가속하는 장치다. AXIM 경로는\n"
               "ART가 아니라 D-cache와 프리페치 큐가 담당한다. 위 표에서 각 경로가 무엇에 반응하는지\n"
               "직접 읽어낼 수 있다.\n");

    /* ---------------- 7. 외부 메모리 ---------------- */
    probe_section("7. 외부 메모리 (SDRAM / QSPI)");
    if (s_sdram_ok) {
        uint32_t words = EXT_WS_BYTES / 4u, bytes = EXT_WS_BYTES;
        struct { const char *lab; int mpu; int cm; } sd[] = {
            { "map:Device  D$WB ", 0, CM_WB  },   /* 기본맵. D$가 켜져 있어도 캐싱되지 않는다 */
            { "MPU:Normal  D$OFF", 1, CM_OFF },
            { "MPU:Normal  D$WB ", 1, CM_WB  },
        };
        unsigned si;
        probe_puts("[SDRAM] 보드 실장 SDRAM (FMC SDRAM bank1, 0xC0000000). 크기는 1절에서 실측했고,\n"
                   "        버스폭/CAS 지연/클럭은 이 절에서 재지 않았다 (FMC 레지스터 덤프 절 참조).\n"
                   "  아래 세 블록의 대비가 이 칩 메모리 시스템의 핵심이다:\n"
                   "   - map:Device  = MPU 미설정 시의 기본 메모리맵. Device라 D-cache를 켜도 캐싱되지 않는다.\n"
                   "   - MPU:Normal  = MPU 리전 0으로 8MB를 Normal WBWA(non-shareable)로 재정의한 상태.\n"
                   "  즉 첫 블록과 셋째 블록의 차이가 곧 'MPU 설정만으로 얻는 SDRAM 성능'이다.\n"
                   "  copyLDM은 읽기+쓰기 합산 규약이다.\n");
        for (si = 0; si < sizeof(sd) / sizeof(sd[0]); si++) {
            mpu_sdram_set(sd[si].mpu);
            cache_mode(sd[si].cm);
            c  = run_min(k_read32,  0, (void *)SDRAM_ADDR, words, 0, 3, MODE_COLD);
            c2 = run_min(k_write32, (void *)SDRAM_ADDR, 0, words, 0, 3, MODE_COLD);
            c3 = run_min(k_copyldm, (uint8_t *)SDRAM_ADDR + bytes, (void *)SDRAM_ADDR, words, 0, 3, MODE_COLD);
            probe_printf("  [%s] CCR.DC=%u CACR.FORCEWT=%u  속성=%s\n",
                         sd[si].lab, dc_bit(), fwt_bit(), sdram_attr_now());
            probe_puts("    read32  "); pr_bw(bytes, c);  probe_puts("\n");
            probe_puts("    write32 "); pr_bw(bytes, c2); probe_puts("\n");
            probe_puts("    copyLDM "); pr_bw(bytes * 2u, c3); probe_puts("  (읽기+쓰기 합산)\n");
        }
        mpu_sdram_set(0);
    } else {
        probe_puts("[SDRAM] 초기화 실패 또는 쓰기/되읽기 불일치 - 모든 SDRAM 측정을 건너뛰었다.\n");
    }

    if (s_qspi_ok) {
        uint32_t words = EXT_WS_BYTES / 4u, bytes = EXT_WS_BYTES, acc = 128u;
        probe_puts("\n[QSPI] 보드 실장 QSPI NOR 메모리맵 읽기 (0x90000000, 읽기 전용).\n"
                   "       부품 식별(JEDEC ID)은 이 절에서 재지 않았다 - 앞의 QSPI JEDEC ID 절을 볼 것.\n"
                   "       주소 범위는 1절의 폴트 경계 탐색이 실측한 값이다.\n");
        cache_mode(CM_OFF);
        c  = run_min(k_read32, 0, (void *)QSPI_ADDR, words, 0, 3, MODE_COLD);
        c2 = run_min(k_stride, 0, (void *)QSPI_ADDR, acc, 65536u, 3, MODE_COLD);
        probe_puts("  D$OFF read32       "); pr_bw(bytes, c); probe_puts("\n");
        probe_printf("  D$OFF 64KB 스트라이드 %4u.%02u cyc/접근 (랜덤 접근 근사)\n",
                     cpa100(c2, acc) / 100u, cpa100(c2, acc) % 100u);
        cache_mode(CM_WB);
        c  = run_min(k_read32, 0, (void *)QSPI_ADDR, words, 0, 3, MODE_COLD);
        c2 = run_min(k_stride, 0, (void *)QSPI_ADDR, acc, 65536u, 3, MODE_COLD);
        probe_puts("  D$WB  read32       "); pr_bw(bytes, c); probe_puts("\n");
        probe_printf("  D$WB  64KB 스트라이드 %4u.%02u cyc/접근\n",
                     cpa100(c2, acc) / 100u, cpa100(c2, acc) % 100u);
    } else {
        probe_puts("\n[QSPI] 메모리맵 모드 진입 실패 (0x90000000 읽기가 버스 폴트) - 측정을 건너뛰었다.\n");
    }

    /* ---------------- 8. 정렬 페널티 ---------------- */
    probe_section("8. 정렬 페널티 (32비트 로드)");
    {
        uint32_t saved_ccr = SCB->CCR;
        int unalign_trap = (saved_ccr & SCB_CCR_UNALIGN_TRP_Msk) ? 1 : 0;
        uint32_t acc = 1024u;
        if (unalign_trap) {
            SCB->CCR &= ~SCB_CCR_UNALIGN_TRP_Msk;
            __DSB(); __ISB();
            probe_puts("주: CCR.UNALIGN_TRP가 켜져 있어 측정 동안 임시로 껐다 (끝나면 원복).\n");
        }
        probe_puts("모두 64B 간격으로 접근해 접근당 라인 상황을 같게 맞췄다 (D-cache WB, 워킹셋 64KB).\n"
                   "SRAM1은 Normal WBWA라 비정렬 접근이 합법이다. SDRAM은 기본맵에서 Device이고,\n"
                   "Device 메모리의 비정렬 접근은 CCR.UNALIGN_TRP와 무관하게 항상 UsageFault이므로\n"
                   "그대로는 잴 수 없다. 그래서 이 표의 SDRAM 열은 MPU 리전 0으로 SDRAM 8MB를\n"
                   "Normal WBWA로 재정의한 상태에서 쟀다. 그럼에도 모든 측정을 폴트 가드 안에서\n"
                   "돌리므로, 폴트가 나면 그 칸만 '측정 불가(폴트)'로 표시되고 진행은 계속된다.\n\n");
        mpu_sdram_set(1);
        cache_mode(CM_WB);
        probe_printf("측정 조건 증명: CCR.DC=%u, CACR.FORCEWT=%u, CCR.UNALIGN_TRP=%u, SDRAM 속성=%s\n\n",
                     dc_bit(), fwt_bit(),
                     (SCB->CCR & SCB_CCR_UNALIGN_TRP_Msk) ? 1u : 0u, sdram_attr_now());
        probe_puts("정렬 조건                   SRAM1 [Normal WBWA]     SDRAM [MPU:Normal WBWA]\n");
        probe_puts("-----------------------------------------------------------------------------\n");
        {
            struct { const char *nm; uint32_t delta; } al[] = {
                { "4B 정렬 (라인 내부)",   0u },
                { "+1B 어긋남 (라인 내부)", 1u },
                { "+2B 어긋남 (라인 내부)", 2u },
                { "라인 경계 걸침 (+30B)", 30u },
                { "라인 경계 걸침 (+31B)", 31u },
            };
            unsigned i;
            for (i = 0; i < sizeof(al) / sizeof(al[0]); i++) {
                uint32_t a, b = 0;
                int oka = 0, okb = 0;
                a = cpa100(run_min_guarded(k_stride, 0, (uint8_t *)s_sram1_buf + al[i].delta,
                                           acc, 64u, 3, MODE_COLD, &oka), acc);
                if (s_sdram_ok)
                    b = cpa100(run_min_guarded(k_stride, 0, (uint8_t *)SDRAM_ADDR + al[i].delta,
                                               acc, 64u, 3, MODE_COLD, &okb), acc);
                pr_pad(al[i].nm, 28);
                if (oka) probe_printf("%4u.%02u cyc/접근       ", a / 100u, a % 100u);
                else     probe_puts("측정 불가(폴트)         ");
                if (!s_sdram_ok)  probe_puts("   - (SDRAM 사용 불가)\n");
                else if (!okb)    probe_puts("측정 불가(폴트)\n");
                else              probe_printf("%4u.%02u cyc/접근\n", b / 100u, b % 100u);
            }
        }
        mpu_sdram_set(0);
        if (unalign_trap) {
            SCB->CCR = saved_ccr;
            __DSB(); __ISB();
        }
    }

    /* ---------------- 마무리 ---------------- */
    /* 상태 원복은 출력보다 먼저 한다. 그래야 '원복했다'는 문장 옆에 원복 뒤의
     * 레지스터 값을 그대로 찍어서 증명할 수 있다. */
    mpu_sdram_set(0);
    ARM_MPU_Disable();
    ARM_MPU_ClrRegion(MPU_SDRAM_REGION);
    __DSB(); __ISB();
    {
        uint32_t rasr_now;
        MPU->RNR = MPU_SDRAM_REGION;
        rasr_now = MPU->RASR;

        probe_section("메모리 측정 마무리");
        probe_printf("가드 구간에서 잡아낸 버스 폴트 %u회 (전부 복구했다). 마지막 CFSR=0x%08X\n",
                     s_fault_count, s_fault_cfsr);
        probe_printf("MPU 원복 확인: MPU_CTRL=0x%08X (진입 시 0x%08X), 리전 %u RASR=0x%08X → MPU %s.\n",
                     MPU->CTRL, mpu_ctrl_at_entry, MPU_SDRAM_REGION, rasr_now,
                     (MPU->CTRL & MPU_CTRL_ENABLE_Msk)
                         ? "아직 켜져 있음 (비정상)"
                         : "꺼짐 - SDRAM은 다시 기본맵 Device로 돌아갔다");
    }
    probe_puts("이 절의 모든 수치는 CPU 로드/스토어 기준이다. DMA / DMA2D / LTDC 경로는\n"
               "AHB 마스터가 달라 대역폭 특성이 다르며, 여기서 측정하지 않았다.\n"
               "또한 모든 측정은 인터럽트 차단(PRIMASK=1) 상태에서 3회 반복 최솟값이다.\n");

    /* 상태 원복: 폴트 가드 제거 + 캐시/I-cache 상태 복원 */
    guard_remove();
    probe_art_enable();
    probe_art_reset();
    cache_mode(cm_saved);
    if ((ccr_saved & SCB_CCR_UNALIGN_TRP_Msk) != (SCB->CCR & SCB_CCR_UNALIGN_TRP_Msk)) {
        SCB->CCR = (SCB->CCR & ~SCB_CCR_UNALIGN_TRP_Msk) | (ccr_saved & SCB_CCR_UNALIGN_TRP_Msk);
        __DSB(); __ISB();
    }
    probe_printf("I-cache 원복: CCR.IC=%u → ", (SCB->CCR & SCB_CCR_IC_Msk) ? 1u : 0u);
    if (!ic_saved) probe_icache_disable();
    probe_printf("%u (진입 시 상태로 되돌렸다)\n", (SCB->CCR & SCB_CCR_IC_Msk) ? 1u : 0u);
    probe_flush();
}
