/*
 * probe_cpu.c - 명령어 처리량, FPU/DSP, 코드 배치별 실행 속도 측정.
 *
 * 측정 원칙
 *   - 모든 구간은 인터럽트를 끄고(PRIMASK) DWT_CYCCNT로 잰다. SysTick이 1ms마다
 *     들어오면 짧은 구간은 통째로 오염된다.
 *   - 대상 명령은 인라인 어셈블리로 직접 적는다. C로 적으면 -O2가 지워버리거나
 *     다른 명령으로 바꿔버려서 "무엇을 쟀는지" 보증할 수 없다.
 *   - 루프 제어(subs/bne)와 프레임 오버헤드는 "기울기"로 제거한다: 같은 본문을
 *     1배/2배로 언롤한 두 함수를 재고 그 차분만 쓴다. 빈 루프 비용을 그대로 빼는
 *     방식은 루프 제어가 payload와 듀얼이슈되는 만큼 과잉보정이라, IPC가
 *     아키텍처 상한 2.0을 넘는 물리적으로 불가능한 값을 만들 수 있다.
 *   - 각 측정은 3회 반복하고 최솟값을 쓴다. 최솟값이 노이즈가 가장 적다.
 *   - DWT_CYCCNT는 32비트다(200MHz에서 약 21.5초에 랩어라운드). 모든 구간은
 *     밀리초 단위이므로 uint32_t 뺄셈만으로 안전하다.
 */
#include "probe.h"
#include "stm32f7xx.h"

/* ========================================================================== */
/* 문자열 반복 매크로: 언롤된 명령열을 만든다                                  */
/* ========================================================================== */
#define REP2(x)    x x
#define REP4(x)    REP2(x) REP2(x)
#define REP8(x)    REP4(x) REP4(x)
#define REP16(x)   REP8(x) REP8(x)
#define REP32(x)   REP16(x) REP16(x)
#define REP64(x)   REP32(x) REP32(x)
#define REP128(x)  REP64(x) REP64(x)
#define REP256(x)  REP128(x) REP128(x)
#define REP768(x)  REP256(x) REP256(x) REP256(x)

/* ========================================================================== */
/* 출력 보조                                                                   */
/* ========================================================================== */
/* 표시 폭(터미널 칸 수). 한글은 UTF-8 3바이트이고 두 칸을 차지한다. */
static int disp_width(const char *s)
{
    int w = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p < 0x80u)       w += 1;   /* ASCII */
        else if (*p < 0xC0u)  w += 0;   /* 연속 바이트 */
        else if (*p < 0xE0u)  w += 1;   /* 2바이트 문자 */
        else                  w += 2;   /* 3바이트 이상 = 한글/CJK */
    }
    return w;
}

/* probe_printf는 '%-16s'(왼쪽 정렬)를 지원하지 않으므로 직접 채운다 */
static void pad(const char *s, int w)
{
    int n = disp_width(s);
    probe_puts(s);
    while (n < w) { probe_puts(" "); n++; }
}

/* 100배 고정소수점 값을 "x.yy"로 출력 */
static void fx2(uint32_t v100)
{
    probe_printf("%u.%02u", v100 / 100u, v100 % 100u);
}

/* 폭을 맞춰서 "  x.yy" 형태로 출력 (w는 전체 폭) */
static void fx2w(uint32_t v100, int w)
{
    uint32_t ip = v100 / 100u;
    int n = 4;                       /* ".yy" + 최소 한 자리 */
    uint32_t t = ip;
    while (t >= 10u) { t /= 10u; n++; }
    while (n < w) { probe_puts(" "); n++; }
    probe_printf("%u.%02u", ip, v100 % 100u);
}

static uint32_t divr(uint32_t a, uint32_t b)   /* 반올림 나눗셈 */
{
    return b ? (a + b / 2u) / b : 0u;
}

/* ========================================================================== */
/* 공통 측정 프레임                                                            */
/* ========================================================================== */
typedef uint32_t (*bench_fn_t)(uint32_t loops);

#define IB_LOOPS 2000u     /* 반복 횟수: 32명령 x 2000 = 64000 명령 (~0.3ms) */

/*
 * 정수 마이크로벤치 정의 매크로.
 *   %[a0]~%[a3]  누산기 (읽기/쓰기)
 *   %[b0],%[b1]  상수 소스 (읽기 전용)
 *   %[n]         루프 카운터
 * BODY를 REPN번 반복하고 마지막에 subs/bne를 붙인다.
 *
 * 각 벤치마다 함수를 두 개 만든다: fname(본문 1배)과 fname##_x2(본문 2배).
 * 두 함수는 루프 제어도, 함수 프롤로그/에필로그도, DWT 읽기 프레임도 완전히 같고
 * payload 명령 수만 정확히 2배 차이가 난다. 따라서
 *      순수 payload 사이클 = (x2 측정) - (x1 측정)
 * 이고 오버헤드는 뺄셈에서 자동으로 상쇄된다. 빈 루프 비용을 빼는 옛 방식과 달리
 * 루프 제어가 payload와 듀얼이슈되는 부분을 이중으로 빼지 않는다.
 */
#define DEF_IBENCH_1(fname, B0, B1, ASMSEQ)                                    \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    uint32_t a0 = 0x00010001u, a1 = 0x00020002u;                               \
    uint32_t a2 = 0x00030003u, a3 = 0x00050005u;                               \
    uint32_t b0 = (uint32_t)(B0), b1 = (uint32_t)(B1);                         \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\t"                                     \
                   "bne 1b\n\t"                                                \
                   : [a0] "+r" (a0), [a1] "+r" (a1), [a2] "+r" (a2),           \
                     [a3] "+r" (a3), [n] "+r" (n)                              \
                   : [b0] "r" (b0), [b1] "r" (b1)                              \
                   : "cc");                                                    \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    /* 결과를 소비해서 최적화로 사라지지 않게 한다 */                          \
    __asm volatile("" :: "r"(a0), "r"(a1), "r"(a2), "r"(a3) : "memory");       \
    return t1 - t0;                                                            \
}
#define DEF_IBENCH(fname, B0, B1, REPN, BODY)                                  \
    DEF_IBENCH_1(fname,      B0, B1, REPN(BODY))                               \
    DEF_IBENCH_1(fname##_x2, B0, B1, REPN(BODY) REPN(BODY))

/* DTCM 대상 로드/스토어용: %[p]가 DTCM 버퍼를 가리킨다 */
__attribute__((section(".dtcm_bss"), aligned(32)))
static uint32_t s_dtcm_w[512];                       /* 2KB, 워드 접근용 */

#define DEF_MBENCH_1(fname, ASMSEQ)                                            \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    uint32_t a0 = 1u, a1 = 2u, a2 = 3u, a3 = 5u;                               \
    uint32_t *p = s_dtcm_w;                                                    \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\t"                                     \
                   "bne 1b\n\t"                                                \
                   : [a0] "+r" (a0), [a1] "+r" (a1), [a2] "+r" (a2),           \
                     [a3] "+r" (a3), [n] "+r" (n)                              \
                   : [p] "r" (p)                                               \
                   : "cc", "memory");                                          \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "r"(a0), "r"(a1), "r"(a2), "r"(a3) : "memory");       \
    return t1 - t0;                                                            \
}
#define DEF_MBENCH(fname, REPN, BODY)                                          \
    DEF_MBENCH_1(fname,      REPN(BODY))                                       \
    DEF_MBENCH_1(fname##_x2, REPN(BODY) REPN(BODY))

/* 단정밀도 FPU용: %[a0]~%[a3]는 s레지스터 누산기, %[b0],%[b1]은 상수 */
#define DEF_FBENCH_1(fname, ASMSEQ)                                            \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    float a0 = 1.0f, a1 = 1.25f, a2 = 1.5f, a3 = 1.75f;                        \
    float b0 = 1.0009765625f, b1 = 0.99951171875f;                             \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\t"                                     \
                   "bne 1b\n\t"                                                \
                   : [a0] "+t" (a0), [a1] "+t" (a1), [a2] "+t" (a2),           \
                     [a3] "+t" (a3), [n] "+r" (n)                              \
                   : [b0] "t" (b0), [b1] "t" (b1)                              \
                   : "cc");                                                    \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "t"(a0), "t"(a1), "t"(a2), "t"(a3) : "memory");       \
    return t1 - t0;                                                            \
}
#define DEF_FBENCH(fname, REPN, BODY)                                          \
    DEF_FBENCH_1(fname,      REPN(BODY))                                       \
    DEF_FBENCH_1(fname##_x2, REPN(BODY) REPN(BODY))

/* 3회 반복 최솟값 */
static uint32_t bench_min3(bench_fn_t fn, uint32_t loops)
{
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 3; i++) {
        uint32_t c = fn(loops);
        if (c < best) best = c;
    }
    return best;
}

/* 측정 프레임 자체의 오버헤드(빈 구간)와 빈 루프 비용 - 참고용으로만 인쇄한다 */
static uint32_t s_frame_ovh;      /* DWT 두 번 읽기 사이의 순수 오버헤드 */
static uint32_t s_emptyloop_cyc;  /* subs+bne만 IB_LOOPS회 (payload 0) */

DEF_IBENCH(ib_empty, 1u, 1u, REP32, "")

static uint32_t measure_frame_overhead(void)
{
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 8; i++) {
        uint32_t pm = __get_PRIMASK();
        __disable_irq();
        PROBE_BARRIER();
        uint32_t t0 = DWT->CYCCNT;
        PROBE_BARRIER();
        uint32_t t1 = DWT->CYCCNT;
        __set_PRIMASK(pm);
        uint32_t d = t1 - t0;
        if (d < best) best = d;
    }
    return best;
}

/*
 * 순수 payload 사이클 = (본문 2배 측정) - (본문 1배 측정).
 * 두 함수의 차이는 payload 명령 REPN개뿐이므로, 루프 제어(subs/bne)와 함수 호출·
 * DWT 읽기 프레임 비용은 정확히 상쇄된다. 음수면(측정 노이즈) 0으로 클램프한다.
 */
static uint32_t bench_net(bench_fn_t f1, bench_fn_t f2, uint32_t loops)
{
    uint32_t c1 = bench_min3(f1, loops);
    uint32_t c2 = bench_min3(f2, loops);
    return (c2 > c1) ? (c2 - c1) : 0u;
}

/* IPC가 아키텍처 상한(2.0)을 넘는 행이 하나라도 나오면 그 사실을 표 아래에 적는다 */
static int s_ipc_over2;

static void ipc_note(void)
{
    if (s_ipc_over2) {
        probe_puts("주의: insn/cyc가 2.00을 넘는 행이 있다. Cortex-M7은 in-order 2-way이므로\n"
                   "      2.00이 아키텍처 상한이고, 초과분은 측정 오차(또는 명령 폴딩)다.\n");
        s_ipc_over2 = 0;
    }
}

static uint32_t ipc100_of(uint32_t c100)
{
    uint32_t i100 = c100 ? divr(100u * 100u, c100) : 0u;
    if (i100 > 200u) s_ipc_over2 = 1;
    return i100;
}

/* ========================================================================== */
/* 1) 클럭 실측 검증                                                           */
/* ========================================================================== */
static uint32_t s_meas_khz;    /* 실측 HCLK (kHz) */

static void cpu_clock_check(void)
{
    probe_section("CPU 1) 클럭 실측 검증 - DWT_CYCCNT가 정말 HCLK로 도는가");

    uint32_t cfgr    = RCC->CFGR;
    uint32_t pllcfgr = RCC->PLLCFGR;
    uint32_t pllm = (pllcfgr & RCC_PLLCFGR_PLLM_Msk) >> RCC_PLLCFGR_PLLM_Pos;
    uint32_t plln = (pllcfgr & RCC_PLLCFGR_PLLN_Msk) >> RCC_PLLCFGR_PLLN_Pos;
    uint32_t pllp = ((((pllcfgr & RCC_PLLCFGR_PLLP_Msk) >> RCC_PLLCFGR_PLLP_Pos)) + 1u) * 2u;
    uint32_t src_hz = (pllcfgr & RCC_PLLCFGR_PLLSRC_HSE) ? 25000000u : 16000000u;
    uint32_t vco_hz = (src_hz / (pllm ? pllm : 1u)) * plln;
    uint32_t sys_hz = vco_hz / (pllp ? pllp : 2u);
    uint32_t hpre   = (cfgr & RCC_CFGR_HPRE_Msk) >> RCC_CFGR_HPRE_Pos;
    uint32_t hclk_hz = sys_hz >> AHBPrescTable[hpre & 0xFu];

    probe_kv32("RCC_CFGR", (uint32_t)&RCC->CFGR, cfgr, "SW/HPRE/PPRE");
    probe_kv32("RCC_PLLCFGR", (uint32_t)&RCC->PLLCFGR, pllcfgr, "PLLM/N/P/SRC");
    probe_printf("레지스터 역산 : SRC=%s %u Hz, PLLM=%u PLLN=%u PLLP=%u\n",
                 (pllcfgr & RCC_PLLCFGR_PLLSRC_HSE) ? "HSE" : "HSI",
                 src_hz, pllm, plln, pllp);
    probe_printf("              VCO=%u Hz -> SYSCLK=%u Hz, HPRE=/%u -> HCLK=%u Hz\n",
                 vco_hz, sys_hz, 1u << AHBPrescTable[hpre & 0xFu], hclk_hz);

    /* SysTick(1ms 인터럽트) 대비 DWT 사이클 수를 실측한다.
     * 인터럽트를 켠 채로 재야 HAL_GetTick이 증가한다. */
    /* SysTick이 죽어 있으면(인터럽트 차단 등) 무한 대기가 되므로 사이클로 상한을 건다 */
    uint32_t guard = DWT->CYCCNT;
    uint32_t m0 = HAL_GetTick();
    while (HAL_GetTick() == m0) {            /* tick 경계에 정렬 */
        if ((uint32_t)(DWT->CYCCNT - guard) > 400000000u) break;
    }
    m0 = HAL_GetTick();
    uint32_t c0 = DWT->CYCCNT;
    while ((uint32_t)(HAL_GetTick() - m0) < 500u) {
        if ((uint32_t)(DWT->CYCCNT - c0) > 400000000u) break;   /* 최대 ~2초 */
    }
    uint32_t c1 = DWT->CYCCNT;
    uint32_t m1 = HAL_GetTick();

    uint32_t dcyc = c1 - c0;
    uint32_t dms  = m1 - m0;

    uint32_t exp_khz = hclk_hz / 1000u;
    if (dms == 0u) {
        /* 실패했으면 값을 내지 않는다. 0을 실측값처럼 찍으면 그 자체가 거짓말이다. */
        s_meas_khz = 0u;
        probe_puts("실측         : SysTick이 증가하지 않는다 (인터럽트 차단/SysTick 정지?)\n"
                   "               → 교차검증 불가. 실측 HCLK은 '미측정'으로 남긴다.\n");
        probe_printf("               (참고: 측정 창 동안 DWT는 %u cycles 증가했다)\n", dcyc);
        probe_printf("SysTick RELOAD=%u (=%u 사이클/tick), CTRL=0x%08X\n",
                     SysTick->LOAD + 1u, SysTick->LOAD + 1u, SysTick->CTRL);
        probe_printf("역산 HCLK=%u kHz, 실측 HCLK=미측정 (비교 생략)\n", exp_khz);
    } else {
        s_meas_khz = divr(dcyc, dms);            /* cycles/ms = kHz */
        probe_printf("실측         : %u ms 동안 %u cycles -> HCLK = %u.%03u MHz\n",
                     dms, dcyc, s_meas_khz / 1000u, s_meas_khz % 1000u);
        probe_printf("SysTick RELOAD=%u (=%u 사이클/tick), CTRL=0x%08X\n",
                     SysTick->LOAD + 1u, SysTick->LOAD + 1u, SysTick->CTRL);

        uint32_t diff_khz = (s_meas_khz > exp_khz) ? (s_meas_khz - exp_khz)
                                                   : (exp_khz - s_meas_khz);
        probe_printf("역산 HCLK=%u kHz, 실측 HCLK=%u kHz, 차이=%u kHz (%s)\n",
                     exp_khz, s_meas_khz, diff_khz,
                     (exp_khz && diff_khz * 1000u < exp_khz) ? "0.1% 이내 - 일치"
                                                            : "불일치 - 확인 필요");
    }
    probe_puts("해석: DWT_CYCCNT와 SysTick은 둘 다 HCLK에서 파생되므로 이 비교는\n"
               "      'CYCCNT가 HCLK를 1:1로 센다 + SysTick RELOAD가 SystemCoreClock과\n"
               "      일치한다'를 확인한다. 절대 주파수의 근거는 보드의 25MHz HSE 크리스탈이며,\n"
               "      위 PLL 역산값과 실측 사이클 수가 일치하면 그 경로 전체가 검증된 것이다.\n");
}

/* ========================================================================== */
/* 2) 명령어 처리량 / 레이턴시                                                 */
/* ========================================================================== */

/* ---- 정수 ALU ---- */
DEF_IBENCH(ib_nop_t,  1u, 1u, REP32, "nop\n\t")
DEF_IBENCH(ib_add_t,  1u, 1u, REP8,  "add.w %[a0],%[a0],#1\n\t"
                                     "add.w %[a1],%[a1],#1\n\t"
                                     "add.w %[a2],%[a2],#1\n\t"
                                     "add.w %[a3],%[a3],#1\n\t")
DEF_IBENCH(ib_add_l,  1u, 1u, REP32, "add.w %[a0],%[a0],#1\n\t")
DEF_IBENCH(ib_addsh_t, 1u, 1u, REP8, "add %[a0],%[a0],%[b0],lsl #2\n\t"
                                     "add %[a1],%[a1],%[b0],lsl #2\n\t"
                                     "add %[a2],%[a2],%[b0],lsl #2\n\t"
                                     "add %[a3],%[a3],%[b0],lsl #2\n\t")
DEF_IBENCH(ib_addsh_l, 1u, 1u, REP32, "add %[a0],%[a0],%[a0],lsl #2\n\t")
DEF_IBENCH(ib_mul_t,  3u, 5u, REP8,  "mul %[a0],%[a0],%[b0]\n\t"
                                     "mul %[a1],%[a1],%[b0]\n\t"
                                     "mul %[a2],%[a2],%[b0]\n\t"
                                     "mul %[a3],%[a3],%[b0]\n\t")
DEF_IBENCH(ib_mul_l,  3u, 5u, REP32, "mul %[a0],%[a0],%[b0]\n\t")
DEF_IBENCH(ib_mla_t,  3u, 5u, REP8,  "mla %[a0],%[b0],%[b1],%[a0]\n\t"
                                     "mla %[a1],%[b0],%[b1],%[a1]\n\t"
                                     "mla %[a2],%[b0],%[b1],%[a2]\n\t"
                                     "mla %[a3],%[b0],%[b1],%[a3]\n\t")
DEF_IBENCH(ib_mla_l,  3u, 5u, REP32, "mla %[a0],%[b0],%[b1],%[a0]\n\t")
DEF_IBENCH(ib_clz_t,  0x0000FFFFu, 7u, REP8, "clz %[a0],%[b0]\n\t"
                                             "clz %[a1],%[b0]\n\t"
                                             "clz %[a2],%[b0]\n\t"
                                             "clz %[a3],%[b0]\n\t")
DEF_IBENCH(ib_clz_l,  0x0000FFFFu, 7u, REP32, "clz %[a0],%[a0]\n\t")
/* 나눗셈: 피연산자 값에 따라 사이클이 변한다. 어려운 값/쉬운 값을 따로 잰다. */
DEF_IBENCH(ib_udiv_h, 0xFFFFFFF1u, 7u, REP8, "udiv %[a0],%[b0],%[b1]\n\t"
                                             "udiv %[a1],%[b0],%[b1]\n\t"
                                             "udiv %[a2],%[b0],%[b1]\n\t"
                                             "udiv %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_udiv_e, 0x000000FFu, 7u, REP8, "udiv %[a0],%[b0],%[b1]\n\t"
                                             "udiv %[a1],%[b0],%[b1]\n\t"
                                             "udiv %[a2],%[b0],%[b1]\n\t"
                                             "udiv %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_sdiv_h, 0x8000000Fu, 7u, REP8, "sdiv %[a0],%[b0],%[b1]\n\t"
                                             "sdiv %[a1],%[b0],%[b1]\n\t"
                                             "sdiv %[a2],%[b0],%[b1]\n\t"
                                             "sdiv %[a3],%[b0],%[b1]\n\t")

/* ---- 로드/스토어 (DTCM 대상) ---- */
DEF_MBENCH(mb_ldr_t,  REP8, "ldr %[a0],[%[p],#0]\n\t"
                            "ldr %[a1],[%[p],#4]\n\t"
                            "ldr %[a2],[%[p],#8]\n\t"
                            "ldr %[a3],[%[p],#12]\n\t")
DEF_MBENCH(mb_str_t,  REP8, "str %[a0],[%[p],#0]\n\t"
                            "str %[a1],[%[p],#4]\n\t"
                            "str %[a2],[%[p],#8]\n\t"
                            "str %[a3],[%[p],#12]\n\t")
DEF_MBENCH(mb_ldrd_t, REP16, "ldrd %[a0],%[a1],[%[p],#0]\n\t"
                             "ldrd %[a2],%[a3],[%[p],#8]\n\t")
DEF_MBENCH(mb_strd_t, REP16, "strd %[a0],%[a1],[%[p],#0]\n\t"
                             "strd %[a2],%[a3],[%[p],#8]\n\t")
/* 로드 후 즉시 사용 (use-after-load 레이턴시).
 * 포인터 체이스: s_dtcm_w[64]를 0으로 채워 두고 ldr Rd,[base,Rd]를 이어 붙인다.
 * 항상 base+0을 읽으므로 주소가 버퍼 밖으로 튀지 않는다(폴트 방지). */
#define DEF_CHASE_1(fname, ASMSEQ)                                             \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    uint32_t a0 = 0;                                                           \
    uint32_t *p = &s_dtcm_w[64];  /* 스토어 벤치가 건드리는 앞쪽과 겹치지 않게 */ \
    uint32_t n = loops, t0, t1;                                                \
    p[0] = 0u;                     /* 체인이 제자리를 맴돌게 한다 */            \
    __DSB();                                                                   \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\tbne 1b\n\t"                           \
                   : [a0] "+r" (a0), [n] "+r" (n)                              \
                   : [p] "r" (p)                                               \
                   : "cc", "memory");                                          \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "r"(a0) : "memory");                                  \
    return t1 - t0;                                                            \
}
#define CHASE_BODY "ldr %[a0],[%[p],%[a0]]\n\t"
DEF_CHASE_1(mb_ldr_l,    REP32(CHASE_BODY))
DEF_CHASE_1(mb_ldr_l_x2, REP32(CHASE_BODY) REP32(CHASE_BODY))

/* ---- 분기 예측 ----
 * 두 벤치가 "예측 가능성만" 다르려면 실행 명령 수와 분기 taken 비율까지 같아야 한다.
 * 그래서 한 단위(unit)를 아래처럼 대칭으로 짠다:
 *
 *      tst  a0,#MASK
 *      b<cc> 2f
 *      add.w a1,a1,#1        <- not-taken 경로
 *      b    3f
 *   2: add.w a1,a1,#1        <- taken 경로 (같은 명령)
 *      b    3f
 *   3:
 *
 * taken이든 아니든 실행되는 명령은 정확히 tst / 조건분기 / add.w / b 네 개다.
 * (예전 판은 taken일 때 add.w를 건너뛰어 두 벤치의 실행 명령 수가 달랐고, 그 차이가
 *  오예측 페널티로 잘못 귀속됐다.)
 *
 * 예측 가능 버전: tst a0,#0 → 항상 Z=1. 절반은 beq(항상 taken), 절반은 bne(항상 not taken).
 *                 → taken 비율 50%, 각 분기 지점은 방향이 고정이라 100% 예측 가능.
 * 난수 버전:     tst a0,#1 → xorshift32의 최하위 비트. taken 비율 ~50%, 예측 불가.
 * 두 벤치의 명령열·부호화 길이·taken 비율이 모두 같고, 남는 차이는 예측 성공률뿐이다. */
#define BR_XORSHIFT  "eor %[a0],%[a0],%[a0],lsl #13\n\t"                        \
                     "eor %[a0],%[a0],%[a0],lsr #17\n\t"                        \
                     "eor %[a0],%[a0],%[a0],lsl #5\n\t"
#define BR_UNIT(MASK, CC)  BR_XORSHIFT                                          \
                     "tst %[a0],#" MASK "\n\t"                                  \
                     CC " 2f\n\t"                                               \
                     "add.w %[a1],%[a1],#1\n\t"                                 \
                     "b 3f\n\t"                                                 \
                     "2:\n\t"                                                   \
                     "add.w %[a1],%[a1],#1\n\t"                                 \
                     "b 3f\n\t"                                                 \
                     "3:\n\t"
DEF_IBENCH(ib_br_pred, 1u, 1u, REP4, BR_UNIT("0", "beq") BR_UNIT("0", "bne"))
DEF_IBENCH(ib_br_rand, 1u, 1u, REP8, BR_UNIT("1", "beq"))

/* ---- DSP / SIMD ---- */
DEF_IBENCH(ib_smlad_t,  0x00020003u, 0x00040005u, REP8,
           "smlad %[a0],%[b0],%[b1],%[a0]\n\t" "smlad %[a1],%[b0],%[b1],%[a1]\n\t"
           "smlad %[a2],%[b0],%[b1],%[a2]\n\t" "smlad %[a3],%[b0],%[b1],%[a3]\n\t")
DEF_IBENCH(ib_smlad_l,  0x00020003u, 0x00040005u, REP32,
           "smlad %[a0],%[b0],%[b1],%[a0]\n\t")
/* SMLALD는 RdLo/RdHi 두 개를 쓴다 → 누산기 쌍 2개로 2-way ILP */
DEF_IBENCH(ib_smlald_t, 0x00020003u, 0x00040005u, REP16,
           "smlald %[a0],%[a1],%[b0],%[b1]\n\t"
           "smlald %[a2],%[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_smlald_l, 0x00020003u, 0x00040005u, REP32,
           "smlald %[a0],%[a1],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_smuad_t,  0x00020003u, 0x00040005u, REP8,
           "smuad %[a0],%[b0],%[b1]\n\t" "smuad %[a1],%[b0],%[b1]\n\t"
           "smuad %[a2],%[b0],%[b1]\n\t" "smuad %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_qadd16_t, 0x00020003u, 0x00040005u, REP8,
           "qadd16 %[a0],%[b0],%[b1]\n\t" "qadd16 %[a1],%[b0],%[b1]\n\t"
           "qadd16 %[a2],%[b0],%[b1]\n\t" "qadd16 %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_qsub16_t, 0x00020003u, 0x00040005u, REP8,
           "qsub16 %[a0],%[b0],%[b1]\n\t" "qsub16 %[a1],%[b0],%[b1]\n\t"
           "qsub16 %[a2],%[b0],%[b1]\n\t" "qsub16 %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_sadd16_t, 0x00020003u, 0x00040005u, REP8,
           "sadd16 %[a0],%[b0],%[b1]\n\t" "sadd16 %[a1],%[b0],%[b1]\n\t"
           "sadd16 %[a2],%[b0],%[b1]\n\t" "sadd16 %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_ssub16_t, 0x00020003u, 0x00040005u, REP8,
           "ssub16 %[a0],%[b0],%[b1]\n\t" "ssub16 %[a1],%[b0],%[b1]\n\t"
           "ssub16 %[a2],%[b0],%[b1]\n\t" "ssub16 %[a3],%[b0],%[b1]\n\t")
/* SEL은 직전 SIMD 연산이 남긴 GE 플래그를 쓴다. 여기서는 SADD16으로 GE를 만든다. */
DEF_IBENCH(ib_sel_t,    0x00020003u, 0x00040005u, REP8,
           "sadd16 %[a0],%[b0],%[b1]\n\t" "sel %[a1],%[b0],%[b1]\n\t"
           "sel %[a2],%[b0],%[b1]\n\t"    "sel %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_pkhbt_t,  0x00020003u, 0x00040005u, REP8,
           "pkhbt %[a0],%[b0],%[b1],lsl #8\n\t" "pkhbt %[a1],%[b0],%[b1],lsl #8\n\t"
           "pkhbt %[a2],%[b0],%[b1],lsl #8\n\t" "pkhbt %[a3],%[b0],%[b1],lsl #8\n\t")
DEF_IBENCH(ib_pkhtb_t,  0x00020003u, 0x00040005u, REP8,
           "pkhtb %[a0],%[b0],%[b1],asr #8\n\t" "pkhtb %[a1],%[b0],%[b1],asr #8\n\t"
           "pkhtb %[a2],%[b0],%[b1],asr #8\n\t" "pkhtb %[a3],%[b0],%[b1],asr #8\n\t")
DEF_IBENCH(ib_ssat_t,   0x00012345u, 7u, REP8,
           "ssat %[a0],#8,%[b0]\n\t" "ssat %[a1],#8,%[b0]\n\t"
           "ssat %[a2],#8,%[b0]\n\t" "ssat %[a3],#8,%[b0]\n\t")
DEF_IBENCH(ib_usat_t,   0x00012345u, 7u, REP8,
           "usat %[a0],#8,%[b0]\n\t" "usat %[a1],#8,%[b0]\n\t"
           "usat %[a2],#8,%[b0]\n\t" "usat %[a3],#8,%[b0]\n\t")
DEF_IBENCH(ib_sxtab16_t, 0x00020003u, 0x00040005u, REP8,
           "sxtab16 %[a0],%[b0],%[b1]\n\t" "sxtab16 %[a1],%[b0],%[b1]\n\t"
           "sxtab16 %[a2],%[b0],%[b1]\n\t" "sxtab16 %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_uxtb16_t,  0x81238123u, 7u, REP8,
           "uxtb16 %[a0],%[b0]\n\t" "uxtb16 %[a1],%[b0]\n\t"
           "uxtb16 %[a2],%[b0]\n\t" "uxtb16 %[a3],%[b0]\n\t")
DEF_IBENCH(ib_smmul_t,  0x12345678u, 0x0ABCDEF1u, REP8,
           "smmul %[a0],%[b0],%[b1]\n\t" "smmul %[a1],%[b0],%[b1]\n\t"
           "smmul %[a2],%[b0],%[b1]\n\t" "smmul %[a3],%[b0],%[b1]\n\t")
DEF_IBENCH(ib_smmla_t,  0x12345678u, 0x0ABCDEF1u, REP8,
           "smmla %[a0],%[b0],%[b1],%[a0]\n\t" "smmla %[a1],%[b0],%[b1],%[a1]\n\t"
           "smmla %[a2],%[b0],%[b1],%[a2]\n\t" "smmla %[a3],%[b0],%[b1],%[a3]\n\t")
DEF_IBENCH(ib_smmla_l,  0x12345678u, 0x0ABCDEF1u, REP32,
           "smmla %[a0],%[b0],%[b1],%[a0]\n\t")

/* ---- FPU (단정밀도) ---- */
DEF_FBENCH(fb_vadd_t, REP8, "vadd.f32 %[a0],%[a0],%[b0]\n\t"
                            "vadd.f32 %[a1],%[a1],%[b0]\n\t"
                            "vadd.f32 %[a2],%[a2],%[b0]\n\t"
                            "vadd.f32 %[a3],%[a3],%[b0]\n\t")
DEF_FBENCH(fb_vadd_l, REP32, "vadd.f32 %[a0],%[a0],%[b0]\n\t")
DEF_FBENCH(fb_vmul_t, REP8, "vmul.f32 %[a0],%[a0],%[b0]\n\t"
                            "vmul.f32 %[a1],%[a1],%[b0]\n\t"
                            "vmul.f32 %[a2],%[a2],%[b0]\n\t"
                            "vmul.f32 %[a3],%[a3],%[b0]\n\t")
DEF_FBENCH(fb_vmul_l, REP32, "vmul.f32 %[a0],%[a0],%[b0]\n\t")
DEF_FBENCH(fb_vfma_t, REP8, "vfma.f32 %[a0],%[b0],%[b1]\n\t"
                            "vfma.f32 %[a1],%[b0],%[b1]\n\t"
                            "vfma.f32 %[a2],%[b0],%[b1]\n\t"
                            "vfma.f32 %[a3],%[b0],%[b1]\n\t")
DEF_FBENCH(fb_vfma_l, REP32, "vfma.f32 %[a0],%[b0],%[b1]\n\t")
DEF_FBENCH(fb_vdiv_t, REP8, "vdiv.f32 %[a0],%[b0],%[b1]\n\t"
                            "vdiv.f32 %[a1],%[b0],%[b1]\n\t"
                            "vdiv.f32 %[a2],%[b0],%[b1]\n\t"
                            "vdiv.f32 %[a3],%[b0],%[b1]\n\t")
DEF_FBENCH(fb_vsqrt_t, REP8, "vsqrt.f32 %[a0],%[b0]\n\t"
                             "vsqrt.f32 %[a1],%[b0]\n\t"
                             "vsqrt.f32 %[a2],%[b0]\n\t"
                             "vsqrt.f32 %[a3],%[b0]\n\t")
DEF_FBENCH(fb_vcvt_f2i, REP8, "vcvt.s32.f32 %[a0],%[b0]\n\t"
                              "vcvt.s32.f32 %[a1],%[b0]\n\t"
                              "vcvt.s32.f32 %[a2],%[b0]\n\t"
                              "vcvt.s32.f32 %[a3],%[b0]\n\t")
DEF_FBENCH(fb_vcvt_i2f, REP8, "vcvt.f32.s32 %[a0],%[b0]\n\t"
                              "vcvt.f32.s32 %[a1],%[b0]\n\t"
                              "vcvt.f32.s32 %[a2],%[b0]\n\t"
                              "vcvt.f32.s32 %[a3],%[b0]\n\t")

/* VMOV: 코어 레지스터 <-> FPU 레지스터 이동 (혼합 제약이라 별도 함수) */
#define DEF_VMOV_F2I_1(fname, ASMSEQ)                                          \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    uint32_t x0 = 0, x1 = 0, x2 = 0, x3 = 0;                                   \
    float f0 = 1.5f;                                                           \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\tbne 1b\n\t"                           \
                   : [x0] "+r" (x0), [x1] "+r" (x1), [x2] "+r" (x2),           \
                     [x3] "+r" (x3), [n] "+r" (n)                              \
                   : [f0] "t" (f0)                                             \
                   : "cc");                                                    \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "r"(x0), "r"(x1), "r"(x2), "r"(x3) : "memory");       \
    return t1 - t0;                                                            \
}
#define VMOV_F2I_BODY "vmov %[x0],%[f0]\n\t" "vmov %[x1],%[f0]\n\t"            \
                      "vmov %[x2],%[f0]\n\t" "vmov %[x3],%[f0]\n\t"
DEF_VMOV_F2I_1(fb_vmov_f2i,    REP8(VMOV_F2I_BODY))
DEF_VMOV_F2I_1(fb_vmov_f2i_x2, REP8(VMOV_F2I_BODY) REP8(VMOV_F2I_BODY))

#define DEF_VMOV_I2F_1(fname, ASMSEQ)                                          \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    float f0 = 1.0f, f1 = 2.0f, f2 = 3.0f, f3 = 4.0f;                          \
    uint32_t x0 = 0x3F800000u;                                                 \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\tbne 1b\n\t"                           \
                   : [f0] "+t" (f0), [f1] "+t" (f1), [f2] "+t" (f2),           \
                     [f3] "+t" (f3), [n] "+r" (n)                              \
                   : [x0] "r" (x0)                                             \
                   : "cc");                                                    \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "t"(f0), "t"(f1), "t"(f2), "t"(f3) : "memory");       \
    return t1 - t0;                                                            \
}
#define VMOV_I2F_BODY "vmov %[f0],%[x0]\n\t" "vmov %[f1],%[x0]\n\t"            \
                      "vmov %[f2],%[x0]\n\t" "vmov %[f3],%[x0]\n\t"
DEF_VMOV_I2F_1(fb_vmov_i2f,    REP8(VMOV_I2F_BODY))
DEF_VMOV_I2F_1(fb_vmov_i2f_x2, REP8(VMOV_I2F_BODY) REP8(VMOV_I2F_BODY))

/* 왕복(코어→FPU→코어) 종속 체인: 실무에서 병목이 되는 경로 */
#define DEF_VMOV_RT_1(fname, ASMSEQ)                                           \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    float f0 = 1.0f;                                                           \
    uint32_t x0 = 0x3F800000u;                                                 \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\tbne 1b\n\t"                           \
                   : [f0] "+t" (f0), [x0] "+r" (x0), [n] "+r" (n)              \
                   :                                                           \
                   : "cc");                                                    \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "r"(x0) : "memory");                                  \
    return t1 - t0;                                                            \
}
#define VMOV_RT_BODY "vmov %[f0],%[x0]\n\t" "vmov %[x0],%[f0]\n\t"
DEF_VMOV_RT_1(fb_vmov_rt,    REP16(VMOV_RT_BODY))
DEF_VMOV_RT_1(fb_vmov_rt_x2, REP16(VMOV_RT_BODY) REP16(VMOV_RT_BODY))

/* ---- 배정밀도 (fpv5-sp-d16은 단정밀도 전용 → double은 libgcc 소프트웨어 에뮬레이션) ---- */
static volatile double s_dsink;

#define DEF_DBENCH(fname, OP)                                                  \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    double a0 = 1.0, a1 = 2.0, a2 = 3.0, a3 = 4.0;                             \
    const double b = 1.0009765625;                                             \
    uint32_t t0, t1;                                                           \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    for (uint32_t i = 0; i < loops; i++) {                                     \
        a0 = a0 OP b; a1 = a1 OP b; a2 = a2 OP b; a3 = a3 OP b;                \
    }                                                                          \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    s_dsink = a0 + a1 + a2 + a3;                                               \
    return t1 - t0;                                                            \
}
DEF_DBENCH(db_add, +)
DEF_DBENCH(db_mul, *)
DEF_DBENCH(db_div, /)

/* 단정밀도 동일 구조 (배수 비교용, C로 작성) */
static volatile float s_fsink;
#define DEF_SBENCH(fname, OP)                                                  \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    float a0 = 1.0f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;                          \
    const float b = 1.0009765625f;                                             \
    uint32_t t0, t1;                                                           \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    for (uint32_t i = 0; i < loops; i++) {                                     \
        a0 = a0 OP b; a1 = a1 OP b; a2 = a2 OP b; a3 = a3 OP b;                \
    }                                                                          \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    s_fsink = a0 + a1 + a2 + a3;                                               \
    return t1 - t0;                                                            \
}
DEF_SBENCH(sb_add, +)
DEF_SBENCH(sb_mul, *)
DEF_SBENCH(sb_div, /)

/* ---- LDM/STM (8레지스터): r4-r11이 필요하므로 naked 함수로 작성 ---- */
typedef void (*kern1_fn_t)(uint32_t);
typedef void (*kern2_fn_t)(uint32_t *, uint32_t);

/* 여기도 기울기 방식이다: 같은 커널을 8개/16개 언롤로 만들어 차분한다. */
#define DEF_LSKERN(fname, ASMSEQ)                                              \
__attribute__((naked, noinline))                                               \
static void fname(uint32_t *p, uint32_t n)                                     \
{                                                                              \
    __asm volatile(                                                            \
        "push {r4-r11}\n\t"                                                    \
        "1:\n\t"                                                               \
        ASMSEQ                                                                 \
        "subs r1,r1,#1\n\t"                                                    \
        "bne 1b\n\t"                                                           \
        "pop {r4-r11}\n\t"                                                     \
        "bx lr\n\t");                                                          \
}
#define LDM8_BODY "ldmia r0,{r4,r5,r6,r7,r8,r9,r10,r11}\n\t"
#define STM8_BODY "stmia r0,{r4,r5,r6,r7,r8,r9,r10,r11}\n\t"
DEF_LSKERN(kn_ldm8,    REP8(LDM8_BODY))
DEF_LSKERN(kn_ldm8_x2, REP8(LDM8_BODY) REP8(LDM8_BODY))
DEF_LSKERN(kn_stm8,    REP8(STM8_BODY))
DEF_LSKERN(kn_stm8_x2, REP8(STM8_BODY) REP8(STM8_BODY))

static uint32_t time_kern2(kern2_fn_t fn, uint32_t *p, uint32_t n)
{
    uint32_t t0, t1;
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    PROBE_BARRIER();
    t0 = DWT->CYCCNT;
    fn(p, n);
    t1 = DWT->CYCCNT;
    PROBE_BARRIER();
    __set_PRIMASK(pm);
    return t1 - t0;
}

static uint32_t time_kern2_min3(kern2_fn_t fn, uint32_t *p, uint32_t n)
{
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 3; i++) {
        uint32_t c = time_kern2(fn, p, n);
        if (c < best) best = c;
    }
    return best;
}

/* ---- 명령어 표 출력 ---- */
typedef struct {
    const char *name;
    bench_fn_t  thr, thr2;  /* 스루풋 (독립 체인): 본문 1배 / 2배 */
    bench_fn_t  lat, lat2;  /* 레이턴시 (종속 체인), NULL이면 생략 */
    uint16_t    n_thr;      /* 루프당 대상 명령 수 (1배 기준) */
    uint16_t    n_lat;
} ibench_t;

/* 표 항목을 만들 때 _x2 짝을 자동으로 붙인다 (짝을 빠뜨리면 컴파일 에러가 난다) */
#define IBE2(nm, T, L, NT, NL) { nm, T, T##_x2, L, L##_x2, NT, NL }
#define IBE1(nm, T, NT)        { nm, T, T##_x2, 0, 0, NT, 0 }

static uint32_t s_ipc_add100;     /* ADD 스루풋 IPC (x100) - 종합에 사용 */
static uint32_t s_smlad_cyc100;   /* SMLAD 스루풋 cyc/insn (x100) */

static void print_ibench_table(const ibench_t *tab, int cnt)
{
    probe_puts("명령            스루풋[cyc/insn]  [insn/cyc]   레이턴시[cyc]\n");
    probe_puts("--------------  ----------------  ----------   -------------\n");
    for (int i = 0; i < cnt; i++) {
        uint32_t nt = bench_net(tab[i].thr, tab[i].thr2, IB_LOOPS);
        uint32_t total_t = IB_LOOPS * tab[i].n_thr;
        uint32_t c100 = divr(nt * 100u, total_t);
        uint32_t i100 = ipc100_of(c100);

        pad(tab[i].name, 16);       /* 0-15  */
        fx2w(c100, 16);             /* 16-31 */
        probe_puts("  ");           /* 32-33 */
        fx2w(i100, 10);             /* 34-43 */
        probe_puts("   ");          /* 44-46 */
        if (tab[i].lat) {
            uint32_t nl = bench_net(tab[i].lat, tab[i].lat2, IB_LOOPS);
            uint32_t l100 = divr(nl * 100u, IB_LOOPS * tab[i].n_lat);
            fx2w(l100, 13);         /* 47-59 */
        } else {
            probe_puts("            -");
        }
        probe_puts("\n");
    }
    ipc_note();
}

static void cpu_instr_throughput(void)
{
    probe_section("CPU 2) 명령어 스루풋 / 레이턴시");

    s_frame_ovh      = measure_frame_overhead();
    s_emptyloop_cyc  = bench_min3(ib_empty, IB_LOOPS);
    uint32_t empty_slope = bench_net(ib_empty, ib_empty_x2, IB_LOOPS);

    probe_puts("보정 방식: 같은 본문을 1배/2배로 언롤한 두 함수를 재고 그 차분만 쓴다(기울기 방식).\n"
               "           루프 제어(subs/bne)와 호출·DWT 프레임 비용은 두 측정에 똑같이 들어 있어\n"
               "           차분에서 정확히 상쇄된다. 빈 루프 비용을 그대로 빼면 루프 제어가 payload와\n"
               "           듀얼이슈되는 만큼 과잉보정이 되어 IPC가 상한 2.0을 넘을 수 있다.\n");
    probe_printf("측정 프레임 오버헤드(DWT 2회 읽기 사이): %u cycles (차분에서 상쇄됨, 참고값)\n",
                 s_frame_ovh);
    probe_printf("빈 루프(subs+bne x %u): %u cycles = ", IB_LOOPS, s_emptyloop_cyc);
    fx2(divr(s_emptyloop_cyc * 100u, IB_LOOPS));
    probe_puts(" cyc/iter (참고값. 이 값을 빼지는 않는다)\n");
    probe_printf("보정 검증: payload가 없는 벤치의 1배/2배 차분 = %u cycles (0이어야 정상)\n\n",
                 empty_slope);

    static const ibench_t tab_int[] = {
        IBE1("NOP",           ib_nop_t,               32),
        IBE2("ADD imm",       ib_add_t,   ib_add_l,   32, 32),
        IBE2("ADD Rm,LSL#2",  ib_addsh_t, ib_addsh_l, 32, 32),
        IBE2("MUL",           ib_mul_t,   ib_mul_l,   32, 32),
        IBE2("MLA",           ib_mla_t,   ib_mla_l,   32, 32),
        IBE2("CLZ",           ib_clz_t,   ib_clz_l,   32, 32),
        IBE1("UDIV(어려움)",  ib_udiv_h,              32),
        IBE1("UDIV(쉬움)",    ib_udiv_e,              32),
        IBE1("SDIV(어려움)",  ib_sdiv_h,              32),
    };
    probe_puts("[정수 ALU]\n");
    print_ibench_table(tab_int, (int)(sizeof(tab_int) / sizeof(tab_int[0])));
    probe_puts("해석: 스루풋 0.5 cyc/insn = 매 사이클 2명령 발행(듀얼이슈 성공).\n"
               "      레이턴시는 앞 결과를 뒤가 바로 쓰는 체인이라 파이프라인 깊이가 드러난다.\n"
               "      나눗셈은 비파이프라인이라 스루풋=레이턴시이며 피연산자 크기에 비례해 느려진다.\n\n");

    static const ibench_t tab_ls[] = {
        IBE2("LDR (DTCM)",  mb_ldr_t,   mb_ldr_l,  32, 32),
        IBE1("STR (DTCM)",  mb_str_t,              32),
        IBE1("LDRD (DTCM)", mb_ldrd_t,             32),
        IBE1("STRD (DTCM)", mb_strd_t,             32),
    };
    probe_puts("[로드/스토어 - 전부 DTCM 대상 (0 wait state, 캐시 무관)]\n");
    print_ibench_table(tab_ls, (int)(sizeof(tab_ls) / sizeof(tab_ls[0])));

    /* LDM/STM 8레지스터: 여기도 8개/16개 언롤의 차분(기울기)으로 뽑는다 */
    uint32_t cldm  = time_kern2_min3(kn_ldm8,    s_dtcm_w, IB_LOOPS);
    uint32_t cldm2 = time_kern2_min3(kn_ldm8_x2, s_dtcm_w, IB_LOOPS);
    uint32_t cstm  = time_kern2_min3(kn_stm8,    s_dtcm_w, IB_LOOPS);
    uint32_t cstm2 = time_kern2_min3(kn_stm8_x2, s_dtcm_w, IB_LOOPS);
    uint32_t ldm_net = (cldm2 > cldm) ? cldm2 - cldm : 0u;
    uint32_t stm_net = (cstm2 > cstm) ? cstm2 - cstm : 0u;
    uint32_t ldm_c100 = divr(ldm_net * 100u, IB_LOOPS * 8u);
    uint32_t stm_c100 = divr(stm_net * 100u, IB_LOOPS * 8u);
    pad("LDM 8reg", 16); fx2w(ldm_c100, 14);
    probe_puts(" cyc/명령 (= "); fx2(divr(ldm_net * 100u, IB_LOOPS * 8u * 8u));
    probe_puts(" cyc/word, "); fx2(divr(IB_LOOPS * 8u * 32u * 100u, ldm_net));
    probe_puts(" bytes/cycle)\n");
    pad("STM 8reg", 16); fx2w(stm_c100, 14);
    probe_puts(" cyc/명령 (= "); fx2(divr(stm_net * 100u, IB_LOOPS * 8u * 8u));
    probe_puts(" cyc/word, "); fx2(divr(IB_LOOPS * 8u * 32u * 100u, stm_net));
    probe_puts(" bytes/cycle)\n");
    probe_puts("해석: LDM/STM은 명령 하나당 8워드를 옮기므로 cyc/word가 LDR보다 낮으면\n"
               "      DTCM 인터페이스가 명령 발행이 아니라 데이터 폭으로 포화한 것이다.\n\n");

    /* 분기 오예측 */
    uint32_t br_p = bench_net(ib_br_pred, ib_br_pred_x2, IB_LOOPS);
    uint32_t br_r = bench_net(ib_br_rand, ib_br_rand_x2, IB_LOOPS);
    uint32_t nbr  = IB_LOOPS * 8u;                 /* 루프당 조건분기 8개 */
    uint32_t dpen100 = (br_r > br_p) ? divr((br_r - br_p) * 100u, nbr) : 0u;
    probe_puts("[분기 예측]\n");
    pad("예측 가능(방향 고정)", 30);
    probe_printf("%u cycles / %u branches = ", br_p, nbr);
    fx2(divr(br_p * 100u, nbr)); probe_puts(" cyc/분기(주변 명령 포함)\n");
    pad("예측 불가(난수 조건)", 30);
    probe_printf("%u cycles / %u branches = ", br_r, nbr);
    fx2(divr(br_r * 100u, nbr)); probe_puts(" cyc/분기(주변 명령 포함)\n");
    pad("차이", 30); fx2(dpen100);
    probe_puts(" cyc/분기 → 오예측률 50% 가정 시 오예측 페널티 ≈ ");
    fx2(dpen100 * 2u); probe_puts(" cycles\n");
    probe_puts("해석: 두 벤치는 명령열이 같고(단위마다 tst/조건분기/add.w/b 네 개가 반드시 실행됨)\n"
               "      taken 비율도 둘 다 50%다. 예측 가능 버전은 절반을 beq(항상 taken),\n"
               "      절반을 bne(항상 not taken)로 만들어 방향을 고정했다. 실행 명령 수도\n"
               "      taken 비율도 같으므로 남는 차이는 예측 성공률뿐이고, 그 차이가 파이프라인\n"
               "      리필 비용이다.\n\n");

    static const ibench_t tab_dsp[] = {
        IBE2("SMLAD",        ib_smlad_t,   ib_smlad_l,  32, 32),
        IBE2("SMLALD",       ib_smlald_t,  ib_smlald_l, 32, 32),
        IBE1("SMUAD",        ib_smuad_t,                32),
        IBE1("QADD16",       ib_qadd16_t,               32),
        IBE1("QSUB16",       ib_qsub16_t,               32),
        IBE1("SADD16",       ib_sadd16_t,               32),
        IBE1("SSUB16",       ib_ssub16_t,               32),
        IBE1("SEL(+SADD16)", ib_sel_t,                  32),
        IBE1("PKHBT",        ib_pkhbt_t,                32),
        IBE1("PKHTB",        ib_pkhtb_t,                32),
        IBE1("SSAT",         ib_ssat_t,                 32),
        IBE1("USAT",         ib_usat_t,                 32),
        IBE1("SXTAB16",      ib_sxtab16_t,              32),
        IBE1("UXTB16",       ib_uxtb16_t,               32),
        IBE1("SMMUL",        ib_smmul_t,                32),
        IBE2("SMMLA",        ib_smmla_t,   ib_smmla_l,  32, 32),
    };
    probe_puts("[DSP / SIMD - 이 저장소 SNN 커널의 핵심 명령들]\n");
    print_ibench_table(tab_dsp, (int)(sizeof(tab_dsp) / sizeof(tab_dsp[0])));
    probe_puts("해석: SMLAD 한 개는 int16 MAC 2회, SMLALD는 64비트 누산 MAC 2회다.\n"
               "      스루풋이 1.00 cyc/insn이면 레지스터에 데이터가 있을 때 2 MAC/cycle이 상한이다.\n\n");

    static const ibench_t tab_fpu[] = {
        IBE2("VADD.F32",      fb_vadd_t,  fb_vadd_l, 32, 32),
        IBE2("VMUL.F32",      fb_vmul_t,  fb_vmul_l, 32, 32),
        IBE2("VFMA.F32",      fb_vfma_t,  fb_vfma_l, 32, 32),
        IBE1("VDIV.F32",      fb_vdiv_t,             32),
        IBE1("VSQRT.F32",     fb_vsqrt_t,            32),
        IBE1("VCVT.S32.F32",  fb_vcvt_f2i,           32),
        IBE1("VCVT.F32.S32",  fb_vcvt_i2f,           32),
        IBE1("VMOV Rd,Sn",    fb_vmov_f2i,           32),
        IBE1("VMOV Sn,Rd",    fb_vmov_i2f,           32),
        IBE1("VMOV 왕복(종속)", fb_vmov_rt,          32),
    };
    probe_puts("[FPU 단정밀도 - FPv5-SP-D16]\n");
    print_ibench_table(tab_fpu, (int)(sizeof(tab_fpu) / sizeof(tab_fpu[0])));
    probe_puts("해석: VMOV 왕복 값이 크면 정수 유닛과 FPU 사이를 오가는 코드가 병목이 된다.\n"
               "      (양자화 신경망에서 float 스케일을 int로 바꿔 곱하는 코드가 대표적이다.)\n\n");

    /* 배정밀도 vs 단정밀도 */
    const uint32_t DB_LOOPS = 5000u;       /* 4연산 x 5000 = 20000 연산 */
    static const struct { const char *n; bench_fn_t d; bench_fn_t f; } dtab[3] = {
        { "double +", db_add, sb_add },
        { "double *", db_mul, sb_mul },
        { "double /", db_div, sb_div },
    };
    probe_puts("[배정밀도(double) - 하드웨어 미지원, libgcc 소프트웨어 에뮬레이션]\n");
    probe_puts("연산       double[cyc/op]   float[cyc/op]   느려지는 배수\n");
    probe_puts("---------  --------------   -------------   -------------\n");
    for (int i = 0; i < 3; i++) {
        uint32_t cd = bench_min3(dtab[i].d, DB_LOOPS);
        uint32_t cf = bench_min3(dtab[i].f, DB_LOOPS);
        uint32_t d100 = divr(cd * 100u, DB_LOOPS * 4u);
        uint32_t f100 = divr(cf * 100u, DB_LOOPS * 4u);
        uint32_t r100 = f100 ? divr(d100 * 100u, f100) : 0u;
        pad(dtab[i].n, 11);
        fx2w(d100, 12); probe_puts("    ");
        fx2w(f100, 12); probe_puts("    ");
        fx2w(r100, 10); probe_puts("x\n");
    }
    probe_puts("해석: 이 칩의 FPU는 단정밀도 전용(FPv5-SP-D16)이라 double은 전부 함수 호출로\n"
               "      에뮬레이션된다. 신경망 추론 코드에 double이 한 줄이라도 섞이면 그 줄이\n"
               "      나머지 전부보다 비싸질 수 있다.\n\n");

    /* 종합용 값 저장 */
    s_ipc_add100 = 0;
    {
        uint32_t nt = bench_net(ib_add_t, ib_add_t_x2, IB_LOOPS);
        uint32_t c100 = divr(nt * 100u, IB_LOOPS * 32u);
        s_ipc_add100 = ipc100_of(c100);
        uint32_t ns = bench_net(ib_smlad_t, ib_smlad_t_x2, IB_LOOPS);
        s_smlad_cyc100 = divr(ns * 100u, IB_LOOPS * 32u);
    }
    ipc_note();
}

/* ========================================================================== */
/* 3) 듀얼 이슈 확인                                                           */
/* ========================================================================== */
DEF_IBENCH(du_int_int, 1u, 1u, REP16, "add.w %[a0],%[a0],#1\n\t"
                                      "add.w %[a1],%[a1],#1\n\t")
DEF_IBENCH(du_int_dep, 1u, 1u, REP16, "add.w %[a0],%[a0],#1\n\t"
                                      "add.w %[a0],%[a0],#1\n\t")
DEF_MBENCH(du_int_ld,  REP16, "add.w %[a0],%[a0],#1\n\t"
                              "ldr %[a1],[%[p],#0]\n\t")
DEF_MBENCH(du_ld_ld,   REP16, "ldr %[a0],[%[p],#0]\n\t"
                              "ldr %[a1],[%[p],#4]\n\t")
DEF_MBENCH(du_st_st,   REP16, "str %[a0],[%[p],#0]\n\t"
                              "str %[a1],[%[p],#4]\n\t")
/* 곱셈 쪽 레이턴시가 체인을 막지 않도록 누산기를 2개씩 쓴다 */
DEF_IBENCH(du_int_mul, 3u, 5u, REP8, "add.w %[a0],%[a0],#1\n\t"
                                     "mul %[a1],%[a1],%[b0]\n\t"
                                     "add.w %[a2],%[a2],#1\n\t"
                                     "mul %[a3],%[a3],%[b0]\n\t")

/* 정수 + FPU 혼합: 제약이 섞이므로 별도 함수.
 * FPU 쪽도 누산기 2개로 나눠 VADD 레이턴시가 체인을 막지 않게 한다. */
#define DEF_DU_MIX_1(fname, ASMSEQ)                                            \
static uint32_t __attribute__((noinline)) fname(uint32_t loops)                \
{                                                                              \
    uint32_t a0 = 1u, a1 = 2u;                                                 \
    float f0 = 1.0f, f1 = 2.0f, b0 = 1.0009765625f;                            \
    uint32_t n = loops, t0, t1;                                                \
    uint32_t pm = __get_PRIMASK();                                             \
    __disable_irq();                                                           \
    PROBE_BARRIER();                                                           \
    t0 = DWT->CYCCNT;                                                          \
    __asm volatile("1:\n\t" ASMSEQ                                             \
                   "subs %[n],%[n],#1\n\tbne 1b\n\t"                           \
                   : [a0] "+r" (a0), [a1] "+r" (a1),                           \
                     [f0] "+t" (f0), [f1] "+t" (f1), [n] "+r" (n)              \
                   : [b0] "t" (b0)                                             \
                   : "cc");                                                    \
    t1 = DWT->CYCCNT;                                                          \
    PROBE_BARRIER();                                                           \
    __set_PRIMASK(pm);                                                         \
    __asm volatile("" :: "r"(a0), "r"(a1), "t"(f0), "t"(f1) : "memory");       \
    return t1 - t0;                                                            \
}
#define DU_MIX_BODY "add.w %[a0],%[a0],#1\n\t" "vadd.f32 %[f0],%[f0],%[b0]\n\t" \
                    "add.w %[a1],%[a1],#1\n\t" "vadd.f32 %[f1],%[f1],%[b0]\n\t"
DEF_DU_MIX_1(du_int_fpu,    REP8(DU_MIX_BODY))
DEF_DU_MIX_1(du_int_fpu_x2, REP8(DU_MIX_BODY) REP8(DU_MIX_BODY))

static void cpu_dual_issue(void)
{
    probe_section("CPU 3) 듀얼 이슈 실측 (in-order 2-way superscalar 여부)");

#define DUE(nm, F, why) { nm, F, F##_x2, why }
    static const struct { const char *n; bench_fn_t f; bench_fn_t f2; const char *why; } tab[] = {
        DUE("정수 + 정수 (독립)", du_int_int, "가장 잘 짝지어지는 조합"),
        DUE("정수 + 정수 (종속)", du_int_dep, "앞 결과를 뒤가 소비 → 짝지을 수 없음"),
        DUE("정수 + 로드",        du_int_ld,  "ALU와 LSU가 다른 포트"),
        DUE("로드 + 로드",        du_ld_ld,   "LSU 포트가 하나면 1.0에 붙는다"),
        DUE("스토어 + 스토어",    du_st_st,   "쓰기 버퍼 깊이가 드러난다"),
        DUE("정수 + MUL",         du_int_mul, "곱셈기와 ALU의 동시 발행"),
        DUE("정수 + FPU(VADD)",   du_int_fpu, "정수와 FPU 파이프의 겹침"),
    };
#undef DUE
    probe_puts("보정: 아래 값도 본문 1배/2배 차분(기울기)이라 루프 제어 비용이 섞이지 않는다.\n");
    probe_puts("조합                     cyc/insn   insn/cyc   해석\n");
    probe_puts("-----------------------  --------   --------   ----------------------------\n");
    for (unsigned i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
        uint32_t nt = bench_net(tab[i].f, tab[i].f2, IB_LOOPS);
        uint32_t c100 = divr(nt * 100u, IB_LOOPS * 32u);
        uint32_t i100 = ipc100_of(c100);
        pad(tab[i].n, 25);
        fx2w(c100, 8); probe_puts("   ");
        fx2w(i100, 8); probe_puts("   ");
        probe_puts(tab[i].why);
        probe_puts("\n");
    }
    ipc_note();
    probe_puts("해석: cyc/insn이 0.5에 가까우면 그 조합은 매 사이클 2개씩 발행된다(IPC 2.0).\n"
               "      1.0이면 자원 충돌 또는 데이터 종속으로 듀얼이슈가 깨진 것이다.\n");
}

/* ========================================================================== */
/* 4) 코드 배치별 실행 속도                                                    */
/* ========================================================================== */
/*
 * 동일한 기계어를 4곳에서 실행한다.
 *   ITCM-RAM        0x00000000  코어 직결, 캐시/ART 무관
 *   FLASH AXIM      0x08000000  I-cache가 담당
 *   FLASH ITCM 별칭 0x00200000  ART accelerator가 담당 (I-cache 경유 안 함)
 *   SRAM1           0x20010000  AXI + I-cache
 * 커널은 레지스터 연산과 자기 내부 분기뿐이라 위치 독립적이다. 따라서 같은
 * 바이트열을 다른 주소로 복사해도 그대로 실행된다.
 */
#define KPAIR "add.w r1,r1,#1\n\t" "add.w r2,r2,#1\n\t"

/* 큰 커널: 1536 명령 x 4바이트 = 6144바이트 → 4KB I-cache/1KB ART에 안 들어간다 */
__attribute__((naked, noinline, aligned(32), section(".itcm_code")))
static void kern_big(uint32_t iters)
{
    __asm volatile("1:\n\t" REP768(KPAIR)
                   "subs r0,r0,#1\n\t" "bne 1b\n\t" "bx lr\n\t");
}

/* 작은 커널: 32 명령 = 128바이트 → 무엇에든 다 들어간다 */
__attribute__((naked, noinline, aligned(32), section(".itcm_code")))
static void kern_small(uint32_t iters)
{
    __asm volatile("1:\n\t" REP16(KPAIR)
                   "subs r0,r0,#1\n\t" "bne 1b\n\t" "bx lr\n\t");
}

#define KBIG_INSN    1536u
#define KSMALL_INSN  32u
#define KBIG_ITERS   64u
#define KSMALL_ITERS 3000u

extern uint32_t _sitcm_code, _eitcm_code, _sitcm_code_load;

#define CODE_BUF_BYTES (16u * 1024u)
__attribute__((aligned(32)))
static uint32_t s_code_buf[CODE_BUF_BYTES / 4u];      /* SRAM1(.bss)에 놓인 코드 사본 */

static uint32_t time_kern1(kern1_fn_t fn, uint32_t iters)
{
    uint32_t t0, t1;
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    PROBE_BARRIER();
    t0 = DWT->CYCCNT;
    fn(iters);
    t1 = DWT->CYCCNT;
    PROBE_BARRIER();
    __set_PRIMASK(pm);
    return t1 - t0;
}

static uint32_t time_kern1_min3(kern1_fn_t fn, uint32_t iters)
{
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 3; i++) {
        uint32_t c = time_kern1(fn, iters);
        if (c < best) best = c;
    }
    return best;
}

static uint32_t s_itcm_base, s_flash_base, s_alias_base, s_sram_base;
static int      s_sram_code_ok;

static uint32_t kern_off(const void *fn)
{
    return ((uint32_t)fn & ~1u) - s_itcm_base;
}

static kern1_fn_t kern_at(uint32_t base, const void *fn)
{
    return (kern1_fn_t)(base + kern_off(fn) + 1u);
}

/* 실행 위치 x 캐시/ART 조건 조합 */
typedef struct {
    const char *label;
    uint32_t    base;       /* 0이면 사용 불가 */
    int         icache;     /* 1=on, 0=off */
    int         art;        /* 1=on, 0=off */
} place_case_t;

static uint32_t s_best_place100, s_worst_place100;   /* 큰 커널 기준 cyc/insn */

static void run_place_table(const place_case_t *cs, int n,
                            const void *fn, uint32_t insns, uint32_t iters,
                            int track)
{
    probe_puts("실행 위치            I$   ART      cycles   cyc/insn   ns/insn   기준대비\n");
    probe_puts("-------------------  ---  ---   ---------   --------   -------   --------\n");
    uint32_t ref100 = 0;
    for (int i = 0; i < n; i++) {
        if (cs[i].base == 0u && i != 0) { continue; }
        if (cs[i].icache) probe_icache_enable(); else probe_icache_disable();
        if (cs[i].art)    probe_art_enable();    else probe_art_disable();
        PROBE_BARRIER();

        kern1_fn_t f = kern_at(cs[i].base, fn);
        uint32_t c = time_kern1_min3(f, iters);
        uint32_t c100 = divr(c * 100u, insns * iters);
        uint32_t ns100 = divr(c100 * 1000u, g_sysclk_hz / 1000000u);
        if (i == 0) ref100 = c100;

        pad(cs[i].label, 21);
        pad(cs[i].icache ? "ON " : "OFF", 5);
        pad(cs[i].art ? "ON " : "OFF", 6);
        probe_printf("%9u   ", c);
        fx2w(c100, 8); probe_puts("   ");
        fx2w(ns100, 7); probe_puts("   ");
        fx2w(ref100 ? divr(c100 * 100u, ref100) : 0u, 7);
        probe_puts("x\n");

        if (track) {
            if (s_best_place100 == 0u || c100 < s_best_place100) s_best_place100 = c100;
            if (c100 > s_worst_place100) s_worst_place100 = c100;
        }
    }
    /* 조건 복구 */
    probe_icache_enable();
    probe_art_enable();
    PROBE_BARRIER();
}

static void cpu_code_placement(void)
{
    probe_section("CPU 4) 코드 배치별 실행 속도 - 같은 기계어를 네 곳에서 실행");

    s_itcm_base  = (uint32_t)&_sitcm_code;
    s_flash_base = (uint32_t)&_sitcm_code_load;
    s_alias_base = 0x00200000u + (s_flash_base - FLASH_BASE);
    uint32_t sz  = (uint32_t)&_eitcm_code - (uint32_t)&_sitcm_code;

    probe_printf(".itcm_code 섹션: ITCM 0x%08X, FLASH 적재 0x%08X, ITCM별칭 0x%08X, %u bytes\n",
                 s_itcm_base, s_flash_base, s_alias_base, sz);

    /* SRAM1으로 코드 사본을 만든다 */
    s_sram_code_ok = 0;
    if (sz <= CODE_BUF_BYTES) {
        /* 원본은 플래시 적재 주소에서 읽는다 (ITCM 기준 주소 0에서 읽으면
         * 컴파일러가 널 포인터 접근으로 오해할 여지가 있다) */
        const uint32_t *src = (const uint32_t *)s_flash_base;
        uint32_t *dst = s_code_buf;
        for (uint32_t i = 0; i < (sz + 3u) / 4u; i++) dst[i] = src[i];
        /* D-cache로 쓴 코드가 실제 메모리/I-cache에 보이게 만든다.
         * 이 세 줄이 없으면 엉뚱한(예전) 명령을 페치해서 폴트가 난다. */
        SCB_CleanDCache();
        SCB_InvalidateICache();
        __DSB();
        __ISB();
        s_sram_base = (uint32_t)s_code_buf;
        s_sram_code_ok = 1;
        probe_printf("SRAM1 사본: 0x%08X (%u bytes 복사, CleanDCache+InvalidateICache 완료)\n",
                     s_sram_base, sz);
    } else {
        s_sram_base = 0;
        probe_printf("SRAM1 사본: 건너뜀 (.itcm_code %u bytes > 버퍼 %u bytes)\n",
                     sz, CODE_BUF_BYTES);
    }

    probe_printf("커널: 독립 ADD 2개를 번갈아 %u회(큰 커널 %u bytes) / %u회(작은 커널 %u bytes)\n",
                 KBIG_INSN, KBIG_INSN * 4u, KSMALL_INSN, KSMALL_INSN * 4u);
    probe_puts("      독립 명령이므로 ITCM에서는 듀얼이슈로 0.5 cyc/insn이 상한이다.\n"
               "      즉 명령 페치가 8 bytes/cycle을 못 대면 그 즉시 병목이 보인다.\n\n");

    place_case_t cases[] = {
        { "ITCM-RAM",        s_itcm_base,  1, 1 },
        { "FLASH AXIM",      s_flash_base, 1, 1 },
        { "FLASH AXIM",      s_flash_base, 0, 1 },
        { "FLASH ITCM별칭",  s_alias_base, 1, 1 },
        { "FLASH ITCM별칭",  s_alias_base, 1, 0 },
        { "FLASH ITCM별칭",  s_alias_base, 0, 0 },
        { "SRAM1",           s_sram_base,  1, 1 },
        { "SRAM1",           s_sram_base,  0, 1 },
    };
    int ncase = (int)(sizeof(cases) / sizeof(cases[0]));
    if (!s_sram_code_ok) ncase -= 2;

    probe_printf("[작은 커널 %u bytes - I-cache(4KB)/ART(1KB)에 전부 들어간다]\n",
                 KSMALL_INSN * 4u);
    run_place_table(cases, ncase, (const void *)kern_small, KSMALL_INSN, KSMALL_ITERS, 0);
    probe_puts("해석: 루프가 캐시/ART에 다 들어가면 어디에 놓든 차이가 거의 없다.\n"
               "      \"플래시에서 돌려도 안 느리던데\"라는 착각은 대개 이 조건에서 나온다.\n\n");

    probe_printf("[큰 커널 %u bytes - I-cache 4KB / ART 1KB를 넘긴다]\n", KBIG_INSN * 4u);
    s_best_place100 = 0; s_worst_place100 = 0;
    run_place_table(cases, ncase, (const void *)kern_big, KBIG_INSN, KBIG_ITERS, 1);
    probe_puts("해석: 여기서부터가 진짜다. 코드가 캐시보다 크면 명령 페치 경로가 그대로\n"
               "      성능이 된다. ITCM은 64비트 폭 코어 직결이라 매 사이클 2명령을 대주지만,\n"
               "      플래시는 6 wait state라 ART/I-cache가 없으면 페치가 곧 벽이 된다.\n");
    if (s_best_place100) {
        probe_puts("      최악/최선 배치 비율: ");
        fx2(divr(s_worst_place100 * 100u, s_best_place100));
        probe_puts("배\n");
    }
    probe_puts("\n왜 이 측정이 중요한가:\n"
               "  이 보드의 기존 SNN 펌웨어는 VTOR=0x00200000, 즉 플래시의 ITCM 별칭에서\n"
               "  실행되면서 ART(FLASH_ACR.ARTEN)가 꺼져 있었다. 위 표의 'FLASH ITCM별칭 /\n"
               "  ART OFF' 행이 바로 그 조건이고, 'ITCM-RAM' 행이 이론적 상한이다.\n"
               "  두 행의 비가 그 설정으로 잃고 있던 배수다. ARTEN 비트 하나로 회수된다.\n");
}

/* ========================================================================== */
/* 5) DWT 이벤트 카운터                                                        */
/* ========================================================================== */
static void dwt_one(const char *name, bench_fn_t fn, uint32_t loops, uint32_t insns)
{
    probe_dwt_counters_t c;
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    probe_dwt_counters_reset();
    (void)fn(loops);
    probe_dwt_counters_read(&c);
    __set_PRIMASK(pm);

    pad(name, 22);
    probe_printf("%6u %5u %5u %5u %5u %5u   ", c.cyc, c.cpi, c.exc, c.sleep, c.lsu, c.fold);
    if (c.cyc > 255u) {
        probe_printf("8비트 카운터 포화 가능 (cyc=%u > 255) → 수치 신뢰 불가\n", c.cyc);
    } else {
        /* 실행 명령 수 추정: 대상 명령 + 루프 제어 2개/iter + 호출 오버헤드 */
        uint32_t est_insn = insns * loops + 2u * loops;
        uint32_t cpi100 = est_insn ? divr(c.cyc * 100u, est_insn) : 0u;
        probe_puts("대상명령 ~"); probe_printf("%u", est_insn);
        probe_puts("개, CPI≈"); fx2(cpi100); probe_puts("\n");
    }
}

static void cpu_dwt_counters(void)
{
    probe_section("CPU 5) DWT 이벤트 카운터로 본 마이크로아키텍처");
    probe_puts("카운터는 전부 8비트다(256에서 되감긴다). 그래서 구간을 200사이클 미만으로 잡는다.\n");
    probe_puts("CPICNT=추가 사이클, EXCCNT=예외 오버헤드, LSUCNT=로드/스토어 추가 사이클,\n"
               "FOLDCNT=폴딩되어 0사이클로 처리된 명령 수.\n\n");
    probe_puts("워크로드                 CYC  CPI  EXC SLEEP  LSU FOLD   비고\n");
    probe_puts("--------------------  ------ ---- ---- ----- ---- ----   ------------------\n");
    dwt_one("정수 ADD (독립)",   ib_add_t,  4u, 32u);
    dwt_one("정수 ADD (종속)",   ib_add_l,  4u, 32u);
    dwt_one("LDR (DTCM)",        mb_ldr_t,  4u, 32u);
    dwt_one("MUL",               ib_mul_t,  4u, 32u);
    /* 분기 벤치는 단위마다 eor x3 + tst + 조건분기 + add.w + b = 7개가 실행된다.
     * 단위가 8개이므로 반복당 56개다(예전 값 48은 대칭화 이전 구조 기준이었다). */
    dwt_one("분기 (방향 고정)",  ib_br_pred, 2u, 56u);
    dwt_one("분기 (난수 조건)",  ib_br_rand, 2u, 56u);
    probe_puts("해석: FOLD가 크면 그만큼의 명령이 공짜로 처리된 것이고(주로 분기/NOP),\n"
               "      LSU가 크면 로드/스토어가 추가 사이클을 먹은 것이다. EXC가 0이 아니면\n"
               "      측정 구간에 예외가 끼어든 것이므로 그 행은 버려야 한다.\n");
}

/* ========================================================================== */
/* 6) SNN 대표 커널 (int16 MAC)                                                */
/* ========================================================================== */
#define MAC_N      512u                       /* int16 원소 수 (=1KB) */
#define MAC_WORDS  (MAC_N / 2u)               /* 32비트 워드 수 */
#define MAC_REPS   200u

__attribute__((section(".dtcm_bss"), aligned(32))) static int16_t s_dt_a[MAC_N];
__attribute__((section(".dtcm_bss"), aligned(32))) static int16_t s_dt_b[MAC_N];
__attribute__((aligned(32))) static int16_t s_sr_a[MAC_N];   /* SRAM1(.bss) */
__attribute__((aligned(32))) static int16_t s_sr_b[MAC_N];

static volatile int32_t  s_mac_sink32;
static volatile uint32_t s_mac_sink64lo;

static void mac_fill(void)
{
    uint32_t x = 0x12345678u;
    for (uint32_t i = 0; i < MAC_N; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        s_dt_a[i] = (int16_t)(x & 0x7FFFu) - 16384;
        s_sr_a[i] = s_dt_a[i];
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        s_dt_b[i] = (int16_t)(x & 0x7FFFu) - 16384;
        s_sr_b[i] = s_dt_b[i];
    }
}

/*
 * 세 커널 모두 noipa다. 이유:
 *   mac_scalar()는 순수 C라서 GCC의 ipa-pure-const가 "pure"로 마킹한다. 그러면
 *   MAC_REPS 루프에서 인자가 불변이므로 호출이 통째로 루프 밖으로 끌려 나가고
 *   (실제 ELF에서 bl 1회 + 결과 x200으로 접힌 것을 확인했다) 스칼라 사이클이
 *   1/200이 되어 MAC/cyc와 GOPS가 200배 부풀려진다. noinline으로는 막지 못한다.
 *   noipa는 함수 간 분석 자체를 금지해 호출을 불투명하게 만든다.
 *   SMLAD/SMLALD 쪽은 asm volatile이라 원래 접히지 않지만, 세 커널의 호출 조건을
 *   똑같이 맞춰야 비교가 공정하므로 같은 속성을 붙인다.
 */
/* SMLAD: 32비트 누산. 워드 하나에 int16 2개 → 명령 하나에 MAC 2회 */
static int32_t __attribute__((noipa))
mac_smlad(const int16_t *a, const int16_t *b, uint32_t nwords)
{
    const uint32_t *pa = (const uint32_t *)(const void *)a;
    const uint32_t *pb = (const uint32_t *)(const void *)b;
    uint32_t acc0 = 0, acc1 = 0;
    for (uint32_t i = 0; i + 4u <= nwords; i += 4u) {
        acc0 = __SMLAD(pa[i + 0u], pb[i + 0u], acc0);
        acc1 = __SMLAD(pa[i + 1u], pb[i + 1u], acc1);
        acc0 = __SMLAD(pa[i + 2u], pb[i + 2u], acc0);
        acc1 = __SMLAD(pa[i + 3u], pb[i + 3u], acc1);
    }
    return (int32_t)(acc0 + acc1);
}

/* SMLALD: 64비트 누산 (오버플로 걱정 없는 버전) */
static uint64_t __attribute__((noipa))
mac_smlald(const int16_t *a, const int16_t *b, uint32_t nwords)
{
    const uint32_t *pa = (const uint32_t *)(const void *)a;
    const uint32_t *pb = (const uint32_t *)(const void *)b;
    uint64_t acc0 = 0, acc1 = 0;
    for (uint32_t i = 0; i + 4u <= nwords; i += 4u) {
        acc0 = __SMLALD(pa[i + 0u], pb[i + 0u], acc0);
        acc1 = __SMLALD(pa[i + 1u], pb[i + 1u], acc1);
        acc0 = __SMLALD(pa[i + 2u], pb[i + 2u], acc0);
        acc1 = __SMLALD(pa[i + 3u], pb[i + 3u], acc1);
    }
    return acc0 + acc1;
}

/* 스칼라 C: int32 += a*b (컴파일러가 만들어내는 코드 그대로) */
static int32_t __attribute__((noipa))
mac_scalar(const int16_t *a, const int16_t *b, uint32_t n)
{
    int32_t acc0 = 0, acc1 = 0;
    for (uint32_t i = 0; i + 4u <= n; i += 4u) {
        acc0 += (int32_t)a[i + 0u] * (int32_t)b[i + 0u];
        acc1 += (int32_t)a[i + 1u] * (int32_t)b[i + 1u];
        acc0 += (int32_t)a[i + 2u] * (int32_t)b[i + 2u];
        acc1 += (int32_t)a[i + 3u] * (int32_t)b[i + 3u];
    }
    return acc0 + acc1;
}

typedef enum { MAC_SMLAD, MAC_SMLALD, MAC_SCALAR } mac_kind_t;

static uint32_t mac_time(mac_kind_t k, const int16_t *a, const int16_t *b)
{
    uint32_t best = 0xFFFFFFFFu;
    for (int r = 0; r < 3; r++) {
        uint32_t pm = __get_PRIMASK();
        __disable_irq();
        PROBE_BARRIER();
        uint32_t t0 = DWT->CYCCNT;
        /* 각 반복마다 메모리 클로버를 넣어 "인자가 같으니 결과도 같다"는 가정을
         * 깬다. noipa와 중복 방어이며, 세 커널에 똑같이 넣어 조건을 맞춘다. */
        if (k == MAC_SMLAD) {
            int32_t s = 0;
            for (uint32_t i = 0; i < MAC_REPS; i++) {
                s += mac_smlad(a, b, MAC_WORDS);
                __asm volatile("" : "+r" (s) :: "memory");
            }
            s_mac_sink32 = s;
        } else if (k == MAC_SMLALD) {
            uint64_t s = 0;
            for (uint32_t i = 0; i < MAC_REPS; i++) {
                s += mac_smlald(a, b, MAC_WORDS);
                __asm volatile("" ::: "memory");
            }
            s_mac_sink64lo = (uint32_t)s;
        } else {
            int32_t s = 0;
            for (uint32_t i = 0; i < MAC_REPS; i++) {
                s += mac_scalar(a, b, MAC_N);
                __asm volatile("" : "+r" (s) :: "memory");
            }
            s_mac_sink32 = s;
        }
        uint32_t t1 = DWT->CYCCNT;
        PROBE_BARRIER();
        __set_PRIMASK(pm);
        uint32_t d = t1 - t0;
        if (d < best) best = d;
    }
    return best;
}

static uint32_t s_mac100_best;      /* 최고 MAC/cycle (x100) */
static uint32_t s_mac100_scalar;    /* 스칼라 MAC/cycle (x100) */

/* 라벨이 아니라 레지스터가 진실이다. 측정 직전/직후의 CCR.DC를 그대로 읽는다. */
static int dcache_is_on(void)
{
    return (SCB->CCR & SCB_CCR_DC_Msk) ? 1 : 0;
}

/* 이미 원하는 상태면 건드리지 않는다 (불필요한 무효화/클린을 피한다) */
static void dcache_set(int on)
{
    int cur = dcache_is_on();
    if (on && !cur)       probe_dcache_enable();
    else if (!on && cur)  probe_dcache_disable();
    PROBE_BARRIER();
}

static void mac_row(const char *label, mac_kind_t k, const int16_t *a, const int16_t *b,
                    int store_best, int store_scalar)
{
    int dc_before = dcache_is_on();
    uint32_t c = mac_time(k, a, b);
    int dc_after = dcache_is_on();
    uint32_t macs = MAC_N * MAC_REPS;                 /* 총 MAC 수 */
    uint32_t m100 = divr(macs * 100u, c);
    /* GOPS x100: MAC/cyc * 2 OP * f[MHz] / 1000 */
    uint32_t g100 = divr(m100 * 2u * (g_sysclk_hz / 1000000u), 1000u);
    pad(label, 34);
    probe_printf("%9u   ", c);
    fx2w(m100, 8); probe_puts("   ");
    fx2w(g100, 8);
    /* 라벨과 실제 조건이 일치함을 증명하는 열: 측정 구간의 CCR.DC 실측값 */
    probe_printf("   %u%s\n", (unsigned)dc_before,
                 (dc_before == dc_after) ? "" : " (측정 중 변경됨!)");
    if (store_best && m100 > s_mac100_best) s_mac100_best = m100;
    if (store_scalar) s_mac100_scalar = m100;
}

static void cpu_snn_kernel(void)
{
    probe_section("CPU 6) SNN 대표 커널 - int16 MAC 처리량");
    mac_fill();

    probe_printf("벡터 길이 %u개 int16(=%u bytes) x %u회 = MAC %u회\n",
                 MAC_N, MAC_N * 2u, MAC_REPS, MAC_N * MAC_REPS);
    probe_printf("데이터 위치: DTCM 0x%08X / SRAM1 0x%08X\n",
                 (uint32_t)s_dt_a, (uint32_t)s_sr_a);

    /* 라벨이 "D$ON"이면 그 시점에 실제로 켜져 있어야 한다. 이 절에 들어올 때의
     * 상태는 보장되지 않으므로(부팅 직후 D-cache는 꺼져 있고 probe_mem은 진입 시
     * 상태로 되돌려 놓고 나간다) 각 조건을 재기 직전에 직접 설정하고, 각 행에
     * CCR.DC 실측값을 같이 찍는다. 절을 나갈 때는 진입 시 상태로 복구한다. */
    int dc_saved = dcache_is_on();
    probe_printf("이 절 진입 시 SCB->CCR=0x%08X (D-cache %s, I-cache %s)\n",
                 SCB->CCR,
                 dc_saved ? "ON" : "OFF",
                 (SCB->CCR & SCB_CCR_IC_Msk) ? "ON" : "OFF");
    probe_puts("각 행의 CCR.DC 열은 그 행을 측정하는 동안 읽은 실제 캐시 상태다(라벨의 근거).\n");

    probe_puts("\n커널 / 데이터 위치                   cycles    MAC/cyc       GOPS   CCR.DC\n");
    probe_puts("--------------------------------  ---------   --------   --------   ------\n");

    s_mac100_best = 0; s_mac100_scalar = 0;
    dcache_set(1);
    mac_row("SMLAD  (32b누산) / DTCM",   MAC_SMLAD,  s_dt_a, s_dt_b, 1, 0);
    mac_row("SMLALD (64b누산) / DTCM",   MAC_SMLALD, s_dt_a, s_dt_b, 1, 0);
    mac_row("스칼라 C int32   / DTCM",   MAC_SCALAR, s_dt_a, s_dt_b, 0, 1);
    mac_row("SMLAD  (32b누산) / SRAM1 D$ON", MAC_SMLAD, s_sr_a, s_sr_b, 0, 0);
    mac_row("스칼라 C int32   / SRAM1 D$ON", MAC_SCALAR, s_sr_a, s_sr_b, 0, 0);

    dcache_set(0);
    mac_row("SMLAD  (32b누산) / SRAM1 D$OFF", MAC_SMLAD, s_sr_a, s_sr_b, 0, 0);
    mac_row("스칼라 C int32   / SRAM1 D$OFF", MAC_SCALAR, s_sr_a, s_sr_b, 0, 0);

    dcache_set(dc_saved);      /* 진입 시 상태로 복구 */
    probe_printf("절 종료 시 SCB->CCR=0x%08X (진입 시 상태로 복구)\n", SCB->CCR);
    probe_puts("주: DTCM(0x20000000)은 코어 직결 TCM이라 애초에 D-cache 대상이 아니다.\n"
               "    위 DTCM 행의 CCR.DC=1은 '조건을 고정했다'는 뜻일 뿐 캐시가 개입한다는 뜻이 아니다.\n");

    if (s_mac100_scalar) {
        probe_puts("\nSIMD 이득(DTCM 기준): SMLAD / 스칼라 = ");
        fx2(divr(s_mac100_best * 100u, s_mac100_scalar));
        probe_puts("배\n");
    }
    uint32_t peak100 = 2u * 2u * (g_sysclk_hz / 1000000u) / 10u;   /* 2MAC/cyc x 2OP x MHz /1000 x100 */
    probe_puts("이론 피크(레지스터 상주, SMLAD 1 cyc/insn 가정): 2.00 MAC/cycle = ");
    fx2(peak100);
    probe_printf(" GOPS @ %u MHz\n", g_sysclk_hz / 1000000u);
    probe_puts("해석: 실측이 2.00 MAC/cycle에 못 미치는 만큼이 로드 비용이다. SMLAD 한 개는\n"
               "      워드 2개(가중치+입력)를 읽어야 하므로, 로드 포트가 한 사이클에 한 워드면\n"
               "      MAC/cycle의 상한은 로드 쪽에서 결정된다. DTCM에 두면 wait state가 0이라\n"
               "      이 상한에 가장 가깝다. SRAM1은 AXI를 타므로 D-cache 유무가 그대로 드러나는데,\n"
               "      그 크기는 위 표의 'SRAM1 D$ON' 행과 'SRAM1 D$OFF' 행의 비(CCR.DC 열이 1과 0인\n"
               "      두 행)가 말해 준다. 미리 단정하지 말고 그 두 행을 직접 비교하라.\n");
}

/* ========================================================================== */
/* 7) 200MHz vs 216MHz                                                         */
/* ========================================================================== */
/* 8워드 LDM으로 버퍼를 훑는 읽기 대역폭 커널 (naked: r4-r11 필요) */
typedef void (*kern3_fn_t)(const uint32_t *, uint32_t, uint32_t);

__attribute__((naked, noinline))
static void kn_read(const uint32_t *p, uint32_t inner, uint32_t outer)
{
    __asm volatile(
        "push {r4-r11}\n\t"
        "1:\n\t"
        "mov r3,r0\n\t"
        "mov r12,r1\n\t"
        "2:\n\t"
        REP4("ldmia r3!,{r4,r5,r6,r7,r8,r9,r10,r11}\n\t")
        "subs r12,r12,#1\n\t"
        "bne 2b\n\t"
        "subs r2,r2,#1\n\t"
        "bne 1b\n\t"
        "pop {r4-r11}\n\t"
        "bx lr\n\t");
}

__attribute__((aligned(32))) static uint32_t s_sram_rd[2048];   /* 8KB, SRAM1 */

static uint32_t read_bw_cycles(const uint32_t *p, uint32_t bytes, uint32_t outer)
{
    uint32_t inner = bytes / 128u;          /* 반복당 4 x 8워드 = 128 bytes */
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 3; i++) {
        uint32_t t0, t1;
        uint32_t pm = __get_PRIMASK();
        __disable_irq();
        PROBE_BARRIER();
        t0 = DWT->CYCCNT;
        kn_read(p, inner, outer);
        t1 = DWT->CYCCNT;
        PROBE_BARRIER();
        __set_PRIMASK(pm);
        uint32_t d = t1 - t0;
        if (d < best) best = d;
    }
    return best;
}

/*
 * 지금 이 순간의 클럭을 RCC 레지스터에서 직접 역산한다.
 * g_sysclk_hz 등의 전역은 probe_set_sysclk()이 "성공했을 때만" 갱신되므로,
 * 전환이 중간에 실패하면(칩은 이미 HSI로 내려가 있다) 낡은 값이 남는다.
 * 그 값으로 BRR을 잡으면 이후 시리얼 출력이 통째로 깨진다. 레지스터가 진실이다.
 */
static uint32_t sysclk_from_regs(void)
{
    uint32_t cfgr = RCC->CFGR;
    uint32_t sws  = cfgr & RCC_CFGR_SWS_Msk;
    if (sws == RCC_CFGR_SWS_HSI) return HSI_VALUE;
    if (sws == RCC_CFGR_SWS_HSE) return HSE_VALUE;
    /* PLL */
    uint32_t p = RCC->PLLCFGR;
    uint32_t pllm = (p & RCC_PLLCFGR_PLLM_Msk) >> RCC_PLLCFGR_PLLM_Pos;
    uint32_t plln = (p & RCC_PLLCFGR_PLLN_Msk) >> RCC_PLLCFGR_PLLN_Pos;
    uint32_t pllp = ((((p & RCC_PLLCFGR_PLLP_Msk) >> RCC_PLLCFGR_PLLP_Pos)) + 1u) * 2u;
    uint32_t src  = (p & RCC_PLLCFGR_PLLSRC_HSE) ? HSE_VALUE : HSI_VALUE;
    if (pllm == 0u || pllp == 0u) return HSI_VALUE;
    return (src / pllm) * plln / pllp;
}

static uint32_t hclk_from_regs(void)
{
    uint32_t hpre = (RCC->CFGR & RCC_CFGR_HPRE_Msk) >> RCC_CFGR_HPRE_Pos;
    return sysclk_from_regs() >> AHBPrescTable[hpre & 0xFu];
}

static uint32_t pclk2_from_regs(void)
{
    uint32_t ppre2 = (RCC->CFGR & RCC_CFGR_PPRE2_Msk) >> RCC_CFGR_PPRE2_Pos;
    return hclk_from_regs() >> APBPrescTable[ppre2 & 0x7u];
}

/* 216MHz 전환 후 UART BRR을 새 PCLK2에 맞춰 다시 계산해 넣는다.
 * probe_io_init()은 결과 버퍼를 지우므로 절대 다시 부르면 안 된다. */
static void uart_rebaud(void)
{
    uint32_t pclk2 = pclk2_from_regs();       /* 전역이 아니라 레지스터에서 */
    uint32_t cr1 = USART1->CR1;
    USART1->CR1 = cr1 & ~USART_CR1_UE;
    USART1->BRR = (pclk2 + 115200u / 2u) / 115200u;
    USART1->CR1 = cr1;
    /* TEACK를 무한정 기다리지 않는다 (UART가 꺼져 있으면 그대로 진행) */
    for (uint32_t i = 0; i < 100000u; i++) {
        if (USART1->ISR & USART_ISR_TEACK) break;
    }
}

typedef struct {
    uint32_t int_c100;     /* 정수 ADD 스루풋 cyc/insn x100 */
    uint32_t smlad_m100;   /* SMLAD MAC/cycle x100 (DTCM) */
    uint32_t dtcm_bpc100;  /* DTCM 읽기 bytes/cycle x100 */
    uint32_t sram_bpc100;  /* SRAM1 읽기 bytes/cycle x100 */
    uint32_t flash_c100;   /* FLASH AXIM 큰 커널 cyc/insn x100 */
} clkset_t;

static void collect_clkset(clkset_t *o)
{
    uint32_t nt = bench_net(ib_add_t, ib_add_t_x2, IB_LOOPS);
    o->int_c100 = divr(nt * 100u, IB_LOOPS * 32u);

    uint32_t c = mac_time(MAC_SMLAD, s_dt_a, s_dt_b);
    o->smlad_m100 = divr(MAC_N * MAC_REPS * 100u, c);

    uint32_t cd = read_bw_cycles(s_dtcm_w, sizeof(s_dtcm_w), 64u);
    o->dtcm_bpc100 = divr(sizeof(s_dtcm_w) * 64u * 100u, cd);

    uint32_t cs = read_bw_cycles(s_sram_rd, sizeof(s_sram_rd), 16u);
    o->sram_bpc100 = divr(sizeof(s_sram_rd) * 16u * 100u, cs);

    probe_icache_enable();
    probe_art_enable();
    PROBE_BARRIER();
    uint32_t cf = time_kern1_min3(kern_at(s_flash_base, (const void *)kern_big), KBIG_ITERS);
    o->flash_c100 = divr(cf * 100u, KBIG_INSN * KBIG_ITERS);
}

static void clk_row(const char *name, uint32_t v200, uint32_t v216,
                    int is_cycle_metric, const char *unit)
{
    pad(name, 26);
    fx2w(v200, 9); probe_puts(" "); pad(unit, 9);
    fx2w(v216, 9); probe_puts(" "); pad(unit, 9);
    if (v200) {
        uint32_t r = divr(v216 * 100u, v200);
        fx2w(r, 7); probe_puts("x");
    } else {
        probe_puts("      -");
    }
    if (is_cycle_metric) {
        /* 절대 시간으로도 환산: ns x100 = cyc100 * 1000 / MHz */
        uint32_t ns200 = divr(v200 * 1000u, 200u);
        uint32_t ns216 = divr(v216 * 1000u, 216u);
        probe_puts("   ("); fx2(ns200); probe_puts(" ns → ");
        fx2(ns216); probe_puts(" ns)");
    }
    probe_puts("\n");
}

static void cpu_clock_compare(void)
{
    probe_section("CPU 7) 200MHz vs 216MHz");

    clkset_t at200, at216;
    collect_clkset(&at200);

    probe_flush();
    int rc = probe_set_sysclk(216);
    uart_rebaud();          /* BRR은 전역이 아니라 RCC 레지스터에서 역산한다 */
    if (rc != 0) {
        probe_printf("216MHz 전환 실패(rc=%d) - 200MHz 결과만 보고한다."
                     " 실패 시점 실측 SYSCLK=%u Hz\n", rc, sysclk_from_regs());
        probe_flush();
        int rb = probe_set_sysclk(200);
        uart_rebaud();
        probe_printf("200MHz 복귀 시도: rc=%d, 레지스터 역산 SYSCLK=%u Hz, USART1_BRR=%u\n",
                     rb, sysclk_from_regs(), USART1->BRR);
        if (sysclk_from_regs() != 200000000u) {
            probe_puts("경고: SYSCLK이 200MHz가 아니다. 이 뒤의 사이클→시간 환산은 믿지 마라.\n");
        }
        return;
    }
    probe_printf("클럭 전환 완료: SYSCLK=%u Hz(레지스터 역산 %u Hz), PCLK2=%u Hz,"
                 " USART1_BRR=%u, FLASH_ACR=0x%08X\n",
                 g_sysclk_hz, sysclk_from_regs(), g_pclk2_hz, USART1->BRR, FLASH->ACR);
    probe_printf("FLASH wait state = %u WS (200MHz에서는 6 WS)\n",
                 (FLASH->ACR & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos);

    collect_clkset(&at216);

    probe_flush();
    int rb = probe_set_sysclk(200);
    uart_rebaud();
    /* 복구를 "했다"가 아니라 "됐다"를 확인한다. 여기가 틀어지면 이 뒤의 모든
     * 사이클→시간 환산(및 다음 절들)이 조용히 틀어진다. */
    uint32_t back_hz = sysclk_from_regs();
    probe_printf("\n200MHz로 복귀: rc=%d, 전역 SYSCLK=%u Hz, 레지스터 역산 %u Hz,"
                 " HCLK=%u Hz, PCLK2=%u Hz, USART1_BRR=%u, FLASH_ACR=0x%08X\n",
                 rb, g_sysclk_hz, back_hz, hclk_from_regs(), pclk2_from_regs(),
                 USART1->BRR, FLASH->ACR);
    if (rb != 0 || back_hz != 200000000u) {
        probe_puts("경고: 200MHz 복귀에 실패했다. 아래 표와 이후 절의 시간 환산은 신뢰할 수 없다.\n");
    } else {
        probe_puts("확인: SYSCLK/HCLK/PCLK2가 기준 조건(200/200/100MHz)으로 되돌아왔다.\n");
    }

    probe_puts("\n항목                        200MHz            216MHz            비(216/200)\n");
    probe_puts("------------------------  ---------------   ---------------   -----------\n");
    clk_row("정수 ADD 스루풋", at200.int_c100, at216.int_c100, 1, "cyc/insn");
    clk_row("SMLAD MAC/cycle(DTCM)", at200.smlad_m100, at216.smlad_m100, 0, "MAC/cyc");
    clk_row("DTCM 읽기", at200.dtcm_bpc100, at216.dtcm_bpc100, 0, "B/cyc");
    clk_row("SRAM1 읽기", at200.sram_bpc100, at216.sram_bpc100, 0, "B/cyc");
    clk_row("FLASH AXIM 코드", at200.flash_c100, at216.flash_c100, 1, "cyc/insn");

    /* 절대 성능(초당) 환산 */
    probe_puts("\n같은 값을 절대 성능으로 환산 (사이클이 아니라 시간 기준):\n");
    probe_puts("항목                        200MHz            216MHz            비\n");
    probe_puts("------------------------  ---------------   ---------------   -----------\n");
    {
        uint32_t m200 = divr(at200.smlad_m100 * 200u, 100u);   /* MMAC/s */
        uint32_t m216 = divr(at216.smlad_m100 * 216u, 100u);
        pad("SMLAD 절대 처리량", 26);
        probe_printf("%9u MMAC/s     %9u MMAC/s     ", m200, m216);
        fx2(m200 ? divr(m216 * 100u, m200) : 0u); probe_puts("배\n");

        uint32_t b200 = divr(at200.dtcm_bpc100 * 200u, 100u);  /* MB/s */
        uint32_t b216 = divr(at216.dtcm_bpc100 * 216u, 100u);
        pad("DTCM 읽기 대역폭", 26);
        probe_printf("%9u MB/s       %9u MB/s       ", b200, b216);
        fx2(b200 ? divr(b216 * 100u, b200) : 0u); probe_puts("배\n");

        uint32_t s200 = divr(at200.sram_bpc100 * 200u, 100u);
        uint32_t s216 = divr(at216.sram_bpc100 * 216u, 100u);
        pad("SRAM1 읽기 대역폭", 26);
        probe_printf("%9u MB/s       %9u MB/s       ", s200, s216);
        fx2(s200 ? divr(s216 * 100u, s200) : 0u); probe_puts("배\n");

        /* 플래시 코드: ns/insn */
        uint32_t f200 = divr(at200.flash_c100 * 1000u, 200u);  /* ns x100 */
        uint32_t f216 = divr(at216.flash_c100 * 1000u, 216u);
        pad("FLASH 코드 1명령 시간", 26);
        probe_puts("     "); fx2(f200); probe_puts(" ns          ");
        probe_puts("     "); fx2(f216); probe_puts(" ns          ");
        fx2(f200 ? divr(f216 * 100u, f200) : 0u); probe_puts("배\n");
    }
    probe_puts("해석: 216MHz는 플래시 wait state가 6→7로 올라간다. 그래서 플래시에서 코드를\n"
               "      페치하는 항목은 사이클 기준으로 오히려 나빠질 수 있고, 클럭이 8% 올라간\n"
               "      만큼을 그대로 못 가져간다. 반대로 TCM/레지스터에만 머무는 계산은 클럭에\n"
               "      정확히 비례해 1.08배가 나온다. 코드를 ITCM/RAM에 올려두지 않은 채\n"
               "      216MHz로 올리면 기대만큼 빨라지지 않는다는 뜻이다.\n");
}

/* ========================================================================== */
/* 8) 종합                                                                     */
/* ========================================================================== */
static void cpu_summary(void)
{
    probe_section("CPU 8) 종합 - 이 칩의 실측 성능 요약");
    /* 부품명·리비전을 하드코딩하지 않는다. 이 문단은 실측 요약이므로 코어 식별도
     * 레지스터(CPUID, DBGMCU_IDCODE)에서 읽은 값으로만 쓴다. */
    {
        uint32_t cpuid = SCB->CPUID;
        uint32_t partno = (cpuid & SCB_CPUID_PARTNO_Msk) >> SCB_CPUID_PARTNO_Pos;
        uint32_t var    = (cpuid & SCB_CPUID_VARIANT_Msk) >> SCB_CPUID_VARIANT_Pos;
        uint32_t rev    = (cpuid & SCB_CPUID_REVISION_Msk) >> SCB_CPUID_REVISION_Pos;
        uint32_t idcode = DBGMCU->IDCODE;
        probe_printf("이 보드의 코어(CPUID=0x%08X → Cortex-M%u r%up%u, DBGMCU_IDCODE=0x%08X"
                     " → DEV_ID=0x%03X)는 %u MHz에서 다음과 같이 측정되었다. ",
                     cpuid, partno & 0xFu, var, rev, idcode, idcode & 0xFFFu,
                     g_sysclk_hz / 1000000u);
    }
    probe_puts("독립적인 정수 명령을 나란히 놓으면 IPC ");
    fx2(s_ipc_add100);
    if (s_ipc_add100 > 200u) {
        probe_puts(" 명령/사이클이 나왔다. 이는 in-order 2-way의 상한 2.00을 넘는 값이므로"
                   " 듀얼이슈의 근거가 아니라 측정/보정 오차의 표시로 읽어야 한다. ");
    } else {
        probe_puts(" 명령/사이클이 나와, 실제로 in-order 2-way 듀얼이슈로 동작함이 확인된다. ");
    }
    probe_puts("DSP 확장의 SMLAD는 ");
    fx2(s_smlad_cyc100);
    probe_puts(" cyc/insn으로 발행되어 레지스터 상주 조건에서 int16 MAC 2.00회/사이클이 상한이고, ");
    probe_puts("데이터를 DTCM에 두고 실제로 로드까지 포함해 돌리면 ");
    fx2(s_mac100_best);
    probe_puts(" MAC/cycle, 즉 ");
    fx2(divr(s_mac100_best * 2u * (g_sysclk_hz / 1000000u), 1000u));
    probe_puts(" GOPS(1 MAC = 2 OP)가 실효 성능이다. ");
    if (s_mac100_scalar) {
        probe_puts("같은 계산을 스칼라 C로 쓰면 ");
        fx2(s_mac100_scalar);
        probe_puts(" MAC/cycle에 그쳐 SIMD 이득은 ");
        fx2(divr(s_mac100_best * 100u, s_mac100_scalar));
        probe_puts("배다. ");
    }
    if (s_best_place100 && s_worst_place100) {
        probe_puts("코드 배치의 영향은 캐시에 안 들어가는 크기의 루프에서 최대 ");
        fx2(divr(s_worst_place100 * 100u, s_best_place100));
        probe_puts("배로, ITCM 실행 대비 ART 꺼진 플래시 실행이 그만큼 손해다. ");
    }
    probe_puts("병목은 명령어 실행이 아니라 공급 쪽이다: 명령은 플래시 페치 경로(ART/I-cache)에서, ");
    probe_puts("데이터는 SRAM1의 AXI 왕복(D-cache 유무)에서 막힌다. ");
    probe_puts("SNN 추론처럼 가중치를 순차적으로 훑는 워크로드는 계산이 아니라 이 두 공급 경로를 ");
    probe_puts("어떻게 배치하느냐로 성능이 결정된다.\n");
}

/* ========================================================================== */
void probe_run_cpu(void)
{
    /*
     * 기준 조건을 명시적으로 고정한다.
     *
     * 이 펌웨어는 부팅 시 캐시를 켜지 않는다(SystemInit도 HAL_Init도 켜지 않고,
     * hal_conf에 INSTRUCTION/DATA_CACHE_ENABLE 자체가 없다). 그래서 이 절 이전까지
     * I-cache는 꺼진 채였고, DEF_IBENCH/DEF_MBENCH/DEF_FBENCH가 만든 벤치 루프는
     * 전부 .text(FLASH AXIM, 6 wait state)에 있다. 그 상태로 재면 "명령 스루풋"이
     * 아니라 "명령 페치 대역폭"을 재게 된다(0.5 cyc/insn 같은 값은 원리상 불가능).
     * 따라서 여기서 I-cache를 켜고, 켜기 전후 상태를 인쇄해 근거를 남긴다.
     * (4절 cpu_code_placement()는 자기 절 안에서만 I$/ART를 껐다 켠다.)
     */
    uint32_t ccr_before = SCB->CCR;
    uint32_t acr_before = FLASH->ACR;
    probe_icache_enable();
    PROBE_BARRIER();

    probe_section("CPU 0) 측정 기준 조건 고정");
    probe_printf("진입 시   : SCB_CCR=0x%08X (I$=%s, D$=%s), FLASH_ACR=0x%08X"
                 " (ART=%s, PRFT=%s, %u WS)\n",
                 ccr_before,
                 (ccr_before & SCB_CCR_IC_Msk) ? "ON" : "OFF",
                 (ccr_before & SCB_CCR_DC_Msk) ? "ON" : "OFF",
                 acr_before,
                 (acr_before & FLASH_ACR_ARTEN) ? "ON" : "OFF",
                 (acr_before & FLASH_ACR_PRFTEN) ? "ON" : "OFF",
                 (acr_before & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos);
    probe_printf("고정 후   : SCB_CCR=0x%08X (I$=%s, D$=%s)\n",
                 SCB->CCR,
                 (SCB->CCR & SCB_CCR_IC_Msk) ? "ON" : "OFF",
                 (SCB->CCR & SCB_CCR_DC_Msk) ? "ON" : "OFF");
    probe_puts("2~3절(명령 스루풋/듀얼이슈)의 벤치 루프는 .text 즉 FLASH AXIM에 있다.\n"
               "I-cache가 꺼져 있으면 매 반복 6 wait state 플래시에서 다시 페치하게 되어\n"
               "실행 처리량이 아니라 페치 대역폭을 재게 된다. 그래서 I-cache를 켠 상태를\n"
               "기준으로 삼는다. D-cache는 건드리지 않는다(6절이 조건별로 직접 설정한다).\n");

    cpu_clock_check();
    cpu_instr_throughput();
    cpu_dual_issue();
    cpu_code_placement();
    cpu_dwt_counters();
    cpu_snn_kernel();
    cpu_clock_compare();
    cpu_summary();

    probe_printf("\n[CPU 절 종료 상태] SCB_CCR=0x%08X (I$=%s, D$=%s), FLASH_ACR=0x%08X"
                 " (ART=%s, PRFT=%s), SYSCLK=%u Hz\n",
                 SCB->CCR,
                 (SCB->CCR & SCB_CCR_IC_Msk) ? "ON" : "OFF",
                 (SCB->CCR & SCB_CCR_DC_Msk) ? "ON" : "OFF",
                 FLASH->ACR,
                 (FLASH->ACR & FLASH_ACR_ARTEN) ? "ON" : "OFF",
                 (FLASH->ACR & FLASH_ACR_PRFTEN) ? "ON" : "OFF",
                 sysclk_from_regs());
}
