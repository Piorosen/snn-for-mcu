/*
 * probe_regs.c - 코어/시스템/페리페럴 레지스터 덤프 및 해석.
 *
 * 이 모듈의 원칙: "칩이 자기 자신에 대해 말하게 한다."
 * 데이터시트가 아니라 실리콘이 내놓는 값을 그대로 찍고, 그 옆에 해석을 붙인다.
 * 해석이 실측과 어긋날 수 있는 항목(TCM SZ 등)은 어긋난다는 사실 자체를 출력한다.
 *
 * 주의 1) probe_printf는 '-'(좌측 정렬) 플래그를 지원하지 않는다. probe_io.c의
 *         probe_kv32()가 "%-16s"를 쓰고 있어 현재 구현으로는 인자 정렬이 깨진다.
 *         (probe_io.c는 다른 담당자의 파일이라 고치지 않고) 같은 모양의 출력을
 *         내는 로컬 kv()를 두어 우회한다. probe_io.c가 고쳐져도 이 파일은 무관하다.
 * 주의 2) 페리페럴 존재 실측은 BusFault를 일부러 유발한다. 벡터 테이블을 SRAM으로
 *         복사해 BusFault 엔트리만 바꿔치기하고, 핸들러에서 스택된 PC를 4바이트
 *         전진시켜 복귀한다. 프로빙용 읽기는 반드시 4바이트 인코딩(ldr.w)이어야 한다.
 */
#include "probe.h"
#include "stm32f7xx.h"

/* ------------------------------------------------------------------------- */
/* 공통 헬퍼                                                                   */
/* ------------------------------------------------------------------------- */

#define FLD(v, pos, w)  (((uint32_t)(v) >> (pos)) & ((1u << (w)) - 1u))
#define BIT_(v, pos)    (((uint32_t)(v) >> (pos)) & 1u)

/* 이름을 yn()으로 두면 <math.h>의 내장 함수 yn(int,double)과 충돌한다 */
static const char *yes_no(uint32_t b) { return b ? "예" : "아니오"; }
static const char *onoff(uint32_t b){ return b ? "ON" : "OFF"; }

/* 이 루프를 그냥 두면 GCC가 strlen() 호출로 치환한다. 그런데 이 프로젝트의 링커
 * 스크립트는 libc.a를 DISCARD하므로 libc 심볼에 의존하지 않는 편이 안전하다.
 * 빈 asm 배리어로 관용구 인식을 막는다. */
static int str_len(const char *s)
{
    int n = 0;
    while (s[n]) { n++; __asm volatile ("" ::: "memory"); }
    return n;
}

static void pad_to(const char *s, int width)
{
    probe_puts(s);
    for (int n = str_len(s); n < width; n++) probe_puts(" ");
}

/* probe_kv32와 같은 모양: "이름             @ 0xADDR = 0xVALUE  ; 설명" */
static void kv(const char *name, uint32_t addr, uint32_t value, const char *desc)
{
    pad_to(name, 16);
    probe_printf(" @ 0x%08X = 0x%08X", addr, value);
    if (desc && desc[0]) probe_printf("  ; %s", desc);
    probe_puts("\n");
}

/* ------------------------------------------------------------------------- */
/* BusFault 프로빙 기반 안전 읽기                                              */
/* ------------------------------------------------------------------------- */

static volatile int      s_probe_active;
static volatile uint32_t s_probe_faulted;
static volatile uint32_t s_probe_bfsr;      /* 마지막 폴트의 BFSR (CFSR[15:8]) */

/* VTOR은 테이블 크기를 올림한 2의 거듭제곱으로 정렬돼야 한다.
 * STM32F746은 16(시스템) + 98(IRQ) = 114 엔트리 → 512바이트 정렬로 충분하다. */
static uint32_t s_vtable[128] __attribute__((aligned(512)));
static uint32_t s_saved_vtor;
static uint32_t s_saved_shcsr;
static int      s_fault_ready;
static uint32_t s_fault_probe_count;
static uint32_t s_fault_hit_count;

/* naked 핸들러에서 b로 진입한다. static이면 심볼이 사라질 수 있어 외부 링키지 + used. */
__attribute__((used)) void probe_regs_busfault_c(uint32_t *frame);

void probe_regs_busfault_c(uint32_t *frame)
{
    uint32_t bfsr = FLD(SCB->CFSR, SCB_CFSR_BUSFAULTSR_Pos, 8);

    if (!s_probe_active) {
        /* 프로빙 구간 밖의 진짜 버스 폴트. 원인을 남기고 정지한다.
         * (probe_system.c의 fault_record는 static이라 호출할 수 없어 자체 기록) */
        volatile uint32_t *log = (volatile uint32_t *)PROBE_DIAG_REGS_LOG;
        log[0] = 0xBADF0000u;
        log[1] = SCB->CFSR;
        log[2] = SCB->BFAR;
        log[3] = frame[6];               /* 폴트를 낸 PC */
        log[4] = *(volatile uint32_t *)0xE000EFA8u;   /* ABFSR */
        __DSB();
        for (;;) { }
    }

    s_probe_faulted = 1;
    s_probe_bfsr    = bfsr;

    /* PRECISERR(bit1)면 스택된 PC가 폴트를 낸 명령을 가리킨다 → 4바이트(ldr.w) 건너뛴다.
     * IMPRECISERR(bit2)면 그 명령은 이미 리타이어했고 PC는 다음 명령이므로 건드리면 안 된다.
     * 둘 다 아닌 경우에도 전진시킨다. 전진하지 않으면 같은 명령으로 복귀해 무한 폴트에
     * 빠지기 때문이다 (행보다는 한 명령 건너뛰는 쪽이 안전하다). */
    if (!(bfsr & (1u << 2))) {
        frame[6] += 4u;
    }

    /* BFSR은 write-1-clear. BFARVALID까지 함께 지운다. */
    SCB->CFSR = SCB_CFSR_BUSFAULTSR_Msk;
    /* HardFault로 승격되어 들어온 경우 FORCED도 지워야 다음 폴트가 정상 처리된다. */
    SCB->HFSR = SCB_HFSR_FORCED_Msk;
    __DSB();
}

__attribute__((naked)) static void probe_regs_busfault(void)
{
    __asm volatile (
        "tst   lr, #4                 \n"
        "ite   eq                     \n"
        "mrseq r0, msp                \n"
        "mrsne r0, psp                \n"
        "b     probe_regs_busfault_c  \n"
    );
}

static void fault_probe_install(void)
{
    if (s_fault_ready) return;

    s_saved_vtor = SCB->VTOR;
    const uint32_t *src = (const uint32_t *)s_saved_vtor;
    for (int i = 0; i < 128; i++) s_vtable[i] = src[i];
    s_vtable[5] = (uint32_t)&probe_regs_busfault;   /* 예외번호 5 = BusFault */
    /* 안전망: BusFault가 어떤 이유로든 HardFault로 승격되어도 같은 핸들러가 받아
     * 복구하도록 예외번호 3(HardFault)도 함께 걸어 둔다. 프로빙 구간 밖에서 들어오면
     * s_probe_active가 0이라 기존과 동일하게 기록 후 정지한다. */
    s_vtable[3] = (uint32_t)&probe_regs_busfault;
    __DSB();
    if (SCB->CCR & SCB_CCR_DC_Msk) SCB_CleanDCache();

    SCB->VTOR = (uint32_t)s_vtable;
    __DSB(); __ISB();

    s_saved_shcsr = SCB->SHCSR;
    SCB->SHCSR |= SCB_SHCSR_BUSFAULTENA_Msk;   /* HardFault로 승격되지 않게 */
    SCB->CFSR   = SCB_CFSR_BUSFAULTSR_Msk;     /* 묵은 상태 클리어 */
    __DSB(); __ISB();
    s_fault_ready = 1;
}

static void fault_probe_remove(void)
{
    if (!s_fault_ready) return;
    SCB->SHCSR = s_saved_shcsr;
    SCB->VTOR  = s_saved_vtor;
    __DSB(); __ISB();
    s_fault_ready = 0;
}

/* 주소 하나를 읽어 본다. 반환 1 = 정상, 0 = BusFault.
 * 반드시 4바이트 인코딩(ldr.w)으로 읽는다.
 *
 * 여기서 절대로 PRIMASK를 세우면 안 된다. PRIMASK=1은 우선순위 0 이상의 모든 예외를
 * 막는데, BusFault의 기본 우선순위가 0이라 함께 막힌다. 그러면 폴트가 우리 핸들러
 * 대신 HardFault로 승격되어 복구 지점을 잃는다. (실제로 이 보드에서 JPEG 주소
 * 0x50051000을 읽다가 그렇게 멈추는 것을 확인했다.)
 * 프로빙은 타이밍 측정이 아니므로 인터럽트가 끼어들어도 결과에 영향이 없다. */
static int safe_read32(uint32_t addr, uint32_t *out)
{
    uint32_t v = 0;
    fault_probe_install();

    s_probe_faulted = 0;
    s_probe_active  = 1;
    PROBE_BARRIER();

    __asm volatile ("ldr.w %0, [%1]" : "=r"(v) : "r"(addr) : "memory");

    PROBE_BARRIER();
    s_probe_active = 0;

    s_fault_probe_count++;
    if (s_probe_faulted) {
        s_fault_hit_count++;
        if (out) *out = 0xFFFFFFFFu;
        return 0;
    }
    if (out) *out = v;
    return 1;
}

static uint32_t read16(uint32_t addr) { return *(volatile uint16_t *)addr; }

/* ------------------------------------------------------------------------- */
/* 1. 코어 식별                                                               */
/* ------------------------------------------------------------------------- */

static const char *cpu_part_name(uint32_t part)
{
    switch (part) {
    case 0xC20: return "Cortex-M0";
    case 0xC21: return "Cortex-M1";
    case 0xC23: return "Cortex-M3";
    case 0xC24: return "Cortex-M4";
    case 0xC27: return "Cortex-M7";
    case 0xC60: return "Cortex-M0+";
    default:    return "미상";
    }
}

/* ARMv7-M ARM(DDI0403) B4.1의 ID_ISAR 필드 정의를 따른다. */
static void decode_isar(const uint32_t *isar)
{
    uint32_t div    = FLD(isar[0], 24, 4);   /* ID_ISAR0.Divide_instrs   */
    uint32_t dbg    = FLD(isar[0], 20, 4);   /* Debug_instrs             */
    uint32_t cop    = FLD(isar[0], 16, 4);   /* Coproc_instrs            */
    uint32_t cmpbr  = FLD(isar[0], 12, 4);   /* CmpBranch_instrs         */
    uint32_t bfield = FLD(isar[0],  8, 4);   /* Bitfield_instrs          */
    uint32_t bcount = FLD(isar[0],  4, 4);   /* BitCount_instrs          */

    uint32_t itw    = FLD(isar[1], 24, 4);   /* Interwork_instrs         */
    uint32_t imm    = FLD(isar[1], 20, 4);   /* Immediate_instrs         */
    uint32_t ifthen = FLD(isar[1], 16, 4);   /* IfThen_instrs            */
    uint32_t ext    = FLD(isar[1], 12, 4);   /* Extend_instrs            */

    uint32_t rev    = FLD(isar[2], 28, 4);   /* Reversal_instrs          */
    uint32_t multu  = FLD(isar[2], 20, 4);   /* MultU_instrs             */
    uint32_t mults  = FLD(isar[2], 16, 4);   /* MultS_instrs             */
    uint32_t mult   = FLD(isar[2], 12, 4);   /* Mult_instrs              */
    uint32_t mai    = FLD(isar[2],  8, 4);   /* MultiAccessInt_instrs    */
    uint32_t mhint  = FLD(isar[2],  4, 4);   /* MemHint_instrs           */
    uint32_t ldst   = FLD(isar[2],  0, 4);   /* LoadStore_instrs         */

    uint32_t tnop   = FLD(isar[3], 24, 4);   /* TrueNOP_instrs           */
    uint32_t t32cp  = FLD(isar[3], 20, 4);   /* T32Copy_instrs           */
    uint32_t tabbr  = FLD(isar[3], 16, 4);   /* TabBranch_instrs         */
    uint32_t sync   = FLD(isar[3], 12, 4);   /* SynchPrim_instrs         */
    uint32_t svc    = FLD(isar[3],  8, 4);   /* SVC_instrs               */
    uint32_t simd   = FLD(isar[3],  4, 4);   /* SIMD_instrs              */
    uint32_t sat    = FLD(isar[3],  0, 4);   /* Saturate_instrs          */

    uint32_t psrm   = FLD(isar[4], 24, 4);   /* PSR_M_instrs             */
    uint32_t syncf  = FLD(isar[4], 20, 4);   /* SynchPrim_instrs_frac    */
    uint32_t barr   = FLD(isar[4], 16, 4);   /* Barrier_instrs           */
    uint32_t wback  = FLD(isar[4], 12, 4);   /* Writeback_instrs         */
    uint32_t wshift = FLD(isar[4],  8, 4);   /* WithShifts_instrs        */
    uint32_t unpriv = FLD(isar[4],  4, 4);   /* Unpriv_instrs            */

    probe_printf("    ID_ISAR0 Divide=%u Debug=%u Coproc=%u CmpBranch=%u Bitfield=%u BitCount=%u\n",
                 div, dbg, cop, cmpbr, bfield, bcount);
    probe_printf("    ID_ISAR1 Interwork=%u Immediate=%u IfThen=%u Extend=%u\n",
                 itw, imm, ifthen, ext);
    probe_printf("    ID_ISAR2 Reversal=%u MultU=%u MultS=%u Mult=%u MultiAccessInt=%u MemHint=%u LoadStore=%u\n",
                 rev, multu, mults, mult, mai, mhint, ldst);
    probe_printf("    ID_ISAR3 TrueNOP=%u T32Copy=%u TabBranch=%u SynchPrim=%u SVC=%u SIMD=%u Saturate=%u\n",
                 tnop, t32cp, tabbr, sync, svc, simd, sat);
    probe_printf("    ID_ISAR4 PSR_M=%u SynchPrimFrac=%u Barrier=%u Writeback=%u WithShifts=%u Unpriv=%u\n",
                 psrm, syncf, barr, wback, wshift, unpriv);

    probe_puts("  [명령어 지원 판정]\n");
    probe_printf("    하드웨어 나눗셈(SDIV/UDIV): %s (Divide_instrs=%u)\n",
                 div >= 1 ? "지원" : "없음", div);
    probe_printf("    포화 연산(SSAT/USAT): %s / QADD·QSUB류: %s (Saturate_instrs=%u)\n",
                 simd >= 1 ? "지원" : "없음", sat >= 1 ? "지원" : "없음", sat);
    probe_printf("    패킹 SIMD(DSP 확장, SADD16/SMLAD/PKHBT 등): %s (SIMD_instrs=%u)\n",
                 simd >= 3 ? "지원 → ARMv7E-M, CMSIS-DSP/CMSIS-NN의 __SIMD32 경로 사용 가능"
                           : (simd >= 1 ? "SSAT/USAT만 (패킹 SIMD 없음)" : "없음"), simd);
    probe_printf("    부호있는 곱셈 확장(SMLABB/SMLAD 등): %s (MultS_instrs=%u)\n",
                 mults >= 3 ? "SMLAD/SMUAD까지 지원" : (mults >= 2 ? "SMLAxy까지" :
                 (mults >= 1 ? "SMULL/SMLAL만" : "없음")), mults);
    probe_printf("    무부호 곱셈 확장(UMAAL): %s (MultU_instrs=%u)\n",
                 multu >= 2 ? "지원" : "없음", multu);
    probe_printf("    배타적 접근(LDREX/STREX): %s, CLREX/바이트·하프워드 변형: %s\n",
                 sync >= 1 ? "지원" : "없음", syncf >= 1 ? "지원" : "없음");
    probe_printf("    메모리 배리어(DSB/DMB/ISB): %s / 코프로세서 명령: %s\n",
                 barr >= 1 ? "지원" : "없음", cop >= 1 ? "지원" : "없음");
}

static void sec_core_id(void)
{
    probe_section("1. 코어 식별 (CPUID / 피처 레지스터)");

    uint32_t cpuid = SCB->CPUID;
    kv("CPUID", 0xE000ED00u, cpuid, NULL);
    uint32_t impl = FLD(cpuid, 24, 8);
    uint32_t var  = FLD(cpuid, 20, 4);
    uint32_t arch = FLD(cpuid, 16, 4);
    uint32_t part = FLD(cpuid,  4, 12);
    uint32_t rev  = FLD(cpuid,  0, 4);
    probe_printf("    Implementer=0x%02X (%s)  Architecture=0x%X (%s)\n",
                 impl, impl == 0x41u ? "ARM Limited" : "미상",
                 arch, arch == 0xFu ? "ARMv7-M 이상 (CPUID로는 구분 불가)" : "미상");
    probe_printf("    PartNo=0x%03X (%s)  Variant=%u  Revision=%u\n",
                 part, cpu_part_name(part), var, rev);
    probe_printf("    => %s r%up%u  ; 이 실리콘의 코어 리비전\n",
                 cpu_part_name(part), var, rev);

    probe_puts("\n  [ARMv7-M 피처 레지스터 원시값]\n");
    uint32_t pfr0 = SCB->ID_PFR[0], pfr1 = SCB->ID_PFR[1];
    kv("ID_PFR0", 0xE000ED40u, pfr0, NULL);
    kv("ID_PFR1", 0xE000ED44u, pfr1, NULL);
    kv("ID_DFR0", 0xE000ED48u, SCB->ID_DFR, NULL);
    kv("ID_AFR0", 0xE000ED4Cu, SCB->ID_AFR, NULL);
    kv("ID_MMFR0", 0xE000ED50u, SCB->ID_MFR[0], NULL);
    kv("ID_MMFR1", 0xE000ED54u, SCB->ID_MFR[1], NULL);
    kv("ID_MMFR2", 0xE000ED58u, SCB->ID_MFR[2], NULL);
    kv("ID_MMFR3", 0xE000ED5Cu, SCB->ID_MFR[3], NULL);

    uint32_t isar[5];
    for (int i = 0; i < 5; i++) isar[i] = SCB->ID_ISAR[i];
    kv("ID_ISAR0", 0xE000ED60u, isar[0], NULL);
    kv("ID_ISAR1", 0xE000ED64u, isar[1], NULL);
    kv("ID_ISAR2", 0xE000ED68u, isar[2], NULL);
    kv("ID_ISAR3", 0xE000ED6Cu, isar[3], NULL);
    kv("ID_ISAR4", 0xE000ED70u, isar[4], NULL);

    probe_printf("    ID_PFR0 State0(A32)=%u State1(T32)=%u  ; T32=3 이면 Thumb-2 전용 프로파일\n",
                 FLD(pfr0, 0, 4), FLD(pfr0, 4, 4));
    probe_printf("    ID_PFR1 M-profile programmers' model=%u  ; 2 = ARMv7-M 예외 모델\n",
                 FLD(pfr1, 8, 4));
    probe_printf("    ID_DFR0 M-profile debug model=%u  ; 1 = 브레이크포인트/워치포인트 기반 디버그\n",
                 FLD(SCB->ID_DFR, 20, 4));
    probe_printf("    ID_MMFR0 PMSA 지원=%u (2=PMSAv6/v7 MPU), Aux control regs=%u\n",
                 FLD(SCB->ID_MFR[0], 4, 4), FLD(SCB->ID_MFR[0], 20, 4));
    probe_printf("    ID_MMFR2 WFI/WFE 등 Mem barrier=%u  ; 캐시 유지 명령 계열 존재 표시\n",
                 FLD(SCB->ID_MFR[2], 24, 4));

    probe_puts("\n");
    decode_isar(isar);

    probe_puts("\n  [인터럽트 컨트롤러]\n");
    uint32_t ictr = SCnSCB->ICTR;
    kv("ICTR", 0xE000E004u, ictr, NULL);
    uint32_t lines = (FLD(ictr, 0, 4) + 1u) * 32u;
    probe_printf("    INTLINESNUM=%u → NVIC가 지원하는 인터럽트 라인 = %u개\n",
                 FLD(ictr, 0, 4), lines);
    probe_printf("    (STM32F746의 실제 사용 IRQ는 98개. 나머지 %u개는 구현만 되어 있고 미배선)\n",
                 lines > 98u ? lines - 98u : 0u);

    /* NVIC 우선순위 비트 실측: IP[0]에 0xFF를 쓰고 되읽는다 */
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    uint8_t saved_ip = NVIC->IP[0];
    NVIC->IP[0] = 0xFFu;
    __DSB();
    uint8_t readback = NVIC->IP[0];
    NVIC->IP[0] = saved_ip;
    __DSB();
    __set_PRIMASK(pm);
    int prio_bits = 0;
    for (int b = 7; b >= 0; b--) {
        if (readback & (1u << b)) prio_bits++;
        else break;
    }
    probe_printf("    NVIC->IP[0]에 0xFF 기록 후 되읽기 = 0x%02X → 구현된 우선순위 비트 = %u개",
                 (uint32_t)readback, (uint32_t)prio_bits);
    probe_printf(" (%u단계)\n", prio_bits ? (1u << prio_bits) : 1u);
    probe_printf("    ; 상위 %u비트만 하드웨어에 존재하고 하위 %u비트는 0으로 읽힌다\n",
                 (uint32_t)prio_bits, (uint32_t)(8 - prio_bits));

    kv("AIRCR", 0xE000ED0Cu, SCB->AIRCR, NULL);
    {
        uint32_t prigroup = FLD(SCB->AIRCR, 8, 3);
        int grp = (int)(7u - prigroup);
        if (grp > prio_bits) grp = prio_bits;
        probe_printf("    PRIGROUP=%u → 선점 우선순위 %d비트 / 서브 우선순위 %d비트\n",
                     prigroup, grp, prio_bits - grp);
    }
    kv("VTOR", 0xE000ED08u, SCB->VTOR, "현재 벡터 테이블 위치");
}

/* ------------------------------------------------------------------------- */
/* 2. 캐시                                                                    */
/* ------------------------------------------------------------------------- */

static void sec_cache(void)
{
    probe_section("2. 캐시 (CLIDR / CTR / CCSIDR)");

    uint32_t clidr = SCB->CLIDR;
    uint32_t ctr   = SCB->CTR;
    kv("CLIDR", 0xE000ED78u, clidr, NULL);
    uint32_t ctype1 = FLD(clidr, 0, 3);
    const char *ct = (ctype1 == 0) ? "캐시 없음" :
                     (ctype1 == 1) ? "I-cache만" :
                     (ctype1 == 2) ? "D-cache만" :
                     (ctype1 == 3) ? "I/D 분리 캐시 (하버드)" :
                     (ctype1 == 4) ? "통합 캐시" : "미상";
    probe_printf("    Ctype1=%u (%s)  LoUIS=%u  LoC=%u  LoUU=%u\n",
                 ctype1, ct, FLD(clidr, 21, 3), FLD(clidr, 24, 3), FLD(clidr, 27, 3));
    probe_printf("    ; LoC=1 → 코히런시 유지가 필요한 캐시 레벨은 L1뿐 (L2 없음)\n");

    kv("CTR", 0xE000ED7Cu, ctr, NULL);
    uint32_t imin = 4u << FLD(ctr, 0, 4);
    uint32_t dmin = 4u << FLD(ctr, 16, 4);
    uint32_t erg  = 4u << FLD(ctr, 20, 4);
    uint32_t cwg  = 4u << FLD(ctr, 24, 4);
    probe_printf("    IminLine=%u bytes, DminLine=%u bytes  ; 캐시 유지 명령의 최소 단위\n",
                 imin, dmin);
    probe_printf("    ERG=%u bytes (배타적 예약 입도), CWG=%u bytes (쓰기 병합 입도)\n", erg, cwg);
    probe_printf("    L1Ip=%u  ; 0b11 = PIPT (물리 인덱스/물리 태그, 앨리어싱 없음)\n",
                 FLD(ctr, 14, 2));

    /* CSSELR을 바꿔가며 CCSIDR을 읽는다. 원래 값은 복원한다. */
    uint32_t saved_csselr = SCB->CSSELR;
    for (int ind = 0; ind < 2; ind++) {
        SCB->CSSELR = (uint32_t)ind;     /* Level=0(L1), InD: 0=D, 1=I */
        __DSB(); __ISB();
        uint32_t ccsidr = SCB->CCSIDR;
        uint32_t sets   = FLD(ccsidr, 13, 15) + 1u;
        uint32_t assoc  = FLD(ccsidr, 3, 10) + 1u;
        uint32_t lsz    = 1u << (FLD(ccsidr, 0, 3) + 4u);   /* 워드수 로그 → 바이트 */
        uint32_t total  = sets * assoc * lsz;
        probe_printf("\n  [%s-cache]  (CSSELR=0x%08X)\n", ind ? "I" : "D", (uint32_t)ind);
        kv("CCSIDR", 0xE000ED80u, ccsidr, NULL);
        probe_printf("    NumSets=%u  Associativity=%u-way  LineSize=%u bytes\n",
                     sets, assoc, lsz);
        probe_printf("    → 용량 = %u x %u x %u = %u bytes (%u KB)\n",
                     sets, assoc, lsz, total, total / 1024u);
        probe_printf("    지원 정책: WT=%s WB=%s ReadAlloc=%s WriteAlloc=%s\n",
                     yes_no(BIT_(ccsidr, 31)), yes_no(BIT_(ccsidr, 30)),
                     yes_no(BIT_(ccsidr, 29)), yes_no(BIT_(ccsidr, 28)));
        probe_printf("    ; 한 way = %u bytes, 인덱스 비트 %u개 → 같은 세트로 충돌하는 주소 간격 %u bytes\n",
                     sets * lsz, sets, sets * lsz);
    }
    SCB->CSSELR = saved_csselr;
    __DSB(); __ISB();

    probe_puts("\n  [캐시 제어 레지스터 현재 상태]\n");
    uint32_t ccr = SCB->CCR;
    kv("CCR", 0xE000ED14u, ccr, NULL);
    probe_printf("    IC(I-cache)=%s  DC(D-cache)=%s  BP(분기예측)=%s\n",
                 onoff(BIT_(ccr, 17)), onoff(BIT_(ccr, 16)), onoff(BIT_(ccr, 18)));
    probe_printf("    STKALIGN=%s DIV_0_TRP=%s UNALIGN_TRP=%s USERSETMPEND=%s NONBASETHRDENA=%s\n",
                 onoff(BIT_(ccr, 9)), onoff(BIT_(ccr, 4)), onoff(BIT_(ccr, 3)),
                 onoff(BIT_(ccr, 1)), onoff(BIT_(ccr, 0)));
    probe_printf("    ; 이 값은 probe_run_registers() 시점, 즉 벤치마크가 캐시를 만지기 전의 상태다\n");

    uint32_t cacr = SCB->CACR;
    kv("CACR", 0xE000EF9Cu, cacr, NULL);
    probe_printf("    SIWT=%s (%s)  ECCEN=%s  FORCEWT=%s\n",
                 onoff(BIT_(cacr, 0)),
                 BIT_(cacr, 0) ? "공유 영역을 write-through로 강제" : "공유 영역도 write-back 허용",
                 onoff(BIT_(cacr, 1)), onoff(BIT_(cacr, 2)));
    probe_printf("    ; ECCEN=0이면 TCM/캐시 ECC 비활성. STM32F7은 ECC 로직 자체가 없어 항상 0으로 읽힌다\n");

    uint32_t actlr = SCnSCB->ACTLR;
    kv("ACTLR", 0xE000E008u, actlr, NULL);
    probe_printf("    DISFOLD(bit0)=%s  ; IT 폴딩 비활성화\n", onoff(BIT_(actlr, 0)));
    probe_printf("    FPEXCODIS(bit2)=%s  ; FPU 예외 출력 비활성화\n", onoff(BIT_(actlr, 2)));
    probe_printf("    DISRAMODE(bit3)=%s  ; write-through 동적 read-allocate 비활성화\n", onoff(BIT_(actlr, 3)));
    probe_printf("    DISITMATBFLUSH(bit4)=%s ; ITM/DWT ATB 플러시 비활성화\n", onoff(BIT_(actlr, 4)));
    probe_printf("    나머지 비트[31:5]=0x%08X  ; 이중발행 억제(DISDI 계열)는 코어 리비전 의존 필드다\n",
                 actlr & 0xFFFFFFE0u);
    probe_printf("    ; DISDEFWBUF/DISMCYCINT는 Cortex-M3/M4의 ACTLR 비트이고 Cortex-M7에는 없다\n");
}

/* ------------------------------------------------------------------------- */
/* 3. TCM / 버스                                                              */
/* ------------------------------------------------------------------------- */

/* Cortex-M7 TRM의 TCM SZ 인코딩: 0=없음, 3=4KB, 이후 한 칸마다 2배 */
static uint32_t tcm_sz_kb(uint32_t sz)
{
    if (sz < 3u) return 0u;
    return 1u << (sz - 1u);      /* sz=3 → 4KB, sz=7 → 64KB */
}

static void sec_tcm_bus(void)
{
    probe_section("3. TCM / 버스 인터페이스");

    uint32_t itcmcr = SCB->ITCMCR;
    uint32_t dtcmcr = SCB->DTCMCR;
    kv("ITCMCR", 0xE000EF90u, itcmcr, NULL);
    probe_printf("    EN=%s RMW=%s RETEN=%s SZ=0x%X → %u KB\n",
                 onoff(BIT_(itcmcr, 0)), onoff(BIT_(itcmcr, 1)), onoff(BIT_(itcmcr, 2)),
                 FLD(itcmcr, 3, 4), tcm_sz_kb(FLD(itcmcr, 3, 4)));
    kv("DTCMCR", 0xE000EF94u, dtcmcr, NULL);
    probe_printf("    EN=%s RMW=%s RETEN=%s SZ=0x%X → %u KB\n",
                 onoff(BIT_(dtcmcr, 0)), onoff(BIT_(dtcmcr, 1)), onoff(BIT_(dtcmcr, 2)),
                 FLD(dtcmcr, 3, 4), tcm_sz_kb(FLD(dtcmcr, 3, 4)));
    probe_puts("    ; SZ는 코어가 TCM 인터페이스에 할당한 주소공간 크기이지, 실제로 붙어 있는\n");
    probe_puts("    ; RAM 용량이 아니다. ST가 그 주소공간의 일부에만 SRAM을 물려 놓았다면\n");
    probe_puts("    ; 위 해석과 실제 크기가 어긋난다. 실제 크기는 probe_mem.c가 별도로 실측한다.\n");
    probe_puts("    ; (RM0385 기준 기대값: ITCM 16KB, DTCM 64KB)\n");
    probe_printf("    RMW=1은 32비트 미만 쓰기를 read-modify-write로 처리한다는 뜻이다.\n");

    uint32_t ahbpcr = SCB->AHBPCR;
    kv("AHBPCR", 0xE000EF98u, ahbpcr, NULL);
    uint32_t psz = FLD(ahbpcr, 1, 3);
    probe_printf("    EN(AHBP 슬레이브 포트)=%s  SZ=%u → %s\n",
                 onoff(BIT_(ahbpcr, 0)), psz,
                 psz == 0 ? "0 (비활성)" :
                 psz == 1 ? "64MB" : psz == 2 ? "128MB" : psz == 3 ? "256MB" :
                 psz == 4 ? "512MB" : "예약값");
    probe_printf("    ; AHBP는 페리페럴 전용 포트다. 켜져 있으면 페리페럴 접근이 AXIM을 거치지 않아\n");
    probe_printf("    ; 코드 페치와 경쟁하지 않는다 (STM32F7에서는 0x40000000~ 접근 경로)\n");

    uint32_t ahbscr = SCB->AHBSCR;
    kv("AHBSCR", 0xE000EFA0u, ahbscr, NULL);
    probe_printf("    CTL=%u (%s)  TPRI=%u  INITCOUNT=%u\n",
                 FLD(ahbscr, 0, 2),
                 FLD(ahbscr, 0, 2) == 0 ? "AHBS 중재: 기본(코어 우선)" : "소프트웨어 지정 중재",
                 FLD(ahbscr, 2, 9), FLD(ahbscr, 11, 5));

    uint32_t abfsr = SCB->ABFSR;
    kv("ABFSR", 0xE000EFA8u, abfsr, NULL);
    probe_printf("    ITCM=%u DTCM=%u AHBP=%u AXIM=%u EPPB=%u  AXIMTYPE=%u\n",
                 BIT_(abfsr, 0), BIT_(abfsr, 1), BIT_(abfsr, 2),
                 BIT_(abfsr, 3), BIT_(abfsr, 4), FLD(abfsr, 8, 2));
    probe_printf("    → %s\n", abfsr == 0 ? "부팅 이후 비동기 버스 폴트 없음"
                                          : "비동기 버스 폴트 이력 있음 (위 비트가 발생 인터페이스)");
}

/* ------------------------------------------------------------------------- */
/* 4. FPU                                                                     */
/* ------------------------------------------------------------------------- */

static void sec_fpu(void)
{
    probe_section("4. FPU (MVFR0/1/2 / CPACR / FPCCR)");

    uint32_t m0 = FPU->MVFR0, m1 = FPU->MVFR1, m2 = FPU->MVFR2;
    kv("MVFR0", 0xE000EF40u, m0, NULL);
    kv("MVFR1", 0xE000EF44u, m1, NULL);
    kv("MVFR2", 0xE000EF48u, m2, NULL);

    uint32_t simd_reg = FLD(m0, 0, 4);   /* A_SIMD registers */
    uint32_t sp       = FLD(m0, 4, 4);   /* Single precision */
    uint32_t dp       = FLD(m0, 8, 4);   /* Double precision */
    uint32_t trap     = FLD(m0, 12, 4);  /* FP exception trapping */
    uint32_t divide   = FLD(m0, 16, 4);
    uint32_t sqrt     = FLD(m0, 20, 4);
    uint32_t shortvec = FLD(m0, 24, 4);
    uint32_t rmode    = FLD(m0, 28, 4);  /* rounding modes */

    probe_printf("    MVFR0 A_SIMD=%u → 레지스터 뱅크 %s\n", simd_reg,
                 simd_reg == 1 ? "16개 x 64비트 (= s0~s31, d0~d15)" :
                 simd_reg == 2 ? "32개 x 64비트" : "없음/미상");
    probe_printf("    단정밀도(FP32) 하드웨어: %s (SP=%u)\n",
                 sp ? "지원" : "없음", sp);
    probe_printf("    배정밀도(FP64) 하드웨어: %s (DP=%u)\n",
                 dp ? "지원" : "**없음**", dp);
    if (!dp) {
        probe_puts("    ; => 이 칩은 FPv5-SP-D16 이다. double 연산은 전부 소프트웨어(aeabi_dxxx)로\n");
        probe_puts("    ;    처리되므로 float(32비트)만 써야 FPU 이득을 본다.\n");
    }
    probe_printf("    나눗셈(VDIV)=%s  제곱근(VSQRT)=%s  FP예외 트래핑=%s\n",
                 divide ? "하드웨어" : "없음", sqrt ? "하드웨어" : "없음",
                 trap ? "지원" : "없음");
    probe_printf("    반올림 모드=%s  Short vectors=%s\n",
                 rmode ? "모든 IEEE 모드 지원" : "RN만", shortvec ? "지원" : "없음(정상)");

    uint32_t fma  = FLD(m1, 12, 4);   /* MVFR1 D_NaN/FtZ/... FMA는 [15:12] */
    uint32_t ftz  = FLD(m1, 0, 4);
    uint32_t dnan = FLD(m1, 4, 4);
    uint32_t hpc  = FLD(m1, 20, 4);   /* FP HPFP: half precision conversion */
    probe_printf("    MVFR1 FMA(VFMA/VFMS 융합 곱셈누산)=%s\n", fma ? "지원" : "없음");
    probe_printf("    반정밀도(FP16) 변환=%s (FP_HP=%u)  ; 연산이 아니라 변환(VCVTB/VCVTT)만이다\n",
                 hpc ? "지원" : "없음", hpc);
    probe_printf("    FtZ(플러시 투 제로) 모드=%u, Default NaN 모드=%u\n", ftz, dnan);
    probe_printf("    MVFR2 VFP_Misc=%u  ; 0 이외면 추가 제어(부동소수 예외 등) 구현\n",
                 FLD(m2, 4, 4));

    uint32_t cpacr = SCB->CPACR;
    kv("CPACR", 0xE000ED88u, cpacr, NULL);
    probe_printf("    CP10=%u CP11=%u → %s\n", FLD(cpacr, 20, 2), FLD(cpacr, 22, 2),
                 (FLD(cpacr, 20, 2) == 3 && FLD(cpacr, 22, 2) == 3)
                     ? "특권/비특권 모두 FPU 접근 허용 (SystemInit이 켰다)"
                     : "FPU 접근 제한 상태");

    uint32_t fpccr = FPU->FPCCR;
    kv("FPCCR", 0xE000EF34u, fpccr, NULL);
    probe_printf("    ASPEN=%s LSPEN=%s  ; 자동/지연 FPU 컨텍스트 스택\n",
                 onoff(BIT_(fpccr, 31)), onoff(BIT_(fpccr, 30)));
    probe_printf("    LSPACT=%s USER=%u THREAD=%u HFRDY=%u MMRDY=%u BFRDY=%u MONRDY=%u\n",
                 onoff(BIT_(fpccr, 0)), BIT_(fpccr, 1), BIT_(fpccr, 3), BIT_(fpccr, 4),
                 BIT_(fpccr, 5), BIT_(fpccr, 6), BIT_(fpccr, 8));
    probe_printf("    ; LSPEN=1이면 예외 진입 시 FPU 레지스터 저장을 실제 사용 시점까지 미룬다\n");
    kv("FPCAR", 0xE000EF38u, FPU->FPCAR, "마지막 지연 스택 프레임 주소");
    uint32_t fpdscr = FPU->FPDSCR;
    kv("FPDSCR", 0xE000EF3Cu, fpdscr, NULL);
    probe_printf("    기본 FPSCR: RMode=%u AHP=%u DN=%u FZ=%u\n",
                 FLD(fpdscr, 22, 2), BIT_(fpdscr, 26), BIT_(fpdscr, 25), BIT_(fpdscr, 24));
}

/* ------------------------------------------------------------------------- */
/* 5. MPU                                                                     */
/* ------------------------------------------------------------------------- */

static const char *mpu_ap(uint32_t ap)
{
    switch (ap) {
    case 0: return "접근 불가";
    case 1: return "특권 RW / 비특권 없음";
    case 2: return "특권 RW / 비특권 RO";
    case 3: return "특권 RW / 비특권 RW";
    case 5: return "특권 RO / 비특권 없음";
    case 6: return "특권 RO / 비특권 RO";
    case 7: return "특권 RO / 비특권 RO";
    default: return "예약값";
    }
}

static const char *mpu_memtype(uint32_t tex, uint32_t c, uint32_t b)
{
    if (tex == 0) {
        if (!c && !b) return "Strongly-ordered (비캐시, 재정렬 없음)";
        if (!c &&  b) return "Device (공유)";
        if ( c && !b) return "Normal, WT, no WA";
        return "Normal, WB, no WA";
    }
    if (tex == 1) {
        if (!c && !b) return "Normal, 비캐시";
        if ( c &&  b) return "Normal, WB, RA+WA";
        return "예약값";
    }
    if (tex == 2) {
        if (!c && !b) return "Device (비공유)";
        return "예약값";
    }
    return "Normal, 캐시 정책을 TEX[1:0]/C/B로 개별 지정";
}

static void sec_mpu(void)
{
    probe_section("5. MPU");

    uint32_t type = MPU->TYPE;
    uint32_t ctrl = MPU->CTRL;
    kv("MPU_TYPE", 0xE000ED90u, type, NULL);
    uint32_t nreg = FLD(type, 8, 8);
    probe_printf("    DREGION=%u개, IREGION=%u개, SEPARATE=%u\n",
                 nreg, FLD(type, 16, 8), BIT_(type, 0));
    probe_printf("    → 데이터/명령 통합 MPU 리전 %u개\n", nreg);
    kv("MPU_CTRL", 0xE000ED94u, ctrl, NULL);
    probe_printf("    ENABLE=%s HFNMIENA=%s PRIVDEFENA=%s\n",
                 onoff(BIT_(ctrl, 0)), onoff(BIT_(ctrl, 1)), onoff(BIT_(ctrl, 2)));
    if (!BIT_(ctrl, 0)) {
        probe_puts("    ; MPU가 꺼져 있다 → 메모리 속성은 기본 메모리 맵(default map)을 따른다.\n");
        probe_puts("    ;   0x20000000~0x3FFFFFFF SRAM = Normal WB-WA, 0x40000000~ = Device.\n");
        probe_puts("    ;   즉 SDRAM(0xC0000000)은 기본맵에서 Normal 비공유 캐시 가능 영역이다.\n");
    }

    if (nreg == 0) return;
    uint32_t saved_rnr = MPU->RNR;
    probe_puts("\n  리전  RBAR         RASR         베이스      크기      권한/속성\n");
    for (uint32_t r = 0; r < nreg; r++) {
        MPU->RNR = r;
        __DSB();
        uint32_t rbar = MPU->RBAR;
        uint32_t rasr = MPU->RASR;
        uint32_t en   = BIT_(rasr, 0);
        uint32_t szf  = FLD(rasr, 1, 5);
        uint32_t base = rbar & 0xFFFFFFE0u;
        probe_printf("  %2u    0x%08X   0x%08X   ", r, rbar, rasr);
        if (!en) {
            probe_puts("(비활성)\n");
            continue;
        }
        uint32_t size_b = (szf >= 4u && szf <= 31u) ? (1u << (szf + 1u)) : 0u;
        probe_printf("0x%08X  ", base);
        if (size_b >= 1024u * 1024u) probe_printf("%4u MB  ", size_b / (1024u * 1024u));
        else if (size_b >= 1024u)    probe_printf("%4u KB  ", size_b / 1024u);
        else                         probe_printf("%4u B   ", size_b);
        probe_printf("%s\n", mpu_ap(FLD(rasr, 24, 3)));
        probe_printf("        TEX=%u C=%u B=%u S=%u XN=%u SRD=0x%02X → %s\n",
                     FLD(rasr, 19, 3), BIT_(rasr, 17), BIT_(rasr, 16), BIT_(rasr, 18),
                     BIT_(rasr, 28), FLD(rasr, 8, 8),
                     mpu_memtype(FLD(rasr, 19, 3), BIT_(rasr, 17), BIT_(rasr, 16)));
    }
    MPU->RNR = saved_rnr;
    __DSB();
}

/* ------------------------------------------------------------------------- */
/* 6. 디버그 유닛 / CoreSight ROM 테이블                                       */
/* ------------------------------------------------------------------------- */

static const char *coresight_part(uint32_t part)
{
    switch (part) {
    case 0x000: return "SCS (Cortex-M3)";
    case 0x001: return "ITM";
    case 0x002: return "DWT";
    case 0x003: return "FPB (Cortex-M3/M4)";
    case 0x008: return "SCS (Cortex-M0)";
    case 0x00A: return "DWT (Cortex-M0)";
    case 0x00B: return "BPU (Cortex-M0)";
    case 0x00C: return "SCS (Cortex-M4/M7)";
    case 0x00D: return "ETM (Cortex-M4)";
    case 0x00E: return "FPB (Cortex-M7)";
    case 0x490: return "CTI";
    case 0x906: return "CoreSight CTI";
    case 0x907: return "CoreSight ETB";
    case 0x908: return "CoreSight CSTF";
    case 0x912: return "CoreSight TPIU";
    case 0x913: return "CoreSight ITM";
    case 0x914: return "CoreSight SWO";
    case 0x923: return "TPIU (Cortex-M3)";
    case 0x924: return "ETM (Cortex-M3)";
    case 0x925: return "ETM (Cortex-M4)";
    case 0x961: return "CoreSight TMC";
    case 0x962: return "CoreSight STM";
    case 0x975: return "ETM (Cortex-M7)";
    case 0x9A1: return "TPIU (Cortex-M4)";
    case 0x9A9: return "TPIU (Cortex-M7)";
    default:    return "미상";
    }
}

static const char *jep106_name(uint32_t cont, uint32_t code)
{
    if (cont == 4u && code == 0x3Bu) return "ARM Limited";
    if (cont == 0u && code == 0x20u) return "STMicroelectronics";
    return "미상";
}

/* 컴포넌트 하나의 PIDR/CIDR를 읽어 식별한다. 접근 불가면 0을 반환. */
static int identify_component(uint32_t base, const char *hint)
{
    uint32_t pidr[8], cidr[4];
    for (int i = 0; i < 4; i++) {
        if (!safe_read32(base + 0xFD0u + 4u * (uint32_t)i, &pidr[4 + i])) return 0;
    }
    for (int i = 0; i < 4; i++) {
        if (!safe_read32(base + 0xFE0u + 4u * (uint32_t)i, &pidr[i])) return 0;
    }
    for (int i = 0; i < 4; i++) {
        if (!safe_read32(base + 0xFF0u + 4u * (uint32_t)i, &cidr[i])) return 0;
    }
    uint32_t part = (pidr[0] & 0xFFu) | ((pidr[1] & 0x0Fu) << 8);
    uint32_t jcode = ((pidr[1] >> 4) & 0x0Fu) | ((pidr[2] & 0x07u) << 4);
    uint32_t jedec = (pidr[2] >> 3) & 1u;
    uint32_t cont  = pidr[4] & 0x0Fu;
    uint32_t rev   = (pidr[2] >> 4) & 0x0Fu;
    uint32_t cls   = (cidr[1] >> 4) & 0x0Fu;
    int cid_ok = ((cidr[0] & 0xFFu) == 0x0Du) && ((cidr[2] & 0xFFu) == 0x05u) &&
                 ((cidr[3] & 0xFFu) == 0xB1u);

    probe_printf("    0x%08X  PIDR=%02X%02X%02X%02X/%02X%02X%02X%02X  CIDR=%02X%02X%02X%02X\n",
                 base, pidr[3] & 0xFFu, pidr[2] & 0xFFu, pidr[1] & 0xFFu, pidr[0] & 0xFFu,
                 pidr[7] & 0xFFu, pidr[6] & 0xFFu, pidr[5] & 0xFFu, pidr[4] & 0xFFu,
                 cidr[3] & 0xFFu, cidr[2] & 0xFFu, cidr[1] & 0xFFu, cidr[0] & 0xFFu);
    probe_printf("      part=0x%03X (%s) rev=%u  JEP106 cont=%u code=0x%02X (%s)%s\n",
                 part, coresight_part(part), rev, cont, jcode,
                 jedec ? jep106_name(cont, jcode) : "JEDEC 코드 아님", hint);
    probe_printf("      CID 프리앰블=%s  class=0x%X (%s)\n",
                 cid_ok ? "정상" : "불일치", cls,
                 cls == 1 ? "ROM table" : cls == 9 ? "CoreSight 컴포넌트" : "기타");
    return 1;
}

static void sec_debug(void)
{
    probe_section("6. 디버그 유닛 / CoreSight ROM 테이블");

    uint32_t demcr = CoreDebug->DEMCR;
    kv("DEMCR", 0xE000EDFCu, demcr, NULL);
    probe_printf("    TRCENA=%s  ; probe_time_init()이 켰다. DWT/ITM/TPIU 전원 게이트 역할\n",
                 onoff(BIT_(demcr, 24)));
    kv("DHCSR", 0xE000EDF0u, CoreDebug->DHCSR, NULL);
    probe_printf("    C_DEBUGEN=%s S_HALT=%s  ; 디버거(ST-LINK) 연결 여부의 근거\n",
                 onoff(BIT_(CoreDebug->DHCSR, 0)), onoff(BIT_(CoreDebug->DHCSR, 17)));

    uint32_t dwtctrl = DWT->CTRL;
    kv("DWT_CTRL", 0xE0001000u, dwtctrl, NULL);
    probe_printf("    NUMCOMP=%u개 (데이터 워치포인트/비교기)\n", FLD(dwtctrl, 28, 4));
    probe_printf("    NOTRCPKT=%u NOEXTTRIG=%u NOCYCCNT=%u NOPRFCNT=%u\n",
                 BIT_(dwtctrl, 27), BIT_(dwtctrl, 26), BIT_(dwtctrl, 25), BIT_(dwtctrl, 24));
    probe_printf("    → 사이클 카운터 %s, 프로파일 카운터(CPI/EXC/SLEEP/LSU/FOLD) %s\n",
                 BIT_(dwtctrl, 25) ? "없음" : "있음 (CYCCNT 사용 가능)",
                 BIT_(dwtctrl, 24) ? "없음" : "있음");
    probe_printf("    CYCCNTENA=%s, 현재 CYCCNT=0x%08X\n",
                 onoff(BIT_(dwtctrl, 0)), DWT->CYCCNT);

    uint32_t fpctrl = 0, fpremap = 0;
    if (safe_read32(0xE0002000u, &fpctrl)) {
        kv("FP_CTRL", 0xE0002000u, fpctrl, NULL);
        uint32_t ncode = (FLD(fpctrl, 12, 3) << 4) | FLD(fpctrl, 4, 4);
        probe_printf("    NUM_CODE=%u개 (명령 브레이크포인트), NUM_LIT=%u개 (리터럴 비교기)\n",
                     ncode, FLD(fpctrl, 8, 4));
        probe_printf("    ENABLE=%s KEY=%u REV=%u  ; REV=1이면 FPBv2 (전체 주소 비교)\n",
                     onoff(BIT_(fpctrl, 0)), BIT_(fpctrl, 1), FLD(fpctrl, 28, 4));
        (void)safe_read32(0xE0002004u, &fpremap);
        kv("FP_REMAP", 0xE0002004u, fpremap, NULL);
    } else {
        probe_puts("    FP_CTRL(0xE0002000) 읽기 실패 → FPB 미구현?\n");
    }

    probe_puts("\n  [CoreSight ROM table @ 0xE00FF000]\n");
    for (uint32_t i = 0; i < 8u; i++) {
        uint32_t addr = 0xE00FF000u + 4u * i;
        uint32_t ent;
        if (!safe_read32(addr, &ent)) {
            probe_printf("    entry[%u] @ 0x%08X 읽기 실패(BusFault)\n", i, addr);
            continue;
        }
        if (ent == 0u) {
            probe_printf("    entry[%u] = 0x00000000 → 테이블 종료\n", i);
            break;
        }
        /* 엔트리의 [31:12]은 ROM 베이스 기준 부호있는 오프셋이다.
         * 32비트 모듈러 덧셈이 부호 확장을 자동으로 처리하므로 uint32_t로 계산한다. */
        uint32_t cbase = 0xE00FF000u + (ent & 0xFFFFF000u);
        uint32_t present = ent & 1u;
        probe_printf("    entry[%u] = 0x%08X → base 0x%08X, present=%u\n",
                     i, ent, cbase, present);
        if (!identify_component(cbase, "")) {
            probe_printf("      → 이 주소는 읽을 수 없다 (BusFault). 컴포넌트가 물리적으로 없거나 전원 차단\n");
        }
    }

    probe_puts("\n  [TPIU/ETM 재확인 — TRCENA가 켜진 상태에서]\n");
    probe_printf("    DEMCR.TRCENA=%s 인 지금 다시 읽는다.\n", onoff(BIT_(CoreDebug->DEMCR, 24)));
    {
        /* 인덱스를 가정하지 않고, 계산된 베이스가 TPIU/ETM 주소인 엔트리를 찾는다 */
        for (uint32_t i = 0; i < 8u; i++) {
            uint32_t ent;
            if (!safe_read32(0xE00FF000u + 4u * i, &ent) || ent == 0u) break;
            uint32_t cbase = 0xE00FF000u + (ent & 0xFFFFF000u);
            if (cbase != 0xE0040000u && cbase != 0xE0041000u) continue;
            probe_printf("    ROM entry[%u] (base 0x%08X, %s) = 0x%08X → present=%u\n",
                         i, cbase, cbase == 0xE0040000u ? "TPIU" : "ETM", ent, ent & 1u);
        }
        uint32_t v;
        int tpiu_ok = safe_read32(0xE0040FE0u, &v);    /* TPIU PIDR0 */
        probe_printf("    TPIU(0xE0040000) PIDR0 직접 읽기: %s\n",
                     tpiu_ok ? "성공 → 하드웨어 존재" : "BusFault → 접근 불가");
        int etm_ok = safe_read32(0xE0041FE0u, &v);     /* ETM PIDR0 */
        probe_printf("    ETM (0xE0041000) PIDR0 직접 읽기: %s\n",
                     etm_ok ? "성공 → 하드웨어 존재" : "BusFault → 접근 불가");
        probe_puts("    ; ROM 테이블의 present=0 은 \"이 ROM에서 광고하지 않음\"이라는 뜻이다.\n");
        probe_puts("    ; 위 직접 읽기가 성공하면 유닛은 있는데 ROM이 광고하지 않는 것이고,\n");
        probe_puts("    ; BusFault면 TRCENA를 켠 뒤에도 접근 불가 = 이 실리콘에 없거나 완전 차단이다.\n");
    }

    uint32_t idcode = *(volatile uint32_t *)0xE0042000u;   /* DBGMCU_IDCODE */
    kv("DBGMCU_IDCODE", 0xE0042000u, idcode, NULL);
    uint32_t devid = FLD(idcode, 0, 12), revid = FLD(idcode, 16, 16);
    probe_printf("    DEV_ID=0x%03X (%s)  REV_ID=0x%04X (%s)\n", devid,
                 devid == 0x449u ? "STM32F74x/F75x" :
                 devid == 0x451u ? "STM32F76x/F77x" :
                 devid == 0x452u ? "STM32F72x/F73x" : "미상",
                 revid,
                 revid == 0x1000u ? "rev A" : revid == 0x1001u ? "rev Z" :
                 revid == 0x1003u ? "rev Y" : revid == 0x2001u ? "rev X" :
                 revid == 0x2003u ? "rev V" : "미상");
    kv("DBGMCU_CR", 0xE0042004u, *(volatile uint32_t *)0xE0042004u,
       "DBG_SLEEP/STOP/STANDBY + TRACE 설정");
    uint32_t dbgcr = *(volatile uint32_t *)0xE0042004u;
    probe_printf("    DBG_SLEEP=%u DBG_STOP=%u DBG_STANDBY=%u TRACE_IOEN=%u TRACE_MODE=%u\n",
                 BIT_(dbgcr, 0), BIT_(dbgcr, 1), BIT_(dbgcr, 2),
                 BIT_(dbgcr, 5), FLD(dbgcr, 6, 2));
}

/* ------------------------------------------------------------------------- */
/* 7. 팩토리 프로그램 영역                                                     */
/* ------------------------------------------------------------------------- */

static void sec_factory(void)
{
    probe_section("7. 팩토리 프로그램 영역 (읽기 전용)");

    uint32_t uid0 = 0, uid1 = 0, uid2 = 0;
    int ok = safe_read32(0x1FF0F420u, &uid0);
    ok &= safe_read32(0x1FF0F424u, &uid1);
    ok &= safe_read32(0x1FF0F428u, &uid2);
    if (ok) {
        kv("UID[31:0]", 0x1FF0F420u, uid0, NULL);
        kv("UID[63:32]", 0x1FF0F424u, uid1, NULL);
        kv("UID[95:64]", 0x1FF0F428u, uid2, NULL);
        probe_printf("    96비트 고유 ID = %08X%08X%08X\n", uid2, uid1, uid0);
        probe_printf("    ; UID[31:0]의 하위 16비트는 웨이퍼 X좌표, 상위 16비트는 Y좌표,\n");
        probe_printf("    ; UID[39:32]는 웨이퍼 번호, 나머지는 로트 번호(ASCII)다.\n");
        probe_printf("    웨이퍼 X=%u Y=%u, 웨이퍼번호=%u, LOT='%c%c%c%c%c%c%c'\n",
                     uid0 & 0xFFFFu, (uid0 >> 16) & 0xFFFFu, uid1 & 0xFFu,
                     (char)((uid1 >> 8) & 0xFF), (char)((uid1 >> 16) & 0xFF),
                     (char)((uid1 >> 24) & 0xFF), (char)(uid2 & 0xFF),
                     (char)((uid2 >> 8) & 0xFF), (char)((uid2 >> 16) & 0xFF),
                     (char)((uid2 >> 24) & 0xFF));
    } else {
        probe_puts("    UID 읽기 실패 (BusFault)\n");
    }

    uint32_t fsz = read16(0x1FF0F442u);
    probe_printf("FLASH_SIZE       @ 0x1FF0F442 = 0x%04X  ; %u KB (%u MB)\n",
                 fsz, fsz, fsz / 1024u);

    uint32_t pkg = 0;
    if (safe_read32(0x1FF07BF0u, &pkg)) {
        kv("PACKAGE", 0x1FF07BF0u, pkg, NULL);
        if (pkg == 0xFFFFFFFFu) {
            probe_puts("    → 전 비트가 1이다. 이 다이에는 패키지 데이터가 프로그램되어 있지 않다.\n");
            probe_puts("      (ST-LINK 실측과 동일) 패키지 종류를 이 레지스터로는 판정할 수 없다.\n");
        } else {
            probe_printf("    PKG[2:0]=%u  ; RM0385 패키지 인코딩\n", FLD(pkg, 0, 3));
        }
    } else {
        probe_puts("    PACKAGE(0x1FF07BF0) 읽기 실패(BusFault)\n");
    }
    uint32_t pkg2 = 0;
    if (safe_read32(0x1FF0F7E0u, &pkg2)) {
        kv("PACKAGE(hdr)", 0x1FF0F7E0u, pkg2,
           "CMSIS 헤더의 PACKAGE_BASE. 위 주소와 다른 값이면 둘 중 하나는 미프로그램");
    }

    uint32_t vref = read16(0x1FF0F44Au);
    uint32_t ts1  = read16(0x1FF0F44Cu);
    uint32_t ts2  = read16(0x1FF0F44Eu);
    probe_printf("VREFINT_CAL      @ 0x1FF0F44A = 0x%04X  ; %u LSB @3.3V, 30도\n", vref, vref);
    probe_printf("TS_CAL1          @ 0x1FF0F44C = 0x%04X  ; %u LSB @30도\n", ts1, ts1);
    probe_printf("TS_CAL2          @ 0x1FF0F44E = 0x%04X  ; %u LSB @110도\n", ts2, ts2);
    /*
     * 온도 계산식 (RM0385 / 데이터시트):
     *   T[도] = (110 - 30) / (TS_CAL2 - TS_CAL1) * (TS_DATA - TS_CAL1) + 30
     * 전원전압 보정:
     *   VDD[V] = 3.3 * VREFINT_CAL / VREFINT_DATA
     *   TS_DATA_보정 = TS_DATA * VDD / 3.3
     * TS_CAL1/2는 12비트 ADC, VDDA=3.3V 조건에서 팩토리 측정된 값이다.
     */
    if (ts2 > ts1) {
        uint32_t slope_uv = (80u * 1000000u) / (ts2 - ts1);   /* LSB당 밀리도(x1000) */
        probe_printf("    기울기 = 80도 / %u LSB = %u.%03u 도/LSB\n",
                     ts2 - ts1, slope_uv / 1000000u, (slope_uv / 1000u) % 1000u);
        probe_puts("    ; 계산식: T = 80/(TS_CAL2-TS_CAL1) * (TS_DATA-TS_CAL1) + 30\n");
    } else {
        probe_puts("    ; TS_CAL2 <= TS_CAL1 → 캘리브레이션 값이 비정상이거나 미프로그램\n");
    }
}

/* ------------------------------------------------------------------------- */
/* 8. 클럭 / 전원 / 플래시                                                     */
/* ------------------------------------------------------------------------- */

static const uint16_t k_ahb_div[16] = {1,1,1,1,1,1,1,1,2,4,8,16,64,128,256,512};
static const uint8_t  k_apb_div[8]  = {1,1,1,1,2,4,8,16};

static void sec_clock(void)
{
    probe_section("8. 클럭 / 전원 / 플래시");

    uint32_t cr      = RCC->CR;
    uint32_t pllcfgr = RCC->PLLCFGR;
    uint32_t cfgr    = RCC->CFGR;
    kv("RCC_CR", 0x40023800u, cr, NULL);
    probe_printf("    HSION=%u HSIRDY=%u HSEON=%u HSERDY=%u HSEBYP=%u CSSON=%u\n",
                 BIT_(cr, 0), BIT_(cr, 1), BIT_(cr, 16), BIT_(cr, 17),
                 BIT_(cr, 18), BIT_(cr, 19));
    probe_printf("    PLLON=%u PLLRDY=%u PLLI2SON=%u PLLI2SRDY=%u PLLSAION=%u PLLSAIRDY=%u\n",
                 BIT_(cr, 24), BIT_(cr, 25), BIT_(cr, 26), BIT_(cr, 27),
                 BIT_(cr, 28), BIT_(cr, 29));
    probe_printf("    HSITRIM=%u HSICAL=%u\n", FLD(cr, 3, 5), FLD(cr, 8, 8));

    kv("RCC_PLLCFGR", 0x40023804u, pllcfgr, NULL);
    uint32_t pllm = FLD(pllcfgr, 0, 6);
    uint32_t plln = FLD(pllcfgr, 6, 9);
    uint32_t pllp = (FLD(pllcfgr, 16, 2) + 1u) * 2u;
    uint32_t pllq = FLD(pllcfgr, 24, 4);
    uint32_t pllsrc = BIT_(pllcfgr, 22);
    uint32_t src_hz = pllsrc ? 25000000u : 16000000u;   /* 보드 HSE=25MHz, HSI=16MHz */
    probe_printf("    PLLSRC=%u (%s, %u Hz 가정)  PLLM=%u PLLN=%u PLLP=%u PLLQ=%u\n",
                 pllsrc, pllsrc ? "HSE" : "HSI", src_hz, pllm, plln, pllp, pllq);
    uint32_t vco_in  = pllm ? (src_hz / pllm) : 0u;
    uint32_t vco_out = vco_in * plln;
    uint32_t sysclk  = pllp ? (vco_out / pllp) : 0u;
    probe_printf("    VCO 입력 = %u / %u = %u Hz (권장 1~2MHz)\n", src_hz, pllm, vco_in);
    probe_printf("    VCO 출력 = %u x %u = %u Hz\n", vco_in, plln, vco_out);
    probe_printf("    PLL P 출력(SYSCLK 후보) = %u / %u = %u Hz\n", vco_out, pllp, sysclk);
    probe_printf("    PLL Q 출력(48MHz 계열) = %u / %u = %u Hz\n",
                 vco_out, pllq, pllq ? vco_out / pllq : 0u);

    kv("RCC_CFGR", 0x40023808u, cfgr, NULL);
    uint32_t sws = FLD(cfgr, 2, 2);
    const char *swsname = (sws == 0) ? "HSI" : (sws == 1) ? "HSE" : (sws == 2) ? "PLL" : "예약";
    uint32_t hpre  = k_ahb_div[FLD(cfgr, 4, 4)];
    uint32_t ppre1 = k_apb_div[FLD(cfgr, 10, 3)];
    uint32_t ppre2 = k_apb_div[FLD(cfgr, 13, 3)];
    probe_printf("    SW=%u SWS=%u (%s가 SYSCLK 소스)  HPRE=/%u  PPRE1=/%u  PPRE2=/%u\n",
                 FLD(cfgr, 0, 2), sws, swsname, hpre, ppre1, ppre2);
    probe_printf("    RTCPRE=/%u  MCO1=%u(/%u)  MCO2=%u(/%u)\n",
                 FLD(cfgr, 16, 5), FLD(cfgr, 21, 2), FLD(cfgr, 24, 3) >= 4 ? FLD(cfgr, 24, 3) - 2 : 1,
                 FLD(cfgr, 30, 2), FLD(cfgr, 27, 3) >= 4 ? FLD(cfgr, 27, 3) - 2 : 1);

    uint32_t hw_sysclk = (sws == 2) ? sysclk : (sws == 1 ? 25000000u : 16000000u);
    uint32_t hw_hclk   = hw_sysclk / hpre;
    uint32_t hw_pclk1  = hw_hclk / ppre1;
    uint32_t hw_pclk2  = hw_hclk / ppre2;
    probe_puts("\n  [레지스터로 역산한 클럭 vs 소프트웨어가 믿는 값]\n");
    probe_printf("    SYSCLK  역산 %9u Hz   소프트웨어 %9u Hz   %s\n",
                 hw_sysclk, g_sysclk_hz, hw_sysclk == g_sysclk_hz ? "일치" : "**불일치**");
    probe_printf("    HCLK    역산 %9u Hz   소프트웨어 %9u Hz   %s\n",
                 hw_hclk, g_hclk_hz, hw_hclk == g_hclk_hz ? "일치" : "**불일치**");
    probe_printf("    PCLK1   역산 %9u Hz   소프트웨어 %9u Hz   %s\n",
                 hw_pclk1, g_pclk1_hz, hw_pclk1 == g_pclk1_hz ? "일치" : "**불일치**");
    probe_printf("    PCLK2   역산 %9u Hz   소프트웨어 %9u Hz   %s\n",
                 hw_pclk2, g_pclk2_hz, hw_pclk2 == g_pclk2_hz ? "일치" : "**불일치**");
    probe_printf("    SystemCoreClock 전역변수 = %u Hz\n", SystemCoreClock);
    probe_puts("    ; 역산은 HSE=25MHz(보드 크리스탈)를 가정한다. 이 가정이 틀리면 전부 비례해 틀린다.\n");
    probe_puts("    ; probe_time.c의 사이클→시간 환산이 이 값에 의존하므로 불일치는 치명적이다.\n");

    kv("RCC_DCKCFGR1", 0x4002388Cu, RCC->DCKCFGR1, NULL);
    uint32_t d1 = RCC->DCKCFGR1;
    probe_printf("    PLLI2SDIVQ=/%u PLLSAIDIVQ=/%u PLLSAIDIVR=/%u SAI1SEL=%u SAI2SEL=%u TIMPRE=%u\n",
                 FLD(d1, 0, 5) + 1u, FLD(d1, 8, 5) + 1u,
                 1u << FLD(d1, 16, 2), FLD(d1, 20, 2), FLD(d1, 22, 2), BIT_(d1, 24));
    kv("RCC_DCKCFGR2", 0x40023890u, RCC->DCKCFGR2, NULL);
    uint32_t d2 = RCC->DCKCFGR2;
    probe_printf("    USART1SEL=%u USART2SEL=%u USART3SEL=%u UART4SEL=%u UART5SEL=%u USART6SEL=%u\n",
                 FLD(d2, 0, 2), FLD(d2, 2, 2), FLD(d2, 4, 2), FLD(d2, 6, 2),
                 FLD(d2, 8, 2), FLD(d2, 10, 2));
    probe_printf("    I2C1SEL=%u I2C2SEL=%u I2C3SEL=%u I2C4SEL=%u LPTIM1SEL=%u CK48MSEL=%u SDMMC1SEL=%u\n",
                 FLD(d2, 16, 2), FLD(d2, 18, 2), FLD(d2, 20, 2), FLD(d2, 22, 2),
                 FLD(d2, 24, 2), BIT_(d2, 27), BIT_(d2, 28));
    probe_printf("    ; USART1SEL=0 → PCLK2 사용. probe_io.c가 BRR을 PCLK2 기준으로 잡는 근거다\n");

    kv("RCC_BDCR", 0x40023870u, RCC->BDCR, NULL);
    uint32_t bd = RCC->BDCR;
    probe_printf("    LSEON=%u LSERDY=%u LSEBYP=%u LSEDRV=%u RTCSEL=%u RTCEN=%u BDRST=%u\n",
                 BIT_(bd, 0), BIT_(bd, 1), BIT_(bd, 2), FLD(bd, 3, 2),
                 FLD(bd, 8, 2), BIT_(bd, 15), BIT_(bd, 16));

    uint32_t csr = RCC->CSR;
    kv("RCC_CSR", 0x40023874u, csr, NULL);
    probe_printf("    LSION=%u LSIRDY=%u\n", BIT_(csr, 0), BIT_(csr, 1));
    probe_puts("  [이번 부팅의 리셋 원인]\n");
    probe_printf("    LPWRRSTF=%u (저전력 리셋)   WWDGRSTF=%u (윈도우 워치독)\n",
                 BIT_(csr, RCC_CSR_LPWRRSTF_Pos), BIT_(csr, RCC_CSR_WWDGRSTF_Pos));
    probe_printf("    IWDGRSTF=%u (독립 워치독)   SFTRSTF=%u (소프트웨어 리셋)\n",
                 BIT_(csr, RCC_CSR_IWDGRSTF_Pos), BIT_(csr, RCC_CSR_SFTRSTF_Pos));
    probe_printf("    PORRSTF=%u (POR/PDR)        PINRSTF=%u (NRST 핀)\n",
                 BIT_(csr, RCC_CSR_PORRSTF_Pos), BIT_(csr, RCC_CSR_PINRSTF_Pos));
    probe_printf("    BORRSTF=%u (브라운아웃)\n", BIT_(csr, RCC_CSR_BORRSTF_Pos));
    {
        const char *cause = "미상 (플래그 없음 — 이전 실행이 RMVF로 지웠을 수 있음)";
        if (BIT_(csr, RCC_CSR_LPWRRSTF_Pos))      cause = "저전력 리셋";
        else if (BIT_(csr, RCC_CSR_WWDGRSTF_Pos)) cause = "윈도우 워치독";
        else if (BIT_(csr, RCC_CSR_IWDGRSTF_Pos)) cause = "독립 워치독";
        else if (BIT_(csr, RCC_CSR_SFTRSTF_Pos))  cause = "소프트웨어 리셋 (디버거의 리셋 포함)";
        else if (BIT_(csr, RCC_CSR_PINRSTF_Pos))  cause = "NRST 핀 (ST-LINK 하드리셋 또는 리셋 버튼)";
        else if (BIT_(csr, RCC_CSR_PORRSTF_Pos))  cause = "파워온 리셋 (전원 인가)";
        else if (BIT_(csr, RCC_CSR_BORRSTF_Pos))  cause = "브라운아웃 리셋";
        probe_printf("    → 가장 유력한 원인: %s\n", cause);
        probe_puts("    ; PORRSTF와 PINRSTF가 함께 서면 전원 인가 직후이며, 플래그는 RMVF로만 지워진다\n");
    }

    uint32_t acr = FLASH->ACR;
    kv("FLASH_ACR", 0x40023C00u, acr, NULL);
    probe_printf("    LATENCY=%u wait state, PRFTEN=%s, ARTEN=%s, ARTRST=%u\n",
                 FLD(acr, 0, 4), onoff(BIT_(acr, 8)), onoff(BIT_(acr, 9)), BIT_(acr, 11));
    probe_printf("    ; %u WS는 %u MHz에서 VOS Scale1 기준 요구값이다. ART가 꺼지면 플래시\n",
                 FLD(acr, 0, 4), hw_hclk / 1000000u);
    probe_printf("    ; 실행 성능이 크게 떨어진다 (probe_cpu.c가 ART on/off를 비교 측정한다)\n");

    uint32_t optcr = FLASH->OPTCR;
    kv("FLASH_OPTCR", 0x40023C14u, optcr, NULL);
    uint32_t rdp = FLD(optcr, 8, 8);
    probe_printf("    RDP=0x%02X → %s\n", rdp,
                 rdp == 0xAAu ? "Level 0 (보호 없음)" :
                 rdp == 0xCCu ? "Level 2 (영구 차단)" : "Level 1 (디버그 접근 시 플래시 소거)");
    probe_printf("    BOR_LEV=%u → %s\n", FLD(optcr, 2, 2),
                 FLD(optcr, 2, 2) == 3 ? "BOR 꺼짐 (VBOR0, 1.8V)" :
                 FLD(optcr, 2, 2) == 2 ? "BOR Level 1 (2.1V)" :
                 FLD(optcr, 2, 2) == 1 ? "BOR Level 2 (2.4V)" : "BOR Level 3 (2.7V)");
    probe_printf("    WWDG_SW=%u (%s)  IWDG_SW=%u (%s)\n",
                 BIT_(optcr, 4), BIT_(optcr, 4) ? "소프트웨어 제어" : "하드웨어 강제",
                 BIT_(optcr, 5), BIT_(optcr, 5) ? "소프트웨어 제어" : "하드웨어 강제");
    probe_printf("    nRST_STOP=%u (%s)  nRST_STDBY=%u (%s)\n",
                 BIT_(optcr, 6), BIT_(optcr, 6) ? "STOP 진입 시 리셋 안 함" : "STOP 진입 시 리셋",
                 BIT_(optcr, 7), BIT_(optcr, 7) ? "STANDBY 진입 시 리셋 안 함" : "STANDBY 진입 시 리셋");
    probe_printf("    IWDG_STDBY=%u IWDG_STOP=%u\n", BIT_(optcr, 30), BIT_(optcr, 31));
    uint32_t nwrp = FLD(optcr, 16, 8);
    probe_printf("    nWRP=0x%02X → 쓰기보호 섹터: ", nwrp);
    if (nwrp == 0xFFu) probe_puts("없음 (전 섹터 쓰기 가능)");
    else for (int s = 0; s < 8; s++) if (!((nwrp >> s) & 1u)) probe_printf("%d ", s);
    probe_puts("\n");
    uint32_t optcr1 = FLASH->OPTCR1;
    kv("FLASH_OPTCR1", 0x40023C18u, optcr1, NULL);
    probe_printf("    BOOT_ADD0=0x%04X → 0x%08X, BOOT_ADD1=0x%04X → 0x%08X\n",
                 FLD(optcr1, 0, 16), FLD(optcr1, 0, 16) << 14,
                 FLD(optcr1, 16, 16), FLD(optcr1, 16, 16) << 14);
    probe_puts("    ; BOOT_ADDx는 상위 16비트만 저장한다 (실제 주소 = 값 x 0x4000)\n");

    uint32_t pcr1 = PWR->CR1;
    uint32_t pcsr = PWR->CSR1;
    kv("PWR_CR1", 0x40007000u, pcr1, NULL);
    uint32_t vos = FLD(pcr1, 14, 2);
    probe_printf("    VOS=%u → %s\n", vos,
                 vos == 3 ? "Scale 1 (최대 216MHz, 최고 전압)" :
                 vos == 2 ? "Scale 2 (최대 180MHz)" : "Scale 3 (최대 144MHz)");
    probe_printf("    ODEN=%s ODSWEN=%s UDEN=%u LPDS=%u PDDS=%u FPDS=%u DBP=%u\n",
                 onoff(BIT_(pcr1, 16)), onoff(BIT_(pcr1, 17)), FLD(pcr1, 18, 2),
                 BIT_(pcr1, 0), BIT_(pcr1, 1), BIT_(pcr1, 9), BIT_(pcr1, 8));
    kv("PWR_CSR1", 0x40007004u, pcsr, NULL);
    probe_printf("    VOSRDY=%s ODRDY=%s ODSWRDY=%s UDRDY=%u\n",
                 yes_no(BIT_(pcsr, 14)), yes_no(BIT_(pcsr, 16)), yes_no(BIT_(pcsr, 17)), FLD(pcsr, 18, 2));
    probe_printf("    PVDO=%u BRR=%u BRE=%u SBF=%u WUIF=%u\n",
                 BIT_(pcsr, 2), BIT_(pcsr, 3), BIT_(pcsr, 9), BIT_(pcsr, 1), BIT_(pcsr, 0));
    probe_puts("    ; ODEN/ODSWEN이 모두 1이고 ODSWRDY=1이어야 200MHz 이상에서 동작한다.\n");
    probe_puts("    ; over-drive는 코어 전압을 올려 최대 클럭을 얻는 대신 소비전류가 늘어난다.\n");
}

/* ------------------------------------------------------------------------- */
/* 9. 페리페럴 존재 실측                                                       */
/* ------------------------------------------------------------------------- */

enum { E_NONE = 0, E_AHB1, E_AHB2, E_AHB3, E_APB1, E_APB2 };

typedef struct {
    const char *name;
    uint32_t    base;
    uint8_t     bus;
    uint32_t    mask;
} periph_t;

/* F746 헤더에 매크로가 없는 것(F76x/F77x 전용)은 RM0410의 비트 위치를 직접 쓴다 */
#define AHB2ENR_JPEGEN  (1u << 2)
#define AHB2ENR_CRYPEN  (1u << 4)
#define AHB2ENR_HASHEN  (1u << 5)

static const periph_t k_periphs[] = {
    /* AHB2 — 여기서 JPEG/CRYP/HASH는 F746에 없어야 정상이다 */
    { "JPEG",       0x50051000u, E_AHB2, AHB2ENR_JPEGEN },
    { "CRYP",       0x50060000u, E_AHB2, AHB2ENR_CRYPEN },
    { "HASH",       0x50060400u, E_AHB2, AHB2ENR_HASHEN },
    { "RNG",        0x50060800u, E_AHB2, RCC_AHB2ENR_RNGEN },
    { "DCMI",       0x50050000u, E_AHB2, RCC_AHB2ENR_DCMIEN },
    { "USB_OTG_FS", 0x50000000u, E_AHB2, RCC_AHB2ENR_OTGFSEN },
    /* AHB1 */
    { "USB_OTG_HS", 0x40040000u, E_AHB1, RCC_AHB1ENR_OTGHSEN },
    { "ETH_MAC",    0x40028000u, E_AHB1, RCC_AHB1ENR_ETHMACEN },
    { "DMA2D",      0x4002B000u, E_AHB1, RCC_AHB1ENR_DMA2DEN },
    { "DMA1",       0x40026000u, E_AHB1, RCC_AHB1ENR_DMA1EN },
    { "DMA2",       0x40026400u, E_AHB1, RCC_AHB1ENR_DMA2EN },
    { "CRC",        0x40023000u, E_AHB1, RCC_AHB1ENR_CRCEN },
    /* AHB3 */
    { "FMC",        0xA0000000u, E_AHB3, RCC_AHB3ENR_FMCEN },
    { "QUADSPI",    0xA0001000u, E_AHB3, RCC_AHB3ENR_QSPIEN },
    /* 없어야 정상인 대조군 (F746에는 클럭 인에이블 비트 자체가 없다) */
    { "DFSDM1(x)",  0x40017400u, E_NONE, 0u },
    { "DSI(x)",     0x40016C00u, E_NONE, 0u },
    /* APB2 */
    { "TIM1",       0x40010000u, E_APB2, RCC_APB2ENR_TIM1EN },
    { "TIM8",       0x40010400u, E_APB2, RCC_APB2ENR_TIM8EN },
    { "USART1",     0x40011000u, E_APB2, RCC_APB2ENR_USART1EN },
    { "USART6",     0x40011400u, E_APB2, RCC_APB2ENR_USART6EN },
    { "ADC1",       0x40012000u, E_APB2, RCC_APB2ENR_ADC1EN },
    { "ADC2",       0x40012100u, E_APB2, RCC_APB2ENR_ADC2EN },
    { "ADC3",       0x40012200u, E_APB2, RCC_APB2ENR_ADC3EN },
    { "SDMMC1",     0x40012C00u, E_APB2, RCC_APB2ENR_SDMMC1EN },
    { "SPI1",       0x40013000u, E_APB2, RCC_APB2ENR_SPI1EN },
    { "SPI4",       0x40013400u, E_APB2, RCC_APB2ENR_SPI4EN },
    { "TIM9",       0x40014000u, E_APB2, RCC_APB2ENR_TIM9EN },
    { "TIM10",      0x40014400u, E_APB2, RCC_APB2ENR_TIM10EN },
    { "TIM11",      0x40014800u, E_APB2, RCC_APB2ENR_TIM11EN },
    { "SPI5",       0x40015000u, E_APB2, RCC_APB2ENR_SPI5EN },
    { "SPI6",       0x40015400u, E_APB2, RCC_APB2ENR_SPI6EN },
    { "SAI1",       0x40015800u, E_APB2, RCC_APB2ENR_SAI1EN },
    { "SAI2",       0x40015C00u, E_APB2, RCC_APB2ENR_SAI2EN },
    { "LTDC",       0x40016800u, E_APB2, RCC_APB2ENR_LTDCEN },
    /* APB1 */
    { "TIM2",       0x40000000u, E_APB1, RCC_APB1ENR_TIM2EN },
    { "TIM3",       0x40000400u, E_APB1, RCC_APB1ENR_TIM3EN },
    { "TIM4",       0x40000800u, E_APB1, RCC_APB1ENR_TIM4EN },
    { "TIM5",       0x40000C00u, E_APB1, RCC_APB1ENR_TIM5EN },
    { "TIM6",       0x40001000u, E_APB1, RCC_APB1ENR_TIM6EN },
    { "TIM7",       0x40001400u, E_APB1, RCC_APB1ENR_TIM7EN },
    { "TIM12",      0x40001800u, E_APB1, RCC_APB1ENR_TIM12EN },
    { "TIM13",      0x40001C00u, E_APB1, RCC_APB1ENR_TIM13EN },
    { "TIM14",      0x40002000u, E_APB1, RCC_APB1ENR_TIM14EN },
    { "LPTIM1",     0x40002400u, E_APB1, RCC_APB1ENR_LPTIM1EN },
    { "SPI2",       0x40003800u, E_APB1, RCC_APB1ENR_SPI2EN },
    { "SPI3",       0x40003C00u, E_APB1, RCC_APB1ENR_SPI3EN },
    { "SPDIFRX",    0x40004000u, E_APB1, RCC_APB1ENR_SPDIFRXEN },
    { "USART2",     0x40004400u, E_APB1, RCC_APB1ENR_USART2EN },
    { "USART3",     0x40004800u, E_APB1, RCC_APB1ENR_USART3EN },
    { "UART4",      0x40004C00u, E_APB1, RCC_APB1ENR_UART4EN },
    { "UART5",      0x40005000u, E_APB1, RCC_APB1ENR_UART5EN },
    { "I2C1",       0x40005400u, E_APB1, RCC_APB1ENR_I2C1EN },
    { "I2C2",       0x40005800u, E_APB1, RCC_APB1ENR_I2C2EN },
    { "I2C3",       0x40005C00u, E_APB1, RCC_APB1ENR_I2C3EN },
    { "I2C4",       0x40006000u, E_APB1, RCC_APB1ENR_I2C4EN },
    { "CAN1",       0x40006400u, E_APB1, RCC_APB1ENR_CAN1EN },
    { "CAN2",       0x40006800u, E_APB1, RCC_APB1ENR_CAN2EN },
    { "HDMI_CEC",   0x40006C00u, E_APB1, RCC_APB1ENR_CECEN },
    { "DAC",        0x40007400u, E_APB1, RCC_APB1ENR_DACEN },
    { "UART7",      0x40007800u, E_APB1, RCC_APB1ENR_UART7EN },
    { "UART8",      0x40007C00u, E_APB1, RCC_APB1ENR_UART8EN },
};

static volatile uint32_t *enr_of(uint8_t bus)
{
    switch (bus) {
    case E_AHB1: return &RCC->AHB1ENR;
    case E_AHB2: return &RCC->AHB2ENR;
    case E_AHB3: return &RCC->AHB3ENR;
    case E_APB1: return &RCC->APB1ENR;
    case E_APB2: return &RCC->APB2ENR;
    default:     return (volatile uint32_t *)0;
    }
}

static void sec_periph_probe(void)
{
    probe_section("9. 페리페럴 존재 실측 (RCC 인에이블 되읽기 + BusFault 프로빙)");

    probe_puts("방법: (1) RCC ENR 비트에 1을 쓰고 되읽는다(없는 페리페럴은 0으로 남는다).\n");
    probe_puts("      (2) 클럭을 켠 채 베이스 주소를 ldr.w로 읽는다(BusFault=슬레이브 없음).\n");
    probe_puts("      인터럽트를 끄고 읽으며, 전용 핸들러가 PC를 4바이트 전진시켜 복귀한다.\n\n");

    /* 표 셀은 ASCII로 쓴다. 한글은 UTF-8에서 바이트 수와 표시 폭이 달라 열이 어긋난다.
     * KEEP = RCC 인에이블 비트가 1로 유지됨, CLEAR0 = 1을 썼는데 0으로 되읽힘 */
    probe_puts("  peripheral    base         RCC-bit   base-read   value\n");
    probe_puts("  ------------  -----------  --------  ----------  ----------\n");

    uint32_t save_ahb1 = RCC->AHB1ENR, save_ahb2 = RCC->AHB2ENR, save_ahb3 = RCC->AHB3ENR;
    uint32_t save_apb1 = RCC->APB1ENR, save_apb2 = RCC->APB2ENR;
    uint32_t probe0 = s_fault_probe_count, hit0 = s_fault_hit_count;

    int n = (int)(sizeof(k_periphs) / sizeof(k_periphs[0]));
    for (int i = 0; i < n; i++) {
        const periph_t *p = &k_periphs[i];
        volatile uint32_t *enr = enr_of(p->bus);
        int clk_ok = -1;                       /* -1 = 해당 없음 */
        if (enr && p->mask) {
            uint32_t before = *enr;
            *enr = before | p->mask;
            (void)*enr;                        /* 인에이블 반영 지연 흡수 */
            clk_ok = ((*enr & p->mask) != 0u) ? 1 : 0;
        }
        uint32_t v = 0;
        int rd = safe_read32(p->base, &v);
        if (enr) { *enr = (p->bus == E_AHB1) ? save_ahb1 :
                          (p->bus == E_AHB2) ? save_ahb2 :
                          (p->bus == E_AHB3) ? save_ahb3 :
                          (p->bus == E_APB1) ? save_apb1 : save_apb2; (void)*enr; }

        probe_puts("  ");
        pad_to(p->name, 12);
        probe_printf("  0x%08X  ", p->base);
        pad_to(clk_ok < 0 ? "N/A" : (clk_ok ? "KEEP" : "CLEAR0"), 8);
        probe_puts("  ");
        pad_to(rd ? "OK" : "BUSFAULT", 10);
        if (rd) probe_printf("  0x%08X\n", v);
        else    probe_puts("  -\n");
    }

    /* 안전 복구 */
    RCC->AHB1ENR = save_ahb1; RCC->AHB2ENR = save_ahb2; RCC->AHB3ENR = save_ahb3;
    RCC->APB1ENR = save_apb1; RCC->APB2ENR = save_apb2;
    (void)RCC->APB2ENR;
    __DSB();

    probe_printf("\n  이 표에서 시도한 읽기 %u회 중 %u회가 BusFault로 실패했다.\n",
                 s_fault_probe_count - probe0, s_fault_hit_count - hit0);
    probe_puts("  ; CLEAR0 + BUSFAULT가 함께 나오면 그 페리페럴은 이 다이에 없다고 확정할 수 있다.\n");
    probe_puts("  ; KEEP + OK는 버스 응답이 있다는 뜻일 뿐, 기능 블록의 완전성까지 보증하지는 않는다.\n");
}

/* ------------------------------------------------------------------------- */
/* 10. GPIO 현재 설정                                                          */
/* ------------------------------------------------------------------------- */

static void sec_gpio(void)
{
    probe_section("10. GPIO 현재 설정 (보드 배선의 실측 증거)");

    GPIO_TypeDef *const ports[11] = {
        GPIOA, GPIOB, GPIOC, GPIOD, GPIOE, GPIOF, GPIOG, GPIOH, GPIOI, GPIOJ, GPIOK
    };
    static const char port_ch[11] = {'A','B','C','D','E','F','G','H','I','J','K'};

    uint32_t save_ahb1 = RCC->AHB1ENR;
    probe_printf("  부팅 시점의 GPIO 클럭 인에이블 상태 (AHB1ENR[10:0]) = 0x%03X: ",
                 save_ahb1 & 0x7FFu);
    for (int i = 0; i < 11; i++) if ((save_ahb1 >> i) & 1u) probe_printf("%c ", port_ch[i]);
    probe_puts("\n  ; 클럭이 켜진 포트 = 이미 누군가(BSP/probe_io)가 쓰고 있는 포트다.\n");
    probe_puts("  ; 레지스터를 읽기 위해 나머지 포트의 클럭도 잠시 켰다가 원래대로 되돌린다.\n\n");
    RCC->AHB1ENR = save_ahb1 | 0x7FFu;
    (void)RCC->AHB1ENR;

    for (int i = 0; i < 11; i++) {
        GPIO_TypeDef *g = ports[i];
        probe_printf("  GPIO%c MODER=0x%08X OTYPER=0x%08X OSPEEDR=0x%08X PUPDR=0x%08X\n",
                     port_ch[i], g->MODER, g->OTYPER, g->OSPEEDR, g->PUPDR);
        probe_printf("        AFR0 =0x%08X AFR1  =0x%08X IDR    =0x%08X\n",
                     g->AFR[0], g->AFR[1], g->IDR);
    }

    probe_puts("\n  [대체기능(AF)으로 설정된 핀 목록]\n");
    int af_total = 0;
    for (int i = 0; i < 11; i++) {
        GPIO_TypeDef *g = ports[i];
        uint32_t moder = g->MODER;
        int first = 1;
        for (int pin = 0; pin < 16; pin++) {
            if (FLD(moder, pin * 2, 2) != 2u) continue;
            if (first) { probe_printf("    P%c: ", port_ch[i]); first = 0; }
            uint32_t af = FLD(g->AFR[pin >> 3], (uint32_t)(pin & 7) * 4u, 4);
            probe_printf("%u=AF%u ", (uint32_t)pin, af);
            af_total++;
        }
        if (!first) probe_puts("\n");
    }
    probe_printf("    총 %d개 핀이 대체기능으로 설정되어 있다.\n", af_total);
    probe_puts("    ; 이 시점에는 USART1(PA9/PB7=AF7), SDRAM용 FMC(AF12), QSPI(AF9/AF10)가\n");
    probe_puts("    ; 초기화된 뒤다. AF12 핀들이 FMC 데이터/주소 버스 = SDRAM 배선의 증거다.\n");
    probe_puts("    ; 아날로그(MODER=11)나 입력(00)으로 남은 핀은 아직 아무도 안 쓰는 핀이다.\n");

    RCC->AHB1ENR = save_ahb1;
    (void)RCC->AHB1ENR;
    __DSB();
}

/* ------------------------------------------------------------------------- */
/* 11. 외부 메모리 컨트롤러                                                    */
/* ------------------------------------------------------------------------- */

static void sec_extmem(void)
{
    probe_section("11. 외부 메모리 컨트롤러 (FMC SDRAM / QUADSPI)");

    uint32_t save_ahb3 = RCC->AHB3ENR;
    RCC->AHB3ENR = save_ahb3 | RCC_AHB3ENR_FMCEN | RCC_AHB3ENR_QSPIEN;
    (void)RCC->AHB3ENR;

    uint32_t hclk = g_hclk_hz ? g_hclk_hz : 200000000u;

    for (int b = 0; b < 2; b++) {
        uint32_t sdcr = FMC_Bank5_6->SDCR[b];
        uint32_t sdtr = FMC_Bank5_6->SDTR[b];
        probe_printf("\n  [SDRAM Bank%d]\n", b + 1);
        kv(b ? "FMC_SDCR2" : "FMC_SDCR1", 0xA0000140u + 4u * (uint32_t)b, sdcr, NULL);
        uint32_t nc = FLD(sdcr, 0, 2) + 8u;
        uint32_t nr = FLD(sdcr, 2, 2) + 11u;
        uint32_t mw = FLD(sdcr, 4, 2);
        uint32_t nb = BIT_(sdcr, 6);
        uint32_t cas = FLD(sdcr, 7, 2);
        uint32_t wp = BIT_(sdcr, 9);
        uint32_t sdclk = FLD(sdcr, 10, 2);
        uint32_t rburst = BIT_(sdcr, 12);
        uint32_t rpipe = FLD(sdcr, 13, 2);
        uint32_t width = (mw == 0) ? 8u : (mw == 1) ? 16u : (mw == 2) ? 32u : 0u;
        probe_printf("    NC=%u (컬럼 %u비트)  NR=%u (로우 %u비트)  MWID=%u (%u비트 버스)\n",
                     FLD(sdcr, 0, 2), nc, FLD(sdcr, 2, 2), nr, mw, width);
        probe_printf("    NB=%u (%u뱅크)  CAS=%u (%u사이클)  WP=%u (%s)\n",
                     nb, nb ? 4u : 2u, cas, cas, wp, wp ? "쓰기 금지" : "쓰기 허용");
        probe_printf("    SDCLK=%u (%s)  RBURST=%u (%s)  RPIPE=%u (%u HCLK 지연)\n",
                     sdclk,
                     sdclk == 0 ? "클럭 정지" : sdclk == 2 ? "HCLK/2" : sdclk == 3 ? "HCLK/3" : "예약",
                     rburst, rburst ? "버스트 읽기 사용" : "미사용", rpipe, rpipe);
        if (width && nc && nr) {
            /* 용량 = 2^(로우+컬럼) x 뱅크수 x 버스폭/8 */
            uint32_t cells = (1u << (nr - 11u)) * 2048u;      /* 로우 수 */
            uint32_t cols  = 1u << nc;
            uint32_t banks = nb ? 4u : 2u;
            uint32_t bytes = cells * cols * banks * (width / 8u);
            probe_printf("    → 주소 공간 = 2^%u행 x 2^%u열 x %u뱅크 x %u바이트 = %u MB\n",
                         nr, nc, banks, width / 8u, bytes / (1024u * 1024u));
        }
        if (sdclk >= 2u && width) {
            uint32_t div = sdclk;   /* 2 또는 3 */
            uint32_t f = hclk / div;
            /* 64비트 연산은 __aeabi_uldivmod(libgcc)를 부르는데, 이 프로젝트의 링커
             * 스크립트가 libgcc.a를 DISCARD한다. 전부 32비트로만 계산한다. */
            uint32_t bw_x100 = (f / 10000u) * (width / 8u);   /* = MB/s x 100 */
            probe_printf("    → SDRAM 클럭 = HCLK %u Hz / %u = %u.%02u MHz\n",
                         hclk, div, f / 1000000u, (f % 1000000u) / 10000u);
            probe_printf("    → 이론 최대 대역폭 = %u.%02u MB/s (%u비트 버스, SDR 1전송/사이클)\n",
                         bw_x100 / 100u, bw_x100 % 100u, width);
        }
        kv(b ? "FMC_SDTR2" : "FMC_SDTR1", 0xA0000148u + 4u * (uint32_t)b, sdtr, NULL);
        probe_printf("    TMRD=%u TXSR=%u TRAS=%u TRC=%u TWR=%u TRP=%u TRCD=%u (단위: 사이클)\n",
                     FLD(sdtr, 0, 4) + 1u, FLD(sdtr, 4, 4) + 1u, FLD(sdtr, 8, 4) + 1u,
                     FLD(sdtr, 12, 4) + 1u, FLD(sdtr, 16, 4) + 1u, FLD(sdtr, 20, 4) + 1u,
                     FLD(sdtr, 24, 4) + 1u);
        probe_puts("    ; 필드값 + 1이 실제 사이클 수다. TRC/TRP는 Bank1 설정만 유효하다.\n");
    }

    uint32_t sdrtr = FMC_Bank5_6->SDRTR;
    uint32_t sdsr  = FMC_Bank5_6->SDSR;
    kv("FMC_SDRTR", 0xA0000154u, sdrtr, NULL);
    probe_printf("    COUNT=%u  CRE=%u  REIE=%u\n", FLD(sdrtr, 1, 13), BIT_(sdrtr, 0), BIT_(sdrtr, 14));
    {
        uint32_t sdclk_div = FLD(FMC_Bank5_6->SDCR[0], 10, 2);
        uint32_t cnt = FLD(sdrtr, 1, 13);
        uint32_t fs_mhz = (sdclk_div >= 2u) ? (hclk / sdclk_div) / 1000000u : 0u;
        if (cnt && fs_mhz) {
            /* period[ns] = (COUNT+1) / f = (COUNT+1)*1000 / f[MHz]. 32비트로 안전하다. */
            uint32_t period_ns = ((cnt + 1u) * 1000u) / fs_mhz;
            probe_printf("    → 리프레시 주기 = (%u+1) SDCLK / %u MHz = %u.%03u us\n",
                         cnt, fs_mhz, period_ns / 1000u, period_ns % 1000u);
            if (period_ns < 100000u) {
                uint32_t all_us = (period_ns * 8192u) / 1000u;   /* 8192행 전체 시간 [us] */
                probe_printf("    ; 8192행 전체를 %u.%02u ms마다 한 바퀴 (SDRAM 규격은 보통 64ms 이내)\n",
                             all_us / 1000u, (all_us % 1000u) / 10u);
            }
        }
    }
    kv("FMC_SDSR", 0xA0000158u, sdsr, NULL);
    probe_printf("    RE(리프레시 에러)=%u  MODES1=%u  MODES2=%u  BUSY=%u\n",
                 BIT_(sdsr, 0), FLD(sdsr, 1, 2), FLD(sdsr, 3, 2), BIT_(sdsr, 5));
    probe_puts("    ; MODESx: 00=정상, 01=셀프리프레시, 10=파워다운\n");

    probe_puts("\n  [QUADSPI]\n");
    uint32_t qcr = QUADSPI->CR, qdcr = QUADSPI->DCR, qccr = QUADSPI->CCR, qsr = QUADSPI->SR;
    kv("QUADSPI_CR", 0xA0001000u, qcr, NULL);
    uint32_t presc = FLD(qcr, 24, 8);
    probe_printf("    EN=%s ABORT=%u DMAEN=%u SSHIFT=%u DFM=%u FSEL=%u FTHRES=%u APMS=%u PMM=%u\n",
                 onoff(BIT_(qcr, 0)), BIT_(qcr, 1), BIT_(qcr, 2), BIT_(qcr, 4),
                 BIT_(qcr, 6), BIT_(qcr, 7), FLD(qcr, 8, 5), BIT_(qcr, 22), BIT_(qcr, 23));
    {
        uint32_t f = hclk / (presc + 1u);
        probe_printf("    PRESCALER=%u → QSPI 클럭 = HCLK / %u = %u.%02u MHz\n",
                     presc, presc + 1u, f / 1000000u, (f % 1000000u) / 10000u);
        probe_printf("    ; SSHIFT=%u (샘플 시프트 %s) — 고속에서 데이터 홀드 확보용\n",
                     BIT_(qcr, 4), BIT_(qcr, 4) ? "사용" : "미사용");
    }
    kv("QUADSPI_DCR", 0xA0001004u, qdcr, NULL);
    uint32_t fsize = FLD(qdcr, 16, 5);
    /* 용량 = 2^(FSIZE+1) 바이트. 1u<<32는 UB이므로 MB 단위로 직접 계산한다. */
    uint32_t qmb = (fsize >= 19u) ? (1u << (fsize - 19u)) : 0u;
    probe_printf("    FSIZE=%u → 플래시 용량 = 2^%u 바이트 = %u MB, CSHT=%u (칩셀렉트 하이 %u사이클), CKMODE=%u\n",
                 fsize, fsize + 1u, qmb,
                 FLD(qdcr, 8, 3), FLD(qdcr, 8, 3) + 1u, BIT_(qdcr, 0));
    if (qmb == 0u) probe_puts("    ; 1MB 미만이거나 QUADSPI가 초기화되지 않은 상태다\n");
    kv("QUADSPI_CCR", 0xA0001014u, qccr, NULL);
    static const char *modes[4] = {"없음", "1선", "2선", "4선"};
    probe_printf("    INSTRUCTION=0x%02X  IMODE=%s ADMODE=%s ABMODE=%s DMODE=%s\n",
                 FLD(qccr, 0, 8), modes[FLD(qccr, 8, 2)], modes[FLD(qccr, 10, 2)],
                 modes[FLD(qccr, 14, 2)], modes[FLD(qccr, 24, 2)]);
    probe_printf("    ADSIZE=%u (%u바이트 주소)  DCYC=%u 더미사이클  FMODE=%u (%s)  DDRM=%u SIOO=%u\n",
                 FLD(qccr, 12, 2), FLD(qccr, 12, 2) + 1u, FLD(qccr, 18, 5), FLD(qccr, 26, 2),
                 FLD(qccr, 26, 2) == 3 ? "메모리맵 모드" :
                 FLD(qccr, 26, 2) == 2 ? "자동 폴링" :
                 FLD(qccr, 26, 2) == 1 ? "간접 읽기" : "간접 쓰기",
                 BIT_(qccr, 31), BIT_(qccr, 28));
    kv("QUADSPI_SR", 0xA0001008u, qsr, NULL);
    probe_printf("    BUSY=%u FLEVEL=%u\n", BIT_(qsr, 5), FLD(qsr, 8, 6));
    probe_puts("    ; FMODE=11(메모리맵)이면 0x90000000에서 직접 코드/데이터를 읽을 수 있다.\n");
    probe_puts("    ; 그때 실효 대역폭은 (QSPI클럭 x 선 수 / 8)이 상한이며, 더미 사이클이 지연을 만든다.\n");

    RCC->AHB3ENR = save_ahb3;
    (void)RCC->AHB3ENR;
    __DSB();
}

/* ------------------------------------------------------------------------- */
/* 진입점                                                                     */
/* ------------------------------------------------------------------------- */

void probe_run_registers(void)
{
    probe_section("레지스터 덤프: 칩이 스스로 밝히는 사양");
    probe_puts("아래 값은 전부 이 보드의 실리콘에서 직접 읽은 것이다.\n");
    probe_puts("원시값 옆의 해석은 ARMv7-M ARM / Cortex-M7 TRM / RM0385의 필드 정의를 적용한 결과이며,\n");
    probe_puts("해석과 실제 동작이 어긋날 수 있는 항목은 그 사실을 함께 적었다.\n");

    fault_probe_install();

    sec_core_id();
    sec_cache();
    sec_tcm_bus();
    sec_fpu();
    sec_mpu();
    sec_debug();
    sec_factory();
    sec_clock();
    sec_periph_probe();
    sec_gpio();
    sec_extmem();

    fault_probe_remove();

    probe_section("레지스터 덤프 종료");
    probe_printf("BusFault 프로빙 총 %u회 시도, %u회 폴트. 마지막 BFSR=0x%02X\n",
                 s_fault_probe_count, s_fault_hit_count, s_probe_bfsr);
    probe_printf("VTOR을 0x%08X로 복원했고 SHCSR도 원상복구했다.\n", SCB->VTOR);
}
