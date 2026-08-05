/*
 * STM32F746G-DISCO 하드웨어 프로브 펌웨어 - 공통 인터페이스.
 *
 * 출력 경로는 두 갈래다:
 *   1) USART1 (PA9/PB7) → ST-LINK/V2-1 가상 COM 포트 → 호스트에서 실시간 수신
 *   2) 고정 주소(0x20035000)의 결과 버퍼 → 실행이 끝난 뒤 ST-LINK로 통째 회수
 * 두 경로에 같은 텍스트가 나가므로, 시리얼을 놓쳐도 결과를 잃지 않는다.
 */
#ifndef PROBE_H
#define PROBE_H

#include <stdint.h>
#include <stddef.h>

/* ---- 결과 버퍼 (링커 RESULT 영역) ------------------------------------------ */
#define PROBE_RESULT_ADDR   0x20035000u
#define PROBE_RESULT_SIZE   (92u * 1024u)

/* ---- 진단 영역 (링커 DIAG 영역) --------------------------------------------
 * 결과 버퍼와 물리적으로 분리되어 있다. 로그가 아무리 쌓여도 여기는 덮이지 않으므로,
 * 펌웨어가 죽거나 멈춰도 ST-LINK로 이 주소를 읽어 원인을 알 수 있다. */
#define PROBE_DIAG_ADDR       0x20034000u
#define PROBE_DIAG_SIZE       4096u
#define PROBE_DIAG_BOOT_STEP  (PROBE_DIAG_ADDR + 0x000u)  /* 초기화 진행 단계 */
#define PROBE_DIAG_FAULT_LOG  (PROBE_DIAG_ADDR + 0x010u)  /* 예외 핸들러 기록 (8워드) */
#define PROBE_DIAG_REGS_LOG   (PROBE_DIAG_ADDR + 0x040u)  /* probe_regs 폴트 프로빙 */
#define PROBE_DIAG_MEM_LOG    (PROBE_DIAG_ADDR + 0x080u)  /* probe_mem 폴트 프로빙 */
/* 버퍼 선두 매직: 호스트가 "측정이 실제로 끝났는지"를 판별하는 근거 */
#define PROBE_MAGIC_RUNNING 0x50524231u   /* "PRB1" - 실행 중 */
#define PROBE_MAGIC_DONE    0x50524244u   /* "PRBD" - 정상 종료 */

typedef struct {
    uint32_t magic;        /* PROBE_MAGIC_* */
    uint32_t length;       /* text[]에 쌓인 바이트 수 */
    uint32_t overflow;     /* 버퍼가 넘쳐 잘린 바이트 수 */
    uint32_t sysclk_hz;    /* 이 실행의 SYSCLK (호스트가 사이클→시간 환산에 사용) */
    char     text[];       /* 로그 본문 (NUL 종료 보장) */
} probe_result_t;

extern probe_result_t *const g_probe_result;

/* ---- 출력 --------------------------------------------------------------- */
void probe_io_init(uint32_t sysclk_hz, uint32_t pclk2_hz);
void probe_puts(const char *s);
/* 지원 포맷: %s %c %d %u %x %X %% , 폭/제로패딩(%08x, %6u), %l 접두(무시)
 * 부동소수점은 지원하지 않는다. 소수는 probe_printf("%u.%03u", a, b) 식으로 낼 것. */
void probe_printf(const char *fmt, ...);
void probe_section(const char *title);   /* 구분선이 있는 섹션 헤더 */
void probe_kv32(const char *name, uint32_t addr, uint32_t value, const char *desc);
void probe_flush(void);                  /* UART 송신 완료 대기 */
void probe_finish(void);                 /* magic을 DONE으로 바꾸고 flush */

/* ---- 타이밍 ------------------------------------------------------------- */
/* DWT_CYCCNT 기반. probe_time_init()이 DEMCR.TRCENA와 DWT_CTRL.CYCCNTENA를 켠다. */
void     probe_time_init(void);
uint32_t probe_cycles_now(void);
/* 측정 구간 앞뒤로 파이프라인/메모리 순서를 고정한다 */
#define PROBE_BARRIER()  do { __asm volatile("dsb 0xF" ::: "memory"); \
                              __asm volatile("isb 0xF" ::: "memory"); } while (0)

/* DWT 이벤트 카운터 (CPI/EXC/SLEEP/LSU/FOLD) 스냅샷 */
typedef struct {
    uint32_t cyc, cpi, exc, sleep, lsu, fold;
} probe_dwt_counters_t;
void probe_dwt_counters_reset(void);
void probe_dwt_counters_read(probe_dwt_counters_t *out);

/* ---- 시스템 상태 조작 헬퍼 (벤치마크에서 조건을 바꿔가며 재측정할 때 사용) --- */
void probe_icache_enable(void);
void probe_icache_disable(void);
void probe_dcache_enable(void);
void probe_dcache_disable(void);
void probe_art_enable(void);      /* FLASH_ACR.ARTEN + PRFTEN */
void probe_art_disable(void);
void probe_art_reset(void);       /* ART 캐시 무효화 */
int  probe_set_sysclk(uint32_t target_mhz);  /* 200 또는 216. 0=성공 */

/* 현재 클럭 상태 (probe_set_sysclk 이후 갱신됨) */
extern uint32_t g_sysclk_hz;
extern uint32_t g_hclk_hz;
extern uint32_t g_pclk1_hz;
extern uint32_t g_pclk2_hz;

/* ---- 메모리 영역 서술 ---------------------------------------------------- */
typedef enum {
    MEMKIND_TCM,        /* ITCM/DTCM: 코어 직결, 캐시 대상 아님 */
    MEMKIND_SRAM,       /* SRAM1/SRAM2: AXI 경유, 캐시 대상 */
    MEMKIND_FLASH,      /* 내장 플래시 (AXIM 또는 ITCM 별칭) */
    MEMKIND_EXT,        /* 외부 SDRAM / QSPI */
    MEMKIND_BKP,        /* 백업 SRAM */
} probe_memkind_t;

typedef struct {
    const char     *name;
    uint32_t        base;
    uint32_t        size;      /* 데이터시트상 크기 */
    probe_memkind_t kind;
    int             writable;
    int             available;  /* 런타임에 실제로 쓸 수 있는지 (SDRAM/QSPI 초기화 결과) */
} probe_region_t;

/* probe_mem.c가 채운다. 개수는 probe_region_count() */
const probe_region_t *probe_regions(void);
int probe_region_count(void);

/* ---- 각 측정 모듈의 진입점 ---------------------------------------------- */
/* probe_regs.c: 코어/시스템/페리페럴 레지스터를 남김없이 덤프하고 해석한다 */
void probe_run_registers(void);
/* probe_mem.c: 메모리 실제 크기, 대역폭, 레이턴시, 캐시 효과를 측정한다 */
void probe_run_memory(void);
/* probe_cpu.c: 명령어 처리량, FPU/DSP, 코드 배치별 실행 속도를 측정한다 */
void probe_run_cpu(void);

/* 외부 메모리 초기화 (main에서 호출, 실패해도 계속 진행) */
int probe_init_sdram(void);   /* 0 = 성공 */
int probe_init_qspi(void);    /* 0 = 성공 (메모리맵 모드까지) */

#endif /* PROBE_H */
