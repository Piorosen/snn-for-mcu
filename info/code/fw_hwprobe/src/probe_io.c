/*
 * 출력 경로 구현: USART1(ST-LINK VCP) + 0x20035000 결과 버퍼.
 *
 * UART는 HAL을 쓰지 않고 레지스터를 직접 건드린다. HAL_UART_Transmit은 내부에서
 * HAL_GetTick()을 돌려 타임아웃을 재기 때문에 SysTick 인터럽트에 의존하는데,
 * 벤치마크 구간에서는 인터럽트를 끄고 돌리므로 그 의존이 없는 편이 안전하다.
 */
#include "probe.h"
#include "stm32f7xx.h"
#include <stdarg.h>

probe_result_t *const g_probe_result = (probe_result_t *)PROBE_RESULT_ADDR;

/* text[]가 쓸 수 있는 최대 바이트 (헤더 16바이트 제외, NUL 자리 1바이트 확보) */
#define TEXT_CAPACITY (PROBE_RESULT_SIZE - sizeof(probe_result_t) - 1u)

static int s_uart_ready;

void probe_io_init(uint32_t sysclk_hz, uint32_t pclk2_hz)
{
    /* 결과 버퍼 초기화: 여기서부터 실행 중임을 표시 */
    g_probe_result->magic     = PROBE_MAGIC_RUNNING;
    g_probe_result->length    = 0;
    g_probe_result->overflow  = 0;
    g_probe_result->sysclk_hz = sysclk_hz;
    g_probe_result->text[0]   = '\0';

    /* USART1: PA9=TX(AF7), PB7=RX(AF7) — STM32F746G-DISCO의 ST-LINK VCP 배선 */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    (void)RCC->APB2ENR;

    /* PA9: 대체기능 모드, 고속, 푸시풀, AF7 */
    GPIOA->MODER   = (GPIOA->MODER   & ~(3u << (9 * 2)))  | (2u << (9 * 2));
    GPIOA->OSPEEDR = (GPIOA->OSPEEDR & ~(3u << (9 * 2)))  | (3u << (9 * 2));
    GPIOA->PUPDR   = (GPIOA->PUPDR   & ~(3u << (9 * 2)));
    GPIOA->AFR[1]  = (GPIOA->AFR[1]  & ~(0xFu << ((9 - 8) * 4))) | (7u << ((9 - 8) * 4));

    /* PB7: 대체기능 모드, AF7 (수신은 쓰지 않지만 대칭을 위해 설정) */
    GPIOB->MODER   = (GPIOB->MODER   & ~(3u << (7 * 2)))  | (2u << (7 * 2));
    GPIOB->OSPEEDR = (GPIOB->OSPEEDR & ~(3u << (7 * 2)))  | (3u << (7 * 2));
    GPIOB->PUPDR   = (GPIOB->PUPDR   & ~(3u << (7 * 2)))  | (1u << (7 * 2));
    GPIOB->AFR[0]  = (GPIOB->AFR[0]  & ~(0xFu << (7 * 4))) | (7u << (7 * 4));

    /* USART1 클럭 소스를 PCLK2로 고정 (DCKCFGR2.USART1SEL = 00) */
    RCC->DCKCFGR2 &= ~RCC_DCKCFGR2_USART1SEL;

    USART1->CR1 = 0;
    /* OVER8=0 이므로 BRR = fCK / baud */
    USART1->BRR = (pclk2_hz + 115200u / 2u) / 115200u;
    USART1->CR2 = 0;
    USART1->CR3 = 0;
    USART1->CR1 = USART_CR1_TE | USART_CR1_UE;

    /* 송신기 준비 대기 */
    while (!(USART1->ISR & USART_ISR_TEACK)) { }
    s_uart_ready = 1;
}

static void uart_putc(char c)
{
    if (!s_uart_ready) return;
    while (!(USART1->ISR & USART_ISR_TXE)) { }
    USART1->TDR = (uint8_t)c;
}

static void buf_putc(char c)
{
    uint32_t n = g_probe_result->length;
    if (n >= TEXT_CAPACITY) {
        g_probe_result->overflow++;
        return;
    }
    g_probe_result->text[n] = c;
    g_probe_result->text[n + 1] = '\0';
    g_probe_result->length = n + 1;
}

static void emit(char c)
{
    if (c == '\n') {          /* 터미널 호환을 위해 CRLF로 내보낸다 */
        uart_putc('\r');
    }
    uart_putc(c);
    buf_putc(c);
}

void probe_puts(const char *s)
{
    while (*s) emit(*s++);
}

static void emit_uint(uint32_t v, unsigned base, int upper, int width, int zeropad, int left)
{
    char tmp[36];
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    } else {
        while (v && n < (int)sizeof(tmp)) {
            tmp[n++] = digits[v % base];
            v /= base;
        }
    }
    int pad = width - n;
    if (!left) for (int i = pad; i > 0; i--) emit(zeropad ? '0' : ' ');
    while (n > 0) emit(tmp[--n]);
    if (left) for (int i = pad; i > 0; i--) emit(' ');
}

void probe_printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') { emit(*p); continue; }
        p++;
        /* 플래그: '-'(좌측 정렬)와 '0'(제로 패딩). 표를 정렬해 찍을 때 %-16s를 자주
         * 쓰므로 좌측 정렬은 반드시 지원해야 한다. */
        int zeropad = 0, width = 0, left = 0;
        for (;;) {
            if (*p == '-')      { left = 1; p++; }
            else if (*p == '0') { zeropad = 1; p++; }
            else break;
        }
        while (*p >= '0' && *p <= '9') { width = width * 10 + (*p - '0'); p++; }
        while (*p == 'l' || *p == 'h' || *p == 'z') p++;  /* 모두 32비트로 취급 */
        switch (*p) {
        case 'd': case 'i': {
            int32_t v = va_arg(ap, int32_t);
            if (v < 0) { emit('-'); emit_uint((uint32_t)(-v), 10, 0, width ? width - 1 : 0, zeropad, left); }
            else       { emit_uint((uint32_t)v, 10, 0, width, zeropad, left); }
            break;
        }
        case 'u': emit_uint(va_arg(ap, uint32_t), 10, 0, width, zeropad, left); break;
        case 'x': emit_uint(va_arg(ap, uint32_t), 16, 0, width, zeropad, left); break;
        case 'X': emit_uint(va_arg(ap, uint32_t), 16, 1, width, zeropad, left); break;
        case 'p': probe_puts("0x"); emit_uint((uint32_t)(uintptr_t)va_arg(ap, void *), 16, 0, 8, 1, 0); break;
        case 'c': {
            char c = (char)va_arg(ap, int);
            if (!left) for (int i = width - 1; i > 0; i--) emit(' ');
            emit(c);
            if (left)  for (int i = width - 1; i > 0; i--) emit(' ');
            break;
        }
        case 's': {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            int len = 0; while (s[len]) len++;
            int pad = width - len;
            if (!left) for (int i = pad; i > 0; i--) emit(' ');
            while (*s) emit(*s++);
            if (left)  for (int i = pad; i > 0; i--) emit(' ');
            break;
        }
        case '%': emit('%'); break;
        case '\0': va_end(ap); return;
        default: emit('%'); emit(*p); break;
        }
    }
    va_end(ap);
}

void probe_section(const char *title)
{
    probe_puts("\n================================================================\n");
    probe_printf("== %s\n", title);
    probe_puts("================================================================\n");
}

void probe_kv32(const char *name, uint32_t addr, uint32_t value, const char *desc)
{
    probe_printf("%-16s @ 0x%08X = 0x%08X", name, addr, value);
    if (desc && desc[0]) probe_printf("  ; %s", desc);
    emit('\n');
}

void probe_flush(void)
{
    if (!s_uart_ready) return;
    while (!(USART1->ISR & USART_ISR_TC)) { }
}

void probe_finish(void)
{
    probe_flush();
    __asm volatile("dsb 0xF" ::: "memory");
    g_probe_result->magic = PROBE_MAGIC_DONE;
    __asm volatile("dsb 0xF" ::: "memory");
    /* 결과 버퍼는 SRAM1(캐시 대상)에 있다. D-cache가 켜진 채 끝나면 ST-LINK가
     * 보는 물리 메모리에 쓰기가 반영되지 않을 수 있으므로 반드시 클린한다. */
    SCB_CleanDCache();
    probe_flush();
}
