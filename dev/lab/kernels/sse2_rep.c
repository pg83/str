/*
 * The string instructions for every primitive they can do, every size, no
 * branch tree: the microcode decides. Where a CPU starts them fast (FSRM:
 * rep movsb; FSRS: rep stosb, see sse2_memset_rep.c; FSRC: repe cmpsb, which
 * Zen 5 reports), one path may beat a vector size tree the workload's sizes
 * mispredict.
 *   memcpy   rep movsb
 *   memmove  rep movsb forward, std; rep movsb; cld backward when d is
 *            inside [s, s + n)
 *   memcmp   repe cmpsb
 *   strlen   repne scasb for the NUL
 *   memchr   repne scasb for c within n
 *   strcpy   repne scasb for the length, rep movsb for it and the NUL
 *   strcat   two scans, one copy
 *   strcmp   repne scasb for a's length, repe cmpsb over it and the NUL
 * Result (workload traces, clock held, three runs, 2026-11-04), the time
 * against the host's configuration of ours:
 *              Zen 5   Skylake-SP  Broadwell
 *   memcpy     1.09    1.38        1.10      (glibc's __memcpy_erms the same)
 *   memmove    3.10    3.24        2.22      (std is slow everywhere)
 *   memset     0.99    1.39        1.03      (ours is memset_rep on Zen 5)
 *   memcmp     1.83    6.0         4.1       (FSRC on Zen 5 does not make it)
 *   strlen     13      14          4.7       (scasb: a byte a cycle or worse)
 *   memchr     1.7     4.0         2.7
 *   strcpy     12      8.7         2.2
 *   strcat     7.8     7.7         2.9
 *   strcmp     14      13          6.7
 * Only rep stosb on Zen 5 won a trace: the evex512r configuration had it for
 * memset until the programs' calls (dev/lab/pizdos.py trace) showed it
 * losing on 18 of 20 programs and tying on 2, 2026-11-10.
 */
#include <stddef.h>
#include <stdint.h>

void *kl_memcpy_sse2_rep(void *d, const void *s, size_t n) {
    void *dp = d;
    __asm__ volatile("rep movsb" : "+D"(dp), "+S"(s), "+c"(n) : : "memory");
    return d;
}

void *kl_memmove_sse2_rep(void *d, const void *s, size_t n) {
    void *dp = d;

    if ((uintptr_t)d - (uintptr_t)s >= n) { /* d before s, or past its end: forward is safe */
        __asm__ volatile("rep movsb" : "+D"(dp), "+S"(s), "+c"(n) : : "memory");
    } else if (n) {
        const char *sp = (const char *)s + n - 1;
        dp = (char *)d + n - 1;
        __asm__ volatile("std\n\trep movsb\n\tcld" : "+D"(dp), "+S"(sp), "+c"(n) : : "memory");
    }

    return d;
}

int kl_memcmp_sse2_rep(const void *a, const void *b, size_t n) {
    const unsigned char *p = a, *q = b;

    if (!n) {
        return 0;
    }

    /* cmpsb compares [rsi] with [rdi] and steps both: they end one past the
       byte that stopped it, the last one when all were equal (and equal then) */
    __asm__ volatile("repe cmpsb" : "+S"(p), "+D"(q), "+c"(n) : : "memory", "cc");
    return p[-1] - q[-1];
}

static inline size_t scan_nul(const char *s) {
    size_t n = (size_t)-1;
    __asm__ volatile("repne scasb" : "+D"(s), "+c"(n) : "a"(0) : "memory", "cc");
    return ~n - 1;
}

size_t kl_strlen_sse2_rep(const char *s) {
    return scan_nul(s);
}

void *kl_memchr_sse2_rep(const void *s, int c, size_t n) {
    const unsigned char *p = s;
    int found;

    if (!n) {
        return 0;
    }

    __asm__ volatile("repne scasb" : "+D"(p), "+c"(n), "=@ccz"(found) : "a"(c) : "memory");
    return found ? (void *)(p - 1) : 0;
}

char *kl_strcpy_sse2_rep(char *d, const char *s) {
    size_t n = scan_nul(s) + 1;
    char *dp = d;
    __asm__ volatile("rep movsb" : "+D"(dp), "+S"(s), "+c"(n) : : "memory");
    return d;
}

char *kl_strcat_sse2_rep(char *d, const char *s) {
    char *dp = d + scan_nul(d);
    size_t n = scan_nul(s) + 1;
    __asm__ volatile("rep movsb" : "+D"(dp), "+S"(s), "+c"(n) : : "memory");
    return d;
}

int kl_strcmp_sse2_rep(const char *a, const char *b) {
    const unsigned char *p = (const unsigned char *)a, *q = (const unsigned char *)b;
    size_t n = scan_nul(a) + 1; /* a's bytes and its NUL: b differs or ends within them */
    __asm__ volatile("repe cmpsb" : "+S"(p), "+D"(q), "+c"(n) : : "memory", "cc");
    return p[-1] - q[-1];
}
