/*
 * common.h: shared helpers and short paths for the string library.
 *
 * Included by every implementation file. Everything here is SSE2 and compiles
 * under each file's own -march, so the same short path is emitted once per file.
 *
 * Memory safety rules, every function obeys them:
 *   - the page size is 4096 B; an aligned 16- or 32-byte load never crosses
 *     a page, so it may be issued whenever its first byte is readable;
 *   - an unaligned load is issued only after checking that it stays inside
 *     the page, or when it lies entirely inside [p, p+n);
 *   - groups of aligned loads are issued only when the whole group lies in
 *     one page (a 128-byte aligned group always does);
 *   - reads may extend past a NUL but never past the page that holds it;
 *   - stores never touch bytes outside [dst, dst+n).
 */
#ifndef STR_COMMON_H
#define STR_COMMON_H
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#define PAGE_SIZE 4096
#define PAGE_MASK (PAGE_SIZE - 1)

/* Copies and fills of at least REP_THRESHOLD bytes go to the copy_huge and
   fill_huge slots: rep movsb/stosb where dispatch.c found them fast (ERMS),
   the vector loops elsewhere; below it vector loops win on every host measured. */
#define REP_THRESHOLD 4096

typedef unsigned char u8;

#define NOINLINE __attribute__((noinline))
/* A condition the branch predictor cannot learn: a select (cmov), not a branch. */
#if defined(__has_builtin) && __has_builtin(__builtin_unpredictable)
#define UNPREDICTABLE(x) __builtin_unpredictable(x)
#else
#define UNPREDICTABLE(x) (x)
#endif
#define ALWAYS_INLINE static inline __attribute__((always_inline))
#ifdef __clang__
    #define NO_UNROLL _Pragma("clang loop unroll(disable)")
#else
    #define NO_UNROLL _Pragma("GCC unroll 1")
#endif

/* The one table of primitives: the public functions and the kernels they hand
   the long cases to. Every implementation does its own hot path and reaches
   any other primitive only through the str_* entry points below, which jump
   through this table, so dispatch.c can fill the
   slots from cpuid with any mix: a 64-byte first block can continue in a
   64-byte tail on one machine and in a 32-byte one on another. */
struct str_table {
    void *(*memcpy)(void *, const void *, size_t);
    void *(*memmove)(void *, const void *, size_t);
    void *(*memset)(void *, int, size_t);
    int (*memcmp)(const void *, const void *, size_t);
    size_t (*strlen)(const char *);
    char *(*strcpy)(char *, const char *);
    char *(*strcat)(char *, const char *);
    int (*strcmp)(const char *, const char *);
    size_t (*strspn)(const char *, const char *);
    size_t (*strcspn)(const char *, const char *);
    void *(*memchr)(const void *, int, size_t);
    size_t (*strnlen)(const char *, size_t);
    /* kernels; the contracts every implementation keeps: */
    void *(*copy_large)(u8 *d, const u8 *s, size_t n);         /* n > 64, any overlap */
    void *(*fill_large)(u8 *d, int c, size_t n);               /* n > 128 */
    int (*memcmp_large)(const u8 *a, const u8 *b, size_t n);   /* n > 128, the first 64 bytes equal */
    size_t (*strlen_tail)(const u8 *a, const u8 *s);           /* a 16-aligned, s < a, no NUL in [s, a) */
    int (*strcmp_tail)(const u8 *a, const u8 *b);              /* any pointers */
    void *(*strcpy_long)(u8 *d, const u8 *s, const u8 *a);     /* no NUL in [s, a), a 16-aligned, past s by 1..128, by 16 at most unless 32-aligned */
    void *(*copy_huge)(u8 *d, const u8 *s, size_t n);          /* n >= REP_THRESHOLD, no backward overlap */
    void *(*fill_huge)(u8 *d, int c, size_t n);                /* n >= REP_THRESHOLD */
    void *(*memchr_large)(const u8 *s, u8 c, size_t n);        /* no match in [s, A), n > A - s, A = align_down(s, 16) + 32; aligned loads only (C11) */
    size_t (*strnlen_large)(const u8 *s, const u8 *from, size_t n); /* strnlen(s, n) with no NUL in [s, A), A = align_down(from, 16) + 32 > s, n > A - s; aligned loads only */
};

extern struct str_table str_table;

/* What the host runs well, as bits of str_host(); dispatch.c picks a
   configuration from them. STR_ZMM: 512-bit registers cost nothing extra. */
enum { STR_SSE42 = 1, STR_AVX2 = 2, STR_AVX512 = 4, STR_ZMM = 8, STR_ERMS = 16 };
unsigned str_host(void);
/* The decisions are pure functions of a cpuid snapshot, so that the test can
   make them for any host on the one it runs on. */
struct str_cpuid {
    unsigned max_leaf;                             /* cpuid(0).eax */
    unsigned vendor_ebx, vendor_edx, vendor_ecx;   /* cpuid(0) */
    unsigned eax1, ecx1;                           /* cpuid(1) */
    unsigned ebx7;                                 /* cpuid(7, 0) */
    unsigned ecx81;                                /* cpuid(0x80000001) */
    unsigned xcr0;                                 /* xgetbv(0), 0 without OSXSAVE */
};

void str_cpuid_read(struct str_cpuid *c);
unsigned str_host_of(const struct str_cpuid *c);
const struct str_config *str_choose_for(unsigned host);
/* The configurations dispatch.c chooses from, by name; the test runs every
   one whose needs the host meets. */
struct str_config {
    const char *name;
    unsigned needs;
    struct str_table table;
};

extern const struct str_config str_configs[];
extern const unsigned str_nconfigs;
const struct str_config *str_choose(void);   /* the configuration for this host */
void str_dispatch(void);                     /* str_table = str_choose()->table */

/* The entry points: each jumps through its slot of the table. Implementations
   call each other only through these. */
void *str_memcpy(void *, const void *, size_t);
void *str_memmove(void *, const void *, size_t);
void *str_memset(void *, int, size_t);
int str_memcmp(const void *, const void *, size_t);
size_t str_strlen(const char *);
char *str_strcpy(char *, const char *);
char *str_strcat(char *, const char *);
int str_strcmp(const char *, const char *);
size_t str_strspn(const char *, const char *);
size_t str_strcspn(const char *, const char *);
void *str_memchr(const void *, int, size_t);
void *str_copy_large(u8 *d, const u8 *s, size_t n);
void *str_fill_large(u8 *d, int c, size_t n);
int str_memcmp_large(const u8 *a, const u8 *b, size_t n);
size_t str_strlen_tail(const u8 *a, const u8 *s);
int str_strcmp_tail(const u8 *a, const u8 *b);
void *str_strcpy_long(u8 *d, const u8 *s, const u8 *a);
void *str_copy_huge(u8 *d, const u8 *s, size_t n);
void *str_fill_huge(u8 *d, int c, size_t n);
void *str_memchr_large(const u8 *s, u8 c, size_t n);
size_t str_strnlen_large(const u8 *s, const u8 *from, size_t n);

/* derived.c: the functions made of the entry points above */
int str_bcmp(const void *, const void *, size_t);
void str_bcopy(const void *, void *, size_t);
void str_bzero(void *, size_t);
void str_explicit_bzero(void *, size_t);
void *str_mempcpy(void *, const void *, size_t);
size_t str_strnlen(const char *, size_t);
char *str_strncat(char *, const char *, size_t);
size_t str_strlcpy(char *, const char *, size_t);
size_t str_strlcat(char *, const char *, size_t);
char *str_strpbrk(const char *, const char *);
char *str_strsep(char **, const char *);
char *str_strtok_r(char *, const char *, char **);
char *str_strtok(char *, const char *);
char *str_strdup(const char *);
char *str_strndup(const char *, size_t);
void *str_memmem(const void *, size_t, const void *, size_t);
char *str_strstr(const char *, const char *);

/* sse2.c: every primitive, the baseline the table starts as */
void *memcpy_sse2(void *, const void *, size_t);
void *memmove_sse2(void *, const void *, size_t);
void *memset_sse2(void *, int, size_t);
int memcmp_sse2(const void *, const void *, size_t);
size_t strlen_sse2(const char *);
char *strcpy_sse2(char *, const char *);
char *strcat_sse2(char *, const char *);
int strcmp_sse2(const char *, const char *);
size_t strspn_sse2(const char *, const char *);
size_t strcspn_sse2(const char *, const char *);
void *memchr_sse2(const void *, int, size_t);
void *memchr_large_sse2(const u8 *s, u8 c, size_t n);
size_t strnlen_sse2(const char *, size_t);
size_t strnlen_large_sse2(const u8 *s, const u8 *from, size_t n);
void *copy_large_sse2(u8 *d, const u8 *s, size_t n);
void *fill_large_sse2(u8 *d, int c, size_t n);
int memcmp_large_sse2(const u8 *a, const u8 *b, size_t n);
size_t strlen_tail_sse2(const u8 *a, const u8 *s);
int strcmp_tail_sse2(const u8 *a, const u8 *b);
void *strcpy_long_sse2(u8 *d, const u8 *s, const u8 *a);
void *copy_huge_sse2(u8 *d, const u8 *s, size_t n);
void *fill_huge_sse2(u8 *d, int c, size_t n);
void *copy_huge_rep(u8 *d, const u8 *s, size_t n);   /* rep movsb: the ERMS hosts */
void *fill_huge_rep(u8 *d, int c, size_t n);         /* rep stosb */
size_t strspn_table_sse2(const u8 *s, const u8 *set);
size_t strcspn_table_sse2(const u8 *s, const u8 *set);
size_t find_byte_or_nul_sse2(const u8 *s, u8 c);
size_t skip_byte_sse2(const u8 *s, u8 c);
/* sse42.c */
size_t strspn_sse42(const char *, const char *);
size_t strcspn_sse42(const char *, const char *);
/* avx2.c */
void *memcpy_avx2(void *, const void *, size_t);
void *memmove_avx2(void *, const void *, size_t);
void *memset_avx2(void *, int, size_t);
int memcmp_avx2(const void *, const void *, size_t);
size_t strlen_avx2(const char *);
char *strcpy_avx2(char *, const char *);
char *strcat_avx2(char *, const char *);
int strcmp_avx2(const char *, const char *);
size_t strspn_avx2(const char *, const char *);
size_t strcspn_avx2(const char *, const char *);
size_t strspn_avx2_zen(const char *, const char *);   /* the shuffle scan's tables from 384 bytes, not 1.5 KiB */
size_t strcspn_avx2_zen(const char *, const char *);
void *memchr_avx2(const void *, int, size_t);
void *memchr_large_avx2(const u8 *s, u8 c, size_t n);
size_t strnlen_avx2(const char *, size_t);
size_t strnlen_large_avx2(const u8 *s, const u8 *from, size_t n);
void *copy_large_avx2(u8 *d, const u8 *s, size_t n);
void *fill_large_avx2(u8 *d, int c, size_t n);
int memcmp_large_avx2(const u8 *a, const u8 *b, size_t n);
size_t strlen_tail_avx2(const u8 *a, const u8 *s);
int strcmp_tail_avx2(const u8 *a, const u8 *b);
void *strcpy_long_avx2(u8 *d, const u8 *s, const u8 *a);
void *copy_huge_avx2(u8 *d, const u8 *s, size_t n);           /* the loop below 32 KiB, rep movsb beyond */
void *fill_huge_avx2(u8 *d, int c, size_t n);                  /* rep stosb below 8 MiB, non-temporal stores beyond */
/* evex.c: AVX-512 with 256-bit registers */
char *strcpy_evex(char *, const char *);
char *strcat_evex(char *, const char *);
int strcmp_evex(const char *, const char *);
int strcmp_tail_evex(const u8 *a, const u8 *b);
void *memchr_evex(const void *, int, size_t);
size_t strnlen_evex(const char *, size_t);
/* evex512.c: AVX-512 with 512-bit registers */
void *memcpy_evex512(void *, const void *, size_t);
void *memmove_evex512(void *, const void *, size_t);
void *memset_evex512(void *, int, size_t);
int memcmp_evex512(const void *, const void *, size_t);
size_t strlen_evex512(const char *);
void *memchr_evex512(const void *, int, size_t);
size_t strnlen_evex512(const char *, size_t);
void *fill_large_evex512(u8 *d, int c, size_t n);
size_t strlen_tail_evex512(const u8 *a, const u8 *s);
int strcmp_tail_evex512(const u8 *a, const u8 *b);
void *strcpy_long_evex512(u8 *d, const u8 *s, const u8 *a);

/* The same functions laid out by the post pass with the JCC-erratum padding
   (jcc-*.o, see dev/post.py), for Skylake-SP. */
void *memcpy_evex512_jcc(void *, const void *, size_t);
void *memmove_evex512_jcc(void *, const void *, size_t);
void *memset_evex512_jcc(void *, int, size_t);
int memcmp_evex512_jcc(const void *, const void *, size_t);
size_t strlen_evex512_jcc(const char *);
char *strcpy_evex_jcc(char *, const char *);
char *strcat_evex_jcc(char *, const char *);
int strcmp_evex_jcc(const char *, const char *);
int strcmp_tail_evex_jcc(const u8 *a, const u8 *b);
void *memchr_evex_jcc(const void *, int, size_t);
void *memchr_large_avx2_jcc(const u8 *s, u8 c, size_t n);
size_t strnlen_evex_jcc(const char *, size_t);
size_t strnlen_large_avx2_jcc(const u8 *s, const u8 *from, size_t n);
size_t strspn_avx2_jcc(const char *, const char *);
size_t strcspn_avx2_jcc(const char *, const char *);
size_t strspn_sse42_jcc(const char *, const char *);
size_t strcspn_sse42_jcc(const char *, const char *);
void *copy_large_avx2_jcc(u8 *d, const u8 *s, size_t n);
void *fill_large_avx2_jcc(u8 *d, int c, size_t n);
int memcmp_large_avx2_jcc(const u8 *a, const u8 *b, size_t n);
size_t strlen_tail_avx2_jcc(const u8 *a, const u8 *s);
int strcmp_tail_avx2_jcc(const u8 *a, const u8 *b);
void *strcpy_long_avx2_jcc(u8 *d, const u8 *s, const u8 *a);
void *copy_huge_rep_jcc(u8 *d, const u8 *s, size_t n);
void *copy_huge_avx2_jcc(u8 *d, const u8 *s, size_t n);
void *fill_huge_rep_jcc(u8 *d, int c, size_t n);

/* -------------------------------------------------------------- basics */

static inline size_t page_left(const void *p) {
    return PAGE_SIZE - ((uintptr_t)p & PAGE_MASK);
}

static inline const u8 *align_down(const void *p, size_t a) {
    return (const u8 *)((uintptr_t)p & ~(uintptr_t)(a - 1));
}

/* True when `bytes` bytes from both a and b stay inside their pages.
   Conservative: the OR can exceed the limit when each offset alone passes
   (a at 0x7c0 and b at 0x800, say). Fine for an entry's first block, whose
   fallback is a few bytes; a tail that compares byte by byte to the nearer
   page end on a miss must use the exact test, or that walk can be 2 KiB. */
static inline int both_in_page(const void *a, const void *b, size_t bytes) {
    return (((uintptr_t)a | (uintptr_t)b) & PAGE_MASK) <= PAGE_SIZE - bytes;
}

static inline int both_have(const void *a, const void *b, size_t bytes) {
    return page_left(a) >= bytes && page_left(b) >= bytes;
}

static inline __m128i load16(const void *p) {
    return _mm_loadu_si128((const __m128i *)p);
}

static inline __m128i load16a(const void *p) {
    return _mm_load_si128((const __m128i *)p);
}

static inline void store16(void *p, __m128i v) {
    _mm_storeu_si128((__m128i *)p, v);
}

static inline void store16a(void *p, __m128i v) {
    _mm_store_si128((__m128i *)p, v);
}

static inline uint64_t load64(const void *p) {
    uint64_t v;
    __builtin_memcpy(&v, p, 8);
    return v;
}

static inline void store64(void *p, uint64_t v) {
    __builtin_memcpy(p, &v, 8);
}

static inline uint32_t load32(const void *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

static inline void store32(void *p, uint32_t v) {
    __builtin_memcpy(p, &v, 4);
}

static inline uint16_t load16b(const void *p) {
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return v;
}

static inline void store16b(void *p, uint16_t v) {
    __builtin_memcpy(p, &v, 2);
}

static inline unsigned mask16(__m128i v) {
    return (unsigned)_mm_movemask_epi8(v);
}

static inline unsigned zero_mask16(__m128i v) {
    return mask16(_mm_cmpeq_epi8(v, _mm_setzero_si128()));
}

static inline unsigned eq_mask16(__m128i a, __m128i b) {
    return mask16(_mm_cmpeq_epi8(a, b));
}

static inline unsigned ne_mask16(__m128i a, __m128i b) {
    return eq_mask16(a, b) ^ 0xffffu;
}

/* Bit i set when a[i] != b[i] or a[i] == 0. */
static inline unsigned diff_or_zero_mask16(__m128i a, __m128i b) {
    return ne_mask16(a, b) | zero_mask16(a);
}

static inline __m128i splat16(int c) {
    return _mm_set1_epi8((char)c);
}

static inline unsigned ctz(unsigned m) { return (unsigned)__builtin_ctz(m); } /* uncoverable: m != 0 by contract, the zero branch is the compiler's */
static inline int byte_diff(const u8 *a, const u8 *b, size_t i) {
    return (int)a[i] - (int)b[i];
}

/* memcmp's answer from the mask of the differing bytes of [0, n), 1 <= n <=
   64 (bit i: byte i differs; bits at n and beyond do not matter): the first
   difference, or byte n - 1 when there is none, which is equal then. No
   branch on the data: half the comparisons of a workload come out equal,
   and a branch on that is a coin toss (FreeBSD's memcmp does the same). */
static inline int memcmp_answer(const u8 *a, const u8 *b, uint64_t m, size_t n) {
    return byte_diff(a, b, (size_t)__builtin_ctzll(m | 1ull << (n - 1)));
}

/* 65..128 bytes: lo for bytes 0..63, hi for bytes n - 64..n - 1. */
static inline int memcmp_answer2(const u8 *a, const u8 *b, uint64_t lo, uint64_t hi, size_t n) {
    size_t lo_i = (size_t)__builtin_ctzll(lo | 1ull << 63), hi_i = n - 64 + (size_t)__builtin_ctzll(hi | 1ull << 63);
    return byte_diff(a, b, UNPREDICTABLE(lo != 0) ? lo_i : hi_i);
}

static inline void rep_movsb(u8 *d, const u8 *s, size_t n) {
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(n) : : "memory");
}

static inline void rep_stosb(u8 *d, int c, size_t n) {
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(c) : "memory");
}

static inline int overlap_backward(const u8 *d, const u8 *s, size_t n) {
    return d > s && d < s + n;
}

/* Hide "returns its argument" from the compiler: otherwise it reuses the
   argument after a call instead of the result, which costs a saved register
   and prevents the wrappers from tail-calling the kernels. */
#define RETURN_OPAQUE(p) do { void *r_ = (p); __asm__("" : "+r"(r_)); return r_; } while (0)

/* ------------------------------------------------------- short copy/fill */

/* n <= 32. Every load happens before every store, so overlap is fine. The
   size classes in rising order, each but 16..32 jumped to: one taken branch
   at most, where the descending tree took four for one byte. Broadwell, the
   calls of pizdos's programs (dev/lab/pizdos.py trace, ns a call): CPython's
   dict copies of 1 to 3 bytes 2.69 to 1.37 (glibc 1.98), djvu's single bytes
   3.44 to 2.50 (glibc 2.84); the classes from 8 bytes up pay three compares
   more, CPython's fannkuch, its copies of 8 to 32 bytes, 2.55 to 3.28. */
static inline void copy_small(u8 *d, const u8 *s, size_t n) {
    if (__builtin_expect(n < 2, 0)) {
        if (n) {
            d[0] = s[0];
        }
    } else if (__builtin_expect(n < 4, 0)) {
        uint16_t x = load16b(s);
        u8 y = s[n - 1];
        store16b(d, x);
        d[n - 1] = y;
    } else if (__builtin_expect(n < 8, 0)) {
        uint32_t x = load32(s), y = load32(s + n - 4);
        store32(d, x);
        store32(d + n - 4, y);
    } else if (__builtin_expect(n < 16, 0)) {
        uint64_t x = load64(s), y = load64(s + n - 8);
        store64(d, x);
        store64(d + n - 8, y);
    } else {
        __m128i x = load16(s), y = load16(s + n - 16);
        store16(d, x);
        store16(d + n - 16, y);
    }
}

/* 33..64: all loads first, then all stores, overlap-safe. */
static inline void copy_33_64(u8 *d, const u8 *s, size_t n) {
    __m128i x0 = load16(s), x1 = load16(s + 16), x2 = load16(s + n - 32), x3 = load16(s + n - 16);
    store16(d, x0);
    store16(d + 16, x1);
    store16(d + n - 32, x2);
    store16(d + n - 16, x3);
}

/* n <= 32. The 8- and 4-byte stores go straight from the vector, 2..3
   bytes as three byte stores of c (glibc's): no move to a general register
   first. The classes in rising order as copy_small's: one taken branch at
   most, none for 16..32 (glibc's descending order with every class jumped
   to cost that one 1.27x in Broadwell's matrix and CPython's 16..32-byte
   fills 1.27x in pizdos's calls; this one, CPython's exceptions, 92% of
   them 16..32, 2.20 to 2.45 ns a call, glibc 2.51). */
static inline void fill_small(u8 *d, __m128i v, int c, size_t n) {
    if (__builtin_expect(n < 2, 0)) {
        if (n) {
            d[0] = (u8)c;
        }
    } else if (__builtin_expect(n < 4, 0)) {
        d[0] = (u8)c;
        d[1] = (u8)c;
        d[n - 1] = (u8)c;
    } else if (__builtin_expect(n < 8, 0)) {
        _mm_storeu_si32(d, v);
        _mm_storeu_si32(d + n - 4, v);
    } else if (__builtin_expect(n < 16, 0)) {
        _mm_storel_epi64((__m128i *)d, v);
        _mm_storel_epi64((__m128i *)(d + n - 8), v);
    } else {
        store16(d, v);
        store16(d + n - 16, v);
    }
}

/* 33..64. */
static inline void fill_33_64(u8 *d, __m128i v, size_t n) {
    store16(d, v);
    store16(d + 16, v);
    store16(d + n - 32, v);
    store16(d + n - 16, v);
}

/* --------------------------------------------------------- short compare */

/* 1 <= n <= 16 (the callers settle 0 before). A 16-byte load is taken only
   when both pointers have room in their pages; bit 16 bounds the search
   without a variable shift. */
static inline int memcmp_le16(const u8 *a, const u8 *b, size_t n) {
    if (both_in_page(a, b, 16)) {
        return memcmp_answer(a, b, ne_mask16(load16(a), load16(b)), n);
    }

    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return byte_diff(a, b, i);
        }
    }

    return 0;
}

/* 17..32: two overlapping blocks, their masks in one at their positions. */
static inline int memcmp_17_32(const u8 *a, const u8 *b, size_t n) {
    uint64_t m = ne_mask16(load16(a), load16(b)) | (uint64_t)ne_mask16(load16(a + n - 16), load16(b + n - 16)) << (n - 16);
    return memcmp_answer(a, b, m, n);
}

/* 33..64: four blocks, two from each end. */
static inline int memcmp_33_64(const u8 *a, const u8 *b, size_t n) {
    uint64_t m = ne_mask16(load16(a), load16(b)) | (uint64_t)ne_mask16(load16(a + 16), load16(b + 16)) << 16 |
                 (uint64_t)ne_mask16(load16(a + n - 32), load16(b + n - 32)) << (n - 32) |
                 (uint64_t)ne_mask16(load16(a + n - 16), load16(b + n - 16)) << (n - 16);
    return memcmp_answer(a, b, m, n);
}

/* First 16 bytes of strcmp when both strings have room in their pages.
   Returns 1 with the answer in *result, else 0 with the pointers advanced. */
static inline int strcmp_block16(const u8 **pa, const u8 **pb, int *result) {
    const u8 *a = *pa, *b = *pb;

    if (!both_in_page(a, b, 16)) {
        return 0;
    }

    unsigned m = diff_or_zero_mask16(load16(a), load16(b));

    if (m) {
        *result = byte_diff(a, b, ctz(m));
        return 1;
    }

    *pa = a + 16;
    *pb = b + 16;
    return 0;
}

/* Compare up to the nearer page end, eight bytes at a time (x ^ y marks the
   differing bytes, (x - 1..) & ~x & 0x80.. the first NUL exactly, the
   lowest set bit of either is the first stop), the rest byte by byte.
   Returns 1 with the answer in *result, or 0 with both pointers advanced to
   the page boundary. */
static inline int strcmp_to_page_end(const u8 **pa, const u8 **pb, int *result) {
    const u8 *a = *pa, *b = *pb;
    size_t n = page_left(a), k = page_left(b);

    if (n > k) {
        n = k;
    }

    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint64_t x = load64(a + i), y = load64(b + i);
        uint64_t stop = ((x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull) | (x ^ y);

        if (stop) {
            size_t j = i + (size_t)__builtin_ctzll(stop) / 8; /* uncoverable: stop != 0, the zero branch is the compiler's */
            *result = byte_diff(a, b, j); /* 0 when both are NUL */
            return 1;
        }
    }

    for (; i < n; i++) {
        if (a[i] != b[i]) {
            *result = byte_diff(a, b, i);
            return 1;
        }

        if (!a[i]) {
            *result = 0;
            return 1;
        }
    }

    *pa = a + n;
    *pb = b + n;
    return 0;
}

/* ------------------------------------------------------------- short NUL */

/* First 64 bytes of strlen with aligned blocks; the first block is aligned
   down to cover s and its mask is shifted by the misalignment. Returns the
   length, or SIZE_MAX with *next = the 16-aligned address 64 bytes on. */
static inline size_t strlen_prefix64(const u8 *s, const u8 **next) {
    const u8 *a = align_down(s, 16);
    unsigned m = zero_mask16(load16a(a)) >> ((uintptr_t)s & 15);

    if (m) {
        return ctz(m);
    }

    for (int k = 1; k < 4; k++) {
        a += 16;
        m = zero_mask16(load16a(a));

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }
    }

    *next = a + 16;
    return SIZE_MAX;
}

/* ---------------------------------------------------------- long strcpy */

/* One pass over a string that has no NUL before the 16-byte boundary after s
   (the caller checked at least that much). Each aligned source block is tested

   for NUL before the previous one is stored, and the last 16 bytes of the
   string are copied as one overlapping pair, as in FreeBSD's stpcpy. */
static inline void *strcpy_long16(u8 *d, const u8 *s) {
    const u8 *a = align_down(s, 16) + 16;
    __m128i next = load16a(a); /* aligned: never crosses a page */
    unsigned m = zero_mask16(next);

    if (m) {
        copy_small(d, s, (size_t)(a - s) + ctz(m) + 1); /* at most 32 bytes */
        RETURN_OPAQUE(d);
    }

    store16(d, load16(s)); /* [s, s+16) lies within the two blocks just checked */
    size_t off = (size_t)(a - s);

    for (;;) {
        __m128i block = next;
        a += 16;
        next = load16a(a);
        m = zero_mask16(next);
        store16(d + off, block);
        off += 16;

        if (m) {
            break;
        }
    }

    const u8 *nul = a + ctz(m);
    store16(d + (nul - s) - 15, load16(nul - 15)); /* the string is longer than 16 here */
    RETURN_OPAQUE(d);
}

#endif
