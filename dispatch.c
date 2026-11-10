/*
 * dispatch.c: the table of primitives, chosen from cpuid, and the public entry
 * points. str_table starts as the SSE2 set, so a call before the constructor
 * ran is correct on any x86-64. The constructor replaces it with one of the
 * configurations below: each one names every slot explicitly, and they mix
 * freely, since implementations reach each other only through the table. A
 * Skylake-SP gets the 256-bit AVX-512 first blocks over the AVX2 kernels, a
 * Zen 4 the 512-bit ones, an AVX2 machine the AVX2 set; strspn and strcspn
 * are SSE4.2 (pcmpistri) on the SSE4.2 hosts and shuffle-based 32-byte scans
 * above; copies and fills of 4096 bytes
 * and more go to the copy_huge and fill_huge slots, rep movsb/stosb on the
 * hosts with ERMS.
 *
 * The entry points str_memcpy and so on jump through the table; they are the
 * public functions (a libc build renames them with -Dstr_memcpy=memcpy ... on
 * the command line) and the way implementations call each other.
 */
#include "common.h"

/* clang-format off */
struct str_table str_table = {
    .memcpy = memcpy_sse2, .memmove = memmove_sse2, .memset = memset_sse2, .memcmp = memcmp_sse2,
    .strlen = strlen_sse2, .strcpy = strcpy_sse2, .strcat = strcat_sse2, .strcmp = strcmp_sse2,
    .strspn = strspn_sse2, .strcspn = strcspn_sse2,
    .copy_large = copy_large_sse2, .fill_large = fill_large_sse2, .memcmp_large = memcmp_large_sse2,
    .strlen_tail = strlen_tail_sse2, .strcmp_tail = strcmp_tail_sse2, .strcpy_long = strcpy_long_sse2,
    .copy_huge = copy_huge_sse2, .fill_huge = fill_huge_sse2,
    .memchr = memchr_sse2, .memchr_large = memchr_large_sse2,
};

const struct str_config str_configs[] = {
    {"sse2", 0, {
        .memcpy = memcpy_sse2, .memmove = memmove_sse2, .memset = memset_sse2, .memcmp = memcmp_sse2,
        .strlen = strlen_sse2, .strcpy = strcpy_sse2, .strcat = strcat_sse2, .strcmp = strcmp_sse2,
        .strspn = strspn_sse2, .strcspn = strcspn_sse2,
        .copy_large = copy_large_sse2, .fill_large = fill_large_sse2, .memcmp_large = memcmp_large_sse2,
        .strlen_tail = strlen_tail_sse2, .strcmp_tail = strcmp_tail_sse2, .strcpy_long = strcpy_long_sse2,
        .copy_huge = copy_huge_sse2, .fill_huge = fill_huge_sse2,
        .memchr = memchr_sse2, .memchr_large = memchr_large_sse2,
    }},
    {"sse42", STR_SSE42, {
        .memcpy = memcpy_sse2, .memmove = memmove_sse2, .memset = memset_sse2, .memcmp = memcmp_sse2,
        .strlen = strlen_sse2, .strcpy = strcpy_sse2, .strcat = strcat_sse2, .strcmp = strcmp_sse2,
        .strspn = strspn_sse42, .strcspn = strcspn_sse42,
        .copy_large = copy_large_sse2, .fill_large = fill_large_sse2, .memcmp_large = memcmp_large_sse2,
        .strlen_tail = strlen_tail_sse2, .strcmp_tail = strcmp_tail_sse2, .strcpy_long = strcpy_long_sse2,
        .copy_huge = copy_huge_sse2, .fill_huge = fill_huge_sse2,
        .memchr = memchr_sse2, .memchr_large = memchr_large_sse2,
    }},
    {"avx2", STR_SSE42 | STR_AVX2 | STR_ERMS, {
        .memcpy = memcpy_avx2, .memmove = memmove_avx2, .memset = memset_avx2, .memcmp = memcmp_avx2,
        .strlen = strlen_avx2, .strcpy = strcpy_avx2, .strcat = strcat_avx2, .strcmp = strcmp_avx2,
        .strspn = strspn_avx2, .strcspn = strcspn_avx2,
        .copy_large = copy_large_avx2, .fill_large = fill_large_avx2, .memcmp_large = memcmp_large_avx2,
        .strlen_tail = strlen_tail_avx2, .strcmp_tail = strcmp_tail_avx2, .strcpy_long = strcpy_long_avx2,
        .copy_huge = copy_huge_avx2, .fill_huge = fill_huge_rep,
        .memchr = memchr_avx2, .memchr_large = memchr_large_avx2,
    }},
    /* Skylake-SP and its Cascade/Cooper Lake siblings: AVX-512 mostly at 256
       bits (glibc keeps to 256 bits there for the clock licenses), and the
       copies the post pass laid out with branches kept off 32-byte boundaries

       for the JCC erratum (*_jcc). memmove, memset, memcmp and strlen take
       their 64-byte entries: on a Xeon Gold 6148 with turbo, the workload's
       traces of them run 0.63x, 0.80x, 0.86x and 0.71x the time of the
       256-bit ones, and the mixed trace 0.90x (11.6 ns a call against 12.8;
       glibc 14.7). The core's clock is the same either way (3.20 GHz against
       3.18 by APERF/MPERF): the EVEX code at 256 bits takes the license too.
       memcpy's 64-byte entry: the mixed trace 0.987x with turbo, 0.971x with
       the clock held. strcpy, strcat, strspn and strcmp lose with 64-byte
       entries (up to 1.23x), memchr comes out even (dev/lab, 2026-11-04;
       dev/lab/tune.py proposes, the mixed trace decides). */
    {"evex", STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ERMS, {
        .memcpy = memcpy_evex512_jcc, .memmove = memmove_evex512_jcc, .memset = memset_evex512_jcc, .memcmp = memcmp_evex512_jcc,
        .strlen = strlen_evex512_jcc, .strcpy = strcpy_evex_jcc, .strcat = strcat_evex_jcc, .strcmp = strcmp_evex_jcc,
        .strspn = strspn_avx2_jcc, .strcspn = strcspn_avx2_jcc,
        .copy_large = copy_large_avx2_jcc, .fill_large = fill_large_avx2_jcc, .memcmp_large = memcmp_large_avx2_jcc,
        .strlen_tail = strlen_tail_avx2_jcc, .strcmp_tail = strcmp_tail_evex_jcc, .strcpy_long = strcpy_long_avx2_jcc,
        .copy_huge = copy_huge_rep_jcc, .fill_huge = fill_huge_rep_jcc,
        .memchr = memchr_evex_jcc, .memchr_large = memchr_large_avx2_jcc,
    }},
    /* AVX-512 at 512 bits: Zen 4 and 5, Ice Lake and later. The memory functions
       and strlen take one 64-byte block first; strcpy, strcat and strcmp keep
       the 16-byte first blocks (measured best), their strcpy_long and
       strcmp_tail kernels are 64-byte; the copy and memcmp kernels stay
       32-byte. Zen 5's memset was rep stosb, whose short start won the
       workload's memset trace (0.80x, the evex512r configuration, 2026-11-04);
       on the 20 programs' own calls it lost on 18 and tied on 2
       (dev/lab/pizdos.py trace, 2026-11-10: 0.77 of glibc's time over them
       against this configuration's 0.32). */
    {"evex512", STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, {
        .memcpy = memcpy_evex512, .memmove = memmove_evex512, .memset = memset_evex512, .memcmp = memcmp_evex512,
        .strlen = strlen_evex512, .strcpy = strcpy_evex, .strcat = strcat_evex, .strcmp = strcmp_evex,
        .strspn = strspn_avx2_zen, .strcspn = strcspn_avx2_zen,
        .copy_large = copy_large_avx2, .fill_large = fill_large_evex512, .memcmp_large = memcmp_large_avx2,
        .strlen_tail = strlen_tail_evex512, .strcmp_tail = strcmp_tail_evex512, .strcpy_long = strcpy_long_evex512,
        .copy_huge = copy_huge_avx2, .fill_huge = fill_huge_rep,
        .memchr = memchr_evex512, .memchr_large = memchr_large_avx2,
    }},
};

const unsigned str_nconfigs = sizeof str_configs / sizeof str_configs[0];
/* clang-format on */

static inline void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}

static inline int bit(unsigned word, unsigned i) {
    return (word >> i) & 1;
}

/* The instructions; everything the decisions need, so that they can be made
   again from any snapshot (the test feeds it other hosts). */
void str_cpuid_read(struct str_cpuid *c) {
    unsigned r[4];
    cpuid(0, 0, r);
    c->max_leaf = r[0];
    c->vendor_ebx = r[1];
    c->vendor_edx = r[3];
    c->vendor_ecx = r[2];
    /* leaf 7 past the highest leaf reads as the highest one on Intel, garbage
       either way: str_host_of looks at max_leaf before it believes ebx7 */
    cpuid(1, 0, r);
    c->eax1 = r[0];
    c->ecx1 = r[2];
    cpuid(7, 0, r);
    c->ebx7 = r[1];
    cpuid(0x80000001u, 0, r);
    c->ecx81 = r[2];
    c->xcr0 = 0;

    if (bit(c->ecx1, 27)) { /* uncoverable: xgetbv faults without OSXSAVE, a CPU this test does not run on */
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        c->xcr0 = lo;
    }
}

unsigned str_host_of(const struct str_cpuid *c) {
    if (c->max_leaf < 7) {
        return 0;
    }

    int intel = c->vendor_ebx == 0x756e6547 && c->vendor_edx == 0x49656e69 && c->vendor_ecx == 0x6c65746e;
    unsigned family = (c->eax1 >> 8) & 0xf, model = ((c->eax1 >> 4) & 0xf) | ((c->eax1 >> 12) & 0xf0);
    unsigned ecx1 = c->ecx1, ebx7 = c->ebx7, ecx81 = c->ecx81;
    unsigned have = bit(ebx7, 9) ? STR_ERMS : 0;   /* fast rep movsb/stosb */

    /* x86-64-v2: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF/SAHF */
    if (!(bit(ecx1, 0) && bit(ecx1, 9) && bit(ecx1, 19) && bit(ecx1, 20) && bit(ecx1, 23) && bit(ecx1, 13) && bit(ecx81, 0))) {
        return have;
    }

    have |= STR_SSE42;

    /* x86-64-v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE with YMM state */
    if (!(bit(ecx1, 27) && bit(ecx1, 28) && bit(ecx1, 29) && bit(ecx1, 12) && bit(ecx1, 22)
          && bit(ebx7, 5) && bit(ebx7, 3) && bit(ebx7, 8) && bit(ecx81, 5))) {
        return have;
    }

    if ((c->xcr0 & 0x06) != 0x06) {
        return have;
    }

    have |= STR_AVX2;

    /* x86-64-v4: AVX512F, BW, CD, DQ, VL with opmask and ZMM state */
    if (!(bit(ebx7, 16) && bit(ebx7, 30) && bit(ebx7, 28) && bit(ebx7, 17) && bit(ebx7, 31) && (c->xcr0 & 0xe0) == 0xe0)) {
        return have;
    }

    have |= STR_AVX512;
    /* 512-bit registers lower the clock of Skylake-SP and its Cascade/Cooper Lake
       siblings (family 6 model 0x55); glibc keeps to 256 bits there too */

    if (!(intel && family == 6 && model == 0x55)) {
        have |= STR_ZMM;
    }

    return have;
}

unsigned str_host(void) {
    struct str_cpuid c;
    str_cpuid_read(&c);
    return str_host_of(&c);
}

/* The last configuration whose needs the host meets. */
const struct str_config *str_choose_for(unsigned host) {
    const struct str_config *best = &str_configs[0];

    for (unsigned i = 1; i < str_nconfigs; i++) {
        if ((str_configs[i].needs & ~host) == 0) {
            best = &str_configs[i];
        }
    }

    return best;
}

const struct str_config *str_choose(void) {
    return str_choose_for(str_host());
}

void str_dispatch(void) {
    str_table = str_choose()->table;
}

__attribute__((constructor)) static void select_table(void) {
    str_dispatch();
}

void *str_memcpy(void *d, const void *s, size_t n) {
    return str_table.memcpy(d, s, n);
}

void *str_memmove(void *d, const void *s, size_t n) {
    return str_table.memmove(d, s, n);
}

void *str_memset(void *d, int c, size_t n) {
    return str_table.memset(d, c, n);
}

int str_memcmp(const void *a, const void *b, size_t n) {
    return str_table.memcmp(a, b, n);
}

size_t str_strlen(const char *s) {
    return str_table.strlen(s);
}

char *str_strcpy(char *d, const char *s) {
    return str_table.strcpy(d, s);
}

char *str_strcat(char *d, const char *s) {
    return str_table.strcat(d, s);
}

int str_strcmp(const char *a, const char *b) {
    return str_table.strcmp(a, b);
}

size_t str_strspn(const char *s, const char *set) {
    return str_table.strspn(s, set);
}

size_t str_strcspn(const char *s, const char *set) {
    return str_table.strcspn(s, set);
}

void *str_memchr(const void *s, int c, size_t n) {
    return str_table.memchr(s, c, n);
}

void *str_copy_large(u8 *d, const u8 *s, size_t n) {
    return str_table.copy_large(d, s, n);
}

void *str_fill_large(u8 *d, int c, size_t n) {
    return str_table.fill_large(d, c, n);
}

int str_memcmp_large(const u8 *a, const u8 *b, size_t n) {
    return str_table.memcmp_large(a, b, n);
}

size_t str_strlen_tail(const u8 *a, const u8 *s) {
    return str_table.strlen_tail(a, s);
}

int str_strcmp_tail(const u8 *a, const u8 *b) {
    return str_table.strcmp_tail(a, b);
}

void *str_strcpy_long(u8 *d, const u8 *s, const u8 *a) {
    return str_table.strcpy_long(d, s, a);
}

void *str_copy_huge(u8 *d, const u8 *s, size_t n) {
    return str_table.copy_huge(d, s, n);
}

void *str_fill_huge(u8 *d, int c, size_t n) {
    return str_table.fill_huge(d, c, n);
}

void *str_memchr_large(const u8 *s, u8 c, size_t n) {
    return str_table.memchr_large(s, c, n);
}
