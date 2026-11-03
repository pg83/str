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
};

const struct str_config str_configs[] = {
    {"sse2", 0, {
        .memcpy = memcpy_sse2, .memmove = memmove_sse2, .memset = memset_sse2, .memcmp = memcmp_sse2,
        .strlen = strlen_sse2, .strcpy = strcpy_sse2, .strcat = strcat_sse2, .strcmp = strcmp_sse2,
        .strspn = strspn_sse2, .strcspn = strcspn_sse2,
        .copy_large = copy_large_sse2, .fill_large = fill_large_sse2, .memcmp_large = memcmp_large_sse2,
        .strlen_tail = strlen_tail_sse2, .strcmp_tail = strcmp_tail_sse2, .strcpy_long = strcpy_long_sse2,
        .copy_huge = copy_huge_sse2, .fill_huge = fill_huge_sse2,
    }},
    {"sse42", STR_SSE42, {
        .memcpy = memcpy_sse2, .memmove = memmove_sse2, .memset = memset_sse2, .memcmp = memcmp_sse2,
        .strlen = strlen_sse2, .strcpy = strcpy_sse2, .strcat = strcat_sse2, .strcmp = strcmp_sse2,
        .strspn = strspn_sse42, .strcspn = strcspn_sse42,
        .copy_large = copy_large_sse2, .fill_large = fill_large_sse2, .memcmp_large = memcmp_large_sse2,
        .strlen_tail = strlen_tail_sse2, .strcmp_tail = strcmp_tail_sse2, .strcpy_long = strcpy_long_sse2,
        .copy_huge = copy_huge_sse2, .fill_huge = fill_huge_sse2,
    }},
    {"avx2", STR_SSE42 | STR_AVX2 | STR_ERMS, {
        .memcpy = memcpy_avx2, .memmove = memmove_avx2, .memset = memset_avx2, .memcmp = memcmp_avx2,
        .strlen = strlen_avx2, .strcpy = strcpy_avx2, .strcat = strcat_avx2, .strcmp = strcmp_avx2,
        .strspn = strspn_avx2, .strcspn = strcspn_avx2,
        .copy_large = copy_large_avx2, .fill_large = fill_large_avx2, .memcmp_large = memcmp_large_avx2,
        .strlen_tail = strlen_tail_avx2, .strcmp_tail = strcmp_tail_avx2, .strcpy_long = strcpy_long_avx2,
        .copy_huge = copy_huge_rep, .fill_huge = fill_huge_rep,
    }},
    /* AVX-512 at 256 bits: Skylake-SP and its Cascade/Cooper Lake siblings, where
       512-bit instructions lower the clock (glibc keeps to 256 bits there too).
       The same CPUs have the JCC erratum, so the code is the copy the post pass
       laid out with branches kept off 32-byte boundaries (*_jcc). */
    {"evex", STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ERMS, {
        .memcpy = memcpy_evex_jcc, .memmove = memmove_evex_jcc, .memset = memset_evex_jcc, .memcmp = memcmp_evex_jcc,
        .strlen = strlen_evex_jcc, .strcpy = strcpy_evex_jcc, .strcat = strcat_evex_jcc, .strcmp = strcmp_evex_jcc,
        .strspn = strspn_avx2_jcc, .strcspn = strcspn_avx2_jcc,
        .copy_large = copy_large_avx2_jcc, .fill_large = fill_large_avx2_jcc, .memcmp_large = memcmp_large_avx2_jcc,
        .strlen_tail = strlen_tail_avx2_jcc, .strcmp_tail = strcmp_tail_avx2_jcc, .strcpy_long = strcpy_long_avx2_jcc,
        .copy_huge = copy_huge_rep_jcc, .fill_huge = fill_huge_rep_jcc,
    }},
    /* AVX-512 at 512 bits: Zen 4 and 5, Ice Lake and later. The memory functions
       and strlen take one 64-byte block first; strcpy, strcat and strcmp keep
       the 16-byte first blocks (measured best), their strcpy_long and
       strcmp_tail kernels are 64-byte; the copy and memcmp kernels stay
       32-byte. */
    {"evex512", STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, {
        .memcpy = memcpy_evex512, .memmove = memmove_evex512, .memset = memset_evex512, .memcmp = memcmp_evex512,
        .strlen = strlen_evex512, .strcpy = strcpy_evex, .strcat = strcat_evex, .strcmp = strcmp_evex,
        .strspn = strspn_avx2, .strcspn = strcspn_avx2,
        .copy_large = copy_large_avx2, .fill_large = fill_large_evex512, .memcmp_large = memcmp_large_avx2,
        .strlen_tail = strlen_tail_evex512, .strcmp_tail = strcmp_tail_evex512, .strcpy_long = strcpy_long_evex512,
        .copy_huge = copy_huge_rep, .fill_huge = fill_huge_rep,
    }},
};
const unsigned str_nconfigs = sizeof str_configs / sizeof str_configs[0];
/* clang-format on */

static inline void cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}

static inline int bit(unsigned word, unsigned i) { return (word >> i) & 1; }

unsigned str_host(void)
{
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7)
        return 0;
    int intel = r[1] == 0x756e6547 && r[3] == 0x49656e69 && r[2] == 0x6c65746e;
    cpuid(1, 0, r);
    unsigned ecx1 = r[2];
    unsigned family = (r[0] >> 8) & 0xf, model = ((r[0] >> 4) & 0xf) | ((r[0] >> 12) & 0xf0);
    cpuid(7, 0, r);
    unsigned ebx7 = r[1];
    cpuid(0x80000001u, 0, r);
    unsigned ecx81 = r[2];
    unsigned have = bit(ebx7, 9) ? STR_ERMS : 0;   /* fast rep movsb/stosb */
    /* x86-64-v2: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF/SAHF */
    if (!(bit(ecx1, 0) && bit(ecx1, 9) && bit(ecx1, 19) && bit(ecx1, 20) && bit(ecx1, 23) && bit(ecx1, 13) && bit(ecx81, 0)))
        return have;
    have |= STR_SSE42;
    /* x86-64-v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE with YMM state */
    if (!(bit(ecx1, 27) && bit(ecx1, 28) && bit(ecx1, 29) && bit(ecx1, 12) && bit(ecx1, 22)
          && bit(ebx7, 5) && bit(ebx7, 3) && bit(ebx7, 8) && bit(ecx81, 5)))
        return have;
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 0x06) != 0x06)
        return have;
    have |= STR_AVX2;
    /* x86-64-v4: AVX512F, BW, CD, DQ, VL with opmask and ZMM state */
    if (!(bit(ebx7, 16) && bit(ebx7, 30) && bit(ebx7, 28) && bit(ebx7, 17) && bit(ebx7, 31) && (lo & 0xe0) == 0xe0))
        return have;
    have |= STR_AVX512;
    if (!(intel && family == 6 && model == 0x55))
        have |= STR_ZMM;
    return have;
}

/* The last configuration whose needs the host meets. */
const struct str_config *str_choose(void)
{
    unsigned host = str_host();
    const struct str_config *best = &str_configs[0];
    for (unsigned i = 1; i < str_nconfigs; i++)
        if ((str_configs[i].needs & ~host) == 0)
            best = &str_configs[i];
    return best;
}

void str_dispatch(void) { str_table = str_choose()->table; }

__attribute__((constructor)) static void select_table(void) { str_dispatch(); }

void *str_memcpy(void *d, const void *s, size_t n) { return str_table.memcpy(d, s, n); }
void *str_memmove(void *d, const void *s, size_t n) { return str_table.memmove(d, s, n); }
void *str_memset(void *d, int c, size_t n) { return str_table.memset(d, c, n); }
int str_memcmp(const void *a, const void *b, size_t n) { return str_table.memcmp(a, b, n); }
size_t str_strlen(const char *s) { return str_table.strlen(s); }
char *str_strcpy(char *d, const char *s) { return str_table.strcpy(d, s); }
char *str_strcat(char *d, const char *s) { return str_table.strcat(d, s); }
int str_strcmp(const char *a, const char *b) { return str_table.strcmp(a, b); }
size_t str_strspn(const char *s, const char *set) { return str_table.strspn(s, set); }
size_t str_strcspn(const char *s, const char *set) { return str_table.strcspn(s, set); }
void *str_copy_large(u8 *d, const u8 *s, size_t n) { return str_table.copy_large(d, s, n); }
void *str_fill_large(u8 *d, int c, size_t n) { return str_table.fill_large(d, c, n); }
int str_memcmp_large(const u8 *a, const u8 *b, size_t n) { return str_table.memcmp_large(a, b, n); }
size_t str_strlen_tail(const u8 *a, const u8 *s) { return str_table.strlen_tail(a, s); }
int str_strcmp_tail(const u8 *a, const u8 *b) { return str_table.strcmp_tail(a, b); }
void *str_strcpy_long(u8 *d, const u8 *s) { return str_table.strcpy_long(d, s); }
void *str_copy_huge(u8 *d, const u8 *s, size_t n) { return str_table.copy_huge(d, s, n); }
void *str_fill_huge(u8 *d, int c, size_t n) { return str_table.fill_huge(d, c, n); }
