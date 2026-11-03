/* Correctness test against glibc (lib/build.py links the library objects next
   to static glibc). Covers exhaustive small sizes and alignments, selected
   large sizes, every overlap direction, guard pages on both ends, and random
   fuzzing; the whole suite runs for each level's table the host supports and
   for the dispatched table. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common.h"

/* Every call goes through junk(): the unused argument registers hold garbage, so
   a function that reads an argument it was not given (a tail call's second
   argument prepared on the wrong path, say) cannot pass by luck. */
static inline int junk(void)
{
    __asm__ volatile("movq $0x5a5a5a5a5a5a5a5a, %%rsi\n\tmovq $0x5a5a5a5a5a5a5a5a, %%rdx\n\tmovq $0x5a5a5a5a5a5a5a5a, %%rcx" ::: "rsi", "rdx", "rcx");
    return 0;
}
#define T (junk(), Table)

/* Every check goes through T; main runs the whole suite once per table. */
static const struct str_table *Table;

static size_t checks;
static const char *phase = "start";
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL %s line %d check %zu: %s\n", phase, __LINE__, checks, #x); exit(2); } } while (0)
static int sign(int v) { return (v > 0) - (v < 0); }
static uint32_t state = 0x9e3779b9;
static uint32_t rnd(void) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static void pattern(unsigned char *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (unsigned char)(1 + (i * 37 + seed * 11 + (i >> 7)) % 255);
}

static unsigned char *guard_pages(size_t pages)
{
    /* PROT_NONE page, then `pages` writable pages, then PROT_NONE page. */
    unsigned char *m = mmap(0, (pages + 2) * 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(m != MAP_FAILED);
    CHECK(!mprotect(m + 4096, pages * 4096, PROT_READ | PROT_WRITE));
    return m + 4096;
}

static void check_copy(unsigned char *d, const unsigned char *s, size_t n, unsigned char *ref, size_t span)
{
    /* d and s may overlap; ref holds a copy of the whole span containing both. */
    memcpy(ref, d < s ? d : s, span);
    size_t doff = (size_t)(d - (d < s ? d : s)), soff = (size_t)(s - (d < s ? d : s));
    memmove(ref + doff, ref + soff, n);
    CHECK(T->memmove(d, s, n) == d);
    CHECK(!memcmp(d < s ? d : s, ref, span));
}

static void suite(void)
{
    CHECK(sysconf(_SC_PAGESIZE) == 4096);
    const size_t cap = 1u << 21;
    unsigned char *a = malloc(cap + 512), *b = malloc(cap + 512), *ref = malloc(cap + 512), *ref2 = malloc(cap + 512);
    CHECK(a && b && ref && ref2);

    phase = "small sizes, all alignments";
    for (size_t n = 0; n <= 300; n++)
        for (size_t ao = 0; ao < 48; ao += (ao < 20 ? 1 : 7))
            for (size_t bo = 0; bo < 48; bo += (bo < 20 ? 1 : 7)) {
                unsigned char *s = a + ao, *d = b + bo;
                pattern(s, n + 2, (unsigned)n);
                memset(d, 0x55, n + 2);
                CHECK(T->memcpy(d, s, n) == d);
                CHECK(!memcmp(d, s, n) && d[n] == 0x55);
                CHECK(T->memcmp(d, s, n) == 0);
                if (n) {
                    size_t at = rnd() % n;
                    unsigned char old = d[at];
                    d[at] = (unsigned char)(old ^ 0x80);
                    CHECK(sign(T->memcmp(d, s, n)) == sign(memcmp(d, s, n)));
                    CHECK(sign(T->memcmp(s, d, n)) == sign(memcmp(s, d, n)));
                    d[at] = old;
                    d[n - 1] = (unsigned char)(old + 1);
                    CHECK(sign(T->memcmp(s, d, n)) == sign(memcmp(s, d, n)));
                }
                const int fills[] = {0, 0x5a, -1, 0x1ff};
                for (size_t k = 0; k < 4; k++) {
                    memset(d, 0x55, n + 2);
                    CHECK(T->memset(d, fills[k], n) == d);
                    memset(ref, fills[k], n);
                    CHECK(!memcmp(d, ref, n) && d[n] == 0x55);
                }
                s[n] = 0;
                CHECK(T->strlen((char *)s) == n);
                memset(d, 0x55, n + 2);
                CHECK(T->strcpy((char *)d, (char *)s) == (char *)d);
                CHECK(!memcmp(d, s, n + 1) && d[n + 1] == 0x55);
                CHECK(T->strcmp((char *)s, (char *)d) == 0);
                if (n) {
                    size_t at = rnd() % n;
                    d[at] = (unsigned char)(1 + rnd() % 255);
                    CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                    CHECK(sign(T->strcmp((char *)d, (char *)s)) == sign(strcmp((char *)d, (char *)s)));
                    d[at] = 0; /* d is a prefix of s */
                    CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                    CHECK(sign(T->strcmp((char *)d, (char *)s)) == sign(strcmp((char *)d, (char *)s)));
                }
                d[0] = 'x'; d[1] = 0; d[n + 2] = 0x55;
                CHECK(T->strcat((char *)d, (char *)s) == (char *)d);
                CHECK(d[0] == 'x' && !memcmp(d + 1, s, n + 1) && d[n + 2] == 0x55);
            }

    phase = "overlap";
    const size_t sizes[] = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 200,
                            255, 256, 257, 300, 511, 512, 513, 1000, 2047, 2048, 2049, 4095, 4096, 4097, 8192,
                            65535, 65536, 65537, 1000003, 1048576, 1048577};
    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        size_t n = sizes[k];
        const long deltas[] = {-4097, -4096, -300, -257, -256, -129, -128, -65, -64, -33, -32, -17, -16, -15, -8, -3, -1, 0,
                               1, 3, 8, 15, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 300, 4096, 4097};
        for (size_t j = 0; j < sizeof deltas / sizeof *deltas; j++)
            for (size_t align = 0; align < 3; align++) {
                unsigned char *s = a + 8192 + align * 5;
                unsigned char *d = s + deltas[j];
                size_t span = n + (size_t)(deltas[j] < 0 ? -deltas[j] : deltas[j]);
                pattern(a, span + 16384, (unsigned)(n + j));
                check_copy(d, s, n, ref, span);
                pattern(a, span + 16384, (unsigned)(n + j + 1));
                if (d + n <= s || s + n <= d) {
                    memcpy(ref, s, n);
                    CHECK(T->memcpy(d, s, n) == d);
                    CHECK(!memcmp(d, ref, n));
                }
            }
    }

    phase = "large sizes";
    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++)
        for (size_t ao = 0; ao < 64; ao += 13)
            for (size_t bo = 0; bo < 64; bo += 17) {
                size_t n = sizes[k];
                unsigned char *s = a + ao, *d = b + bo;
                pattern(s, n + 1, (unsigned)k);
                memset(d, 0x55, n + 1);
                CHECK(T->memcpy(d, s, n) == d && !memcmp(d, s, n) && d[n] == 0x55);
                CHECK(T->memcmp(d, s, n) == 0);
                if (n) {
                    for (int t = 0; t < 8; t++) {
                        size_t at = t < 2 ? (t ? n - 1 : 0) : rnd() % n;
                        unsigned char old = d[at];
                        d[at] = (unsigned char)(old ^ (1 + rnd() % 255));
                        CHECK(sign(T->memcmp(d, s, n)) == sign(memcmp(d, s, n)));
                        CHECK(sign(T->memcmp(s, d, n)) == sign(memcmp(s, d, n)));
                        d[at] = old;
                    }
                }
                CHECK(T->memset(d, 0xa7, n) == d);
                memset(ref, 0xa7, n);
                CHECK(!memcmp(d, ref, n) && d[n] == 0x55);
                s[n] = 0;
                CHECK(T->strlen((char *)s) == n);
                CHECK(T->strcpy((char *)d, (char *)s) == (char *)d && !memcmp(d, s, n + 1));
                CHECK(T->strcmp((char *)s, (char *)d) == 0);
                if (n) {
                    size_t at = rnd() % n;
                    d[at] ^= 0x40;
                    CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                }
            }

    phase = "guard pages";
    unsigned char *ga = guard_pages(3), *gb = guard_pages(3), *gs = guard_pages(1);
    for (size_t n = 0; n <= 12288; n += (n < 8192 ? 1 : 97)) {
        unsigned char *s = ga + 12288 - n, *d = gb + 12288 - n; /* ranges end at the guard */
        pattern(s, n, (unsigned)n);
        CHECK(T->memcpy(d, s, n) == d && !memcmp(d, s, n));
        CHECK(T->memmove(d, s, n) == d && !memcmp(d, s, n));
        CHECK(!T->memcmp(d, s, n));
        if (n) {
            d[n - 1] ^= 1;
            CHECK(sign(T->memcmp(d, s, n)) == sign(memcmp(d, s, n)));
            CHECK(sign(T->memcmp(s, d, n)) == sign(memcmp(s, d, n)));
            d[0] ^= 1;
            CHECK(sign(T->memcmp(d, s, n)) == sign(memcmp(d, s, n)));
        }
        CHECK(T->memset(d, 0x11, n) == d);
        memset(ref, 0x11, n);
        CHECK(!memcmp(d, ref, n));
        if (n) {
            /* strings ending exactly at the guard: NUL is the last mapped byte */
            s[n - 1] = 0;
            CHECK(T->strlen((char *)s) == n - 1);
            CHECK(T->strcpy((char *)d, (char *)s) == (char *)d && !memcmp(d, s, n));
            CHECK(!T->strcmp((char *)s, (char *)d));
            if (n > 1) {
                d[n - 2] ^= 0x80;
                CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                CHECK(sign(T->strcmp((char *)d, (char *)s)) == sign(strcmp((char *)d, (char *)s)));
            }
            /* strings at the start of the mapping, before the low guard */
            unsigned char *s0 = ga, *d0 = gb;
            if (n - 1 < 12288) {
                pattern(s0, n - 1, 3);
                s0[n - 1] = 0;
                CHECK(T->strlen((char *)s0) == n - 1);
                CHECK(T->strcpy((char *)d0, (char *)s0) == (char *)d0);
                CHECK(!T->strcmp((char *)s0, (char *)d0));
            }
            /* sets ending at the guard */
            const size_t setlens[] = {1, 2, 3, 15, 16, 17, 31, 32, 33, 100};
            for (size_t k = 0; k < sizeof setlens / sizeof *setlens; k++) {
                size_t m = setlens[k];
                unsigned char *set = gs + 4096 - m - 1;
                for (size_t i = 0; i < m; i++)
                    set[i] = (unsigned char)('a' + i % 26);
                set[m] = 0;
                for (size_t i = 0; i + 1 < n; i++)
                    s[i] = (unsigned char)((i % 5) ? 'a' + i % 26 : '0' + i % 10);
                CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
                for (size_t i = 0; i + 1 < n; i++)
                    s[i] = (unsigned char)('a' + i % (m < 26 ? m : 26));
                CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
                if (m == 1 && n > 1) {
                    for (size_t i = 0; i + 1 < n; i++) s[i] = 'a';
                    CHECK(T->strspn((char *)s, (char *)set) == n - 1);
                    CHECK(T->strcspn((char *)s, (char *)set) == 0);
                    s[0] = 'b';
                    CHECK(T->strspn((char *)s, (char *)set) == 0);
                    CHECK(T->strcspn((char *)s, (char *)set) == (n > 2 ? 1 : 1));
                }
            }
            CHECK(T->strspn((char *)s, "") == 0);
            CHECK(T->strcspn((char *)s, "") == strlen((char *)s));
        }
    }

    /* Two strings ending at different distances from their guards: the page
       checks of the two pointers disagree, near the end of one page and far
       from the end of the other. */
    phase = "page offsets";
    const size_t lens[] = {33, 40, 64, 65, 100, 127, 128, 129, 141, 200, 255, 256, 257, 353, 600, 4100};
    for (size_t k = 0; k < sizeof lens / sizeof *lens; k++)
        for (size_t ga_ = 0; ga_ < 320; ga_ += 7)
            for (size_t gb_ = 0; gb_ < 320; gb_ += 11) {
                size_t n = lens[k];
                unsigned char *s = ga + 12288 - ga_ - n - 1, *d = gb + 12288 - gb_ - n - 1;
                pattern(s, n, (unsigned)(n + ga_));
                s[n] = 0;
                memcpy(d, s, n + 1);
                CHECK(!T->strcmp((char *)s, (char *)d));
                size_t at = rnd() % n;
                d[at] = (unsigned char)(d[at] == 0x40 ? 0x41 : 0x40);
                CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                CHECK(sign(T->strcmp((char *)d, (char *)s)) == sign(strcmp((char *)d, (char *)s)));
                memset(d, 0x55, n + 1);
                CHECK(T->strcpy((char *)d, (char *)s) == (char *)d && !memcmp(d, s, n + 1));
            }

    phase = "random";
    unsigned char set[520];
    for (int t = 0; t < 300000; t++) {
        size_t n = rnd() % (t % 50 == 0 ? 70000 : 400), ao = rnd() % 64, bo = rnd() % 64;
        unsigned char *s = a + 4096 + ao, *d = b + 4096 + bo;
        for (size_t i = 0; i < n; i++) s[i] = (unsigned char)(1 + rnd() % 255);
        s[n] = 0;
        CHECK(T->strlen((char *)s) == n);
        memcpy(ref, s, n + 1);
        CHECK(T->memcpy(d, s, n) == d && !memcmp(d, ref, n));
        CHECK(!T->memcmp(d, s, n));
        if (n) {
            size_t at = rnd() % n;
            d[at] = (unsigned char)(rnd() % 256);
            CHECK(sign(T->memcmp(d, s, n)) == sign(memcmp(d, s, n)));
            d[at] = (unsigned char)(1 + rnd() % 255);
            CHECK(sign(T->strcmp((char *)d, (char *)s)) == sign(strcmp((char *)d, (char *)s)));
        }
        int v = (int)(rnd() % 512) - 256;
        CHECK(T->memset(d, v, n) == d);
        memset(ref, v, n);
        CHECK(!memcmp(d, ref, n));
        /* overlap within a */
        long delta = (long)(rnd() % 2001) - 1000;
        unsigned char *od = s + delta;
        size_t span = n + (size_t)(delta < 0 ? -delta : delta);
        pattern(a, span + 8192, (unsigned)t);
        check_copy(od, s, n, ref, span);
        /* sets */
        size_t m = rnd() % 40;
        if (t % 7 == 0) m = rnd() % 300;
        for (size_t i = 0; i < m; i++) set[i] = (unsigned char)(1 + rnd() % (t % 3 ? 60 : 255));
        set[m] = 0;
        for (size_t i = 0; i < n; i++) s[i] = (unsigned char)(1 + rnd() % (t % 2 ? 60 : 255));
        CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
        CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
        if (m) {
            size_t at = rnd() % (n + 1);
            if (at < n) {
                s[at] = set[rnd() % m];
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
            }
        }
    }
    printf("PASS %zu checks: small sizes x alignments, overlap directions, large sizes, guard pages, random\n", checks);
    free(a); free(b); free(ref); free(ref2);
}

int main(void)
{
    /* every configuration the host runs, then the host's own choice */
    unsigned host = str_host();
    for (unsigned i = 0; i < str_nconfigs; i++) {
        if (str_configs[i].needs & ~host)
            continue;
        str_table = str_configs[i].table;
        Table = &str_table;
        printf("%s: ", str_configs[i].name);
        checks = 0;
        suite();
    }
    str_dispatch();
    Table = &str_table;
    printf("dispatched (%s): ", str_choose()->name);
    checks = 0;
    suite();
    return 0;
}
