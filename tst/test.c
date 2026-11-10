/* Correctness test against the host C library. Covers exhaustive small sizes
   and alignments, selected large sizes, every overlap direction, guard pages
   on both ends, every pair of page offsets of two pointers (the page checks
   of the two disagree in ways a sweep of one misses), every page offset of
   one, random fuzzing; the whole suite runs for each configuration the host
   can execute and for the dispatched table, then a quick suite with every
   slot replaced by every other implementation in turn (the contracts between
   first blocks and kernels hold for any mix), and the host decisions are made
   again from synthetic cpuid snapshots of other machines. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include "common.h"

/* Every call goes through junk(): the unused argument registers hold garbage, so
   a function that reads an argument it was not given (a tail call's second
   argument prepared on the wrong path, say) cannot pass by luck. */
static inline int junk(void) {
    __asm__ volatile("movq $0x5a5a5a5a5a5a5a5a, %%rsi\n\tmovq $0x5a5a5a5a5a5a5a5a, %%rdx\n\tmovq $0x5a5a5a5a5a5a5a5a, %%rcx" ::: "rsi", "rdx", "rcx");
    return 0;
}
#define T (junk(), Table)

/* Every check goes through T; main runs the whole suite once per table. */
static const struct str_table *Table;

static size_t checks;
static const char *phase = "start";
static char mix[160] = "";   /* which one-slot mix runs, for the message */
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL %s%s line %d check %zu: %s\n", phase, mix, __LINE__, checks, #x); exit(2); } } while (0)
static int sign(int v) {
    return (v > 0) - (v < 0);
}

static uint32_t state = 0x9e3779b9;
static uint32_t rnd(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void pattern(unsigned char *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char)(1 + (i * 37 + seed * 11 + (i >> 7)) % 255);
    }
}

static unsigned char *guard_pages(size_t pages) {
    /* PROT_NONE page, then `pages` writable pages, then PROT_NONE page. */
    unsigned char *m = mmap(0, (pages + 2) * 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(m != MAP_FAILED);
    CHECK(!mprotect(m + 4096, pages * 4096, PROT_READ | PROT_WRITE));
    return m + 4096;
}

/* The functions of derived.c are not slots: they call the entry points, so
   through str_table, which main points at each table in turn. */
#define D(f) (junk(), f)

/* The first occurrence of n[0, nl) in h[0, hl), the reference for memmem and strstr. */
static const unsigned char *naive_find(const unsigned char *h, size_t hl, const unsigned char *n, size_t nl) {
    for (size_t i = 0; i + nl <= hl; i++) {
        if (!memcmp(h + i, n, nl)) {
            return h + i;
        }
    }

    return 0;
}

/* derived.c on this table: each function at the sizes and offsets where the
   primitives it calls change paths, quick enough for every mix. */
static void derived_quick(unsigned char *a, unsigned char *b, unsigned char *ref) {
    phase = "derived functions";

    for (size_t n = 0; n <= 80; n++) {
        for (size_t off = 0; off < 40; off += 13) {
            unsigned char *s = a + 4096 + off, *d = b + 4096 + off * 3 % 40;
            pattern(s, n, (unsigned)n);
            s[n] = 0;

            for (size_t lim = n > 2 ? n - 2 : 0; lim <= n + 2; lim++) {
                CHECK(D(str_strnlen)((char *)s, lim) == (lim < n ? lim : n));
                memset(d, 0x55, n + 8);
                CHECK(D(str_strlcpy)((char *)d, (char *)s, lim) == n);
                CHECK(lim == 0 ? d[0] == 0x55 : !memcmp(d, s, lim <= n ? lim - 1 : n) && d[lim <= n ? lim - 1 : n] == 0 && d[lim <= n ? lim : n + 1] == 0x55);
                /* strlcat and strncat onto "xy" */
                memset(d, 0x55, n + 8);
                d[0] = 'x'; d[1] = 'y'; d[2] = 0;
                size_t room = lim + 2, want = room <= 2 ? room + n : 2 + n;
                CHECK(D(str_strlcat)((char *)d, (char *)s, room) == want);
                size_t kept = room <= 2 ? 0 : (n < room - 3 ? n : room - 3);
                CHECK(room <= 2 ? d[2] == 0 && d[3] == 0x55 : !memcmp(d + 2, s, kept) && d[2 + kept] == 0 && d[3 + kept] == 0x55);
                memset(d, 0x55, n + 8);
                d[0] = 'x'; d[1] = 0;
                CHECK(D(str_strncat)((char *)d, (char *)s, lim) == (char *)d);
                size_t took = lim < n ? lim : n;
                CHECK(d[0] == 'x' && !memcmp(d + 1, s, took) && d[1 + took] == 0 && d[2 + took] == 0x55);
            }

            CHECK(D(str_mempcpy)(d, s, n) == d + n && !memcmp(d, s, n));
            CHECK(!D(str_bcmp)(d, s, n));

            if (n) {
                d[n - 1] ^= 1;
                CHECK(D(str_bcmp)(d, s, n) != 0);
            }

            memset(d, 0x55, n + 2);
            D(str_bzero)(d, n);
            CHECK((n == 0 || (d[0] == 0 && d[n - 1] == 0)) && d[n] == 0x55);
            memset(d, 0x55, n + 2);
            D(str_explicit_bzero)(d, n);
            CHECK((n == 0 || (d[0] == 0 && d[n - 1] == 0)) && d[n] == 0x55);
            D(str_bcopy)(s, d, n + 1);
            CHECK(!memcmp(d, s, n + 1));
            D(str_bcopy)(d, d + 1, n);   /* overlapping, forward */
            CHECK(!memcmp(d + 1, s, n));
            /* strpbrk, strsep and strtok_r on words of s split at the bytes of a set */
            const char *set = n % 3 ? ",; " : "\x7f\x01";

            for (size_t i = 0; i < n; i++) {
                s[i] = (unsigned char)(i % 7 == 3 ? set[i % 2] : 'a' + i % 26);
            }

            CHECK(D(str_strpbrk)((char *)s, set) == strpbrk((char *)s, set));
            memcpy(d, s, n + 1);
            memcpy(ref, s, n + 1);
            char *rest = (char *)d, *ref_rest = (char *)ref;

            for (;;) {
                char *t1 = D(str_strsep)(&rest, set), *t2 = strsep(&ref_rest, set);
                CHECK((t1 == 0) == (t2 == 0));

                if (!t1) {
                    break;
                }

                CHECK(t1 - (char *)d == t2 - (char *)ref && !strcmp(t1, t2));
            }

            CHECK(!memcmp(d, ref, n + 1));
            memcpy(d, s, n + 1);
            memcpy(ref, s, n + 1);
            char *save = 0, *ref_save = 0;
            char *t1 = D(str_strtok_r)((char *)d, set, &save), *t2 = strtok_r((char *)ref, set, &ref_save);

            for (;;) {
                CHECK((t1 == 0) == (t2 == 0));

                if (!t1) {
                    break;
                }

                CHECK(t1 - (char *)d == t2 - (char *)ref && !strcmp(t1, t2));
                t1 = D(str_strtok_r)(0, set, &save);
                t2 = strtok_r(0, set, &ref_save);
            }

            CHECK(D(str_strtok_r)(0, set, &save) == 0);   /* and once more past the end */
            CHECK(!memcmp(d, ref, n + 1));
            /* memmem and strstr: the needle a piece of s, a byte changed or not */
            pattern(s, n, (unsigned)n);
            s[n] = 0;

            for (size_t nl = 0; nl <= n && nl <= 9; nl++) {
                size_t at = n - nl;
                memcpy(d, s + at, nl);
                d[nl] = 0;
                CHECK(D(str_memmem)(s, n, d, nl) == naive_find(s, n, d, nl));
                CHECK(D(str_strstr)((char *)s, (char *)d) == (char *)naive_find(s, n, d, nl));

                if (nl) {
                    d[nl - 1] ^= 0x20;
                    CHECK(D(str_memmem)(s, n, d, nl) == naive_find(s, n, d, nl));
                    CHECK(D(str_strstr)((char *)s, (char *)d) == (char *)naive_find(s, n, d, nl));
                }
            }
        }
    }
}

static void check_copy(unsigned char *d, const unsigned char *s, size_t n, unsigned char *ref, size_t span) {
    /* d and s may overlap; ref holds a copy of the whole span containing both. */
    memcpy(ref, d < s ? d : s, span);
    size_t doff = (size_t)(d - (d < s ? d : s)), soff = (size_t)(s - (d < s ? d : s));
    memmove(ref + doff, ref + soff, n);
    CHECK(T->memmove(d, s, n) == d);
    CHECK(!memcmp(d < s ? d : s, ref, span));
}

static void suite(int full) {
    CHECK(sysconf(_SC_PAGESIZE) == 4096);
    const size_t cap = 1u << 21;
    unsigned char *a = malloc(cap + 512), *b = malloc(cap + 512), *ref = malloc(cap + 512), *ref2 = malloc(cap + 512);
    CHECK(a && b && ref && ref2);

    phase = "small sizes, all alignments";

    for (size_t n = 0; n <= 300; n++) {
        for (size_t ao = 0; ao < 48; ao += (ao < 20 ? 1 : 7)) {
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

                /* memchr: a byte that is there, with the int's high bits set as the
                   standard converts it, the first of several, and one that is not */
                CHECK(T->memchr(s, 0x100 | s[n / 2], n) == memchr(s, s[n / 2], n));
                CHECK(T->memchr(s, 0x9a, n) == memchr(s, 0x9a, n));
                CHECK(T->memchr(s, s[n + 1], 0) == 0);

                if (n) {
                    size_t at = rnd() % n;
                    unsigned char old = s[at];
                    s[at] = 0x9a;
                    CHECK(T->memchr(s, 0x9a, n) == memchr(s, 0x9a, n)); /* at `at` unless the pattern has one before */
                    CHECK(T->memchr(s, 0x9a, at) == memchr(s, 0x9a, at));
                    CHECK(T->memchr(s, 0x9a, SIZE_MAX) == memchr(s, 0x9a, n)); /* C11: the object may end at the match; s + n wraps */
                    s[at] = old;
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
        }
    }

    phase = "long strcat and set scans";

    for (size_t n = 0; n <= 600; n += (n < 70 ? 1 : 37)) {
        for (size_t m = 0; m <= 600; m += (m < 70 ? 1 : 41)) {
            unsigned char *s = a + 4096 + (n * 5 % 48), *d = b + 4096 + (m * 7 % 48);
            pattern(s, n, (unsigned)n);
            s[n] = 0;
            pattern(d, m, (unsigned)m + 3);
            d[m] = 0;
            d[m + n + 1] = 0x55;
            memcpy(ref, d, m);
            memcpy(ref + m, s, n + 1);
            CHECK(T->strcat((char *)d, (char *)s) == (char *)d);
            CHECK(!memcmp(d, ref, m + n + 1) && d[m + n + 1] == 0x55);
        }
    }

    {
        const size_t setlens[] = {1, 2, 3, 15, 16, 17, 31, 32, 33, 64, 200};

        for (size_t k = 0; k < sizeof setlens / sizeof *setlens; k++) {
            size_t m = setlens[k];
            unsigned char *set = b + 8192 + k * 7;

            for (size_t i = 0; i < m; i++) {
                set[i] = (unsigned char)(1 + (i * 97 + 13) % 255);
            }

            set[m] = 0;

            /* the runs reach past pcmpistri's 2 KiB (Zen: 1 KiB) into the shuffle scan */
            for (size_t run = 0; run <= 4300; run += (run < 70 ? 1 : run < 600 ? 29 : 900)) {
                for (size_t off = 0; off < 48; off += 5) {
                    unsigned char *s = a + 4096 + off;

                    for (size_t i = 0; i < run; i++)   /* members, then a byte outside the set */
                        s[i] = set[i % m];
                    s[run] = (unsigned char)(m == 200 ? 200 : 254);   /* 254 is in no set here, 200 not in the long one */
                    s[run + 1] = set[0];
                    s[run + 2] = 0;
                    CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));

                    for (size_t i = 0; i < run; i++)   /* non-members, then a member */
                        s[i] = (unsigned char)(m == 200 ? 200 + i % 50 : 254 - i % 2);
                    s[run] = set[m / 2];
                    s[run + 1] = 0;
                    CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
                    s[run] = 0;   /* and the NUL instead */
                    CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                    CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
                }
            }
        }
    }

    derived_quick(a, b, ref);

    if (!full) {
        goto guards;
    }

    phase = "overlap";
    const size_t sizes[] = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 200,
                            255, 256, 257, 300, 511, 512, 513, 1000, 2047, 2048, 2049, 4095, 4096, 4097, 8192,
                            65535, 65536, 65537, 1000003, 1048576, 1048577};

    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        size_t n = sizes[k];
        const long deltas[] = {-4097, -4096, -300, -257, -256, -129, -128, -65, -64, -33, -32, -17, -16, -15, -8, -3, -1, 0,
                               1, 3, 8, 15, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 300, 4096, 4097};

        for (size_t j = 0; j < sizeof deltas / sizeof *deltas; j++) {
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
    }

    phase = "large sizes";

    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        for (size_t ao = 0; ao < 64; ao += 13) {
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
        }
    }

guards:;
    phase = "guard pages";
    unsigned char *ga = guard_pages(3), *gb = guard_pages(3), *gs = guard_pages(1);

    /* objects across a page boundary between two mapped pages: an entry's
       page-end path checks the first page's part and hands over the rest
       (memchr_sse2 once left 13..32 bytes of the second page unchecked) */

    for (size_t k = 1; k <= 64; k++) {
        unsigned char *s = ga + 4096 - k, *d = gb + 4096 - k;

        for (size_t n = 1; n <= 200; n++) {
            pattern(s, n + 1, (unsigned)(n + k));
            memcpy(d, s, n + 1);
            CHECK(T->memchr(s, 0x9a, n) == memchr(s, 0x9a, n));
            CHECK(!T->memcmp(s, d, n));
            const size_t at[] = {k - 1, k, k + 1, k + 15, k + 16, k + 31, k + 32, k + 63, n - 1};

            for (size_t i = 0; i < sizeof at / sizeof *at; i++) {
                if (at[i] >= n) {
                    continue;
                }

                unsigned char keep = s[at[i]];
                s[at[i]] = 0x9a;
                CHECK(T->memchr(s, 0x9a, n) == memchr(s, 0x9a, n));
                CHECK(T->memchr(s, 0x9a, at[i]) == memchr(s, 0x9a, at[i])); /* the match right past n, maybe in the head block */
                CHECK(sign(T->memcmp(s, d, n)) == sign(memcmp(s, d, n)));
                s[at[i]] = 0; /* a string ending there */
                CHECK(T->strlen((char *)s) == strlen((char *)s));
                CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                /* copied and appended: the first block cut short by the page end, the NUL past it */
                unsigned char *t = ref2 + 64, *u = ref2 + 4096;
                CHECK(T->strcpy((char *)t, (char *)s) == (char *)t && !memcmp(t, s, strlen((char *)s) + 1));
                u[0] = 'x';
                u[1] = 0;
                CHECK(T->strcat((char *)u, (char *)s) == (char *)u && u[0] == 'x' && !memcmp(u + 1, s, strlen((char *)s) + 1));
                s[at[i]] = keep;
            }
        }
    }

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
            CHECK(T->memchr(s, 0x9a, n) == memchr(s, 0x9a, n));     /* scans to the guard */
            CHECK(T->memchr(s, 0, n) == memchr(s, 0, n));           /* the last byte before it at the latest */
            /* C11: the object may end at the match, n past it and past the page:
               no load may touch the guard (the unaligned 4-block groups of the
               long kernels did, dev/proof found it) */
            CHECK(T->memchr(s, 0, n + 1000) == memchr(s, 0, n));
            CHECK(T->memchr(s, 0, SIZE_MAX) == memchr(s, 0, n));
            CHECK(T->memchr(s, s[0], n) == s);
            CHECK(T->memchr(s, 0x9a, n - 1) == memchr(s, 0x9a, n - 1));
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

                for (size_t i = 0; i < m; i++) {
                    set[i] = (unsigned char)('a' + i % 26);
                }

                set[m] = 0;

                for (size_t i = 0; i + 1 < n; i++) {
                    s[i] = (unsigned char)((i % 5) ? 'a' + i % 26 : '0' + i % 10);
                }

                CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));

                for (size_t i = 0; i + 1 < n; i++) {
                    s[i] = (unsigned char)('a' + i % (m < 26 ? m : 26));
                }

                CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));

                if (m == 1 && n > 1) {
                    for (size_t i = 0; i + 1 < n; i++) {
                        s[i] = 'a';
                    }

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

    if (!full) {
        munmap(ga - 4096, 5 * 4096);
        munmap(gb - 4096, 5 * 4096);
        munmap(gs - 4096, 3 * 4096);
        free(a); free(b); free(ref); free(ref2);
        return;
    }

    /* Two strings ending at different distances from their guards: the page
       checks of the two pointers disagree, near the end of one page and far
       from the end of the other. */
    phase = "page offsets";
    const size_t lens[] = {33, 40, 64, 65, 100, 127, 128, 129, 141, 200, 255, 256, 257, 353, 600, 4100};

    for (size_t k = 0; k < sizeof lens / sizeof *lens; k++) {
        for (size_t ga_ = 0; ga_ < 320; ga_ += 7) {
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
                /* strcat: a destination of its own length m, then s appended up to the guard */
                size_t m = (n + gb_) % 40;
                unsigned char *dd = gb + 12288 - gb_ - m - n - 2;
                pattern(dd, m, 5);
                dd[m] = 0;
                CHECK(T->strcat((char *)dd, (char *)s) == (char *)dd && dd[m + n] == 0 && !memcmp(dd + m, s, n + 1));
            }
        }
    }

    /* Every pair of page offsets of two pointers. The decisions of a two-pointer
       function depend on both offsets at once (and on their bits: both_in_page
       ORs them), so a sweep of one pointer with the other fixed misses cases. The
       strings start in the middle page of a three-page mapping and may run into
       the next one; the lengths pin the first-block logic, with every offset pair

       for a few of them and every offset of the first pointer against every
       thirteenth of the second for the rest. */
    phase = "offset pairs";
    {
        const size_t full_lens[] = {1, 33, 65};
        const size_t grid_lens[] = {2, 5, 15, 16, 17, 31, 32, 47, 48, 63, 64, 100, 127, 128, 129, 255, 256, 257, 300};

        for (int pass = 0; pass < 2; pass++) {
            const size_t *lens = pass ? grid_lens : full_lens;
            size_t nlens = pass ? sizeof grid_lens / sizeof *grid_lens : sizeof full_lens / sizeof *full_lens;
            size_t step = pass ? 13 : 1;

            for (size_t oa = 0; oa < 4096; oa++) {
                for (size_t ob = 0; ob < 4096; ob += step) {
                    for (size_t k = 0; k < nlens; k++) {
                        size_t n = lens[k];
                        unsigned char *s = ga + 4096 + oa, *d = gb + 4096 + ob;
                        pattern(s, n, (unsigned)(oa * 7 + ob));
                        s[n] = 0;
                        memcpy(d, s, n + 1);
                        CHECK(!T->strcmp((char *)s, (char *)d));
                        CHECK(!T->memcmp(s, d, n));
                        d[n - 1] ^= 0x40; /* the last byte differs */
                        CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                        CHECK(sign(T->strcmp((char *)d, (char *)s)) == sign(strcmp((char *)d, (char *)s)));
                        CHECK(sign(T->memcmp(s, d, n)) == sign(memcmp(s, d, n)));
                        d[n - 1] ^= 0x40;
                        d[0] ^= 0x40; /* the first byte differs */
                        CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                        d[0] ^= 0x40;
                        d[n] = 'x'; /* d longer than s */
                        d[n + 1] = 0;
                        CHECK(sign(T->strcmp((char *)s, (char *)d)) == sign(strcmp((char *)s, (char *)d)));
                        memset(d, 0x55, n + 2);
                        CHECK(T->strcpy((char *)d, (char *)s) == (char *)d && !memcmp(d, s, n + 1) && d[n + 1] == 0x55);
                        CHECK(T->memcpy(d, s, n) == d && !memcmp(d, s, n));
                    }
                }
            }
        }
    }

    /* Every page offset of one pointer: the string ends before the guard when it
       fits, else runs into the next page. */
    phase = "single offsets";
    {
        const size_t one_lens[] = {0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 127, 128, 129, 191, 192, 255, 256, 257, 383, 384, 511, 512, 513, 1000, 4095, 4096, 4097};

        for (size_t oa = 0; oa < 4096; oa++) {
            for (size_t k = 0; k < sizeof one_lens / sizeof *one_lens; k++) {
                size_t n = one_lens[k];

                if (oa + n + 1 > 8192) {
                    continue;
                }

                unsigned char *s = ga + (oa + n + 1 <= 4096 ? 8192 : 4096) + oa, *d = gb + 4096 + (n > 300 ? 0 : oa * 3 % 4096);
                pattern(s, n, (unsigned)(n + oa));
                s[n] = 0;
                CHECK(T->strlen((char *)s) == n);
                CHECK(T->strcpy((char *)d, (char *)s) == (char *)d && !memcmp(d, s, n + 1));
                CHECK(T->memset(d, (int)oa, n) == d && (n == 0 || (d[0] == (unsigned char)oa && d[n - 1] == (unsigned char)oa)));

                if (n && k % 3 == 0) {
                    for (size_t i = 0; i < n; i++) {
                        s[i] = (unsigned char)('a' + (i * 7 + oa) % 26);
                    }

                    CHECK(T->strspn((char *)s, "abcdefghijklm") == strspn((char *)s, "abcdefghijklm"));
                    CHECK(T->strcspn((char *)s, "xyz") == strcspn((char *)s, "xyz"));
                    CHECK(T->strcspn((char *)s, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") == strcspn((char *)s, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
                    CHECK(T->strspn((char *)s, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ") == strspn((char *)s, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"));
                }
            }
        }

        /* and of the set: its load has page logic of its own; the set ends before
           the guard when it fits the page, else runs into the next (mapped) one */
        unsigned char *gs2 = guard_pages(2);

        for (size_t os = 0; os < 4096; os++) {
            const size_t setlens[] = {1, 2, 15, 16, 17, 31, 32, 33, 40};

            for (size_t k = 0; k < sizeof setlens / sizeof *setlens; k++) {
                size_t m = setlens[k];
                unsigned char *set = (os + m + 1 <= 4096 ? gs + os : gs2 + os);

                for (size_t i = 0; i < m; i++) {
                    set[i] = (unsigned char)('A' + i % 26);
                }

                set[m] = 0;
                unsigned char *s = ga + 4096 + (os * 5 % 4096);

                for (size_t i = 0; i < 70; i++) {
                    s[i] = (unsigned char)(i < 20 + os % 50 ? 'A' + i % (m < 26 ? m : 26) : 'a' + i % 26);
                }

                s[70] = 0;
                CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
                s[0] = 'z';
                CHECK(T->strspn((char *)s, (char *)set) == strspn((char *)s, (char *)set));
                CHECK(T->strcspn((char *)s, (char *)set) == strcspn((char *)s, (char *)set));
            }
        }

        munmap(gs2 - 4096, 4 * 4096);
    }

    phase = "random";
    unsigned char set[520];

    for (int t = 0; t < 300000; t++) {
        size_t n = rnd() % (t % 50 == 0 ? 70000 : 400), ao = rnd() % 4096, bo = rnd() % 4096;
        unsigned char *s = a + 4096 + ao, *d = b + 4096 + bo;

        for (size_t i = 0; i < n; i++) {
            s[i] = (unsigned char)(1 + rnd() % 255);
        }

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

        if (t % 7 == 0) {
            m = rnd() % 300;
        }

        for (size_t i = 0; i < m; i++) {
            set[i] = (unsigned char)(1 + rnd() % (t % 3 ? 60 : 255));
        }

        set[m] = 0;

        for (size_t i = 0; i < n; i++) {
            s[i] = (unsigned char)(1 + rnd() % (t % 2 ? 60 : 255));
        }

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

    printf("PASS %zu checks: small sizes x alignments, long strcat and set scans, overlap directions, large sizes, guard pages, offset pairs, single offsets, random\n", checks);
    munmap(ga - 4096, 5 * 4096);
    munmap(gb - 4096, 5 * 4096);
    munmap(gs - 4096, 3 * 4096);
    free(a); free(b); free(ref); free(ref2);
}

/* The slots of a table as an array, for the mixes. */
#define NSLOTS (sizeof(struct str_table) / sizeof(void *))
static void slots_of(const struct str_table *t, void **slots) {
    memcpy(slots, t, sizeof *t);
}

static void table_of(struct str_table *t, void *const *slots) {
    memcpy(t, slots, sizeof *t);
}

/* Other hosts, as cpuid snapshots: the decisions are pure, so they can be made
   here for machines this one is not. */
#define B(i) (1u << (i))
#define V2_ECX1 (B(0) | B(9) | B(13) | B(19) | B(20) | B(23))                     /* SSE3 SSSE3 CX16 SSE4.1 SSE4.2 POPCNT */
#define V3_ECX1 (V2_ECX1 | B(12) | B(22) | B(27) | B(28) | B(29))                /* FMA MOVBE OSXSAVE AVX F16C */
#define V3_EBX7 (B(3) | B(5) | B(8))                                             /* BMI1 AVX2 BMI2 */
#define ERMS_EBX7 B(9)
#define V4_EBX7 (V3_EBX7 | B(16) | B(17) | B(28) | B(30) | B(31))                /* AVX512 F DQ CD BW VL */
#define V2_ECX81 B(0)                                                            /* LAHF/SAHF */
#define V3_ECX81 (V2_ECX81 | B(5))                                               /* LZCNT */
#define INTEL 0x756e6547, 0x49656e69, 0x6c65746e
#define AMD 0x68747541, 0x69746e65, 0x444d4163

/* The public entry points, through the dispatched table. */
static void entries(void) {
    phase = "entry points";
    char s[64] = "hello, world", d[64];
    CHECK(str_strlen(s) == 12);
    CHECK(str_strcpy(d, s) == d && !strcmp(d, s));
    CHECK(str_strcat(d, s) == d && str_strlen(d) == 24);
    CHECK(str_strcmp(s, d) < 0 && str_strcmp(d, s) > 0 && str_strcmp(s, s) == 0);
    CHECK(str_memcpy(d, s, 13) == d && str_memcmp(d, s, 13) == 0);
    CHECK(str_memmove(d + 1, d, 13) == d + 1 && !strcmp(d + 1, s));
    CHECK(str_memset(d, 'x', 5) == d && d[0] == 'x' && d[4] == 'x' && d[5] == 'o');
    CHECK(str_memcmp(s, d, 13) < 0);
    CHECK(str_strspn(s, "ehl") == 4 && str_strcspn(s, ",") == 5);
    CHECK(str_memchr(s, 'w', 13) == s + 7 && str_memchr(s, 'z', 13) == 0);
    printf("entry points: PASS %zu checks\n", checks);
}

/* derived.c once more, on the dispatched table: memmem and strstr on every
   needle and haystack of small alphabets, plainly and behind a prefix that
   spends the scan's budget (the rest found by two-way), on long and
   periodic needles; every function on objects that end at a guard page;
   the edges of each. */
static void derived(void) {
    phase = "memmem and strstr, small alphabets";
    unsigned char *h = malloc(1 << 17), *nd = malloc(4096);
    CHECK(h && nd);
    const size_t prefix = 200;   /* the needle's first byte, then one not in it: a place every other byte, none a match */

    for (int alpha = 0; alpha < 3; alpha++) {
        /* {a, b}: needles up to 6 bytes, haystacks up to 10; {a, b, c}: 4 and 6; {0, 1}: NULs inside, memmem only */
        const unsigned char base = alpha == 2 ? 0 : 'a';
        const size_t k = alpha == 1 ? 3 : 2, maxn = alpha == 1 ? 4 : 6, maxh = alpha == 1 ? 6 : 10;

        for (size_t nl = 1; nl <= maxn; nl++) {
            size_t ncount = 1;

            for (size_t i = 0; i < nl; i++) {
                ncount *= k;
            }

            for (size_t ni = 0; ni < ncount; ni++) {
                for (size_t i = 0, v = ni; i < nl; i++, v /= k) {
                    nd[i] = (unsigned char)(base + v % k);
                }

                nd[nl] = 0;

                for (size_t hl = 0; hl <= maxh; hl++) {
                    size_t hcount = 1;

                    for (size_t i = 0; i < hl; i++) {
                        hcount *= k;
                    }

                    for (size_t hi = 0; hi < hcount; hi++) {
                        for (size_t f = 0; f <= (nl > 1 ? prefix : 0); f += prefix) {
                            for (size_t i = 0; i < f; i++) {
                                h[i] = i % 2 ? 'x' : nd[0];
                            }

                            for (size_t i = 0, v = hi; i < hl; i++, v /= k) {
                                h[f + i] = (unsigned char)(base + v % k);
                            }

                            h[f + hl] = 0;
                            const unsigned char *want = naive_find(h, f + hl, nd, nl);
                            CHECK(D(str_memmem)(h, f + hl, nd, nl) == want);

                            if (alpha != 2) {
                                CHECK(D(str_strstr)((char *)h, (char *)nd) == (char *)want);
                            }
                        }
                    }
                }
            }
        }
    }

    phase = "memmem and strstr, long and periodic";

    for (int t = 0; t < 20000; t++) {
        size_t q = 1 + rnd() % 8, nl = 1 + rnd() % (t % 10 == 0 ? 700 : 40);
        size_t hl = t % 50 == 0 ? rnd() % 100000 : rnd() % (nl > 40 ? 5000 : 3000);
        unsigned char seed[8];

        for (size_t i = 0; i < q; i++) {
            seed[i] = (unsigned char)('a' + rnd() % 3);
        }

        for (size_t i = 0; i < nl; i++) {
            nd[i] = seed[i % q];
        }

        if (rnd() % 2) {
            nd[rnd() % nl] = (unsigned char)('a' + rnd() % 4);
        }

        nd[nl] = 0;

        for (size_t i = 0; i < hl; i++) {
            h[i] = rnd() % 16 ? seed[i % q] : (unsigned char)('a' + rnd() % 4);
        }

        if (rnd() % 2 && hl >= nl) {
            memcpy(h + rnd() % (hl - nl + 1), nd, nl);
        }

        h[hl] = 0;
        const unsigned char *want = naive_find(h, hl, nd, nl);
        CHECK(D(str_memmem)(h, hl, nd, nl) == want);
        CHECK(D(str_strstr)((char *)h, (char *)nd) == (char *)want);
    }

    phase = "derived functions at a guard page";
    unsigned char *g = guard_pages(2), *gd = guard_pages(2);

    for (size_t n = 0; n <= 300; n++) {
        unsigned char *s = g + 8192 - n - 1, *d = gd + 8192 - n - 1;   /* the NULs the last mapped bytes */
        pattern(s, n, (unsigned)n);
        s[n] = 0;
        CHECK(D(str_strnlen)((char *)s, n + 1000) == n);
        CHECK(D(str_strnlen)((char *)s, SIZE_MAX) == n);
        CHECK(D(str_strlcpy)((char *)d, (char *)s, n + 1) == n && !memcmp(d, s, n + 1));
        CHECK(D(str_strlcpy)((char *)d, (char *)s, SIZE_MAX) == n && !memcmp(d, s, n + 1));
        d[0] = 0;
        CHECK(D(str_strlcat)((char *)d, (char *)s, n + 1) == n && !memcmp(d, s, n + 1));
        d[0] = 0;
        CHECK(D(str_strncat)((char *)d, (char *)s, SIZE_MAX) == (char *)d && !memcmp(d, s, n + 1));
        char *dup = D(str_strdup)((char *)s);
        CHECK(dup && !strcmp(dup, (char *)s));
        free(dup);
        dup = D(str_strndup)((char *)s, n + 50);
        CHECK(dup && strlen(dup) == n && !memcmp(dup, s, n));
        free(dup);
        dup = D(str_strndup)((char *)s, n / 2);
        CHECK(dup && strlen(dup) == n / 2 && !memcmp(dup, s, n / 2));
        free(dup);
        CHECK(D(str_strpbrk)((char *)s, "\x01\x02") == strpbrk((char *)s, "\x01\x02"));

        for (size_t nl = 1; nl <= n && nl <= 40; nl += 1 + nl / 4) {
            CHECK(D(str_memmem)(s, n, s + n - nl, nl) == naive_find(s, n, s + n - nl, nl));
            CHECK(D(str_strstr)((char *)s, (char *)s + n - nl) == (char *)naive_find(s, n, s + n - nl, nl));
        }

        /* a place every other byte up to the guard: two-way finds the haystack's NUL by memchr as it goes */
        for (size_t i = 0; i < n; i++) {
            s[i] = i % 2 ? 'x' : 'a';
        }

        CHECK(D(str_strstr)((char *)s, "ab") == 0 && D(str_strstr)((char *)s, "axaxay") == 0);
        CHECK(D(str_memmem)(s, n, "ab", 2) == 0);
    }

    phase = "derived functions, edges";
    char text[64] = "  a,b ,, c ", *p = 0, *save = 0;
    CHECK(D(str_memmem)(h, 0, nd, 0) == h && D(str_memmem)(h, 3, nd, 4) == 0);
    CHECK(D(str_strstr)(text, "") == text && D(str_strstr)(text + 11, "") == text + 11 && D(str_strstr)(text + 11, "a") == 0);
    CHECK(D(str_strsep)(&p, ",") == 0 && p == 0);
    CHECK(D(str_strtok_r)(0, ",", &save) == 0);
    CHECK(!strcmp(D(str_strtok)(text, " ,"), "a") && !strcmp(D(str_strtok)(0, " ,"), "b") && !strcmp(D(str_strtok)(0, " ,"), "c"));
    CHECK(D(str_strtok)(0, " ,") == 0 && D(str_strtok)(0, " ,") == 0);
    strcpy(text, "ab");
    CHECK(D(str_strlcpy)(text, "xyz", 0) == 3 && !strcmp(text, "ab"));
    CHECK(D(str_strlcat)(text, "xyz", 1) == 4 && !strcmp(text, "ab"));
    CHECK(D(str_strncat)(text, "xyz", 0) == text && !strcmp(text, "ab"));
    CHECK(D(str_bcmp)(text, "zz", 0) == 0);
    char *dup = D(str_strndup)("abc", 0);
    CHECK(dup && dup[0] == 0);
    free(dup);
    /* malloc failing: the address space limited to what the process has plus
       16 MiB, a copy of a 64 MiB string asked for */
    {
        struct rlimit old, lim;
        size_t big = (size_t)64 << 20;
        char *s = malloc(big + 1);
        CHECK(s && !getrlimit(RLIMIT_AS, &old));
        memset(s, 'a', big);
        s[big] = 0;
        long pages = 0;
        FILE *f = fopen("/proc/self/statm", "r");
        CHECK(f && fscanf(f, "%ld", &pages) == 1);
        fclose(f);
        lim.rlim_cur = (rlim_t)pages * 4096 + ((rlim_t)16 << 20);
        lim.rlim_max = old.rlim_max;
        CHECK(!setrlimit(RLIMIT_AS, &lim));
        char *d1 = D(str_strdup)(s), *d2 = D(str_strndup)(s, big);
        CHECK(!setrlimit(RLIMIT_AS, &old));
        CHECK(!d1 && !d2);
        free(s);
    }

    munmap(g - 4096, 4 * 4096);
    munmap(gd - 4096, 4 * 4096);
    free(h);
    free(nd);
    printf("derived: PASS %zu checks: memmem and strstr on small alphabets and periodic needles, guard pages, edges\n", checks);
}

static void hosts(void) {
    const struct {
        const char *name;
        struct str_cpuid c;
        unsigned bits;
        const char *config;
    } machines[] = {
        {"no leaf 7", {6, INTEL, 0, 0, 0, 0, 0}, 0, "sse2"},
        {"core 2", {7, INTEL, 0x6f6, B(0) | B(9) | B(13), 0, 0, 0}, 0, "sse2"},
        {"nehalem", {7, INTEL, 0x106a5, V2_ECX1, 0, V2_ECX81, 0}, STR_SSE42, "sse42"},
        {"haswell", {7, INTEL, 0x306c3, V3_ECX1, V3_EBX7 | ERMS_EBX7, V3_ECX81, 0x7}, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"},
        {"haswell, os without ymm", {7, INTEL, 0x306c3, V3_ECX1, V3_EBX7 | ERMS_EBX7, V3_ECX81, 0x1}, STR_SSE42 | STR_ERMS, "sse42"},
        {"avx2 without erms", {7, AMD, 0x800f12, V3_ECX1, V3_EBX7, V3_ECX81, 0x7}, STR_SSE42 | STR_AVX2, "sse42"},
        {"skylake-sp", {7, INTEL, 0x50654, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7}, STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ERMS, "evex"},
        {"skylake-sp, os without zmm", {7, INTEL, 0x50654, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0x7}, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"},
        {"ice lake", {7, INTEL, 0x606a6, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7}, STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, "evex512"},
        {"zen 4", {7, AMD, 0xa60f12, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7}, STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, "evex512"},
        {"zen 5", {7, AMD, 0xb20f40, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7, 0x30, 0x80000028, 0x597fffcf},
         STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS | STR_FSRS, "evex512r"},
        {"zen 5, extended leaves short of 0x80000021", {7, AMD, 0xb20f40, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7, 0x30, 0x80000020, 0x597fffcf},
         STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, "evex512"},
        {"zen 4 with the extended leaves, no fast short rep stosb", {7, AMD, 0xa60f12, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7, 0, 0x80000028, 0x597ffbcf},
         STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, "evex512"},
        {"sapphire rapids", {7, INTEL, 0x806f8, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7, 1u << 11, 0x80000008, 1u << 10},
         STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS | STR_FSRS, "evex512r"},
        {"skylake-sp with garbage past its extended leaves", {7, INTEL, 0x50654, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7, 0, 0x80000008, 0x960},
         STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ERMS, "evex"},
        {"avx-512 without erms", {7, AMD, 0xa60f12, V3_ECX1, V4_EBX7, V3_ECX81, 0xe7}, STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM, "sse42"},
        {"vendor like intel in ebx only", {7, 0x756e6547, 0, 0, 0x50654, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7}, STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, "evex512"},
        {"vendor like intel in ebx and edx", {7, 0x756e6547, 0x49656e69, 0, 0x50654, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7}, STR_SSE42 | STR_AVX2 | STR_AVX512 | STR_ZMM | STR_ERMS, "evex512"},
    };
    phase = "hosts";

    for (size_t i = 0; i < sizeof machines / sizeof *machines; i++) {
        unsigned bits = str_host_of(&machines[i].c);

        if (bits != machines[i].bits || strcmp(str_choose_for(bits)->name, machines[i].config)) {
            fprintf(stderr, "FAIL hosts: %s: bits %#x, configuration %s\n", machines[i].name, bits, str_choose_for(bits)->name);
            exit(2);
        }

        checks++;
    }

    /* every feature bit missing on its own from a Zen 4: the level that needs it is
       lost, the ones below stay (each test of the checks fails in turn) */
    const struct { int word; unsigned bit; unsigned keep; const char *config; } one_missing[] = {
        {1, 0, STR_ERMS, "sse2"}, {1, 9, STR_ERMS, "sse2"}, {1, 19, STR_ERMS, "sse2"}, {1, 20, STR_ERMS, "sse2"},
        {1, 23, STR_ERMS, "sse2"}, {1, 13, STR_ERMS, "sse2"}, {3, 0, STR_ERMS, "sse2"},
        {1, 27, STR_SSE42 | STR_ERMS, "sse42"}, {1, 28, STR_SSE42 | STR_ERMS, "sse42"}, {1, 29, STR_SSE42 | STR_ERMS, "sse42"},
        {1, 12, STR_SSE42 | STR_ERMS, "sse42"}, {1, 22, STR_SSE42 | STR_ERMS, "sse42"}, {2, 5, STR_SSE42 | STR_ERMS, "sse42"},
        {2, 3, STR_SSE42 | STR_ERMS, "sse42"}, {2, 8, STR_SSE42 | STR_ERMS, "sse42"}, {3, 5, STR_SSE42 | STR_ERMS, "sse42"},
        {2, 16, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"}, {2, 30, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"},
        {2, 28, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"}, {2, 17, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"},
        {2, 31, STR_SSE42 | STR_AVX2 | STR_ERMS, "avx2"},
    };

    for (size_t i = 0; i < sizeof one_missing / sizeof *one_missing; i++) {
        struct str_cpuid c = {7, AMD, 0xa60f12, V3_ECX1, V4_EBX7 | ERMS_EBX7, V3_ECX81, 0xe7};
        unsigned *word = one_missing[i].word == 1 ? &c.ecx1 : one_missing[i].word == 2 ? &c.ebx7 : &c.ecx81;
        *word &= ~B(one_missing[i].bit);
        unsigned bits = str_host_of(&c);

        if (bits != one_missing[i].keep || strcmp(str_choose_for(bits)->name, one_missing[i].config)) {
            fprintf(stderr, "FAIL hosts: without word %d bit %u: bits %#x, configuration %s\n", one_missing[i].word, one_missing[i].bit, bits, str_choose_for(bits)->name);
            exit(2);
        }

        checks++;
    }

    printf("hosts: PASS %zu machines\n", checks);
}

/* The bits of a configuration's needs that say what it executes; the others
   (ERMS, ZMM, FSRS) say where it is fast, and the test runs it anyway. */
#define EXECUTES (STR_SSE42 | STR_AVX2 | STR_AVX512)

int main(void) {
    /* every configuration the host can execute, then the host's own choice */
    unsigned host = str_host();

    for (unsigned i = 0; i < str_nconfigs; i++) {
        if (str_configs[i].needs & EXECUTES & ~host) {
            continue;
        }

        str_table = str_configs[i].table;
        Table = &str_table;
        printf("%s: ", str_configs[i].name);
        checks = 0;
        suite(1);
    }

    str_dispatch();
    Table = &str_table;
    printf("dispatched (%s): ", str_choose()->name);
    checks = 0;
    suite(1);
    /* every slot of the first and of the best configuration replaced by every
       other implementation the host can run, one at a time: the quick suite */
    const struct str_config *bases[2] = {&str_configs[0], str_choose()};
    size_t mixes = 0;
    checks = 0;

    for (int bi = 0; bi < 2; bi++) {
        void *base[NSLOTS];
        slots_of(&bases[bi]->table, base);

        for (size_t j = 0; j < NSLOTS; j++) {
            void *tried[16];
            size_t ntried = 0;

            for (unsigned k = 0; k < str_nconfigs; k++) {
                if (str_configs[k].needs & EXECUTES & ~host) {
                    continue;
                }

                void *other[NSLOTS];
                slots_of(&str_configs[k].table, other);
                int seen = other[j] == base[j];

                for (size_t m = 0; m < ntried && !seen; m++) {
                    seen = tried[m] == other[j];
                }

                if (seen) {
                    continue;
                }

                tried[ntried++] = other[j];
                void *slots[NSLOTS];
                memcpy(slots, base, sizeof slots);
                slots[j] = other[j];
                table_of(&str_table, slots);
                Table = &str_table;
                snprintf(mix, sizeof mix, " (%s with slot %zu from %s)", bases[bi]->name, j, str_configs[k].name);
                suite(0);
                mixes++;
            }
        }
    }

    mix[0] = 0;
    printf("mixes: PASS %zu checks in %zu one-slot mixes of %s and %s\n", checks, mixes, bases[0]->name, bases[1]->name);
    checks = 0;
    hosts();
    checks = 0;
    entries();
    checks = 0;
    derived();
    return 0;
}
