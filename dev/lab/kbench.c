/*
 * kbench: the kernel lab's driver. Every implementation of every primitive
 * in the binary is a kernel (kernels-gen.c lists them, written by lab.py
 * from nm of the link inputs); ours are built here from the configurations
 * and the slot implementations: each configuration whole, then the host's
 * configuration with one slot of the primitive (its entry or a kernel behind
 * it) replaced by every other implementation of that slot.
 *
 * Every kernel the host can run is validated first, in a child process of
 * its own (a crash or a wrong answer drops the kernel, not the run): sizes,
 * alignments, contents, overlap, guard pages on both sides. Then the matrix:
 * per primitive, scenario, size and alignment pair all kernels of the
 * primitive in a shuffled order per round, a batch of calls each, the minimum
 * and median ns per call over the rounds. Regimes: thr (independent calls)
 * and lat (each call's pointers depend on the previous result). Then the
 * workload traces (workload.py) per primitive with every kernel.
 *
 * Output: CSV on stdout (prim,scenario,regime,n,align,kernel,min_ns,med_ns,iters,ghz,pmu),
 * progress on stderr, facts (host, features, kernels, validation, the glibc
 * and our dispatch) as JSON in --meta FILE.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "dev/lab/kl.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "kbench: %s:%d: %s\n", __FILE__, __LINE__, #x); exit(2); } } while (0)

static const char *PN[P_N] = {"memcpy", "memmove", "memset", "memcmp", "strlen", "strcpy", "strcat", "strcmp", "strspn", "strcspn", "memchr"};
static const char *SLOTN[] = {"memcpy", "memmove", "memset", "memcmp", "strlen", "strcpy", "strcat", "strcmp", "strspn", "strcspn", "memchr",
                              "copy_large", "fill_large", "memcmp_large", "strlen_tail", "strcmp_tail", "strcpy_long", "copy_huge", "fill_huge", "memchr_large"};
#define NSLOTS ((int)(sizeof(struct str_table) / sizeof(void *)))
#define SLOT(f) ((int)(offsetof(struct str_table, f) / sizeof(void *)))

typedef void *(*f_cpy)(void *, const void *, size_t);
typedef void *(*f_set)(void *, int, size_t);
typedef int (*f_cmp)(const void *, const void *, size_t);
typedef size_t (*f_len)(const char *);
typedef char *(*f_scpy)(char *, const char *);
typedef int (*f_scmp)(const char *, const char *);
typedef size_t (*f_spn)(const char *, const char *);
typedef void *(*f_chr)(const void *, int, size_t);

/* ------------------------------------------------------------------ host */

static unsigned host;
static char brand[49];

/* sched_setaffinity by its number: <sched.h> with _GNU_SOURCE wants the kernel's
   headers, which Debian's libc6-dev does not carry */
static int pin(int cpu) {
    unsigned long mask[16] = {0};
    long r;
    mask[cpu / 64] |= 1ul << (cpu % 64);
    __asm__ volatile("syscall" : "=a"(r) : "a"(203), "D"(0), "S"(sizeof mask), "d"(mask) : "rcx", "r11", "memory");
    return (int)r;
}

static void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}

static unsigned detect(void) {
    unsigned r[4], f = 0;
    cpuid(0, 0, r);
    unsigned max = r[0];
    cpuid(1, 0, r);
    unsigned ecx1 = r[2];
    unsigned xcr0 = 0;

    if (ecx1 & (1u << 27)) {
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = lo;
    }

    int ymm = (xcr0 & 6) == 6, zmm = ymm && (xcr0 & 0xe0) == 0xe0;

    if (ecx1 & (1u << 9)) {
        f |= F_SSSE3;
    }

    if (ecx1 & (1u << 20)) {
        f |= F_SSE42;
    }

    if (ecx1 & (1u << 22)) {
        f |= F_MOVBE;
    }

    if ((ecx1 & (1u << 28)) && ymm) {
        f |= F_AVX;
    }

    if (max >= 7) {
        cpuid(7, 0, r);
        unsigned b = r[1], d = r[3];
        cpuid(0x80000001u, 0, r);
        int lzcnt = (r[2] >> 5) & 1, fma = (ecx1 >> 12) & 1;

        if ((b & (1u << 5)) && (b & (1u << 3)) && (b & (1u << 8)) && lzcnt && fma && (f & F_MOVBE) && (f & F_AVX)) {
            f |= F_AVX2;
        }

        unsigned v4 = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);

        if ((b & v4) == v4 && zmm && (f & F_AVX2)) {
            f |= F_AVX512;
        }

        if (b & (1u << 9)) {
            f |= F_ERMS;
        }

        if (b & (1u << 11)) {
            f |= F_RTM;
        }

        if (d & (1u << 4)) {
            f |= F_FSRM;
        }
    }

    for (unsigned i = 0; i < 3; i++) {
        cpuid(0x80000002u + i, 0, r);
        memcpy(brand + 16 * i, r, 16);
    }

    brand[48] = 0;
    return f;
}

/* ---------------------------------------------------------------- kernels */

typedef struct {
    char name[200];
    const char *origin;
    int prim;
    void *fn;
    unsigned needs;
    int ours;
    struct str_table table;
    struct str_table *tgt; /* where installing it writes: str_table, or the base revision's (0: str_table) */
    int status; /* 0 untested, 1 valid, -1 failed, -2 not for this host */
    char why[160];
    double est;   /* ns per call, for the batch size */
    size_t iters; /* this cell's batch */
    double s[64]; /* this cell's samples */
    uint64_t sum; /* a trace's checksum */
    double ghz[64]; /* the core's average clock in each trace pass, when the MSRs can be read */
} Kernel;

static Kernel *K;
static int NK, CAPK;
static const struct str_config *base_config;

static Kernel *add_kernel(void) {
    if (NK == CAPK) {
        CAPK = CAPK ? 2 * CAPK : 256;
        K = realloc(K, CAPK * sizeof *K);
        CHECK(K);
    }

    Kernel *k = &K[NK++];
    memset(k, 0, sizeof *k);
    return k;
}

static const char *slot_fn_name(int slot, void *fn) {
    for (unsigned i = 0; i < kl_nslots; i++) {
        if (kl_slots[i].slot == slot && kl_slots[i].fn == fn) {
            return kl_slots[i].name;
        }
    }

    return "?";
}

static unsigned slot_fn_needs(int slot, void *fn) {
    for (unsigned i = 0; i < kl_nslots; i++) {
        if (kl_slots[i].slot == slot && kl_slots[i].fn == fn) {
            return kl_slots[i].needs;
        }
    }

    return 0;
}

/* The slots a primitive's calls go through: its own, then the kernels behind it. */
static int prim_slots(int p, int *out) {
    int n = 0;
    out[n++] = p;

    switch (p) {
    case P_MEMCPY: case P_MEMMOVE: out[n++] = SLOT(copy_large); out[n++] = SLOT(copy_huge); break;
    case P_MEMSET: out[n++] = SLOT(fill_large); out[n++] = SLOT(fill_huge); break;
    case P_MEMCMP: out[n++] = SLOT(memcmp_large); break;
    case P_STRLEN: out[n++] = SLOT(strlen_tail); break;
    case P_STRCPY: out[n++] = SLOT(strcpy_long); break;
    case P_STRCAT: out[n++] = SLOT(strlen); out[n++] = SLOT(strcpy); out[n++] = SLOT(strlen_tail); out[n++] = SLOT(strcpy_long); break;
    case P_STRCMP: out[n++] = SLOT(strcmp_tail); break;
    case P_MEMCHR: out[n++] = SLOT(memchr_large); break;
    default: break;
    }

    return n;
}

static int same_slots(const struct str_table *a, const struct str_table *b, int p) {
    int sl[8], n = prim_slots(p, sl);

    for (int i = 0; i < n; i++) {
        if (((void *const *)a)[sl[i]] != ((void *const *)b)[sl[i]]) {
            return 0;
        }
    }

    return 1;
}

static int have_ours(int p, const struct str_table *t) {
    for (int i = 0; i < NK; i++) {
        if (K[i].ours && K[i].prim == p && same_slots(&K[i].table, t, p)) {
            return 1;
        }
    }

    return 0;
}

static unsigned table_needs(const struct str_table *t, int p) {
    int sl[8], n = prim_slots(p, sl);
    unsigned needs = 0;

    for (int i = 0; i < n; i++) {
        needs |= slot_fn_needs(sl[i], ((void *const *)t)[sl[i]]);
    }

    return needs;
}

static void build_kernels(void) {
    CHECK(NSLOTS == (int)(sizeof SLOTN / sizeof *SLOTN));

    for (unsigned i = 0; i < kl_ndefs; i++) {
        int dup = 0; /* an alias of a variant already listed (FreeBSD's resolved names) */

        for (int j = 0; j < NK; j++) {
            if (K[j].prim == kl_defs[i].prim && K[j].fn == kl_defs[i].fn) {
                dup = 1;
            }
        }

        if (dup) {
            continue;
        }

        Kernel *k = add_kernel();
        snprintf(k->name, sizeof k->name, "%s/%s", kl_defs[i].origin, kl_defs[i].name);
        k->origin = kl_defs[i].origin;
        k->prim = kl_defs[i].prim;
        k->fn = kl_defs[i].fn;
        k->needs = kl_defs[i].needs;
    }

    /* ours: every configuration whole, as dispatch installs it (the post pass's
       copies for that configuration, their calls to each other direct); then the
       host's configuration through the table (every slot's plain implementation,
       whose calls go through str_table), and that with one slot of the primitive
       replaced by every other implementation of the slot */
    base_config = str_choose();
    struct str_table plain = base_config->table;

    int jcc = 0; /* the post pass laid this configuration out with the JCC padding: its copies are the *_jcc code */

    for (int i = 0; kl_jcc_configs[i]; i++) {
        jcc |= !strcmp(kl_jcc_configs[i], base_config->name);
    }

    for (int s = 0; s < NSLOTS; s++) {
        const char *nm = slot_fn_name(s, ((void **)&plain)[s]);
        const char *dot = strchr(nm, '.');
        char want[160];
        snprintf(want, sizeof want, "%s%s", dot ? dot + 1 : "", jcc ? "_jcc" : "");

        for (unsigned i = 0; dot && i < kl_nslots; i++) {
            if (kl_slots[i].slot == s && !kl_slots[i].spec && !strcmp(kl_slots[i].name, want)) {
                ((void **)&plain)[s] = kl_slots[i].fn;
            }
        }
    }

    for (int p = 0; p < P_N; p++) {
        for (unsigned c = 0; c < str_nconfigs; c++) {
            const struct str_table *t = &str_configs[c].table;

            if (have_ours(p, t)) {
                continue;
            }

            Kernel *k = add_kernel();
            snprintf(k->name, sizeof k->name, "ours/%s", str_configs[c].name);
            k->origin = "ours";
            k->prim = p;
            k->ours = 1;
            k->table = *t;
            k->fn = ((void *const *)t)[p];
            unsigned n = str_configs[c].needs;
            k->needs = (n & STR_SSE42 ? F_SSE42 : 0) | (n & STR_AVX2 ? F_AVX2 : 0) | (n & STR_AVX512 ? F_AVX512 : 0);
        }

        /* the base revision's configurations whole (lab.py --base) */
        for (unsigned c = 0; c < kl_base_nconfigs; c++) {
            Kernel *k = add_kernel();
            snprintf(k->name, sizeof k->name, "base/%s", kl_base_configs[c].name);
            k->origin = "base";
            k->prim = p;
            k->ours = 1;
            k->table = kl_base_configs[c].table;
            k->tgt = kl_base_table;
            k->fn = ((void *const *)&k->table)[p];
            unsigned n = kl_base_configs[c].needs;
            k->needs = (n & STR_SSE42 ? F_SSE42 : 0) | (n & STR_AVX2 ? F_AVX2 : 0) | (n & STR_AVX512 ? F_AVX512 : 0);
        }

        if (!have_ours(p, &plain)) {
            Kernel *k = add_kernel();
            snprintf(k->name, sizeof k->name, "ours/%s.t", base_config->name);
            k->origin = "ours";
            k->prim = p;
            k->ours = 1;
            k->table = plain;
            k->fn = ((void **)&plain)[p];
            k->needs = table_needs(&plain, p);
        }

        int sl[8], n = prim_slots(p, sl);
        int vary = n; /* strcat too: its strlen and strcpy slots change it as much as its entry */

        for (int j = 0; j < vary; j++) {
            for (unsigned i = 0; i < kl_nslots; i++) {
                if (kl_slots[i].slot != sl[j] || kl_slots[i].spec) {
                    continue;
                }

                struct str_table t = plain;
                ((void **)&t)[sl[j]] = kl_slots[i].fn;

                if (have_ours(p, &t)) {
                    continue;
                }

                Kernel *k = add_kernel();
                snprintf(k->name, sizeof k->name, "ours/%s.t+%s=%s", base_config->name, SLOTN[sl[j]], kl_slots[i].name);
                k->origin = "ours";
                k->prim = p;
                k->ours = 1;
                k->table = t;
                k->fn = ((void **)&t)[p];
                k->needs = table_needs(&t, p);
            }
        }
    }
}

static void install(const Kernel *k) {
    if (k->ours) {
        *(k->tgt ? k->tgt : &str_table) = k->table;
    }
}

/* ------------------------------------------------------------- references */

typedef unsigned char u8x;

static int ref_cmp(const u8x *a, const u8x *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return a[i] - b[i];
        }
    }

    return 0;
}

static size_t ref_len(const u8x *s) {
    size_t i = 0;

    while (s[i]) {
        i++;
    }

    return i;
}

static int ref_scmp(const u8x *a, const u8x *b) {
    size_t i = 0;

    while (a[i] && a[i] == b[i]) {
        i++;
    }

    return a[i] - b[i];
}

static size_t ref_spn(const u8x *s, const u8x *set, int accept) {
    u8x map[256] = {0};

    for (; *set; set++) {
        map[*set] = 1;
    }

    size_t i = 0;

    while (s[i] && map[s[i]] == accept) {
        i++;
    }

    return i;
}

static const u8x *ref_chr(const u8x *s, int c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == (u8x)c) {
            return s + i;
        }
    }

    return 0;
}

static int sgn(int x) {
    return (x > 0) - (x < 0);
}

/* -------------------------------------------------------------- validation */

static char vwhy[160];
#define VFAIL(...) do { snprintf(vwhy, sizeof vwhy, __VA_ARGS__); return 0; } while (0)

static void fill_pattern(u8x *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        p[i] = (u8x)(1 + (i * 7 + seed) % 251); /* never 0, 0xfc..0xff absent */
    }
}

/* One region with an inaccessible page before and after it. */
typedef struct { u8x *lo, *hi; } Area;
static Area area_new(size_t pages) {
    u8x *m = mmap(0, (pages + 2) * 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(m != MAP_FAILED);
    CHECK(!mprotect(m + 4096, pages * 4096, PROT_READ | PROT_WRITE));
    return (Area){m + 4096, m + 4096 + pages * 4096};
}

static const size_t VOFF[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 48, 49, 63};
#define NVOFF (sizeof VOFF / sizeof *VOFF)
static const size_t VOFF_BIG[] = {0, 1, 17, 31, 63};

static size_t vsizes[512];
static int nvsizes;
static void make_vsizes(void) {
    for (size_t n = 0; n <= 160; n++) {
        vsizes[nvsizes++] = n;
    }

    static const size_t more[] = {191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 513, 767, 768, 1023, 1024, 1025,
                                  2047, 2048, 2049, 3000, 4095, 4096, 4097, 8191, 8192, 8193, 12289};

    for (size_t i = 0; i < sizeof more / sizeof *more; i++) {
        vsizes[nvsizes++] = more[i];
    }
}

/* One case of a kernel: objects at s and d (d may be unused), n bytes;
   sroom and droom are the bytes accessible from s and from d on. */
static int vcase(const Kernel *k, u8x *s, size_t sroom, u8x *d, size_t droom, size_t n) {
    size_t dend = k->prim == P_STRCPY ? n + 1 : n; /* the bytes the call may write */
    const size_t G = droom > dend ? (droom - dend < 16 ? droom - dend : 16) : 0;

    switch (k->prim) {
    case P_MEMCPY: case P_MEMMOVE: {
        fill_pattern(s, n, (unsigned)n);
        memset(d, 0x55, n + G);
        void *r = ((f_cpy)k->fn)(d, s, n);

        if (r != d) {
            VFAIL("returns %p, not d", r);
        }

        if (ref_cmp(s, d, n)) {
            VFAIL("copy differs");
        }

        for (size_t i = 0; i < G; i++) {
            if (d[n + i] != 0x55) {
                VFAIL("writes past the end (+%zu)", i);
            }
        }

        return 1;
    }

    case P_MEMSET: {
        static const int cs[] = {0, 0x5a, 0xa5, 0x1ff};

        for (int ci = 0; ci < 4; ci++) {
            memset(d, 0x33, n + G);
            void *r = ((f_set)k->fn)(d, cs[ci], n);

            if (r != d) {
                VFAIL("returns %p, not d", r);
            }

            for (size_t i = 0; i < n; i++) {
                if (d[i] != (u8x)cs[ci]) {
                    VFAIL("byte %zu is %02x, c=%x", i, d[i], cs[ci]);
                }
            }

            for (size_t i = 0; i < G; i++) {
                if (d[n + i] != 0x33) {
                    VFAIL("writes past the end (+%zu)", i);
                }
            }
        }

        return 1;
    }

    case P_MEMCMP: {
        fill_pattern(s, n, 3);
        memcpy(d, s, n);

        if (((f_cmp)k->fn)(s, d, n) != 0) {
            VFAIL("equal blocks compare unequal");
        }

        size_t pl = page_left(s); /* and positions around the page boundary */
        size_t ps[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, n / 2, n - 1, pl - 1, pl, pl + 1, pl + 15, pl + 16, pl + 31, pl + 32, pl + 63};

        for (size_t i = 0; i < sizeof ps / sizeof *ps && n; i++) {
            size_t p = ps[i];

            if (p >= n) {
                continue;
            }

            u8x keep = d[p];
            static const u8x pairs[][2] = {{0x80, 0x7f}, {0x7f, 0x80}, {0x01, 0xff}, {0xff, 0x00}};

            for (int j = 0; j < 4; j++) {
                u8x ks = s[p];
                s[p] = pairs[j][0];
                d[p] = pairs[j][1];
                int r = ((f_cmp)k->fn)(s, d, n), e = ref_cmp(s, d, n);
                s[p] = ks;

                if (sgn(r) != sgn(e)) {
                    VFAIL("difference at %zu (%02x vs %02x): %d", p, pairs[j][0], pairs[j][1], r);
                }
            }

            d[p] = keep;
        }

        return 1;
    }

    case P_STRLEN: {
        fill_pattern(s, n, 5);
        s[n] = 0;
        size_t r = ((f_len)k->fn)((char *)s);

        if (r != n) {
            VFAIL("%zu", r);
        }

        return 1;
    }

    case P_STRCPY: {
        fill_pattern(s, n, 9);
        s[n] = 0;
        memset(d, 0x55, n + 1 + G);
        char *r = ((f_scpy)k->fn)((char *)d, (char *)s);

        if (r != (char *)d) {
            VFAIL("returns %p, not d", (void *)r);
        }

        if (ref_cmp(s, d, n + 1)) {
            VFAIL("copy differs");
        }

        for (size_t i = 0; i < G; i++) {
            if (d[n + 1 + i] != 0x55) {
                VFAIL("writes past the NUL (+%zu)", i);
            }
        }

        return 1;
    }

    case P_STRCAT: {
        static const size_t ms[] = {0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100};
        fill_pattern(s, n, 11);
        s[n] = 0;

        for (size_t j = 0; j < sizeof ms / sizeof *ms; j++) {
            size_t m = ms[j];

            if (m + n + 1 > droom) {
                continue;
            }

            size_t g = droom - (m + n + 1) < 16 ? droom - (m + n + 1) : 16;
            memset(d, 0x55, m + n + 1 + g);
            fill_pattern(d, m, 13);
            d[m] = 0;
            char *r = ((f_scpy)k->fn)((char *)d, (char *)s);

            if (r != (char *)d) {
                VFAIL("returns %p, not d", (void *)r);
            }

            if (ref_cmp(s, d + m, n + 1)) {
                VFAIL("append to %zu bytes differs", m);
            }

            for (size_t i = 0; i < g; i++) {
                if (d[m + n + 1 + i] != 0x55) {
                    VFAIL("writes past the NUL (+%zu), prefix %zu", i, m);
                }
            }
        }

        return 1;
    }

    case P_STRCMP: {
        fill_pattern(s, n, 17);
        s[n] = 0;
        memcpy(d, s, n + 1);

        if (((f_scmp)k->fn)((char *)s, (char *)d) != 0) {
            VFAIL("equal strings compare unequal");
        }

        size_t pl = page_left(s); /* and positions around the page boundary */
        size_t ps[] = {0, 1, 15, 16, 31, 32, 63, 64, n / 2, n - 1, pl - 1, pl, pl + 1, pl + 15, pl + 16, pl + 31, pl + 32, pl + 63};

        for (size_t i = 0; i < sizeof ps / sizeof *ps && n; i++) {
            size_t p = ps[i];

            if (p >= n) {
                continue;
            }

            static const u8x vals[][2] = {{0x80, 0x7f}, {0x7f, 0x80}, {0x01, 0xff}, {0x00, 0x41}, {0x41, 0x00}};

            for (int j = 0; j < 5; j++) {
                u8x ks = s[p], kd = d[p];
                s[p] = vals[j][0];
                d[p] = vals[j][1];
                int r = ((f_scmp)k->fn)((char *)s, (char *)d), e = ref_scmp(s, d);
                s[p] = ks;
                d[p] = kd;

                if (sgn(r) != sgn(e)) {
                    VFAIL("difference at %zu (%02x vs %02x): %d", p, vals[j][0], vals[j][1], r);
                }
            }
        }

        /* the bytes after the NUL differ: still equal */
        if (n + 2 <= sroom && n + 2 <= droom) {
            s[n + 1] = 1;
            d[n + 1] = 2;

            if (((f_scmp)k->fn)((char *)s, (char *)d) != 0) {
                VFAIL("compares past the NUL");
            }
        }

        return 1;
    }

    case P_MEMCHR: {
        fill_pattern(s, n + (n + 1 <= sroom ? 1 : 0), 19);

        if (((f_chr)k->fn)(s, 0xfe, n)) {
            VFAIL("finds an absent byte");
        }

        size_t pl = page_left(s); /* and positions around the page boundary */
        size_t ps[] = {0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, n / 2, n - 1, pl - 1, pl, pl + 1, pl + 15, pl + 16, pl + 17, pl + 31, pl + 32, pl + 33, pl + 63, pl + 64};

        for (size_t i = 0; i < sizeof ps / sizeof *ps && n; i++) {
            size_t p = ps[i];

            if (p >= n) {
                continue;
            }

            u8x keep = s[p];
            s[p] = 0xfe;
            const void *r = ((f_chr)k->fn)(s, 0x1fe, n), *e = ref_chr(s, 0xfe, n);
            s[p] = keep;

            if (r != e) {
                VFAIL("byte at %zu: %p, expected %p", p, r, e);
            }
        }

        if (n + 1 <= sroom) { /* a match just past the end is not found */
            s[n] = 0xfe;
            const void *r = ((f_chr)k->fn)(s, 0xfe, n);
            s[n] = 1;

            if (r) {
                VFAIL("finds the byte after the end");
            }
        }

        return 1;
    }

    default: return 1;
    }
}

/* strspn/strcspn: sets of many sizes; strings that stop at position p. */
static const char *vsets[] = {"a", "ab", " \t", "abc", " \t\r\n", "abcdefghijklmno", "abcdefghijklmnop", "abcdefghijklmnopq",
                              "abcdefghijklmnopqrstuvwxyz01234", "abcdefghijklmnopqrstuvwxyz012345", "abcdefghijklmnopqrstuvwxyz0123456",
                              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/",
                              "\x80", "\xff\x80\x81", "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90",
                              "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\r\n\x7f\xc0\xd0\xe0\xf0"};
#define NVSETS (sizeof vsets / sizeof *vsets)

static int vspan(const Kernel *k, u8x *s, size_t p, const u8x *set, size_t room) {
    int accept = k->prim == P_STRSPN;
    u8x in[256], out[256];
    int nin = 0, nout = 0;
    u8x map[256] = {0};

    for (const u8x *q = set; *q; q++) {
        map[*q] = 1;
    }

    for (int b = 1; b < 256; b++) {
        if (map[b]) {
            in[nin++] = (u8x)b;
        } else {
            out[nout++] = (u8x)b;
        }
    }

    if (p + 3 > room) {
        return 1;
    }

    const u8x *run = accept ? in : out;
    int nrun = accept ? nin : nout;
    const u8x *stop = accept ? out : in;
    int nstop = accept ? nout : nin;

    for (size_t i = 0; i < p; i++) {
        s[i] = nrun ? run[(i * 7) % nrun] : 1;
    }

    for (int variant = 0; variant < 2; variant++) {
        if (variant == 0) {
            if (!nstop) {
                continue;
            }

            s[p] = stop[p % nstop];
            s[p + 1] = run && nrun ? run[0] : 1;
            s[p + 2] = 0;
        } else {
            s[p] = 0;
        }

        size_t r = ((f_spn)k->fn)((char *)s, (const char *)set), e = ref_spn(s, set, accept);

        if (r != e) {
            VFAIL("set of %zu, stop at %zu (%s): %zu, expected %zu", ref_len(set), p, variant ? "NUL" : "byte", r, e);
        }
    }

    return 1;
}

static int validate_kernel(Kernel *k) {
    Area A = area_new(8), B = area_new(8);
    size_t span = (size_t)(A.hi - A.lo);

    if (k->prim == P_STRSPN || k->prim == P_STRCSPN) {
        static const size_t ps[] = {0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 127, 128, 129, 255, 256, 511, 1000, 4096, 6000};

        for (size_t si = 0; si < NVSETS; si++) {
            /* the set at an ordinary place and at the very end of its page */
            size_t sl = strlen(vsets[si]) + 1;
            u8x *set1 = B.lo + 100, *set2 = B.hi - sl;
            memcpy(set1, vsets[si], sl);
            memcpy(set2, vsets[si], sl);

            for (size_t oi = 0; oi < NVOFF; oi++) {
                for (size_t pi = 0; pi < sizeof ps / sizeof *ps; pi++) {
                    if (!vspan(k, A.lo + VOFF[oi], ps[pi], set1, span - VOFF[oi])) {
                        return 0;
                    }

                    if (oi < 3 && !vspan(k, A.lo + VOFF[oi], ps[pi], set2, span - VOFF[oi])) {
                        return 0;
                    }
                }
            }

            for (size_t p = 0; p < 300; p++) /* the string's NUL at the page end */

                if (!vspan(k, A.hi - p - 3, p, set1, p + 3)) {
                    return 0;
                }
        }

        /* the empty set */
        memcpy(A.lo, "abc", 4);
        size_t r = ((f_spn)k->fn)((char *)A.lo, "");

        if (r != (k->prim == P_STRSPN ? 0u : 3u)) {
            VFAIL("empty set: %zu", r);
        }

        return 1;
    }

    for (int si = 0; si < nvsizes; si++) {
        size_t n = vsizes[si];
        int big = n > 1100;
        const size_t *offs = big ? VOFF_BIG : VOFF;
        size_t noffs = big ? 5 : NVOFF;
        size_t need = (k->prim == P_STRCAT ? 2 * n + 200 : n + 64);

        if (need + 64 > span) {
            continue;
        }

        int two = k->prim == P_MEMCPY || k->prim == P_MEMMOVE || k->prim == P_MEMCMP || k->prim == P_STRCPY || k->prim == P_STRCAT || k->prim == P_STRCMP;

        for (size_t i = 0; i < noffs; i++) {
            for (size_t j = 0; j < (two ? noffs : 1); j++) {
                u8x *s = A.lo + offs[i], *d = B.lo + offs[two ? j : i];

                if (!vcase(k, s, (size_t)(A.hi - s), d, (size_t)(B.hi - d), n)) {
                    return 0;
                }
            }
        }

        if (k->prim == P_MEMMOVE) {
            long ds[] = {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, (long)n / 2, (long)n - 1};
            static u8x ref[8 * 4096];

            for (size_t i = 0; i < (big ? 3u : 5u); i++) {
                for (size_t di = 0; di < sizeof ds / sizeof *ds; di++) {
                    for (int sign = -1; sign <= 1; sign += 2) {
                        long delta = ds[di] * sign;

                        if (!delta) {
                            continue;
                        }

                        size_t lo = (size_t)(ds[di] + 64), off = lo + offs[i];

                        if (off + n + (size_t)ds[di] + 64 > span) {
                            continue;
                        }

                        u8x *s = A.lo + off, *d = s + delta;
                        fill_pattern(A.lo, span, (unsigned)(n + di));
                        memcpy(ref, A.lo, span);
                        memmove(ref + (d - A.lo), ref + (s - A.lo), n);
                        void *r = ((f_cpy)k->fn)(d, s, n);

                        if (r != d) {
                            VFAIL("overlap %ld: returns %p, not d", delta, r);
                        }

                        if (ref_cmp(A.lo, ref, span)) {
                            VFAIL("overlap %ld, n %zu: differs", delta, n);
                        }
                    }
                }
            }
        }
    }

    /* across a page boundary inside the area: the object starts k bytes before
       a page end and goes on into the next page, readable; the page-end paths
       of the entries hand over to kernels there */
    {
        static const size_t ks[] = {1, 2, 3, 7, 8, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 63, 64, 65, 96, 127};
        static const size_t kd[] = {1, 7, 16, 31, 33, 63, 200};
        int two = k->prim == P_MEMCPY || k->prim == P_MEMMOVE || k->prim == P_MEMCMP || k->prim == P_STRCPY || k->prim == P_STRCAT || k->prim == P_STRCMP;

        for (size_t n = 0; n <= 300; n += (n < 140 ? 1 : 7)) {
            for (size_t i = 0; i < sizeof ks / sizeof *ks; i++) {
                for (size_t j = 0; j < (two ? sizeof kd / sizeof *kd : 1); j++) {
                    u8x *s = A.lo + 4096 - ks[i], *d = B.lo + 2 * 4096 - (two ? kd[j] : ks[i]);

                    if (!vcase(k, s, (size_t)(A.hi - s), d, (size_t)(B.hi - d), n)) {
                        return 0;
                    }
                }
            }
        }
    }

    /* guard pages: the objects' last bytes against the inaccessible page after
       them, their first bytes against the one before them */

    for (size_t n = 0; n <= 8192; n += (n < 300 ? 1 : n < 1100 ? 37 : 511)) {
        if (k->prim == P_STRCAT) {
            break; /* below */
        }

        int str = k->prim == P_STRLEN || k->prim == P_STRCPY || k->prim == P_STRCMP;
        size_t ls = n + (str ? 1 : 0), ld = n + (k->prim == P_STRCPY || k->prim == P_STRCMP ? 1 : 0);
        u8x *ss[2] = {A.hi - ls, A.lo}, *dd[2] = {B.hi - ld, B.lo};

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (!vcase(k, ss[i], (size_t)(A.hi - ss[i]), dd[j], (size_t)(B.hi - dd[j]), n)) {
                    return 0;
                }
            }
        }
    }

    /* strcat: the appended string's NUL at the end of its page; the result at the end of its */
    if (k->prim == P_STRCAT) {
        for (size_t n = 0; n <= 300; n++) {
            u8x *s = A.hi - n - 1;

            if (!vcase(k, s, n + 1, B.lo + 1, span - 1, n)) {
                return 0;
            }

            if (!vcase(k, s, n + 1, B.hi - (2 * n + 1), 2 * n + 1, n)) {
                return 0;
            }
        }
    }

    return 1;
}

static void validate_all(int jobs) {
    make_vsizes();
    int running = 0, next = 0;
    pid_t *pids = calloc(NK, sizeof *pids);
    int *pipes = calloc(NK, sizeof *pipes);

    for (int i = 0; i < NK; i++) {
        if ((K[i].needs & ~host) != 0) {
            K[i].status = -2;
            snprintf(K[i].why, sizeof K[i].why, "needs %#x, host %#x", K[i].needs, host);
        }
    }

    for (;;) {
        while (running < jobs && next < NK) {
            Kernel *k = &K[next];

            if (k->status == -2) {
                next++;
                continue;
            }

            int fd[2];
            CHECK(!pipe(fd));
            pid_t pid = fork();
            CHECK(pid >= 0);

            if (!pid) {
                close(fd[0]);
                alarm(120); /* a kernel that loops forever is a failure too */
                install(k);
                int ok = validate_kernel(k);

                if (!ok) {
                    CHECK(write(fd[1], vwhy, strlen(vwhy)) >= 0);
                }

                _exit(ok ? 0 : 3);
            }

            close(fd[1]);
            pids[next] = pid;
            pipes[next] = fd[0];
            running++;
            next++;
        }

        if (!running) {
            break;
        }

        int st;
        pid_t pid = wait(&st);
        CHECK(pid > 0);

        for (int i = 0; i < NK; i++) {
            if (pids[i] == pid) {
                Kernel *k = &K[i];
                ssize_t got = read(pipes[i], k->why, sizeof k->why - 1);
                k->why[got > 0 ? got : 0] = 0;
                close(pipes[i]);

                if (WIFEXITED(st) && WEXITSTATUS(st) == 0) {
                    k->status = 1;
                } else {
                    k->status = -1;

                    if (WIFSIGNALED(st)) {
                        snprintf(k->why, sizeof k->why, WTERMSIG(st) == SIGALRM ? "hangs" : "signal %d", WTERMSIG(st));
                    }

                    fprintf(stderr, "INVALID %s: %s\n", k->name, k->why);
                }

                pids[i] = 0;
                running--;
            }
        }
    }

    int ok = 0, bad = 0, skip = 0;

    for (int i = 0; i < NK; i++) {
        ok += K[i].status == 1, bad += K[i].status == -1, skip += K[i].status == -2;
    }

    fprintf(stderr, "validation: %d valid, %d invalid, %d not for this host\n", ok, bad, skip);
    free(pids);
    free(pipes);
}

/* ------------------------------------------------------------------ timing */

typedef struct {
    int prim;
    const char *scenario;
    size_t n, ao, bo;
    char aname[16];
    u8x *s, *d; /* the objects */
    int c;      /* memset value, memchr byte */
    const char *set;
    size_t strcat_len;
} Cell;

static volatile uintptr_t vzero;
static volatile uintptr_t vsink;

static uint64_t now(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

__attribute__((noinline)) static void run(const Kernel *k, const Cell *c, size_t iters, int lat) {
    uintptr_t acc = 0, off = 0, z = lat ? vzero : 0;
    u8x *s = c->s, *d = c->d;
    size_t n = c->n;
#define LOOP(CALL) do { if (!lat) { for (size_t i = 0; i < iters; i++) { acc += (uintptr_t)(CALL); } } \
                        else { for (size_t i = 0; i < iters; i++) { uintptr_t r = (uintptr_t)(CALL); acc += r; off = r & z; } } } while (0)

    switch (k->prim) {
    case P_MEMCPY: case P_MEMMOVE: { f_cpy f = (f_cpy)k->fn; LOOP(f(d + off, s + off, n)); break; }
    case P_MEMSET: { f_set f = (f_set)k->fn; int v = c->c; LOOP(f(d + off, v, n)); break; }
    case P_MEMCMP: { f_cmp f = (f_cmp)k->fn; LOOP((intptr_t)f(s + off, d + off, n)); break; }
    case P_STRLEN: { f_len f = (f_len)k->fn; LOOP(f((char *)s + off)); break; }
    case P_STRCPY: { f_scpy f = (f_scpy)k->fn; LOOP(f((char *)d + off, (char *)s + off)); break; }
    case P_STRCAT: { f_scpy f = (f_scpy)k->fn; size_t m = c->strcat_len; LOOP((d[m + off] = 0, f((char *)d + off, (char *)s + off))); break; }
    case P_STRCMP: { f_scmp f = (f_scmp)k->fn; LOOP((intptr_t)f((char *)s + off, (char *)d + off)); break; }
    case P_STRSPN: case P_STRCSPN: { f_spn f = (f_spn)k->fn; const char *set = c->set; LOOP(f((char *)s + off, set)); break; }
    case P_MEMCHR: { f_chr f = (f_chr)k->fn; int v = c->c; LOOP(f(s + off, v, n)); break; }
    }
#undef LOOP
    vsink += acc;
}

static double measure(Kernel *k, const Cell *c, size_t iters, int lat) {
    install(k);
    uint64_t t0 = now();
    run(k, c, iters, lat);
    uint64_t t1 = now();
    return (double)(t1 - t0) / (double)iters;
}

/* The core clock: a chain of dependent adds retires one add per cycle. */
static double clock_ghz(void) {
    uint64_t x = 0, t0 = now();

    for (long i = 0; i < 50000000; i++) {
        __asm__ volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    }

    return 2e8 / (double)(now() - t0);
}

static double ghz_start, ghz_end;

/* The core's real clock over a stretch of code: IA32_APERF counts its cycles,
   IA32_MPERF counts at the TSC's rate, both only while it runs; the TSC rate
   times their ratio is the average clock, AVX-512 licenses and turbo in it.
   Needs /dev/cpu/N/msr (root): the lab hosts have it. */
static int msr_fd = -1;
static double tsc_ghz;

static uint64_t rdmsr(unsigned reg) {
    uint64_t v = 0;
    return pread(msr_fd, &v, 8, reg) == 8 ? v : 0;
}

static uint64_t rdtsc(void) {
    unsigned lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (uint64_t)hi << 32 | lo;
}

static void msr_open(int cpu) {
    char path[64];
    snprintf(path, sizeof path, "/dev/cpu/%d/msr", cpu);
    FILE *f = fopen(path, "r"); /* <fcntl.h> wants the kernel's headers; pread on the stream's descriptor */
    msr_fd = f ? fileno(f) : -1;

    if (msr_fd < 0 || !rdmsr(0xe7)) {
        msr_fd = -1;
        return;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t c0 = rdtsc();

    do {
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while ((t1.tv_sec - t0.tv_sec) * 1000000000l + (t1.tv_nsec - t0.tv_nsec) < 50000000);

    tsc_ghz = (double)(rdtsc() - c0) / (double)((t1.tv_sec - t0.tv_sec) * 1000000000l + (t1.tv_nsec - t0.tv_nsec));
}

/* --pmu EV,...: the core's counters over one more pass of each trace per
   kernel or table, after the timed passes, per call in the column pmu
   (EV=count;...). perf_event_open and ioctl by their numbers (the kernel's
   headers again), for this thread in user mode. EV: cycles, instructions,
   branches, branch-misses, or rHEX, a raw event of the CPU (Intel: umask << 8
   | event, cmask << 24; r20c4 the taken branches retired, Broadwell to Ice
   Lake). More events than the core has counters take turns, each scaled. */
static int pmu_n, pmu_fd[8];
static const char *pmu_ev[8];

static void pmu_open(char *spec) {
    static const char *hw[] = {"cycles", "instructions", "cache-references", "cache-misses", "branches", "branch-misses"};
    char *sv;

    for (char *e = strtok_r(spec, ",", &sv); e; e = strtok_r(0, ",", &sv)) {
        struct { uint32_t type, size; uint64_t config, period, sample_type, read_format, flags; uint32_t wakeup, bp_type; uint64_t config1; } at = {0};
        at.size = sizeof at; /* PERF_ATTR_SIZE_VER0 */
        at.config = ~0ull;
        at.flags = 1 | 1 << 5 | 1 << 6; /* disabled, exclude_kernel, exclude_hv */
        at.read_format = 1 | 2;          /* the time enabled and running: more events than counters share them */

        for (unsigned i = 0; i < sizeof hw / sizeof *hw; i++) {
            if (!strcmp(e, hw[i])) {
                at.config = i; /* PERF_TYPE_HARDWARE */
            }
        }

        if (e[0] == 'r' && e[1]) {
            at.type = 4; /* PERF_TYPE_RAW */
            at.config = strtoull(e + 1, 0, 16);
        }

        CHECK(at.config != ~0ull && pmu_n < 8);
        long fd = syscall(298, &at, 0, -1, -1, 0);

        if (fd < 0) {
            fprintf(stderr, "kbench: --pmu %s: perf_event_open failed\n", e);
            exit(2);
        }

        pmu_fd[pmu_n] = (int)fd;
        pmu_ev[pmu_n++] = e;
    }
}

static void pmu_start(void) {
    for (int i = 0; i < pmu_n; i++) {
        syscall(16, pmu_fd[i], 0x2403, 0); /* PERF_EVENT_IOC_RESET */
    }

    for (int i = 0; i < pmu_n; i++) {
        syscall(16, pmu_fd[i], 0x2400, 0); /* PERF_EVENT_IOC_ENABLE */
    }
}

static void pmu_stop(char *out, size_t cap, double calls) {
    size_t w = 0;
    out[0] = 0;

    for (int i = 0; i < pmu_n; i++) {
        syscall(16, pmu_fd[i], 0x2401, 0); /* PERF_EVENT_IOC_DISABLE */
    }

    for (int i = 0; i < pmu_n && w < cap; i++) {
        uint64_t v[3] = {0};
        CHECK(read(pmu_fd[i], v, sizeof v) == sizeof v);
        double n = v[2] ? (double)v[0] * (double)v[1] / (double)v[2] : 0; /* the count over the time it ran, scaled to the time enabled */
        w += (size_t)snprintf(out + w, cap - w, "%s%s=%.3f", i ? ";" : "", pmu_ev[i], n / calls);
    }
}

static unsigned long long rng_state = 88172645463325252ull;
static unsigned rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (unsigned)(rng_state >> 11);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int samples = 5;
static double target_ns = 40000;

static void time_cell(Kernel **ks, int nk, const Cell *c, const char *regime) {
    int lat = !strcmp(regime, "lat");

    /* calibrate: a batch of about target_ns per kernel, from the last estimate */
    for (int i = 0; i < nk; i++) {
        Kernel *k = ks[i];
        install(k);
        run(k, c, 2, lat);

        for (int tries = 0; tries < 4; tries++) {
            double est = k->est > 0.3 ? k->est : 0.3;
            size_t it = (size_t)(target_ns / est);

            if (it < 2) {
                it = 2;
            }

            if (it > 50000000) {
                it = 50000000;
            }

            double ns = measure(k, c, it, lat);
            k->iters = it;
            double total = ns * (double)it;
            k->est = ns;

            if (total > target_ns / 3 && total < target_ns * 3) {
                break;
            }
        }
    }

    int order[1024];
    CHECK(nk <= 1024 && samples <= 64);

    for (int r = 0; r < samples; r++) {
        for (int i = 0; i < nk; i++) {
            order[i] = i;
        }

        for (int i = nk - 1; i > 0; i--) {
            int j = (int)(rnd() % (unsigned)(i + 1)), t = order[i];
            order[i] = order[j];
            order[j] = t;
        }

        for (int i = 0; i < nk; i++) {
            Kernel *k = ks[order[i]];
            k->s[r] = measure(k, c, k->iters, lat);
        }
    }

    for (int i = 0; i < nk; i++) {
        Kernel *k = ks[i];
        double v[64];
        memcpy(v, k->s, samples * sizeof *v);
        qsort(v, samples, sizeof *v, cmp_double);
        k->est = v[0];
        printf("%s,%s,%s,%zu,%s,%s,%.4f,%.4f,%zu,,\n", PN[c->prim], c->scenario, regime, c->n, c->aname, k->name, v[0], v[samples / 2], k->iters);
    }
}

/* ---------------------------------------------------------------- the matrix */

static size_t *sizes;
static int nsizes;
typedef struct { int ao, bo; char name[16]; } Align; /* -1: the object ends at a page end */
static Align al2[64], al1[64];
static int nal2, nal1;
static u8x *bufA, *bufB;
static size_t bufcap;

static size_t place(int off, size_t len) {
    if (off >= 0) {
        return (size_t)off;
    }

    return (4096 - len % 4096) % 4096;
}

static const char *spn_sets[] = {"a", " \t\r\n", "abcdefghijklmnop", "abcdefghijklmnopqrstuvwxyz012345",
                                 "abcdefghijklmnopqrstuvwx", "abcdefghijklmnopqrstuvwxyz0123456789ABCD"};
static const char *cspn_sets[] = {",", ",;=\t", "!\"#$%&'()*+,-./:", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ",
                                  "!\"#$%&'()*+,-./:;<=>?@[\\", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ 0123456"};
/* 24 and 40: the two paths between the pcmpistri sets and the long ones (a pair of pcmpistrm, then truffle) */
static const char *set_names[] = {"set_1", "set_4", "set_16", "set_32", "set_24", "set_40"};

static int scenarios(int p, const char **out) {
    switch (p) {
    case P_MEMMOVE: out[0] = "disjoint"; out[1] = "ovl_dst+1"; out[2] = "ovl_dst-1"; out[3] = "ovl_dst+half"; out[4] = "ovl_dst-half"; return 5;
    case P_MEMSET: out[0] = "byte"; out[1] = "zero"; return 2;
    case P_MEMCMP: case P_STRCMP: out[0] = "equal"; out[1] = "diff_first"; out[2] = "diff_last"; return 3;
    case P_STRSPN: case P_STRCSPN: for (int i = 0; i < 6; i++) out[i] = set_names[i]; return 6;
    case P_MEMCHR: out[0] = "absent"; out[1] = "last"; return 2;
    default: out[0] = "full"; return 1;
    }
}

static int two_pointer(int p, const char *sc) {
    switch (p) {
    case P_MEMCPY: case P_MEMCMP: case P_STRCPY: case P_STRCAT: case P_STRCMP: return 1;
    case P_MEMMOVE: return !strcmp(sc, "disjoint");
    default: return 0;
    }
}

/* Lay out the cell's objects in the buffers and fill them. */
static void setup(Cell *c, const Align *al) {
    size_t n = c->n;
    int p = c->prim;
    size_t lens = n + 1, lend = n + 1;

    if (p == P_STRCAT) {
        lend = 2 * n + 1;
    }

    if (p == P_MEMCPY || p == P_MEMMOVE || p == P_MEMSET || p == P_MEMCMP || p == P_MEMCHR) {
        lens = lend = n ? n : 1;
    }

    size_t ao = place(al->ao, lens), bo = place(al->bo, lend);
    c->ao = ao;
    c->bo = bo;
    snprintf(c->aname, sizeof c->aname, "%s", al->name);
    u8x *a = bufA + 4096 * 2, *b = bufB + 4096 * 2;
    c->s = a + ao;
    c->d = b + bo;

    switch (p) {
    case P_MEMMOVE:

        if (strcmp(c->scenario, "disjoint")) {
            long delta = !strcmp(c->scenario, "ovl_dst+1") ? 1 : !strcmp(c->scenario, "ovl_dst-1") ? -1
                       : !strcmp(c->scenario, "ovl_dst+half") ? (long)(n / 2 ? n / 2 : 1) : -(long)(n / 2 ? n / 2 : 1);
            c->s = a + ((n / 2 + 4095) & ~(size_t)4095) + ao; /* room for the negative shift */
            c->d = c->s + delta;
        }

        fill_pattern(c->s, n, 1);
        break;
    case P_MEMCPY: fill_pattern(c->s, n, 1); break;
    case P_MEMSET: c->c = !strcmp(c->scenario, "zero") ? 0 : 0x5a; break;
    case P_MEMCMP: case P_STRCMP:
        fill_pattern(c->s, n, 2);
        memcpy(c->d, c->s, n);

        if (p == P_STRCMP) {
            c->s[n] = c->d[n] = 0;
        }

        if (n && !strcmp(c->scenario, "diff_first")) {
            c->d[0] ^= 0x80;
        }

        if (n && !strcmp(c->scenario, "diff_last")) {
            c->d[n - 1] ^= 0x80;
        }

        break;
    case P_STRLEN: case P_STRCPY: fill_pattern(c->s, n, 3); c->s[n] = 0; break;
    case P_STRCAT:
        fill_pattern(c->s, n, 3);
        c->s[n] = 0;
        fill_pattern(c->d, n, 4);
        c->d[n] = 0;
        c->strcat_len = n;
        break;
    case P_STRSPN: case P_STRCSPN: {
        int si = 0;

        while (strcmp(set_names[si], c->scenario)) {
            si++;
        }

        const char *set = p == P_STRSPN ? spn_sets[si] : cspn_sets[si];
        c->set = set;
        size_t sl = strlen(set);

        for (size_t i = 0; i < n; i++) {
            c->s[i] = p == P_STRSPN ? (u8x)set[(i * 7) % sl] : (u8x)('a' + i % 26);
        }

        c->s[n] = p == P_STRSPN ? 'Z' : (u8x)set[0];
        c->s[n + 1] = 0;
        break;
    }

    case P_MEMCHR:
        fill_pattern(c->s, n, 5);
        c->c = 0xfe;

        if (n && !strcmp(c->scenario, "last")) {
            c->s[n - 1] = 0xfe;
        }

        break;
    }
}

static int want_prim[P_N];
static int want_mixed, want_libs, no_trace, want_hybrid;
static const char *kfilter, *kexclude;

static int kernel_wanted(const Kernel *k) {
    if (k->status != 1 || !want_prim[k->prim]) {
        return 0;
    }

    if (kexclude) {
        char buf[512];
        snprintf(buf, sizeof buf, "%s", kexclude);
        char *sv;

        for (char *t = strtok_r(buf, ",", &sv); t; t = strtok_r(0, ",", &sv)) {
            if (strstr(k->name, t)) {
                return 0;
            }
        }
    }

    if (!kfilter) {
        return 1;
    }

    char buf[512];
    snprintf(buf, sizeof buf, "%s", kfilter);
    char *sv;

    for (char *t = strtok_r(buf, ",", &sv); t; t = strtok_r(0, ",", &sv)) {
        if (strstr(k->name, t)) {
            return 1;
        }
    }

    return 0;
}

static void matrix(const char *regimes) {
    Kernel **ks = malloc(NK * sizeof *ks);

    for (int p = 0; p < P_N; p++) {
        if (!want_prim[p]) {
            continue;
        }

        int nk = 0;

        for (int i = 0; i < NK; i++) {
            if (K[i].prim == p && kernel_wanted(&K[i])) {
                ks[nk++] = &K[i];
            }
        }

        if (!nk) {
            continue;
        }

        const char *sc[8];
        int nsc = scenarios(p, sc);
        char rbuf[64];
        snprintf(rbuf, sizeof rbuf, "%s", regimes);
        char *sv;

        for (char *regime = strtok_r(rbuf, ",", &sv); regime; regime = strtok_r(0, ",", &sv)) {
            for (int j = 0; j < nsc; j++) {
                time_t t0 = time(0);

                for (int i = 0; i < nk; i++) {
                    ks[i]->est = 0;
                }

                int two = two_pointer(p, sc[j]);
                int ovl = p == P_MEMMOVE && !two;

                for (int si = 0; si < nsizes; si++) {
                    size_t n = sizes[si];

                    if (2 * n + 3 * 4096 > bufcap) {
                        continue;
                    }

                    const Align *list = two ? al2 : al1;
                    int nal = two ? nal2 : nal1;

                    for (int ai = 0; ai < nal; ai++) {
                        if (ovl && ai >= 3) {
                            break; /* overlap: the first three source offsets */
                        }

                        Cell c = {.prim = p, .scenario = sc[j], .n = n};
                        setup(&c, &list[ai]);
                        time_cell(ks, nk, &c, regime);
                    }

                    fflush(stdout);
                }

                fprintf(stderr, "%s %s %s: %d kernels, %ld s\n", PN[p], sc[j], regime, nk, (long)(time(0) - t0));
            }
        }
    }

    free(ks);
}

/* -------------------------------------------------------------- the traces */

typedef struct { uint8_t op, pad[3]; uint32_t n; uint64_t a, b; } Rec;
enum { W_MEMCPY, W_MEMMOVE, W_MEMMOVE_OVL, W_MEMSET, W_MEMCMP, W_STRLEN, W_STRCPY, W_STRCAT, W_STRCMP, W_STRSPN, W_STRCSPN, W_MEMCHR };

static inline __attribute__((always_inline)) uint64_t trace_body(const Kernel *k, const Rec *recs, uint64_t count, u8x *src, u8x *dst) {
    uint64_t sum = 0;

    for (uint64_t i = 0; i < count; i++) {
        const Rec *q = &recs[i];

        switch (q->op) {
        case W_MEMCPY: case W_MEMMOVE: sum += (uintptr_t)((f_cpy)k->fn)(dst + q->b, src + q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMMOVE_OVL: sum += (uintptr_t)((f_cpy)k->fn)(dst + q->b, dst + q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMSET: sum += (uintptr_t)((f_set)k->fn)(dst + q->b, (int)q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMCMP: sum += (uint64_t)(sgn(((f_cmp)k->fn)(src + q->a, src + q->b, q->n)) + 1); break;
        case W_STRLEN: sum += ((f_len)k->fn)((const char *)src + q->a); break;
        case W_STRCPY: sum += (uintptr_t)((f_scpy)k->fn)((char *)dst + q->b, (const char *)src + q->a) - (uintptr_t)(dst + q->b); break;
        case W_STRCAT: dst[q->b + q->n] = 0; sum += (uintptr_t)((f_scpy)k->fn)((char *)dst + q->b, (const char *)src + q->a) - (uintptr_t)(dst + q->b); break;
        case W_STRCMP: sum += (uint64_t)(sgn(((f_scmp)k->fn)((const char *)src + q->a, (const char *)src + q->b)) + 1); break;
        case W_STRSPN: case W_STRCSPN: sum += ((f_spn)k->fn)((const char *)src + q->a, (const char *)src + q->b); break;
        case W_MEMCHR: { const u8x *r = ((f_chr)k->fn)(src + q->a, (int)q->b, q->n); sum += r ? (uint64_t)(r - (src + q->a)) : q->n; break; }
        }
    }

    return sum;
}

/* Whole tables on the mixed trace: every function in the workload's
   proportions, as a program calls them. A clock license one function takes
   (AVX-512 on Skylake-SP) slows the others too, which the per-function
   traces cannot show. A table: each configuration of ours the host runs,
   glibc as its IFUNCs chose, and --table NAME=CONFIG:SLOT=FN,... */
typedef struct {
    char name[64];
    struct str_table t; /* public slots only for the other libraries; a null one: the library has no such function */
    int ours;
    struct str_table *tgt; /* as Kernel's */
    const struct str_table *base; /* --hybrid: the base revision's configuration, installed beside (its functions call through it) */
    int on; /* runs the current trace: has every function it calls */
    double s[64], ghz[64];
    size_t iters;
    uint64_t sum;
} Table;

static Table tables[64];

static void table_install(const Table *x) {
    *(x->tgt ? x->tgt : &str_table) = x->t;

    if (x->base) {
        *kl_base_table = *x->base;
    }
}

static int ntables;
static const char *table_specs[16];
static int ntable_specs;

/* --libs: whole libraries on the workload (the stand bench.c --workload had):
   each library's own function for every primitive, as its dispatch or alias
   resolves it here (FreeBSD's names are the variants its archlevel picks, LLVM
   libc the build for the host's level, asmlib its dispatchers); a function it
   lacks, or one validation rejected, leaves the traces calling it out. noop,
   functions that return at once, measures the stand's own cost per call: the
   summary subtracts it. */
static void *noop_cpy(void *d, const void *s, size_t n) {
    (void)s;
    (void)n;
    return d;
}

static void *noop_set(void *d, int c, size_t n) {
    (void)c;
    (void)n;
    return d;
}

static int noop_cmp(const void *a, const void *b, size_t n) {
    (void)a;
    (void)b;
    (void)n;
    return 0;
}

static size_t noop_len(const char *s) {
    (void)s;
    return 0;
}

static char *noop_scpy(char *d, const char *s) {
    (void)s;
    return d;
}

static int noop_scmp(const char *a, const char *b) {
    (void)a;
    (void)b;
    return 0;
}

static size_t noop_spn(const char *s, const char *set) {
    (void)s;
    (void)set;
    return 0;
}

static void *noop_chr(const void *s, int c, size_t n) {
    (void)s;
    (void)c;
    (void)n;
    return 0;
}

static void *lib_fn(const char *prefix, int p) {
    char want[2][96];
    snprintf(want[0], sizeof want[0], "%s%s", prefix, PN[p]);
    snprintf(want[1], sizeof want[1], "%s__%s", prefix, PN[p]); /* FreeBSD's __strcspn, __memchr */

    for (unsigned i = 0; i < kl_ndefs; i++) {
        if (kl_defs[i].prim == p && (!strcmp(kl_defs[i].name, want[0]) || !strcmp(kl_defs[i].name, want[1]))) {
            for (int j = 0; j < NK; j++) {
                if (K[j].prim == p && K[j].fn == kl_defs[i].fn) {
                    return K[j].status == 1 ? kl_defs[i].fn : 0;
                }
            }
        }
    }

    return 0;
}

static void library(const char *name, const char *prefix, int noop) {
    Table *x = &tables[ntables++];
    snprintf(x->name, sizeof x->name, "%s", name);

    if (noop) {
        void *const f[] = {(void *)noop_cpy, (void *)noop_cpy, (void *)noop_set, (void *)noop_cmp, (void *)noop_len, (void *)noop_scpy,
                           (void *)noop_scpy, (void *)noop_scmp, (void *)noop_spn, (void *)noop_spn, (void *)noop_chr};
        memcpy(&x->t, f, sizeof f);
        return;
    }

    for (int p = 0; p < P_N; p++) {
        ((void **)&x->t)[p] = lib_fn(prefix, p);
    }
}

static void build_tables(void) {
    for (unsigned c = 0; c < str_nconfigs; c++) {
        unsigned n = str_configs[c].needs;
        unsigned needs = (n & STR_SSE42 ? F_SSE42 : 0) | (n & STR_AVX2 ? F_AVX2 : 0) | (n & STR_AVX512 ? F_AVX512 : 0);

        if (needs & ~host) {
            continue;
        }

        Table *x = &tables[ntables++];
        snprintf(x->name, sizeof x->name, "ours/%s", str_configs[c].name);
        x->t = str_configs[c].table;
        x->ours = 1;
    }

    for (unsigned c = 0; c < kl_base_nconfigs; c++) {
        unsigned n = kl_base_configs[c].needs;

        if (((n & STR_SSE42 ? F_SSE42 : 0) | (n & STR_AVX2 ? F_AVX2 : 0) | (n & STR_AVX512 ? F_AVX512 : 0)) & ~host) {
            continue;
        }

        Table *x = &tables[ntables++];
        snprintf(x->name, sizeof x->name, "base/%s", kl_base_configs[c].name);
        x->t = kl_base_configs[c].table;
        x->ours = 1;
        x->tgt = kl_base_table;

        /* --hybrid: the host's configuration of ours with one public function the
           base revision's, for each: which of them a difference of the two on
           the mixed trace comes from */

        for (int p = 0; want_hybrid && strcmp(kl_base_configs[c].name, base_config->name) == 0 && p < P_N; p++) {
            Table *h = &tables[ntables++];
            CHECK(ntables <= (int)(sizeof tables / sizeof *tables));
            snprintf(h->name, sizeof h->name, "hyb/%s", PN[p]);
            h->t = base_config->table;
            ((void **)&h->t)[p] = ((void *const *)&kl_base_configs[c].table)[p];
            h->ours = 1;
            h->base = &kl_base_configs[c].table;
        }
    }

    Table *g = &tables[ntables++];
    snprintf(g->name, sizeof g->name, "glibc/dispatch");
    void *const pub[] = {resolved_glibc((void *)memcpy), resolved_glibc((void *)memmove), resolved_glibc((void *)memset), resolved_glibc((void *)memcmp),
                         resolved_glibc((void *)strlen), resolved_glibc((void *)strcpy), resolved_glibc((void *)strcat), resolved_glibc((void *)strcmp),
                         resolved_glibc((void *)strspn), resolved_glibc((void *)strcspn), resolved_glibc((void *)memchr)};
    memcpy(&g->t, pub, sizeof pub);

    /* --hybrid: the host's configuration of ours with one public function
       glibc's, for each: which of ours lose to glibc's in the mix */

    for (int p = 0; want_hybrid && p < P_N; p++) {
        Table *h = &tables[ntables++];
        CHECK(ntables <= (int)(sizeof tables / sizeof *tables));
        snprintf(h->name, sizeof h->name, "hybg/%s", PN[p]);
        h->t = base_config->table;
        ((void **)&h->t)[p] = pub[p];
        h->ours = 1;
    }

    if (want_libs) {
        library("freebsd", "fbsd_", 0);
        library("musl", "musl_", 0);
        library("llvm", host & F_AVX512 ? "llvm_v4_" : host & F_AVX2 ? "llvm_v3_" : "llvm_v2_", 0);
        library("asmlib", "asm_A_", 0);
        library("noop", 0, 1);
    }

    for (int i = 0; i < ntable_specs; i++) {
        char buf[512], *sv;
        snprintf(buf, sizeof buf, "%s", table_specs[i]);
        char *eq = strchr(buf, '='), *colon = eq ? strchr(eq, ':') : 0;
        CHECK(eq && colon); /* NAME=CONFIG:SLOT=FN,... */
        *eq = *colon = 0;
        Table *x = &tables[ntables++];
        snprintf(x->name, sizeof x->name, "ours/%s", buf);
        x->ours = 1;
        int found = 0;

        for (unsigned c = 0; c < str_nconfigs; c++) {
            if (!strcmp(str_configs[c].name, eq + 1)) {
                x->t = str_configs[c].table;
                found = 1;
            }
        }

        CHECK(found);

        for (char *a = strtok_r(colon + 1, ",", &sv); a; a = strtok_r(0, ",", &sv)) {
            char *e = strchr(a, '=');
            CHECK(e);
            *e = 0;
            int slot = -1, done = 0;

            for (int s = 0; s < NSLOTS; s++) {
                if (!strcmp(SLOTN[s], a)) {
                    slot = s;
                }
            }

            CHECK(slot >= 0);

            for (unsigned j = 0; j < kl_nslots; j++) {
                if (kl_slots[j].slot == slot && !strcmp(kl_slots[j].name, e + 1)) {
                    ((void **)&x->t)[slot] = kl_slots[j].fn;
                    done = 1;
                }
            }

            /* a public slot: any kernel of the primitive (dev/lab/kernels, the
               other libraries), by its symbol */

            for (unsigned j = 0; !done && slot < P_N && j < kl_ndefs; j++) {
                if (kl_defs[j].prim == slot && !strcmp(kl_defs[j].name, e + 1)) {
                    ((void **)&x->t)[slot] = kl_defs[j].fn;
                    done = 1;
                }
            }

            if (!done) {
                fprintf(stderr, "kbench: --table: no %s implementation %s\n", a, e + 1);
                exit(2);
            }
        }
    }
}

static inline __attribute__((always_inline)) uint64_t mixed_body(const struct str_table *t, const Rec *recs, uint64_t count, u8x *src, u8x *dst) {
    uint64_t sum = 0;

    for (uint64_t i = 0; i < count; i++) {
        const Rec *q = &recs[i];

        switch (q->op) {
        case W_MEMCPY: sum += (uintptr_t)t->memcpy(dst + q->b, src + q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMMOVE: sum += (uintptr_t)t->memmove(dst + q->b, src + q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMMOVE_OVL: sum += (uintptr_t)t->memmove(dst + q->b, dst + q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMSET: sum += (uintptr_t)t->memset(dst + q->b, (int)q->a, q->n) - (uintptr_t)(dst + q->b); break;
        case W_MEMCMP: sum += (uint64_t)(sgn(t->memcmp(src + q->a, src + q->b, q->n)) + 1); break;
        case W_STRLEN: sum += t->strlen((const char *)src + q->a); break;
        case W_STRCPY: sum += (uintptr_t)t->strcpy((char *)dst + q->b, (const char *)src + q->a) - (uintptr_t)(dst + q->b); break;
        case W_STRCAT: dst[q->b + q->n] = 0; sum += (uintptr_t)t->strcat((char *)dst + q->b, (const char *)src + q->a) - (uintptr_t)(dst + q->b); break;
        case W_STRCMP: sum += (uint64_t)(sgn(t->strcmp((const char *)src + q->a, (const char *)src + q->b)) + 1); break;
        case W_STRSPN: sum += t->strspn((const char *)src + q->a, (const char *)src + q->b); break;
        case W_STRCSPN: sum += t->strcspn((const char *)src + q->a, (const char *)src + q->b); break;
        case W_MEMCHR: { const u8x *r = (t->memchr)(src + q->a, (int)q->b, q->n); /* glibc's string.h has a memchr macro */ sum += r ? (uint64_t)(r - (src + q->a)) : q->n; break; }
        }
    }

    return sum;
}

/* The stand: the loops over a trace, four copies 64-aligned with 0, 16, 32 or
   48 bytes of nops (run once a replay) before the loop, --stand K picks one.
   The loop's place against the 32- and 64-byte boundaries is the stand's own
   share of a measurement: its compare-and-branch pairs and the indirect call
   are cut by a 32-byte boundary or not (Skylake's JCC erratum: such a branch
   is not kept in the decoded-uop cache), and its uops share the frontend with
   the function's. */
#define STAND(K)                                                                                                                                    \
    __attribute__((noinline, aligned(64))) static uint64_t trace_run_##K(const Kernel *k, const Rec *recs, uint64_t count, u8x *src, u8x *dst) {  \
        __asm__ volatile(".skip " #K " * 16, 0x90");                                                                                            \
        return trace_body(k, recs, count, src, dst);                                                                                            \
    }                                                                                                                                           \
    __attribute__((noinline, aligned(64))) static uint64_t mixed_run_##K(const struct str_table *t, const Rec *recs, uint64_t count, u8x *src,      \
                                                                       u8x *dst) {                                                               \
        __asm__ volatile(".skip " #K " * 16, 0x90");                                                                                            \
        return mixed_body(t, recs, count, src, dst);                                                                                            \
    }

STAND(0)
STAND(1)
STAND(2)
STAND(3)

static int stand;
static uint64_t (*const trace_runs[])(const Kernel *, const Rec *, uint64_t, u8x *, u8x *) = {trace_run_0, trace_run_1, trace_run_2, trace_run_3};
static uint64_t (*const mixed_runs[])(const struct str_table *, const Rec *, uint64_t, u8x *, u8x *) = {mixed_run_0, mixed_run_1, mixed_run_2, mixed_run_3};

static uint64_t trace_run(const Kernel *k, const Rec *recs, uint64_t count, u8x *src, u8x *dst) {
    return trace_runs[stand](k, recs, count, src, dst);
}

static uint64_t mixed_run(const struct str_table *t, const Rec *recs, uint64_t count, u8x *src, u8x *dst) {
    return mixed_runs[stand](t, recs, count, src, dst);
}

/* The functions a record calls, as public slots of a table. */
static int rec_slot(int op) {
    static const int slot[] = {[W_MEMCPY] = P_MEMCPY, [W_MEMMOVE] = P_MEMMOVE, [W_MEMMOVE_OVL] = P_MEMMOVE, [W_MEMSET] = P_MEMSET, [W_MEMCMP] = P_MEMCMP,
                               [W_STRLEN] = P_STRLEN, [W_STRCPY] = P_STRCPY, [W_STRCAT] = P_STRCAT, [W_STRCMP] = P_STRCMP, [W_STRSPN] = P_STRSPN,
                               [W_STRCSPN] = P_STRCSPN, [W_MEMCHR] = P_MEMCHR};
    return slot[op];
}

static void tables_trace(const char *name, const Rec *recs, uint64_t count, u8x *src, u8x *dst, const u8x *image, size_t dsz, int passes) {
    time_t t0 = time(0);
    unsigned used = 0;
    int ref = -1, non = 0;

    for (uint64_t i = 0; i < count; i++) {
        used |= 1u << rec_slot(recs[i].op);
    }

    for (int i = 0; i < ntables; i++) {
        Table *x = &tables[i];
        x->on = 1;

        for (int p = 0; p < P_N; p++) {
            if ((used >> p & 1) && !((void **)&x->t)[p]) {
                x->on = 0;
            }
        }

        if (!x->on) {
            continue;
        }

        non++;

        if (x->ours) {
            table_install(x);
        }

        memcpy(dst, image, dsz);
        uint64_t a = now();
        x->sum = mixed_run(&x->t, recs, count, src, dst);
        double est = (double)(now() - a) / (double)count;
        x->iters = est * count > 1e7 ? 1 : (size_t)(1e7 / (est * count)) + 1;

        if (!strcmp(x->name, "glibc/dispatch")) {
            ref = i;
        }
    }

    for (int i = 0; i < ntables; i++) {
        if (tables[i].on && ref >= 0 && strcmp(tables[i].name, "noop") && tables[i].sum != tables[ref].sum) {
            fprintf(stderr, "TABLES %s: %s checksum %" PRIu64 " != glibc %" PRIu64 "\n", name, tables[i].name, tables[i].sum, tables[ref].sum);
        }
    }

    int order[64], np = passes < 63 ? passes : 63;

    for (int r = 0; r < np; r++) {
        for (int i = 0; i < ntables; i++) {
            order[i] = i;
        }

        for (int i = ntables - 1; i > 0; i--) {
            int j = (int)(rnd() % (unsigned)(i + 1)), x = order[i];
            order[i] = order[j];
            order[j] = x;
        }

        for (int i = 0; i < ntables; i++) {
            Table *x = &tables[order[i]];

            if (!x->on) {
                continue;
            }

            if (x->ours) {
                table_install(x);
            }

            memcpy(dst, image, dsz);
            uint64_t a = now(), ap = msr_fd >= 0 ? rdmsr(0xe8) : 0, mp = msr_fd >= 0 ? rdmsr(0xe7) : 0;

            for (size_t it = 0; it < x->iters; it++) {
                vsink += mixed_run(&x->t, recs, count, src, dst);
            }

            uint64_t b = now();
            x->s[r] = (double)(b - a) / (double)(count * x->iters);
            x->ghz[r] = msr_fd >= 0 ? tsc_ghz * (double)(rdmsr(0xe8) - ap) / (double)(rdmsr(0xe7) - mp + 1) : 0;
        }
    }

    for (int i = 0; i < ntables; i++) {
        if (!tables[i].on) {
            continue;
        }

        double v[64], g[64];
        memcpy(v, tables[i].s, np * sizeof *v);
        memcpy(g, tables[i].ghz, np * sizeof *g);
        qsort(v, np, sizeof *v, cmp_double);
        qsort(g, np, sizeof *g, cmp_double);
        char pm[256] = "";

        if (pmu_n) {
            Table *x = &tables[i];

            if (x->ours) {
                table_install(x);
            }

            memcpy(dst, image, dsz);
            pmu_start();

            for (size_t it = 0; it < x->iters; it++) {
                vsink += mixed_run(&x->t, recs, count, src, dst);
            }

            pmu_stop(pm, sizeof pm, (double)(count * x->iters));
        }

        printf("lib,trace:%s,trace,%" PRIu64 ",-,%s,%.4f,%.4f,%zu,%.3f,%s\n", name, count, tables[i].name, v[0], v[np / 2], tables[i].iters, g[np / 2], pm);
    }

    fflush(stdout);
    fprintf(stderr, "trace %s: %d tables, %ld s\n", name, non, (long)(time(0) - t0));
}

/* The primitive a trace calls, by its name: NAME, or NAME/... for a part of
   NAME's trace (split.py's buckets of sizes). */
static int trace_prim(const char *full) {
    char name[64];
    snprintf(name, sizeof name, "%s", full);
    char *slash = strchr(name, '/');

    if (slash) {
        *slash = 0;
    }

    static const struct { const char *t; int p; } map[] = {{"memcpy", P_MEMCPY}, {"memmove", P_MEMMOVE}, {"memset", P_MEMSET}, {"memcmp", P_MEMCMP},
        {"strlen", P_STRLEN}, {"strcpy", P_STRCPY}, {"strcat", P_STRCAT}, {"strcmp", P_STRCMP}, {"strspn", P_STRSPN}, {"strcspn", P_STRCSPN},
        {"memchr", P_MEMCHR}, {"strspn_long", P_STRSPN}, {"strcspn_long", P_STRCSPN}};

    for (size_t i = 0; i < sizeof map / sizeof *map; i++) {
        if (!strcmp(map[i].t, name)) {
            return map[i].p;
        }
    }

    return -1;
}

static void traces(const char *path, int passes) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    char magic[4];
    uint32_t version, ntraces;
    uint64_t ssz, dsz;
    CHECK(fread(magic, 1, 4, f) == 4 && !memcmp(magic, "STRW", 4));
    CHECK(fread(&version, 4, 1, f) == 1 && fread(&ssz, 8, 1, f) == 1 && fread(&dsz, 8, 1, f) == 1 && fread(&ntraces, 4, 1, f) == 1 && version == 1);
    u8x *src, *dst, *image;
    CHECK(!posix_memalign((void **)&src, 4096, ssz) && !posix_memalign((void **)&dst, 4096, dsz) && !posix_memalign((void **)&image, 4096, dsz));
    CHECK(fread(src, 1, ssz, f) == ssz && fread(image, 1, dsz, f) == dsz);
    Kernel **ks = malloc(NK * sizeof *ks);

    for (uint32_t t = 0; t < ntraces; t++) {
        uint32_t nl;
        char name[64];
        uint64_t count;
        CHECK(fread(&nl, 4, 1, f) == 1 && nl < 64 && fread(name, 1, nl, f) == nl);
        name[nl] = 0;
        CHECK(fread(&count, 8, 1, f) == 1);
        Rec *recs = malloc(count * sizeof *recs);
        CHECK(recs && fread(recs, sizeof *recs, count, f) == count);
        int p = trace_prim(name);

        if (count && (want_libs || (want_mixed && !strcmp(name, "mixed")))) {
            tables_trace(name, recs, count, src, dst, image, dsz, passes);
        }

        if (p < 0 || !want_prim[p] || !count) {
            free(recs);
            continue;
        }

        int nk = 0;

        for (int i = 0; i < NK; i++) {
            if (K[i].prim == p && kernel_wanted(&K[i])) {
                ks[nk++] = &K[i];
            }
        }

        /* the reference checksum: glibc's dispatched implementation */
        uint64_t want = 0;
        int have_want = 0;
        time_t t0 = time(0);

        for (int i = 0; i < nk; i++) {
            Kernel *k = ks[i];
            install(k);
            memcpy(dst, image, dsz);
            uint64_t a = now(), sum = trace_run(k, recs, count, src, dst), b = now();
            k->est = (double)(b - a) / (double)count;
            k->iters = k->est * count > 1e7 ? 1 : (size_t)(1e7 / (k->est * count)) + 1; /* about 10 ms a pass */

            if (!have_want && !strcmp(k->origin, "glibc")) {
                want = sum, have_want = 1;
            }

            k->sum = sum;
        }

        for (int i = 0; i < nk; i++) {
            if (have_want && ks[i]->sum != want) {
                fprintf(stderr, "TRACE %s: %s checksum %" PRIu64 " != glibc %" PRIu64 "\n", name, ks[i]->name, ks[i]->sum, want);
            }
        }

        int order[1024];

        for (int r = 0; r < passes && r < 63; r++) {
            for (int i = 0; i < nk; i++) {
                order[i] = i;
            }

            for (int i = nk - 1; i > 0; i--) {
                int j = (int)(rnd() % (unsigned)(i + 1)), x = order[i];
                order[i] = order[j];
                order[j] = x;
            }

            for (int i = 0; i < nk; i++) {
                Kernel *k = ks[order[i]];
                install(k);
                memcpy(dst, image, dsz);
                uint64_t a = now(), ap = msr_fd >= 0 ? rdmsr(0xe8) : 0, mp = msr_fd >= 0 ? rdmsr(0xe7) : 0;

                for (size_t it = 0; it < k->iters; it++) {
                    vsink += trace_run(k, recs, count, src, dst);
                }

                uint64_t b = now();
                k->s[r] = (double)(b - a) / (double)(count * k->iters);
                k->ghz[r] = msr_fd >= 0 ? tsc_ghz * (double)(rdmsr(0xe8) - ap) / (double)(rdmsr(0xe7) - mp + 1) : 0;
            }
        }

        for (int i = 0; i < nk; i++) {
            double v[64];
            int np = passes < 63 ? passes : 63;
            memcpy(v, ks[i]->s, np * sizeof *v);
            qsort(v, np, sizeof *v, cmp_double);
            double g[64];
            memcpy(g, ks[i]->ghz, np * sizeof *g);
            qsort(g, np, sizeof *g, cmp_double);
            char pm[256] = "";

            if (pmu_n) {
                Kernel *k = ks[i];
                install(k);
                memcpy(dst, image, dsz);
                pmu_start();

                for (size_t it = 0; it < k->iters; it++) {
                    vsink += trace_run(k, recs, count, src, dst);
                }

                pmu_stop(pm, sizeof pm, (double)(count * k->iters));
            }

            printf("%s,trace:%s,trace,%" PRIu64 ",-,%s,%.4f,%.4f,%zu,%.3f,%s\n", PN[p], name, count, ks[i]->name, v[0], v[np / 2], ks[i]->iters, g[np / 2], pm);
        }

        fflush(stdout);
        fprintf(stderr, "trace %s: %d kernels, %ld s\n", name, nk, (long)(time(0) - t0));
        free(recs);
    }

    free(ks);
}

/* -------------------------------------------------------------------- main */

static void parse_sizes(const char *spec) {
    static const size_t quick[] = {0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 256, 384, 512,
                                   1024, 2048, 4096, 8192, 65536, 1048576};
    sizes = calloc(1024, sizeof *sizes);
    nsizes = 0;

    if (!strcmp(spec, "quick")) {
        for (size_t i = 0; i < sizeof quick / sizeof *quick; i++) {
            sizes[nsizes++] = quick[i];
        }

        return;
    }

    if (!strcmp(spec, "full")) {
        for (size_t n = 0; n <= 64; n++) {
            sizes[nsizes++] = n;
        }

        static const size_t more[] = {72, 80, 88, 96, 104, 112, 120, 127, 128, 129, 144, 160, 176, 192, 208, 224, 240, 255, 256, 257,
                                      288, 320, 352, 384, 416, 448, 480, 511, 512, 513, 640, 768, 896, 1023, 1024, 1025, 1536, 2048, 3072,
                                      4096, 6144, 8192, 12288, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};

        for (size_t i = 0; i < sizeof more / sizeof *more; i++) {
            sizes[nsizes++] = more[i];
        }

        return;
    }

    for (const char *s = spec; *s;) {
        char *e;
        sizes[nsizes++] = strtoul(s, &e, 10);
        s = *e == ',' ? e + 1 : e;
    }
}

static void parse_aligns(const char *spec, Align *out, int *n, int two) {
    *n = 0;
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", spec);
    char *sv;

    for (char *t = strtok_r(buf, ",", &sv); t; t = strtok_r(0, ",", &sv)) {
        Align a = {0, 0, ""};
        snprintf(a.name, sizeof a.name, "%s", t);
        char *colon = strchr(t, ':');

        if (colon) {
            *colon = 0;
        }

        a.ao = !strcmp(t, "pe") ? -1 : atoi(t);

        if (two) {
            a.bo = colon ? (!strcmp(colon + 1, "pe") ? -1 : atoi(colon + 1)) : a.ao;
        }

        if (!two) {
            snprintf(a.name, sizeof a.name, "%s", t);
        }

        out[(*n)++] = a;
    }
}

static void meta(const char *path) {
    FILE *f = fopen(path, "w");
    CHECK(f);
    fprintf(f, "{\n \"brand\": \"%s\",\n \"features\": %u,\n \"clock_ghz\": [%.3f, %.3f],\n \"ours_config\": \"%s\",\n \"base_rev\": \"%s\",\n \"stand\": %d,\n \"glibc\": {",
            brand, host, ghz_start, ghz_end, base_config->name, kl_base_rev, stand);
    static void *const gl[] = {(void *)memcpy, (void *)memmove, (void *)memset, (void *)memcmp, (void *)strlen, (void *)strcpy,
                               (void *)strcat, (void *)strcmp, (void *)strspn, (void *)strcspn, (void *)memchr};

    for (int p = 0; p < P_N; p++) {
        fprintf(f, "%s\"%s\": \"%s\"", p ? ", " : "", PN[p], glibc_variant(resolved_glibc(gl[p])));
    }

    fprintf(f, "},\n \"kernels\": [\n");

    for (int i = 0; i < NK; i++) {
        Kernel *k = &K[i];
        char why[200];
        size_t w = 0;

        for (const char *q = k->why; *q && w + 2 < sizeof why; q++) {
            if (*q == '"' || *q == '\\') {
                why[w++] = '\\';
            }

            why[w++] = (*q >= 32 && *q < 127) ? *q : '?';
        }

        why[w] = 0;
        fprintf(f, "  {\"name\": \"%s\", \"origin\": \"%s\", \"prim\": \"%s\", \"needs\": %u, \"status\": %d, \"why\": \"%s\"}%s\n", k->name, k->origin,
                PN[k->prim], k->needs, k->status, why, i + 1 < NK ? "," : "");
    }

    fprintf(f, " ]\n}\n");
    fclose(f);
}

static void usage(void) {
    fprintf(stderr, "kbench [--cpu N] [--samples S] [--target-us T] [--prims list] [--kernels substrings] [--exclude substrings]\n"
                    "       [--sizes quick|full|list] [--aligns2 list] [--aligns1 list] [--regimes thr,lat] [--trace FILE] [--passes P]\n"
                    "       [--jobs J] [--meta FILE] [--list] [--validate-only] [--no-matrix] [--no-trace] [--mixed|--libs [--table NAME=CONFIG:SLOT=FN,...]]\n"
                    "       [--pmu cycles,instructions,branches,branch-misses,rHEX] [--stand 0..3]\n"
                    "       [--hybrid: ours with each public function glibc's, and the base revision's if linked]\n");
    exit(2);
}

int main(int argc, char **argv) {
    int cpu = -1, jobs = 4, list = 0, validate_only = 0, no_matrix = 0, passes = 7;
    const char *prims = 0, *sizespec = "full", *regimes = "thr", *trace = 0, *metafile = 0;
    const char *a2 = "0:0,1:1,0:1,1:0,3:3,7:7,8:8,15:15,16:16,17:17,31:31,32:32,33:33,48:48,63:63,0:32,32:0,1:63,63:1,0:63,pe:pe,pe:0";
    const char *a1 = "0,1,3,7,8,15,16,17,31,32,33,48,63,pe";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : 0;
#define OPT(s) (!strcmp(a, s) && v && (++i, 1))
        if (OPT("--cpu")) {
            cpu = atoi(v);
        } else if (OPT("--samples")) {
            samples = atoi(v);
        } else if (OPT("--target-us")) {
            target_ns = atof(v) * 1000;
        } else if (OPT("--prims")) {
            prims = v;
        } else if (OPT("--kernels")) {
            kfilter = v;
        } else if (OPT("--exclude")) {
            kexclude = v;
        } else if (OPT("--sizes")) {
            sizespec = v;
        } else if (OPT("--aligns2")) {
            a2 = v;
        } else if (OPT("--aligns1")) {
            a1 = v;
        } else if (OPT("--regimes")) {
            regimes = v;
        } else if (OPT("--trace")) {
            trace = v;
        } else if (OPT("--table")) {
            CHECK(ntable_specs < 16);
            table_specs[ntable_specs++] = v;
        } else if (OPT("--stand")) {
            stand = atoi(v) & 3;
        } else if (OPT("--pmu")) {
            pmu_open(argv[i]);
        } else if (!strcmp(a, "--no-trace")) {
            no_trace = 1;
        } else if (!strcmp(a, "--mixed")) {
            want_mixed = 1;
        } else if (!strcmp(a, "--hybrid")) {
            want_hybrid = 1;
        } else if (!strcmp(a, "--libs")) {
            want_libs = 1;
        } else if (OPT("--passes")) {
            passes = atoi(v);
        } else if (OPT("--jobs")) {
            jobs = atoi(v);
        } else if (OPT("--meta")) {
            metafile = v;
        } else if (!strcmp(a, "--list")) {
            list = 1;
        } else if (!strcmp(a, "--validate-only")) {
            validate_only = 1;
        } else if (!strcmp(a, "--no-matrix")) {
            no_matrix = 1;
        } else {
            usage();
        }
#undef OPT
    }

    if (samples < 1) {
        samples = 1;
    }

    if (samples > 63) {
        samples = 63;
    }

    host = detect();

    for (int p = 0; p < P_N; p++) {
        want_prim[p] = !prims || strstr(prims, PN[p]) != 0;
    }

    if (prims) /* exact names: "memcpy" must not select memmove's rows by substring luck, nor strspn strcspn's */

        for (int p = 0; p < P_N; p++) {
            char buf[256];
            snprintf(buf, sizeof buf, "%s", prims);
            want_prim[p] = 0;
            char *sv;

            for (char *t = strtok_r(buf, ",", &sv); t; t = strtok_r(0, ",", &sv)) {
                if (!strcmp(t, PN[p])) {
                    want_prim[p] = 1;
                }
            }
        }

    build_kernels();
    /* asmlib's entries read the cache limits its dispatcher sets on the first call */
    {
        static u8x x[64], y[64];

        for (int i = 0; i < NK; i++) {
            if (!strcmp(K[i].origin, "asmlib") && strstr(K[i].name, "asm_A_") && !(K[i].needs & ~host)) {
                if (K[i].prim == P_MEMCPY || K[i].prim == P_MEMMOVE) {
                    ((f_cpy)K[i].fn)(x, y, 8);
                }

                if (K[i].prim == P_MEMSET) {
                    ((f_set)K[i].fn)(x, 0, 8);
                }

                if (K[i].prim == P_MEMCMP) {
                    ((f_cmp)K[i].fn)(x, y, 8);
                }
            }
        }
    }

    fprintf(stderr, "host: %s features %#x; ours: %s; %d kernels\n", brand, host, base_config->name, NK);

    if (list) {
        for (int i = 0; i < NK; i++) {
            printf("%-8s %-60s needs %#x%s\n", PN[K[i].prim], K[i].name, K[i].needs, (K[i].needs & ~host) ? " (not here)" : "");
        }

        return 0;
    }

    validate_all(jobs);

    if (metafile) {
        meta(metafile);
    }

    if (validate_only) {
        return 0;
    }

    if (cpu >= 0) {
        CHECK(!pin(cpu));
    }

    msr_open(cpu >= 0 ? cpu : 0);
    ghz_start = clock_ghz();
    fprintf(stderr, "clock: %.3f GHz%s\n", ghz_start, msr_fd >= 0 ? "; APERF/MPERF readable: traces report the clock of each pass" : "");
    parse_sizes(sizespec);
    parse_aligns(a2, al2, &nal2, 1);
    parse_aligns(a1, al1, &nal1, 0);
    size_t maxn = 0;

    for (int i = 0; i < nsizes; i++) {
        maxn = sizes[i] > maxn ? sizes[i] : maxn;
    }

    bufcap = 2 * maxn + 6 * 4096;
    CHECK(!posix_memalign((void **)&bufA, 4096, bufcap) && !posix_memalign((void **)&bufB, 4096, bufcap));
    memset(bufA, 1, bufcap);
    memset(bufB, 1, bufcap);
    printf("prim,scenario,regime,n,align,kernel,min_ns,med_ns,iters,ghz,pmu\n");

    if (!no_matrix) {
        matrix(regimes);
    }

    if (trace && !no_trace) {
        if (want_mixed || want_libs) {
            build_tables();
        }

        traces(trace, passes);
    }

    ghz_end = clock_ghz();
    fprintf(stderr, "clock: %.3f GHz at the end\n", ghz_end);

    if (metafile) {
        meta(metafile);
    }

    return 0;
}
