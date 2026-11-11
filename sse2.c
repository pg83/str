/*
 * sse2.c: every primitive for baseline x86-64 (SSE2 only), the set str_table
 * starts as. Compile with -march=x86-64. Also home of what the other files
 * share: table-driven strspn/strcspn and single-byte scans.
 *
 * Semantics are the plain C library ones: a string is a byte sequence up to
 * the first NUL, comparisons use unsigned char, no locale, no Unicode.
 * strcpy/strcat require enough destination space and nonoverlapping strings.
 */
#include "common.h"

/* ------------------------------------------------------------ NUL scans */

/* a is 16-aligned, s <= a, no NUL in [s, a). */
size_t strlen_tail_sse2(const u8 *a, const u8 *s) {
    for (;;) {
        unsigned m = zero_mask16(load16a(a));

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }

        a += 16;
    }
}

size_t strlen_sse2(const char *str) {
    const u8 *s = (const u8 *)str, *next = 0;
    size_t n = strlen_prefix64(s, &next);
    return n != SIZE_MAX ? n : str_strlen_tail(next, s);
}

/* First index i with s[i] == c or s[i] == 0, aligned 16-byte blocks. */
size_t find_byte_or_nul_sse2(const u8 *s, u8 c) {
    __m128i needle = splat16(c);
    const u8 *a = align_down(s, 16);
    __m128i v = load16a(a);
    unsigned m = (eq_mask16(v, needle) | zero_mask16(v)) >> ((uintptr_t)s & 15);

    if (m) {
        return ctz(m);
    }

    for (;;) {
        a += 16;
        v = load16a(a);
        m = eq_mask16(v, needle) | zero_mask16(v);

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }
    }
}

/* First index i with s[i] != c (a NUL counts as different). */
size_t skip_byte_sse2(const u8 *s, u8 c) {
    __m128i needle = splat16(c);
    const u8 *a = align_down(s, 16);
    unsigned m = (eq_mask16(load16a(a), needle) ^ 0xffffu) >> ((uintptr_t)s & 15);

    if (m) {
        return ctz(m);
    }

    for (;;) {
        a += 16;
        m = eq_mask16(load16a(a), needle) ^ 0xffffu;

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }
    }
}

/* ------------------------------------------------------------------ copy */

/* Forward loop, n > 128, valid when d <= s or the ranges do not overlap.
   The head and tail vectors are loaded first and stored last, so the aligned
   loop may clobber source bytes only after they were consumed. */
static void *copy_forward(u8 *d, const u8 *s, size_t n) {
    __m128i head = load16(s), tail = load16(s + n - 16);
    size_t skew = 16 - ((uintptr_t)d & 15);
    u8 *dp = d + skew;
    const u8 *sp = s + skew;
    size_t left = n - skew;

    while (left >= 64) {
        __m128i x0 = load16(sp), x1 = load16(sp + 16), x2 = load16(sp + 32), x3 = load16(sp + 48);
        store16a(dp, x0);
        store16a(dp + 16, x1);
        store16a(dp + 32, x2);
        store16a(dp + 48, x3);
        dp += 64;
        sp += 64;
        left -= 64;
    }

    while (left >= 16) {
        store16a(dp, load16(sp));
        dp += 16;
        sp += 16;
        left -= 16;
    }

    store16(d + n - 16, tail);
    store16(d, head);
    RETURN_OPAQUE(d);
}

/* Backward loop, n > 128, for s < d < s + n: the aligned loop runs down from
   the end; head and tail are stored last. */
static void *copy_backward(u8 *d, const u8 *s, size_t n) {
    __m128i head = load16(s), tail = load16(s + n - 16);
    size_t off = n - ((uintptr_t)(d + n) & 15);

    while (off >= 64) {
        off -= 64;
        __m128i x0 = load16(s + off), x1 = load16(s + off + 16), x2 = load16(s + off + 32), x3 = load16(s + off + 48);
        store16a(d + off, x0);
        store16a(d + off + 16, x1);
        store16a(d + off + 32, x2);
        store16a(d + off + 48, x3);
    }

    while (off >= 16) {
        off -= 16;
        store16a(d + off, load16(s + off));
    }

    store16(d, head);
    store16(d + n - 16, tail);
    RETURN_OPAQUE(d);
}

/* n > 64. */
void *copy_large_sse2(u8 *d, const u8 *s, size_t n) {
    if (n <= 128) {
        __m128i x0 = load16(s), x1 = load16(s + 16), x2 = load16(s + 32), x3 = load16(s + 48);
        __m128i y0 = load16(s + n - 64), y1 = load16(s + n - 48), y2 = load16(s + n - 32), y3 = load16(s + n - 16);
        store16(d, x0);
        store16(d + 16, x1);
        store16(d + 32, x2);
        store16(d + 48, x3);
        store16(d + n - 64, y0);
        store16(d + n - 48, y1);
        store16(d + n - 32, y2);
        store16(d + n - 16, y3);
        RETURN_OPAQUE(d);
    }

    if (overlap_backward(d, s, n)) {
        return copy_backward(d, s, n);
    }

    if (n >= REP_THRESHOLD) {
        return str_copy_huge(d, s, n);
    }

    return copy_forward(d, s, n);
}

/* ---------------------------------------------------- huge copy and fill */

/* n >= REP_THRESHOLD, no backward overlap: the vector loops of this unit,
   and the string instructions, which dispatch.c puts in the slots of the
   hosts whose rep movsb/stosb are fast (ERMS). */
void *copy_huge_sse2(u8 *d, const u8 *s, size_t n) {
    return copy_forward(d, s, n);
}

void *copy_huge_rep(u8 *d, const u8 *s, size_t n) {
    rep_movsb(d, s, n);
    RETURN_OPAQUE(d);
}

void *fill_huge_rep(u8 *d, int c, size_t n) {
    rep_stosb(d, c, n);
    RETURN_OPAQUE(d);
}

/* memcpy and memmove share one body: every short path is overlap-safe, and
   the large path picks the direction by overlap. */
static inline void *copy(void *dst, const void *src, size_t n) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;

    if (n <= 32) {
        copy_small(d, s, n);
    } else if (n <= 64) {
        copy_33_64(d, s, n);
    } else {
        return str_copy_large(d, s, n);
    }

    return dst;
}

void *memcpy_sse2(void *dst, const void *src, size_t n) {
    return copy(dst, src, n);
}

void *memmove_sse2(void *dst, const void *src, size_t n) {
    return copy(dst, src, n);
}

/* ------------------------------------------------------------------ fill */

/* The aligned loop for anything above 128 bytes. */
static inline void *fill_loop(u8 *d, __m128i v, size_t n) {
    store16(d, v);
    size_t skew = 16 - ((uintptr_t)d & 15);
    u8 *dp = d + skew;
    size_t left = n - skew;

    while (left >= 64) {
        store16a(dp, v);
        store16a(dp + 16, v);
        store16a(dp + 32, v);
        store16a(dp + 48, v);
        dp += 64;
        left -= 64;
    }

    while (left >= 16) {
        store16a(dp, v);
        dp += 16;
        left -= 16;
    }

    store16(d + n - 16, v);
    RETURN_OPAQUE(d);
}

/* n > 64. */
void *fill_large_sse2(u8 *d, int c, size_t n) {
    __m128i v = splat16(c);

    if (n >= REP_THRESHOLD) {
        return str_fill_huge(d, c, n);
    }

    return fill_loop(d, v, n);
}

void *fill_huge_sse2(u8 *d, int c, size_t n) {
    return fill_loop(d, splat16(c), n);
}

void *memset_sse2(void *dst, int c, size_t n) {
    u8 *d = (u8 *)dst;
    __m128i v = splat16(c);

    if (n <= 32) {
        fill_small(d, v, c, n);
    } else if (n <= 64) {
        fill_33_64(d, v, n);
    } else if (n <= 128) {
        store16(d, v);
        store16(d + 16, v);
        store16(d + 32, v);
        store16(d + 48, v);
        store16(d + n - 64, v);
        store16(d + n - 48, v);
        store16(d + n - 32, v);
        store16(d + n - 16, v);
    } else {
        return str_fill_large(d, c, n);
    }

    return dst;
}

/* --------------------------------------------------------------- compare */

/* n > 64: 16-byte blocks, then one overlapping block at the end. */
int memcmp_large_sse2(const u8 *a, const u8 *b, size_t n) {
    size_t i = 0;

    while (i + 16 < n) {
        unsigned m = ne_mask16(load16(a + i), load16(b + i));

        if (m) {
            return byte_diff(a, b, i + ctz(m));
        }

        i += 16;
    }

    unsigned m = ne_mask16(load16(a + n - 16), load16(b + n - 16));
    return m ? byte_diff(a, b, n - 16 + ctz(m)) : 0;
}

int memcmp_sse2(const void *pa, const void *pb, size_t n) {
    const u8 *a = (const u8 *)pa, *b = (const u8 *)pb;

    /* 1..32 in one path: two blocks from the start when both have the room in
       their pages, the answer without a branch on it; a branch on the size
       class is a branch the workload's sizes make unpredictable */

    if (n - 1 < 32) {
        if (__builtin_expect(both_in_page(a, b, 32), 1)) {
            uint64_t m = ne_mask16(load16(a), load16(b)) | (uint64_t)ne_mask16(load16(a + 16), load16(b + 16)) << 16;
            return memcmp_answer(a, b, m, n);
        }

        return n <= 16 ? memcmp_le16(a, b, n) : memcmp_17_32(a, b, n);
    }

    if (n == 0) {
        return 0;
    }

    if (n <= 64) {
        return memcmp_33_64(a, b, n);
    }

    /* the first 64 bytes; then the last 64 for 65..128, the kernel beyond */
    uint64_t lo = 0, hi = 0;

    for (size_t i = 0; i < 64; i += 16) {
        lo |= (uint64_t)ne_mask16(load16(a + i), load16(b + i)) << i;
    }

    if (n <= 128) {
        for (size_t i = 0; i < 64; i += 16) {
            hi |= (uint64_t)ne_mask16(load16(a + n - 64 + i), load16(b + n - 64 + i)) << i;
        }

        return memcmp_answer2(a, b, lo, hi, n);
    }

    if (lo) {
        return byte_diff(a, b, (size_t)__builtin_ctzll(lo));
    }

    return str_memcmp_large(a, b, n);
}

/* 16-byte blocks while both pointers have room in their pages, otherwise
   byte by byte up to the nearer page end. */
int strcmp_tail_sse2(const u8 *a, const u8 *b) {
    int result;

    for (;;) {
        if (both_have(a, b, 16)) {
            unsigned m = diff_or_zero_mask16(load16(a), load16(b));

            if (m) {
                return byte_diff(a, b, ctz(m));
            }

            a += 16;
            b += 16;
        } else if (strcmp_to_page_end(&a, &b, &result)) {
            return result;
        }
    }
}

int strcmp_sse2(const char *left, const char *right) {
    const u8 *a = (const u8 *)left, *b = (const u8 *)right;
    int result;

    if (strcmp_block16(&a, &b, &result)) {
        return result;
    }

    return str_strcmp_tail(a, b);
}

/* ---------------------------------------------------------- strcpy/strcat */

void *strcpy_long_sse2(u8 *d, const u8 *s, const u8 *a) {
    (void)a; /* the 16-byte kernel takes its own boundary */
    return strcpy_long16(d, s);
}

char *strcpy_sse2(char *dst, const char *src) {
    const u8 *s = (const u8 *)src;
    u8 *d = (u8 *)dst;
    /* Strings ending in the first block are copied in one pass. */
    const u8 *a = align_down(s, 16);
    unsigned m = zero_mask16(load16a(a)) >> ((uintptr_t)s & 15);

    if (m) {
        copy_small(d, s, ctz(m) + 1);
        return dst;
    }

    return str_strcpy_long(d, s, a + 16);
}

char *strcat_sse2(char *dst, const char *src) {
    str_strcpy(dst + str_strlen(dst), src);
    return dst;
}

/* ---------------------------------------------------------------- memchr */

/* n > 64, the first 32 bytes without a match: aligned blocks from the
   16-byte boundary after them, four per step joined by or, then single
   blocks, then the last 16 bytes as one block that may overlap what was
   scanned. */
void *memchr_large_sse2(const u8 *s, u8 c, size_t n) {
    if (n > PTRDIFF_MAX) {
        n = PTRDIFF_MAX; /* memchr(p, c, SIZE_MAX): C11 lets the object end at the match, so n may pass the address space; s + n must not wrap */
    }

    __m128i v = splat16(c);
    const u8 *end = s + n, *a = align_down(s + 32, 16);

    /* C11 lets the object end at the match too: no load may reach a page past
       the one that holds the first byte not yet checked. 16-byte blocks up to
       a 64-byte boundary, then whole aligned 64-byte groups, each in one page
       (unaligned groups read up to 48 bytes into the next page: a fault when
       it is not there, dev/proof found it) */

    while (((uintptr_t)a & 63) && end - a >= 16) {
        unsigned m = mask16(_mm_cmpeq_epi8(load16a(a), v));

        if (m) {
            return (void *)(a + ctz(m));
        }

        a += 16;
    }

    while (end - a >= 64) {
        __m128i e0 = _mm_cmpeq_epi8(load16a(a), v), e1 = _mm_cmpeq_epi8(load16a(a + 16), v);
        __m128i e2 = _mm_cmpeq_epi8(load16a(a + 32), v), e3 = _mm_cmpeq_epi8(load16a(a + 48), v);

        if (mask16(_mm_or_si128(_mm_or_si128(e0, e1), _mm_or_si128(e2, e3)))) {
            unsigned m = mask16(e0);

            if (m) {
                return (void *)(a + ctz(m));
            }

            m = mask16(e1);

            if (m) {
                return (void *)(a + 16 + ctz(m));
            }

            m = mask16(e2);

            if (m) {
                return (void *)(a + 32 + ctz(m));
            }

            return (void *)(a + 48 + ctz(mask16(e3)));
        }

        a += 64;
    }

    while (end - a >= 16) {
        unsigned m = eq_mask16(load16a(a), v);

        if (m) {
            return (void *)(a + ctz(m));
        }

        a += 16;
    }

    if (a < end) {
        unsigned m = eq_mask16(load16(end - 16), v) >> (16 - (end - a));

        if (m) {
            return (void *)(a + ctz(m));
        }
    }

    return 0;
}

/* One 16-byte block first, whatever n: unaligned when the page allows, and
   the index of the first match (16 for none) against n, which settles
   everything up to 16 bytes. Near the page end the aligned block around s,
   shifted, settles what the page holds; the rest, if n reaches into the
   next page, is its first aligned block or the kernel. Then 17..64 in
   overlapping blocks, the kernel beyond. */
void *memchr_sse2(const void *p, int c, size_t n) {
    const u8 *s = (const u8 *)p;
    __m128i v = splat16(c);
    unsigned m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);

    if (left >= 16) {
        m = eq_mask16(load16(s), v);

        if (m) {
            size_t i = ctz(m); /* a match past n decides too: none before it */
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= 16) {
            return 0;
        }
    } else {
        m = eq_mask16(load16a(align_down(s, 16)), v) >> ((uintptr_t)s & 15);

        if (m) {
            size_t i = ctz(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= left) {
            return 0;
        }
    }

    /* C11 lets the object end at the match, n past it: from here no load may
       reach a page past the one holding the first byte not yet checked, so
       aligned blocks only (the last-W-bytes loads this had did: a fault next
       to an unmapped page, dev/proof found it) */
    const u8 *a = align_down(s, 16) + 16;
    m = eq_mask16(load16a(a), v);

    if (m) {
        size_t i = (size_t)(a - s) + ctz(m);
        return i < n ? (void *)(s + i) : 0;
    }

    if ((size_t)(a - s) + 16 >= n) {
        return 0;
    }

    return str_memchr_large(s, (u8)c, n); /* scanned up to a + 16, the kernel's align_down(s + 32, 16) */
}

/* strnlen: memchr's blocks for the NUL, the offset for the answer (n without
   one): the first i < n with s[i] == 0, else n. The kernel from the two
   blocks in line on. */
size_t strnlen_sse2(const char *str, size_t n) {
    const u8 *s = (const u8 *)str;
    unsigned m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);

    if (left >= 16) {
        m = zero_mask16(load16(s));

        if (m) {
            size_t i = ctz(m); /* a NUL past n decides too: none before it */
            return i < n ? i : n;
        }

        if (n <= 16) {
            return n;
        }
    } else {
        m = zero_mask16(load16a(align_down(s, 16))) >> ((uintptr_t)s & 15);

        if (m) {
            size_t i = ctz(m);
            return i < n ? i : n;
        }

        if (n <= left) {
            return n;
        }
    }

    /* aligned blocks only from here, as memchr's: the object may end at the NUL */
    const u8 *a = align_down(s, 16) + 16;
    m = zero_mask16(load16a(a));

    if (m) {
        size_t i = (size_t)(a - s) + ctz(m);
        return i < n ? i : n;
    }

    if ((size_t)(a - s) + 16 >= n) {
        return n;
    }

    return str_strnlen_large(s, s, n); /* scanned up to a + 16, the kernel's align_down(s + 32, 16) */
}

/* The kernel: strnlen from A = align_down(from, 16) + 32 on (no NUL before
   it, n past it), memchr_large's blocks for the NUL, the answer the offset
   from s. */
size_t strnlen_large_sse2(const u8 *s, const u8 *from, size_t n) {
    if (n > PTRDIFF_MAX) {
        n = PTRDIFF_MAX; /* strnlen(s, SIZE_MAX): s + n must not wrap; no string is that long */
    }

    const u8 *end = s + n, *a = align_down(from + 32, 16);

    while (((uintptr_t)a & 63) && end - a >= 16) {
        unsigned m = zero_mask16(load16a(a));

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }

        a += 16;
    }

    while (end - a >= 64) {
        __m128i x0 = load16a(a), x1 = load16a(a + 16), x2 = load16a(a + 32), x3 = load16a(a + 48);

        if (zero_mask16(_mm_min_epu8(_mm_min_epu8(x0, x1), _mm_min_epu8(x2, x3)))) {
            unsigned m = zero_mask16(x0);

            if (m) {
                return (size_t)(a - s) + ctz(m);
            }

            m = zero_mask16(x1);

            if (m) {
                return (size_t)(a - s) + 16 + ctz(m);
            }

            m = zero_mask16(x2);

            if (m) {
                return (size_t)(a - s) + 32 + ctz(m);
            }

            return (size_t)(a - s) + 48 + ctz(zero_mask16(x3));
        }

        a += 64;
    }

    while (end - a >= 16) {
        unsigned m = zero_mask16(load16a(a));

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }

        a += 16;
    }

    if (a < end) {
        unsigned m = zero_mask16(load16(end - 16)) >> (16 - (end - a));

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }
    }

    return n;
}

/* --------------------------------------------------------- strspn/strcspn */

/* 256-entry membership table. Byte stores are independent of each other;
   a bitmap would serialize on read-modify-write of the same word. */
static inline void table_build(u8 table[256], const u8 *set) {
    __m128i zero = _mm_setzero_si128();

    for (int i = 0; i < 256; i += 16) {
        store16a(table + i, zero);
    }

    for (; *set; set++) {
        table[*set] = 1;
    }
}

size_t strspn_table_sse2(const u8 *s, const u8 *set) {
    u8 table[256] __attribute__((aligned(16)));
    table_build(table, set);

    /* table[0] is 0, so the NUL terminates the scan without its own test. */
    if (!table[s[0]]) {
        return 0;
    }

    if (!table[s[1]]) {
        return 1;
    }

    if (!table[s[2]]) {
        return 2;
    }

    if (!table[s[3]]) {
        return 3;
    }

    const u8 *p = align_down(s, 4); /* 4-byte groups stay inside one page */
    unsigned c0, c1, c2, c3;

    do {
        p += 4;
        c0 = table[p[0]];
        c1 = table[p[1]];
        c2 = table[p[2]];
        c3 = table[p[3]];
    } while (c0 & c1 & c2 & c3);

    size_t count = (size_t)(p - s);
    return (c0 & c1) == 0 ? count + c0 : count + 2 + c2;
}

size_t strcspn_table_sse2(const u8 *s, const u8 *set) {
    u8 table[256] __attribute__((aligned(16)));
    table_build(table, set);
    table[0] = 1; /* the NUL stops the scan like a rejected byte */

    if (table[s[0]]) {
        return 0;
    }

    if (table[s[1]]) {
        return 1;
    }

    if (table[s[2]]) {
        return 2;
    }

    if (table[s[3]]) {
        return 3;
    }

    const u8 *p = align_down(s, 4);
    unsigned c0, c1, c2, c3;

    do {
        p += 4;
        c0 = table[p[0]];
        c1 = table[p[1]];
        c2 = table[p[2]];
        c3 = table[p[3]];
    } while ((c0 | c1 | c2 | c3) == 0);

    size_t count = (size_t)(p - s);
    return (c0 | c1) != 0 ? count + 1 - c0 : count + 3 - c2;
}

size_t strspn_sse2(const char *str, const char *accept) {
    const u8 *s = (const u8 *)str, *set = (const u8 *)accept;

    if (!set[0]) {
        return 0;
    }

    if (!set[1]) {
        return skip_byte_sse2(s, set[0]);
    }

    return strspn_table_sse2(s, set);
}

size_t strcspn_sse2(const char *str, const char *reject) {
    const u8 *s = (const u8 *)str, *set = (const u8 *)reject;

    if (!set[0]) {
        return str_strlen(str);
    }

    if (!set[1]) {
        return find_byte_or_nul_sse2(s, set[0]);
    }

    return strcspn_table_sse2(s, set);
}
