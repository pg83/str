/*
 * evex512.c: first blocks with 512-bit registers, for the CPUs where zmm costs
 * nothing extra: Zen 4 and 5, Ice Lake and later (Skylake-SP stays at 256
 * bits: 512-bit instructions lower its clock, as glibc also avoids there).
 * Compile with -march=x86-64-v4 -mprefer-vector-width=512. The point is the
 * size distribution of real calls (median 8 bytes for strings, 14-29 for
 * memory blocks, a long tail): one 64-byte block settles 9 strings in 10, one
 * masked 64-byte load and store copies 90% of the blocks, with one branch on
 * the size instead of a tree of them. Longer inputs go through the table.
 */
#include "common.h"

static inline __mmask64 low_mask64(size_t n) { return (__mmask64)_bzhi_u64(~0ull, (unsigned)n); } /* n <= 64 */
static inline __m512i load512(const void *p) {
    return _mm512_loadu_si512(p);
}

static inline __m512i load512a(const void *p) {
    return _mm512_load_si512(p);
}

static inline void store512(void *p, __m512i v) {
    _mm512_storeu_si512(p, v);
}

static inline void store512a(void *p, __m512i v) {
    _mm512_store_si512(p, v);
}

static inline __m512i mload64(const void *p, __mmask64 k) {
    return _mm512_maskz_loadu_epi8(k, p);
}

static inline void mstore64(void *p, __mmask64 k, __m512i v) {
    _mm512_mask_storeu_epi8(p, k, v);
}

static inline unsigned long long zero_mask64(__m512i v) {
    return _mm512_testn_epi8_mask(v, v);
}

static inline unsigned ctz64(unsigned long long m) {
    return (unsigned)__builtin_ctzll(m);
}

/* ------------------------------------------------------------ NUL scans */

/* a is 16-aligned, s <= a, no NUL in [s, a). The 64-byte block around a is
   checked with the bytes before a masked off (they may lie before s); then
   single blocks up to a 256-byte boundary, then four at a time: a 256-aligned
   group never crosses a page. */
size_t strlen_tail_evex512(const u8 *a, const u8 *s) {
    const u8 *start = align_down(a, 64);
    unsigned long long first = zero_mask64(load512a(start)) >> (a - start);

    if (first) {
        return (size_t)(a - s) + ctz64(first);
    }

    a = start + 64;

    while ((uintptr_t)a & 255) {
        unsigned long long m = zero_mask64(load512a(a));

        if (m) {
            return (size_t)(a - s) + ctz64(m);
        }

        a += 64;
    }

    for (;;) {
        __m512i x0 = load512a(a), x1 = load512a(a + 64), x2 = load512a(a + 128), x3 = load512a(a + 192);
        __m512i m = _mm512_min_epu8(_mm512_min_epu8(x0, x1), _mm512_min_epu8(x2, x3));

        if (_mm512_testn_epi8_mask(m, m)) {
            unsigned long long k = zero_mask64(x0);

            if (k) {
                return (size_t)(a - s) + ctz64(k);
            }

            k = zero_mask64(x1);

            if (k) {
                return (size_t)(a - s) + 64 + ctz64(k);
            }

            k = zero_mask64(x2);

            if (k) {
                return (size_t)(a - s) + 128 + ctz64(k);
            }

            return (size_t)(a - s) + 192 + ctz64(zero_mask64(x3));
        }

        a += 256;
    }
}

size_t strlen_evex512(const char *str) {
    const u8 *s = (const u8 *)str;
    unsigned long long m;

    if (page_left(s) >= 64) {
        m = zero_mask64(load512(s));
    } else {
        /* too close to the page end for an unaligned load: the aligned block around s, shifted */
        m = zero_mask64(load512a(align_down(s, 64))) >> ((uintptr_t)s & 63);
    }

    if (m) {
        return ctz64(m);
    }

    return str_strlen_tail(align_down(s, 64) + 64, s);
}

/* ---------------------------------------------------------------- memchr */

/* One 64-byte block first, whatever n: unaligned when the page allows, and
   the index of the first match (64 for none) against n, which settles
   everything up to 64 bytes without a masked load (one costs 1.1x against
   glibc on Zen 5). Near the page end the aligned block around s, shifted,
   settles what the page holds; the rest, if n reaches into the next page,
   is its first aligned block or the kernel. Then the last 64 bytes for
   65..128, the kernel beyond. */
void *memchr_evex512(const void *p, int c, size_t n) {
    const u8 *s = (const u8 *)p;
    __m512i v = _mm512_set1_epi8((char)c);
    unsigned long long m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);
    const u8 *a = align_down(s, 64) + 64; /* the first aligned block not yet checked */

    if (left >= 64) {
        m = _mm512_cmpeq_epi8_mask(load512(s), v);

        size_t i = _tzcnt_u64(m); /* 64 for no match: no branch on it, half the calls of a mix have none and would mispredict */

        if (n <= 64) {
            return i < n ? (void *)(s + i) : 0;
        }

        if (i < 64) {
            return (void *)(s + i);
        }
    } else {
        m = _mm512_cmpeq_epi8_mask(load512a(align_down(s, 64)), v) >> ((uintptr_t)s & 63);

        if (m) {
            size_t i = ctz64(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= left) {
            return 0;
        }

        /* the next page's first aligned block */
        m = _mm512_cmpeq_epi8_mask(load512a(s + left), v);

        if (m) {
            size_t i = left + ctz64(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= left + 64) {
            return 0;
        }

        a = s + left + 64;
    }

    /* C11 lets the object end at the match, n past it: from here no load may
       reach a page past the one holding the first byte not yet checked, so
       aligned blocks only (the last-W-bytes loads this had did: a fault next
       to an unmapped page, dev/proof found it) */

    for (int k = 0; k < 2; k++, a += 64) { /* two blocks in line: every n up to 128 decided here */
        m = _mm512_cmpeq_epi8_mask(load512a(a), v);

        if (m) {
            size_t i = (size_t)(a - s) + ctz64(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if ((size_t)(a - s) + 64 >= n) {
            return 0;
        }
    }

    /* the kernel goes on where these stopped: from the last block checked
       (its first 32 bytes, scanned), so it starts at a; the answers stay
       pointers, n counts from there */
    const u8 *from = a - 32;
    return str_memchr_large(from, (u8)c, n - (size_t)(from - s));
}

/* strnlen: memchr's blocks for the NUL (memchr_evex512), the answer the offset. */
size_t strnlen_evex512(const char *str, size_t n) {
    const u8 *s = (const u8 *)str;
    unsigned long long m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);
    const u8 *a = align_down(s, 64) + 64; /* the first aligned block not yet checked */

    if (left >= 64) {
        m = zero_mask64(load512(s));

        size_t i = _tzcnt_u64(m); /* 64 for no NUL: no branch on it */

        if (n <= 64) {
            return i < n ? i : n;
        }

        if (i < 64) {
            return i;
        }
    } else {
        m = zero_mask64(load512a(align_down(s, 64))) >> ((uintptr_t)s & 63);

        if (m) {
            size_t i = ctz64(m);
            return i < n ? i : n;
        }

        if (n <= left) {
            return n;
        }

        /* the next page's first aligned block */
        m = zero_mask64(load512a(s + left));

        if (m) {
            size_t i = left + ctz64(m);
            return i < n ? i : n;
        }

        if (n <= left + 64) {
            return n;
        }

        a = s + left + 64;
    }

    /* C11 lets the object end at the match, n past it: from here no load may
       reach a page past the one holding the first byte not yet checked, so
       aligned blocks only (the last-W-bytes loads this had did: a fault next
       to an unmapped page, dev/proof found it) */

    for (int k = 0; k < 2; k++, a += 64) { /* two blocks in line: every n up to 128 decided here */
        m = zero_mask64(load512a(a));

        if (m) {
            size_t i = (size_t)(a - s) + ctz64(m);
            return i < n ? i : n;
        }

        if ((size_t)(a - s) + 64 >= n) {
            return n;
        }
    }

    /* the kernel goes on where these stopped: from the last block checked
       (its first 32 bytes, scanned), so it starts at a */
    const u8 *from = a - 32;
    return str_strnlen_large(s, from, n);
}

/* ------------------------------------------------------------------ copy */

static inline void *copy(u8 *d, const u8 *s, size_t n) {
    /* every load happens before any store: overlap-safe, so memmove is the same code */
    if (n <= 64) {
        __mmask64 k = low_mask64(n);
        mstore64(d, k, mload64(s, k));
        return d;
    }

    if (n <= 128) {
        __m512i x = load512(s), y = load512(s + n - 64);
        store512(d, x);
        store512(d + n - 64, y);
        return d;
    }

    if (n <= 256) {
        __m512i x0 = load512(s), x1 = load512(s + 64), y0 = load512(s + n - 128), y1 = load512(s + n - 64);
        store512(d, x0);
        store512(d + 64, x1);
        store512(d + n - 128, y0);
        store512(d + n - 64, y1);
        return d;
    }

    return str_copy_large(d, s, n);
}

void *memcpy_evex512(void *dst, const void *src, size_t n) {
    return copy((u8 *)dst, (const u8 *)src, n);
}

void *memmove_evex512(void *dst, const void *src, size_t n) {
    return copy((u8 *)dst, (const u8 *)src, n);
}

/* ------------------------------------------------------------------ fill */

/* n > 256. No vector argument (the library's convention: nothing vector is
   live across a call), the splat is remade. Four unaligned stores at each end,
   aligned 256-byte steps between; the loop is not unrolled (the compiler
   would, with a long prologue for a count that is usually 1-16). */
static inline void *fill_loop(u8 *d, int c, size_t n) {
    __m512i v = _mm512_set1_epi8((char)c);
    store512(d, v);
    store512(d + 64, v);
    store512(d + 128, v);
    store512(d + 192, v);
    u8 *end = d + n - 256;
    u8 *dp = (u8 *)align_down(d + 256, 64);

    NO_UNROLL
    while (dp < end) {
        store512a(dp, v);
        store512a(dp + 64, v);
        store512a(dp + 128, v);
        store512a(dp + 192, v);
        dp += 256;
    }

    store512(end, v);
    store512(end + 64, v);
    store512(end + 128, v);
    store512(end + 192, v);
    RETURN_OPAQUE(d);
}

/* n > 128, the slot's contract: memset_evex512 settles up to 256 bytes
   itself, but another first block may hand any size above 128 to this slot. */
void *fill_large_evex512(u8 *d, int c, size_t n) {
    if (n <= 256) {
        __m512i v = _mm512_set1_epi8((char)c);
        store512(d, v);
        store512(d + 64, v);
        store512(d + n - 128, v);
        store512(d + n - 64, v);
        RETURN_OPAQUE(d);
    }

    if (n >= REP_THRESHOLD) {
        return str_fill_huge(d, c, n);
    }

    return fill_loop(d, c, n);
}

void *memset_evex512(void *dst, int c, size_t n) {
    u8 *d = (u8 *)dst;
    __m512i v = _mm512_set1_epi8((char)c);

    if (n <= 64) {
        mstore64(d, low_mask64(n), v);
        return dst;
    }

    if (n <= 128) {
        store512(d, v);
        store512(d + n - 64, v);
        return dst;
    }

    if (n <= 256) {
        store512(d, v);
        store512(d + 64, v);
        store512(d + n - 128, v);
        store512(d + n - 64, v);
        return dst;
    }

    return str_fill_large(d, c, n);
}

/* --------------------------------------------------------------- compare */

int memcmp_evex512(const void *pa, const void *pb, size_t n) {
    const u8 *a = (const u8 *)pa, *b = (const u8 *)pb;
    unsigned long long m;

    if (n <= 64) {
        if (n == 0) {
            return 0;
        }

        /* one masked compare, the answer without a branch on it (memcmp_answer) */
        __mmask64 k = low_mask64(n);
        return memcmp_answer(a, b, _mm512_mask_cmpneq_epu8_mask(k, mload64(a, k), mload64(b, k)), n);
    }

    /* the first block and, for 65..128, the last 64 bytes: one answer; the kernel beyond */
    m = _mm512_cmpneq_epu8_mask(load512(a), load512(b));

    if (n <= 128) {
        return memcmp_answer2(a, b, m, _mm512_cmpneq_epu8_mask(load512(a + n - 64), load512(b + n - 64)), n);
    }

    if (m) {
        return byte_diff(a, b, ctz64(m));
    }

    return str_memcmp_large(a, b, n);
}

/* Bit i set when x[i] != y[i] or x[i] == 0: where a comparison stops. */
static inline unsigned long long stop_mask64(__m512i x, __m512i y) {
    return ~_mm512_mask_cmpeq_epi8_mask(_mm512_test_epi8_mask(x, x), x, y);
}

/* a 64-aligned and 256 bytes from each pointer inside its page: four blocks
   at a time, b unaligned. The or of the xors for the differences and the
   bytewise minimum for a NUL, one masked test; the exact block only on a hit.
   Elsewhere one masked step up to the next 64-byte boundary of a or the page
   end of b, whichever is nearer: the lanes beyond it are zero, so the first
   stop is at most n, and a never-touched next page cannot fault. The first
   step aligns a, so the loads from a never split a line. */
int strcmp_tail_evex512(const u8 *a, const u8 *b) {
    /* Two unaligned blocks first: strings up to about 160 bytes end here
       without the alignment step or the four-block group. */

    if (page_left(a) >= 128 && page_left(b) >= 128) {
        unsigned long long m = stop_mask64(load512(a), load512(b));

        if (m) {
            return byte_diff(a, b, ctz64(m));
        }

        m = stop_mask64(load512(a + 64), load512(b + 64));

        if (m) {
            return byte_diff(a, b, 64 + ctz64(m));
        }

        size_t step = (size_t)(align_down(a, 64) + 128 - a); /* 65..128: a aligned, [a, a+128) done */
        a += step;
        b += step;
    }

    for (;;) {
        if (((uintptr_t)a & 63) == 0) {
            size_t left = page_left(a) < page_left(b) ? page_left(a) : page_left(b);

            if (left >= 256) {
                /* groups of four blocks while they fit in both pages: the or of
                   the four xors with b and the minimum of a's four blocks, one
                   masked test of each (glibc's form; four compares to masks
                   and a kor chain ran 1.4x slower on Zen 5) */
                size_t groups = left / 256;

                do {
                    __m512i x0 = load512a(a), x1 = load512a(a + 64), x2 = load512a(a + 128), x3 = load512a(a + 192);
                    __m512i d0 = _mm512_xor_si512(x0, load512(b)), d1 = _mm512_xor_si512(x1, load512(b + 64));
                    __m512i d2 = _mm512_xor_si512(x2, load512(b + 128)), d3 = _mm512_xor_si512(x3, load512(b + 192));
                    __m512i lo = _mm512_min_epu8(_mm512_min_epu8(x0, x1), _mm512_min_epu8(x2, x3));
                    __m512i d = _mm512_ternarylogic_epi32(d0, d1, _mm512_ternarylogic_epi32(d2, d3, d0, 0xfe), 0xfe);
                    unsigned long long k = _mm512_mask_testn_epi8_mask(_mm512_test_epi8_mask(lo, lo), d, d) + 1;

                    if (k) {
                        unsigned long long m = _mm512_mask_testn_epi8_mask(_mm512_test_epi8_mask(x0, x0), d0, d0) + 1;

                        if (m) {
                            return byte_diff(a, b, ctz64(m));
                        }

                        m = _mm512_mask_testn_epi8_mask(_mm512_test_epi8_mask(x1, x1), d1, d1) + 1;

                        if (m) {
                            return byte_diff(a, b, 64 + ctz64(m));
                        }

                        m = _mm512_mask_testn_epi8_mask(_mm512_test_epi8_mask(x2, x2), d2, d2) + 1;

                        if (m) {
                            return byte_diff(a, b, 128 + ctz64(m));
                        }

                        return byte_diff(a, b, 192 + ctz64(_mm512_mask_testn_epi8_mask(_mm512_test_epi8_mask(x3, x3), d3, d3) + 1));
                    }

                    a += 256;
                    b += 256;
                    __asm__("" : "+r"(a), "+r"(b)); /* two pointers, not one index (see the evex tail) */
                } while (--groups);

                continue;
            }
        }

        size_t n = 64 - ((uintptr_t)a & 63); /* within a's page too */

        if (page_left(b) < n) {
            n = page_left(b);
        }

        __mmask64 km = low_mask64(n);
        unsigned long long m = stop_mask64(mload64(a, km), mload64(b, km));

        if (m) {
            size_t i = ctz64(m);

            if (i < n) {
                return byte_diff(a, b, i);
            }
        }

        a += n;
        b += n;
    }
}

/* ---------------------------------------------------------------- strcpy */

/* n <= 128, all n source bytes readable. */
static inline void copy_le128(u8 *d, const u8 *s, size_t n) {
    if (n <= 64) {
        __mmask64 k = low_mask64(n);
        mstore64(d, k, mload64(s, k));
    } else {
        __m512i x = load512(s), y = load512(s + n - 64);
        store512(d, x);
        store512(d + n - 64, y);
    }
}

/* Aligned 64-byte blocks from the one around s: the first two settle the
   strings up to 128 bytes in one copy; then single blocks to a 256-byte
   boundary and four at a time (a 256-aligned group never crosses a page),
   each tested before it is stored. The last 64 bytes of the string go out as
   one overlapping store ending at the NUL. */
void *strcpy_long_evex512(u8 *d, const u8 *s, const u8 *known) {
    /* the 64-byte block holding the first byte not known clear, the known
       bytes (and any before s) shifted out of its mask */
    const u8 *a = align_down(known, 64);
    __m512i x = load512a(a);
    unsigned long long m = zero_mask64(x) >> (known - a);

    if (m) {
        size_t n = (size_t)(known - s) + ctz64(m) + 1; /* at most 192: known is at most 128 past s */

        if (n <= 128) {
            copy_le128(d, s, n);
        } else {
            store512(d, load512(s));
            store512(d + 64, load512(s + 64));
            store512(d + n - 64, load512(s + n - 64));
        }

        RETURN_OPAQUE(d);
    }

    if (a < s) {
        /* the block holds s itself: the next one too, before a head store can take 64 bytes */
        a += 64;
        x = load512a(a);
        m = zero_mask64(x);

        if (m) {
            copy_le128(d, s, (size_t)(a - s) + ctz64(m) + 1);
            RETURN_OPAQUE(d);
        }

        store512(d, load512(s)); /* within the two blocks checked */
    } else {
        /* [s, a + 64) has no NUL: the bytes before a in whole blocks, the last
           reaching into the block at a at most */

        for (size_t off = 0; off < (size_t)(a - s); off += 64) {
            store512(d + off, load512(s + off));
        }
    }

    store512(d + (a - s), x);
    a += 64;

    while ((uintptr_t)a & 255) {
        x = load512a(a);
        m = zero_mask64(x);

        if (m) {
            goto last;
        }

        store512(d + (a - s), x);
        a += 64;
    }

    for (;;) {
        __m512i x0 = load512a(a), x1 = load512a(a + 64), x2 = load512a(a + 128), x3 = load512a(a + 192);
        __m512i lo = _mm512_min_epu8(_mm512_min_epu8(x0, x1), _mm512_min_epu8(x2, x3));

        if (_mm512_testn_epi8_mask(lo, lo)) {
            u8 *o = d + (a - s);

            if ((m = zero_mask64(x0))) {
                goto last;
            }

            store512(o, x0);
            a += 64;

            if ((m = zero_mask64(x1))) {
                goto last;
            }

            store512(o + 64, x1);
            a += 64;

            if ((m = zero_mask64(x2))) {
                goto last;
            }

            store512(o + 128, x2);
            a += 64;
            m = zero_mask64(x3);
            goto last;
        }

        u8 *o = d + (a - s);
        store512(o, x0);
        store512(o + 64, x1);
        store512(o + 128, x2);
        store512(o + 192, x3);
        a += 256;
    }

last:;
    const u8 *nul = a + ctz64(m); /* nul >= s + 64 here */
    store512(d + (nul - s) - 63, load512(nul - 63));
    RETURN_OPAQUE(d);
}
