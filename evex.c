/*
 * evex.c: first blocks with AVX-512 at 256 bits (AVX-512F/BW/CD/DQ/VL):
 * memchr, strcmp and its tail, strcpy and strcat, for every configuration
 * with AVX-512 (the memory functions and strlen are evex512.c's: their
 * 64-byte entries measured better on Skylake-SP too, dispatch.c).
 *
 * Compile with -march=x86-64-v4 -mprefer-vector-width=256. strcpy looks at
 * one 32-byte block first when the page allows (one vzeroupper on exit), 16
 * bytes otherwise, and stores the string through its NUL masked; strcmp
 * starts with 16 bytes. Longer inputs go through the table to the kernels.
 * 512-bit registers are never used here: on Skylake-era Intel they lower
 * the clock.
 */
#include "common.h"

static inline void mstore16(void *p, __mmask16 k, __m128i v) {
    _mm_mask_storeu_epi8(p, k, v);
}

static inline __m256i load256(const void *p) {
    return _mm256_loadu_si256((const __m256i *)p);
}

static inline __m256i load256a(const void *p) {
    return _mm256_load_si256((const __m256i *)p);
}

static inline void store256(void *p, __m256i v) {
    _mm256_storeu_si256((__m256i *)p, v);
}

static inline __mmask32 low_mask32(size_t n) {
    return (__mmask32)_bzhi_u32(0xffffffffu, (unsigned)n);
}

/* Bits up to and including the lowest set bit of m (m != 0): the bytes of a
   string through its terminator, given the NUL compare mask. One blsmsk. */
static inline unsigned through_first(unsigned m) {
    return m ^ (m - 1);
}

/* ---------------------------------------------------------------- memchr */

/* One 32-byte block first, whatever n: unaligned when the page allows, and
   the index of the first match (32 for none) against n, which settles
   everything up to 32 bytes without a masked load. Near the page end the
   aligned block around s, shifted, settles what the page holds; the rest,

   if n reaches into the next page, is its first aligned block or the
   kernel. Then the last 32 bytes for 33..64, the kernel beyond. */
void *memchr_evex(const void *p, int c, size_t n) {
    const u8 *s = (const u8 *)p;
    __m256i v = _mm256_set1_epi8((char)c);
    unsigned m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);
    const u8 *a = align_down(s, 32) + 32; /* the first aligned block not yet checked */

    if (left >= 32) {
        m = _mm256_cmpeq_epi8_mask(load256(s), v);

        size_t i = _tzcnt_u32(m); /* 32 for no match: no branch on it, half the calls of a mix have none and would mispredict */

        if (n <= 32) {
            return i < n ? (void *)(s + i) : 0;
        }

        if (i < 32) {
            return (void *)(s + i);
        }
    } else {
        m = _mm256_cmpeq_epi8_mask(load256a(align_down(s, 32)), v) >> ((uintptr_t)s & 31);

        if (m) {
            size_t i = ctz(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= left) {
            return 0;
        }

        /* the next page's first aligned block */
        m = _mm256_cmpeq_epi8_mask(load256a(s + left), v);

        if (m) {
            size_t i = left + ctz(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= left + 32) {
            return 0;
        }

        a = s + left + 32;
    }

    /* C11 lets the object end at the match, n past it: from here no load may
       reach a page past the one holding the first byte not yet checked, so
       aligned blocks only (the last-W-bytes loads this had did: a fault next
       to an unmapped page, dev/proof found it) */

    for (int k = 0; k < 2; k++, a += 32) { /* two blocks in line: every n up to 64 decided here */
        m = _mm256_cmpeq_epi8_mask(load256a(a), v);

        if (m) {
            size_t i = (size_t)(a - s) + ctz(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if ((size_t)(a - s) + 32 >= n) {
            return 0;
        }
    }

    /* the kernel goes on where these stopped: from the last block checked
       (its first 32 bytes, scanned), so it starts at a; the answers stay
       pointers, n counts from there */
    const u8 *from = a - 32;
    return str_memchr_large(from, (u8)c, n - (size_t)(from - s));
}

/* strnlen: memchr's blocks for the NUL (memchr_evex), the answer the offset. */
size_t strnlen_evex(const char *str, size_t n) {
    const u8 *s = (const u8 *)str;
    unsigned m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);
    const u8 *a = align_down(s, 32) + 32; /* the first aligned block not yet checked */

    if (left >= 32) {
        m = _mm256_testn_epi8_mask(load256(s), load256(s));

        size_t i = _tzcnt_u32(m); /* 32 for no NUL: no branch on it */

        if (n <= 32) {
            return i < n ? i : n;
        }

        if (i < 32) {
            return i;
        }
    } else {
        m = _mm256_testn_epi8_mask(load256a(align_down(s, 32)), load256a(align_down(s, 32))) >> ((uintptr_t)s & 31);

        if (m) {
            size_t i = ctz(m);
            return i < n ? i : n;
        }

        if (n <= left) {
            return n;
        }

        /* the next page's first aligned block */
        m = _mm256_testn_epi8_mask(load256a(s + left), load256a(s + left));

        if (m) {
            size_t i = left + ctz(m);
            return i < n ? i : n;
        }

        if (n <= left + 32) {
            return n;
        }

        a = s + left + 32;
    }

    /* C11 lets the object end at the match, n past it: from here no load may
       reach a page past the one holding the first byte not yet checked, so
       aligned blocks only (the last-W-bytes loads this had did: a fault next
       to an unmapped page, dev/proof found it) */

    for (int k = 0; k < 2; k++, a += 32) { /* two blocks in line: every n up to 64 decided here */
        m = _mm256_testn_epi8_mask(load256a(a), load256a(a));

        if (m) {
            size_t i = (size_t)(a - s) + ctz(m);
            return i < n ? i : n;
        }

        if ((size_t)(a - s) + 32 >= n) {
            return n;
        }
    }

    /* the kernel goes on where these stopped: from the last block checked
       (its first 32 bytes, scanned), so it starts at a */
    const u8 *from = a - 32;
    return str_strnlen_large(s, from, n);
}

/* --------------------------------------------------------------- compare */

/* 16 bytes first: at a few bytes a 32-byte block costs more than it saves. */
int strcmp_evex(const char *left, const char *right) {
    const u8 *a = (const u8 *)left, *b = (const u8 *)right;

    for (;;) {
        if (__builtin_expect(both_in_page(a, b, 32), 1)) {
            /* k: the lanes that are equal and not NUL. k + 1 clears them up to the first
               other lane, whose index is then the trailing zero count; all 32 equal and
               nonzero wraps to 0. One dependent chain of two compares, no kor. Strings
               differ or end within the block 9 times in 10: that path falls through. */
            __m256i x = load256(a);
            unsigned k = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, load256(b)) + 1;

            if (__builtin_expect(k != 0, 1)) {
                return byte_diff(a, b, ctz(k));
            }

            /* three more blocks at fixed offsets from the start, one page test for
               all of them: strings of up to 128 bytes end here. Single blocks
               overlap under speculation; a group joined by a tree costs its
               whole latency on the hit (2x on 65..127 bytes), and blocks moved
               to the 32-byte boundary first wait for that arithmetic (1.4x). */

            if (both_in_page(a, b, 128)) {
                x = load256(a + 32);
                k = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, load256(b + 32)) + 1;

                if (k != 0) {
                    return byte_diff(a, b, 32 + ctz(k));
                }

                x = load256(a + 64);
                k = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, load256(b + 64)) + 1;

                if (k != 0) {
                    return byte_diff(a, b, 64 + ctz(k));
                }

                x = load256(a + 96);
                k = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, load256(b + 96)) + 1;

                if (k != 0) {
                    return byte_diff(a, b, 96 + ctz(k));
                }

                /* the tail from the 32-byte boundary before a + 128: up to 31
                   bytes compared again, a aligned for its groups */
                const u8 *next = align_down(a + 128, 32);
                b += next - a;
                a = next;
            } else {
                a += 32;
                b += 32;
            }

            return str_strcmp_tail(a, b);
        }

        /* one of the two may be within 32 bytes of its page end (both_in_page is
           conservative: both may still have more, so n is at most 32): masked
           loads of the bytes up to the nearer end never touch the next page. The
           lanes beyond are zero, so the first lane that is not equal-and-nonzero
           is at most n: an answer before n, else both advance by n. */
        size_t n = page_left(a) < page_left(b) ? page_left(a) : page_left(b);

        if (n > 32) {
            n = 32;
        }

        __mmask32 k = low_mask32(n);
        __m256i x = _mm256_maskz_loadu_epi8(k, a);
        unsigned m = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, _mm256_maskz_loadu_epi8(k, b)) + 1;

        if (m && ctz(m) < n) /* m == 0: all 32 equal and nonzero, n == 32 */
            return byte_diff(a, b, ctz(m));
        a += n;
        b += n;
    }
}

/* Bit i set when x[i] != y[i] or x[i] == 0: where a comparison stops. */
static inline unsigned stop_mask32(__m256i x, __m256i y) {
    return ~_mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, y);
}

/* The strcmp tail with mask compares: a group of four blocks is xors with
   b, a min tree, and one masked test (the AVX2 tail's and, min and movemask
   sequence measured 1.3x slower on Skylake-SP, four compares to masks and
   a kortest 1.5x). a 32-aligned and 128 bytes from each pointer inside its
   page for a group, b unaligned; elsewhere one masked step up to the next 32-byte
   boundary of a or the page end of b, whichever is nearer: the lanes beyond
   it are zero, so the first stop is at most n, and a never-touched next
   page cannot fault. No byte loop at a page end. */
int strcmp_tail_evex(const u8 *a, const u8 *b) {
    for (;;) {
        if (((uintptr_t)a & 31) == 0) {
            size_t left = page_left(a) < page_left(b) ? page_left(a) : page_left(b);

            if (left >= 128) {
                /* groups of four blocks while they fit in both pages (glibc's
                   loop): the or of the four xors with b (loads folded, the
                   last three joined by ternary logic) and the minimum of a's
                   four blocks; one test of the minimum for a NUL masks the
                   test of the or for a difference, so one mask and one
                   increment per group decide. On a hit the xors are at hand:
                   the first block with a lane that is NUL or differs. */
                size_t groups = left / 128;

                do {
                    __m256i x0 = load256a(a), x1 = load256a(a + 32), x2 = load256a(a + 64), x3 = load256a(a + 96);
                    __m256i d0 = _mm256_xor_si256(x0, load256(b)), d1 = _mm256_xor_si256(x1, load256(b + 32));
                    __m256i d2 = _mm256_xor_si256(x2, load256(b + 64)), d3 = _mm256_xor_si256(x3, load256(b + 96));
                    __m256i lo = _mm256_min_epu8(_mm256_min_epu8(x0, x1), _mm256_min_epu8(x2, x3));
                    __m256i d = _mm256_ternarylogic_epi32(d0, d1, _mm256_ternarylogic_epi32(d2, d3, d0, 0xfe), 0xfe);
                    unsigned k = _mm256_mask_testn_epi8_mask(_mm256_test_epi8_mask(lo, lo), d, d) + 1;

                    if (k) {
                        unsigned m = _mm256_mask_testn_epi8_mask(_mm256_test_epi8_mask(x0, x0), d0, d0) + 1;

                        if (m) {
                            return byte_diff(a, b, ctz(m));
                        }

                        m = _mm256_mask_testn_epi8_mask(_mm256_test_epi8_mask(x1, x1), d1, d1) + 1;

                        if (m) {
                            return byte_diff(a, b, 32 + ctz(m));
                        }

                        m = _mm256_mask_testn_epi8_mask(_mm256_test_epi8_mask(x2, x2), d2, d2) + 1;

                        if (m) {
                            return byte_diff(a, b, 64 + ctz(m));
                        }

                        return byte_diff(a, b, 96 + ctz(_mm256_mask_testn_epi8_mask(_mm256_test_epi8_mask(x3, x3), d3, d3) + 1));
                    }

                    a += 128;
                    b += 128;
                    /* opaque: else the two pointers become one index, and the
                       xors with an indexed memory operand split into two uops
                       each on Skylake (1.3x on 4 KiB) */
                    __asm__("" : "+r"(a), "+r"(b));
                } while (--groups);

                continue;
            }
        }

        /* a masked step to a's next 32-byte boundary, within b's page too */
        size_t n = 32 - ((uintptr_t)a & 31);

        if (page_left(b) < n) {
            n = page_left(b);
        }

        __mmask32 km = low_mask32(n);
        unsigned m = stop_mask32(_mm256_maskz_loadu_epi8(km, a), _mm256_maskz_loadu_epi8(km, b));

        if (m) {
            size_t i = ctz(m);

            if (i < n) {
                return byte_diff(a, b, i);
            }
        }

        a += n;
        b += n;
    }
}

/* ---------------------------------------------------------- strcpy/strcat */

/* Strings longer than the first block: two passes, out of line so that the
   hot path above keeps no registers across a call. */
/* Either the first block has no NUL, or s is too close to its page end for
   an unaligned load; strcpy_long16 handles both from the aligned block. */
static NOINLINE void *strcpy_long(u8 *d, const u8 *s) {
    const u8 *a = align_down(s, 16);
    unsigned m = zero_mask16(load16a(a)) >> ((uintptr_t)s & 15);

    if (m) {
        copy_small(d, s, ctz(m) + 1);
        RETURN_OPAQUE(d);
    }

    return str_strcpy_long(d, s, a + 16);
}

char *strcpy_evex(char *dst, const char *src) {
    const u8 *s = (const u8 *)src;
    u8 *d = (u8 *)dst;
    size_t left = page_left(s);

    if (left >= 32) {
        /* a string ending within the block: one load, one masked store */
        __m256i v = load256(s);
        unsigned m = _mm256_cmpeq_epi8_mask(v, _mm256_setzero_si256());

        if (m) {
            _mm256_mask_storeu_epi8(d, through_first(m), v);
            return dst;
        }

        /* three aligned blocks in line: strings of up to 128 bytes are copied
           here as whole 32-byte blocks and the last 32 bytes (through the
           wrapper and the kernel they cost 1.4-1.8x), the long copy beyond */
        const u8 *a = align_down(s, 32) + 32;

        for (int k = 0; k < 3; k++, a += 32) {
            m = _mm256_cmpeq_epi8_mask(load256a(a), _mm256_setzero_si256());

            if (m) {
                size_t n = (size_t)(a - s) + ctz(m) + 1; /* 33..128: the first block had no NUL */
                size_t off = 0;

                for (; off + 32 < n; off += 32) {
                    store256(d + off, load256(s + off)); /* within [s, a + 32) */
                }

                store256(d + n - 32, load256(s + n - 32));
                return dst;
            }
        }

        return str_strcpy_long(d, s, a); /* [s, a) has no NUL */
    } else if (left >= 16) {
        __m128i v = load16(s);
        unsigned m = _mm_cmpeq_epi8_mask(v, _mm_setzero_si128());

        if (m) {
            mstore16(d, (__mmask16)through_first(m), v);
            return dst;
        }
    }

    return strcpy_long(d, s);
}

/* Out of line, so that the common case below keeps no frame and no call. */
static NOINLINE char *strcat_long(char *dst, const char *src) {
    str_strcpy(dst + str_strlen(dst), src);
    RETURN_OPAQUE(dst); /* else the caller keeps dst in a callee-saved register instead of a tail call */
}

char *strcat_evex(char *dst, const char *src) {
    const u8 *d = (const u8 *)dst, *s = (const u8 *)src;

    if (both_in_page(d, s, 32)) {
        /* both strings end within their first 32 bytes: one load each, one masked store */
        unsigned md = _mm256_cmpeq_epi8_mask(load256(d), _mm256_setzero_si256());

        if (md) {
            __m256i v = load256(s);
            unsigned ms = _mm256_cmpeq_epi8_mask(v, _mm256_setzero_si256());

            if (ms) {
                _mm256_mask_storeu_epi8(dst + ctz(md), through_first(ms), v);
                return dst;
            }
        }
    }

    return strcat_long(dst, src);
}
