/*
 * evex.c: first blocks with AVX-512 at 256 bits (AVX-512F/BW/CD/DQ/VL).
 *
 * Compile with -march=x86-64-v4 -mprefer-vector-width=256. What changes
 * against AVX2 is the short path: memcpy/memmove/memset/memcmp of 16 bytes
 * and less take one masked 128-bit load/store/compare, with no size tree and
 * no page check, and no vzeroupper because only xmm registers are involved.
 * strlen and strcpy look at one 32-byte block first when the page allows (one
 * vzeroupper on exit), 16 bytes otherwise; strcmp starts with 16 bytes.
 * 17..64 bytes of the memory functions reuse the common short paths; longer
 * inputs go through the table to the kernels. 512-bit registers are never
 * used here: on Skylake-era Intel they lower the clock.
 */
#include "common.h"

/* Low n bits set; n <= 16. Masked-off bytes of a load are never touched, so a
   masked load is safe wherever the n bytes themselves are. */
static inline __mmask16 low_mask(size_t n) { return (__mmask16)_bzhi_u32(0xffffffffu, (unsigned)n); }
static inline __m128i mload16(const void *p, __mmask16 k) { return _mm_maskz_loadu_epi8(k, p); }
static inline void mstore16(void *p, __mmask16 k, __m128i v) { _mm_mask_storeu_epi8(p, k, v); }
static inline __m256i load256(const void *p) { return _mm256_loadu_si256((const __m256i *)p); }
static inline void store256(void *p, __m256i v) { _mm256_storeu_si256((__m256i *)p, v); }
static inline __mmask32 low_mask32(size_t n) { return (__mmask32)_bzhi_u32(0xffffffffu, (unsigned)n); }
/* Bits up to and including the lowest set bit of m (m != 0): the bytes of a
   string through its terminator, given the NUL compare mask. One blsmsk. */
static inline unsigned through_first(unsigned m) { return m ^ (m - 1); }

/* ------------------------------------------------------------ NUL scans */

size_t strlen_evex(const char *str)
{
    const u8 *s = (const u8 *)str, *next = 0;
    size_t left = page_left(s);
    if (left >= 32) {
        /* one 32-byte block covers most strings; the mask needs no shift */
        unsigned m = _mm256_cmpeq_epi8_mask(load256(s), _mm256_setzero_si256());
        if (m)
            return ctz(m);
        return str_strlen_tail(align_down(s + 32, 32), s); /* bytes before it were checked */
    }
    if (left >= 16) {
        unsigned m = _mm_cmpeq_epi8_mask(load16(s), _mm_setzero_si128());
        if (m)
            return ctz(m);
        const u8 *a = align_down(s + 16, 16);
        for (int k = 0; k < 3; k++) {
            m = zero_mask16(load16a(a));
            if (m)
                return (size_t)(a - s) + ctz(m);
            a += 16;
        }
        return str_strlen_tail(a, s);
    }
    size_t n = strlen_prefix64(s, &next);
    return n != SIZE_MAX ? n : str_strlen_tail(next, s);
}

/* ------------------------------------------------------------------ copy */

/* masked: one masked load and store for n <= 16. memmove skips it: after a
   partially overlapping store the masked load forwards worse than two plain
   loads, and overlapping tiny moves follow such stores by nature. */
static inline void *copy(void *dst, const void *src, size_t n, int masked)
{
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;
    if (masked && n <= 16) {
        __mmask16 k = low_mask(n);
        mstore16(d, k, mload16(s, k));
    } else if (n <= 32) {
        copy_small(d, s, n);
    } else if (n <= 64) {
        /* two 32-byte blocks from the ends, loads before stores (overlap-safe) */
        __m256i x = load256(s), y = load256(s + n - 32);
        store256(d, x);
        store256(d + n - 32, y);
    } else {
        return str_copy_large(d, s, n);
    }
    return dst;
}

void *memcpy_evex(void *dst, const void *src, size_t n) { return copy(dst, src, n, 1); }
void *memmove_evex(void *dst, const void *src, size_t n) { return copy(dst, src, n, 0); }

/* ------------------------------------------------------------------ fill */

/* n <= 32: one masked 32-byte store when it stays in its page (a masked
   store into the next page is assisted by microcode on Intel), 16 bytes
   masked or the short path otherwise; 33..64: two 32-byte stores. */
void *memset_evex(void *dst, int c, size_t n)
{
    u8 *d = (u8 *)dst;
    if (n <= 32) {
        if (__builtin_expect(page_left(d) >= 32, 1))
            _mm256_mask_storeu_epi8(d, low_mask32(n), _mm256_set1_epi8((char)c));
        else if (n <= 16)
            mstore16(d, low_mask(n), splat16(c));
        else
            fill_small(d, splat16(c), n);
    } else if (n <= 64) {
        __m256i v = _mm256_set1_epi8((char)c);
        store256(d, v);
        store256(d + n - 32, v);
    } else {
        return str_fill_large(d, c, n);
    }
    return dst;
}

/* --------------------------------------------------------------- compare */

int memcmp_evex(const void *pa, const void *pb, size_t n)
{
    const u8 *a = (const u8 *)pa, *b = (const u8 *)pb;
    if (n <= 32) {
        /* one masked 256-bit compare for everything up to 32 bytes; n == 0 loads nothing */
        __mmask32 k = low_mask32(n);
        unsigned m = _mm256_mask_cmpneq_epu8_mask(k, _mm256_maskz_loadu_epi8(k, a), _mm256_maskz_loadu_epi8(k, b));
        return m ? byte_diff(a, b, ctz(m)) : 0;
    }
    if (n <= 64) {
        /* the first and the last 32 bytes */
        unsigned m = _mm256_cmpneq_epu8_mask(load256(a), load256(b));
        if (m)
            return byte_diff(a, b, ctz(m));
        m = _mm256_cmpneq_epu8_mask(load256(a + n - 32), load256(b + n - 32));
        return m ? byte_diff(a, b, n - 32 + ctz(m)) : 0;
    }
    return str_memcmp_large(a, b, n);
}

/* 16 bytes first: at a few bytes a 32-byte block costs more than it saves. */
int strcmp_evex(const char *left, const char *right)
{
    const u8 *a = (const u8 *)left, *b = (const u8 *)right;
    for (;;) {
        if (__builtin_expect(both_in_page(a, b, 32), 1)) {
            /* k: the lanes that are equal and not NUL. k + 1 clears them up to the first
               other lane, whose index is then the trailing zero count; all 32 equal and
               nonzero wraps to 0. One dependent chain of two compares, no kor. Strings
               differ or end within the block 9 times in 10: that path falls through. */
            __m256i x = load256(a);
            unsigned k = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, load256(b)) + 1;
            if (__builtin_expect(k != 0, 1))
                return byte_diff(a, b, ctz(k));
            return str_strcmp_tail(a + 32, b + 32);
        }
        /* one of the two may be within 32 bytes of its page end (both_in_page is
           conservative: both may still have more, so n is at most 32): masked
           loads of the bytes up to the nearer end never touch the next page. The
           lanes beyond are zero, so the first lane that is not equal-and-nonzero
           is at most n: an answer before n, else both advance by n. */
        size_t n = page_left(a) < page_left(b) ? page_left(a) : page_left(b);
        if (n > 32)
            n = 32;
        __mmask32 k = low_mask32(n);
        __m256i x = _mm256_maskz_loadu_epi8(k, a);
        unsigned m = _mm256_mask_cmpeq_epi8_mask(_mm256_test_epi8_mask(x, x), x, _mm256_maskz_loadu_epi8(k, b)) + 1;
        if (m && ctz(m) < n) /* m == 0: all 32 equal and nonzero, n == 32 */
            return byte_diff(a, b, ctz(m));
        a += n;
        b += n;
    }
}

/* ---------------------------------------------------------- strcpy/strcat */

/* Strings longer than the first block: two passes, out of line so that the
   hot path above keeps no registers across a call. */
/* Either the first block has no NUL, or s is too close to its page end for
   an unaligned load; strcpy_long16 handles both from the aligned block. */
static NOINLINE void *strcpy_long(u8 *d, const u8 *s)
{
    const u8 *a = align_down(s, 16);
    unsigned m = zero_mask16(load16a(a)) >> ((uintptr_t)s & 15);
    if (m) {
        copy_small(d, s, ctz(m) + 1);
        RETURN_OPAQUE(d);
    }
    return str_strcpy_long(d, s);
}

char *strcpy_evex(char *dst, const char *src)
{
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
static NOINLINE char *strcat_long(char *dst, const char *src)
{
    str_strcpy(dst + str_strlen(dst), src);
    RETURN_OPAQUE(dst); /* else the caller keeps dst in a callee-saved register instead of a tail call */
}

char *strcat_evex(char *dst, const char *src)
{
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

