/*
 * memset of large blocks by non-temporal stores, as glibc 2.40 and later
 * fill from __x86_memset_non_temporal_threshold on: from the threshold the
 * head and the last 128 bytes by ordinary stores, 128 bytes a step by
 * vmovntdq between them, then an sfence; below it rep stosb from 4 KiB and
 * avx2's memset under that, called by name (in a table of --table this
 * kernel is the memset slot: str_memset would come back here).
 *   ntall (from 4 KiB), nt1m .. nt32m, ntnever   the threshold
 */
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"

static inline void *nt_fill(void *dst, int c, size_t n) {
    u8 *d = (u8 *)dst;
    __m256i v = _mm256_set1_epi8((char)c);
    u8 *end = d + n - 128;
    u8 *p = (u8 *)(((uintptr_t)d | 31) + 1);
    _mm256_storeu_si256((__m256i *)d, v);

    while (p < end) {
        _mm256_stream_si256((__m256i *)p, v);
        _mm256_stream_si256((__m256i *)(p + 32), v);
        _mm256_stream_si256((__m256i *)(p + 64), v);
        _mm256_stream_si256((__m256i *)(p + 96), v);
        p += 128;
    }

    _mm_sfence();
    _mm256_storeu_si256((__m256i *)end, v);
    _mm256_storeu_si256((__m256i *)(end + 32), v);
    _mm256_storeu_si256((__m256i *)(end + 64), v);
    _mm256_storeu_si256((__m256i *)(end + 96), v);
    return dst;
}

static inline void *rep_fill(void *d, int c, size_t n) {
    void *p = d;
    __asm__ volatile("rep stosb" : "+D"(p), "+c"(n) : "a"(c) : "memory");
    return d;
}

/* the threshold from 4 KiB (where the library's huge path starts) to 32 MiB,
   and never: the same wrapper without them, the base to compare with. From
   4 KiB to the threshold rep stosb here, not the library's huge path */
#define KERNEL(name, t) \
    void *kl_memset_avx2_##name(void *d, int c, size_t n) { \
        return n >= (t) ? nt_fill(d, c, n) : n >= 4096 ? rep_fill(d, c, n) : memset_avx2(d, c, n); \
    }

KERNEL(ntall, (size_t)4096)
KERNEL(nt1m, (size_t)1 << 20)
KERNEL(nt2m, (size_t)2 << 20)
KERNEL(nt4m, (size_t)4 << 20)
KERNEL(nt8m, (size_t)8 << 20)
KERNEL(nt16m, (size_t)16 << 20)
KERNEL(nt32m, (size_t)32 << 20)
KERNEL(ntnever, SIZE_MAX)
