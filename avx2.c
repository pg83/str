/*
 * avx2.c: the primitives for x86-64-v3 (AVX2, BMI2, SSE4.2 and friends).
 *
 * Compile with -march=x86-64-v3. Short paths come from common.h; inputs
 * longer than 64 bytes go through the table to the kernels, whose 32-byte
 * versions are defined here and serve the AVX-512 first blocks as well.
 */
#include "common.h"
#include "pcmpistri.h"

static inline __m256i load256(const void *p) {
    return _mm256_loadu_si256((const __m256i *)p);
}

static inline __m256i load256a(const void *p) {
    return _mm256_load_si256((const __m256i *)p);
}

static inline void store256(void *p, __m256i v) {
    _mm256_storeu_si256((__m256i *)p, v);
}

static inline void store256a(void *p, __m256i v) {
    _mm256_store_si256((__m256i *)p, v);
}

static inline unsigned mask256(__m256i v) {
    return (unsigned)_mm256_movemask_epi8(v);
}

static inline __m256i eq_zero(__m256i v) {
    return _mm256_cmpeq_epi8(v, _mm256_setzero_si256());
}

/* ------------------------------------------------------- strspn/strcspn */

/* Byte classification by two shuffles (Hyperscan's "truffle"): a byte b is in
   the set when bit (b >> 4 & 7) is set in table[(b >> 7) * 16 + (b & 15)].
   pshufb indexes by the low four bits and gives 0 where bit 7 of the index is
   set, so the first table answers for b < 0x80 as it is and the second for
   b >= 0x80 after b ^ 0x80. Any set, 32 bytes per step, no pcmpistri. */
/* The two tables are built as four 64-bit words (a byte store followed by a
   16-byte load of the same bytes would stall on store forwarding; 8-byte words
   forward) and assembled into vectors from registers. Four words in
   registers chosen by masks instead of the indexed array were measured 1.6x
   slower: more operations per byte than the forwarding costs. */
static inline void truffle_tables(const u8 *set, int with_nul, __m128i *low, __m128i *high) {
    uint64_t t[4] = {with_nul ? 1u : 0u, 0, 0, 0};

    for (; *set; set++) {
        unsigned b = *set;
        t[((b >> 7) << 1) | ((b >> 3) & 1)] |= (uint64_t)(1u << ((b >> 4) & 7)) << (8 * (b & 7));
    }

    *low = _mm_set_epi64x((long long)t[1], (long long)t[0]);
    *high = _mm_set_epi64x((long long)t[3], (long long)t[2]);
}

/* The bytes of x that are in the set, one bit each. */
static inline unsigned truffle_members(__m256i x, __m256i low, __m256i high, __m256i bits) {
    __m256i t = _mm256_or_si256(_mm256_shuffle_epi8(low, x), _mm256_shuffle_epi8(high, _mm256_xor_si256(x, _mm256_set1_epi8((char)0x80))));
    __m256i b = _mm256_shuffle_epi8(bits, _mm256_and_si256(_mm256_srli_epi16(x, 4), _mm256_set1_epi8(7)));
    return ~mask256(_mm256_cmpeq_epi8(_mm256_and_si256(t, b), _mm256_setzero_si256()));
}

#define TRUFFLE_BITS _mm256_broadcastsi128_si256(_mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0))

/* The tables for a set of at most 32 bytes held in two registers, `count`
   bytes long (the lanes beyond may hold anything). Built in vectors: the
   position of each byte's bit, (b >> 7) * 16 + (b & 15), and the bit itself
   are computed for all lanes at once, and each lane then contributes a
   one-hot 32-byte word selected by its position. Some 40 cycles for 32
   bytes against 200 for the scalar build (a store-forwarded read-modify-
   write per byte); the two 32-byte stores forward to the byte broadcasts. */
static inline void truffle_tables32(__m128i set1, __m128i set2, size_t count, int with_nul, __m128i *low, __m128i *high) {
    __m256i s = _mm256_inserti128_si256(_mm256_castsi128_si256(set1), set2, 1);
    __m256i pos = _mm256_or_si256(_mm256_and_si256(s, _mm256_set1_epi8(15)), _mm256_and_si256(_mm256_srli_epi16(s, 3), _mm256_set1_epi8(16)));
    __m256i val = _mm256_shuffle_epi8(TRUFFLE_BITS, _mm256_and_si256(_mm256_srli_epi16(s, 4), _mm256_set1_epi8(7)));
    __m256i iota = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m256i t = _mm256_castsi128_si256(_mm_cvtsi32_si128(with_nul));
    u8 p[32] __attribute__((aligned(32))), v[32] __attribute__((aligned(32)));

    _mm256_store_si256((__m256i *)p, pos);
    _mm256_store_si256((__m256i *)v, val);

    for (size_t k = 0; k < count; k++) {
        t = _mm256_or_si256(t, _mm256_and_si256(_mm256_cmpeq_epi8(iota, _mm256_set1_epi8((char)p[k])), _mm256_set1_epi8((char)v[k])));
    }

    *low = _mm256_castsi256_si128(t);
    *high = _mm256_extracti128_si256(t, 1);
}

/* The scan from `from` on (the bytes of s before it are settled): aligned
   32-byte blocks, the first with the bytes before `from` shifted out. members:
   stop at the first byte in the set (strcspn, the NUL counts as one), else at
   the first byte not in it (strspn, the NUL is not). */
static NOINLINE size_t truffle_scan(const u8 *from, const u8 *s, __m128i low16, __m128i high16, int members) {
    __m256i low = _mm256_broadcastsi128_si256(low16), high = _mm256_broadcastsi128_si256(high16), bits = TRUFFLE_BITS;
    unsigned flip = members ? 0 : 0xffffffffu;
    const u8 *a = align_down(from, 32); /* aligned blocks never cross a page */
    unsigned m = (truffle_members(load256a(a), low, high, bits) ^ flip) >> (from - a);

    if (m) {
        return (size_t)(from - s) + ctz(m);
    }

    for (;;) {
        a += 32;
        m = truffle_members(load256a(a), low, high, bits) ^ flip;

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }
    }
}

/* The scan with the tables of a set in two registers (count bytes) or, for
   a longer set, built from the string. */
static NOINLINE size_t truffle_scan32(const u8 *from, const u8 *s, __m128i set1, __m128i set2, size_t count, int members) {
    __m128i low, high;
    truffle_tables32(set1, set2, count, members, &low, &high);
    return truffle_scan(from, s, low, high, members);
}

static NOINLINE size_t truffle_scan_long(const u8 *from, const u8 *s, const u8 *set, int members) {
    __m128i low, high;
    truffle_tables(set, members, &low, &high);
    return truffle_scan(from, s, low, high, members);
}

#define PCMPISTRI_BLOCKS 128    /* Intel: 2 KiB */
#define PCMPISTRI_BLOCKS_ZEN 64 /* Zen 5: 1 KiB */
#define PCMPISTRM_PAIR_BLOCKS 4 /* 64 bytes */

/* A set of up to 16 bytes: pcmpistri, one load of the set and no tables,
   over `blocks` 16-byte blocks of text; a set of 17..32 bytes: two pcmpistrm
   per block over the first PCMPISTRM_PAIR_BLOCKS (a parser's token); the
   shuffle scan with the tables built from the registers beyond, and the
   scalar table build for a longer set. pcmpistri runs a block per cycle or
   so on Intel and the shuffle scan 32 bytes in 1.6 cycles, but the tables
   cost some 40 cycles; a pcmpistrm pair costs some 13 cycles per block, so
   the tables win from the fourth block on. Inlined into the four entries: `members` and `blocks`
   are constants there, so the loops carry no branch on them. */
ALWAYS_INLINE size_t set_scan(const char *str, const char *setstr, int members, int blocks) {
    const u8 *s = (const u8 *)str, *set = (const u8 *)setstr;
    __m128i set1, set2;
    const u8 *from = s;
    size_t valid;

    if (load_set16(set, &set1)) {
        __m128i x = load_head16(s, &valid);
        unsigned i = members ? cspn_index(set1, x) : (unsigned)_mm_cmpistri(set1, x, SPN_MODE);
        unsigned z = zero_mask16(set1);

        if (i < valid) {
            return i;
        }

        from = after_head(s, valid);

        for (int k = 0; k < blocks; k++, from += 16) {
            x = load16a(from);

            /* the flags of the same pcmpistri: a byte to stop at (CF) or, for
               strcspn, the NUL (ZF) */

            if (members ? !_mm_cmpistra(set1, x, CSPN_MODE) : _mm_cmpistrc(set1, x, SPN_MODE)) {
                return (size_t)(from - s) + (members ? cspn_stop(set1, x) : (unsigned)_mm_cmpistri(set1, x, SPN_MODE));
            }
        }

        return truffle_scan32(from, s, set1, _mm_setzero_si128(), z ? ctz(z) : 16, members);
    }

    if (load_set_second16(set, &set2)) {
        set1 = load16(set); /* 16 non-NUL bytes: page-safe, see load_set16 */
        __m128i x = load_head16(s, &valid);
        unsigned i = members ? cspn2_index(set1, set2, x) : ctz(spn2_misses(set1, set2, x) | 1u << 16);

        if (i < valid) {
            return i;
        }

        from = after_head(s, valid);

        for (int k = 0; k < PCMPISTRM_PAIR_BLOCKS; k++, from += 16) {
            x = load16a(from);
            i = members ? cspn2_index(set1, set2, x) : ctz(spn2_misses(set1, set2, x) | 1u << 16);

            if (i < 16) {
                return (size_t)(from - s) + i;
            }
        }

        unsigned z = zero_mask16(set2);
        return truffle_scan32(from, s, set1, set2, 16 + (z ? ctz(z) : 16), members);
    }

    return truffle_scan_long(from, s, set, members);
}

size_t strcspn_avx2(const char *str, const char *reject) {
    if (!reject[0]) {
        return str_strlen(str);
    }

    return set_scan(str, reject, 1, PCMPISTRI_BLOCKS);
}

size_t strspn_avx2(const char *str, const char *accept) {
    if (!accept[0]) {
        return 0;
    }

    return set_scan(str, accept, 0, PCMPISTRI_BLOCKS);
}

size_t strcspn_avx2_zen(const char *str, const char *reject) {
    if (!reject[0]) {
        return str_strlen(str);
    }

    return set_scan(str, reject, 1, PCMPISTRI_BLOCKS_ZEN);
}

size_t strspn_avx2_zen(const char *str, const char *accept) {
    if (!accept[0]) {
        return 0;
    }

    return set_scan(str, accept, 0, PCMPISTRI_BLOCKS_ZEN);
}

/* ------------------------------------------------------------ NUL scans */

/* a is 16-aligned, s <= a, no NUL in [s, a). The 32-byte block around a is
   checked with the bytes before a masked off (they may lie before s), then
   three single blocks: with the caller's block, strings of up to 128 bytes
   end in singles. Then four blocks per step from the 128-byte boundary below
   (a 128-aligned group never crosses a page; what it scans again is known
   nonzero), joined by a min tree: the minimum of four blocks is zero exactly
   where one has a NUL, so a group costs one compare and one movemask. The
   blocks are loaded again only in the group with the NUL, which lets the
   compiler fold the loads of the others into the mins. */
size_t strlen_tail_avx2(const u8 *a, const u8 *s) {
    const u8 *start = align_down(a, 32);
    unsigned m = mask256(eq_zero(load256a(start))) >> (a - start);

    if (m) {
        return (size_t)(a - s) + ctz(m);
    }

    for (int k = 1; k < 4; k++) {
        m = mask256(eq_zero(load256a(start + 32 * k)));

        if (m) {
            return (size_t)(start + 32 * k - s) + ctz(m);
        }
    }

    a = align_down(start + 128, 128); /* at least start + 32 > s */

    for (;;) {
        __m256i lo = _mm256_min_epu8(load256a(a), load256a(a + 32));
        __m256i hi = _mm256_min_epu8(load256a(a + 64), load256a(a + 96));
        m = mask256(eq_zero(_mm256_min_epu8(lo, hi)));

        if (m) {
            for (int k = 0; k < 3; k++) {
                unsigned mk = mask256(eq_zero(load256a(a + 32 * k)));

                if (mk) {
                    return (size_t)(a + 32 * k - s) + ctz(mk);
                }
            }

            return (size_t)(a - s) + 96 + ctz(m);
        }

        a += 128;
    }
}

/* One 32-byte block at s when the page allows (all 32 bytes count: 9 strings
   in 10 end there, median 8 bytes); else the aligned block around s, which
   never crosses a page, with the bytes before s shifted out. Either way the
   next aligned block is where the tail continues. */
size_t strlen_avx2(const char *str) {
    const u8 *s = (const u8 *)str;
    unsigned m;

    if (page_left(s) >= 32) {
        m = mask256(eq_zero(load256(s)));
    } else {
        m = mask256(eq_zero(load256a(align_down(s, 32)))) >> ((uintptr_t)s & 31);
    }

    if (m) {
        return ctz(m);
    }

    /* three aligned blocks in line (strings of up to 128 bytes end here; the
       tail's call and shift cost 1.1-1.6x on them), the tail beyond */
    const u8 *a = align_down(s + 32, 32);

    for (int k = 0; k < 3; k++, a += 32) {
        m = mask256(eq_zero(load256a(a)));

        if (m) {
            return (size_t)(a - s) + ctz(m);
        }
    }

    return str_strlen_tail(a, s);
}

/* ---------------------------------------------------------------- memchr */

/* n > 64, the first 32 bytes without a match: aligned blocks from the
   32-byte boundary after them (what the first of them scans again is known;
   both sides unaligned would run 2x slower on Broadwell), four per step
   joined by or, then single blocks, then the last 32 bytes as one block
   that may overlap what was scanned. */
void *memchr_large_avx2(const u8 *s, u8 c, size_t n) {
    if (n > PTRDIFF_MAX) {
        n = PTRDIFF_MAX; /* memchr(p, c, SIZE_MAX): C11 lets the object end at the match, so n may pass the address space; s + n must not wrap */
    }

    __m256i v = _mm256_set1_epi8((char)c);
    const u8 *end = s + n, *a = align_down(s + 32, 32);

    /* C11 lets the object end at the match too: no load may reach a page past
       the one that holds the first byte not yet checked. 32-byte blocks up to
       a 128-byte boundary, then whole aligned 128-byte groups, each in one
       page (unaligned groups read up to 96 bytes into the next page: a fault
       when it is not there, dev/proof found it) */

    while (((uintptr_t)a & 127) && end - a >= 32) {
        unsigned m = mask256(_mm256_cmpeq_epi8(load256a(a), v));

        if (m) {
            return (void *)(a + ctz(m));
        }

        a += 32;
    }

    while (end - a >= 128) {
        __m256i e0 = _mm256_cmpeq_epi8(load256a(a), v), e1 = _mm256_cmpeq_epi8(load256a(a + 32), v);
        __m256i e2 = _mm256_cmpeq_epi8(load256a(a + 64), v), e3 = _mm256_cmpeq_epi8(load256a(a + 96), v);

        if (mask256(_mm256_or_si256(_mm256_or_si256(e0, e1), _mm256_or_si256(e2, e3)))) {
            unsigned m = mask256(e0);

            if (m) {
                return (void *)(a + ctz(m));
            }

            m = mask256(e1);

            if (m) {
                return (void *)(a + 32 + ctz(m));
            }

            m = mask256(e2);

            if (m) {
                return (void *)(a + 64 + ctz(m));
            }

            return (void *)(a + 96 + ctz(mask256(e3)));
        }

        a += 128;
    }

    while (end - a >= 32) {
        unsigned m = mask256(_mm256_cmpeq_epi8(load256a(a), v));

        if (m) {
            return (void *)(a + ctz(m));
        }

        a += 32;
    }

    if (a < end) {
        unsigned m = mask256(_mm256_cmpeq_epi8(load256(end - 32), v)) >> (32 - (end - a));

        if (m) {
            return (void *)(a + ctz(m));
        }
    }

    return 0;
}

/* One 32-byte block first, whatever n: unaligned when the page allows, and
   the index of the first match (32 for none) against n, which settles
   everything up to 32 bytes. Near the page end the aligned block around s,
   shifted, settles what the page holds; the rest, if n reaches into the
   next page, is its first aligned block or the kernel. Then the last 32
   bytes for 33..64, the kernel beyond. */
void *memchr_avx2(const void *p, int c, size_t n) {
    const u8 *s = (const u8 *)p;
    __m256i v = _mm256_set1_epi8((char)c);
    unsigned m;

    if (n == 0) {
        return 0; /* the pointer may be unreadable */
    }

    size_t left = page_left(s);
    const u8 *a = align_down(s, 32) + 32; /* the first aligned block not yet checked */

    if (left >= 32) {
        m = mask256(_mm256_cmpeq_epi8(load256(s), v));

        size_t i = _tzcnt_u32(m); /* 32 for no match: no branch on it, half the calls of a mix have none and would mispredict */

        if (n <= 32) {
            return i < n ? (void *)(s + i) : 0;
        }

        if (i < 32) {
            return (void *)(s + i);
        }
    } else {
        m = mask256(_mm256_cmpeq_epi8(load256a(align_down(s, 32)), v)) >> ((uintptr_t)s & 31);

        if (m) {
            size_t i = ctz(m);
            return i < n ? (void *)(s + i) : 0;
        }

        if (n <= left) {
            return 0;
        }

        /* the next page's first aligned block */
        m = mask256(_mm256_cmpeq_epi8(load256a(s + left), v));

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
        m = mask256(_mm256_cmpeq_epi8(load256a(a), v));

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

/* ------------------------------------------------------------------ copy */

/* n > 64. Head and tail vectors are loaded first and stored last, so an
   aligned loop may clobber source bytes only after they were consumed. */
/* The forward loop, aligned 128-byte steps, for 257..4095 bytes without backward
   overlap; the first and the last 32 bytes are the caller's. */
static inline void copy_forward_loop(u8 *d, const u8 *s, size_t n) {
    size_t skew = 32 - ((uintptr_t)d & 31);
    u8 *dp = d + skew;
    const u8 *sp = s + skew;
    size_t left = n - skew; /* at least 225 */
    /* the last 128 bytes loaded first (with a forward overlap the loop's
       stores reach them) and stored after the loop from the end down: no
       32-byte steps after it, whose count of 0..3 the sizes mispredict
       (glibc's way; Broadwell, ours with glibc's memcpy ran the mixed trace
       2.3% faster, 0.1 mispredictions a call of it fewer) */
    __m256i t0 = load256(s + n - 128), t1 = load256(s + n - 96), t2 = load256(s + n - 64), t3 = load256(s + n - 32);
    /* the next group is loaded before this one is stored: a load right after
       a store to the same offset in another page waits for the store (4K
       aliasing), and source and destination at the same offset are common.
       Two register sets by turns, so that no copies are needed. */
    __m256i x0 = load256(sp), x1 = load256(sp + 32), x2 = load256(sp + 64), x3 = load256(sp + 96);

    for (;;) {
        if (left < 256) {
            break;
        }

        __m256i y0 = load256(sp + 128), y1 = load256(sp + 160), y2 = load256(sp + 192), y3 = load256(sp + 224);
        store256a(dp, x0);
        store256a(dp + 32, x1);
        store256a(dp + 64, x2);
        store256a(dp + 96, x3);
        dp += 128;
        sp += 128;
        left -= 128;

        if (left < 256) {
            x0 = y0;
            x1 = y1;
            x2 = y2;
            x3 = y3;
            break;
        }

        x0 = load256(sp + 128);
        x1 = load256(sp + 160);
        x2 = load256(sp + 192);
        x3 = load256(sp + 224);
        store256a(dp, y0);
        store256a(dp + 32, y1);
        store256a(dp + 64, y2);
        store256a(dp + 96, y3);
        dp += 128;
        sp += 128;
        left -= 128;
    }

    store256a(dp, x0);
    store256a(dp + 32, x1);
    store256a(dp + 64, x2);
    store256a(dp + 96, x3);
    store256(d + n - 32, t3); /* less than 128 bytes after dp: within these */
    store256(d + n - 64, t2);
    store256(d + n - 96, t1);
    store256(d + n - 128, t0);
}

/* n > 256: its own function. The loops want a register beyond the
   caller-saved ones (rbx), and the push and pop it costs at the entry sat on
   every copy of 65..256 bytes: on Broadwell 1.27x glibc on the trace's
   129..256-byte calls, where the stores queue for lines in L3 and a store of
   the prologue waits behind them. */
static NOINLINE void *copy_loop_avx2(u8 *d, const u8 *s, size_t n) {
    if (overlap_backward(d, s, n)) {
        /* down from the aligned end; the first 128 bytes and the last 32
           loaded first and stored after the loop, as the forward one */
        __m256i h0 = load256(s), h1 = load256(s + 32), h2 = load256(s + 64), h3 = load256(s + 96), tail = load256(s + n - 32);
        size_t off = n - ((uintptr_t)(d + n) & 31);

        while (off > 128) {
            off -= 128;
            __m256i x0 = load256(s + off), x1 = load256(s + off + 32), x2 = load256(s + off + 64), x3 = load256(s + off + 96);
            store256a(d + off, x0);
            store256a(d + off + 32, x1);
            store256a(d + off + 64, x2);
            store256a(d + off + 96, x3);
        }

        store256(d + n - 32, tail);
        store256(d + 96, h3);
        store256(d + 64, h2);
        store256(d + 32, h1);
        store256(d, h0);
        RETURN_OPAQUE(d);
    }

    if (n >= REP_THRESHOLD) {
        return str_copy_huge(d, s, n);
    }

    __m256i head = load256(s);
    copy_forward_loop(d, s, n);
    store256(d, head);
    RETURN_OPAQUE(d);
}

void *copy_large_avx2(u8 *d, const u8 *s, size_t n) {
    if (n <= 128) {
        __m256i x0 = load256(s), x1 = load256(s + 32), y0 = load256(s + n - 64), y1 = load256(s + n - 32);
        store256(d, x0);
        store256(d + 32, x1);
        store256(d + n - 32, y1);
        store256(d + n - 64, y0);
        RETURN_OPAQUE(d);
    }

    if (n <= 256) {
        __m256i x0 = load256(s), x1 = load256(s + 32), x2 = load256(s + 64), x3 = load256(s + 96);
        __m256i y0 = load256(s + n - 128), y1 = load256(s + n - 96), y2 = load256(s + n - 64), y3 = load256(s + n - 32);
        store256(d, x0);
        store256(d + 32, x1);
        store256(d + 64, x2);
        store256(d + 96, x3);
        store256(d + n - 32, y3); /* the tail from the end down: up, 1.25x slower on Broadwell (lines in L3) */
        store256(d + n - 64, y2);
        store256(d + n - 96, y1);
        store256(d + n - 128, y0);
        RETURN_OPAQUE(d);
    }

    return copy_loop_avx2(d, s, n);
}

static inline void *copy(void *dst, const void *src, size_t n) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;

    if (n <= 32) {
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

void *memcpy_avx2(void *dst, const void *src, size_t n) {
    return copy(dst, src, n);
}

void *memmove_avx2(void *dst, const void *src, size_t n) {
    return copy(dst, src, n);
}

/* n >= REP_THRESHOLD, no backward overlap: the loop up to COPY_LOOP_LIMIT,
   rep movsb beyond. rep movsb with the source and the destination at
   different offsets in their lines runs 5x slower than the loop at 4..8 KiB
   on Broadwell and no faster at 16; on Zen 5 it is 1.5x slower up to 16 KiB
   and ahead from 32. Skylake-SP is even at 4 KiB and ahead from 8: it keeps
   copy_huge_rep. */
#define COPY_LOOP_LIMIT 32768

void *copy_huge_avx2(u8 *d, const u8 *s, size_t n) {
    if (n >= COPY_LOOP_LIMIT) {
        rep_movsb(d, s, n);
        RETURN_OPAQUE(d);
    }

    __m256i head = load256(s);
    copy_forward_loop(d, s, n);
    store256(d, head);
    RETURN_OPAQUE(d);
}

/* ------------------------------------------------------------------ fill */

static inline void *fill_loop(u8 *d, __m256i v, size_t n) {
    /* four unaligned stores at the start, aligned 128-byte steps, four
       unaligned ending at n written from the end down. Broadwell (lines in
       L3), memset trace at 129..256 / 257..512 bytes: the last four up (as
       glibc) 33.8 / 36.8 ns, down 26.3 / 29.8; a frame's push and pop around
       the loop had made it 55 at 257..512. One unaligned store at each end
       and 0..3 aligned 32-byte ones after the loop (LLVM's, the fewest
       stores) is a little faster still where the stores queue (the memset
       trace, 18.85 ns a call against 19.08), and slower in the mix (12.72 ns
       a call against 12.50), where the queue drains between calls and the
       branches on the 0..3 mispredict. The loop is not unrolled (the
       compiler would, with a long prologue for a count that is usually
       small). */
    store256(d, v);
    store256(d + 32, v);
    store256(d + 64, v);
    store256(d + 96, v);
    u8 *end = d + n - 128;
    u8 *dp = (u8 *)align_down(d + 128, 32);

    NO_UNROLL
    while (dp < end) {
        store256a(dp, v);
        store256a(dp + 32, v);
        store256a(dp + 64, v);
        store256a(dp + 96, v);
        dp += 128;
    }

    store256(end + 96, v);
    store256(end + 64, v);
    store256(end + 32, v);
    store256(end, v);
    RETURN_OPAQUE(d);
}

/* n > 128: the aligned stores (eight unaligned ones for 129..256,
   overlapping and crossing lines, measured slower). */
void *fill_large_avx2(u8 *d, int c, size_t n) {
    __m256i v = _mm256_set1_epi8((char)c);

    if (n >= REP_THRESHOLD) {
        return str_fill_huge(d, c, n);
    }

    return fill_loop(d, v, n);
}

/* n <= 32 with 16-byte and smaller stores, no upper halves touched; 33..64
   two 32-byte stores (four 16-byte ones measured 1.1-1.2x slower on
   Skylake-SP and Zen 5), 65..128 four, in line: through the kernel they
   cost 1.1-1.3x. */
void *memset_avx2(void *dst, int c, size_t n) {
    u8 *d = (u8 *)dst;

    if (n <= 32) {
        fill_small(d, splat16(c), c, n);
    } else if (n <= 128) {
        __m256i v = _mm256_set1_epi8((char)c);
        store256(d, v);
        store256(d + n - 32, v);

        if (n > 64) {
            store256(d + 32, v);
            store256(d + n - 64, v);
        }
    } else {
        return str_fill_large(d, c, n);
    }

    return dst;
}

/* --------------------------------------------------------------- compare */

/* Two 32-byte blocks at o1 < o2 (they may overlap): 1 with the answer in
   *result when they differ, else 0. */
static inline int pair_differs(const u8 *a, const u8 *b, size_t o1, size_t o2, int *result) {
    __m256i e0 = _mm256_cmpeq_epi8(load256(a + o1), load256(b + o1));
    __m256i e1 = _mm256_cmpeq_epi8(load256(a + o2), load256(b + o2));

    if (mask256(_mm256_and_si256(e0, e1)) == 0xffffffffu) {
        return 0;
    }

    unsigned m = mask256(e0) ^ 0xffffffffu;
    *result = m ? byte_diff(a, b, o1 + ctz(m)) : byte_diff(a, b, o2 + ctz(mask256(e1) ^ 0xffffffffu));
    return 1;
}

/* n > 128, the first 64 bytes equal (the entry compared them: a difference
   is usually early, and this way it costs one compare, not a tail call).
   Groups of four blocks joined by and, loaded aligned on a (the bytes of the
   second block it loads again are equal), then the last 33..128 bytes as two
   pairs that may overlap what was compared: bytes before a block's start
   were already verified, so the first difference is found in address order. */
int memcmp_large_avx2(const u8 *a, const u8 *b, size_t n) {
    int result;
    size_t i = 64 - ((uintptr_t)(a + 64) & 31); /* a + i is 32-aligned, 33 <= i <= 64 */

    while (n - i > 128) {
        __m256i e0 = _mm256_cmpeq_epi8(load256a(a + i), load256(b + i)), e1 = _mm256_cmpeq_epi8(load256a(a + i + 32), load256(b + i + 32));
        __m256i e2 = _mm256_cmpeq_epi8(load256a(a + i + 64), load256(b + i + 64)), e3 = _mm256_cmpeq_epi8(load256a(a + i + 96), load256(b + i + 96));
        unsigned m = mask256(_mm256_and_si256(_mm256_and_si256(e0, e1), _mm256_and_si256(e2, e3))) + 1;

        if (m) {
            unsigned m0 = mask256(e0) + 1;

            if (m0) {
                return byte_diff(a, b, i + ctz(m0));
            }

            m0 = mask256(e1) + 1;

            if (m0) {
                return byte_diff(a, b, i + 32 + ctz(m0));
            }

            m0 = mask256(e2) + 1;

            if (m0) {
                return byte_diff(a, b, i + 64 + ctz(m0));
            }

            return byte_diff(a, b, i + 96 + ctz(m));
        }

        i += 128;
    }

    /* 1..128 bytes are left in [i, n) */
    if (n - i > 64 && pair_differs(a, b, i, i + 32, &result)) {
        return result;
    }

    if (pair_differs(a, b, n - 64, n - 32, &result)) {
        return result;
    }

    return 0;
}

/* n <= 32 in one 32-byte compare when both pointers have the room in their
   pages: eq + 1 sets the bit of the first differing lane (all equal wraps to
   0), bzhi drops the lanes at n and beyond. No split at 16, no test for 0.
   Longer: the first block, then the last 32 bytes for 33..64, the second
   block and the last 64 bytes for 65..128, the kernel beyond. */
int memcmp_avx2(const void *pa, const void *pb, size_t n) {
    const u8 *a = (const u8 *)pa, *b = (const u8 *)pb;

    if (n - 1 < 32) { /* 1..32: with n == 0 the pointers may be unreadable (the end of a buffer) */

        if (__builtin_expect(both_in_page(a, b, 32), 1)) {
            return memcmp_answer(a, b, ~mask256(_mm256_cmpeq_epi8(load256(a), load256(b))), n);
        }

        return n <= 16 ? memcmp_le16(a, b, n) : memcmp_17_32(a, b, n);
    }

    if (n > 32) {
        /* the answer from all the blocks' masks at once, no branch on the data
           (memcmp_answer); a branch only for the kernel beyond 128 */
        uint64_t lo = (uint32_t)~mask256(_mm256_cmpeq_epi8(load256(a), load256(b)));

        if (n <= 64) {
            lo |= (uint64_t)(uint32_t)~mask256(_mm256_cmpeq_epi8(load256(a + n - 32), load256(b + n - 32))) << (n - 32);
            return memcmp_answer(a, b, lo, n);
        }

        lo |= (uint64_t)(uint32_t)~mask256(_mm256_cmpeq_epi8(load256(a + 32), load256(b + 32))) << 32;

        if (n <= 128) {
            uint64_t hi = (uint32_t)~mask256(_mm256_cmpeq_epi8(load256(a + n - 64), load256(b + n - 64))) |
                          (uint64_t)(uint32_t)~mask256(_mm256_cmpeq_epi8(load256(a + n - 32), load256(b + n - 32))) << 32;
            return memcmp_answer2(a, b, lo, hi, n);
        }

        if (lo) {
            return byte_diff(a, b, (size_t)__builtin_ctzll(lo));
        }

        return str_memcmp_large(a, b, n);
    }

    return 0; /* n == 0 */
}

/* Bit i set when a[i] and b[i] are equal and a[i] is not NUL. */
static inline __m256i same_and_nonzero(__m256i x, __m256i y) {
    return _mm256_andnot_si256(eq_zero(x), _mm256_cmpeq_epi8(x, y));
}

/* x & eq(x, y) is nonzero exactly in the lanes that are equal and not NUL,
   so the min of four such blocks is zero where a lane differs or ends: two
   operations per block, one compare per group. */
static inline __m256i same_nonzero_value(__m256i x, __m256i y) {
    return _mm256_and_si256(x, _mm256_cmpeq_epi8(x, y));
}

/* Four blocks per step (the entry has done 128 bytes, so a string here is
   longer and a group is the first step), as many steps as both
   pointers have room for in their pages, counted once per page; single
   blocks near a page end, words and bytes at the very end. The tests are
   exact, not both_in_page: on its false miss the walk to the page end would
   run to the nearer one, up to 2 KiB. The loads are written unaligned, but
   the avx2 entry hands over a on a 32-byte boundary and the steps keep it
   there: both sides unaligned run 2x slower on Broadwell, one side 1.2x. */
int strcmp_tail_avx2(const u8 *a, const u8 *b) {
    int result;

    for (;;) {
        size_t left = page_left(a) < page_left(b) ? page_left(a) : page_left(b);

        if (left >= 128) {
            size_t groups = left / 128;

            do {
                __m256i s0 = same_nonzero_value(load256(a), load256(b)), s1 = same_nonzero_value(load256(a + 32), load256(b + 32));
                __m256i s2 = same_nonzero_value(load256(a + 64), load256(b + 64)), s3 = same_nonzero_value(load256(a + 96), load256(b + 96));
                /* opaque: else the compiler rewrites the four and + min into the or
                   of four negated compares and a min of the raw blocks, 13 vector
                   operations per group instead of 8 (1.15x on Broadwell) */
                __asm__("" : "+x"(s0), "+x"(s1), "+x"(s2), "+x"(s3));
                unsigned m = mask256(eq_zero(_mm256_min_epu8(_mm256_min_epu8(s0, s1), _mm256_min_epu8(s2, s3))));

                if (m) {
                    unsigned m0 = mask256(eq_zero(s0));

                    if (m0) {
                        return byte_diff(a, b, ctz(m0));
                    }

                    m0 = mask256(eq_zero(s1));

                    if (m0) {
                        return byte_diff(a, b, 32 + ctz(m0));
                    }

                    m0 = mask256(eq_zero(s2));

                    if (m0) {
                        return byte_diff(a, b, 64 + ctz(m0));
                    }

                    return byte_diff(a, b, 96 + ctz(m));
                }

                a += 128;
                b += 128;
            } while (--groups);
        } else if (both_have(a, b, 32)) {
            unsigned m = mask256(same_and_nonzero(load256(a), load256(b))) ^ 0xffffffffu;

            if (m) {
                return byte_diff(a, b, ctz(m));
            }

            a += 32;
            b += 32;
        } else if (strcmp_to_page_end(&a, &b, &result)) {
            return result;
        }
    }
}

/* One 32-byte block when both strings have the room in their pages: the
   lanes that are equal and not NUL, + 1, and the trailing zero count give
   the first differing or terminating byte (all 32 wrap to 0). Three more
   blocks for the strings equal so far; the tail for the few longer still. */
int strcmp_avx2(const char *left, const char *right) {
    const u8 *a = (const u8 *)left, *b = (const u8 *)right;

    if (__builtin_expect(both_in_page(a, b, 32), 1)) {
        unsigned m = mask256(same_and_nonzero(load256(a), load256(b))) + 1;

        if (__builtin_expect(m != 0, 1)) {
            return byte_diff(a, b, ctz(m));
        }

        /* three more blocks at fixed offsets from the start, one page test for
           all of them: strings of up to 128 bytes end here. Single blocks
           overlap under speculation; a group joined by a tree costs its whole
           latency on the hit (2x on 65..127 bytes), and blocks moved to the
           32-byte boundary first wait for that arithmetic. */

        if (both_in_page(a, b, 128)) {
            m = mask256(same_and_nonzero(load256(a + 32), load256(b + 32))) + 1;

            if (m) {
                return byte_diff(a, b, 32 + ctz(m));
            }

            m = mask256(same_and_nonzero(load256(a + 64), load256(b + 64))) + 1;

            if (m) {
                return byte_diff(a, b, 64 + ctz(m));
            }

            m = mask256(same_and_nonzero(load256(a + 96), load256(b + 96))) + 1;

            if (m) {
                return byte_diff(a, b, 96 + ctz(m));
            }

            /* the tail from the 32-byte boundary before a + 128: up to 31 bytes
               compared again, a aligned for its groups (both sides unaligned
               run 2x slower on Broadwell, one side 1.2x) */
            const u8 *next = align_down(a + 128, 32);
            b += next - a;
            a = next;
        } else {
            a += 32;
            b += 32;
        }
    } else {
        int result;

        if (strcmp_block16(&a, &b, &result)) {
            return result;
        }
    }

    return str_strcmp_tail(a, b);
}

/* ---------------------------------------------------------- strcpy/strcat */

/* One pass for a string with no NUL in [s, a), a 16-aligned and at most 128
   past s (the caller checked that much). The aligned 32-byte block after the known bytes
   first, then single blocks to a 128-byte boundary, then four per step (a
   128-aligned group never crosses a page) joined by a min tree, each block
   tested before it is stored; the last 32 bytes of the string go out as one
   overlapping store. */
void *strcpy_long_avx2(u8 *d, const u8 *s, const u8 *a) {
    if ((uintptr_t)a & 31) {
        /* reach a 32-byte boundary: check the 16 bytes in between */
        unsigned m16 = zero_mask16(load16a(a));

        if (m16) {
            copy_small(d, s, (size_t)(a - s) + ctz(m16) + 1); /* at most 32 bytes */
            RETURN_OPAQUE(d);
        }

        a += 16;
    }

    __m256i x = load256a(a);
    unsigned m = mask256(eq_zero(x));

    if (m) {
        size_t n = (size_t)(a - s) + ctz(m) + 1; /* at most 160 bytes: a is at most 128 past s */

        if (n <= 32) {
            copy_small(d, s, n);
        } else {
            /* whole blocks while another fits before the end, then the last 32 bytes */
            size_t off = 0;

            for (; off + 32 < n; off += 32) {
                store256(d + off, load256(s + off));
            }

            store256(d + n - 32, load256(s + n - 32));
        }

        RETURN_OPAQUE(d);
    }

    /* the known bytes [s, a) in whole blocks; the last may reach into the
       block at a, which has no NUL */
    size_t off = 0;

    for (; off < (size_t)(a - s); off += 32) {
        store256(d + off, load256(s + off));
    }

    off = (size_t)(a - s);

    for (;;) {
        /* x is the block at a and has no NUL: it goes out, the next is tested before */
        store256(d + off, x);
        a += 32;
        off += 32;

        if (!((uintptr_t)a & 127)) {
            break;
        }

        x = load256a(a);
        m = mask256(eq_zero(x));

        if (m) {
            goto last;
        }
    }

    /* groups of four: the next group is loaded and tested before this one is
       stored (a load right after a store to the same offset in another
       page waits for the store, 4K aliasing; the string's source and
       destination at the same offset are common), two register sets by
       turns so that no copies are needed */
    __m256i x0 = load256a(a), x1 = load256a(a + 32), x2 = load256a(a + 64), x3 = load256a(a + 96);
    m = mask256(eq_zero(_mm256_min_epu8(_mm256_min_epu8(x0, x1), _mm256_min_epu8(x2, x3))));

    while (!m) {
        __m256i y0 = load256a(a + 128), y1 = load256a(a + 160), y2 = load256a(a + 192), y3 = load256a(a + 224);
        m = mask256(eq_zero(_mm256_min_epu8(_mm256_min_epu8(y0, y1), _mm256_min_epu8(y2, y3))));
        store256(d + off, x0);
        store256(d + off + 32, x1);
        store256(d + off + 64, x2);
        store256(d + off + 96, x3);
        a += 128;
        off += 128;

        if (m) {
            x0 = y0;
            x1 = y1;
            x2 = y2;
            x3 = y3;
            break;
        }

        x0 = load256a(a + 128);
        x1 = load256a(a + 160);
        x2 = load256a(a + 192);
        x3 = load256a(a + 224);
        m = mask256(eq_zero(_mm256_min_epu8(_mm256_min_epu8(x0, x1), _mm256_min_epu8(x2, x3))));
        store256(d + off, y0);
        store256(d + off + 32, y1);
        store256(d + off + 64, y2);
        store256(d + off + 96, y3);
        a += 128;
        off += 128;
    }

    /* the group at a has the NUL: the blocks before it go out */
    {
        unsigned m0 = mask256(eq_zero(x0));

        if (m0) {
            m = m0;
            goto last;
        }

        store256(d + off, x0);
        a += 32;
        off += 32;
        m0 = mask256(eq_zero(x1));

        if (m0) {
            m = m0;
            goto last;
        }

        store256(d + off, x1);
        a += 32;
        off += 32;
        m0 = mask256(eq_zero(x2));

        if (m0) {
            m = m0;
            goto last;
        }

        store256(d + off, x2);
        a += 32;
        off += 32;
    }

last:;
    const u8 *nul = a + ctz(m);
    store256(d + (nul - s) - 31, load256(nul - 31)); /* the string is longer than 32 here */
    RETURN_OPAQUE(d);
}

static NOINLINE void *strcpy_long(u8 *d, const u8 *s, const u8 *a) {
    return str_strcpy_long(d, s, a);
}

/* strcpy up to 128 bytes past the 32-byte boundary after s: 0 when done,
   else the boundary a with no NUL in [s, a), for strcpy_long. The first 32
   bytes as one block (copy_small when the NUL is in them); then aligned
   blocks, each stored at its offset in d as soon as it is found free of the
   NUL (whole, so the string's bytes only), and the block with the NUL by
   the 32 bytes ending at it: no copy loop after the search, whose count of
   blocks the lengths mispredict (glibc's way; Broadwell, the mix: ours
   with glibc's strcpy 1.3 ns a call of it faster before). Strings of up to
   128 bytes end here (through strcpy_long they cost 1.4-1.8x). */
ALWAYS_INLINE const u8 *strcpy_short(u8 *d, const u8 *s) {
    unsigned m;

    if (page_left(s) >= 32) {
        m = mask256(eq_zero(load256(s)));
    } else {
        m = mask256(eq_zero(load256a(align_down(s, 32)))) >> ((uintptr_t)s & 31);
    }

    if (m) {
        copy_small(d, s, ctz(m) + 1);
        return 0;
    }

    const u8 *a = align_down(s, 32) + 32;
    __m256i y = load256a(a);
    m = mask256(eq_zero(y));

    if (m) {
        size_t n = (size_t)(a - s) + ctz(m) + 1; /* 2..64 */

        if (n <= 32) {
            copy_small(d, s, n);
        } else {
            __m256i head = load256(s), tail = load256(s + n - 32); /* [s, s+32) is readable: the string reaches a */
            store256(d, head);
            store256(d + n - 32, tail);
        }

        return 0;
    }

    store256(d, load256(s)); /* [s, a + 32) has no NUL */
    store256(d + (a - s), y);

    for (int k = 0; k < 2; k++) {
        a += 32;
        y = load256a(a);
        m = mask256(eq_zero(y));

        if (m) {
            size_t n = (size_t)(a - s) + ctz(m) + 1; /* 34..128: the blocks before are stored */
            store256(d + n - 32, load256(s + n - 32));
            return 0;
        }

        store256(d + (a - s), y);
    }

    return a + 32;
}

char *strcpy_avx2(char *dst, const char *src) {
    const u8 *a = strcpy_short((u8 *)dst, (const u8 *)src);

    if (a) {
        return strcpy_long((u8 *)dst, (const u8 *)src, a);
    }

    return dst;
}

/* Out of line, so that the common case below keeps no frame and no call. */
static NOINLINE char *strcat_long(char *dst, const char *src) {
    str_strcpy(dst + str_strlen(dst), src);
    RETURN_OPAQUE(dst);
}

/* src from e on past the 128 bytes strcpy_short copies; strcat's answer: dst */
static NOINLINE char *strcat_copy(char *dst, u8 *e, const char *src, const u8 *a) {
    str_strcpy_long(e, (const u8 *)src, a);
    RETURN_OPAQUE(dst);
}

/* dst's end by aligned blocks: the one holding dst with the bytes before it
   shifted out (no page test), three more in line, the call to strlen only
   beyond; then strcpy's short path from there. Calls (and the registers
   they save: three pushes, the stores queued behind the copy's) only for a
   dst of 97 bytes or more or a src past strcpy_short: up to 1.12x glibc on
   Broadwell with strlen and strcpy by calls for every dst past 31 bytes. */
char *strcat_avx2(char *dst, const char *src) {
    const u8 *d = (const u8 *)dst;
    const u8 *a = align_down(d, 32);
    unsigned m = mask256(eq_zero(load256a(a))) >> ((uintptr_t)d & 31);
    u8 *e;

    if (m) {
        e = (u8 *)dst + ctz(m);
    } else {
        for (int k = 0; k < 3 && !m; k++) {
            a += 32;
            m = mask256(eq_zero(load256a(a)));
        }

        if (!m) {
            return strcat_long(dst, src);
        }

        e = (u8 *)dst + (a - d) + ctz(m);
    }

    const u8 *rest = strcpy_short(e, (const u8 *)src);

    if (rest) {
        return strcat_copy(dst, e, src, rest);
    }

    return dst;
}
