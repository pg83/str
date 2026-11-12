/*
 * derived.c: the rest of string.h and strings.h that is made of the
 * primitives: every function here reaches memory only through the str_*
 * entry points (so through the table: any configuration, any mix of its
 * slots) and its own few loads and stores, and calls nothing of a C library:
 * strdup and strndup, which would need malloc, are left to it, so the
 * archive takes no name from outside. Compiled once, for baseline x86-64:
 * the work is the primitives'.
 *
 * memmem and strstr scan for the needle's first byte and compare at each
 * place it is found; when that does much more work than it advances (the
 * first byte everywhere, long partial matches) they go on by the two-way
 * algorithm, linear in the haystack whatever the needle. strstr with a
 * needle under 16 bytes scans 16 bytes of the haystack at a time itself
 * (SSE2, the baseline: this unit's own loads, in their pages).
 */
#include "common.h"

int str_bcmp(const void *a, const void *b, size_t n) {
    return str_memcmp(a, b, n);
}

void str_bcopy(const void *s, void *d, size_t n) {
    str_memmove(d, s, n);
}

void str_bzero(void *d, size_t n) {
    str_memset(d, 0, n);
}

/* The stores are memset's, behind a call the caller's compiler cannot see
   into: they are not removed as dead. */
void str_explicit_bzero(void *d, size_t n) {
    str_memset(d, 0, n);
}

void *str_mempcpy(void *d, const void *s, size_t n) {
    return (u8 *)str_memcpy(d, s, n) + n;
}

/* The selects (UNPREDICTABLE): whether the string fits, whether a delimiter
   is there, a program decides call by call; a branch on it mispredicts
   half the time on the workload. */
char *str_strncat(char *d, const char *s, size_t n) {
    char *e = d + str_strlen(d);
    size_t k = str_strnlen(s, n);
    str_memcpy(e, s, k);
    e[k] = 0;
    return d;
}

size_t str_strlcpy(char *d, const char *s, size_t n) {
    size_t l = str_strlen(s);

    if (n) {
        size_t k = UNPREDICTABLE(l < n - 1) ? l : n - 1;
        str_memcpy(d, s, k);
        d[k] = 0;
    }

    return l;
}

size_t str_strlcat(char *d, const char *s, size_t n) {
    size_t l = str_strnlen(d, n);

    if (l == n) {
        return l + str_strlen(s);
    }

    return l + str_strlcpy(d + l, s, n - l);
}

char *str_strpbrk(const char *s, const char *set) {
    s += str_strcspn(s, set);
    return UNPREDICTABLE(*s != 0) ? (char *)s : 0;
}

char *str_strsep(char **sp, const char *sep) {
    char *s = *sp;

    if (!s) {
        return 0;
    }

    char *e = s + str_strcspn(s, sep);

    if (*e) {
        *e = 0;
        *sp = e + 1;
    } else {
        *sp = 0;
    }

    return s;
}

/* Past the last token *last is null, and a call with s null and *last null
   finds none. */
char *str_strtok_r(char *s, const char *sep, char **last) {
    if (!s) {
        s = *last;

        if (!s) {
            return 0;
        }
    }

    s += str_strspn(s, sep);

    if (!*s) {
        *last = 0;
        return 0;
    }

    char *e = s + str_strcspn(s, sep);

    if (*e) {
        *e = 0;
        *last = e + 1;
    } else {
        *last = 0;
    }

    return s;
}

static char *strtok_last;

char *str_strtok(char *s, const char *sep) {
    return str_strtok_r(s, sep, &strtok_last);
}

/* ------------------------------------------------------------ two-way */

/* The maximal suffix of n[0, l) under the order of the bytes (rev: the
   opposite one): where it starts, and its period in *period (Crochemore and
   Perrin's computation: a the suffix so far, b the one compared with it, k
   bytes of them equal, p the period of a's). */
static size_t maximal_suffix(const u8 *n, size_t l, int rev, size_t *period) {
    size_t a = 0, b = 1, k = 1, p = 1;

    while (b + k <= l) {
        u8 x = n[a + k - 1], y = n[b + k - 1];

        if (x == y) {
            if (k == p) {
                b += p;
                k = 1;
            } else {
                k++;
            }
        } else if (rev ? x < y : x > y) {
            b += k;
            k = 1;
            p = b - a;
        } else {
            a = b;
            b = a + 1;
            k = p = 1;
        }
    }

    *period = p;
    return a;
}

/* The first occurrence of n[0, l), l >= 2, in the haystack from h: [h, end),
   or with grow (strstr) known to be free of NULs up to end and found to its
   NUL by memchr as the window needs; null if none. The needle is split at
   the critical place ms, the later of the two maximal suffixes; a window is
   compared from ms to the right, then from ms to the left. A mismatch on the
   right moves the window past it; one on the left by the period p, with the
   first mem0 bytes of the next window known to match when the needle has
   that period, else p is max(ms, l - ms + 1) and nothing is known. Before
   that, the window's last byte: a needle that cannot end there moves the
   window to where its last occurrence of that byte would (Boyer and Moore's
   bad character), past the window when the needle lacks it. */
static const u8 *two_way(const u8 *h, const u8 *end, const u8 *n, size_t l, int grow) {
    size_t p, p2;
    size_t ms = maximal_suffix(n, l, 0, &p), ms2 = maximal_suffix(n, l, 1, &p2);

    if (ms2 > ms) {
        ms = ms2;
        p = p2;
    }

    size_t mem0 = 0;

    if (!str_memcmp(n, n + p, ms)) {
        mem0 = l - p;
    } else {
        p = (ms > l - ms ? ms : l - ms + 1);
    }

    /* the bad-character shifts, at most 255: a smaller shift is still safe */
    u8 shift[256];
    str_memset(shift, l < 255 ? (int)l : 255, 256);

    for (size_t i = 0; i < l; i++) {
        size_t s = l - 1 - i;
        shift[n[i]] = (u8)(s < 255 ? s : 255);
    }

    size_t mem = 0;

    for (;;) {
        while ((size_t)(end - h) < l) {
            if (!grow) {
                return 0;
            }

            const u8 *z = str_memchr(end, 0, l | 63);

            if (z) {
                end = z;
                grow = 0;
            } else {
                end += l | 63;
            }
        }

        size_t k = shift[h[l - 1]];

        if (k) {
            h += k;
            mem = 0;
            continue;
        }

        for (k = ms > mem ? ms : mem; k < l && n[k] == h[k]; k++) {
        }

        if (k < l) {
            h += k - ms + 1;
            mem = 0;
            continue;
        }

        for (k = ms; k > mem && n[k - 1] == h[k - 1]; k--) {
        }

        if (k <= mem) {
            return h;
        }

        h += p;
        mem = mem0;
    }
}

/* A scan's budget: a place found costs CANDIDATE, a compare past the
   second byte its length; past the bytes advanced plus BUDGET the rest goes
   to two_way. */
#define CANDIDATE 8
#define BUDGET 256

void *str_memmem(const void *hay, size_t hl, const void *needle, size_t nl) {
    const u8 *h = hay, *n = needle;

    if (!nl) {
        return (void *)h;
    }

    if (hl < nl) {
        return 0;
    }

    if (nl == 1) {
        return str_memchr(h, n[0], hl);
    }

    const u8 *start = h, *end = h + hl, *last = end - nl;
    size_t work = 0;

    for (;;) {
        const u8 *p = str_memchr(h, n[0], (size_t)(last - h) + 1);

        if (!p) {
            return 0;
        }

        if (p[1] == n[1]) {
            if (!str_memcmp(p + 2, n + 2, nl - 2)) {
                return (void *)p;
            }

            work += nl;
        }

        if (p == last) {
            return 0;
        }

        h = p + 1;
        work += CANDIDATE;

        if (work > (size_t)(h - start) + 2 * nl + BUDGET) {
            return (void *)two_way(h, end, n, nl, 0);
        }
    }
}

/* The haystack from h for a needle of l < 16 bytes, both with 16 bytes of
   their pages ahead (nv the needle's 16 bytes, need the bits of its l): 16
   bytes of the haystack at a time, the needle's first byte and the NUL in one
   mask, a candidate compared with the needle in one 16-byte compare (the
   haystack's NUL within those bytes differs from the needle's byte there).
   The loads stay in their pages past the strings' ends, as every first block's

   do. The place found, null for none, or at a page's end the place to go on
   from by the slower way in *rest. */
static const u8 *strstr_short(const u8 *h, __m128i nv, u8 n0, unsigned need, const u8 **rest) {
    __m128i first = splat16(n0);

    while (page_left(h) >= 16) {
        __m128i hv = load16(h);
        unsigned z = zero_mask16(hv), stop = eq_mask16(hv, first) | z;

        while (stop) {
            unsigned i = ctz(stop);

            if (z >> i & 1) {
                return 0;
            }

            if (page_left(h + i) < 16) {
                *rest = h + i;
                return 0;
            }

            if ((eq_mask16(load16(h + i), nv) & need) == need) {
                return h + i;
            }

            stop &= stop - 1;
        }

        h += 16;
    }

    *rest = h;
    return 0;
}

char *str_strstr(const char *hay, const char *needle) {
    const u8 *h = (const u8 *)hay, *n = (const u8 *)needle;

    if (!n[0]) {
        return (char *)h;
    }

    if (page_left(n) >= 16) {
        __m128i nv = load16(n);
        unsigned nz = zero_mask16(nv);

        if (nz) {
            const u8 *rest = 0;
            const u8 *p = strstr_short(h, nv, n[0], (1u << ctz(nz)) - 1, &rest);

            if (p || !rest) {
                return (char *)p;
            }

            h = rest;
        }
    }

    u8 first[2];   /* the needle's first byte as a set: strcspn stops at it or at the NUL */
    first[0] = n[0];
    first[1] = 0;
    const u8 *start = h;
    size_t work = 0;

    for (;;) {
        h += str_strcspn((const char *)h, (const char *)first);

        if (!*h) {
            return 0;
        }

        size_t j = 1;

        while (n[j] && h[j] == n[j]) {
            j++;
        }

        if (!n[j]) {
            return (char *)h;
        }

        if (!h[j]) {
            return 0;   /* the haystack ends inside this window: no later one fits */
        }

        h++;
        work += CANDIDATE + j;

        if (work > (size_t)(h - start) + BUDGET) {
            return (char *)two_way(h, h, n, str_strlen(needle), 1);
        }
    }
}
