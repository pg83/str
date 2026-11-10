#!/usr/bin/env python3
"""A workload for the string functions: a long trace of calls with sizes,
alignments and contents drawn from measured distributions, run as a whole and
timed as a whole. One number per library, plus the same trace split per
function.

Sizes are log-normal, three numbers per class: median, sigma of ln n and the
share of zero sizes. Strings: median 8, sigma 1.2 (half of the strings are
under 8 bytes, 10% over 32, 1% over 128), fitted to identifiers, string
literals and lines of 46 686 source files of a monorepo; memory
blocks: memcpy 16/1.8, memmove 14/1.3, memcmp 14/0.95, memset 29/1.6, from
the automemcpy paper's published histograms, which a log-normal matches within
5-10% per size bucket. Other parameters:
  alignment - strings at any offset; memory blocks 8-aligned half the time.
  contents - strcmp: 30% equal, else a common prefix of geometric length
    (mean 1) and a difference; memcmp: 50% equal, else one different byte at
    a uniform position; memset: 80% zero; memmove: 50% overlapping.
  strspn skips leading whitespace (" \t", 0-3 bytes, geometric); strcspn
    scans a token for one of 1-3 delimiters.
  call mix: each call's function drawn by the weights memcpy 25%, strlen
    20%, strcmp 12%, memset 10%, memcmp 10%, strcpy 8%, strcspn 5%, memmove
    5%, strspn 3%, strcat 2% (common sense, stated here so it can be argued
    with), so that every function's branches see the others' in the history,
    as in a program. The stand's own dispatch mispredicts then: noop, the
    library of empty functions, measures it and libs.py subtracts it.
  per-function traces: drawn on their own, not cut out of the mix, and long
    (PER_FUNCTION calls): a pass of a short trace replays it many times, and
    the branch predictor learns the replayed sequence (a trace of 5 000 strcat
    calls: glibc's size branches at 0.05 mispredictions a call).
  working set - 2 MiB of sources and 2 MiB of destinations, L2-sized.

File format (little-endian): 'STRW' u32 version u64 src_size u64 dst_size
u32 ntraces, the two pool images, then per trace u32 namelen name u64 count
and count records of {u8 op, u8[3] zero, u32 n, u64 a, u64 b}.
"""
import math
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = ['memcpy', 'memmove', 'memmove_overlap', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat', 'strcmp', 'strspn', 'strcspn', 'memchr', 'memcpy_dst']
MIX = {'memcpy': 25, 'strlen': 20, 'strcmp': 12, 'memset': 10, 'memcmp': 10, 'strcpy': 8, 'strcspn': 5, 'memmove': 5, 'memchr': 5, 'strspn': 3, 'strcat': 2}
SRC_SIZE = 8 << 20   # the first WORK_SET bytes are the working set of the mix (L2-sized); the long-string traces live beyond
WORK_SET = 2 << 20
DST_SIZE = 2 << 20
RECORDS = 1 << 18
PER_FUNCTION = 1 << 17


# median, sigma of ln n, share of zero-length calls: the log-normal fits of the measured data
LOGNORMAL = {'string': (8, 1.2, 0.04), 'memcpy': (16, 1.8, 0.025), 'memmove': (14, 1.3, 0.01), 'memcmp': (14, 0.95, 0.001), 'memset': (29, 1.6, 0.007)}


def make_draw(rng):
    """draw(kind) -> size, kind 'string' or a memory function."""
    def draw(kind):
        median, sigma, zero = LOGNORMAL[kind]
        if rng.random() < zero:
            return 0
        return max(1, min(4096, int(round(math.exp(rng.gauss(math.log(median), sigma))))))
    return draw


class Pool:
    def __init__(self, size):
        self.data = bytearray(size)
        self.pos = 0

    def place(self, blob, rng, align=None):
        """Append blob at a random offset (0..15 past the cursor, or aligned), return its offset."""
        if align:
            self.pos = (self.pos + align - 1) // align * align
        else:
            self.pos += rng.randrange(16)
        if self.pos + len(blob) + 64 > len(self.data):
            raise RuntimeError('pool full')
        self.data[self.pos:self.pos + len(blob)] = blob
        off = self.pos
        self.pos += len(blob)
        return off


LETTERS = b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'


def text(rng, n):
    return bytes(rng.choice(LETTERS) for _ in range(n))


def generate(seed=1, records=RECORDS, per_function=PER_FUNCTION):
    rng = random.Random(seed)
    src, dst = Pool(SRC_SIZE), Pool(DST_SIZE)
    draw = make_draw(rng)
    sets = {'ws': src.place(b' \t\0', rng), 'd1': src.place(b',\0', rng), 'd2': src.place(b';\0', rng), 'd3': src.place(b'=\t\0', rng), 'd4': src.place(b' ,;\0', rng)}
    objects = {op: [] for op in OPS}
    counts = {'memcmp': 4096, 'strlen': 8192, 'strcpy': 8192, 'strcmp': 4096, 'strspn': 4096, 'strcspn': 4096, 'memchr': 4096}
    for _ in range(counts['memcmp']):
        n = draw('memcmp')
        a = rng.randbytes(n)
        b = bytearray(a)
        if n and rng.random() < 0.5:
            k = rng.randrange(n)
            b[k] ^= 1 + rng.randrange(255)
        objects['memcmp'].append((n, src.place(a, rng), src.place(bytes(b), rng)))
    for _ in range(counts['strlen']):
        n = draw('string')
        objects['strlen'].append((n, src.place(text(rng, n) + b'\0', rng), 0))
    objects['strcpy'] = objects['strlen']
    for _ in range(counts['strcmp']):
        n = draw('string')
        a = text(rng, n)
        if rng.random() < 0.3:
            b = a
        else:
            prefix = min(n, int(rng.expovariate(1.0)))
            tail = text(rng, draw('string'))
            b = a[:prefix] + bytes([(a[prefix] if prefix < n else 65) ^ 1]) + tail if prefix < n else a + b'x' + tail
        objects['strcmp'].append((n, src.place(a + b'\0', rng), src.place(b + b'\0', rng)))
    for _ in range(counts['strspn']):
        k = min(int(rng.expovariate(1.0)), 3)
        n = draw('string')
        s = bytes(rng.choice(b' \t') for _ in range(k)) + text(rng, n) + b'\0'
        objects['strspn'].append((k, src.place(s, rng), sets['ws']))
    for _ in range(counts['memchr']):
        # a delimiter in a buffer of string-like length, present in half the cases
        n = draw('string')
        buf = bytearray(text(rng, n))
        c = 0x01
        if n and rng.random() < 0.5:
            buf[rng.randrange(n)] = ord(',')
            c = ord(',')
        objects['memchr'].append((n, src.place(bytes(buf), rng), c))
    # strcat: a destination string of its own per object (the upper half of the
    # destination pool), room after it for the source and its NUL
    strcat_dst = Pool(DST_SIZE)
    strcat_dst.pos = DST_SIZE // 2
    for _ in range(8192):
        n, a, _ = rng.choice(objects['strlen'])
        m = min(draw('string'), 64)
        slot = strcat_dst.place(text(rng, m) + b'\0' + bytes(n + 1), rng)
        dst.data[slot:slot + m + 1] = strcat_dst.data[slot:slot + m + 1]
        objects['strcat'].append((m, a, slot))   # n = the destination's own length: the runner re-terminates it before each call
    for _ in range(counts['strcspn']):
        n = draw('string')
        setname = rng.choice(['d1', 'd2', 'd3', 'd4'])
        delim = {'d1': b',', 'd2': b';', 'd3': b'=', 'd4': b','}[setname]
        s = text(rng, n) + delim + text(rng, rng.randrange(8)) + b'\0'
        objects['strcspn'].append((n, src.place(s, rng), sets[setname]))
    def record(op):
        if op in ('memcpy', 'memmove'):
            n = draw(op)
            align = 8 if rng.random() < 0.5 else 1
            a = rng.randrange(0, WORK_SET - 4096, align)
            if op == 'memmove' and rng.random() < 0.5:
                b = rng.randrange(DST_SIZE // 4, DST_SIZE // 2 - 8192, align)
                delta = rng.randrange(1, max(2, n + 1)) * rng.choice((-1, 1))
                return ('memmove_overlap', n, b, b + delta)
            return (op, n, a, rng.randrange(0, DST_SIZE // 2 - 4096, align))
        if op == 'memset':
            n = draw('memset')
            align = 8 if rng.random() < 0.5 else 1
            b = rng.randrange(0, DST_SIZE // 2 - 4096, align)
            return (op, n, 0 if rng.random() < 0.8 else rng.randrange(256), b)
        if op == 'strcpy':
            n, a, _ = rng.choice(objects['strcpy'])
            return (op, n, a, rng.randrange(0, DST_SIZE // 2 - 4096))
        n, a, b = rng.choice(objects[op])   # memcmp, strlen, strcmp, strspn, strcspn, memchr, strcat
        return (op, n, a, b)

    ops = list(MIX)
    weights = [MIX[o] for o in ops]
    traces = [('mixed', [record(op) for op in rng.choices(ops, weights, k=records)])]
    for op in ops:
        traces.append((op, [record(op) for _ in range(per_function)]))
    # the set scans on long strings with larger sets (a parser's whitespace and delimiter
    # sets, lines of 64-1024 bytes): not in the mix, a trace of their own
    src.pos = max(src.pos, WORK_SET)
    long_sets = []
    for _ in range(16):
        k = 8 + rng.randrange(40)
        long_sets.append(src.place(bytes(sorted(set(rng.sample(range(1, 128), k)))) + b'\0', rng))
    spn_long, cspn_long = [], []
    for _ in range(4096):
        n = 64 + rng.randrange(961)
        setpos = rng.choice(long_sets)
        setbytes = src.data[setpos:src.data.index(0, setpos)]
        body = bytes(rng.choice(setbytes) for _ in range(n)) + b'x' + text(rng, 8) + b'\0'
        spn_long.append(('strspn', n, src.place(body, rng), setpos))
        others = bytes(b for b in LETTERS if b not in setbytes) or b'a'
        body = bytes(rng.choice(others) for _ in range(n)) + bytes([setbytes[0]]) + text(rng, 8) + b'\0'
        cspn_long.append(('strcspn', n, src.place(body, rng), setpos))
    traces.append(('strspn_long', spn_long))
    traces.append(('strcspn_long', cspn_long))
    return bytes(src.data), bytes(dst.data), traces


def write(path, seed=1):
    src, dst, traces = generate(seed)
    with open(path, 'wb') as f:
        f.write(b'STRW' + struct.pack('<IQQI', 1, len(src), len(dst), len(traces)))
        f.write(src)
        f.write(dst)
        for name, records in traces:
            f.write(struct.pack('<I', len(name)) + name.encode() + struct.pack('<Q', len(records)))
            f.write(b''.join(struct.pack('<B3xIQQ', OPS.index(op), n, a, b) for op, n, a, b in records))
    return traces


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = write(out)
    print('%s: %d records in %d traces, %.1f MB' % (out, len(traces[0][1]), len(traces), out.stat().st_size / 1e6))
    for name, records in traces[1:]:
        sizes = sorted(r[1] for r in records)
        print('  %-8s %6d calls, size median %d, p90 %d' % (name, len(records), sizes[len(sizes) // 2], sizes[int(len(sizes) * 0.9)]))
