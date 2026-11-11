#!/usr/bin/env python3
"""Large fills by size, cold and warm, as a kbench workload: where stores
around the caches (non-temporal ones) start to pay against rep stosb.

    dev/lab/bigfill.py OUT [--sizes 262144,...] [--pool 256]

A trace fill/cold/S fills S bytes at the next place of a pool of POOL MiB,
larger than any L3 here, so that every fill finds its lines in memory;
fill/warm/S fills the same S bytes again and again, which stay in the
caches as far as they fit. workload.py's format, version 2 (the pools zeros,
not stored). kbench runs them with --libs, the fills to compare as tables:

    dev/lab/run.py --tag T --trace OUT -- --prims none --no-matrix --libs \\
        --table rep=avx2:memset=kl_memset_avx2_ntnever,fill_huge=fill_huge_rep \\
        --table nt=avx2:memset=kl_memset_avx2_ntall,fill_huge=fill_huge_rep
"""
import argparse
import struct
from pathlib import Path

W_MEMSET = 3   # workload.py's OPS
SIZES = [256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20, 32 << 20, 64 << 20]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path)
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)))
    ap.add_argument('--pool', type=int, default=256, help='MiB')
    args = ap.parse_args()
    pool = args.pool << 20
    rec = struct.Struct('<B3xIQQ')
    traces = []
    for n in map(int, args.sizes.split(',')):
        places = max(2, pool // n)
        cold = [rec.pack(W_MEMSET, n, 0, (i % places) * n) for i in range(max(16, 2 * places))]
        warm = [rec.pack(W_MEMSET, n, 0, 0) for _ in range(max(16, min(2 * places, 256)))]
        traces.append(('fill/cold/%d' % n, cold))
        traces.append(('fill/warm/%d' % n, warm))
    with open(args.out, 'wb') as f:
        f.write(b'STRW' + struct.pack('<IQQI', 2, 4096, pool, len(traces)))
        for name, recs in traces:
            f.write(struct.pack('<I', len(name)) + name.encode() + struct.pack('<Q', len(recs)) + b''.join(recs))
    for name, recs in traces:
        print('%-20s %6d calls' % (name, len(recs)))


if __name__ == '__main__':
    main()
