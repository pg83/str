#!/usr/bin/env python3
"""A workload split by size: which sizes a trace's time comes from.

    dev/lab/split.py IN OUT [--traces memset,memcpy] [--edges 0,16,32,64,128,256,512,1024,2048,4096] [--by-op]

Reads a workload (workload.py's format), writes one with the same pools and,
per trace named (fnmatch patterns: 'pz/*'), one trace per size bucket of its
records (lo < n <= hi; the first bucket from 0 inclusive) named TRACE/lo-hi,
or with --by-op per function and bucket, TRACE/FUNCTION/lo-hi (a program's
trace calls several: pizdos.py trace). kbench runs a trace NAME/... for
NAME's primitive, and every trace in --libs: per bucket, the kernels' or the
libraries' ns per call on those calls alone, in the order the trace has them.
"""
import argparse
import fnmatch
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workload import OPS  # noqa: E402

REC = struct.Struct('<B3xIQQ')


def read(path):
    b = Path(path).read_bytes()
    assert b[:4] == b'STRW'
    version, ssz, dsz, nt = struct.unpack_from('<IQQI', b, 4)
    assert version in (1, 2, 3)
    o = 4 + struct.calcsize('<IQQI')
    zeros = b''
    if version == 3:   # each pool's continuation in zeros, not stored
        zeros = b[o:o + 16]
        o += 16
    pools = zeros + b[o:o + ssz + dsz] if version != 2 else b''   # version 2: zeros, not stored
    o += len(pools) - len(zeros)
    traces = []
    for _ in range(nt):
        nl, = struct.unpack_from('<I', b, o)
        name = b[o + 4:o + 4 + nl].decode()
        count, = struct.unpack_from('<Q', b, o + 4 + nl)
        o += 4 + nl + 8
        traces.append((name, b[o:o + count * REC.size]))
        o += count * REC.size
    return version, ssz, dsz, pools, traces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--traces', default='memset,memcpy')
    ap.add_argument('--edges', default='0,16,32,64,128,256,512,1024,2048,4096')
    ap.add_argument('--by-op', action='store_true')
    args = ap.parse_args()
    edges = [int(x) for x in args.edges.split(',')]
    version, ssz, dsz, pools, traces = read(args.inp)
    out = []
    for name, recs in traces:
        if not any(fnmatch.fnmatchcase(name, t) for t in args.traces.split(',')):
            continue
        recs = [recs[i:i + REC.size] for i in range(0, len(recs), REC.size)]
        ops = sorted({r[0] for r in recs}) if args.by_op else [None]
        for op in ops:
            for lo, hi in zip(edges, edges[1:]):
                keep = [r for r in recs if (op is None or r[0] == op) and (lo < REC.unpack(r)[1] <= hi or (lo == edges[0] and REC.unpack(r)[1] == lo))]
                if keep:
                    out.append(('%s/%s%d-%d' % (name, '' if op is None else OPS[op] + '/', lo, hi), b''.join(keep)))
    with open(args.out, 'wb') as f:
        f.write(b'STRW' + struct.pack('<IQQI', version, ssz, dsz, len(out)) + pools)
        for name, recs in out:
            f.write(struct.pack('<I', len(name)) + name.encode() + struct.pack('<Q', len(recs) // REC.size) + recs)
    for name, recs in out:
        print('%-36s %7d calls' % (name, len(recs) // REC.size))


if __name__ == '__main__':
    main()
