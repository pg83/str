#!/usr/bin/env python3
"""Propose each host's table from a lab run's traces, and what others still do better.

    dev/lab/tune.py TAG [--hosts local,lab2,lab3] [--gain 0.03]

For the configuration a host's dispatch installs (meta.json), every slot's
implementations are compared on the workload traces of the primitives that
go through the slot: the rows ours/<config>.t+<slot>=<fn> against
ours/<config>.t, the same table with the slot as it is. A slot that several
primitives share (copy_large: memcpy and memmove; strlen: strlen and strcat)
is chosen by the sum over them weighted by their share of the workload's
calls. A replacement must gain --gain (3%) on that sum. On the configuration
with the JCC-erratum copies (evex, Skylake-SP) only *_jcc implementations are
candidates, elsewhere only the plain ones.

Prints, per host: the proposal as a kbench --table spec (verify it on the
mixed trace: run.py ... -- --mixed --table SPEC), the evidence per slot, and
per trace the best kernel from outside ours where it beats ours' best.
"""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from workload import MIX, PRIMITIVES  # noqa: E402  the mix's weights; the primitives are what the slots serve


def load(path):
    traces = collections.defaultdict(dict)
    with gzip.open(path, 'rt') as f:
        for r in csv.DictReader(f):
            if r['regime'] == 'trace':
                traces[r['scenario'][6:]][r['kernel']] = float(r['med_ns'])
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tag')
    ap.add_argument('--hosts', default='local,lab2,lab3')
    ap.add_argument('--gain', type=float, default=0.03)
    args = ap.parse_args()
    for host in args.hosts.split(','):
        d = ROOT / '.lab' / args.tag / host
        if not (d / 'out.csv.gz').exists():
            continue
        meta = json.loads((d / 'meta.json').read_text())
        cfg = meta['ours_config']
        jcc = cfg == 'evex'
        traces = load(d / 'out.csv.gz')
        base = 'ours/%s.t' % cfg
        # slot -> impl -> {prim: ratio to the base table}
        by_slot = collections.defaultdict(lambda: collections.defaultdict(dict))
        for prim in PRIMITIVES:
            ks = traces.get(prim, {})
            if base not in ks:
                continue
            for k, v in ks.items():
                if not k.startswith(base + '+'):
                    continue
                slot, fn = k[len(base) + 1:].split('=', 1)
                if fn.endswith('_jcc') != jcc:
                    continue
                by_slot[slot][fn][prim] = v / ks[base]
        print('== %s (%s), configuration %s' % (host, meta['brand'].strip(), cfg))
        changes = []
        for slot in sorted(by_slot):
            prims = sorted({p for fn in by_slot[slot].values() for p in fn})
            total = sum(MIX[p] for p in prims)
            scored = []
            for fn, rs in by_slot[slot].items():
                if set(rs) != set(prims):
                    continue
                scored.append((sum(MIX[p] * rs[p] for p in prims) / total, fn, rs))
            if not scored:
                continue
            scored.sort()
            score, fn, rs = scored[0]
            mark = score <= 1 - args.gain
            print('   %-13s %s %-28s %.3f  (%s)%s' % (slot, '->' if mark else '  ', fn, score, ', '.join('%s %.3f' % (p, rs[p]) for p in prims),
                                                    '' if mark else '  keep'))
            if mark:
                changes.append('%s=%s' % (slot, fn))
        print('   proposal: %s' % ('%s-tuned=%s:%s' % (cfg, cfg, ','.join(changes)) if changes else 'none'))
        print('   others ahead of ours (trace: best outside ours / best of ours):')
        for name, ks in sorted(traces.items()):
            ours = min((v for k, v in ks.items() if k.startswith('ours/')), default=None)
            other = min(((v, k) for k, v in ks.items() if not k.startswith('ours/') and not k.startswith('kl/')), default=None)
            if ours and other and other[0] < ours:
                print('      %-14s %-42s %.2f' % (name, other[1], other[0] / ours))
        print()


if __name__ == '__main__':
    main()
