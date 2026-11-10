#!/usr/bin/env python3
"""One node of ./build proof (dev/proof/targets.py), run where the build runs
it (here, or a --dist host): a function's proof in one class or one of its
classes, the sum of a function's classes, or the proof of a copy whose code
is another's.

    node.py --elf E --spec S --func F --out O [--classes N --align K] [--callee O ...]
    node.py --elf E --spec S --func F --out O --of O1 O2 ...       the classes' sum
    node.py --elf E --spec S --func F --out O --same-as F2 --of O2  another copy's proof

--callee: the outputs of the proofs whose contracts the spec takes for the
function's callees (strcat, strspn's shuffle scan): each callee must be one
of them, proved, whichever way this node goes. The solvers: --bin, a directory with z3 and bitwuzla (the proof_solvers node's).
STR_PROOF_WORKERS, STR_PROOF_TIMEOUT, STR_PROOF_MEMORY (build.py sets them
from -D) go to prove.py; STR_PROOF_NODE_TIMEOUT, seconds, ends a proof that
takes longer (a failure). The solvers' answers are kept in ~/proof-cache on
the host (PROVE_CACHE; STR_PROOF_CACHE=0: every query asked). While it runs, prove.py's line a minute on how
far it is goes to ~/proof-status/ on the host it runs on (PROVE_STATUS:
<func>-<spec>-<classes>-<align>.txt), to see from outside where a proof of hours
is. The output: the RESULT line and how the proof ended; the node fails
unless proved."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def table_slots():
    """The slots of struct str_table, in common.h's order."""
    text = (HERE.parent.parent / 'common.h').read_text()
    body = text[text.index('struct str_table {'):]
    return re.findall(r'\(\*(\w+)\)', body[:body.index('};')])


def slot_implementations(img, k, nslots):
    """The names of what every configuration's table holds in slot k (str_configs: a name, the needs
    and their padding, then the table)."""
    word = lambda at, n: int.from_bytes(bytes(img.byte(at + i) for i in range(n)), 'little')  # noqa: E731
    stride = 16 + 8 * nslots
    return [img.names.get(word(img.symbols['str_configs'] + stride * j + 16 + 8 * k, 8), '?')
            for j in range(word(img.symbols['str_nconfigs'], 4))]


def entry_slot(img, addr):
    """The slot an entry point is a jump through (jmp [rip + str_table + 8 k]), or None."""
    j = img.decode(addr)
    o = j['ops'][0] if j['ops'] else None
    if j['mnemonic'] != 'jmp' or o is None or o['kind'] != 'Memory' or not o['mem'].get('rip_rel'):
        return None
    k, r = divmod(j['offset'] + j['len'] + o['mem']['disp'] - img.symbols['str_table'], 8)
    return k if r == 0 else None


def derived_callees(img, spec, func, callees):
    """derived.c's: each callee an entry point that jumps through its slot, every configuration's
    implementation there proved by the slot's specs among the callees' proofs; malloc the C
    library's, taken by its contract. The table the constructor installs (before it runs the
    plain sse2 functions are there, no proof's)."""
    import prove
    import targets
    proved = {}
    for o in callees:
        r = result_of(o)
        if r and r.get('result') == 'proved':
            proved.setdefault(r['func'], set()).add(r['spec'])
    names = {a: n for n, a in img.symbols.items()}
    slots, missing = table_slots(), []
    for t in prove.contract_callees(spec, img, img.symbols[func]):
        n = names.get(t, hex(t))
        if n == 'malloc' and img.symbols.get('malloc') == t:
            continue
        k = entry_slot(img, t)
        if k is None or k >= len(slots) or n != 'str_' + slots[k]:
            missing.append('%s: not the jump through its slot' % n)
            continue
        for impl in slot_implementations(img, k, len(slots)):
            missing += ['%s by %s' % (impl, x) for x in targets.SLOT_SPECS.get(slots[k], (slots[k],)) if x not in proved.get(impl, ())]
    return missing


def result_of(path):
    """The RESULT line of a node's output, as a dict; None without one."""
    for line in Path(path).read_text().splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[7:])
    return None


def write(out, r, text=''):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text + 'RESULT ' + json.dumps(r) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--elf', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--func', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bin', help='the directory with z3 and bitwuzla')
    ap.add_argument('--classes', type=int)
    ap.add_argument('--align', type=int, default=0)
    ap.add_argument('--setmax', type=int, help='strspn and strcspn: the short spec\'s longest set (prove.py --setmax)')
    ap.add_argument('--of', nargs='+', default=[], help="the classes' outputs, or the other copy's")
    ap.add_argument('--same-as', help='the copy whose code this one has')
    ap.add_argument('--callee', nargs='+', default=[], help='the outputs of the proofs of the callees the spec takes by contract')
    args = ap.parse_args()
    import machine as X
    import prove
    import run
    img = X.Image(args.elf)

    # every callee the spec takes by its contract proved on its own
    if not args.of and args.spec in __import__('targets').DERIVED:
        missing = derived_callees(img, args.spec, args.func, args.callee)
        if missing:
            sys.exit('%s: callees without a proof among the dependencies: %s' % (args.func, ', '.join(missing)))
    elif not args.of or args.same_as:
        proved = {}
        for o in args.callee:
            r = result_of(o)
            if r and r.get('result') == 'proved':
                proved[r['func']] = r
        names = {a: n for n, a in img.symbols.items()}
        need = prove.contract_callees(args.spec, img, img.symbols[args.func])
        missing = [names.get(t, hex(t)) for t in need if names.get(t) not in proved]
        if missing:
            sys.exit('%s: callees without a proof among the dependencies: %s' % (args.func, ', '.join(missing)))

    if args.same_as:  # the same code, checked the way run.py groups copies (sse2 and sse42 decode tzcnt as bsf)
        bmi = lambda f: not f.startswith(('sse2.', 'sse42.'))  # noqa: E731
        if bmi(args.func) != bmi(args.same_as) or run.fingerprint(img, img.symbols[args.func], prove.contract_stops(args.spec, img, img.symbols[args.func])) != \
                run.fingerprint(img, img.symbols[args.same_as], prove.contract_stops(args.spec, img, img.symbols[args.same_as])):
            sys.exit('%s: its code is not %s\'s any more: targets.SAME is stale' % (args.func, args.same_as))
        r = result_of(args.of[0])
        if not r or r.get('result') != 'proved':
            sys.exit('%s: %s is not proved' % (args.func, args.same_as))
        write(args.out, {**r, 'func': args.func, 'same_as': args.same_as}, 'the code of %s, proved with it\n' % args.same_as)
        return

    if args.of:  # the classes: a case split, all of them every pointer
        rs = [result_of(o) for o in args.of]
        bad = [o for o, r in zip(args.of, rs) if not r or r.get('result') != 'proved']
        if bad:
            sys.exit('%s: classes not proved: %s' % (args.func, ', '.join(bad)))
        write(args.out, {'func': args.func, 'spec': args.spec, 'classes': len(rs), 'result': 'proved',
                         'paths': sum(r['paths'] for r in rs), 'obligations': sum(r['obligations'] for r in rs),
                         'seconds': round(sum(r['seconds'] for r in rs), 1)})
        return

    env = dict(os.environ, PROVE_TIMEOUT=os.environ.get('STR_PROOF_TIMEOUT', '1800'), PROVE_MEMORY=os.environ.get('STR_PROOF_MEMORY', '3000'))
    status = Path.home() / 'proof-status'
    try:
        status.mkdir(exist_ok=True)
        env['PROVE_STATUS'] = str(status)
    except OSError:  # a home of no writes (a sandbox): no status lines
        pass
    if os.environ.get('STR_PROOF_CACHE', '1') != '0':
        try:
            cache = Path.home() / 'proof-cache'
            cache.mkdir(exist_ok=True)
            env['PROVE_CACHE'] = str(cache)
        except OSError:  # as for the status lines
            pass
    if args.bin:
        env['PATH'] = str(Path(args.bin).resolve()) + os.pathsep + env.get('PATH', '')
        if (Path(args.bin) / 'bitwuzla').exists():
            env['PROVE_BITWUZLA'] = str((Path(args.bin) / 'bitwuzla').resolve())
    cmd = [sys.executable, str(HERE / 'prove.py'), '--elf', args.elf, '--func', args.func, '--spec', args.spec, '--induct',
           '--classes', str(args.classes or 1), '--align', str(args.align), '--workers', os.environ.get('STR_PROOF_WORKERS', '1')] + \
        (['--setmax', str(args.setmax)] if args.setmax else [])
    limit = int(os.environ.get('STR_PROOF_NODE_TIMEOUT') or 0) or None
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out = p.communicate(timeout=limit)[0]
    except subprocess.TimeoutExpired:
        p.terminate()  # SIGTERM: prove.py takes its solvers down
        out = p.communicate()[0] + '\nkilled after %d s' % limit
    lines = out.splitlines()
    r = next((json.loads(x[7:]) for x in lines if x.startswith('RESULT ')), None)
    tail = '\n'.join(x for x in lines[-20:] if not x.startswith('RESULT ')) + '\n'
    if r is None:
        r = {'func': args.func, 'spec': args.spec, 'classes': args.classes or 1, 'align': args.align, 'result': 'error',
             'detail': (lines[-1] if lines else 'rc=%d' % p.returncode)}
    write(args.out, r, tail)
    print('%s %s class %s/%s: %s' % (args.spec, args.func, args.align, args.classes or 1, r['result']), flush=True)
    if r['result'] != 'proved':
        sys.exit('%s %s class %s/%s: %s' % (args.spec, args.func, args.align, args.classes or 1, r.get('detail', '')[:300]))


if __name__ == '__main__':
    main()
