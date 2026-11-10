#!/usr/bin/env python3
"""Ours in real programs: pizdos's, run by its loader with our library. A
measurement of its own, beside kbench's synthetic workload.

    ix run lib/c bld/compiler bin/go/lang/25 bin/zstd -- python3 dev/lab/pizdos.py build NAME=SPEC ...
    dev/lab/pizdos.py run --tag T [--hosts local,lab2,lab3] [--runs 5] [--only aot,magic,py]
    dev/lab/pizdos.py report 'T*' [--base NAME] [--hosts ...]
    ix run ... -- python3 dev/lab/pizdos.py trace [--only ...]      the programs' calls as kbench traces

pizdos (github.com/pizd0s/kernel, pinned below) runs wasm modules that its
loader wasm2o compiles to x86-64. A module's memory.copy and memory.fill and
lib/c's string functions are calls of our library: libstr.o in the convention
of the module's code (strcc.py --abi wasm2o), linked into the loader, the
functions of the configuration str_choose picks for the host. The programs
(README.md, "Real programs") are the compiler compiling itself, the decoders
of pizdos's stand over its corpus and CPython on its benchmarks.

build: pizdos unpacked into .lab/pizdos/src/ makes its inputs with its own
./build. Then, per NAME=SPEC, our libstr.o and a wasm2o embedding it; SPEC is
a revision or `work` (the working tree), optionally followed by
:CONFIG.SLOT=FN,... (those slots of dispatch.c's configurations replaced
first: an experiment needs no commit). The library is built by the SPEC's own
strcc.py and checked by its tst/wasm2o.py, so only revisions from a1c43d5 on
(--abi wasm2o) build. The programs are compiled once, by the first wasm2o:
their code does not depend on the library, only their calls do. All of it
goes into .lab/pizdos/bundle/ with count (count.c: the counters, opened as
perf stat opens them; the lab hosts have no perf) and this script.

run: the bundle shipped to each host once per content, the clock held as
run.py holds it, one core (local: 2; a lab host: the idlest but 0). Each run
calls every program with every wasm2o in turns, GOMAXPROCS=1 and GOGC=off,
and records the whole process's cycles and instructions in user mode (the
startup too, the same for every wasm2o), its times and the digest of its
output, which must not depend on the wasm2o: .lab/T/<host>/pizdos.jsonl.

trace: the programs' calls of memcpy, memmove and memset, recorded by a
wasm2o whose GOT entries are wrappers (wasm2o/trace.go), as a kbench workload,
.lab/pizdos/traces.bin: a trace pz/NAME per program, WINDOWS runs of WINDOW
consecutive calls spread over it, the addresses compacted by pages (order,
identity and alignment kept), the pools zeros and not stored (version 2).
run.py --trace .lab/pizdos/traces.bin -- --prims none --no-matrix --libs
replays them with every library; libs.py lists them after the workload's.

report: per host and program the base's median cycles (--metric: or
instructions, the times) and every other wasm2o's ratio to it, the median and
the range of the ratios run by run (the calls of one run follow each other);
the geometric mean over the programs, CPython's benchmarks as one program.
"""
import argparse
import collections
import datetime
import glob
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LAB = ROOT / '.lab/pizdos'
BUNDLE = LAB / 'bundle'
PIZDOS = 'bfc5e0a173d06dbd7342dd8574b17c92317b3e98'
PIZDOS_SHA256 = '9fbe803c82dbab07248174a8da773098af1bcfa5ff88b2e97acdacc907cd15b3'
PIZDOS_URLS = ['https://github.com/pizd0s/kernel/archive/%s.tar.gz' % PIZDOS, 'https://codeload.github.com/pizd0s/kernel/tar.gz/%s' % PIZDOS]
DECODERS = ('im', 'pdf', 'djvu', 'svg', 'magic')
# a decoder's passes over its files in one process: long enough to time
REPEAT = {'im': 1, 'pdf': 1, 'djvu': 1, 'svg': 1, 'magic': 200}
# CPython's benchmarks (pizdos's dev/stand/pybench.py) and their repeats
PY = {'richards': 5, 'deltablue': 13, 'nbody': 5, 'spectral_norm': 9, 'fannkuch': 5, 'nqueens': 48, 'raytrace': 16, 'float': 11,
      'unpack_sequence': 34, 'generators': 12, 'coroutines': 60, 'calls': 40, 'dicts': 10, 'exceptions': 13}
ENV = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'GOMAXPROCS': '1', 'GOGC': 'off', 'MAGICK_THREAD_LIMIT': '1', 'OMP_NUM_THREADS': '1',
       'PYTHONHASHSEED': '0'}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# ---------------------------------------------------------------- build

def pizdos_tree():
    """pizdos at PIZDOS and the inputs its ./build makes: the compiler's
    reactor (aot.aot, embedded in every wasm2o), the compiler as a command
    and its input, the decoders, CPython and the corpus."""
    sys.path.insert(0, str(HERE))
    from deps import download

    tree = LAB / 'src' / PIZDOS
    if not (tree / 'build').exists():
        data = download(PIZDOS_URLS, PIZDOS_SHA256)
        part = tree.with_name(PIZDOS + '.part')
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(part, filter='data')
        (part / ('kernel-' + PIZDOS)).rename(tree)
        part.rmdir()
    targets = ['aot', 'aot_aot', 'aot_command', 'stand_corpus', 'python', *('stand_%s_wasm' % k for k in DECODERS)]
    sh(['./build', '-j', str(os.cpu_count()), *targets], cwd=tree)
    return tree


def str_tree(name, spec):
    """Our tree of SPEC in .lab/pizdos/str/NAME, its slots replaced: the revision and the replacements."""
    rev, _, slots = spec.partition(':')
    tree = LAB / 'str' / name
    shutil.rmtree(tree, ignore_errors=True)
    tree.mkdir(parents=True)
    if rev == 'work':
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        commit = head + '+work'
        files = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT, capture_output=True, check=True).stdout
        for f in files.decode().split('\0'):
            if f and (ROOT / f).is_file():
                (tree / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / f, tree / f)
    else:
        commit = subprocess.run(['git', 'rev-parse', '--verify', rev + '^{commit}'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        archive = subprocess.run(['git', 'archive', commit], cwd=ROOT, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tree, filter='data')
    replaced = []
    if slots:
        path = tree / 'dispatch.c'
        text = path.read_text()
        for edit in slots.split(','):
            m = re.fullmatch(r'(\w+)\.(\w+)=(\w+)', edit)
            if not m:
                raise SystemExit('%s: a replacement is CONFIG.SLOT=FN, not %r' % (name, edit))
            config, slot, fn = m.groups()
            start = text.find('{"%s",' % config)
            if start < 0:
                raise SystemExit('%s: no configuration %s in dispatch.c' % (name, config))
            end = text.index('}},', start)
            block, n = re.subn(r'\.%s = \w+,' % slot, '.%s = %s,' % (slot, fn), text[start:end])
            if n != 1:
                raise SystemExit('%s: %s has %d slots %s' % (name, config, n, slot))
            text = text[:start] + block + text[end:]
            replaced.append(edit)
        path.write_text(text)
    return tree, commit, replaced


def wasm2o_of(tree, out, pizdos):
    """libstr.o of our tree in wasm2o's convention, checked, and a wasm2o embedding it."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'strcc.log', 'w') as log:
        sh([sys.executable, str(tree / 'dev/strcc.py'), str(tree), '--abi', 'wasm2o', '-o', str(out / 'libstr.a'), '--out', str(out)], stdout=log, stderr=subprocess.STDOUT)
    sh([sys.executable, str(tree / 'tst/wasm2o.py'), '--object', str(out / 'libstr.o')], stdout=subprocess.DEVNULL)
    go_build(pizdos, pizdos / 'bin/wasm2o', out / 'libstr.o', out / 'wasm2o')


def go_build(pizdos, src, libstr, out):
    """wasm2o from its sources in src, with pizdos's runtime, compiler and libstr embedded."""
    go = shutil.which('go')
    if not go:
        raise SystemExit('no go on PATH (ix run ... bin/go/lang/25)')
    tmp = out.parent / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, str(pizdos / 'dev/go.py'), 'build', go, str(src), str(out), '--runtime', str(pizdos / 'lib/goruntime'),
        '--embed', str(pizdos / '.build/bin/aot/aot.aot'), '--embed', str(libstr)], env=dict(os.environ, TMPDIR=str(tmp)))
    shutil.rmtree(tmp)


def counter(out):
    cc = os.environ.get('CC', 'cc')
    flags = [*shlex.split(os.environ.get('CPPFLAGS', '')), *shlex.split(os.environ.get('CFLAGS', ''))]
    sh([cc, *flags, '-O2', '-static', str(HERE / 'count.c'), '-o', str(out), *shlex.split(os.environ.get('LDFLAGS', ''))])


def build(args):
    pizdos = pizdos_tree()
    b = pizdos / '.build'
    names = []
    variants = []
    for v in args.variants:
        name, eq, spec = v.partition('=')
        if not eq or not re.fullmatch(r'[\w.-]+', name) or name in names:
            raise SystemExit('a wasm2o is NAME=REV[:CONFIG.SLOT=FN,...], each NAME once: %r' % v)
        names.append(name)
        tree, commit, replaced = str_tree(name, spec)
        out = LAB / 'var' / name
        print('%s: libstr.o of %s%s, wasm2o' % (name, commit[:12], ' with ' + ','.join(replaced) if replaced else ''), flush=True)
        wasm2o_of(tree, out, pizdos)
        variants.append(dict(name=name, spec=spec, commit=commit, slots=replaced, libstr_sha256=digest(out / 'libstr.o')))
    shutil.rmtree(BUNDLE, ignore_errors=True)
    for d in ('variants', 'objs', 'inputs', 'pyroot/usr/bin', 'pyroot/usr/lib/python3.14/lib-dynload'):
        (BUNDLE / d).mkdir(parents=True, exist_ok=True)
    for v in variants:
        (BUNDLE / 'variants' / v['name']).mkdir()
        shutil.copy2(LAB / 'var' / v['name'] / 'wasm2o', BUNDLE / 'variants' / v['name'] / 'wasm2o')
    # the programs, compiled once per pizdos (by the first wasm2o)
    objs = LAB / 'objs' / PIZDOS
    sources = {'aotcmd': b / 'bin/aot/aot-command.wasm', 'python': b / 'bin/python/python.wasm', **{k: b / 'stand/decode' / k / 'input.wasm' for k in DECODERS}}
    for name, wasm in sources.items():
        obj = objs / (name + '.o')
        if not obj.exists():
            objs.mkdir(parents=True, exist_ok=True)
            print('compile %s' % name, flush=True)
            sh([str(LAB / 'var' / variants[0]['name'] / 'wasm2o'), '-o', str(obj) + '.part', str(wasm)])
            os.replace(str(obj) + '.part', obj)
        shutil.copy2(obj, BUNDLE / ('pyroot/usr/bin/python3' if name == 'python' else 'objs/' + obj.name))
    shutil.copy2(b / 'bin/aot/aot.wasm', BUNDLE / 'inputs/aot.wasm')
    os.chmod(BUNDLE / 'pyroot/usr/bin/python3', 0o755)
    shutil.copy2(b / 'bin/python/python314.zip', BUNDLE / 'pyroot/usr/lib/python314.zip')
    shutil.copy2(pizdos / 'dev/stand/pybench.py', BUNDLE / 'pyroot/pybench.py')
    with tarfile.open(b / 'stand/decode-corpus-1.tgz') as tar:
        tar.extractall(BUNDLE, filter='data')
    (BUNDLE / 'decode-corpus-1').rename(BUNDLE / 'corpus')
    counter(BUNDLE / 'count')
    shutil.copy2(Path(__file__), BUNDLE / 'pizdos.py')
    corpus = sorted(p.relative_to(BUNDLE).as_posix() for p in (BUNDLE / 'corpus').rglob('*') if p.is_file() and p.name not in ('MANIFEST', 'SHA256SUMS'))
    works = [dict(name='aot', argv=['{wasm2o}', 'run', 'objs/aotcmd.o'], stdin='inputs/aot.wasm')]
    for k in DECODERS:
        files = corpus if k == 'magic' else [f for f in corpus if f.startswith('corpus/%s/' % k)]
        works.append(dict(name=k, argv=['{wasm2o}', 'stand', '--repeat', str(REPEAT[k]), k, 'objs/%s.o' % k, *files], stdin=None))
    for bench, repeat in PY.items():
        works.append(dict(name='py/' + bench, argv=['{wasm2o}', 'run', '--root', 'pyroot', '-a', '/usr/bin/python3', 'pyroot/usr/bin/python3', '--', '-S',
                                                     '/pybench.py', bench, str(repeat)], stdin=None))
    manifest = dict(pizdos=PIZDOS, built=datetime.datetime.now(datetime.timezone.utc).isoformat(), variants=variants, works=works)
    (BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('%s: %s, id %s' % (BUNDLE, ', '.join(names), bundle_id()))


def bundle_id():
    h = hashlib.sha256()
    for p in sorted(BUNDLE.rglob('*')):
        if p.is_file():
            h.update(p.relative_to(BUNDLE).as_posix().encode() + b'\0' + digest(p).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- trace

RAW = struct.Struct('<IIqqQ')   # wasm2o/trace.go's record: function, length, destination, source or byte, return address
WINDOWS, WINDOW = 4, 1 << 15
PAGE = 4096
W_MEMMOVE_OVL, W_MEMSET, W_MEMCPY_DST = 2, 3, 12   # workload.py's OPS: both objects in the destination pool


def tracer(pizdos, libstr, out):
    """A wasm2o whose str.memcpy, str.memmove and str.memset record every call (wasm2o/trace.go)."""
    src = out / 'src'
    shutil.rmtree(src, ignore_errors=True)
    shutil.copytree(pizdos / 'bin/wasm2o', src)
    for f in ('trace.go', 'trace_amd64.s'):
        shutil.copy2(HERE / 'wasm2o' / f, src / f)
    for name, old, new in (('load.go', '\tif strings.HasPrefix(sym, symStr) {\n\t\treturn strFunction(sym)\n',
                            '\tif strings.HasPrefix(sym, symStr) {\n\t\tif at := strTraced(sym); at != 0 {\n\t\t\treturn at\n\t\t}\n\n\t\treturn strFunction(sym)\n'),
                           ('main.go', '\tos.Exit(code)\n', '\tdumpStrTrace()\n\tos.Exit(code)\n')):
        text = (src / name).read_text()
        if text.count(old) != 1:
            raise SystemExit('wasm2o/%s: the tracer\'s anchor is not there once (pizdos moved on?)' % name)
        (src / name).write_text(text.replace(old, new))
    go_build(pizdos, src, libstr, out / 'wasm2o')
    return out / 'wasm2o'


def windows(raw):
    """WINDOWS runs of WINDOW calls, spread over the trace (all of it if shorter)."""
    n = len(raw) // RAW.size
    if n <= WINDOWS * WINDOW:
        return [raw]
    return [raw[RAW.size * s:RAW.size * (s + WINDOW)] for s in ((2 * k + 1) * n // (2 * WINDOWS) - WINDOW // 2 for k in range(WINDOWS))]


def compact(ranges):
    """The pages the ranges touch, packed in order: an address's offset in the pool."""
    pages = set()
    for off, n in ranges:
        pages.update(range(off // PAGE, (off + max(n, 1) - 1) // PAGE + 1))
    at = {p: i * PAGE for i, p in enumerate(sorted(pages))}
    return (lambda off: at[off // PAGE] + off % PAGE), (len(pages) + 1) * PAGE


def workload(raws):
    """The programs' calls as kbench traces (workload.py's format, version 2: the
    pools are zeros, not stored), all in the destination pool, as all were in
    the program's memory, its pages compacted per program; the call site's
    index in the record's three spare bytes."""
    traces, dsz = [], PAGE
    for name, raw in raws:
        recs = [r for w in windows(raw) for r in RAW.iter_unpack(w)]
        dmap, dn = compact([(d, n) for op, n, d, s, _ in recs] + [(s, n) for op, n, d, s, _ in recs if op != 2])
        dsz = max(dsz, dn)
        sites, out = {}, []
        for op, n, d, s, site in recs:
            k = sites.setdefault(site, len(sites)) & 0xffffff
            if op == 2:
                out.append((W_MEMSET, k, n, s & 255, dmap(d)))
            else:
                out.append((W_MEMCPY_DST if op == 0 else W_MEMMOVE_OVL, k, n, dmap(s), dmap(d)))
        traces.append(('pz/' + name, out))
        print('pz/%s: %d calls of %d, %d sites, %.1f MiB of memory' % (name, len(out), len(raw) // RAW.size, len(sites), dn / 2**20), flush=True)
    blob = io.BytesIO()
    blob.write(b'STRW' + struct.pack('<IQQI', 2, PAGE, dsz, len(traces)))   # version 2: zero pools, not stored
    rec = struct.Struct('<B3sIQQ')
    for name, out in traces:
        blob.write(struct.pack('<I', len(name)) + name.encode() + struct.pack('<Q', len(out)))
        blob.write(b''.join(rec.pack(op, k.to_bytes(3, 'little'), n, a, b) for op, k, n, a, b in out))
    return blob.getvalue()


def trace(args):
    if not (BUNDLE / 'manifest.json').exists():
        raise SystemExit('no bundle: dev/lab/pizdos.py build NAME=SPEC ... first')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    pizdos = LAB / 'src' / manifest['pizdos']
    work = LAB / 'trace'
    wasm2o = tracer(pizdos, LAB / 'var' / manifest['variants'][0]['name'] / 'libstr.o', work)
    raws = []
    for w in manifest['works']:
        if not selected(w['name'], args.only):
            continue
        argv = [a.replace('{wasm2o}', str(wasm2o)) for a in w['argv']]
        if '--repeat' in argv:
            argv[argv.index('--repeat') + 1] = '1'   # one pass over the files
        out = work / (w['name'].replace('/', '_') + '.bin')
        out.unlink(missing_ok=True)
        with open(BUNDLE / w['stdin'] if w['stdin'] else os.devnull, 'rb') as i:
            sh(argv, cwd=BUNDLE, stdin=i, stdout=subprocess.DEVNULL, env=dict(ENV, W2O_STR_TRACE=str(out)))
        raws.append((w['name'], out.read_bytes()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(workload(raws))
    print('%s: %d traces, %.1f MB' % (args.out, len(raws), args.out.stat().st_size / 1e6))


# ------------------------------------------------------------- on a host

def selected(work, only):
    return not only or any(work == o or work.startswith(o + '/') for o in only.split(','))


def remote(args):
    """In the bundle: every program with every wasm2o in turns, run after run, a line of JSON per call."""
    manifest = json.loads(Path('manifest.json').read_text())
    os.sched_setaffinity(0, {args.cpu})
    if os.geteuid() == 0:
        os.nice(-5)
    names = [v['name'] for v in manifest['variants']]
    works = [w for w in manifest['works'] if selected(w['name'], args.only)]
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for run in range(args.runs):
        for work in works:
            # the first wasm2o of a run's turn moves on by one each run
            order = names[run % len(names):] + names[:run % len(names)]
            for name in order:
                argv = [a.replace('{wasm2o}', 'variants/%s/wasm2o' % name) for a in work['argv']]
                counts = tmp / 'count.json'
                counts.unlink(missing_ok=True)
                with open(work['stdin'] or os.devnull, 'rb') as i, open(tmp / 'out', 'wb') as o, open(tmp / 'err', 'wb') as e:
                    rc = subprocess.run(['./count', '--json', str(counts), '--', *argv], stdin=i, stdout=o, stderr=e, env=ENV).returncode
                row = dict(run=run, work=work['name'], variant=name)
                row.update(json.loads(counts.read_text()) if counts.exists() else dict(cycles=0, instructions=0, status=rc))
                err = (tmp / 'err').read_text(errors='replace')
                t = re.findall(r'^time ([0-9.]+)$', err, re.M)
                row.update(digest=digest(tmp / 'out'), time=float(t[-1]) if t else None)
                if rc:
                    row['error'] = err[-2000:]
                print(json.dumps(row), flush=True)


# ------------------------------------------------------------------- run

def run_host(name, tag, args, info, failed):
    from run import HOSTS, IDLEST, STATE, ssh  # the lab's hosts as run.py has them

    h = HOSTS[name]
    out = ROOT / '.lab' / tag / name
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    hold = args.clock == 'hold'
    opts = ['--runs', str(args.runs)] + (['--only', args.only] if args.only else [])
    try:
        if not h['ssh']:
            cpu = h['cpu']
            freq, state = [sys.executable, str(HERE / 'freq.py')], ROOT / '.lab/freq.state'
            log = subprocess.run(['sudo', *freq, 'fix' if hold else 'restore', '--state', str(state)], capture_output=True, text=True)
            (out / 'freq.log').write_text(log.stdout + log.stderr)
            try:
                with open(out / 'pizdos.jsonl', 'w') as o, open(out / 'err.log', 'w') as e:
                    sh([sys.executable, 'pizdos.py', 'remote', '--cpu', str(cpu), '--tmp', str(out / 'tmp'), *opts], cwd=BUNDLE, stdout=o, stderr=e)
            finally:
                if hold:
                    subprocess.run(['sudo', *freq, 'restore', '--state', str(state)], stdout=open(out / 'freq.log', 'a'), stderr=subprocess.STDOUT)
            shutil.rmtree(out / 'tmp', ignore_errors=True)
        else:
            host, bundle, rdir = h['ssh'], '/root/lab/pizdos/' + info['bundle'], '/root/lab/' + tag
            have = subprocess.run(ssh(host) + ['test -f %s/.complete' % bundle]).returncode == 0
            if not have:
                print('%s: shipping the bundle' % name, flush=True)
                tar = subprocess.Popen(['tar', '-C', str(BUNDLE), '-cf', '-', '.'], stdout=subprocess.PIPE)
                sh(ssh(host) + ['rm -rf %s.part && mkdir -p %s.part && tar -C %s.part -xf - && rm -rf %s && mv %s.part %s && touch %s/.complete' % (
                    (bundle,) * 7)], stdin=tar.stdout)
                if tar.wait():
                    raise RuntimeError('tar of the bundle failed')
                # the three latest bundles stay: 135 MB each
                sh(ssh(host) + ['cd /root/lab/pizdos && ls -t | tail -n +4 | xargs -r rm -rf'])
            sh(ssh(host) + ['mkdir -p %s && rm -f %s/pizdos.jsonl' % (rdir, rdir)])
            with open(HERE / 'freq.py', 'rb') as f:
                sh(ssh(host) + ['cat > /root/lab/freq.py'], stdin=f)
            cpu = subprocess.run(ssh(host) + [IDLEST], capture_output=True, text=True, check=True).stdout.strip()
            back = 'python3 /root/lab/freq.py restore --state %s >> %s/freq.log 2>&1' % (STATE, rdir)
            inner = ('trap "%s; exit 129" HUP INT TERM; ' % back +
                     'python3 /root/lab/freq.py %s --state %s > %s/freq.log 2>&1; ' % ('fix' if hold else 'restore', STATE, rdir) +
                     'cd %s && python3 pizdos.py remote --cpu %s --tmp %s/tmp %s > %s/pizdos.jsonl 2> %s/err.log; rc=$?; ' % (
                         bundle, cpu, rdir, ' '.join(opts), rdir, rdir) +
                     '%s; exit $rc' % (back if hold else 'true'))
            print('%s: running on cpu %s' % (name, cpu), flush=True)
            try:
                # the session stays open for the run: the lab hosts end a session's processes with it
                rc = subprocess.run(ssh(host) + ['sh -c %s' % shlex.quote(inner)]).returncode
            finally:
                if hold:
                    subprocess.run(ssh(host) + ['python3 /root/lab/freq.py restore --state %s >> %s/freq.log 2>&1' % (STATE, rdir)])
            for rname in ('pizdos.jsonl', 'err.log', 'freq.log'):
                with open(out / rname, 'wb') as f:
                    sh(ssh(host) + ['cat %s/%s' % (rdir, rname)], stdout=f)
            if rc:
                raise RuntimeError('pizdos.py remote exited with %s, see %s' % (rc, out / 'err.log'))
        (out / 'run.json').write_text(json.dumps(dict(info, host=name, note=h['note'], cpu=cpu, started=started,
                                                      ended=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2) + '\n')
        print('%s: done -> %s' % (name, out), flush=True)
    except Exception as e:  # noqa: BLE001
        failed[name] = repr(e)
        print('%s: FAILED %r' % (name, e), flush=True)


def run(args):
    sys.path.insert(0, str(HERE))
    if not (BUNDLE / 'manifest.json').exists():
        raise SystemExit('no bundle: dev/lab/pizdos.py build NAME=SPEC ... first')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = dict(tag=args.tag, commit=commit, bundle=bundle_id(), pizdos=manifest['pizdos'], variants=manifest['variants'], runs=args.runs,
                only=args.only, clock=args.clock)
    failed = {}
    threads = [threading.Thread(target=run_host, args=(n, args.tag, args, info, failed)) for n in args.hosts.split(',')]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('failed: %s' % (failed or 'none'))
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------- report

def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def report(args):
    for host in args.hosts.split(','):
        calls = collections.defaultdict(dict)  # (tag, run, work) -> variant -> row
        meta = None
        for d in sorted(glob.glob(str(ROOT / '.lab' / args.pattern / host))):
            p = Path(d)
            if not (p / 'pizdos.jsonl').exists() or not (p / 'run.json').exists():
                continue
            meta = json.loads((p / 'run.json').read_text())
            for line in (p / 'pizdos.jsonl').read_text().splitlines():
                r = json.loads(line)
                calls[(p.parent.name, r['run'], r['work'])][r['variant']] = r
        if not calls:
            continue
        names = [v['name'] for v in meta['variants']]
        base = args.base or names[0]
        others = [n for n in names if n != base]
        values = collections.defaultdict(list)
        ratios = collections.defaultdict(list)
        bad = set()
        for (tag, run, work), by in calls.items():
            if any(r.get('error') or r['status'] for r in by.values()):
                bad.add('%s: exit %s' % (work, ', '.join('%s %s' % (n, r['status']) for n, r in by.items() if r['status'])))
                continue
            if len({r['digest'] for r in by.values()}) > 1:
                bad.add('%s: outputs differ' % work)
            for n, r in by.items():
                if r[args.metric] is not None:
                    values[(work, n)].append(r[args.metric])
            if base in by and by[base][args.metric]:
                for n in others:
                    if n in by and by[n][args.metric] is not None:
                        ratios[(work, n)].append(by[n][args.metric] / by[base][args.metric])
        works = list(dict.fromkeys(w for (_, _, w) in calls))
        print('%s: %s, base %s, %d runs' % (host, meta['note'], base, len({(t, r) for (t, r, _) in calls})))
        unit = {'cycles': ' Mcyc', 'instructions': ' Mins'}.get(args.metric, ' ms')
        scale = 1e6 if args.metric in ('cycles', 'instructions') else 1e-3
        print('  %-20s %12s' % ('program', base + unit) + ''.join('  %-22s' % ('%s/%s' % (n, base)) for n in others))
        summary = collections.defaultdict(list)
        py = collections.defaultdict(list)
        for w in works:
            if not values[(w, base)]:
                continue
            line = '  %-20s %12.1f' % (w, statistics.median(values[(w, base)]) / scale)
            for n in others:
                rs = ratios[(w, n)]
                if not rs:
                    line += '  %-22s' % '-'
                    continue
                m = statistics.median(rs)
                line += '  %-22s' % ('%.4f (%.3f-%.3f)' % (m, min(rs), max(rs)))
                (py if w.startswith('py/') else summary)[n].append(m)
            print(line)
        line = '  %-20s %12s' % ('geomean', '')
        for n in others:
            ms = summary[n] + ([geomean(py[n])] if py[n] else [])
            line += '  %-22s' % ('%.4f' % geomean(ms) if ms else '-')
        print(line)
        for b in sorted(bad):
            print('  WARNING %s' % b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('variants', nargs='+', metavar='NAME=SPEC')
    r = sub.add_parser('run')
    r.add_argument('--tag', required=True)
    r.add_argument('--hosts', default='local,lab2,lab3')
    r.add_argument('--runs', type=int, default=5)
    r.add_argument('--only', default='', help='programs, comma-separated: aot, im, pdf, djvu, svg, magic, py or py/NAME')
    r.add_argument('--clock', choices=('hold', 'free'), default='hold')
    p = sub.add_parser('report')
    p.add_argument('pattern')
    p.add_argument('--base')
    p.add_argument('--hosts', default='local,lab2,lab3')
    p.add_argument('--metric', choices=('cycles', 'instructions', 'wall', 'user', 'time'), default='cycles',
                   help="time: the driver's own time of its work (the decoders and CPython print it)")
    t = sub.add_parser('trace')
    t.add_argument('--only', default='', help='programs, as for run')
    t.add_argument('--out', type=Path, default=LAB / 'traces.bin')
    m = sub.add_parser('remote')
    m.add_argument('--cpu', type=int, required=True)
    m.add_argument('--runs', type=int, required=True)
    m.add_argument('--only', default='')
    m.add_argument('--tmp', required=True)
    args = ap.parse_args()
    dict(build=build, run=run, report=report, trace=trace, remote=remote)[args.command](args)


if __name__ == '__main__':
    main()
