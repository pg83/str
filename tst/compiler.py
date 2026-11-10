#!/usr/bin/env python3
"""The compiler (dev/opt) against itself: every unit's IR verified after
every pass, every function's code decoded by dev/x86.py (all of it, in
instructions the decoder knows), the archive's objects read back and their
symbols resolved among themselves (nothing outside the library is called),
and every configuration's table pointing at a copy that exists.

    compiler.py --archive .build/libstr.a"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = {'malloc'}  # the C library's: strdup's and strndup's allocation, the one name the archive takes from outside
sys.path.insert(0, str(ROOT / 'dev'))
os.environ['OPT_VERIFY'] = '1'
import x86  # noqa: E402
from elf import read as read_elf  # noqa: E402
from opt.codegen import compile_function  # noqa: E402
from opt.library import UNITS, read_configs  # noqa: E402
from opt.module import Module  # noqa: E402
from opt.pipeline import optimize  # noqa: E402

failures = 0


def fail(message):
    global failures
    failures += 1
    print('FAIL ' + message)


def members(archive):
    """(name, bytes) of an ar archive's members (GNU long names included)."""
    data = Path(archive).read_bytes()
    if data[:8] != b'!<arch>\n':
        raise ValueError('%s: not an archive' % archive)
    at, names, out = 8, b'', []
    while at + 60 <= len(data):
        head = data[at:at + 60]
        name, size = head[:16].decode().rstrip(), int(head[48:58])
        body = data[at + 60:at + 60 + size]
        if name == '//':
            names = body
        elif name != '/':
            if name.startswith('/'):
                start = int(name[1:])
                name = names[start:names.index(b'/\n', start)].decode()
            out.append((name.rstrip('/'), body))
        at += 60 + size + (size & 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    args = ap.parse_args()
    # the units through the passes (verified after each) and the selection; the code decodable
    modules = {}
    for unit, isa in UNITS:
        m = Module(ROOT / (unit + '.c'), isa, None)
        optimize(m)
        modules[unit] = m
        for fn in m.emitted():
            mf, code, relocs = compile_function(fn, isa)
            try:
                decoded = list(x86.decode(code, 0))
            except x86.DecodeError as e:
                fail('%s: %s' % (fn.name, e))
                continue
            if sum(len(bytes.fromhex(j['bytes'])) for j in decoded) != len(code):
                fail('%s: the decoder does not cover the code' % fn.name)
            for offset, rtype, symbol, addend in relocs:
                if not 0 <= offset <= len(code) - 4:
                    fail('%s: relocation at %#x outside the code' % (fn.name, offset))
        print('%s: %d functions' % (unit, len(m.emitted())))
    # the archive: its objects, their symbols resolved within it (but EXTERNAL)
    defined, needed, where = {}, {}, {}
    for name, body in members(args.archive):
        obj = read_elf_bytes(body, name)
        for s in obj.symbols:
            if s.shndx == 0 and s.name:
                needed.setdefault(s.name, set()).add(name)
            elif s.name and s.bind in ('STB_GLOBAL', 'STB_WEAK') and s.type in ('STT_FUNC', 'STT_OBJECT', 'STT_NOTYPE'):
                if s.name in defined:
                    fail('%s: defined in %s and %s' % (s.name, defined[s.name], name))
                defined[s.name] = name
            if s.bind == 'STB_WEAK':
                fail('%s: weak symbol %s' % (name, s.name))
        where[name] = obj
    for sym, users in sorted(needed.items()):
        if sym not in defined and sym not in EXTERNAL:
            fail('%s: undefined, wanted by %s' % (sym, ', '.join(sorted(users))))
    # the configurations' tables: every slot a copy that exists
    configs, initial, slots = read_configs(modules['dispatch'])
    for config in configs:
        for slot, fn in config.slots.items():
            copy = '%s.%s' % (config.name, fn)
            if copy not in defined:
                fail('%s: slot %s wants %s' % (config.name, slot, copy))
    print('%d objects, %d configurations, %d symbols defined' % (len(where), len(configs), len(defined)))
    print('compiler: %s' % ('ok' if not failures else '%d failures' % failures))
    sys.exit(1 if failures else 0)


def read_elf_bytes(body, name):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='-' + name, delete=False) as f:
        f.write(body)
        path = f.name
    try:
        return read_elf(path)
    finally:
        os.unlink(path)


if __name__ == '__main__':
    main()
