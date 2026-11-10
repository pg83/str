#!/usr/bin/env python3
"""The library built for wasm2o (dev/opt/mir.py, WASM2O) against what wasm2o's loader and
the code it compiles rely on (pizdos, bin/wasm2o): one .text, one .rodata and an .eh_frame,
nothing outside, only PC32 and 64-bit relocations; every function an FDE whose CFA follows
the code at every instruction (rsp + 8 at the entry and at every return, every push, pop and
adjustment of rsp counted, a jump landing where its target's CFA is); no instruction naming
rbp, r13, r14, r15 or vector register 15; rbx and r9-r12 pushed by the prologue of a function
that uses them; no access of the stack needing more alignment than its 8 bytes; and
str_choose there, for the loader to call.

    wasm2o.py --object .build/wasm2o/post/libstr.o"""
import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'dev'))
import x86  # noqa: E402
from elf import read as read_elf  # noqa: E402

RESERVED = re.compile(r'^(rbp|ebp|bp|bpl|r1[345][dwb]?|[xyz]mm15)$')
SAVED = {'rbx': 'rbx', 'ebx': 'rbx', 'bx': 'rbx', 'bl': 'rbx'}
for n in range(9, 13):
    SAVED.update({'r%d%s' % (n, s): 'r%d' % n for s in ('', 'd', 'w', 'b')})
# what may address the stack: rsp is aligned to 8 there, and the legacy SSE operations and the
# aligned moves fault on a memory operand that is not aligned to its size
ALIGNED = re.compile(r'^(v?movdqa(32|64)?|v?movap[sd]|v?movntdq[a]?|v?movntp[sd])$')
LEGACY_SSE = re.compile(r'^(p[a-z]+|movdq[au]|movap[sd]|movup[sd]|pcmpistr[im])$')

failures = 0


def fail(message):
    global failures
    failures += 1
    print('FAIL ' + message)


def uleb(data, at):
    x = shift = 0
    while True:
        b = data[at]
        at += 1
        x |= (b & 0x7f) << shift
        shift += 7
        if not b & 0x80:
            return x, at


def fdes(elf, eh_index, text_index):
    """{start: (size, [(offset, cfa)])} from .eh_frame: the CIE's CFA, then each FDE's changes."""
    data = elf.sections[eh_index].data
    relocs = {r.offset: r for r in elf.relocs.get(eh_index, [])}
    out, cies, at = {}, {}, 0
    while at + 4 <= len(data):
        length = struct.unpack_from('<I', data, at)[0]
        if length == 0:
            break
        body = data[at + 4:at + 4 + length]
        pointer = struct.unpack_from('<I', body, 0)[0]
        if pointer == 0:
            if body[4:8] != b'\x01zR\x00' or body[8:13] != b'\x01\x78\x10\x01\x1b' or body[13:16] != b'\x0c\x07\x08':
                fail('the CIE at %#x is not the one the loader reads' % at)
            cies[at] = 8
        else:
            cie = at + 4 - pointer
            r = relocs.get(at + 8)
            if cie not in cies or r is None or r.type != 'R_X86_64_PC32' or elf.symbols[r.symbol].shndx != text_index:
                fail('the FDE at %#x has no CIE or no start in .text' % at)
            else:
                start = elf.symbols[r.symbol].value + r.addend
                size = struct.unpack_from('<I', body, 8)[0]
                aug, k = uleb(body, 12)
                k += aug
                loc, cfa, changes = 0, cies[cie], [(0, cies[cie])]
                while k < len(body):
                    op = body[k]
                    k += 1
                    if op & 0xc0 == 0x40:
                        loc += op & 0x3f
                    elif op == 0x02:
                        loc += body[k]
                        k += 1
                    elif op == 0x03:
                        loc += struct.unpack_from('<H', body, k)[0]
                        k += 2
                    elif op == 0x04:
                        loc += struct.unpack_from('<I', body, k)[0]
                        k += 4
                    elif op == 0x0e:
                        cfa, k = uleb(body, k)
                        changes.append((loc, cfa))
                    elif op == 0:
                        continue
                    else:
                        fail('the FDE of %#x: CFA instruction %#x' % (start, op))
                        break
                    if loc >= size:
                        fail('the FDE of %#x changes the CFA past its range' % start)
                if start in out:
                    fail('two FDEs at %#x' % start)
                out[start] = (size, changes)
        at += 4 + length
    return out


def cfa_at(changes, offset):
    cfa = changes[0][1]
    for at, value in changes:
        if at <= offset:
            cfa = value
    return cfa


def regs_of(j):
    out = []
    for o in j['ops']:
        if o['kind'] == 'Register' and not o.get('implicit'):
            out.append(o['reg'])
        elif o['kind'] == 'Memory':
            out += [r for r in (o['mem']['base'], o['mem']['index']) if r not in ('none', None)]
    return out


def check_function(name, start, code, changes):
    insns = x86.decode(code, start)
    at_offset = {j['offset']: j for j in insns}
    pushed, prologue = set(), True
    for j in insns:
        here = j['offset'] - start
        cfa = cfa_at(changes, here)
        after = cfa_at(changes, here + j['len']) if here + j['len'] < len(code) else None
        m = j['mnemonic']
        regs = regs_of(j)
        for r in regs:
            if RESERVED.match(r):
                fail('%s+%#x: %s names %s' % (name, here, j['gas'], r))
        if prologue and m == 'push':
            pushed.add(SAVED.get(regs[0], regs[0]))
        elif m != 'push':
            prologue = False
        if m not in ('push', 'pop'):
            for r in regs:
                if r in SAVED and SAVED[r] not in pushed:
                    fail('%s+%#x: %s uses %s, which the prologue does not save' % (name, here, j['gas'], SAVED[r]))
        for o in j['ops']:
            if o['kind'] == 'Memory' and o['mem']['base'] == 'rsp' and (ALIGNED.match(m) or LEGACY_SSE.match(m) and not m.startswith('movdqu') and not m.startswith('movup')):
                fail('%s+%#x: %s needs an aligned stack' % (name, here, j['gas']))
        step = None
        if m == 'push':
            step = 8
        elif m == 'pop':
            step = -8
        elif m in ('sub', 'add') and regs[:1] == ['rsp']:
            imm = j['ops'][1]
            step = imm['imm'] if m == 'sub' else -imm['imm']
        elif 'rsp' in regs and j['ops'][0]['kind'] == 'Register' and j['ops'][0]['reg'] == 'rsp' and m not in ('cmp', 'test'):
            fail('%s+%#x: %s writes rsp' % (name, here, j['gas']))
        if m == 'ret' and cfa != 8:
            fail('%s+%#x: a return with the CFA at rsp+%d' % (name, here, cfa))
        if step is not None and after is not None and after != cfa + step:
            fail('%s+%#x: %s moves rsp by %d, the CFA by %d' % (name, here, j['gas'], step, after - cfa))
        if step is None and m not in ('ret', 'jmp', 'ud2', 'int3') and after is not None and after != cfa:
            fail('%s+%#x: the CFA changes at %s' % (name, here, j['gas']))
        if m.startswith('j') and j['ops'] and j['ops'][0]['kind'] == 'NearBranch':
            target = j['ops'][0]['branch']
            if start <= target < start + len(code):
                if target not in at_offset:
                    fail('%s+%#x: a jump into an instruction' % (name, here))
                elif cfa_at(changes, target - start) != cfa:
                    fail('%s+%#x: a jump from CFA rsp+%d to rsp+%d' % (name, here, cfa, cfa_at(changes, target - start)))
            elif m == 'jmp' and cfa != 8:
                fail('%s+%#x: a tail jump with the CFA at rsp+%d' % (name, here, cfa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--object', type=Path, required=True)
    args = ap.parse_args()
    elf = read_elf(args.object)
    names = {s.name: s.index for s in elf.sections}
    alloc = sorted(s.name for s in elf.sections if s.flags & 2)
    if alloc != ['.eh_frame', '.rodata', '.text']:
        fail('allocated sections %s, not .text, .rodata and .eh_frame' % alloc)
    text, eh = names.get('.text'), names.get('.eh_frame')
    for index, relocs in elf.relocs.items():
        for r in relocs:
            if r.type not in ('R_X86_64_PC32', 'R_X86_64_64'):
                fail('%s+%#x: a relocation %s' % (elf.sections[index].name, r.offset, r.type))
            if elf.symbols[r.symbol].shndx == 0:
                fail('%s+%#x: a relocation to %s, outside the object' % (elf.sections[index].name, r.offset, elf.symbols[r.symbol].name))
    frames = fdes(elf, eh, text)
    code = elf.sections[text].data
    functions = sorted((s.value, s.size, s.name) for s in elf.symbols if s.type == 'STT_FUNC' and s.shndx == text)
    if sorted(frames) != [start for start, _, _ in functions]:
        fail('%d functions, %d FDEs, not one at each entry' % (len(functions), len(frames)))
    for start, size, name in functions:
        frame = frames.get(start)
        if frame is None:
            continue
        if frame[0] != size:
            fail('%s: the FDE covers %d bytes of %d' % (name, frame[0], size))
        check_function(name, start, code[start:start + size], frame[1])
    if not any(s.name == 'str_choose' and s.type == 'STT_FUNC' and s.bind == 'STB_GLOBAL' for s in elf.symbols):
        fail('no global str_choose')
    print('%d functions, %d FDEs: %d failures' % (len(functions), len(frames), failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
