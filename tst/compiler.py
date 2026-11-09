#!/usr/bin/env python3
"""Frontend, SSA and native object emitter checks (no system C headers)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'dev'))
from cc import types as T  # noqa: E402
from cc.front import Parser, const  # noqa: E402
from cc.abi import trampolines  # noqa: E402
from cc.emit import emit_object  # noqa: E402
from cc.data import emit_data  # noqa: E402
from cc.lower import lower_unit  # noqa: E402
from cc.opt import optimize  # noqa: E402
from cc.source import Constant, Error, Preprocessor, lex  # noqa: E402
from objfile import Object, Reloc, Section, Symbol  # noqa: E402
import elf  # noqa: E402
import elfout  # noqa: E402
import x86  # noqa: E402

ARCHIVE = None


def parse(text):
    return Parser(Preprocessor().text(text)).parse()


class Frontend(unittest.TestCase):
    def test_library(self):
        for path in sorted(ROOT.glob('*.c')):
            with self.subTest(unit=path.stem):
                pp = Preprocessor()
                unit = Parser(pp.process(path), str(path)).parse()
                self.assertGreater(len(unit.functions), 1)
                self.assertEqual(unit.records['str_table'].size, 160)
                self.assertEqual(unit.records['str_config'].size, 176)
                lower_unit(unit)  # includes SSA dominance and phi-edge checks

    def test_macro_rescan_and_hiding(self):
        pp = Preprocessor()
        ts = pp.text('#define A B\n#define B(x) (x + 1)\n#define R R\nA(B(2)) R\n')
        self.assertEqual(' '.join(t.text for t in ts), '( ( 2 + 1 ) + 1 ) R')

    def test_conditions(self):
        for text, answer in [('0 && (1/0)', 0), ('1 || (1/0)', 1), ('1 ? 7 : 1/0', 7),
                             ('0 ? 1/0 : 9', 9), ('-7/3', -2), ('-7%3', -1)]:
            self.assertEqual(Constant(lex(text)).parse(), answer)
        ts = Preprocessor().text('#if defined(__has_builtin) && __has_builtin(x)\nyes\n#else\nno\n#endif\n')
        self.assertEqual([t.text for t in ts], ['yes'])

    def test_declarators_and_layout(self):
        unit = parse('struct S { char a; int b; void *(*f)(int, const char *); }; '
                     'int *f(int x); int (*g)(int); int a[3][4];')
        by_name = {d.name: d.type for d in unit.declarations}
        self.assertEqual(by_name['f'].kind, 'function')
        self.assertEqual(by_name['f'].base, T.pointer(T.INT))
        self.assertEqual(by_name['g'].kind, 'ptr')
        self.assertEqual(by_name['g'].base.kind, 'function')
        self.assertEqual(by_name['a'].size, 48)
        self.assertEqual(unit.records['S'].fields[1][2], 4)
        self.assertEqual(unit.records['S'].fields[2][2], 8)
        self.assertEqual(unit.records['S'].size, 16)

    def test_integer_semantics(self):
        unit = parse('unsigned a = 0xffffffffu + 1; long b = 0xffffffff; '
                     'int c = (int)0x80000000u >> 31; int d = (unsigned char)511;')
        values = {d.name: const(d.init.expr) for d in unit.declarations}
        self.assertEqual(values, {'a': 0, 'b': 4294967295, 'c': -1, 'd': 255})

    def test_intrinsic_immediate(self):
        with self.assertRaisesRegex(Error, 'must be constant'):
            parse('__m256i f(__m256i x, int n) { return _mm256_srli_epi16(x, n); }')
        with self.assertRaisesRegex(Error, 'expected 2 arguments'):
            parse('__m128i f(__m128i x) { return _mm_cmpeq_epi8(x); }')

    def test_diagnostics(self):
        for text, error in [('int f(void) { return missing; }', 'undeclared'),
                            ('int f(void) { const int x = 1; x = 2; return x; }', 'modifiable'),
                            ('int f(void) { goto missing; }', 'undefined label'),
                            ('int f(void) { int a[3]; return a; }', 'cannot convert')]:
            with self.subTest(source=text), self.assertRaisesRegex(Error, error):
                parse(text)

    def test_loop_phi_and_address_escape(self):
        unit = parse('int f(int n) { int sum = 0; for (int i=0; i<n; i++) sum += i; return sum; } '
                     'int g(int *p) { int x=7; *p = x; return x; } '
                     'void escape(int *); int h(void) { int x=3; escape(&x); return x; }')
        fs = {f.name: f for f in lower_unit(unit)}
        phis = [op for b in fs['f'].blocks for op in b.ops if op.kind == 'phi']
        self.assertEqual(len(phis), 2)
        self.assertEqual(fs['f'].allocas, [])
        self.assertEqual(fs['g'].allocas, [])
        self.assertEqual(len(fs['h'].allocas), 1)
        for f in fs.values():
            f.verify()

    def test_compound_lvalue_evaluated_once(self):
        fn = optimize(lower_unit(parse('int f(int *p) { int i=0; p[i++] += 3; return i; }')))[0]
        returns = [fn.defs[b.term.args[0]] for b in fn.blocks if b.term.kind == 'ret']
        self.assertEqual(len(returns), 1)
        self.assertEqual((returns[0].kind, returns[0].attrs['value']), ('constant', 1))

    def test_stack_aggregate_scalarization(self):
        fs = optimize(lower_unit(parse(
            'int f(int n) { int a[2]; a[0]=3; a[1]=4; if(n) a[1]=8; return a[0]+a[1]; } '
            'void escape(int *); int g(void) { int a[2]; a[0]=3; escape(a); return a[0]; } '
            'int h(void) { int a[2]; a[0]=0x1234; ((unsigned char *)a)[0]=7; return a[0]; }')))
        self.assertEqual(fs[0].allocas, [])
        self.assertEqual(len(fs[1].allocas), 1)
        self.assertEqual(len(fs[2].allocas), 1)
        for fn in fs:
            fn.verify()

    def test_alignment_cycles(self):
        from cc.alignment import alignments
        fs = lower_unit(parse('unsigned shrink(unsigned n) { unsigned x=16; while(n--) x>>=1; return x; } '
                              'unsigned stride(unsigned n) { unsigned x=16; while(n--) x+=2; return x; }'))
        for fn, expected in zip(fs, [1, 2]):
            facts = alignments(fn)
            returned = next(b.term.args[0] for b in fn.blocks if b.term.kind == 'ret')
            self.assertEqual(facts[returned], expected, fn.name)


class Native(unittest.TestCase):
    def test_sse2_copy_reference(self):
        if ARCHIVE is None:
            self.skipTest('pass --archive for native compiler byte comparison')
        from post import read_archive
        pp = Preprocessor()
        source = Parser(pp.process(ROOT / 'sse2.c')).parse()
        fs = optimize(lower_unit(source))
        dispatch = Parser(Preprocessor().process(ROOT / 'dispatch.c')).parse()
        entries = trampolines(optimize(lower_unit(dispatch)))
        selected = [f for f in fs if f.name in ('memcpy_sse2', 'memmove_sse2', 'memset_sse2', 'strcat_sse2')]
        with tempfile.TemporaryDirectory() as where:
            path, ref = Path(where) / 'native.o', Path(where) / 'reference.o'
            with patch.object(elf, 'read', side_effect=AssertionError('compiler must not parse ELF')), \
                 patch.object(x86, 'decode', side_effect=AssertionError('compiler must not decode machine code')):
                path.write_bytes(emit_object(selected, ROOT / 'sse2.c', trampolines=entries))
            ref.write_bytes(dict(read_archive(ARCHIVE))['sse2.c.o'])
            a, b = Object(path), Object(ref)
            for f in selected:
                section = '.text.' + f.name
                self.assertEqual(a.sections[section].data, b.sections[section].data, section)
                self.assertEqual(a.sections[section].relocs, b.sections[section].relocs, section)

    def test_static_data_and_constructor(self):
        unit = parse('int f(void) { return 0; } struct S { char c; long n; int (*f)(void); }; '
                     'const struct S table[] = {{.n=7, .f=f, .c=3}}; '
                     'const char *name = "str"; unsigned zeros[4]; '
                     '__attribute__((constructor)) static void init(void) {}')
        obj = Object()
        emit_data(obj, unit)
        data = obj.sections['.rodata.table']
        self.assertEqual(len(data.data), 24)
        self.assertEqual(data.data[0], 3)
        self.assertEqual(int.from_bytes(data.data[8:16], 'little'), 7)
        self.assertEqual(data.relocs, [Reloc(16, 'R_X86_64_64', 'f', 0)])
        self.assertEqual(obj.sections['.rodata.str1.1'].data, b'str\0')
        self.assertEqual(obj.sections['.bss.zeros'].nobits_size, 16)
        self.assertEqual(obj.sections['.init_array'].relocs, [Reloc(0, 'R_X86_64_64', '.text.init', 0, True)])

    def test_full_library_without_tools_or_decoder(self):
        from cc.library import compile_library
        with tempfile.TemporaryDirectory() as where:
            archive = Path(where) / 'libstr.a'
            with patch('subprocess.run', side_effect=AssertionError('native build must not run an external tool')), \
                 patch('subprocess.Popen', side_effect=AssertionError('native build must not start a tool')), \
                 patch.object(elf, 'read', side_effect=AssertionError('native build must not parse ELF')), \
                 patch.object(x86, 'decode', side_effect=AssertionError('native build must not decode machine code')), \
                 patch.object(x86, 'decode_one', side_effect=AssertionError('native build must not decode an instruction')):
                result = compile_library(ROOT, archive)
            self.assertEqual(len(result['objects']), 11)
            self.assertTrue(archive.is_file())
            dispatch = Object(archive.parent / 'post/dispatch.c.o')
            self.assertEqual(len(dispatch.sections['.data.str_table'].data), 160)
            self.assertEqual(len(dispatch.sections['.rodata.str_configs'].data), 1056)
            if ARCHIVE is not None:
                from post import read_archive
                reference = dict(read_archive(ARCHIVE))
                actual = dict(read_archive(archive))
                self.assertEqual(set(actual), set(reference))
                for member in reference:
                    with self.subTest(member=member):
                        self.assertEqual(actual[member], reference[member])

    def test_conditional_tail_fallthrough(self):
        import ctypes
        import mmap
        import layout
        from cc import machine as M
        from cc.late import conditional_tails
        eax, edi = M.R(M.physical('rax'), 32), M.R(M.physical('rdi'), 32)
        for intervening in (False, True):
            with self.subTest(intervening=intervening), tempfile.TemporaryDirectory() as where:
                items = [M.Instr('cmp', (edi, M.Imm(0)), ('R', 'R')),
                         M.Instr('je', target=True, cc='e'),
                         M.Instr('jmp', (M.Target('callback'),), ('R',), callee='callback')]
                if intervening:
                    items.extend([M.Instr('mov', (eax, M.Imm(7)), ('W', 'R')), M.Instr('ret')])
                noncall = len(items)*16
                items.extend([M.Instr('mov', (eax, M.Imm(42)), ('W', 'R')), M.Instr('ret')])
                instructions = [M.structured(ins, n*16, noncall if n == 1 else None) for n, ins in enumerate(items)]
                instructions[2].j['shrinkwrap_tail'] = True
                section = '.text.branch'
                obj = Object()
                obj.sections[section] = Section(section, 'SHT_PROGBITS', 6, 16, bytes(len(items)*16))
                obj.symbols = [Symbol('branch', 0, len(items)*16, 'STT_FUNC', 'STB_GLOBAL', section)]
                conditional_tails(obj, {section: instructions})
                layouts = {section: layout.layout(obj, section, instructions, func_align=6, jcc_align=False)}
                remap = layout.resolve_relocs(obj, layouts)
                path = Path(where) / 'tail.o'
                path.write_bytes(elfout.write_model(obj, None, layouts, remap))
                emitted = Object(path).sections[section]
                code = bytearray(emitted.data)
                stub = len(code)
                code.extend(bytes.fromhex('b863000000c3'))
                for rel in emitted.relocs:
                    self.assertEqual((rel.type, rel.symbol), ('R_X86_64_PLT32', 'callback'))
                    code[rel.offset:rel.offset+4] = (stub+rel.addend-rel.offset).to_bytes(4, 'little', signed=True)
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(address)
                    self.assertEqual([call(x) for x in (0, 1, 17)], [42, 99, 99])

    def test_scalar_code(self):
        source = 'int identity(int x) { return x; } int answer(void) { return 42; } unsigned char byte(unsigned char *p) { return *p; }'
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'scalar.o'
            with patch.object(elf, 'read', side_effect=AssertionError('compiler must not parse ELF')), \
                 patch.object(x86, 'decode', side_effect=AssertionError('compiler must not decode machine code')):
                data = emit_object(fs, Path('scalar.c'))
            path.write_bytes(data)
            obj = Object(path)
            self.assertEqual(obj.sections['.text.identity'].data.hex(), '89f8c3')
            self.assertEqual(obj.sections['.text.answer'].data.hex(), 'b82a000000c3')
            self.assertEqual(obj.sections['.text.byte'].data.hex(), '0fb607c3')

    def test_scalar_immediate_boundaries(self):
        import ctypes
        import mmap
        fs = optimize(lower_unit(parse('unsigned long f(unsigned long x) { return x <= 128; } '
                                       'unsigned long g(unsigned long x) { return x & ~15ul; } '
                                       'unsigned long choose(unsigned long x) { return x ? 7 : 42; } '
                                       'int guard(int *p) { return p && *p; } '
                                       'unsigned flags(unsigned x) { '
                                       'if (!((x&1) && ((x>>9)&1) && ((x>>19)&1) && ((x>>20)&1) '
                                       '&& ((x>>23)&1) && ((x>>13)&1))) return 7; return 15; }')))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'scalar.o'
            path.write_bytes(emit_object(fs, Path('scalar.c')))
            obj = Object(path)
            pointee = ctypes.c_int(3)
            for name, inputs, expected in [('f', [0, 127, 128, 129, (1 << 64) - 1], [1, 1, 1, 0, 0]),
                                           ('g', [0, 15, 16, 31, (1 << 64) - 1], [0, 0, 16, 16, (1 << 64) - 16]),
                                           ('choose', [0, 1, (1 << 64) - 1], [42, 7, 7]),
                                           ('guard', [0, ctypes.addressof(pointee)], [0, 1]),
                                           ('flags', [0, 0x982201, 0x902201, 0xffffffff], [7, 15, 7, 15])]:
                code = obj.sections['.text.' + name].data
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64)(address)
                    self.assertEqual([call(x) for x in inputs], expected, name)

    def test_status_companions_and_rejoins(self):
        import ctypes
        import mmap
        from cc.opt import callee_of
        source = ('static inline __attribute__((always_inline)) int choose(int key, int seed, int *a, int *b) { '
                  'if (key == 1) { *a=seed+3; *b=seed+5; return 1; } '
                  'if (key == 2) { *a=seed-7; *b=seed+11; return 1; } return 0; } '
                  'static __attribute__((noinline)) int observe(int a, int b) { return 13*a+7*b; } '
                  'int direct(int key, int seed) { int a=seed+101, b=seed+103; '
                  'if (choose(key,seed,&a,&b)) return observe(a,b); return a-b; } '
                  'int rejoin(int key, int seed) { int a=seed+101, b=seed+103, result; '
                  'if (choose(key,seed,&a,&b)) result=observe(a,b); else result=a-b; return result+a; } '
                  'int looped(int count, int seed) { int total=0; for (int i=0; i<count; i++) { '
                  'int a=seed+i+101,b=seed+i+103; '
                  'if (choose(i&3,seed+i,&a,&b)) total+=observe(a,b); else total+=a-b; } return total; }')
        functions = optimize(lower_unit(parse(source)))
        direct = next(fn for fn in functions if fn.name == 'direct')
        calls = [op for block in direct.blocks for op in block.ops
                 if op.kind == 'call' and callee_of(direct, op.args[0]) == 'observe']
        self.assertEqual(len(calls), 1, 'the shared consumer must not be cloned onto each successful status edge')
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'status.o'
            path.write_bytes(emit_object(functions, Path('status.c')))
            obj = Object(path)
            offsets, code = {}, bytearray()
            for symbol in obj.symbols:
                if symbol.type == 'STT_FUNC':
                    code += bytes(-len(code) % 16)
                    offsets[symbol.name] = len(code)
                    code += obj.sections[symbol.section].data
            for symbol in obj.symbols:
                if symbol.type != 'STT_FUNC':
                    continue
                start = offsets[symbol.name]
                for rel in obj.sections[symbol.section].relocs:
                    self.assertEqual(rel.type, 'R_X86_64_PLT32')
                    at = start+rel.offset
                    code[at:at+4] = (offsets[rel.symbol]+rel.addend-at).to_bytes(4, 'little', signed=True)
            with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                memory.write(code)
                base = ctypes.addressof(ctypes.c_char.from_buffer(memory))

                def expected(key, seed):
                    if key == 1:
                        a, b = seed+3, seed+5
                    elif key == 2:
                        a, b = seed-7, seed+11
                    else:
                        a, b = seed+101, seed+103
                    return (13*a+7*b if key in (1, 2) else a-b), a

                for name in ('direct', 'rejoin', 'looped'):
                    call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)(base+offsets[name])
                    for seed in (-1000, -1, 0, 1, 17, 12345):
                        for key in range(-2, 21):
                            if name == 'looped':
                                answer = sum(expected(i & 3, seed+i)[0] for i in range(max(0, key)))
                            else:
                                answer, a = expected(key, seed)
                                if name == 'rejoin':
                                    answer += a
                            self.assertEqual(call(key, seed), answer, (name, key, seed))

    def test_string_compare_flags_and_saved_index(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        loads = '__m128i x=_mm_loadu_si128((const __m128i *)a), y=_mm_loadu_si128((const __m128i *)b); '
        source = ('static __attribute__((noinline)) unsigned clobber(unsigned a,unsigned b,unsigned c,unsigned d) '
                  '{ return a+b+c+d; } '
                  'unsigned index(const unsigned char *a,const unsigned char *b) { '+loads+
                  'return (unsigned)_mm_cmpistri(x,y,0); } '
                  'unsigned mixed(const unsigned char *a,const unsigned char *b) { '+loads+
                  'unsigned i=(unsigned)_mm_cmpistri(x,y,0); '
                  'if (i==16 && _mm_cmpistrz(x,y,0)) return i+32; return i; } '
                  'unsigned across(const unsigned char *a,const unsigned char *b,unsigned seed) { '+loads+
                  'unsigned i=(unsigned)_mm_cmpistri(x,y,0); unsigned v=clobber(seed,3,5,7); '
                  'if (_mm_cmpistrz(x,y,0)) return i+v+1; return i+v; } ')
        for suffix in ('a', 'c', 'z'):
            source += ('unsigned flag_%s(const unsigned char *a,const unsigned char *b,unsigned seed) { ' % suffix
                       + loads + 'if (_mm_cmpistr%s(x,y,0)) return seed+a[16]; return seed+b[16]; } ' % suffix)
        functions = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'flags.o'
            for isa in ('sse42', 'avx2'):
                path.write_bytes(emit_object(functions, Path('flags.c'), isa))
                obj = Object(path)
                starts, code = {}, bytearray()
                for name, section in obj.sections.items():
                    if section.flags & 2 and section.data:
                        code += bytes(-len(code) % section.align)
                        starts[name] = len(code)
                        code += section.data
                symbols = {s.name: starts[s.section]+s.value for s in obj.symbols if s.section in starts}
                symbols.update(starts)
                for name, start in starts.items():
                    for rel in obj.sections[name].relocs:
                        self.assertIn(rel.type, ('R_X86_64_PC32', 'R_X86_64_PLT32'))
                        at = start+rel.offset
                        code[at:at+4] = (symbols[rel.symbol]+rel.addend-at).to_bytes(4, 'little', signed=True)
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    base = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    calls = {name: ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_uint)(base+symbols[name])
                             for name in ('index', 'mixed', 'across', 'flag_a', 'flag_c', 'flag_z')}
                    for m in (0, 1, 2, 7, 15, 16):
                        accepted = bytes(1+(i*37) % 255 for i in range(m))
                        set_bytes = (accepted+b'\0'+b'\xa3'*16)[:16]+b'\x25'
                        a = (ctypes.c_ubyte*17).from_buffer_copy(set_bytes)
                        fill = next(c for c in range(1, 256) if c not in accepted)
                        for n in (0, 1, 7, 15, 16):
                            positions = (None, 0, n//2, n-1) if m and n else (None,)
                            for hit in positions:
                                text = bytearray((bytes([fill])*n+b'\0'+b'\xfe'*16)[:16]+b'\x5b')
                                if hit is not None:
                                    text[hit] = accepted[0]
                                b = (ctypes.c_ubyte*17).from_buffer_copy(text)
                                i = next((p for p in range(n) if text[p] in accepted), 16)
                                z = n < 16
                                flags = {'a': i == 16 and not z, 'c': i < 16, 'z': z}
                                for seed in (0, 1, 17, 0xfffffff0):
                                    answers = {'index': i, 'mixed': 48 if i == 16 and z else i,
                                               'across': (i+seed+15+z) & 0xffffffff,
                                               **{'flag_'+s: (seed+(37 if yes else 91)) & 0xffffffff
                                                  for s, yes in flags.items()}}
                                    for name, expected in answers.items():
                                        self.assertEqual(calls[name](a, b, seed), expected,
                                                         (isa, name, m, n, hit, seed))

    def test_flag_branch_with_parallel_zero(self):
        import ctypes
        import mmap
        from cc import machine as M
        from cc.stringops import fold_flag_branches
        source = lower_unit(parse('unsigned probe(unsigned a,unsigned b) { return a<b; }'))[0]
        fn = M.Function(source)
        entry, yes, no = (fn.block(name) for name in ('entry', 'yes', 'no'))
        a, b = M.R(M.physical('rdi'), 32), M.R(M.physical('rsi'), 32)
        value = M.R(M.physical('r8'), 8)
        entry.ins = [M.Instr('cmp', (a, b), ('R', 'R')),
                     M.Instr('setb', (value,), ('W',), cc='b'),
                     M.Instr('parallel', copies=((M.R(M.physical('rdx')), M.Imm(0)),)),
                     M.Instr('test', (value, value), ('R', 'R')),
                     M.Instr('jne', target=yes, cc='ne'), M.Instr('jmp', target=no)]
        entry.successors = [yes, no]
        for block, result in ((yes, 7), (no, 9)):
            block.ins = [M.Instr('mov', (M.R(M.physical('rax'), 32), M.Imm(result)), ('W', 'R')),
                         M.Instr('ret', implicit_uses=frozenset({M.physical('rax')}))]
        fold_flag_branches(fn)
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'parallel-flags.o'
            with patch('cc.emit.Select.run', return_value=(fn, {})):
                path.write_bytes(emit_object([source], Path('parallel-flags.c')))
            code = Object(path).sections['.text.probe'].data
            with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                memory.write(code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)(address)
                for left in (0, 1, 17, 0x80000000, 0xffffffff):
                    for right in (0, 1, 17, 0x80000000, 0xffffffff):
                        self.assertEqual(call(left, right), 7 if left < right else 9, (left, right))

    def test_stack_argument_survives_call(self):
        import ctypes
        import mmap
        source = ('__attribute__((noinline)) int consume(const int *p) { '
                  'volatile unsigned char scratch[64]; scratch[60]=99; return *p; } '
                  'int caller(void) { int x=7; return consume(&x); } '
                  'int opaque(void) { int x=7; int *p=&x; __asm__("" : "+r"(p)); return consume(p); }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'stack.o'
            path.write_bytes(emit_object(fs, Path('stack.c')))
            obj = Object(path)
            offsets, code = {}, bytearray()
            for symbol in obj.symbols:
                if symbol.type == 'STT_FUNC':
                    code += bytes(-len(code) % 16)
                    offsets[symbol.name] = len(code)
                    code += obj.sections[symbol.section].data
            for symbol in obj.symbols:
                if symbol.type != 'STT_FUNC':
                    continue
                start = offsets[symbol.name]
                for rel in obj.sections[symbol.section].relocs:
                    self.assertEqual(rel.type, 'R_X86_64_PLT32')
                    at = start + rel.offset
                    displacement = offsets[rel.symbol] + rel.addend - at
                    code[at:at + 4] = displacement.to_bytes(4, 'little', signed=True)
            with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                memory.write(code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                for name in ('caller', 'opaque'):
                    call = ctypes.CFUNCTYPE(ctypes.c_int)(address + offsets[name])
                    self.assertEqual(call(), 7, name)

    def test_counted_memory_loops(self):
        import ctypes
        import mmap
        source = ('unsigned char *fill(unsigned char *p, unsigned long n) { '
                  'while(n>=4) { p[0]=7; p[1]=8; p[2]=9; p[3]=10; p+=4; n-=4; } return p; } '
                  'unsigned char *copy(unsigned char *p, const unsigned char *q, unsigned long n) { '
                  'while(n>=4) { p[0]=q[0]; p[1]=q[1]; p[2]=q[2]; p[3]=q[3]; p+=4; q+=4; n-=4; } return p; }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'loops.o'
            path.write_bytes(emit_object(fs, Path('loops.c')))
            obj = Object(path)
            for name in ('fill', 'copy'):
                code = obj.sections['.text.' + name].data
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    signature = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong) if name == 'fill' else ctypes.CFUNCTYPE(
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
                    call = signature(address)
                    for n in range(81):
                        dest = (ctypes.c_ubyte * 96)(*([0xaa] * 96))
                        src = (ctypes.c_ubyte * 96)(*range(96))
                        expected = list(dest)
                        for i in range(n // 4 * 4):
                            expected[i] = 7 + i % 4 if name == 'fill' else src[i]
                        result = call(dest, n) if name == 'fill' else call(dest, src, n)
                        self.assertEqual(result, ctypes.addressof(dest) + n // 4 * 4, (name, n))
                        self.assertEqual(list(dest), expected, (name, n))

    def test_avx2_address_encoding(self):
        from cc import machine as M
        import encode
        for name, offset in [('vmovups', -224), ('vmovaps', 128)]:
            reg = M.R(M.physical('ymm9'), 256)
            mem = M.Mem(32, base=M.physical('rsi'), index=M.physical('r10'), disp=offset)
            operands, access = ((reg, mem), ('W', 'R')) if name == 'vmovups' else ((mem, reg), ('W', 'R'))
            ins = M.structured(M.Instr(name, operands, access), isa='avx2')
            self.assertEqual(ins.j['encoding'], 'VEX')
            self.assertNotEqual(bytes.fromhex(ins.j['bytes'])[0], 0x62)
            self.assertNotEqual(encode.encode(ins.j)[0][0], 0x62)

    def test_alternating_copy_pipeline(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('requires AVX2')
        source = ROOT / 'avx2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        selected = [f for f in functions if f.name in ('copy_huge_avx2', 'copy_loop_avx2')]
        self.assertEqual(len(selected), 2)
        for fn in selected:
            self.assertTrue(hasattr(fn, 'copy_pipeline'))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'copy-pipeline.o'
            path.write_bytes(emit_object(selected, source, 'avx2'))
            obj = Object(path)
            for fn in selected:
                section = obj.sections['.text.' + fn.name]
                wrapper = fn.name == 'copy_loop_avx2'
                self.assertEqual([rel.symbol for rel in section.relocs], ['str_copy_huge'] if wrapper else [])
                with mmap.mmap(-1, len(section.data), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as code, \
                     mmap.mmap(-1, 131072) as data:
                    code.write(section.data)
                    entry = ctypes.addressof(ctypes.c_char.from_buffer(code))
                    call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(entry)
                    base = ctypes.addressof(ctypes.c_char.from_buffer(data))
                    original = bytes((i * 37 + i // 256) & 255 for i in range(len(data)))
                    for n in [257, 319, 383, 384, 385, 511, 512, 513, 767, 768, 769, 1023,
                              4095, 4096, 8191, 16384, 32767, 32768, 32769]:
                        for alignment in range(32):
                            src = 512 + alignment
                            destinations = [256 + alignment, 257 + alignment, 65536 + (alignment * 7) % 32]
                            if wrapper:
                                # The external huge-copy fallback is checked by
                                # the linked library suite. Exercise both local
                                # overlap directions without entering that call.
                                destinations = destinations if n < 4096 else []
                                destinations += [src + 1, src + 31, src + 256]
                            for dst in destinations:
                                data[:] = original
                                expected = bytearray(original)
                                expected[dst:dst+n] = original[src:src+n]
                                self.assertEqual(call(base + dst, base + src, n), base + dst, (fn.name, n, src, dst))
                                self.assertEqual(data[:], expected, (fn.name, n, src, dst))

    def test_vector_tail_allocation_order(self):
        from cc import machine as M
        from cc.regalloc import color
        source = lower_unit(parse('void f(void) { return; }'))[0]
        results = []
        for reverse in (False, True):
            fn = M.Function(source, 'avx2')
            block = fn.block('entry')
            loaded, a, b, result = [fn.reg(T.V128) for _ in range(4)]
            # Both retained values prefer the same unassigned input. Their
            # propagation order must not choose a different input register.
            block.ins = [M.Instr('vmovups', (loaded, M.Mem(16, base=M.physical('rdi'))), ('W', 'R')),
                         M.Instr('vmovups', (a, loaded), ('W', 'R')),
                         M.Instr('vmovups', (b, loaded), ('W', 'R')),
                         M.Instr('vpor', (result, a, b), ('W', 'R', 'R')),
                         M.Instr('ret', implicit_uses=frozenset({result.reg}))]
            fn.vector_tail_values = (b.reg, a.reg) if reverse else (a.reg, b.reg)
            assigned, spills = color(fn)
            self.assertEqual(spills, [])
            results.append(tuple(assigned[v.reg] for v in (loaded, a, b, result)))
        self.assertEqual(*results)

    def test_loaded_loop_headers(self):
        import ctypes
        import mmap
        source = ('unsigned char *firstzero(unsigned char *p) { while(*p) p++; return p; } '
                  'unsigned long indexed(const unsigned char *p) { const unsigned char *s=p; while(*p) p++; return p-s; } '
                  'unsigned long counted(const unsigned char *p) { unsigned long x=7; while(*p) { p++; x+=3; } return x; } '
                  'unsigned char *increment(unsigned char *p) { while(*p) { *p+=1; p++; } return p; }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'headers.o'
            path.write_bytes(emit_object(fs, Path('headers.c')))
            obj = Object(path)
            for name in ('firstzero', 'indexed', 'counted', 'increment'):
                code = obj.sections['.text.' + name].data
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(address)
                    for n in range(65):
                        original = [1+i%253 for i in range(n)] + [0, 31]
                        data = (ctypes.c_ubyte * (n+2))(*original)
                        expected = n if name == 'indexed' else 7+3*n if name == 'counted' else ctypes.addressof(data)+n
                        self.assertEqual(call(data), expected, (name, n))
                        if name == 'increment':
                            original[:n] = [v+1 for v in original[:n]]
                        self.assertEqual(list(data), original, (name, n))

    def test_vector_header_and_mask_exit(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx512bw' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX512BW')
        source = ('unsigned long scanblocks(const unsigned char *p) { unsigned long n=0; '
                  'for(;;) { __m512i x=_mm512_loadu_si512(p); __mmask64 m=_mm512_testn_epi8_mask(x,x); '
                  'if(m) return n+_tzcnt_u64(m); p+=64; n+=64; } } '
                  'unsigned long copyblocks(unsigned char *d, const unsigned char *p) { unsigned long n=0; '
                  'for(;;) { __m512i x=_mm512_loadu_si512(p); __mmask64 m=_mm512_testn_epi8_mask(x,x); '
                  'if(m) return n+_tzcnt_u64(m); _mm512_storeu_si512(d,x); p+=64; d+=64; n+=64; } }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'mask-loop.o'
            path.write_bytes(emit_object(fs, Path('mask-loop.c'), isa='evex512'))
            obj = Object(path)
            for name in ('scanblocks', 'copyblocks'):
                code = obj.sections['.text.' + name].data
                self.assertEqual(obj.sections['.text.' + name].relocs, [])
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    signature = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p) if name == 'scanblocks' else ctypes.CFUNCTYPE(
                        ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)
                    call = signature(address)
                    for n in [0, 1, 63, 64, 65, 127, 128, 191, 255, 256, 319, 383, 447, 510, 511]:
                        src = (ctypes.c_ubyte * 512)(*(1+i%253 for i in range(512)))
                        src[n] = 0
                        dest = (ctypes.c_ubyte * 512)(*([0xaa] * 512))
                        result = call(src) if name == 'scanblocks' else call(dest, src)
                        self.assertEqual(result, n, (name, n))
                        expected = list(src[:n//64*64]) + [0xaa] * (512-n//64*64) if name == 'copyblocks' else [0xaa] * 512
                        self.assertEqual(list(dest), expected, (name, n))

    def test_alignment_loop_exits(self):
        import ctypes
        import mmap
        source = ('unsigned long aligned(unsigned char *s) { '
                  'unsigned char *p=(unsigned char *)((unsigned long)s&~3ul)+4; '
                  'while((unsigned long)p&15) { if(!*p) return p-s; *p=2; p+=4; } return p-s; }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'alignment-loop.o'
            path.write_bytes(emit_object(fs, Path('alignment-loop.c')))
            obj = Object(path)
            code = obj.sections['.text.aligned'].data
            with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                memory.write(code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(address)
                for offset in range(16):
                    for stop in range(5):
                        data = (ctypes.c_ubyte * 64)(*range(1,65))
                        start = ctypes.addressof(data)
                        first = ((start+offset)&~3)+4
                        data[first-start+4*stop] = 0
                        expected = list(data)
                        p = first
                        while p&15 and expected[p-start]:
                            expected[p-start] = 2
                            p += 4
                        self.assertEqual(call(start+offset), p-start-offset, (offset, stop))
                        self.assertEqual(list(data), expected, (offset, stop))

    def test_vector_search_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        features = cpuinfo.read_text() if cpuinfo.exists() else ''
        variants = ['sse2'] + (['avx2'] if 'avx2' in features else []) + (['evex512', 'evex'] if 'avx512bw' in features else [])
        dispatch = Parser(Preprocessor().process(ROOT / 'dispatch.c')).parse()
        entries = trampolines(optimize(lower_unit(dispatch)))
        def check_variant(isa, where):
            source = ROOT / (isa+'.c')
            unit = Parser(Preprocessor().process(source)).parse()
            fn = next(f for f in optimize(lower_unit(unit)) if f.name == 'memchr_'+isa)
            path = Path(where) / (isa+'.o')
            path.write_bytes(emit_object([fn], source, isa, entries))
            section = Object(path).sections['.text.memchr_'+isa]
            code = bytearray(section.data)
            table = (len(code)+15)&~15
            code.extend(bytes(table+160-len(code)))
            calls = []
            with mmap.mmap(-1, 12288) as data:
                base = ctypes.addressof(ctypes.c_char.from_buffer(data))
                def fallback(p, c, n):
                    calls.append((p, c, n))
                    if not (base <= p <= base+8192-n):
                        return 0
                    for i, value in enumerate((ctypes.c_ubyte*n).from_address(p)):
                        if value == (c&255):
                            return p+i
                    return 0
                callback = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_ulong)(fallback)
                address = ctypes.cast(callback, ctypes.c_void_p).value
                slot = entries['str_memchr_large'][1]
                code[table+slot:table+slot+8] = address.to_bytes(8,'little')
                for rel in section.relocs:
                    self.assertEqual((rel.type, rel.symbol), ('R_X86_64_PC32', 'str_table'))
                    code[rel.offset:rel.offset+4] = (table+rel.addend-rel.offset).to_bytes(4,'little',signed=True)
                protect_code = bytes.fromhex('b80a0000000f05c3')
                with mmap.mmap(-1, len(code)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code+protect_code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong)(address)
                    protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(code))
                    self.assertEqual(protect(base+8192, 4096, 0), 0)
                    self.assertIsNone(call(None, -1, 0))
                    for offset in [0, 1, 15, 16, 31, 32, 63, 4032, 4064, 4065, 4095,
                                   8192-512, 8192-129, 8192-65, 8192-33, 8192-17, 8191]:
                        for c in [0x87, 0x187, -1, 0x100]:
                            for n in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 129, 512]:
                                if offset+n > 8192:
                                    continue
                                for at in [None, 0, 1, 15, 16, 17, 31, 32, 63, 64, 127, 128, 511]:
                                    data[:8192] = b'\x11'*8192
                                    if at is not None and offset+at < 8192:
                                        data[offset+at] = c&255
                                    calls.clear()
                                    result = call(base+offset, c, n)
                                    expected = base+offset+at if at is not None and at<n else None
                                    self.assertEqual(result, expected, (offset,c,n,at))
                                    for p, byte, left in calls:
                                        self.assertEqual(byte, c&255, 'byte argument must be extended before the call')
                                        self.assertTrue(base <= p <= base+8192-left)
        with tempfile.TemporaryDirectory() as where:
            for isa in variants:
                with self.subTest(isa=isa):
                    check_variant(isa, where)

    def test_zero_vector_conditional_preheaders(self):
        import ctypes
        import mmap
        from cc import machine as M
        source = ('unsigned long scan(const unsigned char *p, unsigned which) { const unsigned char *s=p; '
                  'if(which) { for(;;p+=16) { __m128i x=_mm_loadu_si128(p); '
                  'unsigned m=_mm_movemask_epi8(_mm_cmpeq_epi8(x,_mm_setzero_si128())); '
                  'if(m) return p-s+__builtin_ctz(m); } } '
                  'for(;;p+=16) { __m128i x=_mm_loadu_si128(p); '
                  'unsigned m=_mm_movemask_epi8(_mm_cmpeq_epi8(x,_mm_setzero_si128())); '
                  'if(m) return p-s+__builtin_ctz(m)+7; } }')
        functions = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'zero.o'
            path.write_bytes(emit_object(functions, Path('zero.c')))
            section = Object(path).sections['.text.scan']
            self.assertEqual(section.relocs, [])
            prefix = bytearray()
            for n in range(16):
                r = M.R(M.physical('xmm'+str(n)), 128)
                prefix.extend(bytes.fromhex(M.structured(M.Instr('pcmpeqd', (r, r), ('W', 'R')), 0).j['bytes']))
            code = prefix + section.data
            with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                memory.write(code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_uint)(address)
                for n in (0, 1, 15, 16, 17, 31, 32, 63, 64, 127, 191):
                    for offset in (0, 1, 15):
                        for which in (0, 1, 2):
                            data = (ctypes.c_ubyte * 256)(*([81]*224+[255]*32))
                            data[offset+n] = 0
                            self.assertEqual(call(ctypes.addressof(data)+offset, which), n+(0 if which else 7), (n, offset, which))

    def test_sse2_search_page_edges(self):
        import ctypes
        import mmap
        source = ROOT / 'sse2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        selected = [f for f in functions if f.name in ('find_byte_or_nul_sse2', 'skip_byte_sse2')]
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'search.o'
            path.write_bytes(emit_object(selected, source))
            obj = Object(path)
            for fn in selected:
                with self.subTest(function=fn.name):
                    section = obj.sections['.text.' + fn.name]
                    self.assertEqual(section.relocs, [])
                    protect_code = bytes.fromhex('b80a0000000f05c3')
                    with mmap.mmap(-1, len(section.data)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                         mmap.mmap(-1, 12288) as src:
                        memory.write(section.data+protect_code)
                        address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                        call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ubyte)(address)
                        protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(section.data))
                        start = ctypes.addressof(ctypes.c_char.from_buffer(src))
                        self.assertEqual(protect(start+8192, 4096, 0), 0)
                        for n in (0, 1, 2, 7, 14, 15, 16, 17, 30, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 511, 700):
                            for offset in (0, 1, 15, 4095, 8191-n):
                                for needle in (0, 1, 81, 0x87, 0xf1, 255):
                                    src[:8192] = b'Q'*8192
                                    if fn.name == 'skip_byte_sse2':
                                        data = bytes([needle])*n + bytes([0 if needle else 1])
                                        expected = n
                                    else:
                                        data = bytes(1+i%253 for i in range(n)) + b'\0'
                                        expected = next(i for i, value in enumerate(data) if value == 0 or value == needle)
                                    src[offset:offset+len(data)] = data
                                    self.assertEqual(call(start+offset, needle), expected, (n, offset, needle))

    def test_sse2_table_scan_page_edges(self):
        import ctypes
        import mmap
        source = ROOT / 'sse2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        selected = [f for f in functions if f.name in ('strspn_table_sse2', 'strcspn_table_sse2')]
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'table-scans.o'
            path.write_bytes(emit_object(selected, source, 'sse2'))
            obj = Object(path)
            for fn in selected:
                section = obj.sections['.text.'+fn.name]
                self.assertEqual(section.relocs, [])
                protect_code = bytes.fromhex('b80a0000000f05c3')
                with mmap.mmap(-1, len(section.data)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                     mmap.mmap(-1, 12288) as data:
                    memory.write(section.data+protect_code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(address)
                    protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(section.data))
                    base = ctypes.addressof(ctypes.c_char.from_buffer(data))
                    self.assertEqual(protect(base+8192, 4096, 0), 0)
                    for members in (b'', b'a', b'az', b'\x01\x87\xff', bytes(range(1, 256))):
                        table = ctypes.create_string_buffer(members+b'\0')
                        inside = set(members)
                        for n in [*range(66), 95, 127, 128, 129, 191, 255, 511]:
                            for text in (b'a'*n, b'\xf1'*n, bytes(1+i%255 for i in range(n))):
                                expected = next((i for i, c in enumerate(text) if (c in inside) == (fn.name == 'strcspn_table_sse2')), n)
                                for offset in (0, 1, 2, 3, 15, 4095, 8191-n):
                                    data[:8192] = b'Q'*8192
                                    data[offset:offset+n+1] = text+b'\0'
                                    self.assertEqual(call(base+offset, table), expected, (fn.name, members, n, offset))

    def test_sse2_compare_page_edges(self):
        import ctypes
        import mmap
        source = ROOT / 'sse2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        selected = [f for f in functions if f.name in ('memcmp_le16', 'memcmp_17_32', 'memcmp_large_sse2')]
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'compare.o'
            path.write_bytes(emit_object(selected, source))
            obj = Object(path)
            for fn in selected:
                with self.subTest(function=fn.name):
                    section = obj.sections['.text.' + fn.name]
                    self.assertEqual(section.relocs, [])
                    protect_code = bytes.fromhex('b80a0000000f05c3')
                    with mmap.mmap(-1, len(section.data)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                         mmap.mmap(-1, 12288) as a, mmap.mmap(-1, 12288) as b:
                        memory.write(section.data+protect_code)
                        address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                        call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(address)
                        protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(section.data))
                        pa = ctypes.addressof(ctypes.c_char.from_buffer(a))
                        pb = ctypes.addressof(ctypes.c_char.from_buffer(b))
                        self.assertEqual(protect(pa+8192, 4096, 0), 0)
                        self.assertEqual(protect(pb+8192, 4096, 0), 0)
                        sizes = range(17) if fn.name == 'memcmp_le16' else range(17, 33)
                        if fn.name == 'memcmp_large_sse2':
                            sizes = (65, 66, 79, 80, 81, 95, 96, 97, 127, 128, 129, 255, 256, 257, 511, 700)
                        for n in sizes:
                            for oa in (0, 1, 15, 4095-n, 8192-n):
                                for ob in (0, 3, 31, 4095-n, 8192-n):
                                    for at in (None, 0, 1, 7, 15, 16, 31, 63, 64, 127, n-1):
                                        a[:8192], b[:8192] = b'Q'*8192, b'Q'*8192
                                        expected = 0
                                        if at is not None and 0 <= at < n:
                                            a[oa+at], b[ob+at] = 0x87, 0xf1
                                            expected = 0x87-0xf1
                                        self.assertEqual(call(pa+oa, pb+ob, n), expected, (n, oa, ob, at))

    def test_vector_compare_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        variants = ['sse2'] + (['evex', 'evex512'] if cpuinfo.exists() and 'avx512bw' in cpuinfo.read_text() else [])
        def check_variant(isa, where):
            source = ROOT / (isa+'.c')
            unit = Parser(Preprocessor().process(source)).parse()
            fn = next(f for f in optimize(lower_unit(unit)) if f.name == 'strcmp_tail_'+isa)
            path = Path(where) / 'compare.o'
            path.write_bytes(emit_object([fn], source, isa))
            section = Object(path).sections['.text.'+fn.name]
            self.assertEqual(section.relocs, [])
            # Linux x86-64 mprotect syscall, independent of the Python
            # process exporting a dynamically linked libc.
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(section.data)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                 mmap.mmap(-1, 12288) as a, mmap.mmap(-1, 12288) as b:
                memory.write(section.data+protect_code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(
                    address)
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(
                    address+len(section.data))
                pa = ctypes.addressof(ctypes.c_char.from_buffer(a))
                pb = ctypes.addressof(ctypes.c_char.from_buffer(b))
                self.assertEqual(protect(pa+8192, 4096, 0), 0)
                self.assertEqual(protect(pb+8192, 4096, 0), 0)
                for n in [0, 1, 31, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 700]:
                    for oa in [0, 1, 63, 3968, 4032, 4095, 8191-n]:
                        for ob in [0, 3, 64, 3968, 4032, 4095, 8191-n]:
                            for at in [None, 0, 63, 128, 511]:
                                a[:8192] = b'Q'*8192
                                b[:8192] = b'Q'*8192
                                a[oa+n] = b[ob+n] = 0
                                expected = 0
                                if at is not None and at < n:
                                    a[oa+at], b[ob+at] = 0x87, 0xf1
                                    expected = 0x87-0xf1
                                self.assertEqual(call(pa+oa, pb+ob), expected, (n, oa, ob, at))
        with tempfile.TemporaryDirectory() as where:
            for isa in variants:
                with self.subTest(isa=isa):
                    check_variant(isa, where)

    def test_vector_strcmp_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        variants = ['sse2'] + (['evex'] if cpuinfo.exists() and 'avx512bw' in cpuinfo.read_text() else [])
        if cpuinfo.exists() and 'avx2' in cpuinfo.read_text():
            variants.append('avx2')
        entries = trampolines(optimize(lower_unit(Parser(Preprocessor().process(ROOT / 'dispatch.c')).parse())))
        def check_variant(isa, where):
            source = ROOT / (isa+'.c')
            functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
            fn = next(f for f in functions if f.name == 'strcmp_'+isa)
            path = Path(where) / 'strcmp.o'
            path.write_bytes(emit_object([fn], source, isa, entries))
            section = Object(path).sections['.text.'+fn.name]
            with mmap.mmap(-1, 12288) as a, mmap.mmap(-1, 12288) as b:
                pa = ctypes.addressof(ctypes.c_char.from_buffer(a))
                pb = ctypes.addressof(ctypes.c_char.from_buffer(b))
                invalid = []
                def fallback(x, y):
                    if not (x is not None and y is not None and pa <= x < pa+8192 and pb <= y < pb+8192):
                        invalid.append((x, y))
                        return 0
                    for left, right in zip(bytes(a[x-pa:8192]), bytes(b[y-pb:8192])):
                        if left != right:
                            return left-right
                        if left == 0:
                            return 0
                    invalid.append((x, y))
                    return 0
                callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(fallback)
                code = bytearray(section.data)
                table = (len(code)+15)&~15
                code.extend(bytes(table+160-len(code)))
                slot = table+entries['str_strcmp_tail'][1]
                code[slot:slot+8] = ctypes.cast(callback, ctypes.c_void_p).value.to_bytes(8, 'little')
                for rel in section.relocs:
                    self.assertEqual((rel.type, rel.symbol), ('R_X86_64_PC32', 'str_table'))
                    code[rel.offset:rel.offset+4] = (table+rel.addend-rel.offset).to_bytes(4, 'little', signed=True)
                protect_code = bytes.fromhex('b80a0000000f05c3')
                with mmap.mmap(-1, len(code)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code+protect_code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(address)
                    protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(code))
                    self.assertEqual(protect(pa+8192, 4096, 0), 0)
                    self.assertEqual(protect(pb+8192, 4096, 0), 0)
                    for n in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 255, 511]:
                        data = bytes(1+i%253 for i in range(n))+b'\0'
                        for oa in [0, 1, 31, 63, 3968, 4032, 4065, 4095, 8191-n]:
                            for ob in [0, 3, 32, 3968, 4032, 4065, 4095, 8191-n]:
                                for at in [None, 0, 15, 31, 32, 63, 95, 128, n-1]:
                                    a[:8192] = b'Q'*8192
                                    b[:8192] = b'Q'*8192
                                    a[oa:oa+n+1] = b[ob:ob+n+1] = data
                                    expected = 0
                                    if at is not None and 0 <= at < n:
                                        a[oa+at], b[ob+at] = 0x87, 0xf1
                                        expected = 0x87-0xf1
                                    self.assertEqual(call(pa+oa, pb+ob), expected, (n, oa, ob, at))
                    self.assertEqual(invalid, [])
        with tempfile.TemporaryDirectory() as where:
            for isa in variants:
                with self.subTest(isa=isa):
                    check_variant(isa, where)

    def test_avx2_strcmp_tail_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        source = ROOT / 'avx2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        fn = next(f for f in functions if f.name == 'strcmp_tail_avx2')
        self.assertTrue(fn.avx_cmp_tail)
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'strcmp-tail.o'
            path.write_bytes(emit_object([fn], source, 'avx2'))
            section = Object(path).sections['.text.' + fn.name]
            self.assertEqual(section.relocs, [])
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(section.data) + len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as image, \
                 mmap.mmap(-1, 12288) as a, mmap.mmap(-1, 12288) as b:
                image.write(section.data + protect_code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(image))
                call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(address)
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address + len(section.data))
                pa = ctypes.addressof(ctypes.c_char.from_buffer(a))
                pb = ctypes.addressof(ctypes.c_char.from_buffer(b))
                self.assertEqual(protect(pa + 8192, 4096, 0), 0)
                self.assertEqual(protect(pb + 8192, 4096, 0), 0)
                for n in (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97,
                          127, 128, 129, 255, 511, 1023, 2047, 4097):
                    data = bytes(1 + k % 253 for k in range(n)) + b'\0'
                    offsets = [k for k in (0, 1, 15, 16, 31, 32, 63, 3968, 4032, 4064,
                                           4065, 4080, 4088, 4095, 8191-n) if k + n < 8192]
                    for oa in offsets:
                        for ob in offsets:
                            for at in (None, 0, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 127, n-1):
                                a[:8192] = b[:8192] = b'Q' * 8192
                                a[oa:oa+n+1] = b[ob:ob+n+1] = data
                                expected = 0
                                if at is not None and 0 <= at < n:
                                    a[oa+at], b[ob+at] = 0x87, 0xf1
                                    expected = 0x87 - 0xf1
                                self.assertEqual(call(pa+oa, pb+ob), expected, (n, oa, ob, at))
                            if n:
                                for at in (0, n//2, n-1):
                                    a[oa:oa+n+1] = b[ob:ob+n+1] = data
                                    a[oa+at] = 0
                                    self.assertEqual(call(pa+oa, pb+ob), -data[at], ('NUL', n, oa, ob, at))

    def test_vector_string_copy_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx512bw' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX512BW')
        unit = Parser(Preprocessor().process(ROOT / 'evex512.c')).parse()
        fn = next(f for f in optimize(lower_unit(unit)) if f.name == 'strcpy_long_evex512')
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'string-copy.o'
            path.write_bytes(emit_object([fn], ROOT / 'evex512.c', 'evex512'))
            section = Object(path).sections['.text.strcpy_long_evex512']
            self.assertEqual(section.relocs, [])
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(section.data)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                 mmap.mmap(-1, 12288) as src, mmap.mmap(-1, 8192) as dest:
                memory.write(section.data+protect_code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(section.data))
                source = ctypes.addressof(ctypes.c_char.from_buffer(src))
                target = ctypes.addressof(ctypes.c_char.from_buffer(dest))
                self.assertEqual(protect(source+8192, 4096, 0), 0)
                sizes = [0, 1, 15, 31, 63, 64, 65, 95, 127, 128, 129, 159, 191, 192, 193,
                         255, 256, 257, 319, 383, 447, 511, 512, 513, 767, 1023, 1535]
                for n in sizes:
                    data = bytes(1+i%253 for i in range(n))+b'\0'
                    for offset in [0, 1, 63, 64, 127, 4095, 8191-n]:
                        src[:8192] = b'Q'*8192
                        src[offset:offset+n+1] = data
                        for known in sorted({0, min(n, 16), min(n, 64), min(n, 128)}):
                            for out in [16, 17, 47, 79]:
                                dest[:] = b'\xa5'*8192
                                self.assertEqual(call(target+out, source+offset, source+offset+known), target+out,
                                                 (n, offset, known, out))
                                self.assertEqual(dest[:out], b'\xa5'*out)
                                self.assertEqual(dest[out:out+n+1], data, (n, offset, known, out))
                                self.assertEqual(dest[out+n+1:], b'\xa5'*(8192-out-n-1), (n, offset, known, out))

    def test_sse2_pipelined_string_copy_page_edges(self):
        import ctypes
        import mmap
        source = ROOT / 'sse2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        fn = next(f for f in functions if f.name == 'strcpy_long_sse2')
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'pipelined-copy.o'
            path.write_bytes(emit_object([fn], source, 'sse2'))
            section = Object(path).sections['.text.'+fn.name]
            self.assertEqual(section.relocs, [])
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(section.data)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                 mmap.mmap(-1, 12288) as src, mmap.mmap(-1, 8192) as dest:
                memory.write(section.data+protect_code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(section.data))
                start = ctypes.addressof(ctypes.c_char.from_buffer(src))
                target = ctypes.addressof(ctypes.c_char.from_buffer(dest))
                self.assertEqual(protect(start+8192, 4096, 0), 0)
                for n in [*range(1, 98), 127, 128, 129, 191, 255, 256, 257, 511, 767, 1023, 1535]:
                    data = bytes(1+i%253 for i in range(n))+b'\0'
                    for offset in [*range(32), 4095, 8191-n]:
                        known = 16-((start+offset)&15)
                        if n < known:
                            continue
                        src[:8192] = b'Q'*8192
                        src[offset:offset+n+1] = data
                        for out in [16, 17, 47, 79]:
                            dest[:] = b'\xa5'*8192
                            self.assertEqual(call(target+out, start+offset, start+offset+known), target+out, (n, offset, out))
                            self.assertEqual(dest[:out], b'\xa5'*out)
                            self.assertEqual(dest[out:out+n+1], data, (n, offset, out))
                            self.assertEqual(dest[out+n+1:], b'\xa5'*(8192-out-n-1), (n, offset, out))

    def test_vector_strlen_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        features = cpuinfo.read_text() if cpuinfo.exists() else ''
        variants = ['sse2'] + (['avx2'] if 'avx2' in features else []) + (['evex', 'evex512'] if 'avx512bw' in features else [])
        entries = trampolines(optimize(lower_unit(Parser(Preprocessor().process(ROOT / 'dispatch.c')).parse())))
        with tempfile.TemporaryDirectory() as where:
            for isa in variants:
                source = ROOT / (isa+'.c')
                functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
                fn = next(f for f in functions if f.name == 'strlen_'+isa)
                selected = [fn]
                if isa == 'avx2':
                    selected.append(next(f for f in functions if f.name == 'strlen_tail_avx2'))
                path = Path(where) / (isa+'.o')
                path.write_bytes(emit_object(selected, source, isa, entries))
                obj = Object(path)
                section = obj.sections['.text.'+fn.name]
                with mmap.mmap(-1, 12288) as src:
                    start = ctypes.addressof(ctypes.c_char.from_buffer(src))
                    invalid = []
                    def fallback(a, s):
                        if not start <= s < start+8192:
                            invalid.append((a, s))
                            return 0
                        n = bytes(src[s-start:8192]).find(b'\0')
                        if n < 0 or not s <= a <= s+n:
                            invalid.append((a, s))
                            return 0
                        return n
                    callback = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(fallback)
                    code = bytearray(section.data)
                    table = (len(code)+15)&~15
                    code.extend(bytes(table+160-len(code)))
                    slot = table+entries['str_strlen_tail'][1]
                    code[slot:slot+8] = ctypes.cast(callback, ctypes.c_void_p).value.to_bytes(8, 'little')
                    for rel in section.relocs:
                        self.assertEqual((rel.type, rel.symbol), ('R_X86_64_PC32', 'str_table'))
                        code[rel.offset:rel.offset+4] = (table+rel.addend-rel.offset).to_bytes(4, 'little', signed=True)
                    tail_offset = None
                    if isa == 'avx2':
                        tail_section = obj.sections['.text.strlen_tail_avx2']
                        self.assertEqual(tail_section.relocs, [])
                        tail_offset = len(code)
                        code.extend(tail_section.data)
                    protect_code = bytes.fromhex('b80a0000000f05c3')
                    with mmap.mmap(-1, len(code)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                        memory.write(code+protect_code)
                        address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                        call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(address)
                        tail = None if tail_offset is None else ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(address+tail_offset)
                        protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(code))
                        self.assertEqual(protect(start+8192, 4096, 0), 0)
                        for n in [0, 1, 2, 7, 14, 15, 16, 17, 30, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 255, 511, 1535]:
                            for offset in [0, 1, 15, 16, 31, 32, 63, 64, 4095, 8191-n]:
                                src[:8192] = b'Q'*8192
                                src[offset:offset+n+1] = bytes(1+i%253 for i in range(n))+b'\0'
                                self.assertEqual(call(start+offset), n, (isa, n, offset))
                                if tail is not None:
                                    for known in (0, 1, 15, 31, 32, 33, 63, 64, 95, 127, 128, 129, n):
                                        if known <= n:
                                            self.assertEqual(tail(start+offset+known, start+offset), n, (isa, n, offset, known))
                        self.assertEqual(invalid, [])

    def test_avx2_long_string_copy_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        source = ROOT / 'avx2.c'
        functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
        fn = next(f for f in functions if f.name == 'strcpy_long_avx2')
        self.assertTrue(fn.avx_copy_long)
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'copy-long.o'
            path.write_bytes(emit_object([fn], source, 'avx2'))
            section = Object(path).sections['.text.' + fn.name]
            self.assertEqual(section.relocs, [])
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(section.data) + len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as image, \
                 mmap.mmap(-1, 12288) as src, mmap.mmap(-1, 12288) as dest:
                image.write(section.data + protect_code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(image))
                call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address + len(section.data))
                start = ctypes.addressof(ctypes.c_char.from_buffer(src))
                target = ctypes.addressof(ctypes.c_char.from_buffer(dest))
                self.assertEqual(protect(start + 8192, 4096, 0), 0)
                self.assertEqual(protect(target + 8192, 4096, 0), 0)
                for n in (0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97,
                          127, 128, 129, 159, 191, 192, 193, 255, 256, 257, 383, 511, 512, 513,
                          767, 1023, 1535, 2047, 4097, 6000):
                    data = bytes(1 + k % 253 for k in range(n)) + b'\0'
                    for offset in (0, 1, 15, 16, 31, 32, 63, 64, 127, 4095, 8191-n):
                        if offset + n >= 8192:
                            continue
                        src[:8192] = b'Q' * 8192
                        src[offset:offset+n+1] = data
                        # The initial 16-byte bridge precedes the first
                        # 32-byte block; later continuations are 32-aligned.
                        cursors = sorted({(start+offset+15) & -16,
                                          *((start+offset+known+31) & -32
                                            for known in (0, min(n, 16), min(n, 32), min(n, 64), min(n, 128), n//2))})
                        for cursor in cursors:
                            if cursor > start + offset + n or cursor - start - offset > 128:
                                continue
                            for out in (*range(32), 63, 127, 8191-n):
                                dest[:8192] = b'\xa5' * 8192
                                self.assertEqual(call(target+out, start+offset, cursor), target+out,
                                                 (n, offset, cursor-start-offset, out))
                                self.assertEqual(dest[:out], b'\xa5' * out)
                                self.assertEqual(dest[out:out+n+1], data, (n, offset, cursor-start-offset, out))
                                self.assertEqual(dest[out+n+1:8192], b'\xa5' * (8191-out-n))

    def test_vector_string_copy_short_paths(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        features = cpuinfo.read_text() if cpuinfo.exists() else ''
        variants = (['avx2'] if 'avx2' in features else []) + (['evex'] if 'avx512bw' in features else [])
        if not variants:
            self.skipTest('execution needs AVX2 or AVX512BW')
        dispatch = Parser(Preprocessor().process(ROOT / 'dispatch.c')).parse()
        entries = trampolines(optimize(lower_unit(dispatch)))
        with tempfile.TemporaryDirectory() as where:
            for isa in variants:
                source = ROOT / (isa + '.c')
                functions = optimize(lower_unit(Parser(Preprocessor().process(source)).parse()))
                names = ('strcpy_avx2', 'strcat_avx2') if isa == 'avx2' else ('strcpy_long', 'strcpy_evex', 'strcat_evex')
                for name in names:
                    fn = next(f for f in functions if f.name == name)
                    concatenating = name.startswith('strcat_')
                    if isa == 'avx2' and concatenating:
                        self.assertTrue(fn.avx_cat)
                    path = Path(where) / (name+'.o')
                    path.write_bytes(emit_object([fn], source, isa, entries))
                    section = Object(path).sections['.text.'+name]
                    with mmap.mmap(-1, 12288) as src, mmap.mmap(-1, 8192) as dest:
                        source_address = ctypes.addressof(ctypes.c_char.from_buffer(src))
                        target = ctypes.addressof(ctypes.c_char.from_buffer(dest))
                        invalid = []
                        def fallback(d, s, known=None):
                            if not (target <= d < target+8192 and source_address <= s < source_address+8192):
                                invalid.append((d, s, known))
                                return d
                            n = bytes(src[s-source_address:8192]).find(b'\0')
                            prefix = bytes(dest[d-target:8192]).find(b'\0') if concatenating else 0
                            if (n < 0 or prefix < 0 or d+prefix+n+1 > target+8192
                                    or known is not None and not s <= known <= s+n+1):
                                invalid.append((d, s, known))
                                return d
                            ctypes.memmove(d+prefix, s, n+1)
                            return d
                        if isa == 'avx2' and concatenating:
                            def copy_fallback(d, end, s, known):
                                if not (target <= d <= end < target+8192 and source_address <= s < source_address+8192):
                                    invalid.append((d, end, s, known))
                                    return d
                                n = bytes(src[s-source_address:8192]).find(b'\0')
                                if n < 0 or end+n+1 > target+8192 or not s <= known <= s+n+1:
                                    invalid.append((d, end, s, known))
                                    return d
                                ctypes.memmove(end, s, n+1)
                                return d
                            prototype = ctypes.CFUNCTYPE(ctypes.c_void_p, *([ctypes.c_void_p]*4))
                            callback = prototype(copy_fallback)
                        else:
                            prototype = ctypes.CFUNCTYPE(ctypes.c_void_p, *([ctypes.c_void_p]* (2 if concatenating else 3)))
                            callback = prototype(fallback)
                        direct = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(fallback)
                        code = bytearray(section.data)
                        stub = len(code)
                        code.extend(bytes.fromhex('ff2500000000')+ctypes.cast(callback if isa == 'avx2' else direct, ctypes.c_void_p).value.to_bytes(8, 'little'))
                        long_stub = len(code)
                        code.extend(bytes.fromhex('ff2500000000')+ctypes.cast(direct, ctypes.c_void_p).value.to_bytes(8, 'little'))
                        table = (len(code)+15)&~15
                        code.extend(bytes(table+160-len(code)))
                        slot = table+entries['str_strcpy_long'][1]
                        code[slot:slot+8] = ctypes.cast(callback, ctypes.c_void_p).value.to_bytes(8, 'little')
                        for rel in section.relocs:
                            if rel.symbol == 'str_table':
                                self.assertEqual(rel.type, 'R_X86_64_PC32')
                                destination = table
                            else:
                                self.assertEqual(rel.type, 'R_X86_64_PLT32')
                                self.assertIn(rel.symbol, ('strcat_long', 'strcat_copy') if concatenating else ('strcpy_long',))
                                destination = long_stub if isa == 'avx2' and rel.symbol == 'strcat_long' else stub
                            code[rel.offset:rel.offset+4] = (destination+rel.addend-rel.offset).to_bytes(4, 'little', signed=True)
                        protect_code = bytes.fromhex('b80a0000000f05c3')
                        with mmap.mmap(-1, len(code)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                            memory.write(code+protect_code)
                            address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                            call = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
                            protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(code))
                            self.assertEqual(protect(source_address+8192, 4096, 0), 0)
                            for n in [0, 1, 2, 3, 4, 7, 8, 14, 15, 16, 17, 30, 31, 32, 33, 63, 64, 95, 96, 127, 128, 129, 159, 191, 255]:
                                data = bytes(1+i%253 for i in range(n))+b'\0'
                                for offset in [0, 1, 15, 31, 63, 4095, 8191-n]:
                                    src[:8192] = b'Q'*8192
                                    src[offset:offset+n+1] = data
                                    for prefix in ([0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 255]
                                                   if concatenating else [0]):
                                        for out in ([0, 1, 15, 16, 17, 31, 47, 79, 8191-prefix-n] if concatenating else [16, 17, 47, 79]):
                                            dest[:] = b'\xa5'*8192
                                            if concatenating:
                                                dest[out:out+prefix+1] = b'Z'*prefix+b'\0'
                                            self.assertEqual(call(target+out, source_address+offset), target+out, (name, n, offset, prefix, out))
                                            self.assertEqual(dest[:out], b'\xa5'*out)
                                            self.assertEqual(dest[out:out+prefix+n+1], b'Z'*prefix+data, (name, n, offset, prefix, out))
                                            self.assertEqual(dest[out+prefix+n+1:], b'\xa5'*(8192-out-prefix-n-1))
                            self.assertEqual(invalid, [])

    def test_vector_memory_compare(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        features = cpuinfo.read_text() if cpuinfo.exists() else ''
        variants = (['avx2'] if 'avx2' in features else []) + (['evex', 'evex512'] if 'avx512bw' in features else [])
        if not variants:
            self.skipTest('execution needs AVX2 or AVX512BW')
        dispatch = Parser(Preprocessor().process(ROOT / 'dispatch.c')).parse()
        entries = trampolines(optimize(lower_unit(dispatch)))
        with tempfile.TemporaryDirectory() as where:
            for isa in variants:
                source = ROOT / (isa+'.c')
                unit = Parser(Preprocessor().process(source)).parse()
                functions = optimize(lower_unit(unit))
                fn = next(f for f in functions if f.name == 'memcmp_'+isa)
                selected = [fn] + [f for f in functions if isa == 'avx2' and f.name in ('memcmp_le16', 'memcmp_17_32', 'memcmp_large_avx2')]
                path = Path(where) / (isa+'.o')
                path.write_bytes(emit_object(selected, source, isa, entries))
                obj = Object(path)
                section = obj.sections['.text.memcmp_'+isa]
                code = bytearray(section.data)
                offsets = {}
                for helper in selected[1:]:
                    name = '.text.'+helper.name
                    part = obj.sections[name]
                    self.assertEqual(part.relocs, [])
                    code.extend(bytes(-len(code) % 16))
                    offsets[name] = offsets[helper.name] = len(code)
                    code.extend(part.data)
                table = (len(code)+15)&~15
                code.extend(bytes(table+160-len(code)))
                with mmap.mmap(-1, 12288) as a, mmap.mmap(-1, 12288) as b:
                    pa = ctypes.addressof(ctypes.c_char.from_buffer(a))
                    pb = ctypes.addressof(ctypes.c_char.from_buffer(b))
                    invalid = []
                    def fallback(x, y, n):
                        if not (pa <= x <= pa+8192-n and pb <= y <= pb+8192-n):
                            invalid.append((x, y, n))
                            return 0
                        for left, right in zip((ctypes.c_ubyte*n).from_address(x), (ctypes.c_ubyte*n).from_address(y)):
                            if left != right:
                                return left-right
                        return 0
                    callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(fallback)
                    slot = table+entries['str_memcmp_large'][1]
                    code[slot:slot+8] = ctypes.cast(callback, ctypes.c_void_p).value.to_bytes(8, 'little')
                    for rel in section.relocs:
                        if rel.symbol == 'str_table':
                            self.assertEqual(rel.type, 'R_X86_64_PC32')
                            target = table
                        else:
                            self.assertIn(rel.type, ('R_X86_64_PLT32', 'R_X86_64_PC32'))
                            target = offsets[rel.symbol]
                        code[rel.offset:rel.offset+4] = (target+rel.addend-rel.offset).to_bytes(4, 'little', signed=True)
                    protect_code = bytes.fromhex('b80a0000000f05c3')
                    with mmap.mmap(-1, len(code)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                        memory.write(code+protect_code)
                        address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                        if isa == 'avx2':
                            memory[slot:slot+8] = (address+offsets['memcmp_large_avx2']).to_bytes(8, 'little')
                        call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(address)
                        protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(code))
                        self.assertEqual(protect(pa+8192, 4096, 0), 0)
                        self.assertEqual(protect(pb+8192, 4096, 0), 0)
                        self.assertEqual(call(None, None, 0), 0)
                        for n in [0, 1, 2, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 79, 95, 96, 97, 127, 128, 129, 191, 256, 257, 383, 511, 512, 700]:
                            for oa in [0, 1, 31, 63, 4095, 8192-n]:
                                for ob in [0, 3, 63, 8192-n]:
                                    for at in [None, 0, 15, 31, 32, 63, 64, 95, 96, 127, 128, 129, 159, 160, 191, 192, 223, 255, 256, 383, n-1]:
                                        a[:8192] = b'Q'*8192
                                        b[:8192] = b'Q'*8192
                                        expected = 0
                                        if at is not None and 0 <= at < n:
                                            a[oa+at], b[ob+at] = 0xf1, 0x87
                                            expected = 0xf1-0x87
                                        self.assertEqual(call(pa+oa, pb+ob, n), expected, (isa, n, oa, ob, at))
                        self.assertEqual(invalid, [])

    def test_vector_test_zero(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        source = ('unsigned test(const unsigned char *p, const unsigned char *q) { '
                  'return _mm256_testz_si256(_mm256_loadu_si256(p), _mm256_loadu_si256(q)); } '
                  'unsigned branch(const unsigned char *p, const unsigned char *q) { '
                  'if(_mm256_testz_si256(_mm256_loadu_si256(p), _mm256_loadu_si256(q))) return 7; return 19; }')
        functions = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'testzero.o'
            path.write_bytes(emit_object(functions, Path('testzero.c'), 'avx2'))
            obj = Object(path)
            for fn in functions:
                section = obj.sections['.text.'+fn.name]
                self.assertEqual(section.relocs, [])
                with mmap.mmap(-1, len(section.data), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(section.data)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)(address)
                    for lane in range(32):
                        for bit in (1, 2, 4, 8, 16, 32, 64, 128):
                            for overlap in (False, True):
                                a = (ctypes.c_ubyte * 32)()
                                b = (ctypes.c_ubyte * 32)()
                                a[lane] = bit
                                b[lane] = bit if overlap else bit ^ 255
                                expected = int(not overlap) if fn.name == 'test' else 19 if overlap else 7
                                self.assertEqual(call(a, b), expected, (fn.name, lane, bit, overlap))

    def test_internal_boolean_parameters(self):
        import ctypes
        import mmap
        from cc.opt import emitted_functions
        source = ('static __attribute__((noinline)) unsigned internal(unsigned x) { return x ? 0u : ~0u; } '
                  'static __attribute__((noinline)) unsigned global_fn(unsigned x) { return x ? 0u : ~0u; } '
                  'static __attribute__((noinline)) unsigned passed_fn(unsigned x) { return x ? 0u : ~0u; } '
                  'unsigned (*global_pointer)(unsigned) = global_fn; '
                  'extern void observe(unsigned (*fn)(unsigned)); '
                  'void expose(void) { observe(passed_fn); } '
                  'unsigned a(unsigned x) { return internal(x != 0); } '
                  'unsigned b(unsigned x) { return global_fn(x != 0); } '
                  'unsigned c(unsigned x) { return passed_fn(x != 0); } '
                  'static __attribute__((noinline)) unsigned bounded(unsigned x) { return x; } '
                  'static __attribute__((noinline)) unsigned wrapped(unsigned x) { return x; } '
                  'unsigned sum(unsigned x) { return bounded((x & 31u) + 16u); } '
                  'unsigned wrap(unsigned x) { return wrapped((x & 0xfffffff0u) + 16u); }')
        unit = parse(source)
        functions = emitted_functions(optimize(lower_unit(unit)))
        by_name = {f.name: f for f in functions}
        self.assertEqual(by_name['internal'].parameter_masks[by_name['internal'].params[0]], 1)
        for name in ('global_fn', 'passed_fn'):
            self.assertEqual(by_name[name].parameter_masks[by_name[name].params[0]], 0xffffffff)
        self.assertEqual(by_name['bounded'].parameter_masks[by_name['bounded'].params[0]], 63)
        self.assertEqual(by_name['wrapped'].parameter_masks[by_name['wrapped'].params[0]], 0xffffffff)
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'arguments.o'
            path.write_bytes(emit_object(functions, Path('arguments.c'), unit=unit))
            obj = Object(path)
            for name in ('internal', 'global_fn', 'passed_fn'):
                section = obj.sections['.text.'+name]
                self.assertEqual(section.relocs, [])
                with mmap.mmap(-1, len(section.data), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(section.data)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_uint)(address)
                    values = (0, 1) if name == 'internal' else (0, 1, 2, 0x80000000, 0xffffffff)
                    for value in values:
                        self.assertEqual(call(value), 0 if value else 0xffffffff, (name, value))

    def test_vector_insert64_lanes(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        source = ' '.join('void lane%d(unsigned long long *out, const unsigned long long *in, long long value) { '
                          '_mm_storeu_si128((__m128i *)out, _mm_insert_epi64('
                          '_mm_loadu_si128((const __m128i *)in), value, %d)); }' % (n, n) for n in (0, 1))
        functions = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'insert.o'
            for isa in ('sse42', 'avx2'):
                path.write_bytes(emit_object(functions, Path('insert.c'), isa))
                obj = Object(path)
                for lane in (0, 1):
                    section = obj.sections['.text.lane%d' % lane]
                    self.assertEqual(section.relocs, [])
                    with mmap.mmap(-1, len(section.data), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                        memory.write(section.data)
                        address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                        call = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong)(address)
                        for value in (0, 1, 0x123456789abcdef0, 1 << 63, (1 << 64)-1):
                            data = [0xfedcba9876543210, 0x13579bdf2468ace0]
                            input_ = (ctypes.c_ulonglong*2)(*data)
                            output = (ctypes.c_ulonglong*2)()
                            call(output, input_, value)
                            data[lane] = value
                            self.assertEqual(list(output), data, (isa, lane, value))

    def test_sse42_span_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'sse4_2' not in cpuinfo.read_text():
            self.skipTest('execution needs SSE4.2')
        source = ROOT / 'sse42.c'
        pp = Preprocessor()
        tokens = pp.process(source)
        tokens += pp.text('__attribute__((noinline)) size_t strspn_table_sse2('
                          'const u8 *s, const u8 *set) { return 9999; } '
                          '__attribute__((noinline)) size_t strcspn_table_sse2('
                          'const u8 *s, const u8 *set) { return 9999; } '
                          '__attribute__((noinline)) size_t str_strlen(const char *s) { '
                          'size_t n = 0; while (s[n]) ++n; return n; }')
        functions = optimize(lower_unit(Parser(tokens).parse()))
        selected = [fn for fn in functions if fn.name in ('strspn_sse42', 'strcspn_sse42', 'strspn_table_sse2', 'strcspn_table_sse2', 'str_strlen')]
        self.assertEqual(len(selected), 5)
        self.assertTrue(next(fn for fn in selected if fn.name == 'strspn_sse42').string_prefixes)
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'span.o'
            path.write_bytes(emit_object(selected, source, 'sse42'))
            obj, starts, code = Object(path), {}, bytearray()
            for name, section in obj.sections.items():
                if not section.flags & 2 or not section.data:
                    continue
                code.extend(bytes(-len(code) % section.align))
                starts[name] = len(code)
                code.extend(section.data)
            symbols = {s.name: starts[s.section] + s.value for s in obj.symbols if s.section in starts}
            symbols.update(starts)
            for name, start in starts.items():
                for rel in obj.sections[name].relocs:
                    self.assertIn(rel.type, ('R_X86_64_PC32', 'R_X86_64_PLT32'))
                    at = start + rel.offset
                    code[at:at+4] = (symbols[rel.symbol] + rel.addend - at).to_bytes(4, 'little', signed=True)
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(code) + len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as image, \
                 mmap.mmap(-1, 12288) as text, mmap.mmap(-1, 12288) as charset:
                image.write(code + protect_code)
                base = ctypes.addressof(ctypes.c_char.from_buffer(image))
                signature = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)
                calls = {True: signature(base + symbols['strspn_sse42']),
                         False: signature(base + symbols['strcspn_sse42'])}
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(base + len(code))
                text_base = ctypes.addressof(ctypes.c_char.from_buffer(text))
                set_base = ctypes.addressof(ctypes.c_char.from_buffer(charset))
                self.assertEqual(protect(text_base + 8192, 4096, 0), 0)
                self.assertEqual(protect(set_base + 8192, 4096, 0), 0)
                for membership, call in calls.items():
                    for size in (0, 1, 2, 15, 16, 17, 31, 32, 33):
                        members = bytes((k * 37) % 255 + 1 for k in range(size))
                        outsider = next(k for k in range(1, 256) if k not in members)
                        set_offsets = [*range(16), 4095-size, 4096-min(size, 15), 8191-size]
                        for n in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128):
                            seed = members if membership and size else bytes([1 if membership else outsider])
                            prefix = bytes(seed[k % len(seed)] for k in range(n))
                            stop = outsider if membership else members[-1] if size else 0
                            for ending in (b'\0', bytes([stop, 0])):
                                data = prefix + ending
                                answer = 9999 if size > 32 else n if size or not membership else 0
                                for offset in (0, 1, 15, 4081, 4096-min(n, 15), 8192-len(data)):
                                    text[:8192] = b'Q' * 8192
                                    text[offset:offset+len(data)] = data
                                    for set_offset in set_offsets:
                                        charset[:8192] = b'R' * 8192
                                        charset[set_offset:set_offset+size+1] = members + b'\0'
                                        self.assertEqual(call(text_base+offset, set_base+set_offset), answer,
                                                         (membership, size, n, ending, offset, set_offset))

    def test_avx2_span_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        source = ROOT / 'avx2.c'
        pp = Preprocessor()
        tokens = pp.process(source)
        tokens += pp.text('__attribute__((noinline)) size_t str_strlen(const char *s) { '
                          'size_t n = 0; while (s[n]) ++n; return n; }')
        functions = optimize(lower_unit(Parser(tokens).parse()))
        names = ('strspn_avx2', 'strspn_avx2_zen', 'strcspn_avx2', 'strcspn_avx2_zen')
        selected = [fn for fn in functions if fn.name in (*names, 'truffle_scan', 'truffle_scan32', 'truffle_scan_long', 'str_strlen')]
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(fn.avx_span for fn in selected if fn.name in names))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'span.o'
            path.write_bytes(emit_object(selected, source, 'avx2'))
            obj, starts, code = Object(path), {}, bytearray()
            for name, section in obj.sections.items():
                if not section.flags & 2 or not section.data:
                    continue
                code.extend(bytes(-len(code) % section.align))
                starts[name] = len(code)
                code.extend(section.data)
            symbols = {s.name: starts[s.section] + s.value for s in obj.symbols if s.section in starts}
            symbols.update(starts)
            for name, start in starts.items():
                for rel in obj.sections[name].relocs:
                    self.assertIn(rel.type, ('R_X86_64_PC32', 'R_X86_64_PLT32'))
                    at = start + rel.offset
                    code[at:at+4] = (symbols[rel.symbol] + rel.addend - at).to_bytes(4, 'little', signed=True)
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(code) + len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as image, \
                 mmap.mmap(-1, 12288) as text, mmap.mmap(-1, 12288) as charset:
                image.write(code + protect_code)
                base = ctypes.addressof(ctypes.c_char.from_buffer(image))
                signature = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)
                calls = {name: signature(base + symbols[name]) for name in names}
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(base + len(code))
                text_base = ctypes.addressof(ctypes.c_char.from_buffer(text))
                set_base = ctypes.addressof(ctypes.c_char.from_buffer(charset))
                self.assertEqual(protect(text_base + 8192, 4096, 0), 0)
                self.assertEqual(protect(set_base + 8192, 4096, 0), 0)
                for size in (0, 1, 2, 15, 16, 17, 31, 32, 33, 254, 255):
                    members = bytes((k * 37) % 255 + 1 for k in range(size))
                    outsider = next((k for k in range(1, 256) if k not in members), 0)
                    set_offsets = [*range(16), 4095-size, 4096-min(size, 15), 8191-size]
                    for n in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                              255, 511, 1023, 1024, 1025, 2047, 2048, 2049, 4097):
                        for membership in (True, False):
                            seed = (members or b'Q') if membership else bytes([outsider or 1])
                            prefix = bytes(seed[k % len(seed)] for k in range(n))
                            stop = outsider if membership else members[-1] if members else 0
                            for ending in (b'\0', bytes([stop, 0])):
                                data = prefix + ending
                                answer = n if (size > 0 if membership else size < 255) else 0
                                for offset in (0, 1, 15, 4081, 4096-min(n, 15), 8192-len(data)):
                                    text[:8192] = b'Q' * 8192
                                    text[offset:offset+len(data)] = data
                                    for set_offset in set_offsets:
                                        charset[:8192] = b'R' * 8192
                                        charset[set_offset:set_offset+size+1] = members + b'\0'
                                        for name, call in calls.items():
                                            if name.startswith('strspn') == membership:
                                                self.assertEqual(call(text_base+offset, set_base+set_offset), answer,
                                                                 (name, size, n, ending, offset, set_offset))

    def test_vector_classification_page_edges(self):
        import ctypes
        import mmap
        cpuinfo = Path('/proc/cpuinfo')
        if not cpuinfo.exists() or 'avx2' not in cpuinfo.read_text():
            self.skipTest('execution needs AVX2')
        source = ROOT / 'avx2.c'
        pp = Preprocessor()
        tokens = pp.process(source)
        tokens += pp.text('size_t classification_probe(const u8 *from, const u8 *start, '
                          'const void *low, const void *high, int members) { '
                          'return truffle_scan(from, start, load16(low), load16(high), members != 0); } '
                          'size_t classification_long_probe(const u8 *from, const u8 *start, '
                          'const u8 *set, int members) { '
                          'return truffle_scan_long(from, start, set, members != 0); } '
                          'size_t classification_short_probe(const u8 *from, const u8 *start, '
                          'const u8 *set, size_t count, int members) { '
                          'return truffle_scan32(from, start, load16(set), load16(set+16), count & 63, members != 0); }')
        functions = optimize(lower_unit(Parser(tokens).parse()))
        scan = next(f for f in functions if f.name == 'truffle_scan')
        self.assertTrue(getattr(scan, 'vector_classification', False))
        driver = next(f for f in functions if f.name == 'classification_probe')
        long_scan = next(f for f in functions if f.name == 'truffle_scan_long')
        self.assertTrue(getattr(long_scan, 'scalar_classification', False))
        long_driver = next(f for f in functions if f.name == 'classification_long_probe')
        short_scan = next(f for f in functions if f.name == 'truffle_scan32')
        self.assertTrue(getattr(short_scan, 'vector_table_build', False))
        short_driver = next(f for f in functions if f.name == 'classification_short_probe')
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'classification.o'
            path.write_bytes(emit_object([driver, scan, long_driver, long_scan, short_driver, short_scan], source, 'avx2'))
            obj = Object(path)
            starts, code = {}, bytearray()
            for name, section in obj.sections.items():
                if not section.flags & 2 or not section.data:
                    continue
                code.extend(bytes(-len(code) % section.align))
                starts[name] = len(code)
                code.extend(section.data)
            symbols = {s.name: starts[s.section]+s.value for s in obj.symbols if s.section in starts}
            symbols.update(starts)
            for name, start in starts.items():
                for rel in obj.sections[name].relocs:
                    self.assertIn(rel.type, ('R_X86_64_PC32', 'R_X86_64_PLT32'))
                    at = start+rel.offset
                    code[at:at+4] = (symbols[rel.symbol]+rel.addend-at).to_bytes(4, 'little', signed=True)
            protect_code = bytes.fromhex('b80a0000000f05c3')
            with mmap.mmap(-1, len(code)+len(protect_code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory, \
                 mmap.mmap(-1, 12288) as data:
                memory.write(code+protect_code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)(address+starts['.text.classification_probe'])
                long_call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_void_p, ctypes.c_int)(address+starts['.text.classification_long_probe'])
                short_call = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+starts['.text.classification_short_probe'])
                protect = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int)(address+len(code))
                base = ctypes.addressof(ctypes.c_char.from_buffer(data))
                self.assertEqual(protect(base+8192, 4096, 0), 0)
                sets = (b'', b'a', b'az', b'\x01\x87\xff', bytes(range(1, 256)),
                        *(bytes(1+(i*73) % 255 for i in range(n)) for n in range(4, 33)))
                for members in (0, 1, -1):
                    for accepted in sets:
                        set_string = ctypes.create_string_buffer(accepted+b'\0')
                        short_set = ctypes.create_string_buffer((accepted+b'\xa5'*32)[:32])
                        inside = set(accepted) | ({0} if members else set())
                        low, high = (ctypes.c_ubyte*16)(), (ctypes.c_ubyte*16)()
                        for byte in inside:
                            table = high if byte & 128 else low
                            table[byte & 15] |= 1 << ((byte >> 4) & 7)
                        allowed = [c for c in range(1, 256) if (c in inside) != bool(members)]
                        for n in (0, 1, 15, 31, 32, 33, 63, 64, 65, 95, 127, 128, 129, 255, 511, 700):
                            for fill in set((allowed[0], allowed[-1]) if allowed else (1,)):
                                expected = n if allowed else 0
                                for offset in (0, 1, 15, 16, 31, 32, 63, 4095, 8191-n):
                                    data[:8192] = b'Q'*8192
                                    data[offset:offset+n+1] = bytes([fill])*n+b'\0'
                                    for known in (0, 1, 31, 32, 63, n):
                                        if known <= expected:
                                            result = call(base+offset+known, base+offset, low, high, members)
                                            self.assertEqual(result, expected, (members, accepted, n, fill, offset, known))
                                            result = long_call(base+offset+known, base+offset, set_string, members)
                                            self.assertEqual(result, expected, ('scalar tables', members, accepted, n, fill, offset, known))
                                            if len(accepted) <= 32:
                                                result = short_call(base+offset+known, base+offset, short_set, len(accepted), members)
                                                self.assertEqual(result, expected, ('vector tables', members, accepted, n, fill, offset, known))
                for byte in range(1, 256):
                    set_string = ctypes.create_string_buffer(bytes([byte, 0]))
                    other = byte % 255+1
                    short_set = ctypes.create_string_buffer(bytes([byte])+bytes([other])*31)
                    for n in (0, 31, 32, 33, 64, 255):
                        offset = 8190-n
                        for fill, stop in ((byte, other), (other, byte)):
                            data[:8192] = b'Q'*8192
                            text = bytes([fill])*n+bytes([stop, 0])
                            data[offset:offset+n+2] = text
                            for members in (0, 1):
                                inside = {byte} | ({0} if members else set())
                                expected = next(i for i, c in enumerate(text) if (c in inside) == bool(members))
                                result = long_call(base+offset, base+offset, set_string, members)
                                self.assertEqual(result, expected, (byte, n, fill, stop, members))
                                result = short_call(base+offset, base+offset, short_set, 1, members)
                                self.assertEqual(result, expected, ('vector tables', byte, n, fill, stop, members))

    def test_integer_path_ranges(self):
        import ctypes
        import mmap
        source = ('unsigned classify(unsigned n) { if(n-1<32) { if(n==0) return 99; return n<=16 ? 1 : 2; } return n==0 ? 0 : 3; } '
                  'unsigned signed_case(int n) { if(n<0) { if(n<=0) return 1; return 99; } return n<2147483647 ? 2 : 3; } '
                  'unsigned wrapped(unsigned n) { if(n<0xfffffffdU) return n+4<4 ? 9 : 7; return n+4<4 ? 1 : 99; } '
                  'unsigned long sum(unsigned long n) { unsigned long x=0; while(n>0) { x+=n; n--; } return x; }')
        fs = optimize(lower_unit(parse(source)))
        cases = [('classify', [0, 1, 16, 17, 32, 33, 0xffffffff], [0, 1, 1, 2, 2, 3, 3]),
                 ('signed_case', [0x80000000, 0xffffffff, 0, 1, 0x7fffffff], [1, 1, 2, 2, 3]),
                 ('wrapped', [0, 1, 0xfffffffb, 0xfffffffc, 0xfffffffd, 0xfffffffe, 0xffffffff], [7, 7, 7, 9, 1, 1, 1]),
                 ('sum', list(range(81)), [n*(n+1)//2 for n in range(81)])]
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'ranges.o'
            path.write_bytes(emit_object(fs, Path('ranges.c')))
            obj = Object(path)
            for name, values, expected in cases:
                code = obj.sections['.text.' + name].data
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64)(address)
                    self.assertEqual([call(v) for v in values], expected, name)

    def test_return_load_paths(self):
        import ctypes
        import mmap
        source = ('int opaque(const unsigned char *); '
                  'int compare(const unsigned char *a, const unsigned char *b, unsigned n) { '
                  'if(n==255) return opaque(a); '
                  'if(n<4) { unsigned j=n&3; return (int)a[j]-(int)b[j]; } '
                  'if(n<8) { unsigned j=(n+1)&7; return (int)a[j]-(int)b[j]; } '
                  'unsigned j=(n+3)&15; return (int)a[j]-(int)b[j]; } '
                  'int changed(unsigned char *a, const unsigned char *b, unsigned n) { '
                  'if(n==255) return opaque(a); '
                  'if(n<4) { unsigned j=n&3; int old=a[j]; a[j]=123; return old-(int)b[j]; } '
                  'if(n<8) { unsigned j=(n+1)&7; return (int)a[j]-(int)b[j]; } '
                  'unsigned j=(n+3)&15; return (int)a[j]-(int)b[j]; }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'paths.o'
            path.write_bytes(emit_object(fs, Path('paths.c')))
            obj = Object(path)
            for name in ('compare', 'changed'):
                code = obj.sections['.text.' + name].data
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(address)
                    for n in range(32):
                        a = (ctypes.c_ubyte * 16)(*(17*i for i in range(16)))
                        b = (ctypes.c_ubyte * 16)(*(255-17*i for i in range(16)))
                        at = n&3 if n<4 else (n+1)&7 if n<8 else (n+3)&15
                        expected = a[at]-b[at]
                        self.assertEqual(call(a, b, n), expected, (name, n))
                        self.assertEqual(a[at], 123 if name == 'changed' and n<4 else 17*at, (name, n))

    def test_sse2_load_alignment(self):
        import ctypes
        import mmap
        source = ('unsigned unaligned(const void *p) { return (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(p), _mm_setzero_si128())); } '
                  'unsigned aligned(const void *p) { return (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128(p), _mm_setzero_si128())); }')
        fs = optimize(lower_unit(parse(source)))
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'alignment.o'
            path.write_bytes(emit_object(fs, Path('alignment.c')))
            obj = Object(path)
            for name in ('unaligned', 'aligned'):
                code = obj.sections['.text.' + name].data
                with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
                    memory.write(code)
                    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                    call = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p)(address)
                    data = (ctypes.c_ubyte * 64)(*(0 if i%3 else i+1 for i in range(64)))
                    start = ctypes.addressof(data)
                    for offset in range(16):
                        p = start + offset
                        if name == 'aligned' and p%16:
                            continue
                        expected = sum((data[offset+i]==0)<<i for i in range(16))
                        self.assertEqual(call(p), expected, (name, offset))

    def test_dispatch_code(self):
        pp = Preprocessor()
        unit = Parser(pp.process(ROOT / 'dispatch.c')).parse()
        fs = optimize(lower_unit(unit))
        entries = trampolines(fs)
        self.assertEqual(len(entries), 20)
        selected = [f for f in fs if f.name in entries]
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / 'dispatch.o'
            with patch.object(elf, 'read', side_effect=AssertionError('compiler must not parse ELF')), \
                 patch.object(x86, 'decode', side_effect=AssertionError('compiler must not decode machine code')):
                path.write_bytes(emit_object(selected, ROOT / 'dispatch.c'))
            obj = Object(path)
            for name, (symbol, offset) in entries.items():
                section = obj.sections['.text.' + name]
                self.assertEqual(section.data, bytes.fromhex('ff2500000000'), name)
                self.assertEqual(section.relocs, [Reloc(2, 'R_X86_64_PC32', symbol, offset - 4)], name)


class Objects(unittest.TestCase):
    def test_compiler_objects(self):
        if ARCHIVE is None:
            self.skipTest('pass --archive to compare the reference objects byte for byte')
        from post import read_archive
        with tempfile.TemporaryDirectory() as where:
            members, exports = [], {}
            for name, data in read_archive(ARCHIVE):
                path = Path(where) / name
                path.write_bytes(data)
                obj = Object(path)
                defined = []
                with patch.object(elf, 'read', side_effect=AssertionError('emitter must not parse ELF')):
                    emitted = elfout.write_model(obj, None, defined=defined)
                self.assertEqual(emitted, data, name)
                members.append((name, emitted))
                exports[name] = defined
            path = Path(where) / 'libstr.a'
            with patch.object(elf, 'read', side_effect=AssertionError('archive emitter must not parse ELF')):
                elfout.write_archive(path, members, defined=exports)
            self.assertEqual(path.read_bytes(), ARCHIVE.read_bytes())

    def test_direct_object_and_archive(self):
        obj = Object()
        obj.sections['.text.f'] = Section('.text.f', 'SHT_PROGBITS', 6, 16, b'\xc3')
        obj.sections['.bss.x'] = Section('.bss.x', 'SHT_NOBITS', 3, 32, b'', nobits_size=96)
        obj.sections['.data.p'] = Section('.data.p', 'SHT_PROGBITS', 3, 8, bytes(8),
                                         relocs=[Reloc(0, 'R_X86_64_64', 'x', 16)])
        obj.symbols = [Symbol('f', 0, 1, 'STT_FUNC', 'STB_GLOBAL', '.text.f'),
                       Symbol('x', 0, 96, 'STT_OBJECT', 'STB_GLOBAL', '.bss.x'),
                       Symbol('p', 0, 8, 'STT_OBJECT', 'STB_GLOBAL', '.data.p')]
        with tempfile.TemporaryDirectory() as where:
            path, archive, defined = Path(where) / 'f.o', Path(where) / 'f.a', []
            with patch.object(elf, 'read', side_effect=AssertionError('emission must not parse ELF')):
                data = elfout.write_model(obj, path, defined=defined)
                elfout.write_archive(archive, [('f.o', data)], defined={'f.o': defined})
            self.assertEqual(defined, ['f', 'x', 'p'])
            image = elf.read(path)
            bss = next(s for s in image.sections if s.name == '.bss.x')
            self.assertEqual((bss.size, bss.align, bss.data), (96, 32, b''))
            rela = next(iter(image.relocs.values()))[0]
            self.assertEqual((rela.offset, rela.type, rela.addend, image.symbols[rela.symbol].name), (0, 'R_X86_64_64', 16, 'x'))
            self.assertEqual(archive.read_bytes()[:8], b'!<arch>\n')


if __name__ == '__main__':
    if '--archive' in sys.argv:
        at = sys.argv.index('--archive')
        ARCHIVE = Path(sys.argv[at + 1])
        del sys.argv[at:at + 2]
    unittest.main()
