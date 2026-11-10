# The kernel lab

Every implementation of every primitive (`memcpy memmove memset memcmp strlen
strcpy strcat strcmp strspn strcspn memchr`) is a *kernel*: validated, then
timed over sizes, alignments and workload traces, on every host we have.
A kernel is the unit, not a library: each glibc IFUNC variant, each of our
configurations and slot choices, each new idea in `kernels/`.

```
ix run lib/c bld/compiler -- ./build lab            # .build/lab/kbench, .build/lab/workload.bin
.build/lab/kbench --list                            # every kernel, what it needs
.build/lab/kbench --validate-only --jobs 12
.build/lab/kbench --cpu 2 --prims strlen --kernels kl/,glibc/__strlen_evex,ours/evex512 --sizes quick --regimes lat
dev/lab/run.py --tag r1 -- --regimes thr,lat        # local, lab2, lab3 in parallel: .lab/r1/<host>/, the clock held
dev/lab/run.py --tag m1 --hosts lab2 --clock free -- --prims none --no-matrix --mixed --table 'old=evex:strlen=strlen_evex_jcc'
dev/lab/report.py r1 --regime lat                   # winners per size bucket, where ours lose
dev/lab/run.py --tag lib-1 -- --prims none --no-matrix --libs   # whole libraries on the workload (repeat: lib-2..5)
dev/lab/libs.py 'lib-*'                             # the platform x library table, glibc = 1
dev/lab/run.py --tag h-1 --hosts lab3 -- --prims none --no-matrix --libs --hybrid --pmu cycles,branch-misses,r0479
                                                    # ours with each function glibc's in the mix; counters per call
ix run lib/c bld/compiler -- python3 dev/lab/base.py HEAD   # .build/lab/kbench-base: ours and HEAD's in one binary
dev/lab/run.py --tag ab-1 --binary .build/lab/kbench-base -- --prims none --no-matrix --libs && dev/lab/libs.py 'ab-*'
dev/lab/split.py .build/lab/workload.bin .lab/split.bin --traces memset --edges 0,128,256,512,4096
dev/lab/run.py --tag s-1 --trace .lab/split.bin -- --prims memset --no-matrix   # a trace per size bucket
ix run lib/c bld/compiler -- ./build lab_canary     # kernels broken on purpose: every one must be rejected
```

## The kernels

- **glibc 2.44** (Debian's static `libc.a`, also kbench's C runtime): every IFUNC
  variant of the eleven primitives, the `_rtm` ones too (glibc runs them on
  CPUs with RTM, Broadwell among them).
- **ours**: every configuration of `str_configs` as dispatch installs it
  (`ours/evex512`, the post pass's copies with direct calls); the host's
  configuration through the table (`ours/evex512.t`); and that with one slot
  of the primitive replaced by every other implementation of the slot
  (`ours/evex512.t+strcmp_tail=strcmp_tail_avx2`). A primitive's slots are its
  entry and the kernels behind it (`copy_large`, `copy_huge`, `strcmp_tail`, …).
- **FreeBSD** `lib/libc/amd64/string`: scalar, baseline and x86_64_v2 variants.
- **musl** 1.2.5: its string files (C, and x86_64 `memcpy`/`memmove`/`memset` in asm).
- **LLVM libc**: the string functions in the Linux configuration, built for
  x86-64-v2, -v3 and -v4 (`llvm/llvm_v3_strlen`).
- **asmlib**: Agner Fog's dispatchers and variants (`asm_memcpyAVX512BW`).
- **kl**: `kernels/*.c`, new ones.

What a kernel needs (SSE4.2, AVX2 = x86-64-v3, AVX-512 = x86-64-v4, ERMS, RTM) comes
from its name; a host runs the kernels it can. One binary serves every host.

## A new kernel

`kernels/<isa>_<anything>.c`, `<isa>` one of `sse2 ssse3 sse42 avx2 evex
evex512` (evex: AVX-512 at 256 bits). The file is compiled like our unit of
that level (`-O3 -march=…`, `-I` the repository for `common.h`). Every global
`kl_<prim>_<isa>_<name>` with the libc signature is a kernel; the `<isa>` of
the name is the file's. A kernel may call our primitives through `str_*` (a
hybrid: its own path for small sets, `str_strspn` beyond) and says so in its
name. The rules are the library's: read only the object's bytes and any byte
of the pages it lies in, write only the result's bytes, compare unsigned
bytes, `memmove` takes any overlap, `memchr` matches `(unsigned char)c`.

## Inputs

Nothing from outside is kept here. `deps.py` fetches, by sha256 with a
fallback URL each: Debian's `libc6-dev` 2.44-3 and `libgcc-16-dev`, musl
1.2.5, asmlib's zip, and the files of `freebsd.lock` and `llvm-libc.lock` one
by one at their pinned commits. `freebsd/` holds the three headers that make
FreeBSD's assembly link on Linux with every variant exported. `./build`
caches all of it by content.

## What is measured

- **Validation**, each kernel in a child process of its own (a crash, a hang
  or a wrong answer drops the kernel, not the run): sizes 0..160 and up to 12
  KiB, 20 offsets per pointer, high bytes, every overlap direction, guard
  pages on both sides of every object, strspn sets of 1 to 64 bytes (also at
  a page end).
- **The matrix**: scenarios (memmove disjoint and overlapping by ±1 and ±n/2;
  memset a byte and zero; memcmp/strcmp equal, differing first and last;
  strspn/strcspn sets of 1, 4, 16 and 32 bytes; memchr absent and last),
  sizes (`full`: 0..64 each, then to 1 MiB, 115 points; `quick`: 34), offset
  pairs (22 for two pointers, 14 for one; `pe` ends the object at a page end).
  In a cell all kernels of the primitive in a shuffled order per round, a
  batch of about 40 us each, 5 rounds, the minimum and median ns per call of
  thread CPU time. Regimes: `thr`, independent calls; `lat`, each call's
  pointers depend on the previous result (what a caller that uses the result
  sees).
- **Traces** (`workload.py`): every function's trace of the workload and the
  long strspn/strcspn ones, each kernel, 7 passes, checksums against glibc.
  `--mixed`: the whole workload, each call's function drawn by the mix's
  weights, for whole tables: every configuration of ours, glibc as
  dispatched, and `--table NAME=CONFIG:SLOT=FN,...` (a configuration with
  slots replaced; a public slot takes any kernel of its primitive by
  symbol). In the mix each function's branches see the others' in the
  history and the store buffer drains between calls: the per-function trace
  of memset is a benchmark of back-to-back memsets, where the order of the
  stores counts 1.3x, and the mix is not (there the branches on the size
  count). A clock license one function takes slows the others, which only
  the mix shows. The per-function traces are drawn on their own and long
  (131 072 calls): a short one replayed many times is learned by the branch
  predictor. `split.py` cuts a trace into size buckets (short, so the
  predictor learns them: for stores and loads, not for branches).
- **Hybrids** (`--hybrid`): the host's configuration of ours with one public
  function glibc's, for each function: which of ours lose to glibc's in the
  mix, the trace's change divided by the function's share being its cost a
  call there.
- **Two revisions in one binary** (`base.py REV`, `./build lab_base`): the
  libstr.a of REV linked beside this tree's with its symbols prefixed
  `base_`, the kernels `base/<config>`, the library `base` in `libs.py`, and
  with `--hybrid` ours with each function the base's. Two kbench builds
  differ in the place of everything; an A/B across binaries measured the
  stand, not the change.
- **The stand**, the loops over a trace, is in every number: built with
  every branch clear of 32-byte boundaries (Skylake's JCC erratum: a branch
  cut by one leaves the uop cache) and every block on a 64-byte line of its
  own (the switch's first case right after its jump cost the empty stand
  5.4 legacy-decoded uops a call of the memcpy trace on Broadwell; now none
  on any trace), and linked first in `.text` with
  glibc's variants after it in a fixed order (`order.txt`), so that neither
  moves with a change of ours or of kbench. `--stand 0..3` runs one of four
  copies 16 bytes apart. Before, moving the stand by 16 bytes took its
  legacy-decoded uops from 14 to 36 a call on Skylake-SP and moved ours'
  strcmp by 25% and glibc's memcmp by 7%, in opposite directions; after, the
  four copies (same shape, different addresses) agree on the mixed trace
  within 1-2%, on the short calls' gross times within 4% (ours and glibc
  alike); a net time can move more, as the empty stand's own does (memcmp
  on Skylake-SP: the gross time of ours equal in all four, the stand 5.2
  or 4.9 ns, the net 2.5..2.9).
- **Counters** (`--pmu cycles,instructions,branches,branch-misses,rHEX`):
  perf_event_open for the thread in user mode, one more pass of each trace
  after the timed ones, per call in the column `pmu`; rHEX is a raw event
  (Intel: umask << 8 | event, cmask << 24), more events than counters take
  turns, scaled.
- **Whole libraries** (`--libs`, the workload stand of the old bench.c
  `--workload`): ours as dispatched, glibc as dispatched, FreeBSD, LLVM libc
  at the host's level, musl, asmlib, each with its own function for every
  primitive, on the mixed trace and every function's trace; noop measures
  the stand's own cost per call, which `libs.py` subtracts. A library runs a
  trace only with a valid function for every primitive in it.
- **The core's clock** of every trace pass from APERF/MPERF where
  `/dev/cpu/N/msr` can be read (root: the lab hosts; here run.py uses sudo):
  turbo and the AVX-512 licenses in numbers (the `ghz` column).

The clock is held at the nominal frequency for a run (`freq.py`: boost or
turbo off, the performance governor, or on a host without cpufreq every core's
HWP request pinned at its guaranteed ratio) and given back after; kbench
measures it at the start and the end with a chain of dependent adds.

`out.csv.gz` has `prim,scenario,regime,n,align,kernel,min_ns,med_ns,iters,ghz,pmu`;
`meta.json` the host, its features, glibc's and our dispatch, every kernel and
its validation; `run.json` the commit, the binary's hash and the options.

## Real programs: pizdos

The workload above draws every call's size on its own from a smooth
distribution; a program's sizes come in peaks at its call sites, and the
branch predictor learns them there. The two disagree: evex512r's memset by
rep stosb for every size ran the memset trace 0.80x on Zen 5 and lost 9% in
file(1)'s decoder, whose 144-byte memset per magic entry takes rep stosb's
slow start from 128 bytes. So real programs are a measurement of their own,
`pizdos.py`, and a change of a first block or a slot is judged by both:

```
ix run lib/c bld/compiler bin/go/lang/25 bin/zstd -- python3 dev/lab/pizdos.py build head=HEAD new=work
dev/lab/pizdos.py run --tag p-1                   # local, lab2, lab3 in parallel, 5 runs, the clock held
dev/lab/pizdos.py report 'p-*'                    # per host and program: the base's cycles, the others' ratios
dev/lab/pizdos.py build head=HEAD try=HEAD:evex512r.memset=memset_evex512   # slots replaced, no commit
```

pizdos (github.com/pizd0s/kernel, pinned in `pizdos.py`) is an OS whose
processes are wasm modules; its loader wasm2o compiles them to x86-64 and
links our library in, built in the convention of the modules' code
(`strcc.py --abi wasm2o`), with the configuration `str_choose` picks for the
host. A module's `memory.copy` and `memory.fill` (but those of a constant
length up to 32 bytes, which the compiler expands) and lib/c's string
functions are calls of ours. The programs:

- **aot**: pizdos's compiler compiling itself, 6.7 G instructions; its
  memmove and memset calls are mostly of 4 to 16 bytes, in peaks;
- **im, pdf, djvu, svg, magic**: the decoders of pizdos's stand (decode
  release 16) over its corpus of 89 files, magic (file(1)) 200 times over;
- **py/NAME**: CPython, pizdos's python.o, on the stand's 14 benchmarks of
  the interpreter.

`build` unpacks pizdos into `.lab/pizdos/src/` and makes these inputs with
pizdos's own `./build` (its cache by content: some two minutes the first
time); then, for every `NAME=SPEC` (a revision, or `work`: the working tree,
optionally `:CONFIG.SLOT=FN,...` replaced in `dispatch.c`), our `libstr.o`
by that tree's own `strcc.py`, checked by its `tst/wasm2o.py` (revisions
from a1c43d5 on), and a wasm2o embedding it. The programs are compiled once,
by the first wasm2o: their code does not depend on our library, only their
calls do. `run` ships the bundle (`.lab/pizdos/bundle/`, 135 MB) to each host
once per content and calls every program with every wasm2o in turns, run
after run, on one core with the clock held as for kbench. A call's figure is
the whole process's cycles in user mode, its startup included (the same for
every wasm2o), by `count.c`: the counters opened for the child before it
execs, as perf stat does; the lab hosts have no perf. The output's digest
must not depend on the wasm2o, and `report` warns when it does. Results:
`.lab/T/<host>/pizdos.jsonl`, a line per call (cycles, instructions, wall,
user and system seconds, the driver's own `time` of its work where it
prints one, the digest); `run.json` with the bundle's id, pizdos's commit,
and every wasm2o's revision, replaced slots and `libstr.o`'s sha256.

A profile's share of a function is not its cost. With avx2's size trees
instead of evex512r's masked first blocks, memmove's and memset's samples on
the self-compile fell by half (1.86% of the cycles to 0.92%) and the
program's cycles stayed the same (-0.07% over 13 runs each): the samples
moved into the callers. The total is what this measures.

The programs' calls go into kbench too. `pizdos.py trace` records their
calls of memcpy, memmove and memset with a wasm2o whose GOT entries of them
are wrappers (`wasm2o/trace.go`: the length, both objects' offsets in the
module's memory, the call site) and writes a kbench workload,
`.lab/pizdos/traces.bin`. Each program gets a trace `pz/NAME` of 4 runs of
32768 consecutive calls spread over the program, the addresses compacted
by 4 KiB pages (their order, identity and alignment kept), both objects in
the destination pool as they were in one memory (`memmove_overlap`,
`memcpy_dst`), the pools zeros and not stored (version 2 of the format;
kbench resets only what a trace writes). The string functions are not
recorded: their bytes decide their work.

```
ix run lib/c bld/compiler bin/go/lang/25 bin/zstd -- python3 dev/lab/pizdos.py trace
dev/lab/run.py --tag pt-1 --trace .lab/pizdos/traces.bin -- --prims none --no-matrix --libs
dev/lab/libs.py 'pt-*'                            # the traces pz/NAME after the workload's
```

A program's calls come in its order: the sizes as its call sites make them,
learned by the branch predictor as in the program, and the same addresses
again. file(1)'s are a 144-byte memset and a 128-byte memmove by turns,
each at one address, 65 000 times a pass. Replayed on Zen 5 (4 runs, ns a
call net of the stand): glibc 2.66, rep stosb for every size 10.15, rep
stosb below 128 bytes (1318a0f) 3.35, evex512's masked memset 2.74; the
workload's memset trace ranks them the other way round. Over the 20
programs, the geometric mean of the ratios to glibc (2026-11-10): Zen 5
evex512r with rep stosb for every size 0.77, below 128 bytes 0.53, with
evex512's memset 0.32; Skylake-SP evex 0.59; Broadwell avx2 1.04, behind
glibc on djvu, im, magic, CPython's dicts, float and raytrace, and svg.
