# str

[![CI](https://github.com/pg83/str/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pg83/str/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Permissively licensed string and memory functions for x86-64: `memcpy`,
`memmove`, `memset`, `memcmp`, `strlen`, `strcpy`, `strcat`, `strcmp`,
`strspn`, `strcspn`. Written in C with the `immintrin.h` intrinsics,
dispatched once by `cpuid` through one table of primitives, and then
rewritten after the compiler by a small post-pass over the objects. Meant to
be linked into a static libc in place of its string routines.

## Model

One global `str_table` holds every primitive: the ten public functions and
the kernels they hand the long cases to (`copy_large`, `fill_large`,
`memcmp_large`, `strlen_tail`, `strcmp_tail`, `strcpy_long`). An
implementation is a plain function named by what it does and how
(`strlen_evex512`, `copy_large_avx2`, `strcpy_long_sse2`); it runs its own
hot path and, when it needs another primitive, calls the entry point that
jumps through the table: `strcat` is `str_strlen` plus `str_strcpy`, a
64-byte `strlen` first block continues in `str_strlen_tail`, `memmove`
without overlap is `str_memcpy`. So the slots compose in any mix, and `dispatch.c`
names a complete table per class of host: 256-bit AVX-512 first blocks over
the AVX2 kernels on Skylake-SP, 512-bit ones on Zen 4 and later (with
512-bit `strcpy_long` and `strcmp_tail`), SSE4.2
`strspn`/`strcspn` everywhere above the baseline.

Anything that depends on the host is a slot, not a build flag. The post pass
lays every unit out twice: plain, and with branches kept off 32-byte
boundaries for the Skylake JCC erratum (`jcc-<unit>.c.o`, functions renamed
`name_jcc`); the padding helps Skylake-SP and costs about 5% elsewhere, so
only the Skylake-SP configuration names the `_jcc` twins.

## Layout

- `sse2.c`, `sse42.c`, `avx2.c`, `evex.c`, `evex512.c` — the implementations,
  split by the `-march` they need: x86-64, x86-64-v2 (SSE4.2), x86-64-v3
  (AVX2), x86-64-v4 at 256 bits and at 512 bits. `common.h` holds the table,
  the short paths (up to 64 bytes) and the helpers they share.
- `dispatch.c` — the configurations, one complete table each, and `str_host()`
  from `cpuid`; a constructor copies the host's configuration into
  `str_table`, and the public `str_memcpy` and friends jump through it. A
  libc build renames them: `-Dstr_memcpy=memcpy ...`.
- `dev/` — the post pass, Python only: `x86.py` decodes x86-64 (legacy,
  VEX, EVEX) into structured instructions, `opt.py` rewrites them (tail and
  branch layout, mask-register and vector-register rewrites, calls through
  the table slots instead of the entry points), `encode.py` encodes them
  again from the same tables, `layout.py` lays the sections out (branch
  relaxation, alignment, relocations), `elf.py` and `elfout.py` read and
  write the objects and the archive, `post.py` drives it. No assembler, no
  `ar`. The library is the post pass's output: a build without it does not
  link.
- `tst/test.c` — the library against the host C library: all sizes and
  alignments, overlaps, guard pages, random sets, for every configuration the
  host can run and for the dispatched one. `tst/roundtrip.py` — the decoder
  and emitter against the compiler at -O0…-O3.

## Build

```sh
./build -j8 checks libstr      # .build/libstr.a, the test, the round-trip
cat .build/post/report.txt     # what the post pass changed
```

`CC` selects the compiler (clang or gcc); the objects are compiled
`-fno-pic`, so the post pass sees plain relocations. The pass itself needs
only Python 3.

## Post pass

The compiler cannot be asked for some things from C, so `dev/opt.py` does
them on the objects, with a control-flow graph and register liveness per
function:

- `rename` — def-use webs of vector registers that touch a `ymm` are moved
  to `ymm16-31` and `vzeroupper` is dropped: nothing dirties the upper halves
  of `ymm0-15` any more. Webs that only use `xmm` stay VEX-encoded.
- `fold` — a masked load whose only use is a masked compare becomes the
  compare's memory operand; the `kand`/`ktest` that follow are simplified
  from the fact that the result is a subset of the mask.
- `tail`, `hot`, `thread` — short `ret` tails are copied over the `jmp` that
  reached them, a conditional branch to such a tail is inverted so the tail
  falls through (short inputs carry most of the weight), a jump to a lone
  `jmp` goes to its target.
- `ktest` — where a block starts with `kmov k,r` and every way in ends with
  `kortest k,k; jcc`, the `kortest` becomes `kmov k,r; test r,r` (fused with
  the branch) and the block's `kmov` goes away.
- `lea` — `lea (a,b),c; add $imm,c` becomes one `lea`.
- `copy` — the ABI copy at the entry (`mov %rdi,%rsi` kept for a tail call)
  moves to the paths that need it; the hot path reads the original register.
- The layout keeps branches and fused compare+branch pairs off 32-byte
  boundaries (the Skylake JCC erratum).

Tried and left off: `vpcmpeqb (mem),zero,k` in place of load + `vptestnmb`
is 20% slower on Skylake-SP (`zcmp`); a `kor` hoisted above `kortest` costs
40% on `strcmp`.

`post.py --roundtrip --check` without passes must reproduce every object
byte for byte (code, relocations, symbols); the CI does that for clang and
gcc at every optimization level.

## Results

Against glibc's own implementations, geometric mean of time per call over
the ten functions at sizes 5, 25, 125, … weighted 1/size: 0.85× on Zen 5,
0.83× on Skylake-SP, 0.84× on Broadwell-EP (lower is faster). In one
process, medians of five runs, our code against the compiler's and against
glibc at 5 bytes: `strlen` 0.86/1.00 on Zen 5 and 0.63/1.00 on Skylake-SP,
`memcmp` 1.00/1.00 and 0.75/1.00, `strcpy` 1.00/1.00 and 0.89/0.62,
`strcat` 1.00/0.59 and 0.97/0.86. What is left behind glibc sits at 125
bytes and beyond, in the AVX2 kernels that every configuration above AVX2
still uses for copies and `memcmp`; the 512-bit configuration has its own
`strcpy_long` and `strcmp_tail`.

## License

MIT, see [LICENSE](LICENSE).
