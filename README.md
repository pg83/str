# str

[![CI](https://github.com/pg83/str/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pg83/str/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Permissively licensed string and memory functions for x86-64: `memcpy`,
`memmove`, `memset`, `memcmp`, `memchr`, `strlen`, `strnlen`, `strcpy`,
`strcat`, `strcmp`, `strspn`, `strcspn`. Written in a small C dialect with
builtin SIMD operations, compiled by the project's Python compiler, and
dispatched once by `cpuid` through one table of primitives. Made of these
(`derived.c`): `bcmp`, `bcopy`, `bzero`, `explicit_bzero`, `mempcpy`,
`strncat`, `strlcpy`, `strlcat`, `strpbrk`, `strsep`, `strtok`, `strtok_r`,
`memmem`, `strstr`. The archive is one object and takes no name from
outside (`strdup` and `strndup`, which need `malloc`, are left to the C
library); meant to be linked into a static libc in place of its string
routines.

## Build

```sh
./build -j8 libstr             # .build/libstr.a; Python only
./build -j8 libstr_wasm2o      # .build/wasm2o/libstr.o: the same code for pizdos's wasm2o
./build -j8 checks             # compiler tests, C test driver, decoder round-trip, the wasm2o contract
```

`libstr_wasm2o` (`strcc.py --abi wasm2o`) compiles the library in the
convention of the code [pizdos](https://github.com/pizd0s/kernel)'s wasm2o
compiles, for its loader to link: arguments in `rax`, `rcx`, `rdx`, `rsi`,
`rdi`, `r8`, `rbx` and `r9`-`r12` kept, `rbp`, `r13`-`r15` and vector
register 15 never touched, no stack alignment assumed, an `.eh_frame` for
the goroutine stacks it runs on. The loader calls `str_choose` and calls the
public slots of the table it returns directly.

## Results

Time per call on the workload of `dev/lab`, relative to glibc on the same
host (lower is faster):

| platform | configuration | str | glibc 2.44 | FreeBSD | LLVM libc | musl 1.2.5 |
|---|---|---|---|---|---|---|
| Zen 5 (Ryzen AI 9 HX 370) | evex512 | **0.65** | 1 (22.4 ns) | 1.34 | 1.46 | 2.27 |
| Skylake-SP (Xeon Gold 6148) | evex | **0.81** | 1 (36.2 ns) | 1.17 | 1.30 | 1.73 |
| Broadwell-EP (Xeon E5-2690 v4) | avx2 | **0.96** | 1 (37.1 ns) | 1.13 | 1.30 | 1.55 |

The workload is one trace of calls in a program's proportions (the
weights of `dev/lab/workload.py`: `memcpy` 25, `strlen` 20, `strcmp` 12,
`memset` and `memcmp` 10, the rest a few each, the derived functions a
small flow), its sizes, alignments and contents drawn from measured
distributions: strings a median of 8 bytes, memory blocks of 14 to 29, and
1% of the copies and fills on a power-law tail from 4 KiB to 8 MiB, as
the programs below make them. Each library runs it with its own function
for every call: str and glibc as each dispatches on the host, LLVM libc
built for the host's level. A figure is the time per call net of the
stand's own cost, the minimum over the passes of four runs with the core's
clock held at nominal; glibc's own time in parentheses. `dev/lab/libs.py`
prints the table, each function's trace under it.

A synthetic trace is not a program: its sizes are drawn call by call, where a
program's come in peaks the branch predictor learns. So every change is
measured on real programs too, by their cycles: pizdos's compiler compiling
itself, its decoders and CPython, run by its loader with this library linked
in (`dev/lab/pizdos.py`; dev/lab/README.md, "Real programs").

## License

MIT, see [LICENSE](LICENSE).
