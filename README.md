# str

[![CI](https://github.com/pg83/str/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pg83/str/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Permissively licensed string and memory functions for x86-64: `memcpy`,
`memmove`, `memset`, `memcmp`, `memchr`, `strlen`, `strcpy`, `strcat`,
`strcmp`, `strspn`, `strcspn`. Written in a small C dialect with builtin
SIMD operations, compiled by the project's Python compiler, and dispatched
once by `cpuid` through one table of primitives. Made of these
(`derived.c`): `bcmp`, `bcopy`, `bzero`, `explicit_bzero`, `mempcpy`,
`strnlen`, `strncat`, `strlcpy`, `strlcat`, `strpbrk`, `strsep`, `strtok`,
`strtok_r`, `strdup`, `strndup`, `memmem`, `strstr`. The archive is one
object; meant to be linked into a static libc in place of its string
routines.

## Build

```sh
./build -j8 libstr             # .build/libstr.a; Python only
./build -j8 checks             # compiler tests, C test driver, decoder round-trip
```

## Results

Time per call on the workload of `dev/lab`, relative to glibc on the same
host (lower is faster):

| platform | configuration | str | glibc 2.44 | FreeBSD | LLVM libc | musl 1.2.5 |
|---|---|---|---|---|---|---|
| Zen 5 (Ryzen AI 9 HX 370) | evex512r | **0.52** | 1 (13.8 ns) | 1.53 | 1.63 | 3.01 |
| Skylake-SP (Xeon Gold 6148) | evex | **0.48** | 1 (15.0 ns) | 1.36 | 1.55 | 2.66 |
| Broadwell-EP (Xeon E5-2690 v4) | avx2 | **0.90** | 1 (14.2 ns) | 1.32 | 1.53 | 2.42 |

The workload is one trace of calls in a program's proportions (`memcpy`
25%, `strlen` 20%, `strcmp` 12%, `memset` and `memcmp` 10% each, the rest a
few percent), its sizes, alignments and contents drawn from measured
distributions: strings a median of 8 bytes, memory blocks of 14 to 29
(`dev/lab/workload.py`). Each library runs it with its own function for
every call: str and glibc as each dispatches on the host, LLVM libc built
for the host's level. A figure is the time per call net of the stand's own
cost, the minimum over the passes of four runs with the core's clock held
at nominal; glibc's own time in parentheses. `dev/lab/libs.py` prints the
table, each function's trace under it.

## License

MIT, see [LICENSE](LICENSE).
