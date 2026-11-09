# str

[![CI](https://github.com/pg83/str/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pg83/str/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Permissively licensed string and memory functions for x86-64: `memcpy`,
`memmove`, `memset`, `memcmp`, `strlen`, `strcpy`, `strcat`, `strcmp`,
`strspn`, `strcspn`. Written in a small C dialect with builtin SIMD
operations, compiled by the project's Python compiler, and dispatched once
by `cpuid` through one table of primitives. Meant to be linked into a
static libc in place of its string routines.

## Build

```sh
./build -j8 libstr             # .build/libstr.a; Python only
./build -j8 checks             # compiler tests, C test driver, decoder round-trip
```

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
