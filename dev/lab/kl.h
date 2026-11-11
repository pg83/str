/* The kernel lab's registry: kernels-gen.c, written by lab.py from nm of every
   link input, defines it; kbench.c reads it. */
#ifndef KL_H
#define KL_H
#include <stddef.h>

enum { P_MEMCPY, P_MEMMOVE, P_MEMSET, P_MEMCMP, P_STRLEN, P_STRCPY, P_STRCAT, P_STRCMP, P_STRSPN, P_STRCSPN, P_MEMCHR, P_STRNLEN, P_N };

/* What a kernel needs from the host. F_AVX2 stands for x86-64-v3 (AVX2, BMI1/2,
   LZCNT, MOVBE, FMA), F_AVX512 for x86-64-v4 (AVX-512 F/BW/VL/CD/DQ). */
enum { F_SSSE3 = 1, F_SSE42 = 2, F_AVX = 4, F_AVX2 = 8, F_AVX512 = 16, F_ERMS = 32, F_FSRM = 64, F_MOVBE = 128, F_RTM = 256 };

typedef struct {
    const char *origin; /* glibc, asmlib, freebsd, musl, llvm, kl */
    const char *name;   /* the symbol */
    int prim;
    void *fn;
    unsigned needs;
} KlDef;

/* Our implementations by slot of struct str_table: the public entries and the
   kernels behind them; the driver builds our kernels from them at run time. */
typedef struct {
    int slot; /* index of the slot in struct str_table read as an array of pointers */
    const char *name;
    void *fn;
    unsigned needs;
    int spec; /* the post pass's copy for one configuration (<unit>.<name>, its calls direct), not a table client */
} KlSlot;

/* derived.c's functions that allocate nothing, in the workload's mix: ours
   (str_*, through str_table) and the other libraries' by their prefix, by
   name, not validated (the mixed trace's checksum against glibc is the check). */
enum { D_BCMP, D_BCOPY, D_BZERO, D_MEMPCPY, D_STRNCAT, D_STRLCPY, D_STRLCAT, D_STRPBRK, D_STRSEP, D_STRTOK, D_STRTOK_R, D_MEMMEM, D_STRSTR, D_N };

typedef struct {
    const char *origin; /* ours, base, freebsd, musl, llvm (glibc's come by their C names) */
    const char *name;   /* the symbol */
    int fn;             /* D_* */
    void *addr;
    unsigned needs;
} KlDerived;

extern const KlDef kl_defs[];
extern const unsigned kl_ndefs;
extern const KlSlot kl_slots[];
extern const unsigned kl_nslots;
extern const KlDerived kl_derived[];
extern const unsigned kl_nderived;
extern const char *const kl_jcc_configs[]; /* the configurations laid out with the JCC padding, 0-terminated */

/* Another revision of ours, its symbols prefixed base_ (lab.py --base): its
   configurations, the table its code calls through, the revision. None: 0. */
extern const struct str_config *const kl_base_configs;
extern const unsigned kl_base_nconfigs;
extern struct str_table *const kl_base_table;
extern const char kl_base_rev[];

/* glibc's variant at an address, and the implementation its IFUNC chose for
   a primitive (the address of memcpy & co. in a static binary is an IPLT stub). */
const char *glibc_variant(void *p);
void *resolved_glibc(void *entry);
#endif
