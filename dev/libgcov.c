/* A minimal gcov runtime for a gcc without libgcc (ix builds it so):
   coverage.py --gcov-runtime builds it as libgcov.a for --coverage with
   -fprofile-update=atomic. The objects register their gcov_info, and at exit
   each is written to its .gcda as libgcc/libgcov-driver.c's write_one_data
   does (no merge with an earlier file: coverage.py removes them before the
   run). The layouts are gcc's libgcov.h and gcov-io.h, 12 to 15; the one
   that changes, the count of counter kinds, comes as GCOV_COUNTERS (8 to gcc
   13, conditions in 14, paths in 15). */
#include <stdint.h>
#include <stdio.h>

#ifndef GCOV_COUNTERS
#error "GCOV_COUNTERS: the counter kinds of the gcc that compiles the objects"
#endif

typedef unsigned gcov_unsigned_t;
typedef int64_t gcov_type;
typedef void (*gcov_merge_fn)(gcov_type *, gcov_unsigned_t);

struct gcov_ctr_info {
    gcov_unsigned_t num;
    gcov_type *values;
};

struct gcov_info;

struct gcov_fn_info {
    const struct gcov_info *key;
    gcov_unsigned_t ident, lineno_checksum, cfg_checksum;
    struct gcov_ctr_info ctrs[1];
};

struct gcov_info {
    gcov_unsigned_t version;
    struct gcov_info *next;
    gcov_unsigned_t stamp, checksum;
    const char *filename;
    gcov_merge_fn merge[GCOV_COUNTERS];
    gcov_unsigned_t n_functions;
    const struct gcov_fn_info *const *functions;
};

static struct gcov_info *registered;
static int written;

void __gcov_merge_add(gcov_type *counters, gcov_unsigned_t n) {
    (void)counters;
    (void)n;
}

void __gcov_init(struct gcov_info *info) {
    info->next = registered;
    registered = info;
}

static void put(FILE *f, gcov_unsigned_t v) {
    fwrite(&v, sizeof v, 1, f);
}

void __gcov_exit(void) {
    if (written++) {
        return;
    }

    for (struct gcov_info *gi = registered; gi; gi = gi->next) {
        FILE *f = fopen(gi->filename, "wb");

        if (!f) {
            continue;
        }

        put(f, 0x67636461);        /* "gcda" */
        put(f, gi->version);
        put(f, gi->stamp);
        put(f, gi->checksum);
        put(f, 0xa1000000);        /* the object summary: one run */
        put(f, 8);
        put(f, 1);
        put(f, 0);

        for (gcov_unsigned_t i = 0; i < gi->n_functions; i++) {
            const struct gcov_fn_info *fn = gi->functions[i];
            put(f, 0x01000000);

            if (!fn || fn->key != gi) {
                put(f, 0);
                continue;
            }

            put(f, 12);
            put(f, fn->ident);
            put(f, fn->lineno_checksum);
            put(f, fn->cfg_checksum);
            const struct gcov_ctr_info *c = fn->ctrs;

            for (int t = 0; t < GCOV_COUNTERS; t++) {
                if (!gi->merge[t]) {
                    continue;
                }

                put(f, 0x01a10000 + ((gcov_unsigned_t)t << 17));
                put(f, c->num * 8);

                for (gcov_unsigned_t k = 0; k < c->num; k++) {
                    put(f, (gcov_unsigned_t)c->values[k]);
                    put(f, (gcov_unsigned_t)((uint64_t)c->values[k] >> 32));
                }

                c++;
            }
        }

        put(f, 0);
        fclose(f);
    }
}
