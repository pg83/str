/*
 * count: a command's cycles and instructions in user mode, its threads and
 * children included, for the measurement on pizdos's programs
 * (dev/lab/pizdos.py) on hosts without perf. The counters are opened for the
 * child before it execs, enabled by the exec and inherited, as perf stat
 * opens them; the child keeps count's standard streams.
 *
 *     count --json FILE -- COMMAND ARG...
 *
 * FILE gets one JSON object: cycles, instructions (scaled when the counters
 * took turns), wall, user and system seconds, the exit status. The exit code
 * is the command's.
 */
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* perf_event_open by its number, PERF_TYPE_HARDWARE: 0 cycles, 1 instructions */
static int counter(pid_t pid, uint64_t config) {
    struct { uint32_t type, size; uint64_t config, period, sample_type, read_format, flags; uint32_t wakeup, bp_type; uint64_t config1; } at = {0};
    at.size = sizeof at; /* PERF_ATTR_SIZE_VER0 */
    at.config = config;
    at.read_format = 1 | 2;                                  /* the time enabled and running */
    at.flags = 1 | 1 << 1 | 1 << 5 | 1 << 6 | 1 << 12;      /* disabled, inherit, exclude_kernel, exclude_hv, enable_on_exec */
    return (int)syscall(298, &at, pid, -1, -1, 0);
}

static double value(int fd) {
    uint64_t v[3] = {0};

    if (read(fd, v, sizeof v) != sizeof v) {
        return -1;
    }

    return v[2] ? (double)v[0] * (double)v[1] / (double)v[2] : 0;
}

static double seconds(struct timeval t) {
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 5 || strcmp(argv[1], "--json") || strcmp(argv[3], "--")) {
        fprintf(stderr, "usage: count --json FILE -- COMMAND ARG...\n");
        return 2;
    }

    int go[2];

    if (pipe(go)) {
        perror("count: pipe");
        return 2;
    }

    pid_t pid = fork();

    if (pid < 0) {
        perror("count: fork");
        return 2;
    }

    if (!pid) {
        char c;
        close(go[1]);

        /* exec only after the parent has the counters on this process */
        if (read(go[0], &c, 1) != 1) {
            _exit(127);
        }

        close(go[0]);
        execvp(argv[4], argv + 4);
        perror("count: exec");
        _exit(127);
    }

    close(go[0]);
    int cycles = counter(pid, 0), instructions = counter(pid, 1);

    if (cycles < 0 || instructions < 0) {
        perror("count: perf_event_open");
        kill(pid, SIGKILL);
        waitpid(pid, 0, 0);
        return 2;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (write(go[1], "x", 1) != 1) {
        perror("count: write");
        return 2;
    }

    close(go[1]);
    int status;
    struct rusage ru;

    if (wait4(pid, &status, 0, &ru) != pid) {
        perror("count: wait4");
        return 2;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    FILE *f = fopen(argv[2], "w");

    if (!f) {
        perror("count: --json");
        return 2;
    }

    int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    fprintf(f, "{\"cycles\": %.0f, \"instructions\": %.0f, \"wall\": %.6f, \"user\": %.6f, \"system\": %.6f, \"status\": %d}\n", value(cycles), value(instructions),
            (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9, seconds(ru.ru_utime), seconds(ru.ru_stime), code);
    fclose(f);
    return code;
}
