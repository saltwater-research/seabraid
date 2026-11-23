// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "sim_cli.h"

#include "sim_budget.h"
#include "sim_providers.h"
#include "sim_util.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum opt_kind { KIND_U32, KIND_U64, KIND_DOUBLE, KIND_STRING, KIND_FLAG };

struct opt {
    const char *name;
    enum opt_kind kind;
    size_t offset;
    const char *help;
};

/* WHY a table: one row per flag keeps the parser's branching constant however many flags are added. */
static const struct opt OPTS[] = {
    {"--hidden", KIND_U32, offsetof(sim_config, hidden), "activation width per token"},
    {"--experts", KIND_U32, offsetof(sim_config, experts), "experts in the model"},
    {"--top-k", KIND_U32, offsetof(sim_config, top_k), "experts picked per token"},
    {"--tokens", KIND_U32, offsetof(sim_config, tokens), "tokens per batch"},
    {"--devices", KIND_U32, offsetof(sim_config, devices), "simulated devices (sessions); each owns a slice of the experts"},
    {"--layers", KIND_U32, offsetof(sim_config, layers), "layers per step"},
    {"--steps", KIND_U32, offsetof(sim_config, steps), "measured steps"},
    {"--warmup", KIND_U32, offsetof(sim_config, warmup), "unmeasured steps before the measured ones"},
    {"--placement-rounds", KIND_U32, offsetof(sim_config, placement_rounds), "placement-planner rounds to run and report (0 = none)"},
    {"--seed", KIND_U64, offsetof(sim_config, seed), "workload random seed"},
    {"--placement-epoch", KIND_U64, offsetof(sim_config, placement_epoch), "stamp every batch with this placement epoch (0 = unversioned)"},
    {"--compact-rows", KIND_FLAG, offsetof(sim_config, compact_rows), "open devices with SEABRAID_PROVIDER_COMPACT_ROWS (takes no value)"},
    {"--fail-rate", KIND_DOUBLE, offsetof(sim_config, fail_rate), "fraction of provider calls to fail on purpose, within [0, 1]"},
    {"--kernel", KIND_STRING, offsetof(sim_config, kernel), "provider kernel, one of the names listed under kernels below"},
    {"--validate-only", KIND_FLAG, offsetof(sim_config, validate_only), "check every flag and limit, then exit 0 without running (takes no value)"},
};

static void set_defaults(sim_config *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    cfg->hidden = 512;
    cfg->experts = 64;
    cfg->top_k = 2;
    cfg->tokens = 16;
    cfg->devices = 1;
    cfg->layers = 4;
    cfg->steps = 20;
    cfg->warmup = 3;
    cfg->seed = 1;
    /* WHY the table's first row: the default kernel is named in one place, sim_providers.c (it is the reference). */
    snprintf(cfg->kernel, sizeof cfg->kernel, "%s", sim_kernel_at(0)->name);
}

/* WHY the first character must be a digit: strtoull skips leading whitespace and accepts '+' and '-', so
 * --seed " -1" parsed as 18446744073709551615. */
static int parse_u64(const char *text, uint64_t *out)
{
    char *end = NULL;
    if (!isdigit((unsigned char)text[0])) {
        return 1;
    }
    errno = 0;
    *out = strtoull(text, &end, 10);
    return errno != 0 || *end != '\0';
}

static int store_value(sim_config *cfg, const struct opt *opt, const char *text)
{
    char *base = (char *)cfg + opt->offset;
    uint64_t wide = 0;
    char *end = NULL;
    switch (opt->kind) {
    case KIND_FLAG:
        *(uint32_t *)(void *)base = 1;
        return 0;
    case KIND_U32:
        if (parse_u64(text, &wide) || wide > UINT32_MAX) {
            return 1;
        }
        *(uint32_t *)(void *)base = (uint32_t)wide;
        return 0;
    case KIND_U64:
        if (parse_u64(text, &wide)) {
            return 1;
        }
        memcpy(base, &wide, sizeof wide);
        return 0;
    case KIND_DOUBLE: {
        /* Same first-character rule as parse_u64, plus an alphabet of decimal digits, '.', 'e', 'E', '+' and '-': a
         * leading digit alone let "0x0.8" and "0X1p-1" through as hex floats, and "nan"/"inf" are refused by the
         * first character. */
        int starts_like_number = (isdigit((unsigned char)text[0]) || text[0] == '.') &&
                                 strspn(text, "0123456789.eE+-") == strlen(text);
        double value = starts_like_number ? strtod(text, &end) : 0.0;
        memcpy(base, &value, sizeof value);
        return !starts_like_number || *end != '\0';
    }
    default:
        if (strlen(text) >= sizeof cfg->kernel) {
            return 1;
        }
        memcpy(base, text, strlen(text) + 1);
        return 0;
    }
}

static const struct opt *find_opt(const char *name)
{
    size_t i;
    for (i = 0; i < sizeof OPTS / sizeof OPTS[0]; i++) {
        if (strcmp(OPTS[i].name, name) == 0) {
            return &OPTS[i];
        }
    }
    return NULL;
}

/* THE OWNER of every per-flag cap (tools/sim/matrix.sh reads them with --print-caps and keeps no copy). Each cap is a
 * bound the simulator needs to build a meaningful workload, not an ABI limit; it stops an absurd value from turning
 * into a hang or a silent huge allocation, and each sits far above any real run:
 * - hidden 2^20: real model widths are under 2^15.
 * - devices 1024: every device is a session with its own worker thread, so thousands are thread exhaustion, not load.
 * - steps and warmup 10^6 each: one latency sample per layer per measured step is kept in memory, and the sum of the
 *   two can never wrap (it used to wrap uint32 and run zero steps).
 * - experts 2^20 and top-k 4096: ranges for the id and pick arrays, not for what can be checked: the JOINT limit in
 *   verifiable_problem bites first (at most 838859 experts, and only with top-k 1 on one device), so most of this range
 *   is refused as a pair.
 * - tokens 2^20, layers 65536: real batches and models (layers under 1000) sit far below, and each loop here is a
 *   per-element or per-pick pass, so a larger value is a hang, not a workload.
 * - placement-rounds 100000: a planner runs for tens of rounds; every round costs tens of microseconds even on one cell, so
 *   the uncapped 4 billion never finished (the part that grows with the cells is in SIM_MAX_WORK, sim_budget.h).
 * The caps bound each flag ALONE and do not bound the work: the product of several is bounded separately, memory by
 * SIM_MAX_BYTES and running time by SIM_MAX_WORK, both in sim_budget.h. Two flags each inside its cap used to give a
 * run that never finishes (--warmup 1000000 --layers 65536 is 6.6e10 layer calls). */
struct cap {
    const char *flag;
    size_t offset;
    uint32_t limit;
};

static const struct cap CAPS[] = {
    {"--hidden", offsetof(sim_config, hidden), 1u << 20},
    {"--experts", offsetof(sim_config, experts), 1u << 20},
    {"--top-k", offsetof(sim_config, top_k), 4096},
    {"--tokens", offsetof(sim_config, tokens), 1u << 20},
    {"--devices", offsetof(sim_config, devices), 1024},
    {"--layers", offsetof(sim_config, layers), 65536},
    {"--steps", offsetof(sim_config, steps), 1000000},
    {"--warmup", offsetof(sim_config, warmup), 1000000},
    {"--placement-rounds", offsetof(sim_config, placement_rounds), 100000},
};

/* One line per cap, "name value": what matrix.sh parses, so the grid check there can never drift from this table. */
void sim_print_caps(FILE *out)
{
    size_t i;
    for (i = 0; i < sizeof CAPS / sizeof CAPS[0]; i++) {
        fprintf(out, "%s %u\n", CAPS[i].flag + 2, (unsigned)CAPS[i].limit);
    }
    fprintf(out, "bytes %.0f\nwork %.0f\n", SIM_MAX_BYTES, SIM_MAX_WORK);
}

static int check_caps(const sim_config *c, char *err, unsigned err_size)
{
    size_t i;
    for (i = 0; i < sizeof CAPS / sizeof CAPS[0]; i++) {
        uint32_t value;
        memcpy(&value, (const char *)c + CAPS[i].offset, sizeof value);
        if (value > CAPS[i].limit) {
            snprintf(err, err_size, "%s must be at most %u", CAPS[i].flag, (unsigned)CAPS[i].limit);
            return 1;
        }
    }
    return 0;
}

/* THE JOINT LIMIT on what the checker can verify (the single owner of this rule).
 * A wrong expert multiplier (id + 2 for id + 1: the "off_by_one" self-test kernel) moves an element by |sum w*a|, out of
 * an element scale M = sum |w*a| * (id + 1). The signed sum cancels: for the synthetic activations (signed, uniform)
 * |sum w*a| / sum |w*a| is about 1.15 / sqrt(p) for p picks, and mean(id + 1) = (E + 1) / 2 for E experts, so the shift
 * relative to M is about 2.3 / (sqrt(p) * (E + 1)). sim_error_bound (sim_util.h) passes anything within
 * (p + d + 3) * FLT_EPSILON of M for d devices, so the shift is seen only while
 *     2.3 / (sqrt(p) * (E + 1)) > (p + d + 3) * FLT_EPSILON.
 * We demand SIM_DETECT_MARGIN times that much room (the 1.15 is an average, and the unlucky element must still show),
 * which, with FLT_EPSILON = 2^-23 and a margin of 4 (rounding 2.3 down to 2), is
 *     (E + 1) * (p + d + 3) * sqrt(p)  <=  2^22.
 * A shape outside it cannot be told from rounding, so it is REFUSED rather than passed. Before this the bound grew with
 * top-k and devices and off_by_one passed at 1048576 experts with top-k 32, at 8192 experts with top-k 4096 and at
 * 262144 experts with top-k 64 on 8 devices. tools/sim/check_cli.sh sweeps experts x top-k x devices against it.
 * WHY 4 and not 2 (measured, not derived): a margin of 2 (limit 2^23) left NO room at p = 1, where the shift is exactly
 * 1 / (id + 1) of the element scale and the float32 error of the kernel itself can cancel it: with ONE checked element
 * (hidden = tokens = layers = steps = 1) off_by_one went undetected 7 of 1000 seeds at 1048575 experts on 4 devices, 6 of
 * 1000 at 699049 on 8, 4 of 1000 at 419429 on 16 and 2 of 1000 at 123360 on 64; at 2^22 the same 18 edge shapes (top-k 1, 2
 * and 8, 1 to 64 devices, cost at the limit) were detected 1000 of 1000 seeds. 1000 seeds cannot show a rate below about
 * 1 in 300; the single-element case of the committed sweep re-checks it on every run. Detection is a measured property of
 * this checker, not a proof. */
#define SIM_DETECT_MARGIN 4.0

static double verifiable_limit(void)
{
    return 2.0 / (SIM_DETECT_MARGIN * (double)FLT_EPSILON);
}

static double verifiable_cost(const sim_config *c)
{
    return ((double)c->experts + 1.0) * ((double)c->top_k + (double)c->devices + 3.0) * sqrt((double)c->top_k);
}

static int verifiable_problem(const sim_config *c, char *err, unsigned err_size)
{
    if (verifiable_cost(c) > verifiable_limit()) {
        snprintf(err, err_size,
                 "--experts %u with --top-k %u and --devices %u cannot be verified: float32 rounding hides a wrong expert "
                 "multiplier; keep (experts + 1) * (top-k + devices + 3) * sqrt(top-k) at most %.0f (it is %.0f)",
                 (unsigned)c->experts, (unsigned)c->top_k, (unsigned)c->devices, verifiable_limit(), verifiable_cost(c));
        return 1;
    }
    return 0;
}

static int memory_problem(const sim_config *c, char *err, unsigned err_size)
{
    if (sim_estimate_bytes(c) > SIM_MAX_BYTES) {
        snprintf(err, err_size,
                 "the workload would allocate about %.1f GiB, above the combined limit of %.0f GiB (it grows with "
                 "--tokens x --hidden x --devices, --tokens x --top-k x --devices, --layers x --steps and, with "
                 "--placement-rounds, --layers x --experts); lower them",
                 sim_estimate_bytes(c) / 1073741824.0, SIM_MAX_BYTES / 1073741824.0);
        return 1;
    }
    return 0;
}

static int work_problem(const sim_config *c, char *err, unsigned err_size)
{
    if (sim_estimate_work(c) > SIM_MAX_WORK) {
        snprintf(err, err_size,
                 "the run would take about %.0f minutes, above the combined limit of %.0f minutes (it grows with "
                 "(--warmup + --steps) x --layers x --tokens x --hidden x --top-k x --devices and, with "
                 "--placement-rounds, --placement-rounds x --layers x --experts); lower them",
                 sim_estimate_work(c) / 6e10, SIM_MAX_WORK / 6e10);
        return 1;
    }
    return 0;
}

static const char *shape_text(const sim_config *c)
{
    if (c->hidden == 0 || c->tokens == 0 || c->layers == 0 || c->steps == 0) {
        return "--hidden, --tokens, --layers and --steps must be at least 1";
    }
    if (c->top_k == 0 || c->top_k > c->experts) {
        return "--top-k must be between 1 and --experts";
    }
    if (c->devices == 0 || c->devices > c->experts) {
        return "--devices must be between 1 and --experts (each device owns at least one expert)";
    }
    return NULL;
}

static int shape_problem(const sim_config *c, char *err, unsigned err_size)
{
    const char *text = shape_text(c);
    if (text == NULL && !(c->fail_rate >= 0.0 && c->fail_rate <= 1.0)) {
        text = "--fail-rate must be within [0, 1]"; /* written as >= and <= so a NaN is refused too */
    }
    if (text != NULL) {
        snprintf(err, err_size, "%s", text);
    }
    return text != NULL;
}

/* WHY the kernel names are NOT listed here: sim_providers.c owns the table of kernels (name, function, help) and this
 * file asks it, so a kernel is added or removed in exactly one place. */
static int kernel_problem(const sim_config *c, char *err, unsigned err_size)
{
    size_t i;
    int used;
    if (sim_kernel_find(c->kernel) != NULL) {
        return 0;
    }
    used = snprintf(err, err_size, "--kernel must be one of:");
    for (i = 0; sim_kernel_at(i) != NULL && used > 0 && (unsigned)used < err_size; i++) {
        used += snprintf(err + used, err_size - (unsigned)used, " %s", sim_kernel_at(i)->name);
    }
    return 1;
}

/* Returns 1 with a one-line reason in `err`, else 0. Order: caps (name the flag), shape, kernel, then the two joint
 * limits, which only mean something for a well-formed shape. */
static int validate(const sim_config *c, char *err, unsigned err_size)
{
    return check_caps(c, err, err_size) || shape_problem(c, err, err_size) || kernel_problem(c, err, err_size) ||
           verifiable_problem(c, err, err_size) || memory_problem(c, err, err_size) || work_problem(c, err, err_size);
}

static void print_default(FILE *out, const sim_config *cfg, const struct opt *opt)
{
    const char *base = (const char *)cfg + opt->offset;
    uint32_t narrow = 0;
    uint64_t wide = 0;
    double real = 0.0;
    switch (opt->kind) {
    case KIND_FLAG:
        return;
    case KIND_U32:
        memcpy(&narrow, base, sizeof narrow);
        fprintf(out, " [default %u]", (unsigned)narrow);
        return;
    case KIND_U64:
        memcpy(&wide, base, sizeof wide);
        fprintf(out, " [default %llu]", (unsigned long long)wide);
        return;
    case KIND_DOUBLE:
        memcpy(&real, base, sizeof real);
        fprintf(out, " [default %g]", real);
        return;
    default:
        fprintf(out, " [default %s]", base);
    }
}

/* WHY generated from OPTS: the usage block lists exactly the flags the parser accepts, with the defaults it applies. */
void sim_print_usage(FILE *out)
{
    sim_config cfg;
    size_t i;
    set_defaults(&cfg);
    fprintf(out, "usage: seabraid_sim [flags]\n\nDrives the public C ABI with synthetic layers and prints one JSON line.\n"
                 "Exit codes: 0 ok, 1 setup or ABI failure, 2 numeric mismatch, 3 bookkeeping mismatch.\n\nflags:\n");
    for (i = 0; i < sizeof OPTS / sizeof OPTS[0]; i++) {
        const char *value = OPTS[i].kind == KIND_STRING ? " NAME" : OPTS[i].kind == KIND_DOUBLE ? " X" : " N";
        if (OPTS[i].kind == KIND_FLAG) {
            value = "";
        }
        fprintf(out, "  %s%s\n      %s", OPTS[i].name, value, OPTS[i].help);
        print_default(out, &cfg, &OPTS[i]);
        fputc('\n', out);
    }
    fprintf(out, "  --print-caps\n      print every per-flag cap as \"name value\" lines and exit 0\n"
                 "  -h, --help\n      print this text and exit 0\n\nkernels (--kernel NAME):\n");
    for (i = 0; sim_kernel_at(i) != NULL; i++) {
        fprintf(out, "  %s\n      %s%s\n", sim_kernel_at(i)->name,
                sim_kernel_at(i)->self_test ? "SELF-TEST, wrong on purpose: " : "", sim_kernel_at(i)->help);
    }
}

/* Consumes argv[*i] (and its value, advancing *i). Returns 0, 1 with `err` set, or SIM_CLI_HELP / SIM_CLI_CAPS. */
static int take_arg(sim_config *cfg, int argc, char **argv, int *i, char *err, unsigned err_size)
{
    const char *arg = argv[*i];
    const struct opt *opt;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
        return SIM_CLI_HELP;
    }
    if (strcmp(arg, "--print-caps") == 0) {
        return SIM_CLI_CAPS;
    }
    opt = find_opt(arg);
    if (opt != NULL && opt->kind == KIND_FLAG) {
        store_value(cfg, opt, "");
        return 0;
    }
    if (opt == NULL || *i + 1 >= argc) {
        snprintf(err, err_size, "unknown flag or missing value: %s", arg);
        return 1;
    }
    (*i)++;
    if (store_value(cfg, opt, argv[*i])) {
        snprintf(err, err_size, "bad value for %s: %s", opt->name, argv[*i]);
        return 1;
    }
    return 0;
}

int sim_parse_args(int argc, char **argv, sim_config *cfg, char *err, unsigned err_size)
{
    int i;
    set_defaults(cfg);
    for (i = 1; i < argc; i++) {
        int taken = take_arg(cfg, argc, argv, &i, err, err_size);
        if (taken) {
            return taken;
        }
    }
    return validate(cfg, err, err_size) ? 1 : 0;
}
