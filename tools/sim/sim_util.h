// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Small shared helpers for the simulator: a seeded PRNG and a monotonic clock. WHY header-only: both are
 * a few lines, used by every module, and have no state beyond the caller's sim_rng. */
#ifndef SEABRAID_SIM_UTIL_H
#define SEABRAID_SIM_UTIL_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct sim_rng {
    uint64_t state;
} sim_rng;

/* splitmix64 step: scrambles any seed (including 0) into a well-mixed 64-bit value. */
static inline uint64_t sim_mix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void sim_rng_seed(sim_rng *rng, uint64_t seed, uint64_t stream)
{
    rng->state = seed ^ (stream * 0xD1B54A32D192ED03ull);
}

static inline uint64_t sim_rng_next(sim_rng *rng)
{
    return sim_mix64(&rng->state);
}

/* Uniform in [0, 1). */
static inline double sim_rng_unit(sim_rng *rng)
{
    return (double)(sim_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

/* Uniform integer in [0, bound); bound must be nonzero. The modulo bias is irrelevant for synthetic data. */
static inline uint32_t sim_rng_below(sim_rng *rng, uint32_t bound)
{
    return (uint32_t)(sim_rng_next(rng) % bound);
}

/* The simulator's pass/fail line for one output element, in units of a rounding bound DERIVED FROM THE COMPUTATION.
 * An element is sum over `picks` of weight * activation * (expert_id + 1), computed in float32, then merged across
 * `devices` partial outputs in float32. Each term has 2 rounded multiplies and the cast of (id + 1), each adds one
 * rounding, and the sums add (picks - 1) + (devices - 1) more, every one a relative error of at most u = FLT_EPSILON / 2
 * of the running magnitude, which never exceeds M = sum |weight * activation * (id + 1)|. So the error is at most
 * (picks + devices + 2) * u * M; the bound below uses (picks + devices + 3) * FLT_EPSILON * M, twice that, and nothing
 * looser. WHY scaled by M and not by |reference|: the activations have both signs, so an element can cancel to near
 * zero while its terms are huge, and |reference| would then reject correct float32 results; M is what the rounding
 * actually scales with. WHY this replaced a flat 1e-4: that was ~700x the real error and let an off-by-one expert
 * multiplier through from 262144 experts. The float32 limit is real: a wrong multiplier (id + 2) shifts an element by
 * about 2.3 / (sqrt(picks) * (experts + 1)) relative, visible only while that exceeds this bound, which GROWS with picks
 * and devices; sim_cli.c (verifiable_problem) derives the joint limit and refuses a shape past it. */
#define SIM_MAX_ERR 1.0 /* the largest passing sim_scaled_error: the error is within the bound */

static inline double sim_error_bound(double magnitude, uint32_t picks, uint32_t devices)
{
    double bound = ((double)picks + (double)devices + 3.0) * (double)FLT_EPSILON * magnitude;
    return bound > FLT_MIN ? bound : FLT_MIN; /* WHY the floor: an all-zero element needs got == 0, not 0/0 */
}

/* |got - want| / bound. WHY every non-finite case is +inf: a NaN would lose every later `a > b` maximum (max ignores
 * NaN), so a NaN output once passed; +inf compares, propagates through a maximum and fails SIM_MAX_ERR. */
static inline double sim_scaled_error(double got, double want, double bound)
{
    double err = fabs(got - want);
    if (!isfinite(got) || !isfinite(want) || !isfinite(err) || !(bound > 0.0)) {
        return INFINITY;
    }
    return err / bound;
}

/* The one pass test. WHY written as <= and never `> SIM_MAX_ERR`: a NaN fails every ordered comparison, so only the
 * positive form rejects it. */
static inline int sim_error_ok(double scaled)
{
    return scaled <= SIM_MAX_ERR;
}

/* calloc of a * b elements of `elem` bytes; NULL (never a wrapped, too-small block) when the element count overflows
 * size_t. WHY: the workload sizes are products of user-chosen uint32 flags, and a wrapped product would be written past. */
static inline int sim_mul_size(size_t a, size_t b, size_t *out)
{
    if (b != 0 && a > SIZE_MAX / b) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static inline void *sim_calloc2(size_t a, size_t b, size_t elem)
{
    size_t count, bytes;
    /* WHY check count * elem here too, not leave it to calloc: an optimising build folds the unit test's oversized
     * constants into the calloc call and -Werror=alloc-size-larger-than rejects it; checking first keeps calloc unreached. */
    if (!sim_mul_size(a, b, &count) || !sim_mul_size(count, elem, &bytes)) {
        return NULL;
    }
    (void)bytes;
    return calloc(count, elem);
}

static inline uint64_t sim_now_ns(void)
{
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts); /* WHY monotonic: latencies must not jump with NTP steps */
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#endif
