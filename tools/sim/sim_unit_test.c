// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Unit checks for the pure helpers in sim_util.h: the tolerance that judges a run (a correct large-expert run must pass,
 * a genuinely wrong or non-finite result must fail) and the overflow-checked allocation. Prints "sim unit: ok" last. */
#include "sim_util.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

static int failures;

static void expect(int condition, const char *what)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

int main(void)
{
    size_t product = 0;
    void *block;

    /* The bound for 2 picks on 1 device is 6 * FLT_EPSILON * magnitude (see sim_error_bound). */
    double mag = 131073.0; /* one term of weight 1, activation 1, id 131072: a 262144-expert run */
    double bound = sim_error_bound(mag, 2, 1);
    double worst = 0.0;
    expect(fabs(bound - 6.0 * (double)FLT_EPSILON * mag) < 1e-12, "the bound is (picks + devices + 3) * eps * magnitude");

    /* Float32 rounding of one ulp-scale step passes; a wrong result fails at every magnitude. */
    expect(sim_error_ok(sim_scaled_error(mag + 0.5 * bound, mag, bound)), "an error inside the bound passes");
    expect(!sim_error_ok(sim_scaled_error(mag + 2.0 * bound, mag, bound)), "an error of twice the bound fails");
    expect(!sim_error_ok(sim_scaled_error(-mag, mag, bound)), "a sign flip fails");
    /* The off-by-one multiplier (id + 2) moves an element by 1 * weight * activation = 1 / mean(id + 1) of the
     * magnitude. At 262144 experts that is above the bound (it used to hide under the flat 1e-4). */
    expect(!sim_error_ok(sim_scaled_error(mag + 1.0, mag, bound)), "an off-by-one multiplier at 262144 experts fails");
    /* An all-zero element has no rounding to forgive. */
    bound = sim_error_bound(0.0, 2, 1);
    expect(sim_error_ok(sim_scaled_error(0.0, 0.0, bound)), "zero equals zero");
    expect(!sim_error_ok(sim_scaled_error(1e-3, 0.0, bound)), "a wrong value where the magnitude is zero fails");

    /* NaN and infinity must FAIL: every ordered comparison with NaN is false, so a maximum used to skip it. */
    expect(!sim_error_ok(sim_scaled_error(NAN, 1.0, 1.0)), "a NaN output fails");
    expect(!sim_error_ok(sim_scaled_error(INFINITY, 1.0, 1.0)), "a +inf output fails");
    expect(!sim_error_ok(sim_scaled_error(-INFINITY, 1.0, 1.0)), "a -inf output fails");
    expect(!sim_error_ok(sim_scaled_error(1.0, NAN, 1.0)), "a NaN reference fails");
    expect(!sim_error_ok(sim_scaled_error(1.0, 1.0, NAN)), "a NaN bound fails");
    expect(!sim_error_ok(NAN), "sim_error_ok rejects NaN");
    expect(!sim_error_ok(INFINITY), "sim_error_ok rejects +inf");
    expect(sim_error_ok(SIM_MAX_ERR), "sim_error_ok accepts exactly the bound");
    /* The workload keeps a running maximum; a NaN output in the middle must still leave a failing maximum. */
    {
        double got[3] = {1.0, NAN, 1.0};
        for (int i = 0; i < 3; i++) {
            double scaled = sim_scaled_error(got[i], 1.0, 1.0);
            worst = scaled > worst ? scaled : worst;
        }
    }
    expect(!sim_error_ok(worst), "a NaN between good outputs leaves a failing maximum");

    expect(sim_mul_size((size_t)1 << 16, (size_t)1 << 16, &product) && product == ((size_t)1 << 32),
           "a fitting product is returned");
    expect(!sim_mul_size(SIZE_MAX / 2 + 1, 2, &product), "a product past size_t is refused");
    expect(sim_calloc2(SIZE_MAX / 2 + 1, 2, 1) == NULL, "calloc2 refuses a wrapping element count");
    expect(sim_calloc2(SIZE_MAX / 2, 1, 4) == NULL, "calloc2 refuses a wrapping byte count");
    block = sim_calloc2(4, 4, 4);
    expect(block != NULL, "calloc2 allocates a small block");
    free(block);

    if (failures == 0) {
        printf("sim unit: ok\n");
    }
    return failures != 0;
}
