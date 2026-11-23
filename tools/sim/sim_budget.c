// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "sim_budget.h"

#include "sim_placement.h"

/* THE OWNER of the memory estimate; the JSON line prints it ("estimated_bytes") and check_cli.sh compares it with the
 * peak RSS /usr/bin/time measures, so it is a bound that is tested, not a hope. Peak RSS of seabraid_sim, measured
 * (Linux x86-64, glibc 2.43, Release, one run per shape, 2026-11-19; check_cli.sh repeats the comparison):
 *   - a one-cell run: 3.0 MiB on 1 device, 3.5 on 64, 9.1 on 512 (the library, stdio, and about 12 KiB per device for its
 *     session and thread): 8 MiB plus 16 KiB per device. (Measured under a bare /usr/bin/time: a wrapper such as nice or
 *     timeout adds its own 5 MiB, because the peak RSS is inherited across exec.)
 *   - per activation cell (tokens * hidden): 20 + 12 per device measured (32.0 bytes on 1 device at 33.5M cells, 44 on 2,
 *     68 on 4, 116 on 8, 212 on 16, 790 on 64), estimated 20 + 15 per device. The sim itself allocates 20 plus 4 per
 *     device (activations, outputs, reference, magnitude); the rest of the per-device part is inferred from the fit to be
 *     libseabraid's per-session copies, not read from its source. The first constant here was 13 per device: it left the
 *     1-device large shapes 3.3 percent over, and --compact-rows under;
 *   - per cell with --compact-rows: the library keeps a compacted copy of the rows, 4 bytes per cell on 1 and 2 devices
 *     (+127 MiB over 33.5M cells). With more devices the run measured LOWER than the plain one (partial routes ship fewer
 *     rows), so 4 per cell is an upper bound there. Platform-sensitive like everything here: a different libc or
 *     allocator moves the per-cell part;
 *   - per pick (tokens * top-k): 8 + 15.5 per device measured (fitted over devices 1, 2, 4, 8, 16, 32), estimated 8 + 18;
 *   - per latency sample (layers * steps): 15 (8 stored, and the glibc qsort that sorts them copies them again): 17;
 *   - placement (layers * experts cells, only when it runs): 38 to 41 per cell: 48.
 * Measured margin (estimate / peak RSS) on the 31 shapes measured on 2026-11-19: 1.09 at the least, on the large
 * 1-device compact-rows run; 1.10 to 1.14 on the other large cell and pick shapes, more on small ones. The fixed part
 * is more than twice the measured one. So this bounds the RSS from above by at least 8 percent on the shapes measured,
 * and that is all it claims: not an allocator-exact count, other libcs (musl, macOS) differ, and a different library
 * build changes the per-device part. check_cli.sh requires estimate >= peak RSS on a handful of shapes, compact-rows
 * included, wherever GNU /usr/bin/time exists. */
#define SIM_BASE_BYTES 8388608.0
#define SIM_DEVICE_BYTES 16384.0

double sim_estimate_bytes(const sim_config *c)
{
    double cells = (double)c->tokens * (double)c->hidden;
    double picks = (double)c->tokens * (double)c->top_k;
    double devices = (double)c->devices;
    double bytes = SIM_BASE_BYTES + SIM_DEVICE_BYTES * devices + cells * (20.0 + 15.0 * devices) +
                   picks * (8.0 + 18.0 * devices) + 4.0 * (double)c->experts +
                   17.0 * (double)c->layers * (double)c->steps;
    if (c->compact_rows) {
        bytes += 4.0 * cells;
    }
    if (sim_placement_skip_reason(c) == NULL) {
        bytes += 48.0 * (double)c->layers * (double)c->experts;
    }
    return bytes;
}

/* THE OWNER of the work estimate, in nanoseconds on the reference box (Ryzen 7 5800, 16 threads, Release or -O2).
 * Wall time per run, measured 2026-11-19 (one run per shape, sequential, nice; 49 shapes, listed with their estimates
 * in the commit that set these constants). A layer call is one submit and one collect per device plus the verify:
 *   - fixed per call: 6.8 us on 1 device (1e6 calls = 6.8 s) and about 1000 ns per device on top at 64 devices;
 *   - per activation cell: 8.4 ns on 1 device with top-k 2, 13 on 4 devices, 18 on 4 devices with top-k 8, so
 *     8 + 2 * top-k * devices bounds it (and is 4 times over at the largest top-k * devices);
 *   - per pick (tokens * top-k): what a run with a single activation per token pays. Measured, net of the cell term:
 *     10 ns on 1 device, 28 on 2, 38 on 4, 41 on 8 (top-k 8), 89 on 16, 108 on 32, 223 on 64; a 1-device run with
 *     262144 tokens and top-k 64 spends 12.8 ns per pick. The first model had no pick term and ran 5.4 times over
 *     its estimate at hidden 1, tokens 65536, top-k 64; the accepted shape `--hidden 1 --tokens 262144 --experts 4096
 *     --top-k 64 --devices 1 --layers 1000 --steps 15 --warmup 1` took about an hour against a ten-minute cap.
 *     22 + 12 per device keeps every measured shape at least 1.14 times under its estimate. Single runs vary by up to
 *     40 percent on this shared box (one shape took 1.85 to 2.82 s in five runs), so the margin is over the slowest run;
 *   - placement, 4M cells (1024 layers x 4096 experts), 4 devices: 0.49 s for one round, 1.31 for 5, 2.99 for 20,
 *     11.9 for 100. A round costs about 50 ns per cell while the drifting heat still has promotions to move and 22 to
 *     29 once it settles; filling and opening the map is a one-time 55 to 85 ns per cell (136 ns per cell at 16M cells,
 *     one round). So: 100 per cell once, 55 per cell per round (up to 2.9 times over on a long run, never under), and
 *     20000 per round of fixed cost (a 128-cell run of 20000 rounds took 4.5 us per round).
 * The estimate is at or above every measured time (the least margin was 1.14, over the slowest run of both a Release
 * (-O3) build and a no-build-type (-O2) one). It is calibrated on an OPTIMISED build:
 * tools/sim/CMakeLists.txt adds -O2 when no build type is given, and a Debug build ran up to 3.6 times over this
 * estimate. */
double sim_estimate_work(const sim_config *c)
{
    double cells = (double)c->tokens * (double)c->hidden;
    double picks = (double)c->tokens * (double)c->top_k;
    double devices = (double)c->devices;
    double layer = cells * (8.0 + 2.0 * (double)c->top_k * devices) + picks * (22.0 + 12.0 * devices) + 6000.0 +
                   2000.0 * devices;
    double work = ((double)c->warmup + (double)c->steps) * (double)c->layers * layer;
    if (sim_placement_skip_reason(c) == NULL) {
        double placement_cells = (double)c->layers * (double)c->experts;
        work += placement_cells * (100.0 + 55.0 * (double)c->placement_rounds) + 20000.0 * (double)c->placement_rounds;
    }
    return work;
}
