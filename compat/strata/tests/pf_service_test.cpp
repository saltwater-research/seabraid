// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/pf_service_test.cpp - the helper's D4 prefill service (strata-ext/include/strata/seabraid/pf_service.hpp).  CPU only: shm.
//
//   1 nothing pending : pf_pending() is 0 on a fresh channel, so the helper's wait loop does not spin on a request that is not there.
//   2 served          : a published request is answered (pf_answered), pf_ok stays what the serve set, the sums region is untouched.
//   3 failed          : a serve that fails answers anyway with pf_ok 0, pf_picks -1 and its reason in pf_err.  The case that matters:
//                       main waits on the answer, so a missing answer is a prompt hung for its whole timeout.
//   4 no reason       : a serve that fails without saying why still leaves a sentence in pf_err.
//   5 region          : the channel the helper creates with --prefill-tokens N has a prefill region of N tokens; with 0 it has none.
// House rule: each check was sabotaged once and watched to FAIL (tests/SABOTAGE.md).
#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/pf_service.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

using namespace seabraid;

int main() {
    static constexpr int64_t H = 64, CAP = 32, NL = 4, NE = 8, BLOB = 256, PF_TOKENS = 16;
    static constexpr int32_t PF_K = 4;
    static constexpr uint32_t WMT = 4;
    const std::string name = "/seabraid_pf_service_" + std::to_string((long) getpid());
    std::string err;
    Channel helper, main_side;
    CHECK(helper.create(name, H, CAP, NL, NE, WMT, BLOB, err, PF_TOKENS, PF_K), ("create: " + err).c_str());
    CHECK(main_side.connect(name, H, CAP, NL, NE, WMT, err), ("connect: " + err).c_str());
    if (!helper.valid() || !main_side.valid()) { std::printf("pf_service_test: cannot continue without a channel\n"); return 1; }
    Header* hh = helper.header();

    // 5 region: the sized-by-tokens region exists, and the header says how big
    CHECK(hh->pf_tokens == PF_TOKENS && hh->pf_k == PF_K, "the helper's header carries the region's tokens and k");
    CHECK(helper.prefill_region().bytes > 0, "a --prefill-tokens N channel has a prefill region");
    CHECK(main_side.header()->pf_tokens == PF_TOKENS, "main sees the same region size through its own mapping");

    // 1 nothing pending
    CHECK(helper.pf_pending() == 0, "a fresh channel has no pending prefill request");

    // 2 served: main publishes, the helper answers through the service
    hh->pf_layer = 3; hh->pf_ntok = 5;
    const uint64_t s1 = main_side.pf_publish();
    CHECK(helper.pf_pending() == s1, "a published request is pending for the helper");
    CHECK(!main_side.pf_answered(s1), "and not yet answered");
    bool called = false;
    const bool r1 = service_prefill(helper, s1, [&](std::string&) { called = true; hh->pf_ok = 1; hh->pf_picks = 7; return true; });
    CHECK(r1 && called, "a successful serve returns true and ran");
    CHECK(main_side.pf_answered(s1), "the request is answered");
    CHECK(helper.pf_pending() == 0, "an answered request is no longer pending");
    CHECK(main_side.header()->pf_ok == 1 && main_side.header()->pf_picks == 7, "the serve's own pf_ok / pf_picks reach main untouched");

    // 3 failed: still answered, loudly
    hh->pf_ok = 1; hh->pf_picks = 7;
    const uint64_t s2 = main_side.pf_publish();
    const bool r2 = service_prefill(helper, s2, [&](std::string& e) { e = "expert 9 is not held here"; return false; });
    CHECK(!r2, "a failed serve returns false");
    CHECK(main_side.pf_answered(s2), "a failed request is STILL answered (main must not wait out its timeout)");
    CHECK(main_side.header()->pf_ok == 0, "a failed serve publishes pf_ok 0");
    CHECK(main_side.header()->pf_picks == -1, "a failed serve publishes pf_picks -1");
    CHECK(std::strstr(main_side.header()->pf_err, "expert 9 is not held here") != nullptr, "the reason reaches main in pf_err");

    // 4 no reason
    const uint64_t s3 = main_side.pf_publish();
    service_prefill(helper, s3, [&](std::string&) { return false; });
    CHECK(main_side.pf_answered(s3) && main_side.header()->pf_err[0] != '\0', "a failure without a reason still leaves a sentence");

    // 5 again: no region without --prefill-tokens
    Channel plain;
    const std::string name2 = name + "_plain";
    CHECK(plain.create(name2, H, CAP, NL, NE, WMT, BLOB, err), ("create plain: " + err).c_str());
    CHECK(plain.valid() && plain.header()->pf_tokens == 0 && plain.prefill_region().bytes == 0, "no --prefill-tokens: no prefill region");

    if (g_fail) { std::printf("pf_service_test: %d FAILED\n", g_fail); return 1; }
    std::printf("pf_service_test: all checks passed\n");
    return 0;
}
