// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// D4 (channel v7, prompt reading) on the HELPER side: answer one prefill request that main published.
//
// OWNER    : this file owns "a request main published is ALWAYS answered, and a failure is stated, not silent".  What serving
//            means (the picks, the per-token reduce) stays in the engine: RemoteExperts::prefill_serve / serve_prefill.
// CONTRACT : service_prefill(ch, seq, serve) -> bool.  `serve(std::string& err)` computes the request in the channel's prefill
//            region and returns false with `err` set when it cannot.  Imports channel.hpp only (no GPU runtime), so
//            tests/pf_service_test.cpp runs it on a real channel with a fake serve.
// FAILURE  : fail-closed and loud.  A failed serve publishes pf_ok = 0, pf_picks = -1 and the reason in pf_err, and STILL
//            answers (pf_answer): main waits on the answer, and an unanswered request would hang a prompt for its whole timeout.
//            Main then computes that layer's handed-over experts itself (prefill.cpp's failure path), so no pick is dropped.
// CHECKER  : tests/pf_service_test.cpp (sabotage recorded in tests/SABOTAGE.md).
#include "strata/seabraid/channel.hpp"

#include <cstdio>
#include <string>

namespace seabraid {

template <class ServeFn>
inline bool service_prefill(const Channel& ch, uint64_t seq, ServeFn&& serve) {
    Header* h = ch.header();
    std::string err;
    const bool ok = serve(err);
    if (!ok) {
        // serve_prefill sets these itself once it has parsed the request; an early failure (device error, wrong role) did not.
        h->pf_ok = 0;
        h->pf_picks = -1;
        std::snprintf(h->pf_err, sizeof(h->pf_err), "%s", err.empty() ? "prefill serve failed (no reason given)" : err.c_str());
    }
    ch.pf_answer(seq);
    return ok;
}

}  // namespace seabraid
