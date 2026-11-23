// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The --retier* usage text (split from retier_loop.cpp, which owns the flags themselves, at its 400-line point).
// OWNER: the help text of every --retier flag; a new flag's line goes here with its parse row in retier_loop.cpp.
// SIZE BUDGET: 60 lines.  FAILURE CONTRACT: none (a constant).  CHECKER: retier_loop_test (the help names each pyramid flag).
#include "strata/seabraid/retier_loop.hpp"

namespace seabraid {

const char* retier_help() {
    return
        "  --retier off|on|zero-op  seabraid D7 live re-tiering of experts between the primary and the helper card.\n"
        "                       Default off: no thread, no snapshot, the engine is unchanged.  on: every\n"
        "                       --retier-every windows a background thread ranks every expert by the adaptive tier's\n"
        "                       routing counts and the engine thread swaps the hottest misses for the coldest helper\n"
        "                       residents (same layer).  zero-op: runs the whole machinery (snapshot, ranker, hand-off,\n"
        "                       log line) but discards the ops and never enters the fence, for the bit-identical test.\n"
        "                       Needs --adapt-every > 0 AND --adapt-swaps > 0 (else no counts exist: startup error).\n"
        "  --retier-every N     a round every N decode windows (default 128; windows, not seconds, because the routing\n"
        "                       counts age per window, so arms of an A/B on the same prompts stay comparable)\n"
        "  --retier-moves N     expert moves per round (a swap is one load + one evict = 2; default 4, max 64)\n"
        "  --retier-margin X    a challenger must out-heat the resident it displaces by this factor (default 1.25, >= 1)\n"
        "  --retier-cooldown N  rounds an expert that moved is frozen (default 8)\n"
        "  --retier-ceiling X   gate on admitting loads/swaps: a layer's helper heat share may not rise above X (default 0.08, 0..1); a layer over it admits no swaps until its heat drifts back under (never actively restored)\n"
        "  --retier-promote legacy|seabraid|off  with --retier on: instead of the card-2 <-> RAM round, promote card 2's hot\n"
        "                       experts to card 1, the coldest card-1 expert of the same layer dropping to card 2 (the\n"
        "                       N-card ladder's first live move).  legacy (alias: on) applies each pair through the\n"
        "                       ownership map's prepare/commit; seabraid plans and applies through Seabraid's placement\n"
        "                       driver (one fenced round, one published routing table) and is the only value the private\n"
        "                       Seabraid helper accepts.  Both run at a quiet window boundary; a timeout or a failed copy\n"
        "                       closes the card (its experts then come from RAM / the drive).  Default off.\n"
        "  --retier-fallback 0|1  with --retier-promote seabraid: also swap between the last helper and the engine's own\n"
        "                       copy (RAM or the drive).  Default 0.\n"
        "  --retier-hops N      with --retier-promote seabraid: one expert may climb up to N rungs in one round (default 1,\n"
        "                       max 64).  A climb of h hops writes 2h experts, so --retier-moves must cover them.\n"
        "  --retier-file-moves 0|1  with --retier-promote seabraid: a helper that opened the model file loads a promoted expert\n"
        "                       itself instead of receiving its bytes (needs the pack's expert-sums.v1 manifest; without a fresh\n"
        "                       one this turns itself off with one warning).  A refused or unavailable reference falls back to\n"
        "                       bytes.  Default 0.\n";
}

}  // namespace seabraid
