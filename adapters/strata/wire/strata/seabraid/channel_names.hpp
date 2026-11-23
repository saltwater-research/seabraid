// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid channel names: the SEABRAID_STRATA_SHM list.  OWNER of the list's grammar and cap.
// CONTRACT: one pure function, parse_channel_list.  The engine's seabraid::channel_name(card, ...) (compat patch 0017,
// remote_experts.cpp) reads SEABRAID_STRATA_SHM, parses it here and picks entry `card`: card id = list position = Seabraid
// rung - 1, fixed for the session.  WHY channel_name is not here: it is the engine's lookup (its default for an unset
// variable is the engine's), and one definition of it lives with its only caller.
// WHY position and not a name lookup: the installer is the one place that maps a card's rank to list order (plan.py
// _admission), and a re-rank is a reinstall that rewrites the units and the list together, so position cannot drift.
// FAILURE: fail-fast; every refusal names the offending entry, the duplicate or the cap; nothing is returned on a refusal.
// SIZE: <=80 lines.  CHECKER: adapters/strata/channel_names_test.cpp.
#include <string>
#include <vector>

namespace seabraid {

inline constexpr int kMaxChannels = 3;   ///< Strata holds three remote tiers (expert_source.hpp `RemoteExperts* remote[3]`)

namespace detail {
inline bool channel_char_ok(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
}
/// One `/name` entry: a leading slash then 1..63 of [A-Za-z0-9_.-] (shm_open names; no second slash, no "..").
inline bool channel_entry_ok(const std::string& e) {
    if (e.size() < 2 || e.size() > 64 || e[0] != '/' || e == "/." || e == "/..") return false;
    for (size_t i = 1; i < e.size(); ++i) if (!channel_char_ok(e[i])) return false;
    return true;
}
}  // namespace detail

/// Parses a SEABRAID_STRATA_SHM value: comma-separated `/name` entries, 1..kMaxChannels of them, no duplicates, no empty entry.
/// On false `out` is empty and `err` names what was refused.
inline bool parse_channel_list(const char* env, std::vector<std::string>& out, std::string& err) {
    out.clear();
    if (env == nullptr || env[0] == '\0') { err = "SEABRAID_STRATA_SHM is empty; want /name[,/name...]"; return false; }
    std::vector<std::string> names;
    const std::string s(env);
    for (size_t start = 0;;) {
        const size_t comma = s.find(',', start);
        const std::string e = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!detail::channel_entry_ok(e)) {
            err = "SEABRAID_STRATA_SHM entry '" + e + "' is not /name (1..63 of A-Z a-z 0-9 _ . -)";
            return false;
        }
        for (const std::string& seen : names)
            if (seen == e) { err = "SEABRAID_STRATA_SHM lists '" + e + "' twice; one helper per channel"; return false; }
        names.push_back(e);
        if (names.size() > (size_t) kMaxChannels) {
            err = "SEABRAID_STRATA_SHM lists more than " + std::to_string(kMaxChannels) + " channels; Strata holds three remote tiers";
            return false;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    out.swap(names);
    return true;
}

}  // namespace seabraid
