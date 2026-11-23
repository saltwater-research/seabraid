// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// OWNER: the grammar of one /proc/<pid>/stat line, and nothing else: where the state, the start time and the wait status sit,
//   and how a wait status decodes.  Moved here from the adapter's helper_exit.cpp so the engine-side PeerWatch (peer_watch.hpp)
//   and the adapter's exit probe read the same fields from one definition.
// CONTRACT: pure text functions over a NUL-terminated line, no I/O, no allocation, safe from any thread.  Field numbers are the
//   proc(5) ones: 3 state, 22 starttime (clock ticks since boot), 52 exit_code (a wait status, readable while the task is a zombie).
// SIZE BUDGET: <=80 lines.
// FAILURE CONTRACT: a line that does not parse answers false and leaves its outputs alone; a caller treats "unreadable" as
//   "no new information", never as death (the pidfd is the authority on death).
// CHECKER: adapters/strata/peer_watch_test.cpp (W1: a ')' inside comm, Z and X states, field 22 and 52) and helper_hardening_test.
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace seabraid::procstat {

/// The first field after the ")" that closes comm.  WHY the LAST ')': comm is the executable name, up to 15 bytes of anything
/// but NUL, so it may itself hold spaces or ")" and only the final ")" is the real end of field 2.
/// Returns the state character's address, or nullptr when the line is not "<pid> (<comm>) <state> ...".
inline const char* state_ptr(const char* line) noexcept {
    const char* close = std::strrchr(line, ')');
    return (close == nullptr || close[1] != ' ' || close[2] == '\0') ? nullptr : close + 2;
}

/// The field `count` places after `field`, or nullptr when the line ends first.
inline const char* skip_fields(const char* field, int count) noexcept {
    for (; field != nullptr && count > 0; --count) {
        field = std::strchr(field, ' ');
        if (field != nullptr) ++field;
    }
    return field;
}

inline bool state_of(const char* line, char& state) noexcept {
    const char* at = state_ptr(line);
    if (at == nullptr) return false;
    state = *at;
    return true;
}

/// 'Z' (zombie) and 'X' (dead): the task is over though its thread group may not be, and it may not be reaped yet.
inline bool dead_state(char state) noexcept { return state == 'Z' || state == 'X'; }

/// Reads the unsigned decimal `skip` fields after the state field (anything that is not a digit there is a failed parse).
inline bool number_after_state(const char* line, int skip, unsigned long long& out) noexcept {
    const char* field = skip_fields(state_ptr(line), skip);
    if (field == nullptr || *field < '0' || *field > '9') return false;
    out = std::strtoull(field, nullptr, 10);
    return true;
}

/// Field 22: when the process started, in clock ticks since boot.  (pid, starttime) names one process for the life of the boot.
inline bool starttime_of(const char* line, uint64_t& out) noexcept {
    unsigned long long value = 0;
    if (!number_after_state(line, 22 - 3, value)) return false;
    out = value;
    return true;
}

/// Field 52, the wait status (kernels before 3.5 have no field 52: false, and the exit is still real).
inline bool exit_status_of(const char* line, long& status) noexcept {
    unsigned long long value = 0;
    if (!number_after_state(line, 52 - 3, value)) return false;
    status = static_cast<long>(value);
    return true;
}

/// A wait status: the signal in the low 7 bits, else the exit code in bits 8-15.
struct WaitStatus { bool signaled; int code; };
inline WaitStatus decode_wait_status(long status) noexcept {
    const bool signaled = (status & 0x7f) != 0;
    return {signaled, signaled ? static_cast<int>(status & 0x7f) : static_cast<int>((status >> 8) & 0xff)};
}

}  // namespace seabraid::procstat
