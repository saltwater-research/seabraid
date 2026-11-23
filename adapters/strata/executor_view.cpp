// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "executor_view.hpp"

#include "diag.hpp"

#include <chrono>
#include <thread>

namespace seabraid::strata_adapter {
namespace {

constexpr uint32_t kMaxRung = 64;  // the quarantine mask is one uint64_t
// The one failure log site: a reason word plus two counts, never paths or data.
bool log_refused(int32_t status, const char* why, unsigned long long a, unsigned long long b) noexcept {
    diag::logf(diag::kError, status, "executor view: refused %s (%llu, %llu)", why, a, b);
    return false;
}

uint64_t fnv1a(uint64_t h, const void* data, size_t n) noexcept {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
uint64_t hash_table(const uint8_t* exec, size_t n, uint32_t layers, uint32_t experts) noexcept {
    return fnv1a(fnv1a(fnv1a(14695981039346656037ULL, &layers, 4), &experts, 4), exec, n);
}
bool rung_clear(uint64_t quarantined, uint32_t rung) noexcept { return rung < kMaxRung && !((quarantined >> rung) & 1U); }
// WHY 1 s: a reader holds a table for a few instructions, so a longer wait means a stuck
// reader, and refusing keeps the old verified table live instead of overwriting memory in use.
bool wait_idle(const std::atomic<uint32_t>& readers) noexcept {
    for (auto end = std::chrono::steady_clock::now() + std::chrono::seconds(1);
         readers.load() != 0 && std::chrono::steady_clock::now() < end;) std::this_thread::yield();
    return readers.load() == 0;
}

bool well_formed(const uint8_t* exec, size_t n, const seabraid_placement_snapshot_v1& s) noexcept {
    const uint64_t cells = uint64_t{s.layers} * s.experts;
    const bool header = s.abi_version == SEABRAID_ABI_VERSION && s.struct_size >= sizeof s && s.fallback_rung < kMaxRung;
    const bool sized = exec != nullptr && cells != 0 && cells == n && cells <= SEABRAID_PLACEMENT_MAX_CELLS;
    return (header && sized) || log_refused(SEABRAID_INVALID, "malformed publish", cells, n);
}
// WHY refuse a quarantined owner: Rust resolves those cells to the fallback, so one is a stale table.
bool no_lost_owner(const uint8_t* exec, size_t n, uint64_t quarantined) noexcept {
    for (size_t i = 0; i < n; ++i)
        if (!rung_clear(quarantined, exec[i])) return log_refused(SEABRAID_INVALID, "rung at cell", i, exec[i]);
    return true;
}
}  // namespace

bool ExecutorView::accepts(const uint8_t* exec, size_t n, const seabraid_placement_snapshot_v1& s) const noexcept {
    if (!well_formed(exec, n, s)) return false;
    const Table* live = current_.load();
    if (live && (live->layers != s.layers || live->experts != s.experts))
        return log_refused(SEABRAID_INVALID, "shape change", s.layers, s.experts);
    if (s.epoch < epoch()) return log_refused(SEABRAID_BUSY_OR_STATE, "epoch going back", s.epoch, epoch());
    // WHY equal epoch needs equal bytes: an idle round may republish its own epoch, but a changed
    // table without an epoch bump would let two different routings share one decision id.
    if (live && s.epoch == epoch() && hash_table(exec, n, s.layers, s.experts) != table_hash())
        return log_refused(SEABRAID_BUSY_OR_STATE, "same epoch, different table", s.epoch, 0);
    return no_lost_owner(exec, n, s.quarantined_rungs);
}

bool ExecutorView::publish(const uint8_t* exec, size_t n, const seabraid_placement_snapshot_v1& s) noexcept {
    // WHY a flag and not a mutex: there is one legal publisher (the engine thread at the fence),
    // so a second caller is a wiring bug to refuse loudly, never to queue behind or let write the idle table.
    if (publishing_.exchange(true)) return log_refused(SEABRAID_BUSY_OR_STATE, "concurrent publish", n, 0);
    Table& idle = tables_[current_.load() == &tables_[0] ? 1 : 0];  // the one no reader can enter
    bool ok = accepts(exec, n, s);
    if (ok && !wait_idle(idle.readers)) ok = log_refused(SEABRAID_BUSY_OR_STATE, "publish, readers hold idle table", n, 0);
    try {
        if (ok) idle.exec.assign(exec, exec + n);
    } catch (...) { ok = log_refused(SEABRAID_INTERNAL, "publish, allocation failed", n, 0); }
    if (ok) {
        idle.quarantined = s.quarantined_rungs, idle.layers = s.layers, idle.experts = s.experts;
        hash_.store(hash_table(exec, n, s.layers, s.experts), std::memory_order_release);
        epoch_.store(s.epoch, std::memory_order_release);
        current_.store(&idle, std::memory_order_seq_cst);
    }
    publishing_.store(false);
    return ok;
}
// WHY increment, then re-check: the publisher rewrites a table only after current_ left it
// and its count read zero, so a reader that still sees current_ unchanged is waited for (seq_cst).
const ExecutorView::Table* ExecutorView::enter() const noexcept {
    for (const Table* t = current_.load(); t != nullptr; t = current_.load()) {
        t->readers.fetch_add(1);
        if (current_.load() == t) return t;
        leave(t);
    }
    return nullptr;
}

bool ExecutorView::routes_to(int64_t layer, int64_t expert, uint8_t rung) const noexcept {
    const Table* t = enter();
    if (t == nullptr) return false;
    const bool hit = layer >= 0 && expert >= 0 && layer < static_cast<int64_t>(t->layers) &&
                     expert < static_cast<int64_t>(t->experts) && rung_clear(t->quarantined, rung) &&
                     t->exec[static_cast<size_t>(layer) * t->experts + static_cast<size_t>(expert)] == rung;
    leave(t);
    return hit;
}

bool ExecutorView::rung_open(uint8_t rung) const noexcept {
    const Table* t = enter();
    const bool open = t != nullptr && rung_clear(t->quarantined, rung);
    if (t) leave(t);
    return open;
}
}  // namespace seabraid::strata_adapter
