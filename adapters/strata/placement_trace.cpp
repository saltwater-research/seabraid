// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The trace writer and its window statistics (see placement_trace.hpp for the owner, contract and failure contract).
#include "placement_trace.hpp"

#include "diag.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <mutex>

namespace seabraid::strata_adapter::trace {
namespace {

enum State : int { kUnknown = 0, kOff = 1, kOn = 2 };

// WHY one atomic for the switch: on() runs once per collect, so it must cost one relaxed load when the trace is off.
std::atomic<int> g_state{kUnknown};
std::atomic<uint32_t> g_every{kDefaultEvery};
std::mutex g_lock;       // guards g_file and the line writes: helper runtimes and the engine thread both emit
std::FILE* g_file = nullptr;

using Clock = std::chrono::steady_clock;

unsigned long long wide(uint64_t value) noexcept { return static_cast<unsigned long long>(value); }

// printf-append that cannot throw into a noexcept caller: a failed allocation drops the record, loudly once.
void put(std::string& out, const char* format, ...) __attribute__((format(printf, 2, 3)));
void put(std::string& out, const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    const int n = std::vsnprintf(buffer, sizeof buffer, format, args);
    va_end(args);
    if (n > 0) out.append(buffer, std::min<size_t>(static_cast<size_t>(n), sizeof buffer - 1));
}

void disable(const char* why) noexcept {
    if (g_file) std::fclose(g_file);
    g_file = nullptr;
    // WHY exchange: two threads failing together must still produce ONE warning.
    if (g_state.exchange(kOff) != kOff) diag::logf(diag::kWarn, 0, "seabraid trace disabled: %s", why);
}

void open_file(const char* path) noexcept {
    std::FILE* file = std::fopen(path, "a");
    if (!file) {
        g_state.store(kOn);   // so disable() sees a transition and warns once
        disable("cannot open SEABRAID_TRACE path for append");
        return;
    }
    g_file = file;
    g_state.store(kOn);
}

uint32_t parse_every(const char* text) noexcept {
    if (!text || !*text) return kDefaultEvery;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    // WHY a ceiling: the window keeps every sample, so the knob bounds the memory a long window can take.
    return (*end != '\0' || value == 0) ? kDefaultEvery : static_cast<uint32_t>(std::min<unsigned long>(value, 1u << 20));
}

void init_from_env() noexcept {
    std::lock_guard<std::mutex> lock(g_lock);
    if (g_state.load() != kUnknown) return;
    const char* path = std::getenv("SEABRAID_TRACE");
    g_every.store(parse_every(std::getenv("SEABRAID_TRACE_EVERY")));
    if (!path || !*path) {
        g_state.store(kOff);
        return;
    }
    open_file(path);
}

uint64_t now_us() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count());
}

std::string iso_time() {
    const auto wall = std::chrono::system_clock::now();
    const std::time_t secs = std::chrono::system_clock::to_time_t(wall);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(wall.time_since_epoch()).count() % 1000;
    std::tm parts{};
    gmtime_r(&secs, &parts);
    char text[96];   // roomy on purpose: the compiler must not have to assume a truncated year
    std::snprintf(text, sizeof text, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", parts.tm_year + 1900, parts.tm_mon + 1,
                  parts.tm_mday, parts.tm_hour, parts.tm_min, parts.tm_sec, static_cast<int>(ms));
    return text;
}

// Writes `body` (the keys after "rec") as one line. WHY the catch: building the prefix allocates, and a trace must
// never let bad_alloc reach a serving thread.
void emit(const char* rec, const std::string& body) noexcept {
    try {
        std::string line;
        put(line, "{\"ts_us\":%llu,\"time\":\"%s\",\"rec\":\"%s\"", wide(now_us()), iso_time().c_str(), rec);
        line += body;
        line += "}\n";
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_state.load() != kOn || !g_file) return;
        if (std::fwrite(line.data(), 1, line.size(), g_file) != line.size() || std::fflush(g_file) != 0)
            disable("write to the SEABRAID_TRACE file failed");
    } catch (const std::exception&) {   // allow-error-path: out of memory drops this one record and nothing else
        return;
    }
}

void put_stats(std::string& out, const char* key, const Stats& s) {
    put(out, ",\"%s\":{\"n\":%llu,\"mean\":%.1f,\"p50\":%llu,\"p99\":%llu}", key, wide(s.n), s.mean, wide(s.p50),
        wide(s.p99));
}

void put_issue(std::string& out, const char* key, const seabraid_placement_issue_v1& issue) {
    if (issue.error == 0) return;
    put(out, ",\"%s\":{\"stage\":\"%s\",\"result\":\"%s\",\"rung\":%u}", key, phase_name(issue.phase),
        issue_name(issue.error), static_cast<unsigned>(issue.rung));
}

}  // namespace

bool on() noexcept {
    int state = g_state.load(std::memory_order_relaxed);
    if (state == kUnknown) {
        init_from_env();
        state = g_state.load(std::memory_order_relaxed);
    }
    return state == kOn;
}

uint32_t every() noexcept {
    on();   // reads the environment on first use
    return g_every.load(std::memory_order_relaxed);
}

void configure(const char* path, uint32_t every_collects) noexcept {
    std::lock_guard<std::mutex> lock(g_lock);
    if (g_file) std::fclose(g_file);
    g_file = nullptr;
    g_every.store(every_collects ? std::min<uint32_t>(every_collects, 1u << 20) : kDefaultEvery);
    if (!path || !*path) {
        g_state.store(kOff);
        return;
    }
    open_file(path);
}

void close_trace() noexcept {
    std::lock_guard<std::mutex> lock(g_lock);
    if (g_file) std::fclose(g_file);
    g_file = nullptr;
    g_state.store(kOff);
}

void Sampler::add(uint64_t us) noexcept {
    try {
        values_.push_back(us);
        sum_ += us;
    } catch (const std::exception&) {   // allow-error-path: a sample lost to out-of-memory only thins one window
        return;
    }
}

Stats Sampler::stats() const noexcept {
    Stats s;
    s.n = values_.size();
    if (s.n == 0) return s;
    std::vector<uint64_t> sorted;
    try {
        sorted = values_;
    } catch (const std::exception&) {   // allow-error-path: no statistics for this window rather than a throw
        return Stats{};
    }
    std::sort(sorted.begin(), sorted.end());
    // Nearest rank: the smallest value with at least p of the samples at or below it.
    const auto rank = [&](double p) { return sorted[static_cast<size_t>(std::max(1.0, std::ceil(p * static_cast<double>(s.n)))) - 1]; };
    s.mean = static_cast<double>(sum_) / static_cast<double>(s.n);
    s.p50 = rank(0.50);
    s.p99 = rank(0.99);
    return s;
}

namespace {

// Whether `call` belongs to the pair `o` whose earlier calls have (`wrote`) or have not yet written.
// A pair stages first, then writes: a stage after a write belongs to the next pair.
bool belongs(const CallSample& call, const seabraid_placement_outcome_v1& o, bool wrote) noexcept {
    if (call.layer != o.layer || (call.expert != o.promote && call.expert != o.displace)) return false;
    return call.replace || !wrote;
}

}  // namespace

std::vector<PairTiming> attribute_calls(const std::vector<seabraid_placement_outcome_v1>& outcomes,
                                        const std::vector<CallSample>& calls) {
    std::vector<PairTiming> result(outcomes.size());
    size_t at = 0;
    for (size_t i = 0; i < outcomes.size(); ++i) {
        if (outcomes[i].kind == SEABRAID_PLACEMENT_SKIPPED) continue;
        bool wrote = false;
        for (; at < calls.size() && belongs(calls[at], outcomes[i], wrote); ++at) {
            (calls[at].replace ? result[i].write_us : result[i].stage_us) += calls[at].us;
            result[i].by_ref = result[i].by_ref || calls[at].by_ref;
            wrote = wrote || calls[at].replace;
            result[i].timed = true;
        }
    }
    return result;
}

const char* phase_name(uint32_t phase) noexcept {
    switch (phase) {
    case SEABRAID_PLACEMENT_STAGE: return "stage";
    case SEABRAID_PLACEMENT_REPLACE: return "replace";
    case SEABRAID_PLACEMENT_RESTORE: return "restore";
    default: return "none";
    }
}

const char* issue_name(uint32_t error) noexcept {
    switch (error) {
    case SEABRAID_PLACEMENT_REFUSED: return "refused";
    case SEABRAID_PLACEMENT_FAILED: return "failed";
    case SEABRAID_PLACEMENT_UNACKNOWLEDGED: return "unacknowledged";
    case SEABRAID_PLACEMENT_PANICKED: return "panicked";
    case SEABRAID_PLACEMENT_MALFORMED_BLOB: return "malformed_blob";
    default: return "none";
    }
}

const char* kind_name(uint32_t kind) noexcept {
    switch (kind) {
    case SEABRAID_PLACEMENT_COMMITTED: return "committed";
    case SEABRAID_PLACEMENT_REJECTED: return "rejected";
    case SEABRAID_PLACEMENT_RESTORED: return "restored";
    case SEABRAID_PLACEMENT_QUARANTINED: return "quarantined";
    case SEABRAID_PLACEMENT_SKIPPED: return "skipped";
    default: return "unknown";
    }
}

// WHY "file" only when the hop committed by reference: the bytes then came from the model file, whatever rung sat below.
static const char* src_name(const MoveFields& f, bool committed) noexcept {
    if (committed && f.timing.by_ref) return "file";
    return f.outcome->upper_rung + 1u == f.fallback ? "fallback" : "card";
}

void move(const MoveFields& f) noexcept {
    if (!on() || !f.outcome) return;
    try {
        const auto& o = *f.outcome;
        const unsigned upper = o.upper_rung, lower = upper + 1u;
        const bool committed = o.kind == SEABRAID_PLACEMENT_COMMITTED;
        std::string body;
        put(body, ",\"round\":%llu,\"epoch\":%llu,\"layer\":%u,\"kind\":\"%s\"", wide(f.round), wide(f.epoch), o.layer,
            kind_name(o.kind));
        put(body, ",\"up\":{\"expert\":%u,\"from_rung\":%u,\"to_rung\":%u}", o.promote, lower, upper);
        put(body, ",\"down\":{\"expert\":%u,\"from_rung\":%u,\"to_rung\":%u}", o.displace, upper, lower);
        put(body, ",\"src\":\"%s\",\"dst\":\"%s\",\"bytes\":%zu", src_name(f, committed), lower == f.fallback ? "fallback" : "card", f.bytes);
        put(body, ",\"stage_us\":%llu,\"write_us\":%llu,\"timed\":%s,\"verified\":%d", wide(f.timing.stage_us),
            wide(f.timing.write_us), f.timing.timed ? "true" : "false", committed && f.verified ? 1 : 0);
        put_issue(body, "issue", o.initial);
        put_issue(body, "restore_issue", o.restore);
        emit("move", body);
    } catch (const std::exception&) {   // allow-error-path: the record is dropped, serving is untouched
        return;
    }
}

void round(const RoundFields& f) noexcept {
    if (!on()) return;
    try {
        std::string body;
        put(body, ",\"round\":%llu,\"epoch\":%llu,\"planned\":%u,\"committed\":%u,\"rejected\":%u,\"restored\":%u,"
                  "\"quarantined\":%u,\"skipped\":%u,\"qmask\":%llu,\"round_us\":%llu",
            wide(f.round), wide(f.epoch), f.planned, f.committed, f.rejected, f.restored, f.quarantined, f.skipped,
            wide(f.qmask), wide(f.round_us));
        if (f.fallback_counted)
            put(body, ",\"fallback\":{\"ram_hits\":%lld,\"file_reads\":%lld}", static_cast<long long>(f.fallback_ram_hits),
                static_cast<long long>(f.fallback_file_reads));
        else
            body += ",\"fallback\":null";
        emit("round", body);
    } catch (const std::exception&) {   // allow-error-path: the record is dropped, serving is untouched
        return;
    }
}

void window(const WindowFields& f) noexcept {
    if (!on()) return;
    try {
        std::string body;
        put(body, ",\"rung\":%u,\"layers\":%llu,\"rows_submitted\":%llu,\"rows_shipped\":%llu,\"failures\":%llu,"
                  "\"timeouts\":%llu,\"window_us\":%llu",
            f.rung, wide(f.layers), wide(f.rows_submitted), wide(f.rows_shipped), wide(f.failures), wide(f.timeouts),
            wide(f.window_us));
        put_stats(body, "submit_to_collect_us", f.submit_to_collect);
        put_stats(body, "blocked_us", f.blocked);
        emit("window", body);
    } catch (const std::exception&) {   // allow-error-path: the record is dropped, serving is untouched
        return;
    }
}

void helper(uint32_t rung, const char* event, const char* reason, uint64_t after_loss_us) noexcept {
    if (!on()) return;
    try {
        std::string body;
        put(body, ",\"rung\":%u,\"event\":\"%s\",\"reason\":\"", rung, event);
        // Reasons are adapter literals; the escape keeps a stray quote or backslash from breaking the line.
        for (const char* c = reason ? reason : ""; *c; ++c) {
            if (*c == '"' || *c == '\\') body += '\\';
            if (static_cast<unsigned char>(*c) >= 0x20) body += *c;
        }
        put(body, "\",\"after_loss_us\":%llu", wide(after_loss_us));
        emit("helper", body);
    } catch (const std::exception&) {   // allow-error-path: the record is dropped, serving is untouched
        return;
    }
}

}  // namespace seabraid::strata_adapter::trace
