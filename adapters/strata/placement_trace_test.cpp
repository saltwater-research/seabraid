// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The trace writer's fixture: the environment switch, the line format of every record kind, fail-open-loud, the window
// statistics and the call-to-outcome attribution. CHECKER for placement_trace.hpp's contract.
#include "placement_trace.hpp"

#include "diag_capture.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n%s", __LINE__, #expression, capture::dump().c_str()); \
        std::abort(); \
    } \
} while (false)

namespace capture = seabraid::strata_adapter::capture;
namespace trace = seabraid::strata_adapter::trace;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

size_t lines(const std::string& text) {
    size_t n = 0;
    for (char c : text) n += c == '\n';
    return n;
}

bool has(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

std::string temp_path() {
    char path[] = "/tmp/seabraid_trace_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    close(fd);
    return path;
}

seabraid_placement_outcome_v1 outcome(uint32_t kind, uint8_t upper) {
    seabraid_placement_outcome_v1 o{};
    o.kind = kind;
    o.layer = 7;
    o.promote = 11;
    o.displace = 22;
    o.upper_rung = upper;
    return o;
}

// Must run first: on() reads the environment only once per process.
void environment_switch_opens_the_file_and_sets_the_cadence() {
    const std::string path = temp_path();
    setenv("SEABRAID_TRACE", path.c_str(), 1);
    setenv("SEABRAID_TRACE_EVERY", "7", 1);
    CHECK(trace::on() && trace::every() == 7);
    trace::helper(2, "up", "env test", 0);
    const std::string text = slurp(path);
    CHECK(lines(text) == 1 && has(text, "\"rec\":\"helper\"") && has(text, "\"rung\":2") && has(text, "\"event\":\"up\""));
    CHECK(has(text, "\"ts_us\":") && has(text, "\"time\":\"20") && has(text, "Z\""));
    trace::close_trace();
    unlink(path.c_str());
    unsetenv("SEABRAID_TRACE");
    unsetenv("SEABRAID_TRACE_EVERY");
}

void trace_off_writes_no_file_and_no_record() {
    const std::string path = temp_path();
    unlink(path.c_str());
    trace::configure(nullptr, 0);
    CHECK(!trace::on());
    auto o = outcome(SEABRAID_PLACEMENT_COMMITTED, 0);
    trace::MoveFields move;
    move.outcome = &o;
    trace::move(move);
    trace::round({});
    trace::window({});
    trace::helper(1, "up", "off", 0);
    CHECK(access(path.c_str(), F_OK) != 0);   // nothing was created
    trace::configure("", 0);
    CHECK(!trace::on() && trace::every() == trace::kDefaultEvery);
}

void every_record_kind_has_its_fields() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    CHECK(trace::on());
    auto committed = outcome(SEABRAID_PLACEMENT_COMMITTED, 1);
    committed.committed_epoch = 9;
    trace::MoveFields move;
    move.round = 3;
    move.epoch = 9;
    move.outcome = &committed;
    move.fallback = 2;   // upper 1 -> lower 2 is the fallback
    move.bytes = 4096;
    move.timing = {120, 340, true};
    trace::move(move);
    trace::RoundFields round;
    round.round = 3;
    round.epoch = 9;
    round.planned = 2;
    round.committed = 1;
    round.rejected = 1;
    round.round_us = 5000;
    round.fallback_counted = true;
    round.fallback_ram_hits = 5;
    round.fallback_file_reads = 2;
    trace::round(round);
    trace::WindowFields window;
    window.rung = 1;
    window.layers = 4096;
    window.rows_submitted = 8192;
    window.rows_shipped = 4000;
    window.submit_to_collect = {4096, 90, 400, 250.5};
    window.blocked = {4096, 10, 30, 12.5};
    trace::window(window);
    trace::helper(1, "closed", "a \"quoted\" reason", 0);
    trace::helper(1, "served_after_loss", "next window", 1234);
    const std::string text = slurp(path);
    CHECK(lines(text) == 5);
    for (const char* field : {"\"rec\":\"move\",\"round\":3,\"epoch\":9,\"layer\":7,\"kind\":\"committed\"",
                              "\"up\":{\"expert\":11,\"from_rung\":2,\"to_rung\":1}",
                              "\"down\":{\"expert\":22,\"from_rung\":1,\"to_rung\":2}",
                              "\"src\":\"fallback\",\"dst\":\"fallback\",\"bytes\":4096",
                              "\"stage_us\":120,\"write_us\":340,\"timed\":true,\"verified\":1",
                              "\"rec\":\"round\",\"round\":3,\"epoch\":9,\"planned\":2,\"committed\":1,\"rejected\":1",
                              "\"round_us\":5000,\"fallback\":{\"ram_hits\":5,\"file_reads\":2}",
                              "\"rec\":\"window\",\"rung\":1,\"layers\":4096,\"rows_submitted\":8192,\"rows_shipped\":4000",
                              "\"submit_to_collect_us\":{\"n\":4096,\"mean\":250.5,\"p50\":90,\"p99\":400}",
                              "\"blocked_us\":{\"n\":4096,\"mean\":12.5,\"p50\":10,\"p99\":30}",
                              "\"reason\":\"a \\\"quoted\\\" reason\"", "\"after_loss_us\":1234"})
        CHECK(has(text, field));
    trace::RoundFields bare;
    trace::round(bare);
    CHECK(has(slurp(path), "\"fallback\":null"));   // no host counters: null, never a made-up zero
    trace::close_trace();
    unlink(path.c_str());
}

void every_outcome_kind_names_its_issue() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    auto rejected = outcome(SEABRAID_PLACEMENT_REJECTED, 0);
    rejected.initial = {SEABRAID_PLACEMENT_STAGE, SEABRAID_PLACEMENT_FAILED, 1};
    auto restored = outcome(SEABRAID_PLACEMENT_RESTORED, 0);
    restored.initial = {SEABRAID_PLACEMENT_REPLACE, SEABRAID_PLACEMENT_REFUSED, 0};
    restored.restore = {SEABRAID_PLACEMENT_RESTORE, SEABRAID_PLACEMENT_UNACKNOWLEDGED, 1};
    auto quarantined = outcome(SEABRAID_PLACEMENT_QUARANTINED, 0);
    quarantined.initial = {SEABRAID_PLACEMENT_REPLACE, SEABRAID_PLACEMENT_UNACKNOWLEDGED, 1};
    auto skipped = outcome(SEABRAID_PLACEMENT_SKIPPED, 0);
    for (const auto* o : {&rejected, &restored, &quarantined, &skipped}) {
        trace::MoveFields move;
        move.outcome = o;
        move.round = 1;
        trace::move(move);
    }
    const std::string text = slurp(path);
    CHECK(lines(text) == 4);
    for (const char* field : {"\"kind\":\"rejected\"", "\"issue\":{\"stage\":\"stage\",\"result\":\"failed\",\"rung\":1}",
                              "\"kind\":\"restored\"", "\"issue\":{\"stage\":\"replace\",\"result\":\"refused\",\"rung\":0}",
                              "\"restore_issue\":{\"stage\":\"restore\",\"result\":\"unacknowledged\",\"rung\":1}",
                              "\"kind\":\"quarantined\"", "\"kind\":\"skipped\"", "\"verified\":0"})
        CHECK(has(text, field));
    // A skipped outcome made no call and has no issue: it must not invent one.
    const size_t last = text.rfind("\"kind\":\"skipped\"");
    CHECK(text.find("\"issue\"", last) == std::string::npos);
    trace::close_trace();
    unlink(path.c_str());
}

void an_unwritable_path_warns_once_and_disables() {
    capture::start();
    trace::configure("/nonexistent_seabraid_dir/trace.jsonl", 0);
    CHECK(!trace::on());
    auto o = outcome(SEABRAID_PLACEMENT_COMMITTED, 0);
    trace::MoveFields move;
    move.outcome = &o;
    for (int i = 0; i < 5; ++i) trace::move(move);   // serving carries on: no throw, no further record
    capture::require("seabraid trace disabled", 1);
    CHECK(capture::count("seabraid trace disabled", 0) == 1);
}

void a_failing_write_warns_once_and_disables() {
    capture::start();
    trace::configure("/dev/full", 0);   // opens, and every write fails with ENOSPC
    CHECK(trace::on());
    auto o = outcome(SEABRAID_PLACEMENT_COMMITTED, 0);
    trace::MoveFields move;
    move.outcome = &o;
    for (int i = 0; i < 5; ++i) trace::move(move);
    CHECK(!trace::on());
    capture::require("seabraid trace disabled: write to the SEABRAID_TRACE file failed", 1);
}

void window_statistics_of_a_hundred_samples() {
    trace::Sampler sampler;
    CHECK(sampler.stats().n == 0 && sampler.stats().p99 == 0);
    for (uint64_t v = 100; v >= 1; --v) sampler.add(v);   // unsorted on purpose
    const trace::Stats s = sampler.stats();
    CHECK(s.n == 100 && s.p50 == 50 && s.p99 == 99 && s.mean == 50.5);
}

void window_statistics_of_tiny_windows() {
    trace::Sampler sampler;
    for (uint64_t v : {40, 10, 30, 20}) sampler.add(v);
    trace::Stats s = sampler.stats();
    CHECK(s.n == 4 && s.p50 == 20 && s.p99 == 40 && s.mean == 25.0);
    sampler.clear();
    sampler.add(7);
    s = sampler.stats();
    CHECK(s.p50 == 7 && s.p99 == 7);
}

// One outlier in a thousand sits above the 99th percentile's rank, so it must not move p99 (nor p50).
void one_outlier_does_not_move_p99() {
    trace::Sampler sampler;
    for (uint64_t v = 1; v <= 1000; ++v) sampler.add(v == 1000 ? 100000 : 10);
    const trace::Stats s = sampler.stats();
    CHECK(s.p50 == 10 && s.p99 == 10 && s.n == 1000);
}

void calls_are_attributed_to_the_outcome_they_belong_to() {
    using seabraid::strata_adapter::trace::CallSample;
    const auto a = outcome(SEABRAID_PLACEMENT_COMMITTED, 0);   // layer 7, experts 11/22
    auto b = outcome(SEABRAID_PLACEMENT_REJECTED, 0);
    b.promote = 33;
    b.displace = 44;
    const auto skipped = outcome(SEABRAID_PLACEMENT_SKIPPED, 0);
    const std::vector<CallSample> calls{
        {false, 1, 7, 11, 10}, {false, 0, 7, 22, 20}, {true, 0, 7, 11, 100}, {true, 1, 7, 22, 200},   // pair a
        {false, 1, 7, 33, 5}, {false, 0, 7, 44, 6},                                                  // pair b: stage only
    };
    const auto t = trace::attribute_calls({a, skipped, b}, calls);
    CHECK(t.size() == 3);
    CHECK(t[0].timed && t[0].stage_us == 30 && t[0].write_us == 300);
    CHECK(!t[1].timed);
    CHECK(t[2].timed && t[2].stage_us == 11 && t[2].write_us == 0);
    // Two hops of one expert in a row (a climb): the second stage after a write starts the second hop's group.
    const std::vector<CallSample> climb{{false, 2, 7, 11, 1}, {false, 1, 7, 22, 2}, {true, 1, 7, 11, 10}, {true, 2, 7, 22, 20},
                                        {false, 1, 7, 11, 100}, {false, 0, 7, 22, 200}, {true, 0, 7, 11, 1000}, {true, 1, 7, 22, 2000}};
    const auto c = trace::attribute_calls({a, a}, climb);
    CHECK(c[0].stage_us == 3 && c[0].write_us == 30 && c[1].stage_us == 300 && c[1].write_us == 3000);
    CHECK(!trace::attribute_calls({a}, {}).front().timed);   // no calls recorded (trace was off): untimed, not zero
}

// A hop whose promoted expert the helper loaded from the model file (an ACKed replace_from_fallback) is recorded src "file", and
// only when it committed: a rolled-back hop moved nobody, so its record keeps naming the rung below.
void by_reference_hops_record_src_file() {
    using seabraid::strata_adapter::trace::CallSample;
    const auto hop = outcome(SEABRAID_PLACEMENT_COMMITTED, 1);   // layer 7, experts 11 up / 22 down
    // stage(down on the upper rung), the by-reference write (a write that ACKed), replace(lower): the promoted expert 11 is never staged.
    const std::vector<CallSample> by_ref{{false, 1, 7, 22, 10}, {true, 1, 7, 11, 300, true}, {true, 2, 7, 22, 40}};
    const auto timing = trace::attribute_calls({hop}, by_ref).front();
    CHECK(timing.timed && timing.by_ref && timing.stage_us == 10 && timing.write_us == 340);
    // A REFUSED ask is recorded as staging work and is not by reference: the byte path followed.
    const std::vector<CallSample> refused{{false, 1, 7, 22, 10}, {false, 1, 7, 11, 5, false}, {false, 2, 7, 11, 6}, {true, 1, 7, 11, 50}, {true, 2, 7, 22, 40}};
    const auto bytes = trace::attribute_calls({hop}, refused).front();
    CHECK(bytes.timed && !bytes.by_ref && bytes.stage_us == 21 && bytes.write_us == 90);

    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    trace::MoveFields move;
    move.round = 1; move.epoch = 2; move.outcome = &hop; move.fallback = 2; move.bytes = 4096;
    move.timing = timing;
    trace::move(move);                                           // committed, by reference
    move.timing = bytes;
    trace::move(move);                                           // committed, by bytes
    auto rolled_back = outcome(SEABRAID_PLACEMENT_RESTORED, 1);
    move.outcome = &rolled_back;
    move.timing = timing;
    trace::move(move);                                           // not committed: src stays the rung below
    const std::string text = slurp(path);
    CHECK(lines(text) == 3);
    CHECK(has(text, "\"kind\":\"committed\",\"up\":{\"expert\":11,\"from_rung\":2,\"to_rung\":1},\"down\":{\"expert\":22,\"from_rung\":1,\"to_rung\":2},\"src\":\"file\",\"dst\":\"fallback\""));
    CHECK(has(text, "\"src\":\"fallback\",\"dst\":\"fallback\""));
    CHECK(text.find("\"src\":\"file\"") == text.rfind("\"src\":\"file\""));   // exactly one record says file
    trace::close_trace();
    unlink(path.c_str());
}

}  // namespace

int main() {
    environment_switch_opens_the_file_and_sets_the_cadence();
    trace_off_writes_no_file_and_no_record();
    every_record_kind_has_its_fields();
    every_outcome_kind_names_its_issue();
    an_unwritable_path_warns_once_and_disables();
    a_failing_write_warns_once_and_disables();
    window_statistics_of_a_hundred_samples();
    window_statistics_of_tiny_windows();
    one_outlier_does_not_move_p99();
    calls_are_attributed_to_the_outcome_they_belong_to();
    by_reference_hops_record_src_file();
    std::puts("placement_trace_test: ok");
    return 0;
}
