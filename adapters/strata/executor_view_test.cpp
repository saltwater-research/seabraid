// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// ExecutorView fixture: publish/read, refusals, quarantine, epoch order, hash, and a
// reader racing the publisher (build with -fsanitize=thread for the memory-safety claim).
#include "executor_view.hpp"

#include "diag_capture.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n", __LINE__, #expression); \
        std::abort(); \
    } \
} while (false)

namespace {

using seabraid::strata_adapter::ExecutorView;
namespace capture = seabraid::strata_adapter::capture;

seabraid_placement_snapshot_v1 snap(uint64_t epoch, uint32_t layers, uint32_t experts, uint64_t mask = 0) {
    seabraid_placement_snapshot_v1 s{};
    s.abi_version = SEABRAID_ABI_VERSION;
    s.struct_size = sizeof s;
    s.epoch = epoch;
    s.quarantined_rungs = mask;
    s.layers = layers;
    s.experts = experts;
    s.fallback_rung = 2;
    return s;
}

void out_of_range_and_epoch(const ExecutorView& view) {
    CHECK(!view.routes_to(-1, 0, 0) && !view.routes_to(0, -1, 0) && !view.routes_to(0, 0, 200));
    CHECK(view.epoch() == 4 && view.table_hash() != 0 && view.rung_open(1) && !view.rung_open(64));
}

void publish_and_read() {
    ExecutorView view;
    CHECK(!view.routes_to(0, 0, 0) && !view.rung_open(0));
    CHECK(view.epoch() == 0 && view.table_hash() == 0);
    const uint8_t t[6] = {0, 1, 2, 1, 0, 2};  // 2 layers x 3 experts
    CHECK(view.publish(t, 6, snap(4, 2, 3)));
    CHECK(view.routes_to(0, 0, 0) && view.routes_to(0, 1, 1) && view.routes_to(0, 2, 2));
    CHECK(view.routes_to(1, 0, 1) && view.routes_to(1, 1, 0) && view.routes_to(1, 2, 2));
    CHECK(!view.routes_to(0, 0, 1) && !view.routes_to(2, 0, 0) && !view.routes_to(0, 3, 2));
    out_of_range_and_epoch(view);
}

void shape_refusal() {
    capture::start();
    ExecutorView view;
    const uint8_t t[6] = {0, 1, 2, 1, 0, 2};
    CHECK(!view.publish(t, 5, snap(1, 2, 3)) && !view.publish(nullptr, 6, snap(1, 2, 3)));
    auto bad_abi = snap(1, 2, 3);
    bad_abi.abi_version = 99;
    CHECK(!view.publish(t, 6, bad_abi) && !view.routes_to(0, 0, 0));
    capture::require("refused malformed publish", 3);
    CHECK(view.publish(t, 6, snap(1, 2, 3)));
    const uint8_t u[4] = {0, 1, 2, 0};
    CHECK(!view.publish(u, 4, snap(2, 2, 2)));  // same epoch rule aside, a new shape is refused
    capture::require("refused shape change", 1);
    const uint8_t huge[6] = {0, 1, 64, 1, 0, 2};
    CHECK(!view.publish(huge, 6, snap(2, 2, 3)) && view.epoch() == 1);
    capture::require("refused rung at cell", 1);
}

void quarantined_rung_falls_back() {
    capture::start();
    ExecutorView view;
    const uint8_t stale[4] = {0, 1, 2, 0};
    CHECK(!view.publish(stale, 4, snap(1, 2, 2, 1U << 1)));  // names a quarantined rung
    capture::require("refused rung at cell", 1);
    const uint8_t resolved[4] = {0, 2, 2, 0};  // Rust resolved rung 1's cells to the fallback
    CHECK(view.publish(resolved, 4, snap(1, 2, 2, 1U << 1)));
    CHECK(!view.rung_open(1) && view.rung_open(0) && view.routes_to(0, 1, 2) && !view.routes_to(0, 1, 1));
    CHECK(capture::count("refused") == 1);  // the accepted publish logged nothing
}

void epoch_is_monotonic() {
    capture::start();
    ExecutorView view;
    const uint8_t a[2] = {0, 1}, b[2] = {1, 0};
    CHECK(view.publish(a, 2, snap(5, 1, 2)) && view.publish(a, 2, snap(5, 1, 2)));  // an idle round repeats
    CHECK(!view.publish(b, 2, snap(4, 1, 2)) && view.epoch() == 5 && view.routes_to(0, 0, 0));
    capture::require("refused epoch going back", 1);
    CHECK(!view.publish(b, 2, snap(5, 1, 2)) && view.routes_to(0, 0, 0));  // other bytes, same epoch
    capture::require("refused same epoch, different table", 1);
    CHECK(view.publish(b, 2, snap(6, 1, 2)) && view.epoch() == 6 && view.routes_to(0, 0, 1));
}

void hash_is_stable() {
    ExecutorView one, two;
    const uint8_t a[4] = {0, 1, 1, 2}, b[4] = {0, 1, 2, 2};
    CHECK(one.publish(a, 4, snap(1, 2, 2)) && two.publish(a, 4, snap(9, 2, 2)));
    const uint64_t h = one.table_hash();
    CHECK(h == two.table_hash());  // epoch excluded
    CHECK(one.publish(a, 4, snap(2, 2, 2)) && one.table_hash() == h);
    CHECK(one.publish(b, 4, snap(3, 2, 2)) && one.table_hash() != h);
    CHECK(two.publish(a, 4, snap(10, 4, 1)) == false && two.table_hash() == h);  // shape is hashed too
}

void reader_races_publisher() {
    ExecutorView view;
    std::vector<uint8_t> zeros(4096, 0), ones(4096, 1);
    CHECK(view.publish(zeros.data(), zeros.size(), snap(1, 64, 64)));
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> hits0{0}, hits1{0};
    auto reader = [&] {
        uint64_t last = 0;
        while (!stop.load()) {
            for (int64_t cell = 0; cell < 4096; cell += 97) {
                hits0 += view.routes_to(cell / 64, cell % 64, 0);
                hits1 += view.routes_to(cell / 64, cell % 64, 1);
                CHECK(view.rung_open(2));
            }
            const uint64_t epoch = view.epoch();
            CHECK(epoch >= last);
            last = epoch;
        }
    };
    std::thread readers[3] = {std::thread(reader), std::thread(reader), std::thread(reader)};
    for (uint64_t epoch = 2; epoch < 4000; ++epoch)
        CHECK(view.publish((epoch % 2 ? ones : zeros).data(), 4096, snap(epoch, 64, 64)));
    stop = true;
    for (auto& r : readers) r.join();
    CHECK(hits0 > 0 && hits1 > 0);  // readers saw both tables while the writer flipped them
}

// A second publisher arrives while the first is busy on a 16M-cell table; it passes n = 0, so
// without the guard it would be refused as "malformed" instead, and the first one must be unharmed.
void second_publisher_is_refused() {
    capture::start();
    ExecutorView view;
    std::vector<uint8_t> big(4096u * 4096u, 1);
    std::atomic<bool> done{false};
    std::thread first([&] {
        // The intruder can win the flag between two tries, so the legitimate publisher retries.
        for (uint64_t epoch = 1; epoch <= 3; ++epoch)
            while (!view.publish(big.data(), big.size(), snap(epoch, 4096, 4096))) std::this_thread::yield();
        done = true;
    });
    const uint8_t one = 0;
    while (!done.load()) view.publish(&one, 0, snap(1, 4096, 4096));
    first.join();
    CHECK(capture::count("refused concurrent publish") > 0);
    CHECK(view.epoch() == 3 && view.routes_to(4095, 4095, 1));
}

}  // namespace

int main() {
    publish_and_read();
    shape_refusal();
    quarantined_rung_falls_back();
    epoch_is_monotonic();
    hash_is_stable();
    reader_races_publisher();
    second_publisher_is_refused();
    std::puts("executor_view_test: ok");
    return 0;
}
