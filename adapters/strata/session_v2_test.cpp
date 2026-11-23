// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// ABI v2 through the Strata provider: a stale placement epoch is refused at the ABI, and
// COMPACT_ROWS changes only how many rows cross the wire, never the output.
// OWNER: this file. The in-process peer computes a deterministic per-row function, so the
// comparison is bit-exact on CPU; it tests the transport and the row selection, not GPU math.
#include "static_qwen_peers.hpp"

#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kBatchTokens = 4;                       // rows 0 and 2 carry no pick, rows 1 and 3 do
constexpr size_t kHidden = kGeometry.n_embd;
constexpr size_t kTopK = kGeometry.n_expert_used;

struct Fixture {
    std::vector<float> x = std::vector<float>(kBatchTokens * kHidden);
    std::vector<int32_t> ids = std::vector<int32_t>(kBatchTokens * kTopK, SEABRAID_SKIP_EXPERT_ID);
    std::vector<float> w = std::vector<float>(kBatchTokens * kTopK, 0.0f);
    std::vector<float> out = std::vector<float>(kBatchTokens * kHidden, -7.0f);

    Fixture() {
        for (size_t i = 0; i < x.size(); ++i) x[i] = 1.0f + 0.5f * float(i / kHidden) + 0.001f * float(i % kHidden);
        for (uint32_t t : {1u, 3u})
            for (size_t k = 0; k < kTopK; ++k) { ids[t * kTopK + k] = int32_t(k); w[t * kTopK + k] = 0.25f * float(k + 1); }
    }
    void all_skip() { std::fill(ids.begin(), ids.end(), SEABRAID_SKIP_EXPERT_ID); std::fill(w.begin(), w.end(), 0.0f); }

    seabraid_batch_v2 batch(uint64_t epoch) const {
        seabraid_batch_v2 b{};
        b.abi_version = SEABRAID_BATCH_V2_VERSION;
        b.struct_size = sizeof b;
        b.operation_id = kLayer;
        b.placement_epoch = epoch;
        b.tokens = kBatchTokens;
        b.activation_format = SEABRAID_FORMAT_ROUTED_F32;
        b.activations = x.data(); b.activation_count = x.size();
        b.expert_ids = ids.data(); b.expert_id_count = ids.size();
        b.weights = w.data(); b.weight_count = w.size();
        return b;
    }
};

// The helper's side of one request: out[row] = sum(weight * (id + 1)) * x[row], zero for skip picks.
// Returns the n_tok it saw, so a test can tell how many rows crossed the wire.
uint32_t serve_one(seabraid::Channel& helper, uint64_t sequence) {
    auto* h = helper.header();
    const auto published = [&] { return h->req_seq.load(std::memory_order_acquire) == sequence; };
    check(seabraid::Channel::wait(published, 2), "provider did not publish");
    const uint32_t rows = h->n_tok;
    for (uint32_t t = 0; t < rows; ++t) {
        float scale = 0.0f;
        for (size_t k = 0; k < kTopK; ++k) {
            const int32_t id = helper.ids()[t * kTopK + k];
            if (id != SEABRAID_SKIP_EXPERT_ID) scale += helper.weights()[t * kTopK + k] * float(id + 1);
        }
        for (size_t j = 0; j < kHidden; ++j) helper.out()[t * kHidden + j] = scale * helper.x()[t * kHidden + j];
    }
    h->out_rows = int32_t(rows);
    h->ok = 1;
    h->resp_seq.store(sequence, std::memory_order_release);
    return rows;
}

struct Session {
    Peers peers;
    std::unique_ptr<StaticQwenProvider> adapter = peers.bind();
    seabraid_config_v1 config = adapter->config();
    seabraid_session* handle = nullptr;
    explicit Session(bool compact) {
        check(bool(adapter), "bind the channel");
        const auto provider = adapter->session_provider(config, compact);
        check(seabraid_session_open(&provider, &handle) == SEABRAID_OK, "open a v3 session");
    }
    ~Session() { seabraid_close(handle); }
    // Submits, serves one helper request when `expect_wire`, collects. Returns the rows the helper saw (0 if none).
    uint32_t run(Fixture& f, uint64_t epoch, uint64_t sequence, bool expect_wire) {
        uint32_t seen = 0;
        std::thread helper([&] { if (expect_wire) seen = serve_one(peers.helper, sequence); });
        uint64_t ticket = 0;
        const auto b = f.batch(epoch);
        check(seabraid_session_submit(handle, &b, &ticket) == SEABRAID_OK, "submit accepted");
        helper.join();
        check(seabraid_collect(handle, ticket, 2000, f.out.data(), f.out.size()) == SEABRAID_OK, "collect");
        return seen;
    }
};

void stale_epoch_is_refused_at_the_abi() {
    Session s(false);
    Fixture f;
    s.run(f, 5, 1, true);
    uint64_t sequence = s.peers.helper.header()->req_seq.load();
    uint64_t ticket = 77;
    auto stale = f.batch(3);
    check(seabraid_session_submit(s.handle, &stale, &ticket) == SEABRAID_BUSY_OR_STATE, "a lower epoch is stale");
    check(ticket == 77, "a refused submit leaves the ticket untouched");
    check(s.peers.helper.header()->req_seq.load() == sequence, "a refused submit publishes nothing on the wire");
    auto unversioned = f.batch(0);
    check(seabraid_session_submit(s.handle, &unversioned, &ticket) == SEABRAID_INVALID,
          "epoch 0 after a nonzero epoch is invalid");
    s.run(f, 6, 2, true);   // the session still serves a newer epoch after two refusals
}

void compact_rows_equals_full_rows() {
    Session full(false), compact(true);
    Fixture a, b;
    check(full.run(a, 0, 1, true) == kBatchTokens, "without COMPACT_ROWS every row crosses the wire");
    check(compact.run(b, 0, 1, true) == 2, "with COMPACT_ROWS only the two rows with a pick cross the wire");
    check(std::memcmp(a.out.data(), b.out.data(), a.out.size() * sizeof(float)) == 0,
          "compacted output is bit-identical to the full output");
    check(a.out[1 * kHidden] != 0.0f && a.out[0] == 0.0f && a.out[2 * kHidden] == 0.0f, "fixture rows are as designed");
    seabraid_session_status_v1 status{};
    status.abi_version = SEABRAID_ABI_VERSION;
    status.struct_size = sizeof status;
    check(seabraid_session_status(compact.handle, &status) == SEABRAID_OK, "status");
    check(status.rows_submitted == kBatchTokens && status.rows_shipped == 2, "row counters show the saving");

    b.all_skip();
    const auto before = compact.peers.helper.header()->req_seq.load();
    compact.run(b, 0, 2, false);
    check(compact.peers.helper.header()->req_seq.load() == before, "a batch with no pick never reaches the helper");
    check(std::all_of(b.out.begin(), b.out.end(), [](float v) { return v == 0.0f; }), "its output is all zeros");
}

void default_provider_flags() {
    Peers peers;
    auto adapter = peers.bind();
    const auto config = adapter->config();
    check(adapter->session_provider(config, false).flags == SEABRAID_PROVIDER_PARTIAL_ROUTES, "COMPACT_ROWS is off by default");
    check(adapter->session_provider(config, true).flags ==
              (SEABRAID_PROVIDER_PARTIAL_ROUTES | SEABRAID_PROVIDER_COMPACT_ROWS), "the option adds exactly COMPACT_ROWS");
}

}  // namespace

int main() {
    default_provider_flags();
    stale_epoch_is_refused_at_the_abi();
    compact_rows_equals_full_rows();
    std::puts("Strata ABI v2 session fixture passed");
}
