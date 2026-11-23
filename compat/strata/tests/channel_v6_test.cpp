// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_v6_test.cpp - the seabraid channel v6 wire contract for engine-neutral helpers (the spec region and per-token output).  CPU only: shm + headers.
//
//   usage: channel_v6_test tests/channel_v6_golden.txt     (the golden file is written by seabraid_layout_print and shared
//                                                           with tests/install/test_plan.py, so the C++ layout and the
//                                                           installer's mirror of it are pinned to the same numbers)
//
// What each block guards (and what it would look like if it broke):
//   1 header      : a field added to Header on one side only shifts every later field in the other process.
//   2 golden      : Layout::of changed without the installer's channel_bytes() following -> shm sized wrong.
//   3 offsets     : a region reordered/overlapped -> two writers share bytes; the data is wrong, nothing crashes.
//   4 spec        : main's LayerSpec/path table must arrive byte-identical in the helper's separate mapping.
//   5 echo        : out_mode/out_rows are what main trusts before reading `out`; weights ride a region of their own.
//   6 connect     : geometry/version disagreement is a refusal at connect, not shifted fields later.
// House rule: each check was sabotaged once and watched to FAIL (tests/SABOTAGE.md records the runs).
#include "strata/seabraid/channel.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

using namespace seabraid;

static size_t up4k(size_t v) { return (v + 4095) & ~(size_t) 4095; }

struct GoldenRow { int64_t H, window, k, L, E, blob; size_t total; };

static bool read_golden(const char* path, size_t& header_bytes, std::vector<GoldenRow>& rows) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string first;
        if (!(ss >> first)) continue;
        if (first == "header_bytes") { ss >> header_bytes; continue; }
        GoldenRow r{};
        r.H = std::atoll(first.c_str());
        ss >> r.window >> r.k >> r.L >> r.E >> r.blob >> r.total;
        rows.push_back(r);
    }
    return header_bytes != 0 && !rows.empty();
}

// A channel's regions in the documented order, with the bytes each one must hold.
static void check_offsets(int64_t H, int64_t cap, int64_t L, int64_t E, int64_t blob, const char* tag) {
    const Layout l = Layout::of(H, cap, L, E, blob);
    const size_t start[] = {l.plan, l.held, l.ids, l.x, l.out, l.blobs, l.tier_ops, l.weights, l.spec, l.total};
    const size_t need[] = {(size_t) L * E * 8, (size_t) L * E, (size_t) cap * 4, (size_t) cap * H * 4, (size_t) cap * H * 4,
                           (size_t) kBlobBatch * blob, (size_t) kTierMaxOps * sizeof(TierWireOp), (size_t) cap * 4, spec_region_bytes(L)};
    const char* name[] = {"plan", "held", "ids", "x", "out", "blobs", "tier_ops", "weights", "spec"};
    CHECK(start[0] >= sizeof(Header), "first region starts after the header");
    for (int i = 0; i < 9; ++i) {
        char what[160];
        std::snprintf(what, sizeof(what), "%s: region %s starts on a 4 KiB page", tag, name[i]);
        CHECK(start[i] % 4096 == 0, what);
        std::snprintf(what, sizeof(what), "%s: region %s is followed by the next one (documented order), not overlapped", tag, name[i]);
        CHECK(start[i + 1] >= start[i] + need[i], what);   // order AND room: a swap or an overlap breaks one of the two
        std::snprintf(what, sizeof(what), "%s: region %s is packed to its page (no wasted page)", tag, name[i]);
        CHECK(start[i + 1] == start[i] + up4k(need[i]), what);
    }
}

// The unit geometry every round-trip block uses.
static constexpr int64_t H = 64, CAP = 32, NL = 4, NE = 8, BLOB = 256;
static constexpr uint32_t WMT = 4;
static constexpr int64_t kHuge = std::numeric_limits<int64_t>::max();

// ---- 1. header
static void check_header(size_t golden_header) {
    CHECK(sizeof(Header) == golden_header, "sizeof(Header) equals the golden header_bytes (installer's CHANNEL_HEADER_BYTES)");
    CHECK(sizeof(RoleSpec) == 32 && sizeof(LayerSpec) == 112, "RoleSpec 32 B, LayerSpec 112 B");
    CHECK(offsetof(Header, engine) == offsetof(Header, out_mode) + sizeof(int32_t), "v6 fields are appended right after out_mode");
    CHECK(offsetof(Header, spec_layers) == offsetof(Header, engine) + 4 && offsetof(Header, spec_files) == offsetof(Header, engine) + 8,
          "engine, spec_layers, spec_files are consecutive");
    CHECK(kVersion == 11, "kVersion is 11 (v11 = v10 under the SEABRD1 tag; v10 = v9 + Header::helper_start, appended; the Header is 2752 bytes, the shm layout totals are unchanged)");
}

// ---- 2. golden sizes
static void check_golden_sizes(const std::vector<GoldenRow>& golden) {
    for (const GoldenRow& r : golden) {
        char what[160];
        std::snprintf(what, sizeof(what), "Layout::of total H=%lld w=%lld k=%lld L=%lld E=%lld blob=%lld matches the golden file", (long long) r.H,
                      (long long) r.window, (long long) r.k, (long long) r.L, (long long) r.E, (long long) r.blob);
        CHECK(Layout::of(r.H, r.window * r.k, r.L, r.E, r.blob).total == r.total, what);
    }
}

// ---- size arithmetic and pure request bounds (no payload consumed on rejection)
static void check_bounds() {
    const auto huge = kHuge;
    CHECK(Layout::of(H, CAP, NL, NE, -1).total == 0, "negative blob size is invalid");
    CHECK(Layout::of(huge, CAP, NL, NE, BLOB).total == 0, "row size exceeding the addressable mapping is invalid");
    CHECK(Layout::of(H, CAP, NL, huge, BLOB).total == 0, "plan size exceeding the addressable mapping is invalid");
    CHECK(PrefillRegion::of(H, 8, huge).bytes == 0, "prefill size exceeding the addressable mapping is invalid");
    const ChannelLimits limits{H, CAP, NL, NE, BLOB, 0, 0};
    CHECK(limits.valid_decode(NL - 1, WMT, 8, WMT), "decode can exactly fill its last layer and full window");
    CHECK(!limits.valid_decode(NL, WMT, 8, WMT), "decode rejects the first layer beyond the mapping");
    CHECK(!limits.valid_decode(0, WMT, 9, WMT), "decode rejects a product beyond the row capacity");
    CHECK(!limits.valid_decode(0, WMT, huge, WMT), "decode rejects a width outside the row capacity before multiplying");
    CHECK(!limits.valid_decode(0, 1, 0, WMT), "decode rejects a zero width");
}

// A live reconnect is refused, a fresh reader attaches, and a new channel has the documented defaults.
static void check_attach_rules(const std::string& name, Channel& helper, Channel& main_side) {
    std::string denied;
    CHECK(!main_side.connect(name, H, CAP, NL, NE, WMT, denied) &&
          denied.find("already mapped") != std::string::npos, "a live Channel cannot be overwritten by reconnecting");
    Channel another;
    CHECK(another.connect(name, H, CAP, NL, NE, WMT, denied), "a fresh reader can attach to the private channel");
    const Header* hh = helper.header();
    CHECK(hh->version == kVersion && hh->engine == kEngineQwen && hh->spec_layers == 0 && hh->spec_files == 0 && hh->out_mode == kOutPerPick,
          "a fresh channel defaults to Qwen, no spec, per-pick rows");
}

// the accessors agree with Layout::of in both mappings
static void check_accessors(Channel& helper) {
    const Layout lay = Layout::of(H, CAP, NL, NE, BLOB);
    CHECK((char*) helper.weights() - (char*) helper.region() == (ptrdiff_t) lay.weights, "weights() sits at Layout::weights");
    CHECK((char*) helper.layer_spec() - (char*) helper.region() == (ptrdiff_t) lay.spec, "layer_spec() sits at Layout::spec");
    CHECK((char*) helper.out() - (char*) helper.region() == (ptrdiff_t) lay.out, "out() sits at Layout::out");
    CHECK(helper.spec_path(kSpecMaxFiles - 1) + kSpecPathMax <= (char*) helper.region() + lay.total, "the last spec path ends inside the channel");
}

// ---- 4. spec round trip: main writes through ITS mapping, the helper reads through ITS mapping
static void check_spec(Channel& helper, Channel& main_side) {
    Header* hh = helper.header();
    Header* mh = main_side.header();
    std::vector<LayerSpec> want(NL);
    for (int l = 0; l < NL; ++l) {
        LayerSpec s{};
        for (int r = 0; r < kRoles; ++r) {
            s.role[r] = RoleSpec{l % 2, 16 + r + l, (uint64_t) 0x100000000ull * (l + 1) + r, (uint64_t) 1000 * (r + 1) + l, (uint64_t) 900 * (r + 1) + l};
        }
        s.hidden = (int32_t) (5120 + l); s.ffn = 2304 + l; s.swiglu_limit = 7.5f + (float) l; s.flags = l == 2 ? kSpecActQ8K : 0;
        want[(size_t) l] = s;
        std::memcpy(&main_side.layer_spec()[l], &s, sizeof(s));
    }
    std::snprintf(main_side.spec_path(0), kSpecPathMax, "%s", "/models/DeepSeek-V4.1-Flash-00001-of-00004.gguf");
    std::snprintf(main_side.spec_path(1), kSpecPathMax, "%s", "/models/DeepSeek-V4.1-Flash-00002-of-00004.gguf");
    mh->engine = kEngineDsv41; mh->spec_layers = NL; mh->spec_files = 2;
    CHECK(std::memcmp(helper.layer_spec(), want.data(), sizeof(LayerSpec) * NL) == 0, "LayerSpec[] arrives byte-identical in the helper's mapping");
    CHECK(std::strcmp(helper.spec_path(1), "/models/DeepSeek-V4.1-Flash-00002-of-00004.gguf") == 0, "path 1 arrives intact");
    CHECK(std::strcmp(helper.spec_path(0), "/models/DeepSeek-V4.1-Flash-00001-of-00004.gguf") == 0, "path 0 arrives intact, not clobbered by path 1");
    CHECK(hh->engine == kEngineDsv41 && hh->spec_layers == NL && hh->spec_files == 2, "engine/spec counts arrive");
    CHECK(helper.layer_spec()[2].flags == kSpecActQ8K && helper.layer_spec()[3].swiglu_limit == 10.5f, "per-layer flags and swiglu_limit arrive");
}

// ---- 5. echo + weights + out rows
static void check_echo(Channel& helper, Channel& main_side) {
    Header* hh = helper.header();
    const Header* mh = main_side.header();
    for (int i = 0; i < CAP; ++i) main_side.weights()[i] = 0.25f * (float) i;
    CHECK(helper.weights()[7] == 1.75f && helper.weights()[CAP - 1] == 0.25f * (CAP - 1), "weights[] written by main is read by the helper");
    hh->out_mode = kOutPerToken;
    hh->out_rows = 3;
    for (int t = 0; t < 3; ++t) for (int d = 0; d < H; ++d) helper.out()[t * H + d] = (float) (t * 1000 + d);
    CHECK(mh->out_mode == kOutPerToken && mh->out_rows == 3, "out_mode and out_rows echo reaches main");
    CHECK(main_side.out()[2 * H + 5] == 2005.0f, "main reads token 2's row where the helper wrote it");
    CHECK(main_side.out()[H - 1] == (float) (H - 1) && main_side.out()[H] == 1000.0f, "rows are H floats apart, no stride surprise");
}

// ---- 6. connect refusals
static void check_refusals(const std::string& name, Channel& helper) {
    Header* hh = helper.header();
    std::string err;
    Channel other;
    CHECK(!other.create(name, kHuge, CAP, NL, NE, WMT, BLOB, err), "invalid geometry fails before replacing an existing mapping");
    CHECK(!other.connect(name, H + 1, CAP, NL, NE, WMT, err) && err.find("geometry mismatch") != std::string::npos, "connect refuses a different hidden");
    CHECK(!other.connect(name, H, CAP, NL, NE, WMT + 1, err) && err.find("geometry mismatch") != std::string::npos, "connect refuses a different window");
    const uint32_t ver = hh->version;
    hh->version = 4;
    CHECK(!other.connect(name, H, CAP, NL, NE, WMT, err) && err.find("magic/version") != std::string::npos, "connect refuses a v4 channel");
    hh->version = ver;
    CHECK(other.connect(name, H, CAP, NL, NE, WMT, err), ("connect works again once the version is right: " + err).c_str());
}

int main(int argc, char** argv) {
    size_t golden_header = 0;
    std::vector<GoldenRow> golden;
    if (argc < 2 || !read_golden(argv[1], golden_header, golden)) {
        std::printf("FAIL: cannot read the golden file (argument 1)\n");
        return 2;
    }
    check_header(golden_header);
    check_golden_sizes(golden);
    check_offsets(64, 32, 4, 8, 256, "unit");   // ---- 3. offsets (the unit geometry)
    check_bounds();

    // ---- channel round trips
    const std::string name = "/seabraid_v6_test_" + std::to_string((long) getpid());
    std::string err;
    Channel helper, main_side;
    CHECK(helper.create(name, H, CAP, NL, NE, WMT, BLOB, err), ("create: " + err).c_str());
    CHECK(main_side.connect(name, H, CAP, NL, NE, WMT, err), ("connect: " + err).c_str());
    if (!helper.valid() || !main_side.valid()) { std::printf("channel_v6_test: cannot continue without a channel\n"); return 1; }
    check_attach_rules(name, helper, main_side);
    check_accessors(helper);
    check_spec(helper, main_side);
    check_echo(helper, main_side);
    check_refusals(name, helper);

    if (g_fail) { std::printf("channel_v6_test: %d FAILED\n", g_fail); return 1; }
    std::printf("channel_v6_test: all checks passed\n");
    return 0;
}
