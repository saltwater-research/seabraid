// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// expert-sums.v1 fixture (no hashing, no GPU): round trip, a pinned byte layout, and every refusal of parse/serialize/freshness.
// Guards: a manifest accepted with a count the length does not back would read past its buffer; a stale manifest accepted as
// fresh would let a helper "verify" bytes against sums of a pack that has since changed.
#include "test_support.hpp"

#include "strata/seabraid/expert_sums.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {
using seabraid::strata_adapter::testing::check;
namespace sums = seabraid::sums;

sums::Manifest sample() {
    sums::Manifest m;
    m.layers = 3; m.experts = 4;
    m.files = {{1000, 111}, {2000, 222}};
    m.blob_bytes = {96, 80, 64};
    for (uint64_t i = 0; i < 12; ++i) m.sums.push_back(0x0102030405060708ull * (i + 1));
    return m;
}

std::string bytes_of(const sums::Manifest& m) {
    std::string out, err;
    check(sums::serialize(m, out, err), ("serialize: " + err).c_str());
    return out;
}

bool refused(std::string b, const char* needle) {
    sums::Manifest out;
    std::string err;
    const bool ok = sums::parse((const uint8_t*) b.data(), b.size(), out, err);
    return !ok && out.sums.empty() && out.files.empty() && err.find(needle) != std::string::npos;
}

void set32(std::string& b, size_t at, uint32_t v) { for (int i = 0; i < 4; ++i) b[at + i] = (char) (v >> (8 * i)); }

// True when every (offset, byte) pair holds: one loop instead of a chain of && the complexity gate would count.
bool bytes_are(const std::string& b, std::initializer_list<std::pair<size_t, int>> want) {
    for (const auto& w : want)
        if ((uint8_t) b[w.first] != w.second) return false;
    return true;
}

void layout_bytes(const std::string& b) {
    check(b.size() == 32 + 2 * 16 + 3 * 4 + 12 * 8, "the manifest is header + files + blob sizes + sums, byte for byte");
    check(b.compare(0, 8, std::string("SBSUMS1\0", 8)) == 0, "the magic is SBSUMS1 and a NUL");
    check(bytes_are(b, {{8, 1}, {12, 1}, {16, 3}, {20, 4}, {24, 2}, {28, 0}}), "header fields sit at their documented offsets, little-endian");
    check(bytes_are(b, {{32, 0xE8}, {33, 0x03}, {40, 111}, {48, 0xD0}, {49, 0x07}, {56, 222}}), "the file entries are {size, mtime_ns} from offset 32");
    check(bytes_are(b, {{64, 96}, {68, 80}, {72, 64}}), "layer blob sizes follow the file entries");
    check(bytes_are(b, {{76, 0x08}, {83, 0x01}}), "the sums follow, least significant byte first");
}

void round_trip(const sums::Manifest& m, const std::string& b) {
    sums::Manifest back;
    std::string err;
    check(sums::parse((const uint8_t*) b.data(), b.size(), back, err), ("round trip parses: " + err).c_str());
    uint64_t s = 0;
    check(back.layers == 3 && back.experts == 4 && back.files.size() == 2, "the round trip returns the geometry and the files");
    check(back.files[1].mtime_ns == 222 && back.blob_bytes == m.blob_bytes && back.sums == m.sums, "the round trip returns the stats, blob sizes and sums");
    check(back.sum_of(2, 3, s) && s == m.sums[11], "sum_of indexes layer-major (last cell)");
    check(back.sum_of(0, 0, s) && s == m.sums[0], "sum_of indexes layer-major (first cell)");
    check(!back.sum_of(3, 0, s) && !back.sum_of(0, 4, s), "sum_of refuses a layer or expert past the geometry");
    check(!back.sum_of(-1, 0, s) && !back.sum_of(0, -1, s), "sum_of refuses a negative index");
}

void parse_refusals(const std::string& b) {
    check(refused(b.substr(0, 31), "shorter than"), "fewer bytes than the header is refused");
    check(refused(b.substr(0, b.size() - 1), "implies"), "a truncated manifest is refused");
    check(refused(b + "x", "implies"), "trailing bytes are refused");
    std::string t = b; t[0] = 'X';           check(refused(t, "magic"), "a wrong magic is refused");
    t = b; set32(t, 8, 2);                   check(refused(t, "format version 2"), "another format version is refused, naming it");
    t = b; set32(t, 12, 2);                  check(refused(t, "sum kind"), "another sum kind is refused");
    t = b; set32(t, 28, 1);                  check(refused(t, "reserved"), "a non-zero reserved field is refused");
    t = b; set32(t, 16, 0);                  check(refused(t, "out of range"), "zero layers is refused");
    t = b; set32(t, 20, 0);                  check(refused(t, "out of range"), "zero experts is refused");
    t = b; set32(t, 24, 0);                  check(refused(t, "out of range"), "zero files is refused");
    t = b; set32(t, 24, 65);                 check(refused(t, "out of range"), "more files than the table holds is refused");
    t = b; set32(t, 16, 1025);               check(refused(t, "out of range"), "layers above 1024 are refused");
    t = b; set32(t, 20, 513);                check(refused(t, "out of range"), "experts above 512 are refused");
    t = b; set32(t, 16, 0xFFFFFFFFu); set32(t, 20, 0xFFFFFFFFu); check(refused(t, "out of range"), "counts whose product would overflow are refused before any multiply is trusted");
    t = b; set32(t, 64, 0);                  check(refused(t, "zero blob size"), "a zero blob size is refused");
}

void serialize_refusals(const sums::Manifest& m) {
    std::string out, err;
    sums::Manifest bad = m;
    bad.sums.pop_back();                     check(!sums::serialize(bad, out, err), "serialize refuses a short sums table");
    bad = m; bad.sum_kind = 9;               check(!sums::serialize(bad, out, err), "serialize refuses an unknown sum kind");
    bad = m; bad.files.clear();              check(!sums::serialize(bad, out, err), "serialize refuses a manifest with no files");
}

void freshness(const sums::Manifest& m) {
    std::string err;
    const sums::FileStat now[2] = {{1000, 111}, {2000, 222}};
    check(sums::matches_files(m, now, 2, err), "an unchanged pack matches");
    const sums::FileStat grown[2] = {{1000, 111}, {2001, 222}};
    const sums::FileStat touched[2] = {{1000, 112}, {2000, 222}};
    check(!sums::matches_files(m, grown, 2, err) && err.find("file 1") != std::string::npos, "a changed size is stale, naming the file");
    check(!sums::matches_files(m, touched, 2, err) && err.find("file 0") != std::string::npos, "a changed mtime is stale");
    check(!sums::matches_files(m, now, 1, err), "a different file count is stale");
    check(!sums::matches_files(m, nullptr, 2, err), "a null stat list is stale, not dereferenced");

    const uint32_t blobs[3] = {96, 80, 64}, other[3] = {96, 80, 65};
    check(sums::matches_geometry(m, 3, 4, blobs, err), "matching geometry and blob sizes pass");
    check(!sums::matches_geometry(m, 3, 5, blobs, err) && err.find("3x4") != std::string::npos, "another expert count is stale");
    check(!sums::matches_geometry(m, 4, 4, blobs, err), "another layer count is stale");
    check(!sums::matches_geometry(m, 3, 4, other, err) && err.find("layer 2") != std::string::npos, "a layer whose blob size changed is stale, naming it");
    check(!sums::matches_geometry(m, 3, 4, nullptr, err), "a null blob size list is stale, not dereferenced");
}
}  // namespace

int main() {
    const sums::Manifest m = sample();
    const std::string b = bytes_of(m);
    layout_bytes(b);
    round_trip(m, b);
    parse_refusals(b);
    serialize_refusals(m);
    freshness(m);
    std::puts("expert-sums: all checks passed");
    return 0;
}
