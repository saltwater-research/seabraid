// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// QualHeat fixture: the walk-file grammar, every refusal, and the bias rule (from_round, later line replaces, clamp at 0).
#include "placement_qual.hpp"

#include "diag_capture.hpp"
#include "test_support.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using seabraid::strata_adapter::QualHeat;
using seabraid::strata_adapter::testing::check;
namespace capture = seabraid::strata_adapter::capture;

std::string write_file(const char* text) {
    char path[] = "/tmp/seabraid_qual_XXXXXX";
    const int fd = mkstemp(path);
    check(fd >= 0 && write(fd, text, std::strlen(text)) == static_cast<ssize_t>(std::strlen(text)), "write a walk file");
    close(fd);
    return path;
}

bool loads(const char* text, QualHeat& qual) {
    const std::string path = write_file(text);
    const bool ok = qual.load(path, 2, 3);
    unlink(path.c_str());
    return ok;
}

void refusals() {
    for (const char* bad : {"1 0 9 5\n", "1 2 0 5\n", "1 0 1\n", "x 0 1 5\n", "1 0 1 nan\n", "1 0 1 inf\n",
                            "-1 0 1 5\n", "# nothing\n", "1 0 1 5 extra\n", ""}) {
        QualHeat qual;
        check(!loads(bad, qual) && qual.entries() == 0, bad);
    }
    QualHeat missing;
    check(!missing.load("/nonexistent/qual-walk", 2, 3), "an unreadable file is refused");
    std::string many;
    for (size_t i = 0; i <= QualHeat::kMaxEntries; ++i) many += "1 0 0 1\n";
    QualHeat big;
    check(!loads(many.c_str(), big), "more than kMaxEntries is refused");
}

void bias_rule() {
    QualHeat qual;
    capture::clear();
    check(loads("# walk cell (1,2) up, then down\n  \n3 1 2 100\n6 1 2 -100\n4 0 0 -1\n", qual), "a walk loads");
    capture::require("QUALIFICATION heat bias active (3 entries)", 1);
    std::vector<double> heat(6, 5.0);
    check(!qual.apply(heat, 2) && heat == std::vector<double>(6, 5.0), "nothing is active before its round");
    heat.assign(6, 5.0);
    check(qual.apply(heat, 3) && heat[5] == 105.0 && heat[0] == 5.0, "round 3: the walk up");
    heat.assign(6, 5.0);
    check(qual.apply(heat, 4) && heat[5] == 105.0 && heat[0] == 4.0, "round 4: both entries");
    heat.assign(6, 5.0);
    check(qual.apply(heat, 7) && heat[5] == 0.0, "round 7: the later line replaces, clamped at 0");
}

}  // namespace

int main() {
    capture::start();
    refusals();
    bias_rule();
    std::puts("Seabraid Strata placement qualification heat fixture passed");
}
