// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Checker for strata-ext/src/log_text.hpp: log text never carries a full path (host only, header-only).
//   g++ -std=c++17 -Istrata-ext/src tests/log_text_test.cpp -o log_text_test
#include "log_text.hpp"

#include <cstdio>

using namespace seabraid::logtext;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main() {
    CHECK(basename_of("/srv/models/m.gguf") == "m.gguf");
    CHECK(basename_of("D:\\data\\m.gguf") == "m.gguf");
    CHECK(basename_of("m.gguf") == "m.gguf");
    CHECK(basename_of("/dir/") == "");
    CHECK(scrub_paths("cannot open /srv/models/m.gguf: No such file") == "cannot open m.gguf: No such file");
    CHECK(scrub_paths("a /x/y b /p/q,") == "a y b q,");
    CHECK(scrub_paths("no paths here") == "no paths here");
    CHECK(scrub_paths("") == "");
    CHECK(scrub_paths("  leading and  double  spaces ") == "  leading and  double  spaces ");
    CHECK(scrub_paths("/only/path") == "path");
    std::printf(g_fail ? "log_text_test: %d FAILED\n" : "log_text_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
