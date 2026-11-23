// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Runs the engine's own --retier* startup rules (retier_parse_flag, retier_check, retier_check_helpers) over every row of
// tests/data/retier_cases.tsv.  The installer's mirror (install/_retier_rules.py) is tested on the SAME file by
// tests/install/test_retier_rules.py, so the two cannot drift: a rule changed here and not there fails there, and the reverse.
//
// WHAT THIS FIXTURE IS NOT: engine_verdict() below re-implements generate.cpp's parse loop (the "unknown argument" text, the
// "<flag> needs a value" text, the 4 / 96 adapt defaults) and calls retier_parse_flag / retier_check / retier_check_helpers directly.  So
// those texts and defaults are compared against a COPY of the engine, not the engine, and the fixture cannot represent quoted spaces
// (arguments split on a single space; "''" stands for an empty argument).  tests/install/test_retier_rules.py pins the copied facts
// to the patches that carry them where a patch does.  Real startup (--serve, model cell counts, the helper tier) is not exercised.
// Usage: seabraid_retier_rules_case_test <retier_cases.tsv>
#include "strata/seabraid/retier_loop.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace seabraid;

static constexpr int64_t kCells = 32;   // any positive model size: the rules under test only ask that the counts match it

// generate.cpp's two flags the checks read; they are not --retier flags, so they are read here.  True when `a` was one of them.
static bool read_adapt(const std::string& a, const char* v, int& every, int& swaps) {
    if (a == "--adapt-every") every = std::atoi(v);
    else if (a == "--adapt-swaps") swaps = std::atoi(v);
    else return false;
    return true;
}

// generate.cpp's parse loop over the whole list.  "" = every argument read; else the text the engine prints before exiting 2.
static std::string parse_all(const std::vector<std::string>& args, RetierConfig& c, int& adapt_every, int& adapt_swaps) {
    std::string err, missing;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        // generate.cpp's next(what): the value, or "<what> needs a value" and exit 2.  Only a flag that is KNOWN calls it.
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= args.size()) { missing = std::string(what) + " needs a value"; return ""; }
            return args[++i].c_str();
        };
        if (a == "--adapt-every" || a == "--adapt-swaps") {
            const char* v = next(a.c_str());
            if (!missing.empty()) return missing;
            read_adapt(a, v, adapt_every, adapt_swaps);
            continue;
        }
        const int rr = retier_parse_flag(a, next, c, err);
        if (!missing.empty()) return missing;
        if (rr == 2) return err;
        if (rr == 0) return "unknown argument: " + a;   // generate.cpp's else-branch text
    }
    return "";
}

// What generate.cpp does with one argument list: the parse loop, then the two startup checks, in that order.  "" = accepted.
static std::string engine_verdict(const std::vector<std::string>& args, int helpers) {
    RetierConfig c;
    int adapt_every = 4, adapt_swaps = 96;   // generate.cpp's Options defaults (the installer mirror names the same two numbers)
    std::string err = parse_all(args, c, adapt_every, adapt_swaps);
    if (!err.empty()) return err;
    const size_t usage = (adapt_every > 0 && adapt_swaps > 0) ? (size_t) kCells : 0;   // the heat is allocated only with both positive
    if (!retier_check(c, adapt_every, adapt_swaps, usage, kCells, helpers >= 1, err)) return err;
    if (!retier_check_helpers(c, helpers, err)) return err;
    return "";
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream in(s);
    while (std::getline(in, cur, sep)) if (!cur.empty()) out.push_back(cur);
    return out;
}

// One row: name, helpers, args, expected text ("-" = accepted / no args).  Returns false (and prints) on a mismatch.
static bool run_row(const std::vector<std::string>& col) {
    std::vector<std::string> args = col[2] == "-" ? std::vector<std::string>{} : split(col[2], ' ');
    for (std::string& a : args) if (a == "''") a.clear();   // the case file's spelling of an empty argument
    const std::string want = col[3] == "-" ? "" : col[3];
    const std::string got = engine_verdict(args, std::atoi(col[1].c_str()));
    if (got == want) return true;
    std::printf("FAIL %s\n  want: %s\n  got:  %s\n", col[0].c_str(), want.c_str(), got.c_str());
    return false;
}

static std::vector<std::string> columns(const std::string& line) {
    std::vector<std::string> col;
    std::istringstream in(line);
    for (std::string cell; std::getline(in, cell, '\t');) col.push_back(cell);
    return col;
}

static bool is_row(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return !line.empty() && line[0] != '#';
}

static bool run_line(const std::string& line) {
    const std::vector<std::string> col = columns(line);
    if (col.size() == 4) return run_row(col);
    std::printf("FAIL malformed row (%zu columns): %s\n", col.size(), line.c_str());
    return false;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::printf("usage: %s <retier_cases.tsv>\n", argv[0]); return 2; }
    std::ifstream f(argv[1]);
    if (!f) { std::printf("FAIL cannot open %s\n", argv[1]); return 1; }
    int rows = 0, fail = 0;
    for (std::string line; std::getline(f, line);) {
        if (!is_row(line)) continue;
        ++rows;
        if (!run_line(line)) ++fail;
    }
    if (rows == 0) { std::printf("FAIL no cases read from %s\n", argv[1]); return 1; }   // an empty file must not be a green run
    std::printf("retier_rules_case_test: %d rows, %d failed\n", rows, fail);
    return fail ? 1 : 0;
}
