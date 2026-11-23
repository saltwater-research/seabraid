// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Test-only fixture helpers shared by the process-spawning fixtures.
// OWNER: `check` and the scratch-executable guard; nothing outside the tests includes
//   this. WHY a registry and not just a destructor: check() ends the fixture with
//   std::exit(1), which skips the destructors of stack objects, so a failed check
//   used to leave its mkstemp copy of the fixture binary in $TMPDIR.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace seabraid::strata_adapter::testing {

inline std::vector<std::string>& scratch_paths() {
    static std::vector<std::string> paths;
    return paths;
}

// Only the process that created the files may remove them: fork()ed test
// controllers that fail a check must not delete the parent's helper copy.
inline pid_t scratch_owner() {
    static const pid_t owner = ::getpid();
    return owner;
}

inline void remove_scratch() noexcept {
    if (::getpid() != scratch_owner()) return;
    for (const auto& path : scratch_paths()) ::unlink(path.c_str());
    scratch_paths().clear();
}

inline void check(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "%s\n", message);
    remove_scratch();
    std::exit(1);
}

// A scratch executable whose mode the test controls; removed on scope exit and
// on a failed check().
struct ScratchFile {
    std::string path;
    ScratchFile() {
        scratch_owner();
        const char* dir = std::getenv("TMPDIR");
        path = std::string(dir && *dir ? dir : "/tmp") + "/seabraid_helper_XXXXXX";
        const int fd = mkstemp(path.data());
        check(fd >= 0, "create a scratch helper file");
        close(fd);
        scratch_paths().push_back(path);
    }
    ScratchFile(const ScratchFile&) = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;
    ~ScratchFile() {
        ::unlink(path.c_str());
        auto& all = scratch_paths();
        for (auto it = all.begin(); it != all.end(); ++it)
            if (*it == path) { all.erase(it); break; }
    }
    void mode(mode_t bits) const { check(chmod(path.c_str(), bits) == 0, "chmod the scratch helper"); }
    // WHY a copy: build outputs inherit the builder's umask (often 002, i.e.
    // group-writable), which the launcher rightly refuses as a helper.
    void copy_of(const char* source) const {
        std::ifstream in(source, std::ios::binary);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << in.rdbuf();
        out.close();
        check(in.good() || in.eof(), "read the fixture binary");
        check(!out.fail(), "write the trusted helper copy");
        mode(0755);
    }
    // Atomically puts a different executable at this path (new inode), the way a
    // hostile or careless writer could replace a helper after it was vetted.
    void replace_with(const char* script) const {
        const std::string next = path + ".next";
        {
            std::ofstream out(next, std::ios::trunc);
            out << script;
        }
        check(chmod(next.c_str(), 0755) == 0, "chmod the replacement");
        check(rename(next.c_str(), path.c_str()) == 0, "swap the replacement in");
    }
};

}  // namespace seabraid::strata_adapter::testing
