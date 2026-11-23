// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Move-only owner of one POSIX descriptor.
// OWNER: the only RAII wrapper for descriptors in the Strata adapter.
// CONTRACT: header-only, no dependencies; get() borrows, release() gives up ownership.
// WHY: three call sites (vetted helper, pidfd, private channel fd) each need a close
//   on every early return; one wrapper means no branch can leak.
#pragma once

#include <unistd.h>

namespace seabraid::strata_adapter {

class UniqueFd {
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) noexcept : fd_(fd) {}
    UniqueFd(UniqueFd&& other) noexcept : fd_(other.release()) {}
    UniqueFd& operator=(UniqueFd&& other) noexcept {
        if (this != &other) reset(other.release());
        return *this;
    }
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    ~UniqueFd() { reset(); }

    int get() const noexcept { return fd_; }
    bool valid() const noexcept { return fd_ >= 0; }
    int release() noexcept {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }
    void reset(int fd = -1) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

}  // namespace seabraid::strata_adapter
