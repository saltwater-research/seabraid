// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid-helper: the one host buffer a kTierSwapFile assembles its blob in (file_fetch.hpp is given it).  Page-locked when the
// driver allows, so the slot copy (cudaMemcpyAsync) is a DMA from it; ordinary aligned memory when it does not, which the driver
// stages itself: slower, never wrong (tier_apply takes any host pointer: it hands it to cudaMemcpyAsync).  WHY here and not in
// file_fetch.cpp: this is the only code that touches the GPU runtime for file moves, and file_fetch.cpp stays host-only so its tests
// run with no Strata and no GPU.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace seabraid::helper {

class FileStaging {
public:
    FileStaging() = default;
    ~FileStaging() { release(); }
    FileStaging(const FileStaging&) = delete;
    FileStaging& operator=(const FileStaging&) = delete;

    /// `bytes` of staging on `device`.  The device is made current first: a helper must never create a context on another card
    /// (the main engine's) just to pin its own staging.  A refused pin is logged and falls back to pageable memory; false only
    /// when even that is refused.
    bool alloc(int device, int64_t bytes) {
        release();
        if (bytes <= 0) return false;
        const cudaError_t set = cudaSetDevice(device);
        const cudaError_t pin = set != cudaSuccess ? set : cudaHostAlloc((void**) &p_, (size_t) bytes, cudaHostAllocPortable);
        if (pin == cudaSuccess) {
            pinned_ = true;
            return true;
        }
        const cudaError_t cleared = cudaGetLastError();   // the failed pin must not poison the next runtime call
        std::fprintf(stderr, "seabraid-helper: cannot pin the file staging buffer (%s, %s); using pageable memory\n", cudaGetErrorString(pin),
                     cudaGetErrorString(cleared));
        p_ = nullptr;
        return posix_memalign((void**) &p_, 4096, (size_t) bytes) == 0;
    }
    uint8_t* data() const { return p_; }
    bool pinned() const { return pinned_; }

private:
    void release() {
        if (p_ == nullptr) return;
        if (pinned_) {
            const cudaError_t e = cudaFreeHost(p_);
            if (e != cudaSuccess) std::fprintf(stderr, "seabraid-helper: cannot free the pinned file staging buffer (%s)\n", cudaGetErrorString(e));
        } else {
            std::free(p_);
        }
        p_ = nullptr;
        pinned_ = false;
    }
    uint8_t* p_ = nullptr;
    bool pinned_ = false;
};

}  // namespace seabraid::helper
