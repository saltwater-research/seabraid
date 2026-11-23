// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/pidfd.hpp"

#include <cerrno>
#include <cstdint>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <thread>
#if defined(__linux__)
#include <poll.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace seabraid {

// Preserve the named helper's CLI defaults. A private launch gets exactly one
// descriptor from its controller and fails closed on an ambiguous transport.
struct QwenHelperOptions {
    std::string shm = "/seabraid0", pack, native;
    int64_t layers = 48, experts = 512, prefill_tokens = 0;
    int device = 0, shm_fd = -1, controller_pid = -1;
    // v9: how this helper serves kTierSwapFile.  kFileIoAuto = direct where every file allows it, else buffered (the helper
    // decides at launch); an explicit mode that cannot be honoured is a launch failure, not a silent downgrade.
    FileIo file_io = kFileIoAuto;
    bool named = false;

    bool parse(int argc, char** argv, std::string& error) {
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            // A trailing option used to parse an empty value as 0 or "".
            if (i + 1 >= argc) { error = option + " requires a value"; return false; }
            const std::string value = argv[++i];
            if (!set(option, value, error)) return false;
        }
        check_combination(error);
        return error.empty();
    }

    bool private_channel() const noexcept { return shm_fd >= 0; }

    bool keep_descriptor_private(std::string& error) const {
        // The launcher clears CLOEXEC for this one exec. Restore it before
        // touching the GPU runtime or other code that could launch a child.
        if (!private_channel() || fcntl(shm_fd, F_SETFD, FD_CLOEXEC) == 0) return true;
        error = std::string("--shm-fd: ") + std::strerror(errno);
        return false;
    }

    bool watch_controller(std::string& error) const {
        if (!private_channel()) return true;
#if defined(__linux__) && defined(SEABRAID_STRATA_HAVE_PIDFD)
        // A pidfd watches the whole engine process even if its spawning thread exits.
        if (::getppid() != controller_pid) {
            error = "private Qwen controller exited before startup";
            return false;
        }
        const int pidfd = seabraid::sys_pidfd_open(controller_pid);
        if (pidfd < 0) {
            error = std::string("cannot watch private Qwen controller: ") + std::strerror(errno);
            return false;
        }
        if (::getppid() != controller_pid) {
            ::close(pidfd);
            error = "private Qwen controller exited during startup";
            return false;
        }
        return start_watcher(pidfd, error);
#else
        error = "private Qwen helpers require Linux pidfd support";
        return false;
#endif
    }

    bool create_channel(Channel& channel, ChannelLimits limits, uint32_t window,
                        std::string& error) const {
        if (!private_channel()) {
            return channel.create(shm, limits.hidden, limits.cap, limits.layers, limits.experts,
                                  window, limits.blob_max, error, limits.pf_tokens, limits.pf_k);
        }
#if defined(__linux__)
        // The mapping alone keeps the capability alive after initialization.
        const bool ready = channel.create_private(shm_fd, limits, window, error);
        ::close(shm_fd);
        return ready;
#else
        error = "--shm-fd requires Linux";
        return false;
#endif
    }

private:
#if defined(__linux__) && defined(SEABRAID_STRATA_HAVE_PIDFD)
    // Detached thread that exits the whole helper when the controller's pidfd turns readable.  WHY a separate
    // function: thread creation can throw, and keeping the catch here keeps watch_controller a flat list of checks.
    static bool start_watcher(int pidfd, std::string& error) {
        try {
            std::thread([pidfd] {
                pollfd watch{pidfd, POLLIN, 0};
                int result;
                do { result = ::poll(&watch, 1, -1); } while (result < 0 && errno == EINTR);
                // Exit the whole helper even if its GPU thread is stuck or idle.
                ::_exit(result == 1 && (watch.revents & POLLIN) ? EXIT_SUCCESS : EXIT_FAILURE);
            }).detach();
            return true;
        } catch (const std::system_error& ex) {
            ::close(pidfd);
            error = std::string("cannot start private Qwen controller watcher: ") + ex.what();
            return false;
        }
    }

#endif
    // Cross-option rules, checked once every option parsed on its own.
    void check_combination(std::string& error) const {
        if (named && shm_fd >= 0) error = "--shm and --shm-fd are mutually exclusive";
        else if (pack.empty()) error = "--pack is required";
        else if (private_channel() && (device < 0 || device >= kDeviceBase))
            error = "--shm-fd requires a local GPU device ordinal";
        else if (private_channel() != (controller_pid > 0))
            error = "--shm-fd and --controller-pid must be supplied together";
    }

    bool set(const std::string& option, const std::string& value, std::string& error) {
        if (option == "--shm") { shm = value; named = true; }
        else if (option == "--pack") pack = value;
        else if (option == "--native") native = value;
        else if (option == "--layers") return parse_number(value, kMinLayers, kMaxLayers, layers, option, error);
        else if (option == "--experts") return parse_number(value, kMinExperts, kMaxExperts, experts, option, error);
        else if (option == "--device") return parse_number(value, 0, kDeviceBase - 1, device, option, error);
        else if (option == "--prefill-tokens") return parse_number(value, 0, kMaxPrefillTokens, prefill_tokens, option, error);
        else return set_more(option, value, error);
        return true;
    }

    // WHY a second function: `set` is one branch per option and the complexity gate caps a function; these are the options
    // that are not plain string/number fields (the private-channel pair and the v9 file mode).
    bool set_more(const std::string& option, const std::string& value, std::string& error) {
        if (option == "--file-io") return parse_file_io(value, error);
        if (option == "--shm-fd") return set_private_number(value, 3, shm_fd, option, error);
        if (option == "--controller-pid") return set_private_number(value, 1, controller_pid, option, error);
        error = "unknown argument " + option;
        return false;
    }

    // Bounds, each from Strata's geometry rules (geometry_checks.hpp):
    //  - experts: static_assert 32 <= n_expert <= 512 (router uses one lane per 32 experts).
    //  - layers: checks only require n_layers >= full_attention_interval (>= 1); 1024 is a
    //    ceiling far above any model and keeps the layers*experts residency table at 512 KiB.
    //  - prefill tokens: ChannelLimits::layout needs tokens * pf_k <= INT32_MAX and the
    //    geometry caps pf_k (n_expert_used) at 16.
    static constexpr int64_t kMinLayers = 1, kMaxLayers = 1024;
    static constexpr int64_t kMinExperts = 32, kMaxExperts = 512;
    static constexpr int64_t kMaxPrefillTokens = INT32_MAX / 16;

    // WHY from_chars and a whole-string check instead of atoi/strtoll: it
    // rejects empty text, signs we did not ask for, trailing junk and overflow
    // in one call, so "12x" or "99999999999999999999" can never become a number.
    template <typename Int>
    static bool parse_number(const std::string& value, int64_t minimum, int64_t maximum, Int& field,
                             const std::string& option, std::string& error) {
        int64_t number = 0;
        const char* end = value.data() + value.size();
        const auto parsed = std::from_chars(value.data(), end, number);
        if (value.empty() || parsed.ec != std::errc() || parsed.ptr != end || number < minimum || number > maximum) {
            error = option + " requires an integer from " + std::to_string(minimum) + " to " + std::to_string(maximum);
            return false;
        }
        field = static_cast<Int>(number);
        return true;
    }

    // WHY "gds" is named and refused here and not "unknown": GPUDirect Storage is a reserved seam (FileIo::kFileIoGds), so an
    // operator who asks for it learns this build does not serve it instead of being told the word is meaningless.
    bool parse_file_io(const std::string& value, std::string& error) {
        if (value == "auto") file_io = kFileIoAuto;
        else if (value == "direct") file_io = kFileIoDirect;
        else if (value == "buffered") file_io = kFileIoBuffered;
        else if (value == "off") file_io = kFileIoOff;
        else {
            error = value == "gds" ? "--file-io gds is not served by this build" : "--file-io requires auto, direct, buffered or off";
            return false;
        }
        return true;
    }

    static bool set_private_number(const std::string& value, int minimum, int& field,
                                   const std::string& option, std::string& error) {
        if (field >= 0) {
            error = option + " may be supplied only once";
            return false;
        }
        return parse_number(value, minimum, INT32_MAX, field, option, error);
    }
};

// WHY first thing in the helper's main and fail-open-loud: the helper only ever needs the rights it started with, so
// no_new_privs makes a setuid/file-capability exec (reachable through the GPU runtime's child processes) unable to
// gain any; if the kernel refuses, the helper still serves, and the caller logs that it runs without the floor.
inline bool forbid_new_privileges(std::string& error) {
#if defined(__linux__) && defined(PR_SET_NO_NEW_PRIVS)
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0) return true;
    error = std::string("prctl(PR_SET_NO_NEW_PRIVS): ") + std::strerror(errno);
#else
    error = "PR_SET_NO_NEW_PRIVS is unavailable on this platform";
#endif
    return false;
}

}  // namespace seabraid
