// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Log hygiene for the seabraid helpers: stderr text may reach journals and bug reports, so it never carries a full
// path.  Owner of the "no full paths in logs" rule for strata-ext; the channel (shm) text is NOT passed through here
// because main reads it verbatim.  Checked by tests/log_text_test.cpp.
#pragma once
#include <string>

namespace seabraid::logtext {

/// "/a/b/c.gguf" -> "c.gguf".  A bare name is returned unchanged.
inline std::string basename_of(const std::string& path) {
    const size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? path : path.substr(cut + 1);
}

/// Reduces every whitespace-delimited token that contains a '/' to its basename, so library messages that embed a
/// path ("cannot open /srv/models/m.gguf: ...") are safe to log.  WHY token-based: the messages are free text, and a
/// token rule is predictable and cheap; a trailing ':' or ',' stays attached to the basename.
inline std::string scrub_paths(const std::string& text) {
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        const size_t end = text.find_first_of(" \t\n", i);
        const size_t stop = end == std::string::npos ? text.size() : end;
        const std::string tok = text.substr(i, stop - i);
        out += tok.find_first_of("/\\") == std::string::npos ? tok : basename_of(tok);
        const size_t next = text.find_first_not_of(" \t\n", stop);
        const size_t gap_end = next == std::string::npos ? text.size() : next;
        out += text.substr(stop, gap_end - stop);
        i = gap_end;
    }
    return out;
}

}  // namespace seabraid::logtext
