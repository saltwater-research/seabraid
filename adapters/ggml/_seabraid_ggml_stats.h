// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#ifndef SEABRAID_GGML_STATS_INTERNAL_H
#define SEABRAID_GGML_STATS_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

/* Increment side of the transfer counter; the public read/reset side is declared in seabraid_ggml.h.
 * Every call is fail-open: it only adds to counters and never changes an adapter result. */

/* `transfers` host-to-device tensor uploads moving `bytes` in total. */
void seabraid_ggml_stats_sent(uint64_t bytes, uint64_t transfers);
/* One device-to-host result download of `bytes`. */
void seabraid_ggml_stats_returned(uint64_t bytes);
/* One batch that completed its upload, compute and download. */
void seabraid_ggml_stats_batch_done(void);

/* Writes the one-line process summary (counts only, no paths or tensor data) and returns its length, or 0
 * when no batch ran since the last reset. */
size_t seabraid_ggml_stats_summary(char *buffer, size_t size);

#endif
