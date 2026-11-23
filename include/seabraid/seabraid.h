/* Copyright 2026 Saltwater Research LLC
 * SPDX-License-Identifier: Apache-2.0 OR MIT */
#ifndef SEABRAID_SEABRAID_H
#define SEABRAID_SEABRAID_H

#include <stddef.h>
#include <stdint.h>

/* Seabraid's structs carry 64-bit pointers and size_t, and every struct_size base (40, 80, ...) is an LP64
 * number. A 32-bit consumer would compile and then fail at run time with a confusing struct_size error, so
 * stop it here: the array size is -1, an error in every C and C++ compiler, when pointers are not 8 bytes.
 * (C89-safe static assert; the identifier is the message.) placement.h includes this header, so it is covered. */
typedef char seabraid_error_this_header_requires_64_bit_pointers[(sizeof(void *) == 8) ? 1 : -1];

/* SEABRAID_DEPRECATED marks the v1 session entry points, which are removed at 1.0 (use seabraid_session_open and
 * seabraid_session_submit). It expands to nothing unless the consumer defines SEABRAID_WARN_DEPRECATED, so
 * existing callers and strict -Werror builds are unaffected until they opt in. */
#if defined(SEABRAID_WARN_DEPRECATED) && defined(__GNUC__)
#define SEABRAID_DEPRECATED __attribute__((__deprecated__))
#elif defined(SEABRAID_WARN_DEPRECATED) && defined(_MSC_VER)
#define SEABRAID_DEPRECATED __declspec(deprecated)
#else
#define SEABRAID_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SEABRAID_ABI_VERSION 1u
#define SEABRAID_MULTI_PROVIDER_VERSION 2u
#define SEABRAID_PROVIDER_PARTIAL_ROUTES (1u << 0)
/* Batch v2 / provider v3 entry points: seabraid_batch_v2, seabraid_provider_v3 and the flags below, used by
 * seabraid_session_open and seabraid_session_submit. */
#define SEABRAID_BATCH_V2_VERSION 2u
#define SEABRAID_PROVIDER_V3_VERSION 3u
#define SEABRAID_PROVIDER_COMPACT_ROWS (1u << 1)
#define SEABRAID_PROVIDER_ACCEPTS_F16 (1u << 2)
#define SEABRAID_PROVIDER_ACCEPTS_BF16 (1u << 3)
#define SEABRAID_PROVIDER_REFERENCE (1u << 4)
#define SEABRAID_SKIP_EXPERT_ID (-1)

/* The layout epoch changes only for a non-append change; appends ride on struct_size. seabraid_abi_epoch()
 * reports the epoch the library was built with; a consumer that was compiled against another epoch must refuse to
 * use it. Every struct crossing the ABI starts with `uint32_t abi_version`. A struct the caller allocates for the
 * library to FILL also has `uint32_t struct_size` second: the caller sets both before the call and the library fails
 * with SEABRAID_INVALID when abi_version is unknown, struct_size is below the v1 size, or struct_size is not a
 * multiple of the struct's alignment (8 for every one of them except seabraid_capabilities_v1, which is 4): a size
 * that would end inside a field is refused, never rounded. The library writes min(caller struct_size, its own
 * current size) bytes and sets the struct_size field to that count, so a newer caller must read struct_size back to
 * learn which tail fields were filled. Padding bytes inside what it writes are zero. In an array of such structs the
 * element stride is the caller's struct_size; see seabraid_placement_outcome_v1 for what a reused array must reset. */
#define SEABRAID_ABI_EPOCH 1u
uint32_t seabraid_abi_epoch(void);

enum seabraid_status {
    SEABRAID_OK = 0,
    SEABRAID_INVALID = 1,
    SEABRAID_BUSY_OR_STATE = 2,
    SEABRAID_TIMEOUT = 3,
    SEABRAID_PROVIDER = 4,
    SEABRAID_INTERNAL = 5
};

enum seabraid_source_modality {
    SEABRAID_SOURCE_TEXT  = 1u << 0,
    SEABRAID_SOURCE_IMAGE = 1u << 1,
    SEABRAID_SOURCE_AUDIO = 1u << 2,
    SEABRAID_SOURCE_VIDEO = 1u << 3,
    SEABRAID_SOURCE_OTHER = 1u << 4
};

enum seabraid_operation { SEABRAID_OPERATION_ROUTED_MOE = 1 };
/* ROUTED_F16 and ROUTED_BF16 are activation formats of seabraid_batch_v2. Seabraid passes them through and never
 * converts. */
enum seabraid_buffer_format {
    SEABRAID_FORMAT_ROUTED_F32 = 1,
    SEABRAID_FORMAT_PARTIAL_F32 = 2,
    SEABRAID_FORMAT_ROUTED_F16 = 3,
    SEABRAID_FORMAT_ROUTED_BF16 = 4
};

/* Library-filled: set abi_version = SEABRAID_ABI_VERSION and struct_size = sizeof before the call. */
typedef struct seabraid_capabilities_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t operation;
    uint32_t input_format;
    uint32_t output_format;
    uint32_t source_modalities;
} seabraid_capabilities_v1;

/* Describes the runtime ABI. Exact formats and model support depend on the
 * provider. Source modalities describe already-encoded and routed activations. It reports the built-in F32 path
 * (input_format ROUTED_F32, output_format PARTIAL_F32) and cannot advertise a provider's 16-bit support: that is
 * the provider's own ACCEPTS_F16 / ACCEPTS_BF16 flag, known to the engine that wrote the provider. */
int32_t seabraid_query_capabilities(seabraid_capabilities_v1 *out);

typedef struct seabraid_session seabraid_session;

/* Library-filled: set abi_version = SEABRAID_ABI_VERSION and struct_size = sizeof before the call. */
typedef struct seabraid_session_status_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t submitted;
    uint64_t collected;
    uint64_t provider_failures;
    uint64_t wait_timeouts;
    uint32_t last_source_modalities;
    uint32_t outstanding;
    /* Appended after the first 48 bytes: a caller with struct_size 48 never sees them. */
    uint64_t rows_submitted; /* sum of tokens over accepted submits */
    uint64_t rows_shipped;   /* rows handed to the provider; fewer only when it set COMPACT_ROWS */
} seabraid_session_status_v1;

/* A caller-side snapshot. An outstanding ticket may already be ready for collection.
 * Timeout counts waits, since a caller may wait on the same ticket more than once.
 * It is a call like any other: from another thread while a call (a blocking seabraid_collect, say) is inside the
 * session it returns SEABRAID_BUSY_OR_STATE and writes nothing, and from the provider callback it is refused too. */
int32_t seabraid_session_status(seabraid_session *session, seabraid_session_status_v1 *out);

typedef struct seabraid_config_v1 {
    uint32_t abi_version;
    uint32_t hidden;
    uint32_t experts;
    uint32_t max_tokens;
    uint32_t top_k;
} seabraid_config_v1;

/* Dense, token-major arrays. Counts are elements. Activations contain
 * tokens * hidden values. Expert IDs and weights contain tokens * top_k values.
 * The caller fills this struct, so it carries abi_version but no struct_size. */
typedef struct seabraid_batch_v1 {
    uint32_t abi_version;
    uint32_t tokens;
    const float *activations;
    size_t activation_count;
    const int32_t *expert_ids;
    size_t expert_id_count;
    const float *weights;
    size_t weight_count;
} seabraid_batch_v1;

/* One callback runs at a time per session on its worker thread. The context
 * must stay valid until seabraid_close returns (see seabraid_close_timeout for
 * the exception). The caller synchronizes shared access. The callback must return and must
 * neither throw across the ABI nor call back into its session. A C++ exception that escapes
 * a callback is undefined behavior and aborts the process (Rust 1.81 and later), so a C++
 * provider must catch every exception inside the callback and return nonzero. All config,
 * batch and output pointers passed to the callback expire when it returns.
 * Complete all device work that touches these buffers before returning. Fill
 * every output element and return SEABRAID_OK, or return nonzero on failure. The library may or may not have zeroed
 * output before the call: write every element, and never rely on its prior contents.
 * Threading: each session owns one dedicated worker thread, and every callback of that session runs on it (the same
 * thread for every call, which matters for thread-bound CUDA or HIP contexts). seabraid_session_status called from
 * another thread while a call such as seabraid_collect is inside the session returns SEABRAID_BUSY_OR_STATE, so a
 * metrics thread cannot poll during a blocking collect. config and batch point at library-owned copies that expire
 * on return. A nonzero return fails the ticket: see seabraid_collect for what the caller then sees. */
typedef struct seabraid_provider_v1 {
    uint32_t abi_version;
    void *context;
    int32_t (*execute)(void *context, const seabraid_config_v1 *config,
                       const seabraid_batch_v1 *batch, float *output,
                       size_t output_count);
} seabraid_provider_v1;

/* One worker for a set of prepared operations sharing one seabraid_config_v1.
 * The engine assigns dense IDs 0 through operation_count - 1 and holds the
 * operation set stable until close. The v1 buffer and callback rules apply.
 * PARTIAL_ROUTES permits SEABRAID_SKIP_EXPERT_ID with zero weight. A row with
 * every pick skipped contributes zero, and the engine merges other experts.
 * flags may hold PARTIAL_ROUTES only: row compaction and the 16-bit formats are provider v3 features, and any other
 * bit is refused with SEABRAID_INVALID. */
typedef struct seabraid_provider_v2 {
    uint32_t abi_version;
    uint32_t operation_count;
    uint32_t flags;
    void *context;
    int32_t (*execute)(void *context, uint32_t operation_id,
                       const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                       float *output, size_t output_count);
} seabraid_provider_v2;

/* Batch v2, submitted with seabraid_session_submit. Caller-filled and sized: zero-initialize ({0}), then set
 * abi_version = SEABRAID_BATCH_V2_VERSION and struct_size = sizeof. Bytes past what the library knows must be zero.
 * Dense, token-major arrays; counts are elements, not bytes. activations holds tokens * hidden elements of
 * activation_format (SEABRAID_FORMAT_ROUTED_F32/_F16/_BF16), aligned to the element size (so an odd address is
 * refused for F16 and BF16; the provider itself always receives a 4-byte-aligned copy); expert_ids and weights hold
 * tokens * top_k values; within one row the expert ids must be distinct (SEABRAID_SKIP_EXPERT_ID may repeat).
 * placement_epoch 0 means unversioned; an engine with live placement passes the epoch of the snapshot it routed
 * with on every batch, and a stale epoch is refused with SEABRAID_BUSY_OR_STATE (re-snapshot, re-route, resubmit).
 * The mark behind that refusal is per session, raised only by an accepted submit (never by a refused one), and the
 * session is not linked to any placement handle: the library cannot tell whether the epoch you pass came from the
 * current table. One accepted huge epoch (even UINT64_MAX) permanently raises the session's mark, so every lower
 * epoch is stale from then on; only closing the session clears it. The check does not make a wrongly routed batch
 * safe: correctness comes from the engine's fence and drain, and a provider that holds only some experts should also
 * check its own ownership of each pick (see ADAPTER-GUIDE.md, Versioned submits and stale epochs).
 * A provider receives the library's own view of this struct: tokens may be fewer than were submitted
 * (SEABRAID_PROVIDER_COMPACT_ROWS), and fields past batch->struct_size do not exist.
 * With COMPACT_ROWS only a row whose picks are ALL SEABRAID_SKIP_EXPERT_ID is dropped. A partially skipped row is
 * shipped unchanged: its skipped picks keep id SEABRAID_SKIP_EXPERT_ID and weight zero. The provider receives no
 * map from kept row to original row (the library scatters the results back itself), and the config and batch it
 * is handed are library-owned copies, not the caller's memory. */
typedef struct seabraid_batch_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t operation_id;
    uint32_t source_modalities;
    uint64_t placement_epoch;
    uint32_t tokens;
    uint32_t activation_format;
    const void *activations;
    size_t activation_count;
    const int32_t *expert_ids;
    size_t expert_id_count;
    const float *weights;
    size_t weight_count;
} seabraid_batch_v2;

/* Provider v3, opened with seabraid_session_open. Caller-filled and sized (zero-initialize, set
 * abi_version = SEABRAID_PROVIDER_V3_VERSION and struct_size = sizeof). flags combine PARTIAL_ROUTES, COMPACT_ROWS,
 * ACCEPTS_F16, ACCEPTS_BF16 and REFERENCE; operation_count is at least 1. config is copied at open. The v1 buffer and
 * callback rules apply to execute. */
typedef struct seabraid_provider_v3 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t operation_count;
    uint32_t flags;
    const seabraid_config_v1 *config;
    void *context;
    int32_t (*execute)(void *context, const seabraid_config_v1 *config,
                       const seabraid_batch_v2 *batch, float *output, size_t output_count);
} seabraid_provider_v3;

/* Callers must serialize calls on each session. The library detects the common violations and returns a status:
 * a call that overlaps another call on the same session from another thread returns SEABRAID_BUSY_OR_STATE and does
 * nothing; any call on a session from its own provider callback (the worker thread) is refused the same way, whether
 * or not the caller is inside a call, because it would wait for itself; and a handle that is already closed, or was never
 * a session, returns SEABRAID_INVALID. This is best effort and not a license: it reads the memory behind the handle,
 * so a double close or a garbage pointer is still undefined behavior when the block was freed and reused or is not
 * readable. The provider must not be null: a null provider returns
 * SEABRAID_INVALID, because silently running the built-in transform would pass for model math.
 * seabraid_open_reference opens the built-in reference transform (an API demonstration and test transform,
 * not model math) when that is what the caller wants. On success, *out is owned by the caller until
 * seabraid_close or seabraid_close_timeout; on failure *out is NULL.
 * A panic caught inside any call poisons the session: every later call except close returns
 * SEABRAID_INTERNAL ("handle poisoned") and does nothing. */
/* Deprecated until removal at 1.0 (SEABRAID_DEPRECATED, opt in with SEABRAID_WARN_DEPRECATED): seabraid_open,
 * seabraid_open_reference, seabraid_open_multi, seabraid_prepare, seabraid_submit, seabraid_submit_routed and
 * seabraid_submit_operation. Open with seabraid_session_open (SEABRAID_PROVIDER_REFERENCE replaces
 * seabraid_open_reference) and submit with seabraid_session_submit. seabraid_collect, seabraid_close and the status
 * calls are not deprecated. */
SEABRAID_DEPRECATED int32_t seabraid_open(const seabraid_provider_v1 *provider, seabraid_session **out);
SEABRAID_DEPRECATED int32_t seabraid_open_reference(seabraid_session **out);
SEABRAID_DEPRECATED int32_t seabraid_open_multi(const seabraid_provider_v2 *provider, seabraid_session **out);

/* Opens a session from a v3 provider and prepares it with provider->config in one call (no seabraid_prepare). Returns
 * SEABRAID_INVALID, with the reason in seabraid_last_error, and sets *out to NULL, unless: out and provider are
 * non-null and aligned; provider->abi_version is SEABRAID_PROVIDER_V3_VERSION; struct_size is at least 40, at most
 * 1024, a multiple of 8, and every byte past what this library knows is zero; operation_count is at least 1; flags
 * holds only known SEABRAID_PROVIDER_* flags; COMPACT_ROWS comes with PARTIAL_ROUTES; REFERENCE (the built-in
 * reference transform, an API demonstration and test transform, not model math) comes with NULL execute and context
 * and no ACCEPTS_* flag, and without REFERENCE execute is non-null; config is non-null and valid. The caller's
 * provider and config memory is copied and may be freed on return; context must outlive the session. */
int32_t seabraid_session_open(const seabraid_provider_v3 *provider, seabraid_session **out);

/* Submits a v2 batch to a prepared session (from seabraid_session_open, or seabraid_open plus seabraid_prepare);
 * the inputs are copied before returning. One ticket may be outstanding per session. A refused batch changes
 * nothing: no buffer, no placement epoch mark, no counter, and *out_ticket stays untouched; the reason is in
 * seabraid_last_error. The order in which several problems are reported is not part of the contract. A batch is
 * refused when:
 * - the session, out_ticket or batch is null or misaligned, or the session is poisoned (INVALID / INTERNAL);
 * - abi_version is not SEABRAID_BATCH_V2_VERSION or the struct_size rules above are broken (INVALID);
 * - the session is not prepared, operation_id is not below operation_count or a ticket is outstanding
 *   (INVALID, BUSY_OR_STATE);
 * - source_modalities holds an unknown SEABRAID_SOURCE_* flag (INVALID);
 * - placement_epoch is stale or 0 after a nonzero one (BUSY_OR_STATE, INVALID);
 * - activation_format is not SEABRAID_FORMAT_ROUTED_F32, _F16 or _BF16, or the provider did not declare it: F32 is
 *   always accepted, F16 and BF16 only with ACCEPTS_F16 and ACCEPTS_BF16 (INVALID);
 * - tokens is outside 1..max_tokens, a count differs from tokens * hidden or tokens * top_k, or an array pointer is
 *   null or misaligned for its element size (INVALID);
 * - an expert id is not valid for the provider, a weight is not finite, or an id repeats within its row (INVALID).
 * Only seabraid_collect, seabraid_close and the status calls follow. No status code is added. */
int32_t seabraid_session_submit(seabraid_session *session, const seabraid_batch_v2 *batch,
                                uint64_t *out_ticket);

SEABRAID_DEPRECATED int32_t seabraid_prepare(seabraid_session *session, const seabraid_config_v1 *config);

/* Copies inputs before returning so callers can immediately reuse their arrays.
 * One ticket may be outstanding per session. */
SEABRAID_DEPRECATED int32_t seabraid_submit(seabraid_session *session, const seabraid_batch_v1 *batch,
                    uint64_t *out_ticket);

/* Source modalities are optional provenance for the submission contract.
 * Combine known flags for mixed rows. Zero leaves the origin unspecified. */
SEABRAID_DEPRECATED int32_t seabraid_submit_routed(seabraid_session *session, const seabraid_batch_v1 *batch,
                           uint32_t source_modalities, uint64_t *out_ticket);

/* Selects a prepared operation. Both legacy submit methods select ID zero.
 * The original provider and the built-in provider accept only ID zero. */
SEABRAID_DEPRECATED int32_t seabraid_submit_operation(seabraid_session *session, uint32_t operation_id,
                              const seabraid_batch_v1 *batch, uint32_t source_modalities,
                              uint64_t *out_ticket);

/* Output must have tokens * hidden elements (of the submitted batch, also under COMPACT_ROWS). A zero timeout polls
 * once. A timeout preserves the ticket and leaves output untouched. Successful collects consume the ticket.
 * If the provider callback returned nonzero (or panicked), collect returns SEABRAID_PROVIDER and: the ticket is
 * consumed (the session is free for the next submit), output is untouched, `collected` is not incremented,
 * `provider_failures` is, and the placement epoch mark raised by the submit stays raised. The callback's own return
 * code appears only in the log text ("provider callback returned status N", ERROR) and in seabraid_last_error on
 * the collecting thread; the status collect returns for a failed callback is SEABRAID_PROVIDER (4). A failure of the
 * library's own bookkeeping around the callback (for example no memory for the output scratch) is reported as it
 * is, SEABRAID_INTERNAL, and counts in `provider_failures` all the same.
 * A callback that PANICKED (only possible for a Rust provider) also consumes the ticket, but the session's worker
 * thread has stopped for good, because the provider's state is unknown and a replacement thread would break the
 * one-thread promise of the callbacks: every later submit returns SEABRAID_PROVIDER ("provider worker stopped")
 * without changing anything, no counter and no placement epoch mark. Close the session. */
int32_t seabraid_collect(seabraid_session *session, uint64_t ticket, uint32_t timeout_ms,
                     float *output, size_t output_count);

/* Drains outstanding execution and consumes the session, even on error. It waits without limit: a provider
 * callback that never returns blocks it forever. Use seabraid_close_timeout where that matters. Works on a
 * poisoned session. Called from the session's own provider callback (the worker thread) it would wait for itself,
 * so it is refused with SEABRAID_BUSY_OR_STATE (logged) and the session stays open; close from another thread. */
int32_t seabraid_close(seabraid_session *session);

/* Like seabraid_close but waits at most timeout_ms for the provider worker. If it does not finish, the worker
 * thread is detached (leaked, with an ERROR record naming the stuck ticket) and SEABRAID_TIMEOUT is returned. The
 * session is consumed either way. After a timeout the leaked worker may still call the provider, so its context
 * and callback code must stay valid for the rest of the process. A worker that is not inside a callback is not
 * stuck: with no ticket outstanding, or a ticket whose callback has returned but was never collected, the close
 * succeeds whatever the bound (zero included) and a failure nobody collected is logged. Refused from the provider
 * callback like seabraid_close. */
int32_t seabraid_close_timeout(seabraid_session *session, uint32_t timeout_ms);

/* Diagnostics. Every failing call is reported through a process-wide sink and is
 * remembered per thread (one record per failing call, except a seabraid_close_timeout that leaks a worker: it
 * logs the ERROR naming the stuck ticket and, for its SEABRAID_TIMEOUT status, the usual DEBUG record). The runtime opens no network connections and collects no
 * data: records carry codes, counts, ids and sizes only. */
enum seabraid_log_level { SEABRAID_LOG_DEBUG = 0, SEABRAID_LOG_INFO = 1, SEABRAID_LOG_WARN = 2, SEABRAID_LOG_ERROR = 3, SEABRAID_LOG_OFF = 4 };

/* One record per failure or notable event. message is NUL-terminated UTF-8, valid only during the call, and never
 * contains file paths, environment values, host names or tensor data. status is a seabraid_status (0 for notices).
 * Levels the library uses: a refusal (SEABRAID_INVALID, SEABRAID_BUSY_OR_STATE, including a stale placement epoch)
 * is WARN; SEABRAID_PROVIDER and SEABRAID_INTERNAL are ERROR; a collect that times out is DEBUG. A min_level of
 * ERROR therefore hides refusals. The placement calls apply the same rule to what a provider callback does during
 * seabraid_placement_apply, which returns SEABRAID_OK with the outcomes: a failure, timeout, unknown code, panic or
 * short stage write is one ERROR record, a clean refusal (no write began) is one WARN record, and a quarantine
 * notice is WARN. The sink runs on whichever thread logs: the calling thread for the library's own
 * records, and the session's worker thread for a seabraid_log made from inside a provider callback. */
typedef void (*seabraid_log_fn)(void *context, uint32_t level, int32_t status, const char *message);

/* Replaces the process-wide sink. A null sink restores the default (records at or above min_level go to stderr as
 * "seabraid: <LEVEL>: <message>"); the default min_level is SEABRAID_LOG_WARN; SEABRAID_LOG_OFF silences everything. The context
 * must stay valid until the sink is replaced AND any call already in progress on another thread has returned: this
 * function does not wait for such calls, so a thread may still be running the old sink after it returns. The simplest
 * safe rule is to keep the old sink and context alive for the process lifetime. The sink may run concurrently on any
 * thread and must not throw. */
int32_t seabraid_set_log_sink(seabraid_log_fn sink, void *context, uint32_t min_level);

/* Lets adapters and providers report through the same sink. level must be DEBUG..ERROR. */
int32_t seabraid_log(uint32_t level, int32_t status, const char *message);

/* Copies this thread's most recent failure message (the one behind the last nonzero status returned on this thread)
 * as a NUL-terminated, possibly truncated string. capacity may be 0; *needed (optional) receives full length + 1.
 * The message persists until the next failure on the same thread: a successful call does not clear it, so read it
 * right after a nonzero status, not later. Each thread has its own (a failure found by the collecting thread is
 * not visible from the worker thread). An empty string means no failure has been recorded on this thread. This call's own argument errors are returned
 * but never recorded, so a bad call here does not replace the message being asked for. */
int32_t seabraid_last_error(char *buffer, size_t capacity, size_t *needed);

/* The library version as a string: the package version, MAJOR.MINOR.PATCH with any pre-release suffix. Static
 * storage. It identifies the build; the compatibility contract is seabraid_abi_epoch() == SEABRAID_ABI_EPOCH plus
 * the library SONAME. */
const char *seabraid_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SEABRAID_SEABRAID_H */
