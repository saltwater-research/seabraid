/* Copyright 2026 Saltwater Research LLC
 * SPDX-License-Identifier: Apache-2.0 */
/* Calls one deprecated v1 entry point. native-tests/check_deprecated.cmake compiles this file twice: without
 * SEABRAID_WARN_DEPRECATED it must be clean under -Wall -Wextra -Werror (the default for every existing caller),
 * and with it the call must raise a deprecation diagnostic. Strict ANSI C, like the header. */
#include "seabraid/seabraid.h"

int deprecated_use(seabraid_session **out)
{
    return (int)seabraid_open_reference(out);
}
