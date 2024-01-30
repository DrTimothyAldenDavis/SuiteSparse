//------------------------------------------------------------------------------
// SPEX_Utilities/spex_create_mpq: create an mpq_t entry
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function safely creates and initializes an mpq_t entry.
 */

// The SPEX_mpq_init function is wrapped in this method to avoid a spurious
// compiler warning in SPEX_MPQ_SET_NULL, about writing past the size of a
// variable.  The warning arises when a high level of optimization is used.
// The warning cannot be supressed entirely because it would require a
// modifcation to GMP itself.

#include "spex_util_internal.h"
SPEX_info spex_create_mpq
(
    mpq_t x                  // mpq_t entry to be initialized
)
{

    SPEX_info info = SPEX_mpq_init(x);
    if (info != SPEX_OK)
    {
        // Out of memory
        SPEX_MPQ_SET_NULL(x);
        return info;
    }
    return SPEX_OK;
}

