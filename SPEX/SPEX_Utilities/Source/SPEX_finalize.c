//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_finalize: finalize SPEX
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// SPEX_finalize frees the working environment for SPEX library.
// This function must be called by the same user thread that called
// SPEX_initialize or SPEX_initialize_expert.

#include "spex_util_internal.h"

SPEX_info SPEX_finalize
(
    void
)
{

    if (!spex_initialized ( )) { return (SPEX_PANIC); }

    SPEX_mpfr_free_cache ( );    // Free mpfr internal cache

    // the primary thread always frees the spex_gmp object
    spex_gmp_finalize (1);

    spex_set_initialized (false);
    return (SPEX_OK);
}

