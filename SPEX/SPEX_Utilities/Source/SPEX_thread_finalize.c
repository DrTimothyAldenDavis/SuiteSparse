//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_thread_finalize: finish SPEX for a single user thread
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// SPEX_thread_finalize frees the working evironment for SPEX for a
// single user thread.

#include "spex_util_internal.h"

SPEX_info SPEX_thread_finalize ( void )
{
    if (!spex_initialized ( )) return (SPEX_PANIC);
    spex_gmp_finalize (0);
    return (SPEX_OK);
}

