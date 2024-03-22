//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_thread_initialize: init SPEX for a single user thread
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// SPEX_thread_initialize initializes the working evironment for SPEX for a
// single user thread.

#include "spex_util_internal.h"

SPEX_info SPEX_thread_initialize ( void )
{
    if (!spex_initialized ( )) return (SPEX_PANIC);
    return (spex_gmp_initialize (0)) ;
}

