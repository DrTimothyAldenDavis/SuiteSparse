//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_free: wrapper for free
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Free the memory allocated by SPEX_calloc, SPEX_malloc, or SPEX_realloc.

#include "spex_util_internal.h"

void SPEX_free
(
    void *p         // pointer to memory space to free
)
{
    SuiteSparse_free (p);
}

