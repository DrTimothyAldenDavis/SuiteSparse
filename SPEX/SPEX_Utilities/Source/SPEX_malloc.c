//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_malloc: wrapper for malloc
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Allocate memory space for SPEX functions.

#include "spex_util_internal.h"

void *SPEX_malloc
(
    size_t size        // size of memory space to allocate
)
{

    return (SuiteSparse_malloc (1, size));
}

