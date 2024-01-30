//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_calloc: wrapper for calloc
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Allocate and initialize memory space for SPEX functions

#include "spex_util_internal.h"

void *SPEX_calloc
(
    size_t nitems,      // number of items to allocate
    size_t size         // size of each item
)
{

    return (SuiteSparse_calloc (nitems, size));
}

