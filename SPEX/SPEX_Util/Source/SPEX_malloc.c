//------------------------------------------------------------------------------
// SPEX_Util/SPEX_malloc: wrapper for malloc
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Allocate memory space for SPEX functions.

#include "spex_util_internal.h"

void *SPEX_malloc
(
    size_t size        // size of memory space to allocate
)
{
    if (!spex_initialized ( )) return (NULL) ;
    return (SuiteSparse_malloc (1, size)) ;
}

