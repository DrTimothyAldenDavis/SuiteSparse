//------------------------------------------------------------------------------
// SPEX_Util/SPEX_calloc: wrapper for calloc
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
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
    if (!spex_initialized ( )) return (NULL) ;

    return (SuiteSparse_calloc (nitems, size)) ;
}

