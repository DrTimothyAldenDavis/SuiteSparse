//------------------------------------------------------------------------------
// SPEX_Util/SPEX_free: wrapper for free
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Free the memory allocated by SPEX_calloc, SPEX_malloc, or SPEX_realloc.

#include "spex_util_internal.h"

void SPEX_free
(
    void *p         // pointer to memory space to free
)
{
    SuiteSparse_free (p) ;
}

