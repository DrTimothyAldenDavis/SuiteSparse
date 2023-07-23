//------------------------------------------------------------------------------
// SPEX_Util/SPEX_cumsum: cumulative sum
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1]
 * in to c.  This function is lightly modified from CSparse.
 */

#include "spex_util_internal.h"

SPEX_info SPEX_cumsum
(
    SuiteSparse_long *p,          // vector to store the sum of c
    SuiteSparse_long *c,          // vector which is summed
    SuiteSparse_long n            // size of c
)
{
    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    if (!p || !c) return SPEX_INCORRECT_INPUT;
    ASSERT(n >= 0);    
    SuiteSparse_long i, nz = 0 ;
    for (i = 0 ; i < n ; i++)
    {
        p [i] = nz ;
        nz += c [i] ;
        c [i] = p [i] ;
    }
    p [n] = nz ;
    return SPEX_OK ;
}
