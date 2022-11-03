//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_permute_x: permute x, as x = Q*x
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function permutes x to get it back in its original form.
 * That is, x = Q*x.
 */

#define SPEX_FREE_ALL \
    SPEX_matrix_free (&x, NULL) ;

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_permute_x
(
    SPEX_matrix **x_handle,     // permuted Solution vector
    SPEX_matrix *x2,            // unpermuted Solution vector (not modified)
    SPEX_LU_analysis *S,        // symbolic analysis with the column ordering Q
    const SPEX_options* option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE (x2, SPEX_DENSE, SPEX_MPQ) ;

    if (x_handle == NULL || !S || !S->q) {return SPEX_INCORRECT_INPUT;}
    (*x_handle) = NULL ;

    //--------------------------------------------------------------------------
    // x (q) = x2
    //--------------------------------------------------------------------------

    int64_t *q = S->q ;     // column permutation
    int64_t m = x2->m ;
    int64_t n = x2->n ;

    // allocate x
    SPEX_matrix *x = NULL ;
    SPEX_CHECK (SPEX_matrix_allocate (&x, SPEX_DENSE, SPEX_MPQ, m, n,
        0, false, true, option)) ;

    // Set x = Q*x2
    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {
            SPEX_CHECK(SPEX_mpq_set(SPEX_2D(x,  q[i], j, mpq),
                                    SPEX_2D(x2,   i,  j, mpq)));
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*x_handle) = x ;
    return SPEX_OK;
}

