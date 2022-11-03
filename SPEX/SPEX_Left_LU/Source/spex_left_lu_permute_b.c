//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_permute_b: permute b, as b = P'*b
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function permutes b for forward substitution.
 * That is, b = P'*b.
 */

#define SPEX_FREE_ALL \
    SPEX_matrix_free (&b, NULL) ;

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_permute_b
(
    SPEX_matrix **b_handle,     // permuted RHS vector
    const SPEX_matrix *b2,      // unpermuted RHS vector (not modified)
    const int64_t *pinv,        // inverse row permutation
    const SPEX_options* option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE (b2, SPEX_DENSE, SPEX_MPZ) ;

    if (b_handle == NULL || !pinv) {return SPEX_INCORRECT_INPUT;}
    (*b_handle) = NULL ;

    //--------------------------------------------------------------------------
    // b(pinv) = b2
    //--------------------------------------------------------------------------

    int64_t m = b2->m ;
    int64_t n = b2->n ;

    // allocate x
    SPEX_matrix *b = NULL ;
    SPEX_CHECK (SPEX_matrix_allocate (&b, SPEX_DENSE, SPEX_MPZ, m, n,
        0, false, true, option)) ;

    // Set b = P'*b2
    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {
            SPEX_CHECK(SPEX_mpz_set(SPEX_2D(b,  pinv[i], j, mpz),
                                    SPEX_2D(b2,   i,     j, mpz)));
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*b_handle) = b ;
    return SPEX_OK;
}

