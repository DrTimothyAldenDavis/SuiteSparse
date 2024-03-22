//------------------------------------------------------------------------------
// SPEX_Utilities/spex_permute_dense_matrix: permute rows of a dense matrix A,
// as A_out = P*A_in
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function gets a copy of a row-wise permuted dense matrix as
 * A_out = P*A_in.
 */

#define SPEX_FREE_ALL        \
    SPEX_matrix_free (&Atmp, NULL);

#include "spex_util_internal.h"

SPEX_info spex_permute_dense_matrix
(
    SPEX_matrix *A_handle,      // permuted A
    const SPEX_matrix A_in,     // unpermuted A (not modified)
    const int64_t *P,           // row permutation
    const SPEX_options option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE (A_in, SPEX_DENSE, SPEX_MPZ);

    if (A_handle == NULL || !P) {return SPEX_INCORRECT_INPUT;}
    (*A_handle) = NULL ;

    //--------------------------------------------------------------------------
    // b(pinv) = b2
    //--------------------------------------------------------------------------

    int64_t m = A_in->m ;
    int64_t n = A_in->n ;

    // allocate x
    SPEX_matrix Atmp = NULL ;
    SPEX_CHECK (SPEX_matrix_allocate (&Atmp, SPEX_DENSE, SPEX_MPZ, m, n,
        0, false, true, option));

    // Set Atmp = P*A
    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {
            SPEX_MPZ_SET(SPEX_2D(Atmp,  P[i], j, mpz),
                                    SPEX_2D(A_in,    i , j, mpz));
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*A_handle) = Atmp ;
    return SPEX_OK;
}

