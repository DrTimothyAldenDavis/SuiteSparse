//------------------------------------------------------------------------------
// SPEX_Util/spex_sparse_realloc: double the space for a sparse mpz matrix
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function expands a CSC SPEX_matrix by doubling its size. This
 * version merely expands x and i and does not initialize/allocate the values!
 * The only purpose of this function is for the factorization, it does not work
 * for general sparse matrices
 */

#include "spex_util_internal.h"

SPEX_info spex_sparse_realloc
(
    SPEX_matrix* A // the matrix to be expanded
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_REQUIRE (A, SPEX_CSC, SPEX_MPZ) ;

    //--------------------------------------------------------------------------
    // double the size of A->x and A->i
    //--------------------------------------------------------------------------

    int64_t nzmax = A->nzmax ;

    bool okx, oki ;
    A->x.mpz = (mpz_t *)
        SPEX_realloc (2*nzmax, nzmax, sizeof (mpz_t), A->x.mpz, &okx) ;
    A->i = (int64_t *)
        SPEX_realloc (2*nzmax, nzmax, sizeof (int64_t), A->i, &oki) ;
    if (!oki || !okx)
    {
        return (SPEX_OUT_OF_MEMORY) ;
    }

    A->nzmax = 2*nzmax ;

    //--------------------------------------------------------------------------
    // set newly allocated mpz entries to NULL
    //--------------------------------------------------------------------------

    for (int64_t p = nzmax ; p < 2*nzmax ; p++)
    {
        SPEX_MPZ_SET_NULL (A->x.mpz [p]) ;
    }

    return (SPEX_OK) ;
}

