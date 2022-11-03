//------------------------------------------------------------------------------
// SPEX_Util/SPEX_matrix_mul: multiplies a matrix by a scalar
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function multiplies the matrix x (CSC, triplet, or dense) by a
 * scalar.  This function requires x to have type mpz_t.
 *
 * On output the values of x are modified.
 */

#include "spex_util_internal.h"

SPEX_info SPEX_matrix_mul   // multiplies x by a scalar
(
    SPEX_matrix *x,         // matrix to be multiplied
    const mpz_t scalar      // scalar to multiply by
)
{

    if (!spex_initialized ( )) return (SPEX_PANIC) ;
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE_TYPE (x, SPEX_MPZ) ;

    //--------------------------------------------------------------------------
    // x = x * scalar
    //--------------------------------------------------------------------------

    int64_t nz;
    info = SPEX_matrix_nnz (&nz, x, NULL) ;
    if (info != SPEX_OK) {return info;}

    for (int64_t i = 0; i < nz; i++)
    {
        // x[i] = x[i]*scalar
        SPEX_CHECK( SPEX_mpz_mul( x->x.mpz[i], x->x.mpz[i], scalar));
    }

    return (SPEX_OK) ;
}

