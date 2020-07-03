//------------------------------------------------------------------------------
// SLIP_LU/slip_matrix_mul: multiplies a matrix by a scalar
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function multiplies the matrix x (CSC, triplet, or dense) by a
 * scalar.  This function requires x to have type mpz_t.
 *
 * On output the values of x are modified.
 */

#include "slip_internal.h"

SLIP_info slip_matrix_mul   // multiplies x by a scalar
(
    SLIP_matrix *x,         // matrix to be multiplied
    const mpz_t scalar      // scalar to multiply by
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SLIP_info info ;
    SLIP_REQUIRE_TYPE (x, SLIP_MPZ) ;

    //--------------------------------------------------------------------------
    // x = x * scalar
    //--------------------------------------------------------------------------

    int64_t nz = SLIP_matrix_nnz (x, NULL) ;
    for (int64_t i = 0; i < nz; i++)
    {
        // x[i] = x[i]*scalar
        SLIP_CHECK( SLIP_mpz_mul( x->x.mpz[i], x->x.mpz[i], scalar));
    }

    return (SLIP_OK) ;
}

