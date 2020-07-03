//------------------------------------------------------------------------------
// SLIP_LU/slip_permute_b: permute b, as b = P'*b
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function permutes b for forward substitution.
 * That is, b = P'*b.
 */

#define SLIP_FREE_ALL \
    SLIP_matrix_free (&b, NULL) ;

#include "slip_internal.h"

SLIP_info slip_permute_b
(
    SLIP_matrix **b_handle,     // permuted RHS vector
    const SLIP_matrix *b2,      // unpermuted RHS vector (not modified)
    const int64_t *pinv,        // inverse row permutation
    const SLIP_options* option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SLIP_info info ;
    SLIP_REQUIRE (b2, SLIP_DENSE, SLIP_MPZ) ;

    if (b_handle == NULL || !pinv) {return SLIP_INCORRECT_INPUT;}
    (*b_handle) = NULL ;

    //--------------------------------------------------------------------------
    // b(pinv) = b2
    //--------------------------------------------------------------------------

    int64_t m = b2->m ;
    int64_t n = b2->n ;

    // allocate x
    SLIP_matrix *b = NULL ;
    SLIP_CHECK (SLIP_matrix_allocate (&b, SLIP_DENSE, SLIP_MPZ, m, n,
        0, false, true, option)) ;

    // Set b = P'*b2
    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {
            SLIP_CHECK(SLIP_mpz_set(SLIP_2D(b,  pinv[i], j, mpz),
                                    SLIP_2D(b2,   i,     j, mpz)));
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*b_handle) = b ;
    return SLIP_OK;
}

