//------------------------------------------------------------------------------
// SLIP_LU/slip_permute_x: permute x, as x = Q*x
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function permutes x to get it back in its original form.
 * That is, x = Q*x.
 */

#define SLIP_FREE_ALL \
    SLIP_matrix_free (&x, NULL) ;

#include "slip_internal.h"

SLIP_info slip_permute_x
(
    SLIP_matrix **x_handle,     // permuted Solution vector
    SLIP_matrix *x2,            // unpermuted Solution vector (not modified)
    SLIP_LU_analysis *S,        // symbolic analysis with the column ordering Q
    const SLIP_options* option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SLIP_info info ;
    SLIP_REQUIRE (x2, SLIP_DENSE, SLIP_MPQ) ;

    if (x_handle == NULL || !S || !S->q) {return SLIP_INCORRECT_INPUT;}
    (*x_handle) = NULL ;

    //--------------------------------------------------------------------------
    // x (q) = x2
    //--------------------------------------------------------------------------

    int64_t *q = S->q ;     // column permutation
    int64_t m = x2->m ;
    int64_t n = x2->n ;

    // allocate x
    SLIP_matrix *x = NULL ;
    SLIP_CHECK (SLIP_matrix_allocate (&x, SLIP_DENSE, SLIP_MPQ, m, n,
        0, false, true, option)) ;

    // Set x = Q*x2
    for (int64_t i = 0 ; i < m ; i++)
    {
        for (int64_t j = 0 ; j < n ; j++)
        {
            SLIP_CHECK(SLIP_mpq_set(SLIP_2D(x,  q[i], j, mpq),
                                    SLIP_2D(x2,   i,  j, mpq)));
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*x_handle) = x ;
    return SLIP_OK;
}

