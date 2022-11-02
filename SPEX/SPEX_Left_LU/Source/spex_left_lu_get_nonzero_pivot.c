//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_get_nonzero_pivot: find a nonzero pivot in a column
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function obtains the first eligible nonzero pivot
 * This is enabled if the user sets option->pivot = SPEX_FIRST_NONZERO
 *
 * Note: This pivoting scheme is NOT recommended for SPEX Left LU.  It is provided
 * for comparison with other pivoting options.
 *
 * On output, the kth pivot is returned.
 */

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_get_nonzero_pivot // find the first eligible nonzero pivot
(
    int64_t *pivot,         // the index of first eligible nonzero pivot
    SPEX_matrix* x,         // kth column of L and U
    int64_t* pivs,          // vector indicating which rows are pivotal
    int64_t n,              // size of x
    int64_t top,            // nonzero pattern is located in xi[top..n-1]
    int64_t* xi             // nonzero pattern of x
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_REQUIRE(x, SPEX_DENSE, SPEX_MPZ);

    SPEX_info info ;
    if (!pivs || !xi || !pivot) {return SPEX_INCORRECT_INPUT;}

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    (*pivot) = -1; // used later to check for singular matrix

    //--------------------------------------------------------------------------
    // Iterate across the nonzeros in x
    //--------------------------------------------------------------------------

    for (int64_t i = top; i < n; i++)
    {
        // inew is the location of the ith nonzero
        int64_t inew = xi[i];
        // check if x[inew] is an eligible pivot
        int sgn ;
        SPEX_CHECK (SPEX_mpz_sgn (&sgn, x->x.mpz[inew])) ;
        if (sgn != 0 && pivs [inew] < 0)
        {
            (*pivot) = inew;
            // End the loop
            return SPEX_OK;
        }
    }

    // return error code since no qualified pivot found
    return SPEX_SINGULAR;
}

