//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_get_largest_pivot: find a pivot entry in a column
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function selects the pivot element as the largest in the
 * column This is activated if the user sets option->pivot = SPEX_LARGEST.
 *
 * Note: This pivoting scheme is NOT recommended for SPEX Left LU.  It is provided
 * for comparison with other pivoting options.
 *
 * On output, the index of the largest pivot is returned.
 */

#define SPEX_FREE_ALL   \
    SPEX_MPZ_CLEAR(big);

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_get_largest_pivot
(
    int64_t *pivot,         // the index of largest pivot
    SPEX_matrix* x,         // kth column of L and U
    int64_t* pivs,          // vector which indicates whether each row
                            // has been pivotal
    int64_t n,              // dimension of problem
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
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t i, inew ;
    int r ;
    (*pivot) = -1 ;
    mpz_t big ;
    SPEX_MPZ_SET_NULL (big) ;
    SPEX_CHECK (SPEX_mpz_init (big)) ;

    //--------------------------------------------------------------------------
    // Iterate accross the nonzeros in x
    //--------------------------------------------------------------------------

    for (i = top; i < n; i++)
    {
        // Location of the ith nonzero
        inew = xi[i];
        // inew can be pivotal
        SPEX_CHECK(SPEX_mpz_cmpabs(&r, big, x->x.mpz[inew]));
        if (pivs[inew] < 0 && r < 0)
        {
            // Current largest pivot location
            (*pivot) = inew;
            // Current largest pivot value
            SPEX_CHECK(SPEX_mpz_set(big, x->x.mpz[inew]));
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    SPEX_FREE_ALL;
    if ((*pivot) == -1)
    {
        return SPEX_SINGULAR;
    }
    else
    {
        return SPEX_OK;
    }
}

