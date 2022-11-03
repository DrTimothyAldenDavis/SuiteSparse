//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_get_smallest_pivot: find the smallest entry in a column
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function selects the pivot element as the smallest in the
 * column.  This is activated if the user sets option->pivot =
 * SPEX_TOL_SMALLEST or SPEX_SMALLEST.
 *
 * On output, the index of kth pivot is returned.
 */

#define SPEX_FREE_ALL        \
    SPEX_MPZ_CLEAR(small);

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_get_smallest_pivot
(
    int64_t *pivot,         // the index of smallest pivot
    SPEX_matrix* x,         // kth column of L and U
    int64_t* pivs,          // vector indicating if each row has been pivotal
    int64_t n,              // dimension of problem
    int64_t top,            // nonzeros are stored in xi[top..n-1]
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

    int64_t i, inew, j, flag ;
    int sgn, r;
    // Flag is non-negative until we have an initial starting value for small
    (*pivot) = -1;
    j = n;
    flag = top;
    mpz_t small; SPEX_MPZ_SET_NULL(small);
    SPEX_CHECK(SPEX_mpz_init(small));

    //--------------------------------------------------------------------------
    // Find an initial pivot. Fails if all terms are 0 in array x
    //--------------------------------------------------------------------------

    while (flag > -1 && flag < n)
    {
        // i location of first nonzero
        inew = xi[flag];

        //check if inew can be pivotal
        SPEX_CHECK(SPEX_mpz_sgn(&sgn, x->x.mpz[inew]));
        if (pivs[inew] < 0 && sgn != 0)
        {
            // Current smallest pivot
            SPEX_CHECK(SPEX_mpz_set(small, x->x.mpz[inew]));
            // Current smallest pivot location
            *pivot = inew;
            // Where to start the search for rest of nonzeros
            j = flag;
            // Exit the while loop
            flag = -5;
        }
        // Increment to next nonzero to search
        flag += 1;
    }

    //--------------------------------------------------------------------------
    // Iterate across remaining nonzeros
    //--------------------------------------------------------------------------

    for (i = j; i < n; i++)
    {
        inew = xi[i];
        // check if inew can be pivotal
        SPEX_CHECK(SPEX_mpz_cmpabs(&r, small, x->x.mpz[inew]));
        if (pivs[inew] < 0 && r > 0)
        {
            SPEX_CHECK(SPEX_mpz_sgn(&sgn, x->x.mpz[inew]));
            if (sgn != 0)
            {
                // Current best pivot location
                *pivot = inew;
                // Current best pivot value
                SPEX_CHECK(SPEX_mpz_set(small, x->x.mpz[inew]));
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    SPEX_FREE_ALL;
    if (*pivot == -1)
    {
        return SPEX_SINGULAR;
    }
    else
    {
        return SPEX_OK;
    }
}

