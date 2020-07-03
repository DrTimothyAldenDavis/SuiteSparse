//------------------------------------------------------------------------------
// SLIP_LU/slip_get_largest_pivot: find a pivot entry in a column
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function selects the pivot element as the largest in the
 * column This is activated if the user sets option->pivot = SLIP_LARGEST.
 *
 * Note: This pivoting scheme is NOT recommended for SLIP LU.  It is provided
 * for comparison with other pivoting options.
 *
 * On output, the index of the largest pivot is returned.
 */

#define SLIP_FREE_ALL   \
    SLIP_MPZ_CLEAR(big);

#include "slip_internal.h"

SLIP_info slip_get_largest_pivot
(
    int64_t *pivot,         // the index of largest pivot
    SLIP_matrix* x,         // kth column of L and U
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

    SLIP_REQUIRE(x, SLIP_DENSE, SLIP_MPZ);

    SLIP_info info ;
    if (!pivs || !xi || !pivot) {return SLIP_INCORRECT_INPUT;}

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t i, inew ;
    int r ;
    (*pivot) = -1 ;
    mpz_t big ;
    SLIP_MPZ_SET_NULL (big) ;
    SLIP_CHECK (SLIP_mpz_init (big)) ;

    //--------------------------------------------------------------------------
    // Iterate accross the nonzeros in x
    //--------------------------------------------------------------------------

    for (i = top; i < n; i++)
    {
        // Location of the ith nonzero
        inew = xi[i];
        // inew can be pivotal
        SLIP_CHECK(SLIP_mpz_cmpabs(&r, big, x->x.mpz[inew]));
        if (pivs[inew] < 0 && r < 0)
        {
            // Current largest pivot location
            (*pivot) = inew;
            // Current largest pivot value
            SLIP_CHECK(SLIP_mpz_set(big, x->x.mpz[inew]));
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    SLIP_FREE_ALL;
    if ((*pivot) == -1)
    {
        return SLIP_SINGULAR;
    }
    else
    {
        return SLIP_OK;
    }
}

