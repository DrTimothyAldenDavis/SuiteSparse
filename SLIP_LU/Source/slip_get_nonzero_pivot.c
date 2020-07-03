//------------------------------------------------------------------------------
// SLIP_LU/slip_get_nonzero_pivot: find a nonzero pivot in a column
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function obtains the first eligible nonzero pivot
 * This is enabled if the user sets option->pivot = SLIP_FIRST_NONZERO
 *
 * Note: This pivoting scheme is NOT recommended for SLIP LU.  It is provided
 * for comparison with other pivoting options.
 *
 * On output, the kth pivot is returned.
 */

#include "slip_internal.h"

SLIP_info slip_get_nonzero_pivot // find the first eligible nonzero pivot
(
    int64_t *pivot,         // the index of first eligible nonzero pivot
    SLIP_matrix* x,         // kth column of L and U
    int64_t* pivs,          // vector indicating which rows are pivotal
    int64_t n,              // size of x
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
        SLIP_CHECK (SLIP_mpz_sgn (&sgn, x->x.mpz[inew])) ;
        if (sgn != 0 && pivs [inew] < 0)
        {
            (*pivot) = inew;
            // End the loop
            return SLIP_OK;
        }
    }

    // return error code since no qualified pivot found
    return SLIP_SINGULAR;
}

