//------------------------------------------------------------------------------
// SLIP_LU/slip_cumsum: cumulative sum
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1]
 * in to c.  This function is lightly modified from CSparse.
 */

#include "slip_internal.h"

SLIP_info slip_cumsum
(
    int64_t *p,          // vector to store the sum of c
    int64_t *c,          // vector which is summed
    int64_t n            // size of c
)
{

    if (!p || !c) return SLIP_INCORRECT_INPUT;

    int64_t i, nz = 0 ;
    for (i = 0 ; i < n ; i++)
    {
        p [i] = nz ;
        nz += c [i] ;
        c [i] = p [i] ;
    }
    p [n] = nz ;
    return SLIP_OK ;
}
