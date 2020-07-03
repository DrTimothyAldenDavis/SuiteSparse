//------------------------------------------------------------------------------
// SLIP_LU/slip_create_mpq_array: create a dense mpq array
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function creates an mpq array of size n.
 * This function is used internally for creating SLIP_MPQ matrices
 */

#include "slip_internal.h"

mpq_t* slip_create_mpq_array
(
    int64_t n              // size of the array
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (n <= 0) {return NULL;}

    //--------------------------------------------------------------------------

    // Malloc space
    mpq_t* x = (mpq_t*) SLIP_calloc(n, sizeof(mpq_t));
    if (!x) {return NULL;}
    for (int64_t i = 0; i < n; i++)
    {
        if (SLIP_mpq_init(x[i]) != SLIP_OK)
        {
            // Out of memory
            SLIP_MPQ_SET_NULL(x[i]);
            for (int64_t j = 0; j < n; j++)
            {
                if ( x[j] != NULL)
                {
                    SLIP_MPQ_CLEAR( x[j]);
                }
            }
            SLIP_FREE(x);
            return NULL;
        }
    }
    return x;
}

