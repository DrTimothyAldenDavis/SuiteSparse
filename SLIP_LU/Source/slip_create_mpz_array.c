//------------------------------------------------------------------------------
// SLIP_LU/slip_create_mpz_array: create a dense mpz array
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function creates an mpz array of size n.
 * Utilized internally for creating SLIP_MPZ matrices
 */

#include "slip_internal.h"

mpz_t* slip_create_mpz_array
(
    int64_t n            // size of the array
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (n <= 0) {return NULL;}

    //--------------------------------------------------------------------------

    // Malloc space
    mpz_t* x = (mpz_t*) SLIP_calloc(n, sizeof(mpz_t));
    if (!x) {return NULL;}
    for (int64_t i = 0; i < n; i++)
    {
        if (SLIP_mpz_init(x[i]) != SLIP_OK)
        {
            // Out of memory
            SLIP_MPZ_SET_NULL(x[i]);
            for (int64_t j = 0; j < n; j++)
            {
                if ( x[j] != NULL)
                {
                    SLIP_MPZ_CLEAR( x[j]);
                }
            }
            SLIP_FREE(x);
            return NULL;
        }
    }
    return x;
}

