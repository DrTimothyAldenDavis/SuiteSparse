//------------------------------------------------------------------------------
// SLIP_LU/slip_create_mpfr_array: create a dense mpfr array
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function creates a MPFR array of desired precision. It is used
            internally for the SLIP_matrix_allocate function */

#include "slip_internal.h"

mpfr_t* slip_create_mpfr_array
(
    int64_t n,           // size of the array
    const SLIP_options *option // command options containing the prec for mpfr
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (n <= 0) {return NULL;}
    uint64_t prec = SLIP_OPTION_PREC (option) ;

    //--------------------------------------------------------------------------

    mpfr_t* x = (mpfr_t*) SLIP_calloc(n, sizeof(mpfr_t));
    if (!x) {return NULL;}
    for (int64_t i = 0; i < n; i++)
    {
        if (SLIP_mpfr_init2(x[i], prec) != SLIP_OK)
        {
            SLIP_MPFR_SET_NULL(x[i]);
            for (int64_t j = 0; j < n; j++)
            {
                if ( x[j] != NULL)
                {
                    SLIP_MPFR_CLEAR( x[j]);
                }
            }
            SLIP_FREE(x);
            return NULL;
        }
    }
    return x;
}

