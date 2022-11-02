//------------------------------------------------------------------------------
// SPEX_Util/spex_create_mpfr_array: create a dense mpfr array
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function creates a MPFR array of desired precision. It is used
            internally for the SPEX_matrix_allocate function */

#include "spex_util_internal.h"

mpfr_t* spex_create_mpfr_array
(
    int64_t n,           // size of the array
    const SPEX_options *option // command options containing the prec for mpfr
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (n <= 0) {return NULL;}
    uint64_t prec = SPEX_OPTION_PREC (option) ;

    //--------------------------------------------------------------------------

    mpfr_t* x = (mpfr_t*) SPEX_calloc(n, sizeof(mpfr_t));
    if (!x) {return NULL;}
    for (int64_t i = 0; i < n; i++)
    {
        if (SPEX_mpfr_init2(x[i], prec) != SPEX_OK)
        {
            SPEX_MPFR_SET_NULL(x[i]);
            for (int64_t j = 0; j < n; j++)
            {
                if ( x[j] != NULL)
                {
                    SPEX_MPFR_CLEAR( x[j]);
                }
            }
            SPEX_FREE(x);
            return NULL;
        }
    }
    return x;
}

