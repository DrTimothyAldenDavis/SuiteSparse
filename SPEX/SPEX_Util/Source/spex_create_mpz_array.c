//------------------------------------------------------------------------------
// SPEX_Util/spex_create_mpz_array: create a dense mpz array
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function creates an mpz array of size n.
 * Utilized internally for creating SPEX_MPZ matrices
 */

#include "spex_util_internal.h"

mpz_t* spex_create_mpz_array
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
    mpz_t* x = (mpz_t*) SPEX_calloc(n, sizeof(mpz_t));
    if (!x) {return NULL;}
    for (int64_t i = 0; i < n; i++)
    {
        if (SPEX_mpz_init(x[i]) != SPEX_OK)
        {
            // Out of memory
            SPEX_MPZ_SET_NULL(x[i]);
            for (int64_t j = 0; j < n; j++)
            {
                if ( x[j] != NULL)
                {
                    SPEX_MPZ_CLEAR( x[j]);
                }
            }
            SPEX_FREE(x);
            return NULL;
        }
    }
    return x;
}

