//------------------------------------------------------------------------------
// SPEX_Utilities/spex_free_mpz_array: free an mpz_t array
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Free a spex mpz_t array
#if defined (__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#include "spex_util_internal.h"

void spex_free_mpz_array
(
    mpz_t **x_handle,       // mpz_t array of size n
    int64_t n
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (x_handle == NULL || (*x_handle) == NULL)
    {
        // nothing to free (not an error)
        return ;
    }

    //--------------------------------------------------------------------------
    // free the mpz_t array x
    //--------------------------------------------------------------------------

    mpz_t *x = (*x_handle) ;

    for (int64_t i = 0 ; i < n ; i++)
    {
        SPEX_MPZ_CLEAR (x [i]) ;
    }

    SPEX_FREE (x) ;
    (*x_handle) = NULL ;
}

