//------------------------------------------------------------------------------
// SPEX_Util/spex_expand_mpq_array: convert mpq array to mpz
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function converts a mpq array of size n into an appropriate
 * mpz array of size n. To do this, the lcm of the denominators is found as a
 * scaling factor. This function allows mpq arrays to be used in SPEX.
 */

#define SPEX_FREE_ALL               \
    SPEX_MPZ_CLEAR(temp);

#include "spex_util_internal.h"

SPEX_info spex_expand_mpq_array
(
    mpz_t* x_out,        // mpz array, on output x_out = x*scale
    mpq_t* x,            // mpq array that needs to be converted
    mpq_t scale,         // scaling factor. x_out = scale*x
    int64_t n,           // size of x
    const SPEX_options* option // Command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------
    // inputs have been checked in the only caller spex_cast_array
    ASSERT(n >= 0);    
    SPEX_info info ;

    //--------------------------------------------------------------------------
    // Define temporary mpz_t variable
    //--------------------------------------------------------------------------
    mpz_t temp;
    SPEX_MPZ_SET_NULL(temp);
    SPEX_CHECK (SPEX_mpz_init(temp)) ;

    // Find LCM of denominators of x
    SPEX_CHECK(SPEX_mpz_set(temp, SPEX_MPQ_DEN(x[0])));
    for (int64_t i = 1; i < n; i++)
    {
        SPEX_CHECK(SPEX_mpz_lcm(temp, SPEX_MPQ_DEN(x[i]), temp));
    }
    SPEX_CHECK(SPEX_mpq_set_z(scale,temp));

    for (int64_t i = 0; i < n; i++)
    {
        // x_out[i] = x[i]*temp
        SPEX_CHECK(SPEX_mpz_divexact(x_out[i], temp, SPEX_MPQ_DEN(x[i])));
        SPEX_CHECK(SPEX_mpz_mul(x_out[i], x_out[i], SPEX_MPQ_NUM(x[i])));
    }
    SPEX_FREE_ALL;
    return SPEX_OK;
}

