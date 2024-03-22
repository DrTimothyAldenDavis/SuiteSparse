//------------------------------------------------------------------------------
// SPEX_Utilities/spex_expand_mpq_array: convert mpq array to mpz
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function converts a mpq array of size n into an appropriate
 * mpz array of size n. To do this, the lcm of the denominators is found as a
 * scaling factor. This function allows mpq arrays to be used in SPEX.
 */

#define SPEX_FREE_ALL               \
    SPEX_mpz_clear (temp);

#include "spex_util_internal.h"

SPEX_info spex_expand_mpq_array
(
    mpz_t *x_out,        // mpz array, on output x_out = x*scale
    mpq_t *x,            // mpq array that needs to be converted
    mpq_t scale,         // scaling factor. x_out = scale*x
    int64_t n,           // size of x
    const SPEX_options option // Command options
)
{

    // inputs have checked in the only caller spex_cast_array
    ASSERT(n >= 0);
    SPEX_info info ;
    mpz_t temp;
    SPEX_mpz_set_null (temp);
    SPEX_MPZ_INIT(temp);

    // Find LCM of denominators of x
    SPEX_MPZ_SET(temp,SPEX_MPQ_DEN(x[0]));
    for (int64_t i = 1; i < n; i++)
    {
        SPEX_MPZ_LCM(temp, SPEX_MPQ_DEN(x[i]), temp);
    }
    SPEX_MPQ_SET_Z(scale,temp);

    for (int64_t i = 0; i < n; i++)
    {
        // x_out[i] = x[i]*temp
        SPEX_MPZ_DIVEXACT(x_out[i], temp, SPEX_MPQ_DEN(x[i]));
        SPEX_MPZ_MUL(x_out[i], x_out[i], SPEX_MPQ_NUM(x[i]));
    }
    SPEX_FREE_ALL;
    return SPEX_OK;
}

