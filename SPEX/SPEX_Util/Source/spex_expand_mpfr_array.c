//------------------------------------------------------------------------------
// SPEX_Util/spex_expand_mpfr_array: convert mpfr aray to mpz
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function converts a mpfr array of size n and precision prec to
 * an appropriate mpz array of size n. To do this, the number is cast to an 
 * equivalent mpq_t (rational number) a conversion which MPFR gaurantees to be
 * exact. Then, this rational array is converted to an mpz_t array
 * This function allows mpfr arrays to be used within SPEX.
 */

#define SPEX_FREE_ALL               \
    SPEX_matrix_free(&x3, NULL);    \

#include "spex_util_internal.h"

SPEX_info spex_expand_mpfr_array
(
    mpz_t* x_out,         // full precision mpz array
    mpfr_t* x,            // mpfr array to be expanded
    mpq_t scale,          // scaling factor used (x_out = scale*x)
    int64_t n,            // size of x
    const SPEX_options *option  // command options containing the prec
                          // and rounding for mpfr
)
{

    //--------------------------------------------------------------------------
    // Input has already been checked
    //--------------------------------------------------------------------------
    ASSERT(n >= 0);
    SPEX_info info ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t i;
    SPEX_matrix* x3 = NULL;
    mpfr_rnd_t round = SPEX_OPTION_ROUND (option) ;
    
    SPEX_CHECK (SPEX_matrix_allocate(&x3, SPEX_DENSE, SPEX_MPQ, n, 1, n,
        false, true, option));

    // Cast the mpfr array to a rational array
    for (i = 0; i < n; i++)
    {
        // x3[i] = x[i]
        SPEX_CHECK(SPEX_mpfr_get_q(x3->x.mpq[i],x[i],round));
    }
    
    // Expand the mpq array
    SPEX_CHECK( spex_expand_mpq_array( x_out, x3->x.mpq, scale, n, option));

    // Free memory
    SPEX_FREE_ALL;
    return SPEX_OK;
}

