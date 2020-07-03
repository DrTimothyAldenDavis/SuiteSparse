//------------------------------------------------------------------------------
// SLIP_LU/slip_expand_double_array: convert double vector to mpz
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function converts a double array of size n to an appropriate
 * mpz array of size n. To do this, the number is multiplied by an appropriate
 * power, then, the GCD is found. This function allows the use of matrices in
 * double precision to work with SLIP LU.
 *
 */

#define SLIP_FREE_ALL               \
    SLIP_MPZ_CLEAR(gcd);            \
    SLIP_MPZ_CLEAR(one);            \
    SLIP_MPQ_CLEAR(temp);           \
    SLIP_matrix_free(&x3, NULL);    \

#include "slip_internal.h"

SLIP_info slip_expand_double_array
(
    mpz_t* x_out,           // integral final array
    double* x,              // double array that needs to be made integral
    mpq_t scale,            // the scaling factor used (x_out = scale * x)
    int64_t n,              // size of x
    const SLIP_options* option // Command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // inputs have been checked in the only caller slip_cast_array

    //--------------------------------------------------------------------------

    int64_t i, k ;
    int r1, r2 = 1;
    bool nz_found = false;
    SLIP_info info ;
    // Double precision accurate to about 2e-16. We multiply by 10e17 to convert
    // (overestimate to be safe)
    double expon = pow(10, 17);
    // Quad precision in case input is huge
    SLIP_matrix* x3 = NULL;
    mpz_t gcd, one; SLIP_MPZ_SET_NULL(gcd); SLIP_MPZ_SET_NULL(one);
    mpq_t temp; SLIP_MPQ_SET_NULL(temp);

    mpfr_rnd_t round = SLIP_OPTION_ROUND (option) ;

    SLIP_CHECK(SLIP_mpq_init(temp));
    SLIP_CHECK(SLIP_mpz_init(gcd));
    SLIP_CHECK(SLIP_mpz_init(one));

    SLIP_CHECK (SLIP_matrix_allocate(&x3, SLIP_DENSE, SLIP_MPFR, n, 1, n,
        false, true, option));

    SLIP_CHECK(SLIP_mpq_set_d(scale, expon));           // scale = 10^17
    for (i = 0; i < n; i++)
    {
        // Set x3[i] = x[i]
        SLIP_CHECK(SLIP_mpfr_set_d(x3->x.mpfr[i], x[i], round));

        // x3[i] = x[i] * 10^17
        SLIP_CHECK(SLIP_mpfr_mul_d(x3->x.mpfr[i], x3->x.mpfr[i], expon, round));

        // x_out[i] = x3[i]
        SLIP_CHECK(SLIP_mpfr_get_z(x_out[i], x3->x.mpfr[i], round));
    }

    //--------------------------------------------------------------------------
    // Compute the GCD to reduce the size of scale
    //--------------------------------------------------------------------------

    SLIP_CHECK(SLIP_mpz_set_ui(one, 1));
    // Find an initial GCD
    for (i = 0; i < n; i++)
    {
        if (!nz_found)
        {
            SLIP_CHECK(SLIP_mpz_cmp_ui(&r1, x_out[i], 0)); // Check if x[i] == 0
            if (r1 != 0)
            {
                nz_found = true;
                k = i;
                SLIP_CHECK(SLIP_mpz_set(gcd, x_out[i]));
            }
        }
        else
        {
            // Compute the GCD, stop if gcd == 1
            SLIP_CHECK(SLIP_mpz_gcd(gcd, gcd, x_out[i]));
            SLIP_CHECK(SLIP_mpz_cmp(&r2, gcd, one));
            if (r2 == 0)
            {
                break;
            }
        }
    }

    if (!nz_found)     // Array is all zeros
    {
        SLIP_FREE_ALL;
        SLIP_mpq_set_z(scale, one);
        return SLIP_OK;
    }

    //--------------------------------------------------------------------------
    // Scale all entries to make as small as possible
    //--------------------------------------------------------------------------

    if (r2 != 0)             // If gcd == 1 then stop
    {
        for (i = k; i < n; i++)
        {
            SLIP_CHECK(SLIP_mpz_divexact(x_out[i], x_out[i], gcd));
        }
        SLIP_CHECK(SLIP_mpq_set_z(temp, gcd));
        SLIP_CHECK(SLIP_mpq_div(scale, scale, temp));
    }
    SLIP_FREE_ALL;
    return SLIP_OK;
}

