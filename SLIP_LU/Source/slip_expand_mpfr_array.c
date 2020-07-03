//------------------------------------------------------------------------------
// SLIP_LU/slip_expand_mpfr_array: convert mpfr aray to mpz
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function converts a mpfr array of size n and precision prec to
 * an appropriate mpz array of size n. To do this, the number is multiplied by
 * the appropriate power of 10 then the gcd is found. This function allows mpfr
 * arrays to be used within SLIP LU.
 */

#define SLIP_FREE_ALL               \
    SLIP_MPFR_CLEAR(expon);         \
    SLIP_MPZ_CLEAR(temp_expon);     \
    SLIP_MPZ_CLEAR(gcd);            \
    SLIP_MPZ_CLEAR(one);            \
    SLIP_MPQ_CLEAR(temp);           \
    SLIP_matrix_free(&x3, NULL);    \

#include "slip_internal.h"

SLIP_info slip_expand_mpfr_array
(
    mpz_t* x_out,         // full precision mpz array
    mpfr_t* x,            // mpfr array to be expanded
    mpq_t scale,          // scaling factor used (x_out = scale*x)
    int64_t n,            // size of x
    const SLIP_options *option  // command options containing the prec
                          // and rounding for mpfr
)
{

    //--------------------------------------------------------------------------
    // Input has already been checked
    //--------------------------------------------------------------------------

    SLIP_info info ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t i, k ;
    int r1, r2 = 1 ;
    bool nz_found = false;
    mpfr_t expon; SLIP_MPFR_SET_NULL(expon);
    mpz_t temp_expon, gcd, one;
    SLIP_matrix* x3 = NULL;
    SLIP_MPZ_SET_NULL(temp_expon);
    SLIP_MPZ_SET_NULL(gcd);
    SLIP_MPZ_SET_NULL(one);
    mpq_t temp; SLIP_MPQ_SET_NULL(temp);

    uint64_t prec = SLIP_OPTION_PREC (option) ;
    mpfr_rnd_t round = SLIP_OPTION_ROUND (option) ;

    SLIP_CHECK(SLIP_mpq_init(temp));
    SLIP_CHECK(SLIP_mpfr_init2(expon, prec));
    SLIP_CHECK(SLIP_mpz_init(temp_expon));
    SLIP_CHECK(SLIP_mpz_init(gcd));
    SLIP_CHECK(SLIP_mpz_init(one));

    SLIP_CHECK (SLIP_matrix_allocate(&x3, SLIP_DENSE, SLIP_MPFR, n, 1, n,
        false, true, option));

    // expon = 2^prec (overestimate)
    SLIP_CHECK(SLIP_mpfr_ui_pow_ui(expon, 2, prec, round)) ;

    for (i = 0; i < n; i++)
    {
        // x3[i] = x[i]*2^prec
        SLIP_CHECK(SLIP_mpfr_mul(x3->x.mpfr[i], x[i], expon, round));

        // x_out[i] = x3[i]
        SLIP_CHECK(SLIP_mpfr_get_z(x_out[i], x3->x.mpfr[i], round));
    }
    SLIP_CHECK(SLIP_mpfr_get_z(temp_expon, expon, round));
    SLIP_CHECK(SLIP_mpq_set_z(scale, temp_expon));

    //--------------------------------------------------------------------------
    // Find the gcd to reduce scale
    //--------------------------------------------------------------------------

    SLIP_CHECK(SLIP_mpz_set_ui(one, 1));
    // Find an initial GCD
    for (i = 0; i < n; i++)
    {
        if (!nz_found)
        {
            SLIP_CHECK(SLIP_mpz_cmp_ui(&r1, x_out[i], 0));
            if (r1 != 0)
            {
                nz_found = true;
                k = i;
                SLIP_CHECK(SLIP_mpz_set(gcd, x_out[i]));
            }
        }
        else
        {
            // Compute the GCD of the numbers, stop if gcd == 1
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

    if (r2 != 0)  // If gcd == 1 stop
    {
        for (i = k; i < n; i++)
        {
            SLIP_CHECK(SLIP_mpz_divexact(x_out[i],x_out[i],gcd));
        }
        SLIP_CHECK(SLIP_mpq_set_z(temp,gcd));
        SLIP_CHECK(SLIP_mpq_div(scale,scale,temp));
    }
    SLIP_FREE_ALL;
    return SLIP_OK;
}

