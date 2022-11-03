//------------------------------------------------------------------------------
// SPEX_Util/spex_cast_array: scale and typecast an array
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Scales and typecasts an input array X, into the output array Y.

// X: an array of type xtype, of size n.
// Y: an array of type ytype, of size n.

// Note about the scaling factors:
//
// This function copies the scaled values of X into the array Y.
//  If Y is mpz_t, the values in X must be scaled to be integral
//  if they are not already integral. As a result, this function
//  will expand the values and set y_scale = this factor.
//`
//  Conversely, if Y is not mpz_t and X is, we apply X's scaling
//  factor here to get the final values of Y. For instance,
//  if Y is FP64 and X is mpz_t, the values of Y are obtained
//  as Y = X / x_scale.
//
//  The final value of x_scale is not modified.
//  The final value of y_scale is set as follows.
//      If Y is mpz_t, y_scale is set to the appropriate value
//      in order to make all of its entries integral.
//
//      If Y is any other data type, this function always sets
//      y_scale = 1.
//

#define SPEX_FREE_ALL \
SPEX_MPQ_CLEAR(temp);       \

#include "spex_util_internal.h"
#pragma GCC diagnostic ignored "-Wunused-variable"

SPEX_info spex_cast_array
(
    void *Y,                // output array, of size n
    SPEX_type ytype,        // type of Y
    void *X,                // input array, of size n
    SPEX_type xtype,        // type of X
    int64_t n,              // size of Y and X
    mpq_t y_scale,          // scale factor applied if Y is mpz_t
    mpq_t x_scale,          // scale factor applied if x is mpz_t
    const SPEX_options *option// Command options. If NULL, set to default values
)
{

    //--------------------------------------------------------------------------
    // check inputs
    // xtype and ytype are checked in SPEX_matrix_copy
    //--------------------------------------------------------------------------

    if (Y == NULL || X == NULL)
    {
        return (SPEX_INCORRECT_INPUT) ;
    }
    SPEX_info info ;
    int r;
    mpq_t temp; SPEX_MPQ_SET_NULL(temp);

    mpfr_rnd_t round = SPEX_OPTION_ROUND (option) ;

    //--------------------------------------------------------------------------
    // Y [0:n-1] = (ytype) X [0:n-1]
    //--------------------------------------------------------------------------

    switch (ytype)
    {

        //----------------------------------------------------------------------
        // output array Y is mpz_t
        // If X is not mpz_t or int64, the values of X are scaled and y_scale is
        // set to be this scaling factor.
        //----------------------------------------------------------------------

        case SPEX_MPZ:
        {
            mpz_t *y = (mpz_t *) Y ;
            switch (xtype)
            {

                case SPEX_MPZ: // mpz_t to mpz_t
                {
                    mpz_t *x = (mpz_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpz_set (y [k], x[k])) ;
                    }
                    // y is a direct copy of x. Set y_scale = x_scale
                    SPEX_CHECK(SPEX_mpq_set(y_scale, x_scale));
                }
                break ;

                case SPEX_MPQ: // mpq_t to mpz_t
                {
                    mpq_t *x = (mpq_t *) X ;
                    SPEX_CHECK (spex_expand_mpq_array(Y, X, y_scale, n,option));
                }
                break ;

                case SPEX_MPFR: // mpfr_t to mpz_t
                {
                    mpfr_t *x = (mpfr_t *) X ;
                    SPEX_CHECK (spex_expand_mpfr_array (Y, X, y_scale, n,
                        option)) ;
                }
                break ;

                case SPEX_INT64: // int64_t to mpz_t
                {
                    int64_t *x = (int64_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpz_set_si (y [k], x [k])) ;
                    }
                    SPEX_CHECK (SPEX_mpq_set_ui (y_scale, 1, 1)) ;
                }
                break ;

                case SPEX_FP64: // double to mpz_t
                {
                    double *x = (double *) X ;
                    SPEX_CHECK (spex_expand_double_array (y, x, y_scale, n,
                        option)) ;
                }
                break ;

            }
        }
        break ;

        //----------------------------------------------------------------------
        // output array Y is mpq_t
        //----------------------------------------------------------------------

        case SPEX_MPQ:
        {
            mpq_t *y = (mpq_t *) Y ;
            switch (xtype)
            {

                case SPEX_MPZ: // mpz_t to mpq_t
                {
                    // In this case, x is mpz_t and y is mpq_t. the scaling
                    // factor x_scale must be used. If x_scale is not equal to
                    // 1, each value in y is divided by x_scale

                    // Check if x_scale == 1
                    SPEX_CHECK(SPEX_mpq_cmp_ui(&r, x_scale, 1, 1));
                    mpz_t *x = (mpz_t *) X ;

                    if (r == 0)
                    {
                        // x_scale = 1. Simply do a direct copy.
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                            SPEX_CHECK (SPEX_mpq_set_z (y [k], x [k])) ;
                        }
                    }
                    else
                    {
                        // x_scale != 1. In this case, we divide each entry
                        // of Y by x_scale
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                            SPEX_CHECK (SPEX_mpq_set_z (y [k], x [k])) ;
                            SPEX_CHECK (SPEX_mpq_div(y[k], y[k], x_scale));
                        }
                    }
                }
                break ;

                case SPEX_MPQ: // mpq_t to mpq_t
                {
                    mpq_t *x = (mpq_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpq_set (y [k], x [k])) ;
                    }
                }
                break ;

                case SPEX_MPFR: // mpfr_t to mpq_t
                {
                    mpfr_t *x = (mpfr_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpfr_get_q( y[k], x[k], round));
                    }
                }
                break ;

                case SPEX_INT64: // int64 to mpq_t
                {
                    int64_t *x = (int64_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpq_set_si (y [k], x [k], 1)) ;
                    }
                }
                break ;

                case SPEX_FP64: // double to mpq_t
                {
                    double *x = (double *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpq_set_d (y [k], x [k])) ;
                    }
                }
                break ;

            }
        }
        break ;

        //----------------------------------------------------------------------
        // output array Y is mpfr_t
        //----------------------------------------------------------------------

        case SPEX_MPFR:
        {
            mpfr_t *y = (mpfr_t *) Y ;
            switch (xtype)
            {

                case SPEX_MPZ: // mpz_t to mpfr_t
                {
                    // x is mpz_t and y is mpfr_t. Like in the above mpq_t
                    // case, if the scaling factor of x is not equal to 1, the
                    // values of y must be scaled.
                    mpz_t *x = (mpz_t *) X ;
                    SPEX_CHECK(SPEX_mpq_cmp_ui(&r, x_scale, 1, 1));

                    if (r == 0)
                    {
                        // x_scale = 1. Simply do a direct copy.
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                            SPEX_CHECK (SPEX_mpfr_set_z (y [k], x [k], round)) ;
                        }
                    }
                    else
                    {
                        // x_scale != 1. In this case, we divide each entry
                        // of Y by x_scale. To do this, we will cast each
                        // x_k to mpq_t, then divide by the scale, then
                        // cast the result to mpfr_t
                        SPEX_CHECK(SPEX_mpq_init(temp));
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                            SPEX_CHECK( SPEX_mpq_set_z( temp, x[k]));
                            SPEX_CHECK( SPEX_mpq_div(temp, temp, x_scale));
                            SPEX_CHECK(SPEX_mpfr_set_q(y[k], temp, round));
                        }
                    }
                }
                break ;

                case SPEX_MPQ: // mpq_t to mpfr_t
                {
                    mpq_t *x = (mpq_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpfr_set_q (y [k], x [k], round)) ;
                    }
                }
                break ;

                case SPEX_MPFR: // mpfr_t to mpfr_t
                {
                    mpfr_t *x = (mpfr_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpfr_set (y [k], x [k], round)) ;
                    }
                }
                break ;

                case SPEX_INT64: // int64 to mpfr_t
                {
                    int64_t *x = (int64_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK(SPEX_mpfr_set_si(y[k], x[k], round));
                    }
                }
                break ;

                case SPEX_FP64:  // double to mpfr_t
                {
                    double *x = (double *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpfr_set_d (y [k], x [k], round)) ;
                    }
                }
                break ;

            }
        }
        break ;

        //----------------------------------------------------------------------
        // output array Y is int64_t
        //----------------------------------------------------------------------

        case SPEX_INT64:
        {
            int64_t *y = (int64_t *) Y ;
            switch (xtype)
            {

                case SPEX_MPZ: // mpz_t to int64_t
                {
                    // x is mpz_t and y is int64_t. Same as above,
                    // if x_scale > 1 it is applied
                    mpz_t *x = (mpz_t *) X ;
                    SPEX_CHECK(SPEX_mpq_cmp_ui(&r, x_scale, 1, 1));

                    if (r == 0)
                    {
                        // x_scale = 1. Simply do a direct copy.
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                           SPEX_CHECK(SPEX_mpz_get_si( &(y[k]), x[k]));
                        }
                    }
                    else
                    {
                        // x_scale != 1. In this case, we divide each entry
                        // of Y by x_scale. To do this, we will cast each
                        // x_k to mpq_t, then divide by the scale, then
                        // cast the result to double and cast the double to int
                        SPEX_CHECK(SPEX_mpq_init(temp));
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                            SPEX_CHECK( SPEX_mpq_set_z( temp, x[k]));
                            SPEX_CHECK( SPEX_mpq_div(temp, temp, x_scale));
                            double temp2;
                            SPEX_CHECK(SPEX_mpq_get_d(&temp2, temp));
                            y[k] = spex_cast_double_to_int64(temp2);
                        }
                    }
                }
                break ;

                case SPEX_MPQ: // mpq_t to int64_t
                {
                    mpq_t *x = (mpq_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        double t ;
                        SPEX_CHECK (SPEX_mpq_get_d (&t, x [k])) ;
                        y [k] = spex_cast_double_to_int64 (t) ;
                    }
                }
                break ;

                case SPEX_MPFR: // mpfr_t to int64_t
                {
                    mpfr_t *x = (mpfr_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK( SPEX_mpfr_get_si( &(y[k]),x[k], round));
                    }
                }
                break ;

                case SPEX_INT64: // int64_t to int64_t
                {
                    memcpy (Y, X, n * sizeof (int64_t)) ;
                }
                break ;

                case SPEX_FP64: // double to int64_t
                {
                    double *x = (double *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        y [k] = spex_cast_double_to_int64 (x [k]) ;
                    }
                }
                break ;

            }
        }
        break ;

        //----------------------------------------------------------------------
        // output array Y is double
        //----------------------------------------------------------------------

        case SPEX_FP64:
        {
            double *y = (double *) Y ;
            switch (xtype)
            {

                case SPEX_MPZ: // mpz_t to double
                {
                    // Same as above, x is mpz_t, y is double. Must
                    // divide by x_scale if x_scale != 1.
                    mpz_t *x = (mpz_t *) X ;
                    SPEX_CHECK(SPEX_mpq_cmp_ui(&r, x_scale, 1, 1));

                    if (r == 0)
                    {
                        // x_scale = 1. Simply do a direct copy.
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                           SPEX_CHECK(SPEX_mpz_get_d( &(y[k]), x[k]));
                        }
                    }
                    else
                    {
                        // x_scale != 1. In this case, we divide each entry
                        // of Y by x_scale. To do this, we will cast each
                        // x_k to mpq_t, then divide by the scale, then
                        // cast the result to double
                        SPEX_CHECK(SPEX_mpq_init(temp));
                        for (int64_t k = 0 ; k < n ; k++)
                        {
                            SPEX_CHECK( SPEX_mpq_set_z( temp, x[k]));
                            SPEX_CHECK( SPEX_mpq_div(temp, temp, x_scale));
                            SPEX_CHECK(SPEX_mpq_get_d(&(y[k]), temp));
                        }
                    }
                }
                break ;

                case SPEX_MPQ: // mpq_t to double
                {
                    mpq_t *x = (mpq_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpq_get_d (&(y [k]), x [k])) ;
                    }
                }
                break ;

                case SPEX_MPFR: // mpfr_t to double
                {
                    mpfr_t *x = (mpfr_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        SPEX_CHECK (SPEX_mpfr_get_d (&(y [k]), x [k],
                            round));
                    }
                }
                break ;

                case SPEX_INT64: // int64_t to double
                {
                    int64_t *x = (int64_t *) X ;
                    for (int64_t k = 0 ; k < n ; k++)
                    {
                        y [k] = (double) (x [k]) ;
                    }
                }
                break ;

                case SPEX_FP64: // double to double
                {
                    memcpy (Y, X, n * sizeof (double)) ;
                }
                break ;

            }
        }
            break ;

    }
    SPEX_FREE_ALL
    return (SPEX_OK) ;
}
