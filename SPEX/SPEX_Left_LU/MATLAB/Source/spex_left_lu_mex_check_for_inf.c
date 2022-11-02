// //------------------------------------------------------------------------------
// SPEX_Left_LU/MATLAB/spex_left_lu_mex_check_for_inf: Check A and b for inf
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Purpose: This function checks if the array x contains Inf's, NaN's, or
// if its values can be represented as int64_t values, useful for input arguments.

#include "SPEX_Left_LU_mex.h"

bool spex_left_lu_mex_check_for_inf     // true if x can be represented as int64_t
(
    double* x, // The array of numeric values
    mwSize n   // size of array
)
{
    
    bool x_is_int64 = true ;

    for (mwSize k = 0; k < n; k++)
    {
        double xk = x [k] ;

        if (mxIsInf (xk))
        {
            spex_left_lu_mex_error (1, "A must not have any Inf values") ;
        }

        if (mxIsNaN (xk))
        {
            spex_left_lu_mex_error (1, "A must not have any NaN values") ;
        }

        if (x_is_int64)
        {
            if (xk < INT64_MIN || xk > INT64_MAX)
            {
                x_is_int64 = false ;
            }
            else
            {
                int64_t xi = (int64_t) xk ;
                if ((double) xi != xk)
                {
                    x_is_int64 = false ;
                }
            }
        }
    }

    return (x_is_int64) ;
}

