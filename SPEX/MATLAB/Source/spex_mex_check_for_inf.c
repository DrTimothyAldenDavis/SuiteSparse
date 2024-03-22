//------------------------------------------------------------------------------
// SPEX/MATLAB/SPEX_mex_check_for_inf.c: Check A&B for inf/NAN
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function checks if the array x contains Inf's, NaN's, or
 * if its values can be represented as int64_t values, useful for input 
 * arguments.
 */

#include "SPEX_mex.h"

bool spex_mex_check_for_inf     // true if x can be represented as int64_t
(
    double *x, // The array of numeric values
    mwSize n   // size of array
)
{
    // Assume it is int64
    bool x_is_int64 = true ;

    // Iterate through all entries
    for (mwSize k = 0; k < n; k++)
    {
        double xk = x [k] ;

        // Check if inf
        if (mxIsInf (xk))
        {
            spex_mex_error (1, "A must not have any Inf values");
        }

        // Check if NAN
        if (mxIsNaN (xk))
        {
            spex_mex_error (1, "A must not have any NaN values");
        }

        // Check if int64
        if (x_is_int64)
        {
            if (xk < INT64_MIN || xk > INT64_MAX)
            {
                x_is_int64 = false ;
                break;
            }
            else
            {
                int64_t xi = (int64_t) xk ;
                if ((double) xi != xk)
                {
                    x_is_int64 = false ;
                    break;
                }
            }
        }
    }

    return (x_is_int64);
}

