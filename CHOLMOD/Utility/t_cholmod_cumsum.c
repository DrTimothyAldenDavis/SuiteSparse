//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_cumsum: cumulative sum
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Cp [0:n] = cumulative sum of Cnz [0:n-1].  Returns the total sum, or -1
// if integer overflow has occurred.

// Cp [0] = 0
// Cp [1] = Cnz [0]
// Cp [2] = Cnz [0] + Cnz [1] ;
// Cp [3] = Cnz [0] + Cnz [1] + Cnz [2] ;
// ...
// Cp [k] = sum (Cnz [0 ... k-1])
// ...
// Cp [n] = sum (Cnz [0 ... n-1])

#include "cholmod_internal.h"

int64_t CHOLMOD(cumsum)
(
    Int *Cp,        // size n+1, output array, the cumsum of Cnz
    Int *Cnz,       // size n, input array
    size_t n        // size of Cp and Cnz
)
{
    Int p = 0 ;
    for (Int k = 0 ; k < n ; k++)
    {
        Cp [k] = p ;
        p += Cnz [k] ;
        if (p < 0)
        {
            // integer overflow has occured
            return (EMPTY) ;
        }
    }
    Cp [n] = p ;
    return ((int64_t) p) ;
}

