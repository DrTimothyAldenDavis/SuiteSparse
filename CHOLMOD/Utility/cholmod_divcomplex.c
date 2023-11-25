//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_divcomplex: complex divide
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// Compute c = a/b

int cholmod_divcomplex
(
    // input:
    double ar, double ai,       // a (real, imaginary)
    double br, double bi,       // b (real, imaginary)
    double *cr, double *ci      // c (real, imaginary)
)
{
    return (SuiteSparse_config_divcomplex (ar, ai, br, bi, cr, ci)) ;
}

