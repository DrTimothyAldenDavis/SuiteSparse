//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_l_allocate_factor: allocate a simplicial factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// For backward compatibilty; L is returned as double precision.
// Use cholmod_l_alloc_factor to allocate a single precision factor.

#define CHOLMOD_INT64
#include "cholmod_internal.h"

cholmod_factor *cholmod_l_allocate_factor       // return the new factor L
(
    // input:
    size_t n,               // L is factorization of an n-by-n matrix
    cholmod_common *Common
)
{
    return (cholmod_l_alloc_factor (n, CHOLMOD_DOUBLE, Common)) ;
}

