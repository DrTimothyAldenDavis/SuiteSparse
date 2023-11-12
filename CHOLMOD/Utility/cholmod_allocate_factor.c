//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_allocate_factor: allocate a simplicial factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// For backward compatibilty; L is returned as double precision.
// Use cholmod_alloc_factor to allocate a single precision factor.

#define CHOLMOD_INT32
#include "cholmod_internal.h"

cholmod_factor *cholmod_allocate_factor         // return the new factor L
(
    // input:
    size_t n,               // L is factorization of an n-by-n matrix
    cholmod_common *Common
)
{
    return (cholmod_alloc_factor (n, CHOLMOD_DOUBLE, Common)) ;
}

