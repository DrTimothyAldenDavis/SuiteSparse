//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_mult_size_t: multiply two size_t values
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// multiply two size_t values and safely check for integer overflow

size_t CHOLMOD(mult_size_t) (size_t a, size_t b, int *ok)
{
    uint64_t x ;
    (*ok) = (*ok) && cholmod_mult_uint64_t (&x, (uint64_t) a, (uint64_t) b) ;
    return (((*ok) && x <= SIZE_MAX) ? ((size_t) x) : 0) ;
}

