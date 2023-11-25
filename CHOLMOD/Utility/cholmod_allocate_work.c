//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_allocate_work: alloc workspace (double, int32)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int cholmod_allocate_work
(
    // input:
    size_t nrow,
    size_t iworksize,
    size_t xworksize,
    cholmod_common *Common
)
{
    return (cholmod_alloc_work (nrow, iworksize, xworksize, CHOLMOD_DOUBLE,
        Common)) ;
}

