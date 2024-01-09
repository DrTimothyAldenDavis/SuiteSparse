//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_set_empty: set an Int array to EMPTY
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

void CHOLMOD(set_empty)
(
    Int *S,     // Int array of size n
    size_t n
)
{
    for (size_t k = 0 ; k < n ; k++)
    {
        S [k] = EMPTY ;
    }
}

