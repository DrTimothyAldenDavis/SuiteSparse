//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_maxrank: find valid value of Common->maxrank
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// Returns 0 on error, or 2, 4, or 8 otherwise.

size_t CHOLMOD(maxrank)     // return validated Common->maxrank
(
    // input:
    size_t n,               // # of rows of L and A
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (0) ;

    //--------------------------------------------------------------------------
    // determine a valid value of maxrank
    //--------------------------------------------------------------------------

    size_t maxrank = Common->maxrank ;

    if (n > 0)
    {
        // guard against size_t overflow (very unlikely)
        size_t max_maxrank = SIZE_MAX / (n * sizeof (float)) ;
        maxrank = MIN (maxrank, max_maxrank) ;
    }

    // maxrank of 2 or less: use 2
    // maxrank of 3 or 4: use 4
    // else use 8
    return ((maxrank <= 2) ? 2 : ((maxrank <= 4) ? 4 : 8)) ;
}

