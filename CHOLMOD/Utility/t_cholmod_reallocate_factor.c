//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_reallocate_factor: reallocate a factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Change the max # of nonzeros that can be held in a factor L.
// The factor must be simplicial.

#include "cholmod_internal.h"

int CHOLMOD(reallocate_factor)
(
    // input:
    size_t nznew,       // new max # of nonzeros the factor matrix can hold
    // input/output:
    cholmod_factor *L,  // factor to reallocate
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_XTYPE_IS_INVALID (L->xtype, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX,
        FALSE) ;
    if (L->is_super)
    {
        ERROR (CHOLMOD_INVALID, "L invalid") ;
        return (FALSE) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // reallocate the sparse matrix
    //--------------------------------------------------------------------------

    nznew = MAX (1, nznew) ;    // ensure L can hold at least 1 entry
    int nint = 1 ;              // reallocate just L->i

    CHOLMOD(realloc_multiple) (nznew, nint, L->xtype + L->dtype,
        &(L->i), NULL, &(L->x), &(L->z), &(L->nzmax), Common) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (Common->status == CHOLMOD_OK) ;
}

