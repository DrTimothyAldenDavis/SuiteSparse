//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_reallocate_triplet: reallocate triplet matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Change the max # of nonzeros that can be held in a triplet matrix T.

#include "cholmod_internal.h"

int CHOLMOD(reallocate_triplet)
(
    // input:
    size_t nznew,       // new max # of nonzeros the triplet matrix can hold
    // input/output:
    cholmod_triplet *T, // triplet matrix to reallocate
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (T, FALSE) ;
    RETURN_IF_XTYPE_IS_INVALID (T->xtype, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX,
        FALSE) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // reallocate the triplet matrix
    //--------------------------------------------------------------------------

    nznew = MAX (1, nznew) ;    // ensure T can hold at least 1 entry
    int nint = 2 ;              // reallocate both T->i and T->j
    CHOLMOD(realloc_multiple) (nznew, nint, T->xtype + T->dtype,
        &(T->i), &(T->j), &(T->x), &(T->z), &(T->nzmax), Common) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (Common->status == CHOLMOD_OK) ;
}

