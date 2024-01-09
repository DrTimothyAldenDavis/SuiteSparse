//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_reallocate_sparse: reallocate sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Change the max # of nonzeros that can be held in a sparse matrix A.

#include "cholmod_internal.h"

int CHOLMOD(reallocate_sparse)
(
    // input:
    size_t nznew,       // new max # of nonzeros the sparse matrix can hold
    // input/output:
    cholmod_sparse *A,  // sparse matrix to reallocate
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_XTYPE_IS_INVALID (A->xtype, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX,
        FALSE) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // reallocate the sparse matrix
    //--------------------------------------------------------------------------

    nznew = MAX (1, nznew) ;    // ensure A can hold at least 1 entry
    int nint = 1 ;              // reallocate just A->i
    CHOLMOD(realloc_multiple) (nznew, nint, A->xtype + A->dtype,
        &(A->i), NULL, &(A->x), &(A->z), &(A->nzmax), Common) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (Common->status == CHOLMOD_OK) ;
}

