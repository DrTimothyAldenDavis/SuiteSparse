//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense_nnz: # of nonzeros in a dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static int64_t TEMPLATE (cholmod_dense_nnz_worker)
(
    cholmod_dense *X        // input dense matrix
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Xx = (Real *) X->x ;
    Real *Xz = (Real *) X->z ;

    Int nrow = (Int) X->nrow ;
    Int ncol = (Int) X->ncol ;
    Int d = (Int) X->d ;

    //--------------------------------------------------------------------------
    // count # of nonzeros in X
    //--------------------------------------------------------------------------

    int64_t xnz = 0 ;
    for (Int j = 0, jx = 0 ; j < ncol ; j++, jx += d)
    {
        for (Int p = jx ; p < jx + nrow ; p++)
        {
            xnz += ENTRY_IS_NONZERO (Xx, Xz, p) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (xnz) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

