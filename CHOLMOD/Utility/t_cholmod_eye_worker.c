//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_eye_worker: dense identity matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_eye_worker)
(
    cholmod_dense *X
)
{

    //--------------------------------------------------------------------------
    // fill the matrix with all 1's on the diagonal
    //--------------------------------------------------------------------------

    Real *Xx = (Real *) X->x ;
    Real *Xz = (Real *) X->z ;

    Int nrow = (Int) X->nrow ;
    Int ncol = (Int) X->ncol ;
    Int n = MIN (nrow, ncol) ;

    Real onex [2] = {1,0} ;
    Real onez [1] = {0} ;

    for (Int k = 0 ; k < n ; k++)
    {
        // X(k,k) = 1
        ASSIGN (Xx, Xz, k + k*nrow, onex, onez, 0) ;
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

