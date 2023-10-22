//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_ones_worker: dense matrix of all ones
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_ones_worker)
(
    cholmod_dense *X
)
{

    //--------------------------------------------------------------------------
    // fill the matrix with all 1's
    //--------------------------------------------------------------------------

    Real *Xx = (Real *) X->x ;
    Real *Xz = (Real *) X->z ;
    size_t nzmax = X->nzmax ;

    Real onex [2] = {1,0} ;
    Real onez [1] = {0} ;

    for (Int k = 0 ; k < nzmax ; k++)
    {
        // X(k) = 1
        ASSIGN (Xx, Xz, k, onex, onez, 0) ;
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

