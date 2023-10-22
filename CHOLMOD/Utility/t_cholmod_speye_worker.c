//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_speye_worker: sparse identity matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_speye_worker)
(
    cholmod_sparse *A
)
{

    //--------------------------------------------------------------------------
    // fill the matrix with all 1's on the diagonal
    //--------------------------------------------------------------------------

    Int  *Ap = (Int  *) A->p ;
    Int  *Ai = (Int  *) A->i ;
    Real *Ax = (Real *) A->x ;
    Real *Az = (Real *) A->z ;

    Int ncol = (Int) A->ncol ;
    Int nrow = (Int) A->nrow ;
    Int n = MIN (nrow, ncol) ;

    Real onex [2] = {1,0} ;
    Real onez [1] = {0} ;

    for (Int k = 0 ; k < n ; k++)
    {
        // A(k,k) = 1
        Ap [k] = k ;
        Ai [k] = k ;
        ASSIGN (Ax, Az, k, onex, onez, 0) ;
    }

    //--------------------------------------------------------------------------
    // finish the rest of A->p for any remaining empty columns
    //--------------------------------------------------------------------------

    for (Int k = n ; k <= ncol ; k++)
    {
	Ap [k] = n ;
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

