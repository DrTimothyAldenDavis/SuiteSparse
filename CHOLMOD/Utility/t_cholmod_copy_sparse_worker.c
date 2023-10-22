//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy_sparse_worker: copy a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_copy_sparse_worker)
(
    cholmod_sparse *C,     // output sparse matrix
    cholmod_sparse *A      // sparse matrix to copy
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // This worker is only used for the unpacked case
    ASSERT (!(A->packed)) ;

    //--------------------------------------------------------------------------
    // get the A and C matrices
    //--------------------------------------------------------------------------

    Int  *Ap  = (Int  *) A->p ;
    Int  *Anz = (Int  *) A->nz ;
    Int  *Ai  = (Int  *) A->i ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;
    size_t ncol = A->ncol ;

    Int  *Ci  = (Int  *) C->i ;
    Real *Cx  = (Real *) C->x ;
    Real *Cz  = (Real *) C->z ;

    //--------------------------------------------------------------------------
    // copy the contents from A to C
    //--------------------------------------------------------------------------

    for (Int j = 0 ; j < ncol ; j++)
    {
        Int p = Ap [j] ;
        Int pend = p + Anz [j] ;
        for ( ; p < pend ; p++)
        {
            // C(i,j) = A (i,j)
            Ci [p] = Ai [p] ;
            ASSIGN (Cx, Cz, p, Ax, Az, p) ;
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

