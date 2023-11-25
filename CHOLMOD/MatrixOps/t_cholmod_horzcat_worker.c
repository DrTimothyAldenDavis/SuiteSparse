//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/t_cholmod_horzcat_worker
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_horzcat_worker)
(
    cholmod_sparse *C,  // output matrix
    cholmod_sparse *A,  // left matrix to concatenate
    cholmod_sparse *B   // right matrix to concatenate
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ap  = A->p ;
    Int *Anz = A->nz ;
    Int *Ai  = A->i ;
    Real *Ax  = A->x ;
    Real *Az  = A->z ;
    bool apacked = A->packed ;
    Int ancol = A->ncol ;

    Int *Bp  = B->p ;
    Int *Bnz = B->nz ;
    Int *Bi  = B->i ;
    Real *Bx  = B->x ;
    Real *Bz  = B->z ;
    bool bpacked = B->packed ;
    Int bncol = B->ncol ;

    Int *Cp = C->p ;
    Int *Ci = C->i ;
    Real *Cx = C->x ;
    Real *Cz = C->z ;
    Int ncol = C->ncol ;

    //--------------------------------------------------------------------------
    // C = [A , B]
    //--------------------------------------------------------------------------

    Int pc = 0 ;

    // copy A as the first A->ncol columns of C
    for (Int j = 0 ; j < ancol ; j++)
    {
        // A(:,j) is the jth column of C
        Int p = Ap [j] ;
        Int pend = (apacked) ? (Ap [j+1]) : (p + Anz [j]) ;
        Cp [j] = pc ;
        for ( ; p < pend ; p++)
        {
            Ci [pc] = Ai [p] ;
            ASSIGN (Cx, Cz, pc, Ax, Az, p) ;
            pc++ ;
        }
    }

    // copy B as the next B->ncol columns of C
    for (Int j = 0 ; j < bncol ; j++)
    {
        // B(:,j) is the (ancol+j)th column of C
        Int p = Bp [j] ;
        Int pend = (bpacked) ? (Bp [j+1]) : (p + Bnz [j]) ;
        Cp [ancol + j] = pc ;
        for ( ; p < pend ; p++)
        {
            Ci [pc] = Bi [p] ;
            ASSIGN (Cx, Cz, pc, Bx, Bz, p) ;
            pc++ ;
        }
    }
    Cp [ncol] = pc ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

