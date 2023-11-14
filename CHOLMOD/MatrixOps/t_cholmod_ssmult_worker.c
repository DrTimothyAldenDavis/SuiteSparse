//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/t_cholmod_ssmult_worker: sparse-times-sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_ssmult_worker)
(
    cholmod_sparse *C,
    cholmod_sparse *A,
    cholmod_sparse *B,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *W = Common->Xwork ;

    Int *Ap  = A->p ;
    Int *Anz = A->nz ;
    Int *Ai  = A->i ;
    Real *Ax  = A->x ;
    Real *Az  = A->z ;
    bool apacked = A->packed ;

    Int *Bp  = B->p ;
    Int *Bnz = B->nz ;
    Int *Bi  = B->i ;
    Real *Bx  = B->x ;
    Real *Bz  = B->z ;
    bool bpacked = B->packed ;

    // get the size of C
    Int nrow = A->nrow ;
    Int ncol = B->ncol ;

    // get workspace
    Real *Wx = Common->Xwork ;  // size nrow, unused if C is pattern
    Real *Wz = Wx + nrow ;      // only used for the zomplex case
    Int *Flag = Common->Flag ;  // size nrow, Flag [0..nrow-1] < mark on input

    Int *Cp = C->p ;
    Int *Ci = C->i ;
    Real *Cx = C->x ;
    Real *Cz = C->z ;

    //--------------------------------------------------------------------------
    // C = A*B
    //--------------------------------------------------------------------------

    Int pc = 0 ;

    for (Int j = 0 ; j < ncol ; j++)
    {
        // clear the Flag array
        CLEAR_FLAG (Common) ;
        Int mark = Common->mark ;

        // start column j of C
        Cp [j] = pc ;

        // for each nonzero B(k,j) in column j, do:
        Int pb = Bp [j] ;
        Int pbend = (bpacked) ? (Bp [j+1]) : (pb + Bnz [j]) ;
        for ( ; pb < pbend ; pb++)
        {
            // B(k,j) is nonzero
            Int k = Bi [pb] ;

            // b = Bx [pb] ;
            Real bx [2] ;
            Real bz [1] ;
            ASSIGN (bx, bz, 0, Bx, Bz, pb) ;

            // add the nonzero pattern of A(:,k) to the pattern of C(:,j)
            // and scatter the values into W
            Int pa = Ap [k] ;
            Int paend = (apacked) ? (Ap [k+1]) : (pa + Anz [k]) ;
            for ( ; pa < paend ; pa++)
            {
                Int i = Ai [pa] ;
                if (Flag [i] != mark)
                {
                    Flag [i] = mark ;
                    Ci [pc++] = i ;
                }
                // W (i) += Ax [pa] * b ;
                MULTADD (Wx, Wz, i, Ax, Az, pa, bx, bz, 0) ;
            }
        }

        // gather the values into C(:,j)
        #ifndef PATTERN
        for (Int p = Cp [j] ; p < pc ; p++)
        {
            Int i = Ci [p] ;
            // Cx [p] = W (i) ;
            ASSIGN (Cx, Cz, p, Wx, Wz, i) ;
            // W (i) = 0 ;
            CLEAR (Wx, Wz, i) ;
        }
        #endif
    }

    Cp [ncol] = pc ;
    ASSERT (MAX (1,pc) == C->nzmax) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

