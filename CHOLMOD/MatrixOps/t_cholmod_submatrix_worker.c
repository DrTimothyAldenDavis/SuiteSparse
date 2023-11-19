//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/t_cholmod_submatrix_worker
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_submatrix_worker)
(
    cholmod_sparse *C,
    cholmod_sparse *A,
    Int nr,
    Int nc,
    Int *cset,          // set of column indices, duplicates OK
    Int *Head,
    Int *Rnext
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ap = A->p ;
    Int *Ai = A->i ;
    Int *Anz = A->nz ;
    Real *Ax = A->x ;
    Real *Az = A->z ;
    bool packed = A->packed ;

    Int *Cp = C->p ;
    Int *Ci = C->i ;
    Real *Cx = C->x ;
    Real *Cz = C->z ;
    Int cncol = C->ncol ;

    //--------------------------------------------------------------------------
    // C = A (rset, cset)
    //--------------------------------------------------------------------------

    Int pc = 0 ;

    if (nr < 0)
    {

        //----------------------------------------------------------------------
        // C = A (:,cset), where cset is not empty
        //----------------------------------------------------------------------

        for (Int cj = 0 ; cj < cncol ; cj++)
        {
            // construct column cj of C, which is column j of A
            PRINT1 (("construct cj = j = "ID"\n", cj)) ;
            Int j = cset [cj] ;
            Cp [cj] = pc ;
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Ci [pc] = Ai [p] ;
                ASSIGN (Cx, Cz, pc, Ax, Az, p) ;
                pc++ ;
                ASSERT (pc <= C->nzmax) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C = A (rset,cset), where rset is not empty but cset might be empty
        //----------------------------------------------------------------------

        for (Int cj = 0 ; cj < cncol ; cj++)
        {
            // construct column cj of C, which is column j of A
            PRINT1 (("construct cj = "ID"\n", cj)) ;
            Int j = (nc < 0) ? cj : (cset [cj]) ;
            PRINT1 (("cj = "ID"\n", j)) ;
            Cp [cj] = pc ;
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                // row (Ai [p]) of A becomes multiple rows (ci) of C
                PRINT2 (("i: "ID" becomes: ", Ai [p])) ;
                for (Int ci = Head [Ai [p]] ; ci != EMPTY ; ci = Rnext [ci])
                {
                    PRINT3 ((""ID" ", ci)) ;
                    Ci [pc] = ci ;
                    ASSIGN (Cx, Cz, pc, Ax, Az, p) ;
                    pc++ ;
                    ASSERT (pc <= C->nzmax) ;
                }
                PRINT2 (("\n")) ;
            }
        }
    }

    // finalize the last column of C
    Cp [cncol] = pc ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

