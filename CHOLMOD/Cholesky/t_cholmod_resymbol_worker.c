//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/t_cholmod_resymbol_worker: recompute symbolic pattern of L
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_resymbol_worker)
(
    cholmod_sparse *A,
    bool pack,
    cholmod_factor *L,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int nrow = A->nrow ;

    Int *Ai = A->i ;
    Int *Ap = A->p ;
    Int *Anz = A->nz ;
    bool apacked = A->packed ;
    bool sorted = A->sorted ;
    int stype = A->stype ;

    Int  *Li = L->i ;
    Real *Lx = L->x ;
    Real *Lz = L->z ;
    Int  *Lp = L->p ;
    Int  *Lnz = L->nz ;

    Int *Flag  = Common->Flag ;      // size nrow
    Int *Head  = Common->Head ;      // size nrow+1
    Int *Iwork = Common->Iwork ;
    Int *Link  = Iwork ;             // size nrow
    Int *Anext = Iwork + nrow ;      // size ncol, unsym. only

    Int pdest = 0 ;

    //--------------------------------------------------------------------------
    // resymbol
    //--------------------------------------------------------------------------

    for (Int k = 0 ; k < nrow ; k++)
    {

        #ifndef NDEBUG
        PRINT1 (("\n\n================== Initial column k = "ID"\n", k)) ;
        for (Int p = Lp [k] ; p < Lp [k] + Lnz [k] ; p++)
        {
            PRINT1 ((" row: "ID"  value: ", Li [p])) ;
            PRINT1 (("\n")) ;
        }
        PRINT1 (("Recomputing LDL, column k = "ID"\n", k)) ;
        #endif

        //----------------------------------------------------------------------
        // compute column k of I+F*F' or I+A
        //----------------------------------------------------------------------

        // flag the diagonal entry
        CLEAR_FLAG (Common) ;
        Int mark = Common->mark ;

        Flag [k] = mark ;
        PRINT1 (("row: "ID" (diagonal)\n", k)) ;

        if (stype != 0)
        {
            // merge column k of A into Flag (lower triangular part only)
            Int p = Ap [k] ;
            Int pend = (apacked) ? (Ap [k+1]) : (p + Anz [k]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                if (i > k)
                {
                    Flag [i] = mark ;
                }
            }
        }
        else
        {
            // for each column j whos first row index is in row k
            for (Int j = Head [k] ; j != EMPTY ; j = Anext [j])
            {
                // merge column j of A into Flag
                PRINT1 (("  ---- A column "ID"\n", j)) ;
                Int p = Ap [j] ;
                Int pend = (apacked) ? (Ap [j+1]) : (p + Anz [j]) ;
                PRINT1 (("  length "ID"  adding\n", pend-p)) ;
                for ( ; p < pend ; p++)
                {
                    #ifndef NDEBUG
                    ASSERT (Ai [p] >= k && Ai [p] < nrow) ;
                    if (Flag [Ai [p]] < mark) PRINT1 ((" row "ID"\n", Ai [p])) ;
                    #endif
                    Flag [Ai [p]] = mark ;
                }
            }
            // clear the kth link list
            Head [k] = EMPTY ;
        }

        //----------------------------------------------------------------------
        // compute pruned pattern of kth column of L = union of children
        //----------------------------------------------------------------------

        // for each column j of L whose parent is k
        for (Int j = Link [k] ; j != EMPTY ; j = Link [j])
        {
            // merge column j of L into Flag
            PRINT1 (("   ---- L column "ID"\n", k)) ;
            ASSERT (j < k) ;
            ASSERT (Lnz [j] > 0) ;
            Int p = Lp [j] ;
            Int pend = p + Lnz [j] ;
            ASSERT (Li [p] == j && Li [p+1] == k) ;
            p++ ;       // skip past the diagonal entry
            for ( ; p < pend ; p++)
            {
                // add to pattern
                ASSERT (Li [p] >= k && Li [p] < nrow) ;
                Flag [Li [p]] = mark ;
            }
        }

        //----------------------------------------------------------------------
        // prune the kth column of L
        //----------------------------------------------------------------------

        PRINT1 (("Final column of L:\n")) ;

        Int p = Lp [k] ;
        Int pend = p + Lnz [k] ;

        if (pack)
        {
            // shift column k upwards
            Lp [k] = pdest ;
        }
        else
        {
            // leave column k in place, just reduce Lnz [k]
            pdest = p ;
        }

        for ( ; p < pend ; p++)
        {
            ASSERT (pdest < pend) ;
            ASSERT (pdest <= p) ;
            Int row = Li [p] ;
            ASSERT (row >= k && row < nrow) ;
            if (Flag [row] == mark)
            {
                // keep this entry
                Li [pdest] = row ;
                // Lx,Lz [pdest] = Lx,Lz [p]
                ASSIGN (Lx, Lz, pdest, Lx, Lz, p) ;
                pdest++ ;
            }
        }

        //----------------------------------------------------------------------
        // prepare this column for its parent
        //----------------------------------------------------------------------

        Lnz [k] = pdest - Lp [k] ;

        PRINT1 ((" L("ID") length "ID"\n", k, Lnz [k])) ;
        ASSERT (Lnz [k] > 0) ;

        // parent is the first entry in the column after the diagonal
        Int parent = (Lnz [k] > 1) ? (Li [Lp [k] + 1]) : EMPTY ;

        PRINT1 (("parent ("ID") = "ID"\n", k, parent)) ;
        ASSERT ((parent > k && parent < nrow) || (parent == EMPTY)) ;

        if (parent != EMPTY)
        {
            Link [k] = Link [parent] ;
            Link [parent] = k ;
        }
    }

    if (pack)
    {
        // finalize Lp
        Lp [nrow] = pdest ;
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

