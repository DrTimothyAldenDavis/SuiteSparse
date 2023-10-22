//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_sparse_to_triplet_worker
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_sparse_to_triplet_worker)
(
    cholmod_triplet *T,
    cholmod_sparse *A
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int nrow = A->nrow ;
    Int ncol = A->ncol ;
    bool packed = (bool) A->packed ;
    Int  *Ap = (Int  *) A->p ;
    Int  *Ai = (Int  *) A->i ;
    Real *Ax = (Real *) A->x ;
    Real *Az = (Real *) A->z ;
    Int *Anz = (Int  *) A->nz ;

    Int  *Ti = (Int  *) T->i ;
    Int  *Tj = (Int  *) T->j ;
    Real *Tx = (Real *) T->x ;
    Real *Tz = (Real *) T->z ;

    Int k = 0 ;

    //--------------------------------------------------------------------------
    // copy entries from A into T
    //--------------------------------------------------------------------------

    #define COPY_ENTRY(entry_test)              \
        if (entry_test)                         \
        {                                       \
            Ti [k] = i ;                        \
            Tj [k] = j ;                        \
            /* Tx (k) = Ax (p) */               \
            ASSIGN (Tx, Tz, k, Ax, Az, p) ;     \
            k++ ;                               \
        }

    if (A->stype == 0)
    {

        //----------------------------------------------------------------------
        // A is unsymmetric
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                COPY_ENTRY (true) ;
            }
        }

    }
    else if (A->stype > 0)
    {

        //----------------------------------------------------------------------
        // A is symmetric, with just upper triangular part stored
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                COPY_ENTRY (i <= j) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A is symmetric, with just lower triangular part stored
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                COPY_ENTRY (i >= j) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // log the number of entries in T
    //--------------------------------------------------------------------------

    T->nnz = k ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

