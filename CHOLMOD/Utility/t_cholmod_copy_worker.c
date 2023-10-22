//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy_worker: copy sparse matrix (change of stype)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// A is symmetric upper or lower, and C is unsymmetric.  Both are square.

#include "cholmod_template.h"

static void TEMPLATE (cholmod_copy_worker)
(
    cholmod_sparse *C,  // output matrix
    cholmod_sparse *A,  // input matrix, not modified
    bool ignore_diag,   // if true, ignore diagonal
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    Int n = A->ncol ;
    ASSERT (A->stype != 0) ;
    ASSERT (C->stype == 0) ;
    ASSERT (n == A->ncol) ;
    ASSERT (n == C->nrow) ;
    ASSERT (n == C->ncol) ;
    ASSERT (C->packed) ;
    ASSERT (C->sorted == A->sorted) ;
    ASSERT (C->dtype == A->dtype) ;
    ASSERT (C->xtype == A->xtype || C->xtype == CHOLMOD_PATTERN) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Wj = (Int *) Common->Iwork ;       // size n integer workspace

    Int  *Ap  = (Int  *) A->p ;
    Int  *Anz = (Int  *) A->nz ;
    Int  *Ai  = (Int  *) A->i ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;
    bool packed = A->packed ;

    Int  *Cp = (Int  *) C->p ;
    Int  *Ci = (Int  *) C->i ;
    Real *Cx = (Real *) C->x ;
    Real *Cz = (Real *) C->z ;
    bool keep_diag = !ignore_diag ;

    //--------------------------------------------------------------------------
    // create an unsymmetric matrix C from a symmetric matrix A
    //--------------------------------------------------------------------------

    // place A(i,j) in C(:,k)
    #define ASSIGN_CIJ(i,j)                             \
    {                                                   \
        Int q = Wj [j]++ ;                              \
        ASSIGN (Cx, Cz, q, Ax, Az, p) ;                 \
        Ci [q] = i ;                                    \
    }

    // place conj(A(i,j)) or A(i,j) in C(:,k)
    #define ASSIGN_CIJ_CONJ_OR_NCONJ(i,j)               \
    {                                                   \
        Int q = Wj [i]++ ;                              \
        ASSIGN_CONJ_OR_NCONJ (Cx, Cz, q, Ax, Az, p) ;   \
        Ci [q] = j ;                                    \
    }

    if (A->stype > 0)
    {

        //----------------------------------------------------------------------
        // A is symmetric upper
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < n ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                // get A(i,j)
                Int i = Ai [p] ;
                // skip entries in strictly lower part
                if (i > j) continue ;
                if (i == j)
                {
                    // place diagonal entry A(i,i) in C(:,i)
                    if (keep_diag) ASSIGN_CIJ (i,i) ;
                }
                else
                {
                    // place A(i,j) in C(:,j) and conj(A(i,j)) in C(:,i)
                    ASSIGN_CIJ (i,j) ;
                    ASSIGN_CIJ_CONJ_OR_NCONJ (i,j) ;
                }
            }
        }

    }
    else // A->stype < 0
    {

        //----------------------------------------------------------------------
        // A is symmetric lower
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < n ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                // get A(i,j)
                Int i = Ai [p] ;
                // skip entries in strictly upper part
                if (i < j) continue ;
                if (i == j)
                {
                    // place diagonal entry A(i,i) in C(:,i)
                    if (keep_diag) ASSIGN_CIJ (i,i) ;
                }
                else
                {
                    // place A(i,j) in C(:,j) and conj(A(i,j)) in C(:,i)
                    ASSIGN_CIJ (i,j) ;
                    ASSIGN_CIJ_CONJ_OR_NCONJ (i,j) ;
                }
            }
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX
#undef NCONJUGATE

