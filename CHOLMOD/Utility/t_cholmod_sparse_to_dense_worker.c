//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_sparse_to_dense_worker: sparse to dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_sparse_to_dense_worker)
(
    cholmod_dense *X,       // output dense matrix, already allocated
    cholmod_sparse *A       // input sparse matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (X->d == X->nrow) ;
    ASSERT (X->nrow == A->nrow) ;
    ASSERT (X->ncol == A->ncol) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Xx = (Real *) X->x ;
    Real *Xz = (Real *) X->z ;

    Int  *Ap  = (Int  *) A->p ;
    Int  *Ai  = (Int  *) A->i ;
    Int  *Anz = (Int  *) A->nz ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;

    Int nrow = (Int) A->nrow ;
    Int ncol = (Int) A->ncol ;
    bool packed = (bool) A->packed ;
    bool upper = (A->stype > 0) ;
    bool lower = (A->stype < 0) ;

    //--------------------------------------------------------------------------
    // copy a sparse matrix A into a dense matrix X
    //--------------------------------------------------------------------------

    for (Int j = 0, jx = 0 ; j < ncol ; j++, jx += nrow)
    {

        //----------------------------------------------------------------------
        // copy A(:,j) into X(:,j)
        //----------------------------------------------------------------------

        Int p = Ap [j] ;
        Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
        for ( ; p < pend ; p++)
        {

            //------------------------------------------------------------------
            // get A(i,j)
            //------------------------------------------------------------------

            Int i = Ai [p] ;
            Int q = i + jx ;

            if (upper)
            {

                //--------------------------------------------------------------
                // A is symmetric with upper part stored
                //--------------------------------------------------------------

                if (i > j) continue ;
                // X(i,j) = A(i,j)
                ASSIGN2 (Xx, Xz, q, Ax, Az, p) ;
                if (i < j)
                {
                    // X(j,i) = conj (A(i,j))
                    Int s = j + i*nrow ;
                    ASSIGN2_CONJ (Xx, Xz, s, Ax, Az, p) ;
                }

            }
            else if (lower)
            {

                //--------------------------------------------------------------
                // A is symmetric with lower part stored
                //--------------------------------------------------------------

                if (i < j) continue ;
                // X(i,j) = A(i,j)
                ASSIGN2 (Xx, Xz, q, Ax, Az, p) ;
                if (i > j)
                {
                    // X(j,i) = conj (A(i,j))
                    Int s = j + i*nrow ;
                    ASSIGN2_CONJ (Xx, Xz, s, Ax, Az, p) ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // A and X are both unsymmetric
                //--------------------------------------------------------------

                // X(i,j) = A(i,j)
                ASSIGN2 (Xx, Xz, q, Ax, Az, p) ;
            }
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

