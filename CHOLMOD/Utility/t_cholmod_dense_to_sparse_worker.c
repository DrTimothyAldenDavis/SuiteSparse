//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense_to_sparse_worker: dense to sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_dense_to_sparse_worker)
(
    cholmod_sparse *C,      // output sparse matrix, already allocated
    cholmod_dense *X        // input dense matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (X->nrow == C->nrow) ;
    ASSERT (X->ncol == C->ncol) ;
    ASSERT (X->d >= X->nrow) ;
    ASSERT (X->dtype == C->dtype) ;
    ASSERT (C->packed) ;
    ASSERT (C->sorted) ;
    ASSERT (C->stype == 0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Xx = (Real *) X->x ;
    Real *Xz = (Real *) X->z ;
    Int nrow = X->nrow ;
    Int ncol = X->ncol ;
    Int d = X->d ;

    Int  *Cp  = (Int  *) C->p ;
    Int  *Ci  = (Int  *) C->i ;
    Real *Cx  = (Real *) C->x ;
    Real *Cz  = (Real *) C->z ;
    bool pattern = (C->xtype == CHOLMOD_PATTERN) ;

    //--------------------------------------------------------------------------
    // copy the dense matrix into the sparse matrix C
    //--------------------------------------------------------------------------

    Int p = 0 ;

    if (pattern)
    {

        //----------------------------------------------------------------------
        // copy just the pattern of the nonzeros of X into C
        //----------------------------------------------------------------------

        for (Int j = 0, jx = 0 ; j < ncol ; j++, jx += d)
        {
            // log the start of C(:,j)
            Cp [j] = p ;
            // find the pattern of nonzeros in X(:,j)
            for (Int i = 0, q = jx ; i < nrow ; i++, q++)
            {
                if (ENTRY_IS_NONZERO (Xx, Xz, q))
                {
                    Ci [p++] = i ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // copy the pattern and values of nonzeros in X into C
        //----------------------------------------------------------------------

        ASSERT (C->xtype == X->xtype) ;

        for (Int j = 0, jx = 0 ; j < ncol ; j++, jx += d)
        {
            // log the start of C(:,j)
            Cp [j] = p ;
            // find the pattern and values of nonzeros in X(:,j)
            for (Int i = 0, q = jx ; i < nrow ; i++, q++)
            {
                if (ENTRY_IS_NONZERO (Xx, Xz, q))
                {
                    ASSIGN (Cx, Cz, p, Xx, Xz, q) ;
                    Ci [p++] = i ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // log the end of the last column of C
    //--------------------------------------------------------------------------

    Cp [ncol] = p ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

