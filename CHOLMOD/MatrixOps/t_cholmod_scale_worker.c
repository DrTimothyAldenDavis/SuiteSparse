//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/t_cholmod_scale_worker: scale a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_scale_worker)
(
    // input:
    cholmod_dense *S,   // scale factors (scalar or vector)
    int scale,          // type of scaling to compute
    // input/output:
    cholmod_sparse *A   // matrix to scale
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int  *Ap  = A->p ;
    Int  *Anz = A->nz ;
    Int  *Ai  = A->i ;
    Real *Ax  = A->x ;
    Real *Az  = A->z ;
    bool packed = A->packed ;
    Int ncol = A->ncol ;

    Real *Sx = S->x ;
    Real *Sz = S->z ;

    //--------------------------------------------------------------------------
    // scale the matrix
    //--------------------------------------------------------------------------

    if (scale == CHOLMOD_ROW)
    {

        //----------------------------------------------------------------------
        // A = diag(s)*A, row scaling
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                // t = S (i) * A (i,j)
                Real tx [2] ;
                Real tz [1] ;
                MULT (tx, tz, 0, Sx, Sz, i, Ax, Az, p) ;
                // A (i,j) = t
                ASSIGN (Ax, Az, p, tx, tz, 0) ;
            }
        }

    }
    else if (scale == CHOLMOD_COL)
    {

        //----------------------------------------------------------------------
        // A = A*diag(s), column scaling
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            // s = S (j)
            Real sx [2] ;
            Real sz [1] ;
            ASSIGN (sx, sz, 0, Sx, Sz, j) ;

            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                // t = A (i,j) * s
                Real tx [2] ;
                Real tz [1] ;
                MULT (tx, tz, 0, Ax, Az, p, sx, sz, 0) ;
                // A (i,j) = t
                ASSIGN (Ax, Az, p, tx, tz, 0) ;
            }
        }

    }
    else if (scale == CHOLMOD_SYM)
    {

        //----------------------------------------------------------------------
        // A = diag(s)*A*diag(s), symmetric scaling
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            // s = S (j)
            Real sx [2] ;
            Real sz [1] ;
            ASSIGN (sx, sz, 0, Sx, Sz, j) ;

            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                // t = A (i,j) * S (i)
                Real tx [2] ;
                Real tz [1] ;
                MULT (tx, tz, 0, Ax, Az, p, Sx, Sz, i) ;
                // A (i,j) = s * t
                MULT (Ax, Az, p, sx, sz, 0, tx, tz, 0) ;
            }
        }

    }
    else if (scale == CHOLMOD_SCALAR)
    {

        //----------------------------------------------------------------------
        // A = s[0] * A, scalar scaling
        //----------------------------------------------------------------------

        // s = S (0)
        Real sx [2] ;
        Real sz [1] ;
        ASSIGN (sx, sz, 0, Sx, Sz, 0) ;

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                // t = s * A (i,j)
                Real tx [2] ;
                Real tz [1] ;
                MULT (tx, tz, 0, sx, sz, 0, Ax, Az, p) ;
                // A (i,j) = t
                ASSIGN (Ax, Az, p, tx, tz, 0) ;
            }
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

