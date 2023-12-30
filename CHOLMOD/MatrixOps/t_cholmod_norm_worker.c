//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/t_cholmod_norm_worker: compute norm of a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// The intermediate values and the final result are always computed in
// double, even if the matrix is single precision.

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// t_cholmod_norm_dense_worker
//------------------------------------------------------------------------------

#ifndef PATTERN

static double TEMPLATE (cholmod_norm_dense_worker)     // return norm
(
    // input:
    cholmod_dense *X,   // matrix to compute the norm of
    int norm,           // type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm
    double *W           // optional size-ncol workspace, always double
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int ncol = X->ncol ;
    Int nrow = X->nrow ;
    Int d = X->d ;
    Real *Xx = X->x ;
    Real *Xz = X->z ;

    double xnorm = 0 ;

    //--------------------------------------------------------------------------
    // compute the norm
    //--------------------------------------------------------------------------

    if (W != NULL)
    {

        //----------------------------------------------------------------------
        // infinity-norm = max row sum, using stride-1 access of X
        //----------------------------------------------------------------------

        DEBUG (for (Int i = 0 ; i < nrow ; i++) ASSERT (W [i] == 0)) ;

        // this is faster than stride-d, but requires O(nrow) workspace
        for (Int j = 0 ; j < ncol ; j++)
        {
            for (Int i = 0 ; i < nrow ; i++)
            {
                W [i] += ABS (Xx, Xz, i+j*d) ;
            }
        }
        for (Int i = 0 ; i < nrow ; i++)
        {
            double s = W [i] ;
            if ((isnan (s) || s > xnorm) && !isnan (xnorm))
            {
                xnorm = s ;
            }
            W [i] = 0 ;
        }

    }
    else if (norm == 0)
    {

        //----------------------------------------------------------------------
        // infinity-norm = max row sum, using stride-d access of X
        //----------------------------------------------------------------------

        for (Int i = 0 ; i < nrow ; i++)
        {
            double s = 0 ;
            for (Int j = 0 ; j < ncol ; j++)
            {
                s += ABS (Xx, Xz, i+j*d) ;
            }
            if ((isnan (s) || s > xnorm) && !isnan (xnorm))
            {
                xnorm = s ;
            }
        }

    }
    else if (norm == 1)
    {

        //----------------------------------------------------------------------
        // 1-norm = max column sum
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            double s = 0 ;
            for (Int i = 0 ; i < nrow ; i++)
            {
                s += ABS (Xx, Xz, i+j*d) ;
            }
            if ((isnan (s) || s > xnorm) && !isnan (xnorm))
            {
                xnorm = s ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // 2-norm = sqrt (sum (X.^2))
        //----------------------------------------------------------------------

        for (Int i = 0 ; i < nrow ; i++)
        {
            double s = ABS (Xx, Xz, i) ;
            xnorm += s*s ;
        }
        xnorm = sqrt (xnorm) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (xnorm) ;
}

#endif

//------------------------------------------------------------------------------
// t_cholmod_norm_sparse_worker
//------------------------------------------------------------------------------

static double TEMPLATE (cholmod_norm_sparse_worker)     // return norm
(
    // input:
    cholmod_sparse *A,  // matrix to compute the norm of
    int norm,           // type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm
    double *W           // optional size-ncol workspace, always double
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
    Int ncol = A->ncol ;
    Int nrow = A->nrow ;
    bool packed = A->packed ;

    double anorm = 0 ;

    if (A->stype > 0)
    {

        //----------------------------------------------------------------------
        // A is symmetric with upper triangular part stored
        //----------------------------------------------------------------------

        // infinity-norm = 1-norm = max row/col sum
        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                double s = ABS (Ax, Az, p) ;
                if (i == j)
                {
                    W [i] += s ;
                }
                else if (i < j)
                {
                    W [i] += s ;
                    W [j] += s ;
                }
            }
        }

    }
    else if (A->stype < 0)
    {

        //----------------------------------------------------------------------
        // A is symmetric with lower triangular part stored
        //----------------------------------------------------------------------

        // infinity-norm = 1-norm = max row/col sum
        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                double s = ABS (Ax, Az, p) ;
                if (i == j)
                {
                    W [i] += s ;
                }
                else if (i > j)
                {
                    W [i] += s ;
                    W [j] += s ;
                }
            }
        }

    }
    else if (norm == 0)
    {

        //----------------------------------------------------------------------
        // A is unsymmetric, compute the infinity-norm
        //----------------------------------------------------------------------

        // infinity-norm = max row sum
        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                W [Ai [p]] += ABS (Ax, Az, p) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A is unsymmetric, compute the 1-norm
        //----------------------------------------------------------------------

        // 1-norm = max column sum
        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            double s ;
            #ifdef PATTERN
            {
                s = pend - p ;
            }
            #else
            {
                s = 0 ;
                for ( ; p < pend ; p++)
                {
                    s += ABS (Ax, Az, p) ;
                }
            }
            #endif
            if ((isnan (s) || s > anorm) && !isnan (anorm))
            {
                anorm = s ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // compute the max row sum
    //--------------------------------------------------------------------------

    if (A->stype || norm == 0)
    {
        for (Int i = 0 ; i < nrow ; i++)
        {
            double s = W [i] ;
            if ((isnan (s) || s > anorm) && !isnan (anorm))
            {
                anorm = s ;
            }
            W [i] = 0 ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (anorm) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

