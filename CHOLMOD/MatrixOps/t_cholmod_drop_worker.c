//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/t_cholmod_drop_worker: drop small entries
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// IF_KEEP: macro to determine if A(i,j) is kept
//------------------------------------------------------------------------------

#ifdef REAL

    #define IF_KEEP(Ax,Az,p,tol)                        \
        double ax = (double) Ax [p] ;                   \
        if (fabs (ax) > tol || isnan (ax))

#elif defined ( COMPLEX )

    #define IF_KEEP(Ax,Az,p,tol)                        \
        double ax = (double) Ax [2*(p)  ] ;             \
        double az = (double) Ax [2*(p)+1] ;             \
        double a = SuiteSparse_config_hypot (ax, az) ;  \
        if (a > tol || isnan (ax) || isnan (az))

#elif defined ( ZOMPLEX )

    #define IF_KEEP(Ax,Az,p,tol)                        \
        double ax = (double) Ax [p] ;                   \
        double az = (double) Az [p] ;                   \
        double a = SuiteSparse_config_hypot (ax, az) ;  \
        if (a > tol || isnan (ax) || isnan (az))

#endif

//------------------------------------------------------------------------------
// t_cholmod_drop_worker
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_drop_worker)
(
    // input:
    double tol,         // keep entries with absolute value > tol
    // input/output:
    cholmod_sparse *A,  // matrix to drop entries from
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ap = A->p ;
    Int *Ai = A->i ;
    Real *Ax = A->x ;
    Real *Az = A->x ;
    Int *Anz = A->nz ;
    bool packed = A->packed ;
    Int ncol = A->ncol ;
    Int nz = 0 ;

    //--------------------------------------------------------------------------
    // drop small numerical entries from A, and entries in ignored part
    //--------------------------------------------------------------------------

    if (A->stype > 0)
    {

        //----------------------------------------------------------------------
        // A is symmetric, with just upper triangular part stored
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            Ap [j] = nz ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                if (i > j) continue ;
                IF_KEEP (Ax, Ax, p, tol)
                {
                    // keep this entry
                    Ai [nz] = i ;
                    ASSIGN (Ax, Az, nz, Ax, Az, p) ;
                    nz++ ;
                }
            }
        }

    }
    else if (A->stype < 0)
    {

        //----------------------------------------------------------------------
        // A is symmetric, with just lower triangular part stored
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            Ap [j] = nz ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                if (i < j) continue ;
                IF_KEEP (Ax, Ax, p, tol)
                {
                    // keep this entry
                    Ai [nz] = i ;
                    ASSIGN (Ax, Az, nz, Ax, Az, p) ;
                    nz++ ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // both parts of A present, just drop small entries
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            Ap [j] = nz ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                IF_KEEP (Ax, Ax, p, tol)
                {
                    // keep this entry
                    Ai [nz] = i ;
                    ASSIGN (Ax, Az, nz, Ax, Az, p) ;
                    nz++ ;
                }
            }
        }
    }

    // finalize the last column of A
    Ap [ncol] = nz ;

    // reduce A->i and A->x in size
    ASSERT (MAX (1,nz) <= A->nzmax) ;
    CHOLMOD(reallocate_sparse) (nz, A, Common) ;
    ASSERT (Common->status >= CHOLMOD_OK) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

#undef IF_KEEP

