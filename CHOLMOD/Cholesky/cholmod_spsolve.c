//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/cholmod_spsolve: solve a linear system with sparse x and b
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Given an LL' or LDL' factorization of A, solve one of the following systems:
//
//      Ax=b        0: CHOLMOD_A        also applies the permutation L->Perm
//      LDL'x=b     1: CHOLMOD_LDLt     does not apply L->Perm
//      LDx=b       2: CHOLMOD_LD
//      DL'x=b      : CHOLMOD_DLt
//      Lx=b        4: CHOLMOD_L
//      L'x=b       5: CHOLMOD_Lt
//      Dx=b        6: CHOLMOD_D
//      x=Pb        7: CHOLMOD_P        apply a permutation (P is L->Perm)
//      x=P'b       8: CHOLMOD_Pt       apply an inverse permutation
//
// where b and x are sparse.  If L and b are real, then x is real.  Otherwise,
// x is complex or zomplex, depending on the Common->prefer_zomplex parameter.
// All xtypes of x and b are supported (real, complex, and zomplex), and
// all dtypes.

#include "cholmod_internal.h"

#ifndef NCHOLESKY

//------------------------------------------------------------------------------
// t_cholmod_spsolve_worker
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_spsolve_worker.c"
#define COMPLEX
#include "t_cholmod_spsolve_worker.c"
#define ZOMPLEX
#include "t_cholmod_spsolve_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_spsolve_worker.c"
#define COMPLEX
#include "t_cholmod_spsolve_worker.c"
#define ZOMPLEX
#include "t_cholmod_spsolve_worker.c"

//------------------------------------------------------------------------------
// cholmod_spsolve
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(spsolve)            // returns the sparse solution X
(
    // input:
    int sys,            // system to solve
    cholmod_factor *L,  // factorization to use
    cholmod_sparse *B,  // right-hand-side
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_dense *X4 = NULL, *B4 = NULL ;
    cholmod_sparse *X = NULL ;

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (L, NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, NULL) ;
    RETURN_IF_XTYPE_INVALID (B, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, NULL) ;
    if (L->n != B->nrow)
    {
        ERROR (CHOLMOD_INVALID, "dimensions of L and B do not match") ;
        return (NULL) ;
    }
    if (B->stype)
    {
        ERROR (CHOLMOD_INVALID, "B cannot be stored in symmetric mode") ;
        return (NULL) ;
    }
    if (L->dtype != B->dtype)
    {
        ERROR (CHOLMOD_INVALID, "dtype of L and B must match") ;
        return (NULL) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace B4 and initial result X
    //--------------------------------------------------------------------------

    Int n = L->n ;
    Int nrhs = B->ncol ;

    // X is real if both L and B are real, complex/zomplex otherwise
    int X_xtype =
        (L->xtype == CHOLMOD_REAL && B->xtype == CHOLMOD_REAL) ?  CHOLMOD_REAL :
        (Common->prefer_zomplex ? CHOLMOD_ZOMPLEX : CHOLMOD_COMPLEX) ;

    // solve up to 4 columns at a time
    Int block = MIN (nrhs, 4) ;

    // initial size of X is at most 4*n
    size_t nzmax = ((size_t) n) * ((size_t) block) ;

    X = CHOLMOD(spzeros) (n, nrhs, nzmax, X_xtype + B->dtype, Common) ;
    B4 = CHOLMOD(zeros) (n, block, B->xtype + B->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        CHOLMOD(free_sparse) (&X, Common) ;
        CHOLMOD(free_dense) (&B4, Common) ;
        return (NULL) ;
    }

    size_t xnz = 0 ;

    //--------------------------------------------------------------------------
    // solve in chunks of 4 columns at a time
    //--------------------------------------------------------------------------

    for (Int jfirst = 0 ; jfirst < nrhs ; jfirst += block)
    {

        //----------------------------------------------------------------------
        // adjust the number of columns of B4
        //----------------------------------------------------------------------

        Int jlast = MIN (nrhs, jfirst + block) ;
        B4->ncol = jlast - jfirst ;

        //----------------------------------------------------------------------
        // scatter B(jfirst:jlast-1) into B4
        //----------------------------------------------------------------------

        switch ((B->xtype + B->dtype) % 8)
        {
            case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                rs_cholmod_spsolve_B_scatter_worker (B4, B, jfirst, jlast) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                cs_cholmod_spsolve_B_scatter_worker (B4, B, jfirst, jlast) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                zs_cholmod_spsolve_B_scatter_worker (B4, B, jfirst, jlast) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                rd_cholmod_spsolve_B_scatter_worker (B4, B, jfirst, jlast) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                cd_cholmod_spsolve_B_scatter_worker (B4, B, jfirst, jlast) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                zd_cholmod_spsolve_B_scatter_worker (B4, B, jfirst, jlast) ;
                break ;
        }

        //----------------------------------------------------------------------
        // solve the system (X4 = A\B4 or other system)
        //----------------------------------------------------------------------

        X4 = CHOLMOD(solve) (sys, L, B4, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            CHOLMOD(free_sparse) (&X, Common) ;
            CHOLMOD(free_dense) (&B4, Common) ;
            CHOLMOD(free_dense) (&X4, Common) ;
            return (NULL) ;
        }
        ASSERT (X4->xtype == X_xtype) ;

        //----------------------------------------------------------------------
        // append the solution onto X
        //----------------------------------------------------------------------

        bool ok = true ;

        switch ((X->xtype + X->dtype) % 8)
        {
            case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                ok = rs_cholmod_spsolve_X_worker (X, X4, jfirst, jlast, &xnz,
                    Common) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                ok = cs_cholmod_spsolve_X_worker (X, X4, jfirst, jlast, &xnz,
                    Common) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                ok = zs_cholmod_spsolve_X_worker (X, X4, jfirst, jlast, &xnz,
                    Common) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                ok = rd_cholmod_spsolve_X_worker (X, X4, jfirst, jlast, &xnz,
                    Common) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                ok = cd_cholmod_spsolve_X_worker (X, X4, jfirst, jlast, &xnz,
                    Common) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                ok = zd_cholmod_spsolve_X_worker (X, X4, jfirst, jlast, &xnz,
                    Common) ;
                break ;
        }

        CHOLMOD(free_dense) (&X4, Common) ;
        if (!ok)
        {
            // out of memory
            CHOLMOD(free_sparse) (&X, Common) ;
            CHOLMOD(free_dense) (&B4, Common) ;
            return (NULL) ;
        }

        //----------------------------------------------------------------------
        // clear B4 for next iteration
        //----------------------------------------------------------------------

        if (jlast < nrhs)
        {
            switch ((B->xtype + B->dtype) % 8)
            {
                case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                    rs_cholmod_spsolve_B_clear_worker (B4, B, jfirst, jlast) ;
                    break ;

                case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                    cs_cholmod_spsolve_B_clear_worker (B4, B, jfirst, jlast) ;
                    break ;

                case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                    zs_cholmod_spsolve_B_clear_worker (B4, B, jfirst, jlast) ;
                    break ;

                case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                    rd_cholmod_spsolve_B_clear_worker (B4, B, jfirst, jlast) ;
                    break ;

                case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                    cd_cholmod_spsolve_B_clear_worker (B4, B, jfirst, jlast) ;
                    break ;

                case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                    zd_cholmod_spsolve_B_clear_worker (B4, B, jfirst, jlast) ;
                    break ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize X, reduce it in size, free workspace, and return result
    //--------------------------------------------------------------------------

    Int *Xp = X->p ;
    Xp [nrhs] = xnz ;
    ASSERT (xnz <= X->nzmax) ;
    CHOLMOD(reallocate_sparse) (xnz, X, Common) ;
    ASSERT (Common->status == CHOLMOD_OK) ;
    CHOLMOD(free_dense) (&B4, Common) ;
    return (X) ;
}
#endif

