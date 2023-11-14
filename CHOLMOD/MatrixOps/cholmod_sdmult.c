//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/cholmod_sdmult: sparse-times-dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Sparse matrix times dense matrix:
// Y = alpha*(A*X) + beta*Y or Y = alpha*(A'*X) + beta*Y,
// where A is sparse and X and Y are dense.
//
// when using A,  X has A->ncol columns and Y has A->nrow rows
// when using A', X has A->nrow columns and Y has A->ncol rows
//
// workspace: none in Common.  Temporary workspace of size 4*(X->nrow) is used
// if A is stored in symmetric form and X has four columns or more.  If the
// workspace is not available, a slower method is used instead that requires
// no workspace.
//
// transpose = 0: use A
// otherwise, use A' (complex conjugate transpose)
//
// transpose is ignored if the matrix is symmetric or Hermitian.
// (the array transpose A.' is not supported).
//
// Supports real, complex, and zomplex matrices, but the xtypes and dtypes of
// A, X, and Y must all match.

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMATRIXOPS

//------------------------------------------------------------------------------
// t_cholmod_sdmult_worker
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_sdmult_worker.c"
#define COMPLEX
#include "t_cholmod_sdmult_worker.c"
#define ZOMPLEX
#include "t_cholmod_sdmult_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_sdmult_worker.c"
#define COMPLEX
#include "t_cholmod_sdmult_worker.c"
#define ZOMPLEX
#include "t_cholmod_sdmult_worker.c"

//------------------------------------------------------------------------------
// cholmod_sdmult
//------------------------------------------------------------------------------

int CHOLMOD(sdmult)
(
    // input:
    cholmod_sparse *A,  // sparse matrix to multiply
    int transpose,      // use A if 0, otherwise use A'
    double alpha [2],   // scale factor for A
    double beta [2],    // scale factor for Y
    cholmod_dense *X,   // dense matrix to multiply
    // input/output:
    cholmod_dense *Y,   // resulting dense matrix
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (X, FALSE) ;
    RETURN_IF_NULL (Y, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (Y, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    size_t ny = transpose ? A->ncol : A->nrow ;     // required length of Y
    size_t nx = transpose ? A->nrow : A->ncol ;     // required length of X
    if (X->nrow != nx || X->ncol != Y->ncol || Y->nrow != ny)
    {
        // X and/or Y have the wrong dimension
        ERROR (CHOLMOD_INVALID, "X and/or Y have wrong dimensions") ;
        return (FALSE) ;
    }
    if (A->xtype != X->xtype || A->xtype != Y->xtype ||
        A->dtype != X->dtype || A->dtype != Y->dtype)
    {
        ERROR (CHOLMOD_INVALID, "A, X, and Y must have same xtype and dtype") ;
        return (FALSE) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace, if required
    //--------------------------------------------------------------------------

    void *w = NULL ;
    size_t e = (A->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((A->xtype == CHOLMOD_REAL) ? 1 : 2) ;

    if (A->stype && X->ncol >= 4)
    {
        w = CHOLMOD(malloc) (4*nx, ex, Common) ;
    }
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;    // out of memory
    }

    //--------------------------------------------------------------------------
    // Y = alpha*op(A)*X + beta*Y via template routine
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (A, "A", Common) >= 0) ;
    DEBUG (CHOLMOD(dump_dense) (X, "X", Common)) ;
    DEBUG (if ((beta [0] != 0)
           || ((beta [1] != 0) && A->xtype != CHOLMOD_REAL))
            CHOLMOD(dump_dense) (Y, "Y", Common)) ;

    float s_alpha [2] ;
    s_alpha [0] = (float) alpha [0] ;
    s_alpha [1] = (float) alpha [1] ;
    float s_beta  [2] ;
    s_beta [0] = (float) beta [0] ;
    s_beta [1] = (float) beta [1] ;

    switch ((A->xtype + A->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_sdmult_worker (A, transpose, s_alpha, s_beta, X, Y, w) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_sdmult_worker (A, transpose, s_alpha, s_beta, X, Y, w) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_sdmult_worker (A, transpose, s_alpha, s_beta, X, Y, w) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_sdmult_worker (A, transpose, alpha, beta, X, Y, w) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_sdmult_worker (A, transpose, alpha, beta, X, Y, w) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_sdmult_worker (A, transpose, alpha, beta, X, Y, w) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    CHOLMOD(free) (4*nx, ex, w, Common) ;
    DEBUG (CHOLMOD(dump_dense) (Y, "Y", Common)) ;
    return (TRUE) ;
}

#endif
#endif

