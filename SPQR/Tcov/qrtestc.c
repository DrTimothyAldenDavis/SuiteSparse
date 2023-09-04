/* ========================================================================== */
/* === qrtest_C ============================================================= */
/* ========================================================================== */

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Test the C wrapper functions. */

#include "SuiteSparseQR_C.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

//------------------------------------------------------------------------------
// qrtest_C64: int64_t version
//------------------------------------------------------------------------------

static void qrtest_C64
(
    cholmod_sparse *A,
    double anorm,
    double errs [5],
    double maxresid [2][2],
    cholmod_common *cc
)
{
    cholmod_dense *B, *X, *Resid ;
    cholmod_sparse *Bsparse, *Xsparse ;
    SuiteSparseQR_C_factorization *QR ;
    double resid, one [2] = {1,0}, minusone [2] = {-1,0} ;
    int64_t m, n ;
#ifndef NEXPERT
    cholmod_dense *Y ;
    int split ;
#endif

    m = A->nrow ;
    n = A->ncol ;

    B = cholmod_l_ones (m, 1, A->xtype, cc) ;

    /* X = A\B */
    X = SuiteSparseQR_C_backslash_default (A, B, cc) ;

    /* Resid = A*X - B */
    Resid = cholmod_l_copy_dense (B, cc) ;
    cholmod_l_sdmult (A, 0, one, minusone, X, Resid, cc) ;

    /* resid = norm (Resid,1) */
    resid = cholmod_l_norm_dense (Resid, 1, cc) / MAX (anorm, 1) ;
    resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
    cholmod_l_free_dense (&Resid, cc) ;
    cholmod_l_free_dense (&X, cc) ;

    maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
    printf ("Resid_C1  %d : %g\n", m>n, resid) ;

    /* X = A\B */
    Bsparse = cholmod_l_dense_to_sparse (B, 1, cc) ;
    Xsparse = SuiteSparseQR_C_backslash_sparse (2, -2, A, Bsparse, cc) ;
    X = cholmod_l_sparse_to_dense (Xsparse, cc) ;
    cholmod_l_free_sparse (&Bsparse, cc) ;
    cholmod_l_free_sparse (&Xsparse, cc) ;

    /* Resid = A*X - B */
    Resid = cholmod_l_copy_dense (B, cc) ;
    cholmod_l_sdmult (A, 0, one, minusone, X, Resid, cc) ;

    /* resid = norm (Resid,1) */
    resid = cholmod_l_norm_dense (Resid, 1, cc) / MAX (anorm, 1) ;
    resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
    cholmod_l_free_dense (&Resid, cc) ;
    cholmod_l_free_dense (&X, cc) ;

    maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
    printf ("Resid_C2  %d : %g\n", m>n, resid) ;

#ifndef NEXPERT

    for (split = 0 ; split <= 1 ; split++)
    {
        if (split)
        {
            /* split symbolic/numeric QR factorization */
            QR = SuiteSparseQR_C_symbolic (2, 1, A, cc) ;
            SuiteSparseQR_C_numeric (-2, A, QR, cc) ;
        }
        else
        {
            /* QR factorization, single pass */
            QR = SuiteSparseQR_C_factorize (2, -2, A, cc) ;
        }

        /* Y = Q'*B */
        Y = SuiteSparseQR_C_qmult (0, QR, B, cc) ;

        /* X = E*(R\Y) */
        X = SuiteSparseQR_C_solve (1, QR, Y, cc) ;

        /* Resid = A*X - B */
        Resid = cholmod_l_copy_dense (B, cc) ;
        cholmod_l_sdmult (A, 0, one, minusone, X, Resid, cc) ;

        /* resid = norm (Resid,1) */
        resid = cholmod_l_norm_dense (Resid, 1, cc) / MAX (anorm, 1) ;
        resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
        cholmod_l_free_dense (&Resid, cc) ;
        cholmod_l_free_dense (&X, cc) ;

        maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
        printf ("Resid_C3  %d : %g\n", m>n, resid) ;

        cholmod_l_free_dense (&Y, cc) ;
        SuiteSparseQR_C_free (&QR, cc) ;
    }
#endif

    cholmod_l_free_dense (&B, cc) ;
}

//------------------------------------------------------------------------------
// qrtest_C32: int32_t version
//------------------------------------------------------------------------------

static void qrtest_C32
(
    cholmod_sparse *A,
    double anorm,
    double errs [5],
    double maxresid [2][2],
    cholmod_common *cc
)
{
    cholmod_dense *B, *X, *Resid ;
    cholmod_sparse *Bsparse, *Xsparse ;
    SuiteSparseQR_C_factorization *QR ;
    double resid, one [2] = {1,0}, minusone [2] = {-1,0} ;
    int32_t m, n ;
#ifndef NEXPERT
    cholmod_dense *Y ;
    int split ;
#endif

    m = A->nrow ;
    n = A->ncol ;

    B = cholmod_ones (m, 1, A->xtype, cc) ;

    /* X = A\B */
    X = SuiteSparseQR_C_backslash_default (A, B, cc) ;

    /* Resid = A*X - B */
    Resid = cholmod_copy_dense (B, cc) ;
    cholmod_sdmult (A, 0, one, minusone, X, Resid, cc) ;

    /* resid = norm (Resid,1) */
    resid = cholmod_norm_dense (Resid, 1, cc) / MAX (anorm, 1) ;
    resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
    cholmod_free_dense (&Resid, cc) ;
    cholmod_free_dense (&X, cc) ;

    maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
    printf ("Resid_C1  %d : %g\n", m>n, resid) ;

    /* X = A\B */
    Bsparse = cholmod_dense_to_sparse (B, 1, cc) ;
    Xsparse = SuiteSparseQR_C_backslash_sparse (2, -2, A, Bsparse, cc) ;
    X = cholmod_sparse_to_dense (Xsparse, cc) ;
    cholmod_free_sparse (&Bsparse, cc) ;
    cholmod_free_sparse (&Xsparse, cc) ;

    /* Resid = A*X - B */
    Resid = cholmod_copy_dense (B, cc) ;
    cholmod_sdmult (A, 0, one, minusone, X, Resid, cc) ;

    /* resid = norm (Resid,1) */
    resid = cholmod_norm_dense (Resid, 1, cc) / MAX (anorm, 1) ;
    resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
    cholmod_free_dense (&Resid, cc) ;
    cholmod_free_dense (&X, cc) ;

    maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
    printf ("Resid_C2  %d : %g\n", m>n, resid) ;

#ifndef NEXPERT

    for (split = 0 ; split <= 1 ; split++)
    {
        if (split)
        {
            /* split symbolic/numeric QR factorization */
            QR = SuiteSparseQR_C_symbolic (2, 1, A, cc) ;
            SuiteSparseQR_C_numeric (-2, A, QR, cc) ;
        }
        else
        {
            /* QR factorization, single pass */
            QR = SuiteSparseQR_C_factorize (2, -2, A, cc) ;
        }

        /* Y = Q'*B */
        Y = SuiteSparseQR_C_qmult (0, QR, B, cc) ;

        /* X = E*(R\Y) */
        X = SuiteSparseQR_C_solve (1, QR, Y, cc) ;

        /* Resid = A*X - B */
        Resid = cholmod_copy_dense (B, cc) ;
        cholmod_sdmult (A, 0, one, minusone, X, Resid, cc) ;

        /* resid = norm (Resid,1) */
        resid = cholmod_norm_dense (Resid, 1, cc) / MAX (anorm, 1) ;
        resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
        cholmod_free_dense (&Resid, cc) ;
        cholmod_free_dense (&X, cc) ;

        maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
        printf ("Resid_C3  %d : %g\n", m>n, resid) ;

        cholmod_free_dense (&Y, cc) ;
        SuiteSparseQR_C_free (&QR, cc) ;
    }
#endif

    cholmod_free_dense (&B, cc) ;
}

//------------------------------------------------------------------------------
// qrtest_C: handles both int32_t and int64_t versions
//------------------------------------------------------------------------------

void qrtest_C
(
    cholmod_sparse *A,
    double anorm,
    double errs [5],
    double maxresid [2][2],
    cholmod_common *cc
)
{
    if (A->itype == CHOLMOD_INT)
    {
        qrtest_C32 (A, anorm, errs, maxresid, cc) ;
    }
    else
    {
        qrtest_C64 (A, anorm, errs, maxresid, cc) ;
    }
}

