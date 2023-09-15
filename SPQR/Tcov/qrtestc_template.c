/* ========================================================================== */
/* === qrtestc_template ===================================================== */
/* ========================================================================== */

// SPQR, Copyright (c) 2008-2023, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
// QRTESTC:  test the SPQR C API
//------------------------------------------------------------------------------

// The #include'ing file must define the following macros:
// QRTEST:  the name of the function

// and one of:
// SINT:    float (and float complex), int32
// SLONG:   float (and float complex), int64
// DINT:    double (and double complex), int32
// DLONG:   double (and double complex), int64

// the SINT and SLONG cases are not yet used.

#include "cholmod_types.h"

static void QRTESTC
(
    cholmod_sparse *A,
    Real anorm,
    Real errs [5],
    Real maxresid [2][2],
    cholmod_common *cc
)
{
    cholmod_dense *B, *X, *Resid ;
    cholmod_sparse *Bsparse, *Xsparse ;
    SuiteSparseQR_C_factorization *QR ;
    Real resid, one [2] = {1,0}, minusone [2] = {-1,0} ;
    Int m, n ;
#ifndef NEXPERT
    cholmod_dense *Y ;
    int split ;
#endif

    m = A->nrow ;
    n = A->ncol ;

    B = CHOLMOD (ones (m, 1, A->xtype, cc)) ;

    /* X = A\B */
    X = SuiteSparseQR_C_backslash_default (A, B, cc) ;

    /* Resid = A*X - B */
    Resid = CHOLMOD (copy_dense (B, cc)) ;
    CHOLMOD (sdmult (A, 0, one, minusone, X, Resid, cc)) ;

    /* resid = norm (Resid,1) */
    resid = CHOLMOD (norm_dense (Resid, 1, cc)) / MAX (anorm, 1) ;
    resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
    CHOLMOD (free_dense (&Resid, cc)) ;
    CHOLMOD (free_dense (&X, cc)) ;

    maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
    printf ("Resid_C1  %d : %g\n", m>n, resid) ;

    /* X = A\B */
    Bsparse = CHOLMOD (dense_to_sparse (B, 1, cc)) ;
    Xsparse = SuiteSparseQR_C_backslash_sparse (2, -2, A, Bsparse, cc) ;
    X = CHOLMOD (sparse_to_dense (Xsparse, cc)) ;
    CHOLMOD (free_sparse (&Bsparse, cc)) ;
    CHOLMOD (free_sparse (&Xsparse, cc)) ;

    /* Resid = A*X - B */
    Resid = CHOLMOD (copy_dense (B, cc)) ;
    CHOLMOD (sdmult (A, 0, one, minusone, X, Resid, cc)) ;

    /* resid = norm (Resid,1) */
    resid = CHOLMOD (norm_dense (Resid, 1, cc)) / MAX (anorm, 1) ;
    resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
    CHOLMOD (free_dense (&Resid, cc)) ;
    CHOLMOD (free_dense (&X, cc)) ;

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
        Resid = CHOLMOD (copy_dense (B, cc)) ;
        CHOLMOD (sdmult (A, 0, one, minusone, X, Resid, cc)) ;

        /* resid = norm (Resid,1) */
        resid = CHOLMOD (norm_dense (Resid, 1, cc)) / MAX (anorm, 1) ;
        resid = (resid < 0 || resid != resid) ? 9e99 : resid ;
        CHOLMOD (free_dense (&Resid, cc)) ;
        CHOLMOD (free_dense (&X, cc)) ;

        maxresid [m>n][0] = MAX (maxresid [m>n][0], resid) ;
        printf ("Resid_C3  %d : %g\n", m>n, resid) ;

        CHOLMOD (free_dense (&Y, cc)) ;
        SuiteSparseQR_C_free (&QR, cc) ;
    }
#endif

    CHOLMOD (free_dense (&B, cc)) ;
}

