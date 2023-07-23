/* ========================================================================== */
/* === qrsimplec.c ========================================================== */
/* ========================================================================== */

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

/* A very simple example of the use of SuiteSparseQR by a C main program.
   Usage:  qrsimplec < Matrix_in_MatrixMarket_format */

#include "SuiteSparseQR_C.h"
int main (int argc, char **argv)
{
    cholmod_common Common, *cc ;
    cholmod_sparse *A ;
    cholmod_dense *X, *B, *Residual = NULL ;
    double rnorm, one [2] = {1,0}, minusone [2] = {-1,0} ;
    int mtype ;

    /* start CHOLMOD */
    cc = &Common ;
    cholmod_start (cc) ;

    /* load A */
    A = (cholmod_sparse *)
        cholmod_read_matrix (stdin, 1, &mtype, cc) ;

    /* B = ones (size (A,1),1) */
    B = cholmod_ones (A->nrow, 1, A->xtype, cc) ;

    /* X = A\B */
    X = SuiteSparseQR_C_backslash_default (A, B, cc) ;

#ifndef NMATRIXOPS
    /* rnorm = norm (B-A*X) */
    Residual = cholmod_copy_dense (B, cc) ;
    cholmod_sdmult (A, 0, minusone, one, X, Residual, cc) ;
    rnorm = cholmod_norm_dense (Residual, 2, cc) ;
    printf ("2-norm of residual: %8.1e\n", rnorm) ;
#else
    printf ("2-norm of residual: not computed (requires CHOLMOD/MatrixOps)\n") ;
#endif
    printf ("rank %" SuiteSparse_long_idd "\n", cc->SPQR_istat [4]) ;

    /* free everything and finish CHOLMOD */
    cholmod_free_dense (&Residual, cc) ;
    cholmod_free_sparse (&A, cc) ;
    cholmod_free_dense (&X, cc) ;
    cholmod_free_dense (&B, cc) ;
    cholmod_finish (cc) ;
    return (0) ;
}
