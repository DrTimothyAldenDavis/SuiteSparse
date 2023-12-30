//------------------------------------------------------------------------------
// CHOLMOD/Demo/cholmod_sl_simple: simple demo program for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Demo Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Read in a real symmetric or complex Hermitian matrix from stdin in
// MatrixMarket format, solve Ax=b where b=[1 1 ... 1]', and print the residual.
//
// Usage: cholmod_sl_simple < matrixfile
//
// There are four versions of this demo:
// cholmod_di_simple:   double, int32
// cholmod_dl_simple:   double, int64
// cholmod_si_simple:   float, int32
// cholmod_sl_simple:   float, int64

#include "cholmod.h"
int main (void)
{
    cholmod_sparse *A ;
    cholmod_dense *x, *b, *r ;
    cholmod_factor *L ;
    double one [2] = {1,0}, m1 [2] = {-1,0} ;       // basic scalars
    cholmod_common c ;
    cholmod_l_start (&c) ;                          // start CHOLMOD
    int dtype = CHOLMOD_SINGLE ;                    // use single precision
    A = cholmod_l_read_sparse2 (stdin, dtype, &c) ; // read in a matrix
    c.precise = true ;
    c.print = (A->nrow > 5) ? 3 : 5 ;
    cholmod_l_print_sparse (A, "A", &c) ;           // print the matrix
    if (A == NULL || A->stype == 0)                 // A must be symmetric
    {
        cholmod_l_free_sparse (&A, &c) ;
        cholmod_l_finish (&c) ;
        return (0) ;
    }
    b = cholmod_l_ones (A->nrow, 1, A->xtype + dtype, &c) ;   // b = ones(n,1)

    double t1 = SUITESPARSE_TIME ;
    L = cholmod_l_analyze (A, &c) ;                   // analyze
    t1 = SUITESPARSE_TIME - t1 ;
    double t2 = SUITESPARSE_TIME ;
    cholmod_l_factorize (A, L, &c) ;                  // factorize
    t2 = SUITESPARSE_TIME - t2 ;
    double t3 = SUITESPARSE_TIME ;
    x = cholmod_l_solve (CHOLMOD_A, L, b, &c) ;       // solve Ax=b
    t3 = SUITESPARSE_TIME - t3 ;
    printf ("analyze   time: %10.3f sec\n", t1) ;
    printf ("factorize time: %10.3f sec\n", t2) ;
    printf ("solve     time: %10.3f sec\n", t3) ;
    printf ("total     time: %10.3f sec\n", t1 + t2 + t3) ;

    cholmod_l_print_factor (L, "L", &c) ;           // print the factorization
    cholmod_l_print_dense (x, "x", &c) ;            // print the solution
    r = cholmod_l_copy_dense (b, &c) ;              // r = b
#ifndef NMATRIXOPS
    cholmod_l_sdmult (A, 0, m1, one, x, r, &c) ;    // r = r-Ax
    double rnorm = cholmod_l_norm_dense (r, 0, &c) ;  // compute inf-norm of r
    double anorm = cholmod_l_norm_sparse (A, 0, &c) ; // compute inf-norm of A
    printf ("\n%s precision results:\n", dtype ? "single" : "double") ;
    printf ("norm(b-Ax) %8.1e\n", rnorm) ;
    printf ("norm(A)    %8.1e\n", anorm) ;
    double relresid = rnorm / anorm ;
    printf ("resid: norm(b-Ax)/norm(A) %8.1e\n", relresid) ;
    fprintf (stderr, "resid: norm(b-Ax)/norm(A) %8.1e\n", relresid) ;
#else
    printf ("residual norm not computed (requires CHOLMOD/MatrixOps)\n") ;
#endif
    cholmod_l_free_factor (&L, &c) ;                // free matrices
    cholmod_l_free_sparse (&A, &c) ;
    cholmod_l_free_dense (&r, &c) ;
    cholmod_l_free_dense (&x, &c) ;
    cholmod_l_free_dense (&b, &c) ;
    cholmod_l_print_common ("common", &c) ;
    cholmod_l_gpu_stats (&c) ;
    cholmod_l_finish (&c) ;                         // finish CHOLMOD
    return (0) ;
}

