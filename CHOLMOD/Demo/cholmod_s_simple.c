//------------------------------------------------------------------------------
// CHOLMOD/Demo/cholmod_s_simple: simple demo program for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Demo Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Read in a real symmetric or complex Hermitian matrix from stdin in
// MatrixMarket format, solve Ax=b where b=[1 1 ... 1]', and print the residual.
// The matrix and its factorization are all in single precision.  Compare with
// cholmod_simple.  Note that all scalars (one, m1, rnorm, anorm) are passed
// to CHOLMOD in double precision, and returned as double.  However, all
// internal computations below are done in single precision.  For input/output
// parameters, scalars are typecast to/from the internal float computations.
//
// Usage: cholmod_s_simple < matrixfile

#include "cholmod.h"
int main (void)
{
    cholmod_sparse *A ;
    cholmod_dense *x, *b, *r ;
    cholmod_factor *L ;
    double one [2] = {1,0}, m1 [2] = {-1,0} ;       // basic scalars
    cholmod_common c ;
    cholmod_start (&c) ;                            // start CHOLMOD
    int dtype = CHOLMOD_SINGLE ;                    // use single precision
    A = cholmod_read_sparse2 (stdin, dtype, &c) ;   // read in a matrix
    c.precise = true ;
    c.print = (A->nrow > 5) ? 4 : 5 ;
    cholmod_print_sparse (A, "A", &c) ;             // print the matrix
    if (A == NULL || A->stype == 0)                 // A must be symmetric
    {
        cholmod_free_sparse (&A, &c) ;
        cholmod_finish (&c) ;
        return (0) ;
    }
    b = cholmod_ones (A->nrow, 1, A->xtype + dtype, &c) ;   // b = ones(n,1)
    L = cholmod_analyze (A, &c) ;                   // analyze
    cholmod_factorize (A, L, &c) ;                  // factorize
    cholmod_print_factor (L, "L", &c) ;             // print the factorization
    x = cholmod_solve (CHOLMOD_A, L, b, &c) ;       // solve Ax=b
    cholmod_print_dense (x, "x", &c) ;              // print the solution
    r = cholmod_copy_dense (b, &c) ;                // r = b
#ifndef NMATRIXOPS
    cholmod_sdmult (A, 0, m1, one, x, r, &c) ;      // r = r-Ax
    double rnorm = cholmod_norm_dense (r, 0, &c) ;  // compute inf-norm of r
    double anorm = cholmod_norm_sparse (A, 0, &c) ; // compute inf-norm of A
    printf ("\n%s precision results:\n", dtype ? "single" : "double") ;
    printf ("norm(b-Ax) %8.1e\n", rnorm) ;
    printf ("norm(A)    %8.1e\n", anorm) ;
    double relresid = rnorm / anorm ;
    printf ("resid: norm(b-Ax)/norm(A) %8.1e\n", relresid) ;
    fprintf (stderr, "resid: norm(b-Ax)/norm(A) %8.1e\n", relresid) ;
#else
    printf ("residual norm not computed (requires CHOLMOD/MatrixOps)\n") ;
#endif
    cholmod_free_factor (&L, &c) ;                  // free matrices
    cholmod_free_sparse (&A, &c) ;
    cholmod_free_dense (&r, &c) ;
    cholmod_free_dense (&x, &c) ;
    cholmod_free_dense (&b, &c) ;
    cholmod_finish (&c) ;                           // finish CHOLMOD
    return (0) ;
}
