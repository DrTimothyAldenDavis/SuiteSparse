//------------------------------------------------------------------------------
// SPEX_Left_LU/Include/SPEX.h: user #include file for SPEX_Left_LU.
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2023, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#ifndef SPEX_LEFT_LU_H
#define SPEX_LEFT_LU_H

// This software package exactly solves a sparse system of linear equations
// using the SPEX Left LU factorization. This code accompanies the paper (submitted
// to ACM Transactions on Mathematical Software):

//    "Algorithm 1021: SPEX Left LU: Exactly Solving Sparse Linear Systems via
//    A Sparse Left-Looking Integer-Preserving LU Factorization", C. Lourenco,
//    J. Chen, E. Moreno-Centeno, T. Davis, ACM Trans. Mathematical Software,
//    June 2022. https://doi.org/10.1145/3519024

//    The theory associated with this software can be found in the paper
//    (published in SIAM journal on matrix analysis and applications):

//    "Exact Solution of Sparse Linear Systems via Left-Looking
//     Roundoff-Error-Free LU Factorization in Time Proportional to
//     Arithmetic Work", C. Lourenco, A. R. Escobedo, E. Moreno-Centeno,
//     T. Davis, SIAM J. Matrix Analysis and Applications.  pp 609-638,
//     vol 40, no 2, 2019.

//    If you use this code, you must first download and install the GMP and
//    MPFR libraries. GMP and MPFR can be found at:
//              https://gmplib.org/
//              http://www.mpfr.org/

//    If you use SPEX Left LU for a publication, we request that you please cite
//    the above two papers.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Authors----------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//    Christopher Lourenco, Jinhao Chen, Erick Moreno-Centeno, and Timothy Davis
//

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Contact Information----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//    Please contact Chris Lourenco (chrisjlourenco@gmail.com)
//    or Tim Davis (timdavis@aldenmath.com, DrTimothyAldenDavis@gmail.com,
//                  davis@tamu.edu)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Copyright--------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//    SPEX Left LU is free software; you can redistribute it and/or modify
//     it under the terms of either:
//
//        * the GNU Lesser General Public License as published by the
//          Free Software Foundation; either version 3 of the License,
//          or (at your option) any later version.
//
//     or
//
//        * the GNU General Public License as published by the Free Software
//          Foundation; either version 2 of the License, or (at your option) any
//          later version.
//
//    or both in parallel, as here.
//
//    See license.txt for license info.
//
// This software is copyright by Christopher Lourenco, Jinhao Chen, Erick
// Moreno-Centeno and Timothy A. Davis. All Rights Reserved.
//

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//---------------------------DISCLAIMER-----------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// SPEX Left LU is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//--------------------------Summary---------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//    This software package solves the linear system Ax = b exactly. The input
//    matrix and right hand side vectors are stored as either integers, double
//    precision numbers, multiple precision floating points (through the mpfr
//    library) or as rational numbers (as a collection of numerators and
//    denominators using the GMP mpq_t data structure). Appropriate routines
//    within the code transform the input into an integral matrix in compressed
//    column form.

//    This package computes the factorization PAQ = LDU. Note that we store the
//    "functional" form of the factorization by only storing L and U. The user
//    is given some freedom to select the permutation matrices P and Q. The
//    recommended default settings select Q using the COLAMD column ordering
//    and select P via a partial pivoting scheme in which the diagonal entry
//    in column k is selected if it is the same magnitude as the smallest
//    entry, otherwise the smallest entry is selected as the kth pivot.
//    Alternative strategies allowed to select Q include the AMD column
//    ordering or no column permutation (Q=I).  For pivots, there are a variety
//    of potential schemes including traditional partial pivoting, diagonal
//    pivoting, tolerance pivoting etc. This package does not allow pivoting
//    based on sparsity criterion.

//    The factors L and U are computed via integer preserving operations via
//    integer-preserving Gaussian elimination. The key part of this algorithm
//    is a Roundoff Error Free (REF) sparse triangular solve function which
//    exploits sparsity to reduce the number of operations that must be
//    performed.

//    Once L and U are computed, a simplified version of the triangular solve
//    is performed which assumes the vector b is dense. The final solution
//    vector x is gauranteed to be exact. This vector can be output in one of
//    three ways: 1) full precision rational arithmetic (as a sequence of
//    numerators and denominators) using the GMP mpq_t data type, 2) double
//    precision while not exact will produce a solution accurate to machine
//    roundoff unless the size of the associated solution exceeds double
//    precision (i.e., the solution is 10^500 or something), 3) variable
//    precision floating point using the GMP mpfr_t data type. The associated
//    precision is user defined.


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//---------------------Include files required by SPEX Left LU-------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <mpfr.h>
#include "SuiteSparse_config.h"
// #include "SPEX_Util.h"


//------------------------------------------------------------------------------
// SPEX_Util/Include/SPEX_Util.h: Include file for utility functions for SPEX
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#ifndef SPEX_UTIL_H
#define SPEX_UTIL_H

// SPEX_Util is a collection of utility functions for the SParse EXact package.
// Included are several routines for memory management, matrix operations, and 
// wrappers to the GMP library.
//
// This is the global include file and should be included in all SPEX_* packages
//
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Authors----------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Unless otherwise noted all Utility functions are authored by:
//
//    Christopher Lourenco, Jinhao Chen, Erick Moreno-Centeno, and Timothy Davis
//

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Contact Information----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//    Please contact Chris Lourenco (chrisjlourenco@gmail.com, lourenco@usna.edu)
//    or Tim Davis (timdavis@aldenmath.com, DrTimothyAldenDavis@gmail.com,
//                  davis@tamu.edu)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Copyright--------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//    SPEX is free software; you can redistribute it and/or modify
//     it under the terms of either:
//
//        * the GNU Lesser General Public License as published by the
//          Free Software Foundation; either version 3 of the License,
//          or (at your option) any later version.
//
//     or
//
//        * the GNU General Public License as published by the Free Software
//          Foundation; either version 2 of the License, or (at your option) any
//          later version.
//
//    or both in parallel, as here.
//
//    See license.txt for license info.
//
// This software is copyright by Christopher Lourenco, Jinhao Chen, Erick
// Moreno-Centeno and Timothy A. Davis. All Rights Reserved.
//

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//---------------------------DISCLAIMER-----------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// SPEX is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//---------------------Include files required by SPEX --------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// #include <stdio.h>
// #include <stdbool.h>
// #include <stdint.h>
// #include <stdlib.h>
// #include <string.h>
// #include <gmp.h>
// #include <mpfr.h>
#include <math.h>
#include <time.h>
// #include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "SuiteSparse_config.h"

//------------------------------------------------------------------------------
// Version
//------------------------------------------------------------------------------

#define SPEX_DATE "Jan 20, 2024"
#define SPEX_VERSION "2.3.2"
#define SPEX_VERSION_MAJOR 2
#define SPEX_VERSION_MINOR 3
#define SPEX_VERSION_SUB   2

#define SPEX__VERSION SUITESPARSE__VERCODE(2,3,2)
#if !defined (SUITESPARSE__VERSION) || \
    (SUITESPARSE__VERSION < SUITESPARSE__VERCODE(7,6,0))
#error "SPEX 2.3.2 requires SuiteSparse_config 7.6.0 or later"
#endif

#if defined ( __cplusplus )
extern "C"
{
#endif

//------------------------------------------------------------------------------
// version
//------------------------------------------------------------------------------

void SPEX_version (int version [3]) ;

//------------------------------------------------------------------------------
// Error codes
//------------------------------------------------------------------------------

// Most SPEX functions return a code that indicates if it was successful
// or not. Otherwise the code returns a pointer to the object that was created
// or it returns void (in the case that an object was deleted)

typedef enum
{
    SPEX_OK = 0,                // all is well
    SPEX_OUT_OF_MEMORY = -1,    // out of memory
    SPEX_SINGULAR = -2,         // the input matrix A is singular
    SPEX_INCORRECT_INPUT = -3,  // one or more input arguments are incorrect
    SPEX_INCORRECT = -4,        // The solution is incorrect
    SPEX_UNSYMMETRIC = -5,      // The input matrix is unsymmetric (for Cholesky)
    SPEX_PANIC = -6             // SPEX used without proper initialization
}
SPEX_info ;

//------------------------------------------------------------------------------
// Pivot scheme codes
//------------------------------------------------------------------------------

// A code in SPEX_options to tell SPEX what type of pivoting to use for pivoting
// in unsymmetric LU factorization.

typedef enum
{
    SPEX_SMALLEST = 0,      // Smallest pivot
    SPEX_DIAGONAL = 1,      // Diagonal pivoting
    SPEX_FIRST_NONZERO = 2, // First nonzero per column chosen as pivot
    SPEX_TOL_SMALLEST = 3,  // Diagonal pivoting with tol for smallest pivot.
                            //   (Default)
    SPEX_TOL_LARGEST = 4,   // Diagonal pivoting with tol. for largest pivot
    SPEX_LARGEST = 5        // Largest pivot
}
SPEX_pivot ;

//------------------------------------------------------------------------------
// Column ordering scheme codes
//------------------------------------------------------------------------------

// A code in SPEX_options to tell SPEX which column ordering to used prior to 
// exact factorization

typedef enum
{
    SPEX_NO_ORDERING = 0,   // None: A is factorized as-is
    SPEX_COLAMD = 1,        // COLAMD: Default
    SPEX_AMD = 2            // AMD
}
SPEX_col_order ;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Data Structures--------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// This struct serves as a global struct to define all user-selectable options.

typedef struct SPEX_options
{
    SPEX_pivot pivot ;     // row pivoting scheme used.
    SPEX_col_order order ; // column ordering scheme used
    double tol ;           // tolerance for the row-pivotin methods
                           // SPEX_TOL_SMALLEST and SPEX_TOL_LARGEST
    int print_level ;      // 0: print nothing, 1: just errors,
                           // 2: terse (basic stats from COLAMD/AMD and
                           // SPEX Left LU), 3: all, with matrices and results
    int32_t prec ;         // Precision used to output file if MPFR is chosen
    mpfr_rnd_t round ;     // Type of MPFR rounding used
    bool check ;           // Set true if the solution to the system should be
                           // checked.  Intended for debugging only; SPEX is
                           // guaranteed to return the exact solution.
} SPEX_options ;

// Purpose: Create SPEX_options object with default parameters
// upon successful allocation, which are defined in SPEX_util_nternal.h
// To free it, simply use SPEX_FREE (*option).
SPEX_info SPEX_create_default_options (SPEX_options **option) ;


//------------------------------------------------------------------------------
// SPEX_matrix: a sparse CSC, sparse triplet, or dense matrix
//------------------------------------------------------------------------------

// SPEX uses a single matrix data type, SPEX_matrix, which can be held in
// one of three kinds of formats:  sparse CSC (compressed sparse column),
// sparse triplet, and dense:

typedef enum
{
    SPEX_CSC = 0,           // matrix is in compressed sparse column format
    SPEX_TRIPLET = 1,       // matrix is in sparse triplet format
    SPEX_DENSE = 2          // matrix is in dense format
}
SPEX_kind ;

// Each of the three formats can have values of 5 different data types: mpz_t,
// mpq_t, mpfr_t, int64_t, and double:

typedef enum
{
    SPEX_MPZ = 0,           // matrix of mpz_t integers
    SPEX_MPQ = 1,           // matrix of mpq_t rational numbers
    SPEX_MPFR = 2,          // matrix of mpfr_t
    SPEX_INT64 = 3,         // matrix of int64_t integers
    SPEX_FP64 = 4           // matrix of doubles
}
SPEX_type ;

// This gives a total of 15 different matrix types.  Not all functions accept
// all 15 matrices types, however.

// Suppose A is an m-by-n matrix with nz <= nzmax entries.
// The p, i, j, and x components are defined as:

// (0) SPEX_CSC:  A sparse matrix in CSC (compressed sparse column) format.
//      A->p is an int64_t array of size n+1, A->i is an int64_t array of size
//      nzmax (with nz <= nzmax), and A->x.type is an array of size nzmax of
//      matrix entries ('type' is one of mpz, mpq, mpfr, int64, or fp64).  The
//      row indices of column j appear in A->i [A->p [j] ... A->p [j+1]-1], and
//      the values appear in the same locations in A->x.type.  The A->j array
//      is NULL.  A->nz is ignored; nz is A->p [A->n].

// (1) SPEX_TRIPLET:  A sparse matrix in triplet format.  A->i and A->j are
//      both int64_t arrays of size nzmax, and A->x.type is an array of values
//      of the same size.  The kth tuple has row index A->i [k], column index
//      A->j [k], and value A->x.type [k], with 0 <= k < A->nz.  The A->p array
//      is NULL.

// (2) SPEX_DENSE:  A dense matrix.  The integer arrays A->p, A->i, and A->j
//      are all NULL.  A->x.type is a pointer to an array of size m*n, stored
//      in column-oriented format.  The value of A(i,j) is A->x.type [p]
//      with p = i + j*A->m.  A->nz is ignored; nz is A->m * A->n.

// The SPEX_matrix may contain 'shallow' components, A->p, A->i, A->j, and
// A->x.  For example, if A->p_shallow is true, then a non-NULL A->p is a
// pointer to a read-only array, and the A->p array is not freed by
// SPEX_matrix_free.  If A->p is NULL (for a triplet or dense matrix), then
// A->p_shallow has no effect.

typedef struct
{
    int64_t m ;         // number of rows
    int64_t n ;         // number of columns
    int64_t nzmax ;     // size of A->i, A->j, and A->x
    int64_t nz ;        // # nonzeros in a triplet matrix .
                        // Ignored for CSC and dense matrices.
    SPEX_kind kind ;    // CSC, triplet, or dense
    SPEX_type type ;    // mpz, mpq, mpfr, int64, or fp64 (double)

    int64_t *p ;        // if CSC: column pointers, an array size is n+1.
                        // if triplet or dense: A->p is NULL.
    bool p_shallow ;    // if true, A->p is shallow.

    int64_t *i ;        // if CSC or triplet: row indices, of size nzmax.
                        // if dense: A->i is NULL.
    bool i_shallow ;    // if true, A->i is shallow.

    int64_t *j ;        // if triplet: column indices, of size nzmax.
                        // if CSC or dense: A->j is NULL.
    bool j_shallow ;    // if true, A->j is shallow.

    union               // A->x.type has size nzmax.
    {
        mpz_t *mpz ;            // A->x.mpz
        mpq_t *mpq ;            // A->x.mpq
        mpfr_t *mpfr ;          // A->x.mpfr
        int64_t *int64 ;        // A->x.int64
        double *fp64 ;          // A->x.fp64
    } x ;
    bool x_shallow ;    // if true, A->x.type is shallow.

    mpq_t scale ;       // scale factor for mpz matrices (never shallow)
                        // For all matrices who's type is not mpz,
                        // mpz_scale = 1. 

} SPEX_matrix ;

//------------------------------------------------------------------------------
// SPEX_matrix_allocate: allocate an m-by-n SPEX_matrix
//------------------------------------------------------------------------------

// if shallow is false: All components (p,i,j,x) are allocated and set to zero,
//                      and then shallow flags are all false.

// if shallow is true:  All components (p,i,j,x) are NULL, and their shallow
//                      flags are all true.  The user can then set A->p,
//                      A->i, A->j, and/or A->x accordingly, from their own
//                      arrays.

SPEX_info SPEX_matrix_allocate
(
    SPEX_matrix **A_handle, // matrix to allocate
    SPEX_kind kind,         // CSC, triplet, or dense
    SPEX_type type,         // mpz, mpq, mpfr, int64, or double
    int64_t m,              // # of rows
    int64_t n,              // # of columns
    int64_t nzmax,          // max # of entries
    bool shallow,           // if true, matrix is shallow.  A->p, A->i, A->j,
                            // A->x are all returned as NULL and must be set
                            // by the caller.  All A->*_shallow are returned
                            // as true.
    bool init,              // If true, and the data types are mpz, mpq, or
                            // mpfr, the entries are initialized (using the
                            // appropriate SPEX_mp*_init function). If false,
                            // the mpz, mpq, and mpfr arrays are allocated but
                            // not initialized. Meaningless for data types 
                            // FP64 or INT64
    const SPEX_options *option
) ;

//------------------------------------------------------------------------------
// SPEX_matrix_free: free a SPEX_matrix
//------------------------------------------------------------------------------

SPEX_info SPEX_matrix_free
(
    SPEX_matrix **A_handle, // matrix to free
    const SPEX_options *option
) ;

//------------------------------------------------------------------------------
// SPEX_matrix_nnz: # of entries in a matrix
//------------------------------------------------------------------------------

SPEX_info SPEX_matrix_nnz     // find the # of entries in A
(
    int64_t *nnz,              // # of entries in A, -1 if A is NULL
    const SPEX_matrix *A,      // matrix to query
    const SPEX_options *option
) ;

//------------------------------------------------------------------------------
// SPEX_matrix_copy: makes a copy of a matrix
//------------------------------------------------------------------------------

// SPEX_matrix_copy: make a copy of a SPEX_matrix, into another kind and type.

SPEX_info SPEX_matrix_copy
(
    SPEX_matrix **C_handle, // matrix to create (never shallow)
    // inputs, not modified:
    SPEX_kind C_kind,       // C->kind: CSC, triplet, or dense
    SPEX_type C_type,       // C->type: mpz_t, mpq_t, mpfr_t, int64_t, or double
    SPEX_matrix *A,         // matrix to make a copy of (may be shallow)
    const SPEX_options *option
) ;

//------------------------------------------------------------------------------
// SPEX_matrix macros
//------------------------------------------------------------------------------

// These macros simplify the access to entries in a SPEX_matrix.
// The type parameter is one of: mpq, mpz, mpfr, int64, or fp64.

// To access the kth entry in a SPEX_matrix using 1D linear addressing,
// in any matrix kind (CSC, triplet, or dense), in any type:
#define SPEX_1D(A,k,type) ((A)->x.type [k])

// To access the (i,j)th entry in a 2D SPEX_matrix, in any type:
#define SPEX_2D(A,i,j,type) SPEX_1D (A, (i)+(j)*((A)->m), type)

//------------------------------------------------------------------------------
// SPEX_LU_analysis: symbolic pre-analysis
//------------------------------------------------------------------------------

// This struct stores the column permutation for LU and the estimate of the
// number of nonzeros in L and U.

typedef struct
{
    int64_t *q ;    // Column permutation for LU factorization, representing
                    // the permutation matrix Q.   The matrix A*Q is factorized.
                    // If the kth column of L, U, and A*Q is column j of the
                    // unpermuted matrix A, then j = S->q [k].
    int64_t lnz ;   // Approximate number of nonzeros in L.
    int64_t unz ;   // Approximate number of nonzeros in U.
                    // lnz and unz are used to allocate the initial space for
                    // L and U; the space is reallocated as needed.
} SPEX_LU_analysis ;

// The symbolic analysis object is created by SPEX_LU_analyze.

// SPEX_LU_analysis_free frees the SPEX_LU_analysis object.
SPEX_info SPEX_LU_analysis_free        
(
    SPEX_LU_analysis **S, // Structure to be deleted
    const SPEX_options *option
) ;

// SPEX_LU_analyze performs the symbolic ordering and analysis for LU factorization.
// Currently, there are three options: no ordering, COLAMD, and AMD.
SPEX_info SPEX_LU_analyze
(
    SPEX_LU_analysis **S, // symbolic analysis (column permutation and nnz L,U)
    const SPEX_matrix *A, // Input matrix
    const SPEX_options *option  // Control parameters
) ;


//------------------------------------------------------------------------------
// Memory management
//------------------------------------------------------------------------------

// SPEX relies on the SuiteSparse memory management functions,
// SuiteSparse_malloc, SuiteSparse_calloc, SuiteSparse_realloc, and
// SuiteSparse_free.

// Allocate and initialize memory space for SPEX
void *SPEX_calloc
(
    size_t nitems,      // number of items to allocate
    size_t size         // size of each item
) ;

// Allocate memory space for SPEX
void *SPEX_malloc
(
    size_t size        // size of memory space to allocate
) ;

// Free the memory allocated by SPEX_calloc, SPEX_malloc, or SPEX_realloc.
void SPEX_free
(
    void *p         // pointer to memory space to free
) ;

// Free a pointer and set it to NULL.
#define SPEX_FREE(p)                        \
{                                           \
    SPEX_free (p) ;                         \
    (p) = NULL ;                            \
}

// SPEX_realloc is a wrapper for realloc.  If p is non-NULL on input, it points
// to a previously allocated object of size old_size * size_of_item.  The
// object is reallocated to be of size new_size * size_of_item.  If p is NULL
// on input, then a new object of that size is allocated.  On success, a
// pointer to the new object is returned.  If the reallocation fails, p is not
// modified, and a flag is returned to indicate that the reallocation failed.
// If the size decreases or remains the same, then the method always succeeds
// (ok is returned as true).

// Typical usage:  the following code fragment allocates an array of 10 int's,
// and then increases the size of the array to 20 int's.  If the SPEX_malloc
// succeeds but the SPEX_realloc fails, then the array remains unmodified,
// of size 10.
//
//      int *p ;
//      p = SPEX_malloc (10 * sizeof (int)) ;
//      if (p == NULL) { error here ... }
//      printf ("p points to an array of size 10 * sizeof (int)\n") ;
//      bool ok ;
//      p = SPEX_realloc (20, 10, sizeof (int), p, &ok) ;
//      if (ok) printf ("p has size 20 * sizeof (int)\n") ;
//      else printf ("realloc failed; p still has size 10 * sizeof (int)\n") ;
//      SPEX_free (p) ;

void *SPEX_realloc      // pointer to reallocated block, or original block
                        // if the realloc failed
(
    int64_t nitems_new,     // new number of items in the object
    int64_t nitems_old,     // old number of items in the object
    size_t size_of_item,    // sizeof each item
    void *p,                // old object to reallocate
    bool *ok                // true if success, false on failure
) ;

//------------------------------------------------------------------------------
// SPEX memory environment routines
//------------------------------------------------------------------------------

// SPEX_initialize: initializes the working evironment for SPEX library.
// It must be called prior to calling any other SPEX_* function.
SPEX_info SPEX_initialize (void) ;

// SPEX_initialize_expert is the same as SPEX_initialize, except that it allows
// for a redefinition of custom memory functions that are used for SPEX and
// GMP.  The four inputs to this function are pointers to four functions with
// the same signatures as the ANSI C malloc, calloc, realloc, and free.
SPEX_info SPEX_initialize_expert
(
    void* (*MyMalloc) (size_t),             // user-defined malloc
    void* (*MyCalloc) (size_t, size_t),     // user-defined calloc
    void* (*MyRealloc) (void *, size_t),    // user-defined realloc
    void  (*MyFree) (void *)                // user-defined free
) ;

// SPEX_finalize: This function finalizes the working evironment for SPEX
// library, and frees any internal workspace created by SPEX.  It must be
// called as the last SPEX_* function called.
SPEX_info SPEX_finalize (void) ;


// SPEX_matrix_check: check and print a SPEX_sparse matrix
SPEX_info SPEX_matrix_check     // returns a SPEX status code
(
    const SPEX_matrix *A,       // matrix to check
    const SPEX_options* option  // defines the print level
) ;


/* Purpose: This function takes as input a mpz_t SPEX_matrix and divides
 * it by an mpz_t constant storing the solution in a mpq_t dense SPEX_matrix
 * array. 
 */
SPEX_info SPEX_matrix_div // divides the x matrix by a scalar
(
    SPEX_matrix **x2_handle,    // x2 = x/scalar
    SPEX_matrix* x,             // input vector x
    const mpz_t scalar,         // the scalar
    const SPEX_options *option
) ;

/* Purpose: This function multiplies matrix x a scalar
 */
SPEX_info SPEX_matrix_mul   // multiplies x by a scalar
(
    SPEX_matrix *x,         // matrix to be multiplied
    const mpz_t scalar      // scalar to multiply by
) ;


/* SPEX_check_solution: checks the solution of the linear system.  Performs a
 * quick rational arithmetic check of A*x=b.
 */
SPEX_info SPEX_check_solution
(
    const SPEX_matrix *A,          // input matrix
    const SPEX_matrix *x,          // solution vector
    const SPEX_matrix *b,          // right hand side
    const SPEX_options* option           // Command options
);

/* Purpose: p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1]
 * into c.  This function is lightly modified from CSparse.
 */
SPEX_info SPEX_cumsum
(
    int64_t *p,          // vector to store the sum of c
    int64_t *c,          // vector which is summed
    int64_t n           // size of c
);


//------------------------------------------------------------------------------
//---------------------------SPEX GMP/MPFR Functions----------------------------
//------------------------------------------------------------------------------

// The following functions are the SPEX interface to the GMP/MPFR libary.
// Each corresponding GMP/MPFR function is given a wrapper to ensure that no
// memory leaks or crashes occur. All covered GMP functions can be found in
// SPEX_gmp.c

// The GMP library does not handle out-of-memory failures.  However, it does
// provide a mechanism for passing function pointers that replace GMP's use of
// malloc, realloc, and free.  This mechanism is used to provide a try/catch
// mechanism for memory allocation errors, using setjmp and longjmp.

// When a GMP function is called, this wrapper keeps track of a list of objects
// allocated by that function.  The list is started fresh each time a GMP
// function is called.  If any allocation fails, the NULL pointer is not
// returned to GMP.  Instead, all allocated blocks in the list are freed,
// and spex_gmp_allocate returns directly to wrapper.

SPEX_info SPEX_mpfr_asprintf (char **str, const char *format, ... ) ;

SPEX_info SPEX_gmp_fscanf (FILE *fp, const char *format, ... ) ;

SPEX_info SPEX_mpz_init (mpz_t x) ;

SPEX_info SPEX_mpz_init2(mpz_t x, const size_t size) ;

SPEX_info SPEX_mpz_set (mpz_t x, const mpz_t y) ;

SPEX_info SPEX_mpz_set_ui (mpz_t x, const uint64_t y) ;

SPEX_info SPEX_mpz_set_si (mpz_t x, const int64_t y) ;

SPEX_info SPEX_mpz_get_d (double *x, const mpz_t y) ;

SPEX_info SPEX_mpz_get_si (int64_t *x, const mpz_t y) ;

SPEX_info SPEX_mpz_set_q (mpz_t x, const mpq_t y) ;

SPEX_info SPEX_mpz_mul (mpz_t a, const mpz_t b, const mpz_t c) ;

#if 0
SPEX_info SPEX_mpz_add (mpz_t a, const mpz_t b, const mpz_t c) ;

SPEX_info SPEX_mpz_addmul (mpz_t x, const mpz_t y, const mpz_t z) ;
#endif

SPEX_info SPEX_mpz_submul (mpz_t x, const mpz_t y, const mpz_t z) ;

SPEX_info SPEX_mpz_divexact (mpz_t x, const mpz_t y, const mpz_t z) ;

SPEX_info SPEX_mpz_gcd (mpz_t x, const mpz_t y, const mpz_t z) ;

SPEX_info SPEX_mpz_lcm (mpz_t lcm, const mpz_t x, const mpz_t y) ;

SPEX_info SPEX_mpz_abs (mpz_t x, const mpz_t y) ;

SPEX_info SPEX_mpz_cmp (int *r, const mpz_t x, const mpz_t y) ;

SPEX_info SPEX_mpz_cmpabs (int *r, const mpz_t x, const mpz_t y) ;

SPEX_info SPEX_mpz_cmp_ui (int *r, const mpz_t x, const uint64_t y) ;

SPEX_info SPEX_mpz_sgn (int *sgn, const mpz_t x) ;

SPEX_info SPEX_mpz_sizeinbase (size_t *size, const mpz_t x, int64_t base) ;

SPEX_info SPEX_mpq_init (mpq_t x) ;

SPEX_info SPEX_mpq_set (mpq_t x, const mpq_t y) ;

SPEX_info SPEX_mpq_set_z (mpq_t x, const mpz_t y) ;

SPEX_info SPEX_mpq_set_d (mpq_t x, const double y) ;

SPEX_info SPEX_mpq_set_ui (mpq_t x, const uint64_t y, const uint64_t z) ;

SPEX_info SPEX_mpq_set_si (mpq_t x, const int64_t y, const uint64_t z) ;

SPEX_info SPEX_mpq_set_num (mpq_t x, const mpz_t y) ;

SPEX_info SPEX_mpq_set_den (mpq_t x, const mpz_t y) ;

SPEX_info SPEX_mpq_get_den (mpz_t x, const mpq_t y) ;

SPEX_info SPEX_mpq_get_d (double *x, const mpq_t y) ;

SPEX_info SPEX_mpq_abs (mpq_t x, const mpq_t y) ;

SPEX_info SPEX_mpq_add (mpq_t x, const mpq_t y, const mpq_t z) ;

SPEX_info SPEX_mpq_mul (mpq_t x, const mpq_t y, const mpq_t z) ;

SPEX_info SPEX_mpq_div (mpq_t x, const mpq_t y, const mpq_t z) ;

SPEX_info SPEX_mpq_cmp (int *r, const mpq_t x, const mpq_t y) ;

SPEX_info SPEX_mpq_cmp_ui (int *r, const mpq_t x,
                    const uint64_t num, const uint64_t den) ;

SPEX_info SPEX_mpq_sgn (int *sgn, const mpq_t x) ;

SPEX_info SPEX_mpq_equal (int *r, const mpq_t x, const mpq_t y) ;

SPEX_info SPEX_mpfr_init2(mpfr_t x, const uint64_t size) ;

SPEX_info SPEX_mpfr_set (mpfr_t x, const mpfr_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_set_d (mpfr_t x, const double y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_set_si (mpfr_t x, int64_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_set_q (mpfr_t x, const mpq_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_set_z (mpfr_t x, const mpz_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_get_z (mpz_t x, const mpfr_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_get_q (mpq_t x, const mpfr_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_get_d (double *x, const mpfr_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_get_si (int64_t *x, const mpfr_t y, const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_mul (mpfr_t x, const mpfr_t y, const mpfr_t z,
                    const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_mul_d (mpfr_t x, const mpfr_t y, const double z,
                    const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_div_d (mpfr_t x, const mpfr_t y, const double z,
                    const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_ui_pow_ui (mpfr_t x, const uint64_t y, const uint64_t z,
                    const mpfr_rnd_t rnd) ;

SPEX_info SPEX_mpfr_sgn (int *sgn, const mpfr_t x) ;

SPEX_info SPEX_mpfr_free_cache (void) ;

SPEX_info SPEX_mpfr_free_str (char *str) ;

#if 0
// These functions are currently unused, but kept here for future reference.
SPEX_info SPEX_gmp_asprintf (char **str, const char *format, ... ) ;
SPEX_info SPEX_gmp_printf (const char *format, ... ) ;
SPEX_info SPEX_mpfr_printf ( const char *format, ... ) ;
SPEX_info SPEX_gmp_fprintf (FILE *fp, const char *format, ... ) ;
SPEX_info SPEX_mpfr_fprintf (FILE *fp, const char *format, ... ) ;
SPEX_info SPEX_mpz_set_d (mpz_t x, const double y) ;

SPEX_info SPEX_mpfr_log2(mpfr_t x, const mpfr_t y, const mpfr_rnd_t rnd) ;
#endif

#endif


//------------------------------------------------------------------------------
// Primary factorization & solve routines
//------------------------------------------------------------------------------

// SPEX_backslash solves the linear system Ax = b. This is the simplest way to
// use the SPEX Left LU package. This function encompasses both factorization and
// solve and returns the solution vector in the user desired type.  It can be
// thought of as an exact version of MATLAB sparse backslash.
SPEX_info SPEX_Left_LU_backslash
(
    // Output
    SPEX_matrix **X_handle,       // Final solution vector
    // Input
    SPEX_type type,               // Type of output desired:
                                  // Must be SPEX_MPQ, SPEX_MPFR,
                                  // or SPEX_FP64
    const SPEX_matrix *A,         // Input matrix
    const SPEX_matrix *b,         // Right hand side vector(s)
    const SPEX_options* option
) ;

// SPEX_Left_LU_factorize performs the SPEX Left LU factorization. 
// This factorization is done via n iterations of the sparse REF 
// triangular solve function. The overall factorization is 
// PAQ = LDU.  The determinant can be obtained as rhos->x.mpz[n-1].
// 
//  L: undefined on input, created on output
//  U: undefined on input, created on output
//  rhos: undefined on input, created on output
//  pinv: undefined on input, created on output
// 
//  A: input only, not modified
//  S: input only, not modified
//  option: input only, not modified
SPEX_info SPEX_Left_LU_factorize
(
    // output:
    SPEX_matrix **L_handle,     // lower triangular matrix
    SPEX_matrix **U_handle,     // upper triangular matrix
    SPEX_matrix **rhos_handle,  // sequence of pivots
    int64_t **pinv_handle,      // inverse row permutation
    // input:
    const SPEX_matrix *A,       // matrix to be factored
    const SPEX_LU_analysis *S,  // column permutation and estimates
                                // of nnz in L and U 
    const SPEX_options* option
) ;

// SPEX_Left_LU_solve solves the linear system LD^(-1)U x = b.
SPEX_info SPEX_Left_LU_solve         // solves the linear system LD^(-1)U x = b
(
    // Output
    SPEX_matrix **X_handle,     // rational solution to the system
    // input:
    const SPEX_matrix *b,       // right hand side vector
    const SPEX_matrix *A,       // Input matrix
    const SPEX_matrix *L,       // lower triangular matrix
    const SPEX_matrix *U,       // upper triangular matrix
    const SPEX_matrix *rhos,    // sequence of pivots
    const SPEX_LU_analysis *S,  // symbolic analysis struct
    const int64_t *pinv,        // inverse row permutation
    const SPEX_options* option
) ;

#if defined ( __cplusplus )
}
#endif

#endif
