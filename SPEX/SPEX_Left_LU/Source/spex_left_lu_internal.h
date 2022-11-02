//------------------------------------------------------------------------------
// SPEX_Left_LU/Source/spex_left_lu_internal: include file for internal use in
// SPEX_Left_LU
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// This file is not intended to be #include'd in user applications.  Use
// SPEX.h instead.

#ifndef SPEX_LEFT_LU_INTERNAL_H
#define SPEX_LEFT_LU_INTERNAL_H

#include "spex_util_internal.h"
#include "SPEX.h"

// ============================================================================
//                           Internal Functions
// ============================================================================


/* Purpose: This function performs sparse REF forward substitution. This is
 * essentially the same as the sparse REF triangular solve applied to each
 * column of the right hand side vectors. Like the normal one, this function
 * expects that the matrix x is dense. As a result,the nonzero pattern is not
 * computed and each nonzero in x is iterated across.  The system to solve is
 * LDx = x.  On output, the mpz_t** x structure is modified.
 */
SPEX_info spex_left_lu_forward_sub
(
    const SPEX_matrix *L,   // lower triangular matrix
    SPEX_matrix *x,         // right hand side matrix
    const SPEX_matrix *rhos // sequence of pivots used in factorization
);

/* Purpose: This function performs sparse REF backward substitution. In essense
 * it solves the sysem Ux = x. Note that prior to this, we expect x to be
 * multiplied by the determinant of A.  The input argument bx is modified on
 * output.
 */
SPEX_info spex_left_lu_back_sub  // performs sparse REF backward substitution
(
    const SPEX_matrix *U,   // input upper triangular matrix
    SPEX_matrix *bx        // right hand side matrix of size n*numRHS
)  ;


/* Purpose: This function performs a depth first search of the graph of the
 * matrix starting at node j. The output of this function is the set of nonzero
 * indices in the xi vector.
 */
void spex_left_lu_dfs // performs a dfs of the graph of the matrix starting at node j
(
    int64_t *top,          // beginning of stack
    int64_t j,             // What node to start DFS at
    SPEX_matrix* L,        // matrix which represents the Graph of L
    int64_t* xi,           // the nonzero pattern
    int64_t* pstack,       // workspace vector
    const int64_t* pinv   // row permutation
);


/* This function performs the pivoting for the SPEX Left LU factorization.
 * The optional Order is:
 *     0: Smallest pivot
 *     1: Natural/Diagonal pivoting
 *     2: Choose first nonzero (not recommended, for comparison only)
 *     3: Diagonal with tolerance and smallest pivot (default)
 *     4: Diagonal with tolerance and largest pivoting
 *     5: Largest pivot (not recommended, for comparison only)
 *
 * On output, the pivs, pinv, and row_perm arrays and rhos matrix are all modified.
 */
SPEX_info spex_left_lu_get_pivot
(
    int64_t *pivot,      // found index of pivot entry
    SPEX_matrix* x,      // kth column of L and U
    int64_t* pivs,       // vector indicating which rows have been pivotal
    int64_t n,           // dimension of the problem
    int64_t top,         // nonzero pattern is located in xi[top..n-1]
    int64_t* xi,         // nonzero pattern of x
    int64_t col,         // current column of A (real kth column i.e., q[k])
    int64_t k,           // iteration of the algorithm
    SPEX_matrix* rhos,   // vector of pivots
    int64_t* pinv,       // row permutation
    int64_t* row_perm,   // opposite of pinv. if pinv[i] = j then row_perm[j] = i
    const SPEX_options *option // command option
);

/* Purpose: This function selects the pivot element as the largest in the
 * column. This is activated if the user sets option->pivot = SPEX_LARGEST.
 * NOTE: This pivoting scheme is NOT recommended for SPEX Left LU.  On output
 * the index of the largest pivot is returned.
 */
SPEX_info spex_left_lu_get_largest_pivot
(
    int64_t *pivot,         // the index of largest pivot
    SPEX_matrix* x,         // kth column of L and U
    int64_t* pivs,          // vector which indicates whether each row has been pivotal
    int64_t n,              // dimension of problem
    int64_t top,            // nonzero pattern is located in xi[top..n-1]
    int64_t* xi             // nonzero pattern of x
);

/* This function obtains the first eligible nonzero pivot.  This is enabled if
 * the user sets option->pivot = SPEX_FIRST_NONZERO.  NOTE: This pivoting
 * scheme is not recommended.  On output, the kth pivot is returned.
 */
SPEX_info spex_left_lu_get_nonzero_pivot // find the first eligible nonzero pivot
(
    int64_t *pivot,      // the index of first eligible nonzero pivot
    SPEX_matrix* x,      // kth column of L and U
    int64_t* pivs,       // vector indicating which rows are pivotal
    int64_t n,           // size of x
    int64_t top,         // nonzero pattern is located in xi[top..n-1]
    int64_t* xi          // nonzero pattern of x
);

/* Purpose: This function selects the pivot element as the smallest in the
 * column. This is activated by default or if the user sets option->pivot =
 * SPEX_SMALLEST.  This is a recommended pivoting scheme for SPEX Left LU.
 * On output, the index of kth pivot is returned.
 */
SPEX_info spex_left_lu_get_smallest_pivot
(
    int64_t *pivot,         // index of smallest pivot
    SPEX_matrix *x,         // kth column of L and U
    int64_t* pivs,          // vector indicating whether each row has been pivotal
    int64_t n,              // dimension of problem
    int64_t top,            // nonzeros are stored in xi[top..n-1]
    int64_t* xi             // nonzero pattern of x
);


/* Purpose: This function permutes b for forward substitution.
 * That is, b = P'*b.
 */
SPEX_info spex_left_lu_permute_b
(
    SPEX_matrix **b_handle,     // permuted RHS vector
    const SPEX_matrix *b2,      // unpermuted RHS vector (not modified)
    const int64_t *pinv,        // inverse row permutation
    const SPEX_options* option
);

/* Purpose: SPEX_permute_x permutes x to get it back in its original form.
 * That is x = Q*x.
 */
SPEX_info spex_left_lu_permute_x
(
    SPEX_matrix **x_handle,    // permuted Solution vector
    SPEX_matrix *x2,           // unpermuted Solution vector (not modified)
    SPEX_LU_analysis *S,  // symbolic analysis with the column ordering Q
    const SPEX_options* option  // Command options
                          // has been checked in the only caller SPEX_Left_LU_solve
) ;

/* Purpose: This function computes the reach of column k of A on the graph of L
 * mathematically that is: xi = Reach(A(:,k))_G_L.
 */
void spex_left_lu_reach    // compute the reach of column k of A on the graph of L
(
    int64_t *top,
    SPEX_matrix* L,         // matrix representing graph of L
    const SPEX_matrix* A,   // input matrix
    int64_t k,              // column of A of interest
    int64_t* xi,            // nonzero pattern
    const int64_t* pinv     // row permutation
)  ;

/* Purpose: This function performs the sparse REF triangular solve; that is,
 * (LD) x = A(:,k). The algorithm is described in the paper; however in essence
 * it computes the nonzero pattern xi, then performs a sequence of IPGE
 * operations on the nonzeros to obtain their final value. All operations are
 * gauranteed to be integral. There are various enhancements in this code used
 * to reduce the overall cost of the operations and minimize operations as much
 * as possible.
 */
SPEX_info spex_left_lu_ref_triangular_solve // performs the sparse REF triangular solve
(
    int64_t *top_output,      // Output the beginning of nonzero pattern
    SPEX_matrix* L,           // partial L matrix
    const SPEX_matrix* A,     // input matrix
    int64_t k,                // iteration of algorithm
    int64_t* xi,              // nonzero pattern vector
    const int64_t* q,         // column permutation
    SPEX_matrix* rhos,        // sequence of pivots
    const int64_t* pinv,      // inverse row permutation
    const int64_t* row_perm,  // row permutation
    int64_t* h,               // history vector
    SPEX_matrix* x            // solution of system ==> kth column of L and U
);

#endif

