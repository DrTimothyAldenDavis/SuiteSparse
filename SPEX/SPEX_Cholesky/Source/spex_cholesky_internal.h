//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_cholesky_internal.h: include file for internal use in SPEX_Cholesky
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// This file is not intended to be #include'd in user applications.  Use
// SPEX.h instead.

#ifndef SPEX_CHOL_INTERNAL_H
#define SPEX_CHOL_INTERNAL_H

// Definition of SPEX macros, SPEX data structures, etc
#include "spex_util_internal.h"

// ============================================================================
//                           Internal Functions
// ============================================================================

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//---------Routines to compute and anayze the elimination tree------------------
// ----These routines are taken and lightly modified from Tim Davis' Csparse----
// -------------------------www.suitesparse.com---------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Routines to compute and postorder the etree
//------------------------------------------------------------------------------

/* Purpose: Compute the elimination tree of A */

SPEX_info spex_cholesky_etree
(
    // Output
    int64_t **tree_handle,      // On output: contains the elimination tree of A
                                // On input: undefined.
    // Input
    const SPEX_matrix A         // Input matrix (must be SPD).
) ;

/* Purpose: post order a forest */

SPEX_info spex_cholesky_post
(
    // Output
    int64_t **post_handle, // On output: post-order of the forest
                           // On input: undefied
    // Input
    const int64_t *parent, // Parent[j] is parent of node j in forest
    const int64_t n        // Number of nodes in the forest
) ;

/* Purpose: Depth-first search and postorder of a tree rooted at node j */

SPEX_info spex_cholesky_tdfs
(
    int64_t *k,         // Index (kth node)
    const int64_t j,    // Root node
    int64_t *head,      // Head of list
    int64_t *next,      // Next node in the list
    int64_t *post,      // Post ordered tree
    int64_t *stack      // Stack of nodes
) ;

//------------------------------------------------------------------------------
// Routines to compute the column counts (number of nonzeros per column) of L
//------------------------------------------------------------------------------

/* Purpose: consider A(i,j), node j in ith row subtree and return lca(jprev,j)
   Used to determine Column counts of cholesky factor*/

SPEX_info spex_cholesky_leaf
(
    int64_t *lca_handle,    // Least common ancestor (jprev,j)
    const int64_t i,        // Index (subtree i)
    const int64_t j,        // Index (node j)
    const int64_t *first,   // first[j] is the first descendant of node j
    int64_t *maxfirst,      // maxfirst[j] is the maximum first descendant of
                            // node j
    int64_t *prevleaf,      // prevleaf[i] is the previous leaf of ith subtree
    int64_t *ancestor,      // ancestor[i] is the ancestor of ith subtree
    int64_t *jleaf          // indicates whether j is the first leaf (value of
                            // 1) or not (value of 2)
) ;

/* Purpose: Obtain the column counts of an SPD matrix for Cholesky factorization
 * This is a modified version of Csparse's cs_chol_counts function
 */

SPEX_info spex_cholesky_counts
(
    // Output
    int64_t **c_handle,     // On ouptut: column counts
                            // On input: undefined
    // Input
    const SPEX_matrix A,    // Input matrix
    const int64_t *parent,  // Elimination tree
    const int64_t *post     // Post-order of the tree
) ;

//------------------------------------------------------------------------------
// Routine to compute the reach (nonzeros of L) using the etree
//------------------------------------------------------------------------------

/* Purpose: This function computes the reach of the kth row of A on the
 * elimination tree of A.
 * On input, k is the iteration of the algorithm, parent contains the
 * elimination tree and w is workspace.
 * On output, xi[top_handle..n-1] contains the nonzero pattern of the
 * kth row of L (or the kth column of L')
 */

SPEX_info spex_cholesky_ereach
(
    // Output
    int64_t *top_handle,    // On output: starting point of nonzero pattern
                            // On input: undefined
    int64_t *xi,            // On output: contains the nonzero pattern in
                            // xi[top..n-1]
                            // On input: undefined
    // Input
    const SPEX_matrix A,    // Matrix to be analyzed
    const int64_t k,        // Node to start at
    const int64_t *parent,  // Elimination tree of A
    int64_t *w              // Workspace array
) ;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------Internal REF Chol Factorization Routines-------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


/* Purpose: Perform the up-looking Cholesky factorization */

SPEX_info spex_cholesky_up_factor
(
    // Output
    SPEX_matrix* L_handle,     // Lower triangular matrix. NULL on input.
    SPEX_matrix* rhos_handle,  // Sequence of pivots. NULL on input.
    // Input
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                               // elimination tree of A, the column pointers of
                               // L, and the exact number of nonzeros of L.
    const SPEX_matrix A,       // Matrix to be factored
    const SPEX_options option  // command options
) ;

/* Purpose: Perform the left-looking Cholesky factorization*/

SPEX_info spex_cholesky_left_factor
(
    // Output
    SPEX_matrix *L_handle,    // Lower triangular matrix. NULL on input.
    SPEX_matrix *rhos_handle, // Sequence of pivots. NULL on input.
    // Input
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                               // elimination tree of A, the column pointers of
                               // L, and the exact number of nonzeros of L.
    const SPEX_matrix A,       // Matrix to be factored
    const SPEX_options option  // command options
) ;

/* Purpose: This function performs a symbolic left-looking factorization.
 * On input, A is the matrix to be factored, parent contains the elimination
 * tree and S contains the row/column permutations and number of nonzeros in L
 * On output, L_handle is allocated to contain the nonzero pattern of L and
 * memory for the values.
 */

SPEX_info spex_cholesky_pre_left_factor
(
    // Output
    SPEX_matrix *L_handle,        // On output: partial L matrix
                                  // On input: undefined
    // Input
    int64_t *xi,                  // Workspace nonzero pattern vector
    const SPEX_matrix A,          // Input Matrix
    const SPEX_symbolic_analysis S  // Symbolic analysis struct containing the
                                  // number of nonzeros in L, the elimination
                                  // tree, the row/coluimn permutation and its
                                  // inverse
) ;

/* Purpose: This function performs the symmetric sparse REF triangular solve.
 * i.e.,(LD) x = A(:,k).
 */

SPEX_info spex_cholesky_left_triangular_solve
(
    // Output
    int64_t *top_output,     // On output: the beginning of nonzero pattern of
                             // L(:,k). The nonzero pattern is contained in
                             // xi[top_output...n-1]
                             // On input: undefined
    SPEX_matrix x,           // On output: Solution of LD x = A(:,k) ==> kth row
                             // of L but really, the ONLY valid values of x are
                             // those in x[xi] since x is a working vector its
                             // other positions are jumbled.
    int64_t *xi,             // On output: Nonzero pattern vector
    // Input
    const SPEX_matrix L,     // Partial L matrix
    const SPEX_matrix A,     // Input matrix
    const int64_t k,         // Iteration of algorithm
    const SPEX_matrix rhos,  // Partial sequence of pivots
    int64_t *h,              // History vector
    const int64_t *parent,   // Elimination tree
    int64_t *c               // Column pointers of L but they don't point to the
                             // top position of each column of L. Instead they
                             // point to the position on each column where the
                             // next value of L will be grabbed, since at
                             // iteration k we need to grab the kth of L in
                             // order to not recompute those values.
) ;

/* Purpose: This function performs the symmetric sparse REF triangular solve.
 * for uplooking Cholesky factorization. i.e., (LD) x = A(1:k-1,k).  At the
 * given iteration k it computes the k-th column of L' (k-th row of L)
 */

SPEX_info spex_cholesky_up_triangular_solve
(
    //Output
    int64_t *top_output,            // On input NULL. On output contains the
                                    // beginning of nonzero pattern
                                    // The nonzero pattern is contained in
                                    // xi[top_output...n-1]
    int64_t *xi,                    // Nonzero pattern vector
    SPEX_matrix x,                  // Solution of system ==> kth row of L
    // Input
    const SPEX_matrix L,            // Partial L matrix
    const SPEX_matrix A,            // Input matrix
    const int64_t k,                // Iteration of algorithm
    const int64_t *parent,          // Elimination tree
    int64_t *c,                     // Column pointers
    const SPEX_matrix rhos,         // sequence of pivots
    int64_t *h                      // History vector
) ;

/* Purpose: This function performs sparse REF forward substitution for Cholesky
 * factorization.  On input, x contains the righ hand side vectors, L is the
 * Cholesky factor of A and rhos is the sequence of pivots used during
 * factorization.  On output, x contains the solution to LD x = x Note that
 * this function assumes that x is stored as a dense matrix
 */

SPEX_info spex_cholesky_forward_sub
(
    // Input/Output
    SPEX_matrix x,               // Right hand side matrix.
                                 // On input: contains b
                                 // On output: contains the solution of LD x = x
    // Input
    const SPEX_matrix L,         // REF Cholesky factor of A (lower triangular)
    const SPEX_matrix rhos       // Sequence of pivots used in factorization
) ;

/* Purpose: This solves the system L'x = b for Cholesky factorization
 * On input, x contains the scaled solution of L D x = b and L is the
 * REF Cholesky factor of A.
 * On output, x is the solution to the linear system Ax = (det A)b.
 */

SPEX_info spex_cholesky_backward_sub
(
    // Output
    SPEX_matrix x,          // Solution vector to A x = det(A) * b
    // Input
    const SPEX_matrix L     // The lower triangular matrix
) ;

/* Purpose: Matrix preordering for integer-preserving Cholesky factorization.
 * On input, S is undefined
 * On output, S contains the row/column permutation of A
 */

SPEX_info spex_cholesky_preorder
(
    // Output
    SPEX_symbolic_analysis *S_handle,   // Symbolic analysis data structure
                                        // On input: undefined
                                        // On output: contains the
                                        // row/column permutation and its
                                        // inverse.
    // Input
    const SPEX_matrix A,            // Input matrix
    const SPEX_options option       // Control parameters (use default if NULL)
) ;

/* Purpose: Permute the matrix A and return PAP = PAP'
 * On input PAP is undefined and A contains the input matrix
 * On output PAP contains the permuted matrix (PAP')
 */

SPEX_info spex_cholesky_permute_A
(
    //Output
    SPEX_matrix* PAP_handle,   // On input: undefined
                               // On output: contains the permuted matrix
    //Input
    const SPEX_matrix A,       // Input matrix
    const bool numeric,        // True if user wants to permute pattern and
                               // numbers, false if only pattern
    const SPEX_symbolic_analysis S  // Symbolic analysis struct that contains
                                // row/column permutations
) ;

/* Purpose: perform the symbolic analysis for the SPEX Cholesky factorization,
 * that is, computing and postordering the elimination tree, getting the column
 * counts of the SPD matrix A, setting the column pointers and exact number of
 * non zeros of L.
 */

SPEX_info spex_cholesky_symbolic_analysis
(
    //Output
    SPEX_symbolic_analysis S,  // Symbolic analysis
    //Input
    const SPEX_matrix A,       // Matrix to be factored
    const SPEX_options option  // Command options
) ;

/* Purpose: Compute the REF Cholesky factorization A = LDL'
 * only appropriate if A is SPD.
 * On input A contains the user's matrix, option->algo indicates which
 * factorization algorithm is used; up-looking (default) or left-looking
 * On output, L contains the REF Cholesky factor of A, rhos contains
 * the REF Cholesky pivot elements and S contains the elimination tree
 * lower triangular matrix and rhos contains the pivots' values
 * used in the factorization
 */

SPEX_info spex_cholesky_factor
(
    // Output
    SPEX_factorization *F_handle,   // Cholesky factorization
    //Input
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                               // elimination tree of A, the column pointers of
                               // L, and the exact number of nonzeros of L.
    const SPEX_matrix A,       // Matrix to be factored
    const SPEX_options option  // Command options
                               // Notably, option->chol_type indicates whether
                               // CHOL_UP (default) or CHOL_LEFT is used.
) ;

#endif
