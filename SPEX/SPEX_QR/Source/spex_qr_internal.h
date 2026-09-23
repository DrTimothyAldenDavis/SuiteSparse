//------------------------------------------------------------------------------
// SPEX_QR/Source/spex_qr_internal: include file for internal use in
// SPEX_QR
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// This file is not intended to be #include'd in user applications.  Use
// SPEX.h instead.

#ifndef SPEX_QR_INTERNAL_H
#define SPEX_QR_INTERNAL_H

#include "spex_util_internal.h"
#include "spex_cholesky_internal.h"

// ============================================================================
//                           Internal Functions
// ============================================================================

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------Internal REF QR Analysis Routines--------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/* Purpose: Matrix preordering for integer-preserving QR factorization. */
SPEX_info spex_qr_preorder(
    // Output
    SPEX_symbolic_analysis *S_handle, // Symbolic analysis data structure
                                      // On input: undefined
                                      // On output: contains the
                                      // row/column permutation and its
                                      // inverse.
    // Input
    const SPEX_matrix A,      // Input matrix
    const SPEX_options option // Control parameters (use default if NULL)
);

/* Purpose: Permute the matrix A and return PAQ'
(P can be null so it would only return AQ')*/
SPEX_info spex_qr_permute_A(
    // Output
    SPEX_matrix *PAQ_handle, // On input: undefined
                             // On output: contains the permuted matrix
    // Input
    const SPEX_matrix A,      // Input matrix
    const bool numeric,       // True if user wants to permute pattern and
                              // numbers, false if only pattern
    const int64_t *Q_perm,    // column permutation
    const int64_t *P_perm,    // row permutation
    const SPEX_options option // Command options (Default if NULL)
);

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//---------Routines to compute and anayze the elimination tree------------------
// ----These routines are taken and lightly modified from Tim Davis' Csparse----
// -------------------------www.suitesparse.com---------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/* Purpose: Compute the column elimination tree of ATA */
SPEX_info spex_qr_etree(
    // Output
    int64_t **tree_handle, // On output: contains the elimination tree of A
                           // On input: undefined.
    // Input
    const SPEX_matrix A // Input matrix (must be SPD).
);

/* Purpose: Obtain the column counts for QR factorization */
SPEX_info spex_qr_counts(
    // Output
    int64_t **c_handle, // On ouptut: column counts
                        // On input: undefined
    int64_t *rnz,       // On output: number of nonzeros in R
    // Input
    const SPEX_matrix A,   // Input matrix
    const int64_t *parent, // Elimination tree
    const int64_t *post    // Post-order of the tree
);

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------Internal REF QR Factorization Routines-------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/* Purpose: Obtain the nonzero structure of Q and R for QR factorization */
SPEX_info spex_qr_nonzero_structure(
    // Output
    SPEX_matrix *R_handle, // On output: partial R matrix
                           // On input: undefined
    SPEX_matrix *Q_handle, // On output: partial R matrix
                           // On input: undefined
    // Input
    const SPEX_matrix A,            // Input Matrix
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                                    // number of nonzeros in L, the elimination
                                    // tree, the row/coluimn permutation and its
                                    // inverse
    const SPEX_options option       // Command options
);

/* Purpose: Perfmorm one interation of IPGS-QR.
 * Computes one row of R and updates n-j columns of Q (finalizing the j+1th
 * column)*/
SPEX_info spex_qr_ipgs(
    // Input/Output
    SPEX_matrix R,    // Right triangular matrix
    SPEX_matrix Q,    // Pair-wise orthogonal matrix
    SPEX_matrix rhos, // sequence of pivots
    int64_t *Qj,      // pointers to elements of the jth column of Q
    int64_t *h,       // History vector
    // Output
    bool *isZeros, // True if j+1th column of Q is linearly dependent
    // Input
    const int64_t j,          // Row of R to compute (col j+1 of Q will be finalized)
    const SPEX_matrix A,      // Matrix to be factored
    const int64_t *Q_perm,    // Column permutation
    const SPEX_options option // Command options
);

/* Purpose: Perform sparse REF backward substitution for potenitally rank
 * deficient matrices
 * */
SPEX_info spex_qr_back_sub(
    SPEX_matrix bx,           // right hand side matrix
    const SPEX_matrix R,      // input upper triangular matrix
    const int64_t rank,       // rank of right triangular matrix
    const SPEX_matrix rhos,   // sequence of pivots
    const SPEX_options option // Command options
);

/* Purpose: Perform a transposed factorization and solve if A is
 * rectangular with more columns than rows. This is essentially
 * a caller for analyze and factorize on A^T and then a different
 * solve. This is currently not user visible as it is only called
 * in SPEX_qr_backslash if A is rectangular with more rows than
 * columns and the factorization itself is not user visible. */
SPEX_info spex_qr_transpose_backslash(
    // Output
    SPEX_matrix *x_handle, // Final solution vector
    // Input
    SPEX_type type,           // Type of output desired. Must be
                                  // SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    const SPEX_matrix A,      // Input matrix
    const SPEX_matrix b,      // Right hand side vector(s)
    const SPEX_options option // Command options
);

/* Purpose: Perform a factorization and solve if A is
 * rectangular with more rows than columns. This is the
 * typical use case for SPEX QR and returns either the exact
 * least squares solution or a basic solution. */
SPEX_info spex_qr_standard_backslash(
    // Output
    SPEX_matrix *x_handle, // Final solution vector
    // Input
    SPEX_type type,           // Type of output desired. Must be
                                  // SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    const SPEX_matrix A,      // Input matrix
    const SPEX_matrix b,      // Right hand side vector(s)
    const SPEX_options option // Command options
);

/* Purpose: Solve the rectangular system Ax = b when A has
 * more columns than rows using the factorization of A^T.
 * Specifically we have A^T = Q D R thus R^T D Q^T x = b
 * First the system R^T D y = b is solved via a tranpose solver
 * Then x is computed as Q D y */
SPEX_info spex_qr_transpose_solve(
        // Output
        SPEX_matrix *x_handle, // On input: undefined.
                               // On output: Rational solution (SPEX_MPQ)
                               // to the system.
        // input
        const SPEX_factorization F, // The QR factorization.
        const SPEX_matrix b,        // Right hand side vector
        const SPEX_options option   // command options
);

/* Purpose: Solve the system (R'D) x = x */
SPEX_info spex_qr_transpose_forward_sub
(
    const SPEX_matrix R,    // upper triangular matrix
    const int64_t rank,     // Rank of A which is also number of rows of zeros
    SPEX_matrix x,          // right hand side matrix of size n*numRHS
    const SPEX_matrix rhos  // sequence of pivots used in factorization
);

#endif
