//------------------------------------------------------------------------------
// SPEX_Cholesky/SPEX_ldl_analyze: Perform the symbolic analysis of A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: perform the symbolic analysis of A for the ldl factorization,
 * that is, preordering A, computing the elimination tree, getting the column
 * counts of A, setting the column pointers and exact number of non zeros of L.
 * 
 * Note that symbolic analysis for LDL and Cholesky are identical. For simplicity,
 * the LDL analyze function just calls the Cholesky analyze
 *
 * Input arguments of the function:
 *
 * S:           Symbolic analysis struct for ldl factorization.
 *              On input it's NULL
 *              On output it contains the row/column permutation, the elimination
 *              tree, and the number of nonzeros in L.
 *
 * A:           User's input matrix (Must be SPEX_MPZ and SPEX_CSC)
 *
 * option:      Command options (Default if NULL)
 *
 */


#include "spex_cholesky_internal.h"

SPEX_info SPEX_ldl_analyze
(
    // Output
    SPEX_symbolic_analysis *S_handle, // Symbolic analysis data structure
    // Input
    const SPEX_matrix A,        // Input matrix. Must be SPEX_MPZ and SPEX_CSC
    const SPEX_options option   // Command options (Default if NULL)
)
{
    SPEX_info info;
    // SPEX_ldl_analyze is identical to SPEX_chol_analyze
    // For simplicity, we just call the chol function
    info = SPEX_cholesky_analyze( S_handle, A, option);
    (*S_handle)->kind = SPEX_LDL_FACTORIZATION;
    return info;
}

