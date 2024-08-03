//------------------------------------------------------------------------------
// SPEX_Cholesky/SPEX_cholesky_analyze: Perform the symbolic analysis of A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: perform the symbolic analysis of A for the Cholesky factorization,
 * that is, preordering A, computing the elimination tree, getting the column
 * counts of A, setting the column pointers and exact number of non zeros of L.
 *
 * Input arguments of the function:
 *
 * S:           Symbolic analysis struct for Cholesky factorization.
 *              On input it's NULL.  On output it contains the row/column
 *              permutation, the elimination tree, and the number of nonzeros
 *              in L.
 *
 * A:           User's input matrix (Must be SPEX_MPZ and SPEX_CSC)
 *
 * option:      Command options (Default if NULL)
 *
 */

#define SPEX_FREE_WORKSPACE         \
{                                   \
    SPEX_matrix_free(&PAP, NULL);   \
}

#define SPEX_FREE_ALL                               \
{                                                   \
    SPEX_FREE_WORKSPACE ;                           \
    SPEX_symbolic_analysis_free (&S, option);       \
}

#include "spex_cholesky_internal.h"

SPEX_info SPEX_cholesky_analyze
(
    // Output
    SPEX_symbolic_analysis *S_handle, // Symbolic analysis data structure
    // Input
    const SPEX_matrix A,        // Input matrix. Must be SPEX_MPZ and SPEX_CSC
    const SPEX_options option   // Command options (Default if NULL)
)
{
    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_CHOL_LEFT
        && algo != SPEX_CHOL_UP)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    SPEX_info info;
    // SPEX Cholesky analyze just calls symmetric analyze
    info = spex_symmetric_analyze( S_handle, A, option);
    if (info == SPEX_OK) (*S_handle)->kind = SPEX_CHOLESKY_FACTORIZATION;
    return info;
}

