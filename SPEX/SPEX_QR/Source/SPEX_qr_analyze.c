//------------------------------------------------------------------------------
// SPEX_QR/SPEX_qr_analyze: Perform the symbolic analysis for QR
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2026, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: perform the symbolic analysis of A for the QR factorization,
 * that is, preordering A, computing the elimination tree, getting the column
 * counts of ATA, setting the column pointers and number of non zeros of R.
 *
 * A must have m >= n
 *
 * Input arguments of the function:
 *
 * S:           Symbolic analysis struct for QR factorization.
 *              On input it's NULL
 *              On output it contains the column permutation, the elimination
 *              tree, and the number of nonzeros in R.
 *
 * A:           User's input matrix (Must be SPEX_MPZ and SPEX_CSC)
 *
 * option:      Command options (Default if NULL)
 *
 */

#define SPEX_FREE_WORKSPACE          \
    {                                \
        SPEX_matrix_free(&AQ, NULL); \
        SPEX_free(post);             \
    }

#define SPEX_FREE_ALL                            \
    {                                            \
        SPEX_FREE_WORKSPACE;                     \
        SPEX_symbolic_analysis_free(&S, option); \
    }

#include "spex_qr_internal.h"

SPEX_info SPEX_qr_analyze
(
    // Output
    SPEX_symbolic_analysis *S_handle, // Symbolic analysis data structure
    // Input
    const SPEX_matrix A,      // Input matrix. Must be SPEX_MPZ and SPEX_CSC
    const SPEX_options option // Command options (Default if NULL)
)
{

    SPEX_info info;
    // SPEX must be initialized
    if (!spex_initialized())
        return SPEX_PANIC;

    // Check inputs
    if (!S_handle || !A)
    {
        return SPEX_INCORRECT_INPUT;
    }

    if (A->n > A->m || A->m <= 0)
    {
        return SPEX_INCORRECT_INPUT;
    }

    // Matrix must be CSC
    SPEX_REQUIRE_KIND(A, SPEX_CSC);

    // Declare permuted matrix and S
    SPEX_matrix AQ = NULL;
    SPEX_symbolic_analysis S = NULL;
    // Declare local variables for symbolic analysis
    int64_t n = A->n, m = A->m;
    int64_t *post = NULL;

    //--------------------------------------------------------------------------
    // Preorder: obtain the row/column ordering of A (Default is COLAMD)
    //--------------------------------------------------------------------------
    SPEX_CHECK(spex_qr_preorder(&S, A, option));

    //--------------------------------------------------------------------------
    // Permute matrix A, that is apply the column ordering from the
    // symbolic analysis step to get the permuted matrix AQ.
    //--------------------------------------------------------------------------
    SPEX_CHECK(spex_qr_permute_A(&AQ, A, false, S->Q_perm, NULL, option));

    //--------------------------------------------------------------------------
    // Symbolic Analysis: compute the column elimination tree of AQ
    //--------------------------------------------------------------------------
    // Obtain elimination tree of AQ
    SPEX_CHECK(spex_qr_etree(&S->parent, AQ));
    // Postorder the column elimination tree of AQ
    SPEX_CHECK(spex_symmetric_post(&post, S->parent, n));

    // Get the column counts of R' aka the row counts of R
    SPEX_CHECK(spex_qr_counts(&(S->cp), &(S->rnz), AQ, S->parent, post));

    //--------------------------------------------------------------------------
    // Set output, free all workspace and return success
    //--------------------------------------------------------------------------

    (*S_handle) = S;
    SPEX_FREE_WORKSPACE;
    return (SPEX_OK);
}
