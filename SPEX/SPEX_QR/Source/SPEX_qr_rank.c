//------------------------------------------------------------------------------
// SPEX_QR/SPEX_qr_rank: give the rank of A
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SPEX QR factorization to exactly calculate
 * the rank of a given matrix A. If A is square or rectangular with more
 * rows than columns the REF QR factorization A = Q D R is computed to find rank.
 * Conversely, if A is rectangular with more columns than rows, the REF QR
 * factorization A^T = Q D R is computed
 *
 * Input/Output arguments:
 *
 * rank:        On output contains the rank of A
 *
 * A:           User's input matrix. It must be populated prior to calling this
 *
 * option:      Struct containing various command parameters for the
 *              factorization. If NULL on input, default values are used.
 */

#define SPEX_FREE_WORKSPACE               \
    {                                     \
        SPEX_matrix_free(&AT, option);    \
        SPEX_factorization_free(&F, NULL); \
        SPEX_symbolic_analysis_free(&S, NULL); \
    }

#define SPEX_FREE_ALL   \
    SPEX_FREE_WORKSPACE \

#include "spex_qr_internal.h"

SPEX_info SPEX_qr_rank
(
        // Output
        int64_t *rank,
        // Input
        const SPEX_matrix A,
        const SPEX_options option
)
{
    //-------------------------------------------------------------------------
    // check inputs
    //-------------------------------------------------------------------------
    SPEX_info info;
    if (!spex_initialized())
        return (SPEX_PANIC);

    // A must be the appropriate dimension
    if (A->n == 0 || A->m == 0 )
    {
        return SPEX_INCORRECT_INPUT;
    }

    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_QR_GS)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    SPEX_REQUIRE(A, SPEX_CSC, SPEX_MPZ);

    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL;
    SPEX_matrix AT = NULL;

    // Determine if we need to factorize A or AT
    if (A->n > A->m)
    {
        // A is short and wide. Factorize AT

        // Compute AT
        SPEX_CHECK( SPEX_transpose(&AT, A, true, option));
        // Symbolic analysis of AT
        SPEX_CHECK(SPEX_qr_analyze(&S, AT, option));
        // Factorize AT
        SPEX_CHECK(SPEX_qr_factorize(&F, AT, S, option));
    }
    else
    {
        // A is tall and skinny. Factorize A
        // Symbolic analysis of A
        SPEX_CHECK(SPEX_qr_analyze(&S, A, option));
        // Factorize A
        SPEX_CHECK(SPEX_qr_factorize(&F, A, S, option));
    }

    // Set the rank
    (*rank) = F->rank;

    //--------------------------------------------------------------------------
    // Free memory and return ok
    //--------------------------------------------------------------------------

    SPEX_FREE_WORKSPACE;
    return (SPEX_OK);
}
