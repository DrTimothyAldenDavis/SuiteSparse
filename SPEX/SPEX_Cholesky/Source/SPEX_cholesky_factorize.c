//------------------------------------------------------------------------------
// SPEX_Cholesky/SPEX_cholesky_factorize: Perform SPEX Chol factorization of A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the integer preserving Cholesky factorization.
 * First it permutes the input matrix according to the symbolic analysis.
 * Then it performs the left-looking or up-looking integer-preserving Cholesky
 * factorization. In order to compute the L matrix, it performs n iterations of
 * a sparse REF symmetric triangular solve function. The overall factorization
 * is PAP' = LDL'
 *
 *
 * Input arguments of the function:
 *
 * F_handle:    Handle to the factorization struct. Null on input.
 *              On output, contains a pointer to the factorization.
 *
 * A:           User's input matrix. Must be SPEX_MPZ and SPEX_CSC.
 *
 * S:           Symbolic analysis struct for Cholesky factorization.
 *              On input it contains the elimination tree and
 *              the number of nonzeros in L.
 *
 * option:      Command options. Default if NULL. Notably, option->chol_type
 *              indicates whether it is performing the default up-looking
 *              factorization (SPEX_CHOL_UP) or the left-looking factorization
 *              (SPEX_CHOL_LEFT).
 */

#define SPEX_FREE_WORKSPACE             \
{                                       \
    SPEX_matrix_free (&PAP, option);   \
}

#define SPEX_FREE_ALL                   \
{                                       \
    SPEX_FREE_WORKSPACE ;               \
}

#include "spex_cholesky_internal.h"

SPEX_info SPEX_cholesky_factorize
(
    // Output
    SPEX_factorization *F_handle,   // Cholesky factorization struct
    //Input
    const SPEX_matrix A,            // Matrix to be factored. Must be SPEX_MPZ
                                    // and SPEX_CSC
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                                    // elimination tree of A, the column
                                    // pointers of L, and the exact number of
                                    // nonzeros of L.
    const SPEX_options option       // command options.
                                    // Notably, option->chol_type indicates
                                    // whether CHOL_UP (default) or CHOL_LEFT
                                    // is used.
)
{

    SPEX_info info;

    if (!spex_initialized())
    {
        return SPEX_PANIC;
    }

    // Check inputs for NULL
    if (!F_handle || !A || !S)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    // Ensure inputs are in the correct format
    if (A->kind != SPEX_CSC || A->type != SPEX_MPZ
        || S->kind != SPEX_CHOLESKY_FACTORIZATION)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    SPEX_matrix PAP = NULL ;
    SPEX_factorization F = NULL ;

    //--------------------------------------------------------------------------
    // Numerically permute matrix A, that is apply the row/column ordering from
    // the symbolic analysis step to get the permuted matrix PAP.
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_cholesky_permute_A(&PAP, A, true, S));

    //--------------------------------------------------------------------------
    // Factorization: Perform the REF Cholesky factorization of
    // A. By default, up-looking Cholesky factorization is done; however,
    // the left looking factorization is done if option->algo=SPEX_CHOL_LEFT
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_cholesky_factor(&F, S, PAP, option));

    //--------------------------------------------------------------------------
    // Set F_handle = F, free all workspace and return success
    //--------------------------------------------------------------------------

    (*F_handle) = F ;
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
