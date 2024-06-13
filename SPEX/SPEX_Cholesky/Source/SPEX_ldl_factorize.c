//------------------------------------------------------------------------------
// SPEX_Cholesky/SPEX_ldl_factorize: Perform SPEX LDL factorization of A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the integer preserving LDL factorization.
 * First it permutes the input matrix according to the symbolic analysis.
 * Then it performs the left-looking or up-looking integer-preserving LDL
 * factorization. In order to compute the L matrix, it performs n iterations of
 * a sparse REF symmetric triangular solve function. The overall factorization
 * is PAP' = LDL'
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
 * option:      Command options. Default if NULL. Notably, option->algo
 *              indicates whether it is performing the default up-looking
 *              factorization (SPEX_LDL_UP) or the left-looking factorization
 *              (SPEX_LDL_LEFT).
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

SPEX_info SPEX_ldl_factorize
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
                                    // Notably, option->algo indicates whether
                                    // SPEX_LDL_UP (default) or SPEX_LDL_LEFT
                                    // is used.
)
{

    SPEX_info info;

    if (!spex_initialized())
    {
        return SPEX_PANIC;
    }

    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_LDL_LEFT
        && algo != SPEX_LDL_UP)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    // Check inputs for NULL
    if (!F_handle || !A || !S)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    // Ensure inputs are in the correct format
    if (A->kind != SPEX_CSC || A->type != SPEX_MPZ
        || S->kind != SPEX_LDL_FACTORIZATION)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    SPEX_matrix PAP = NULL ;
    SPEX_factorization F = NULL ;

    //--------------------------------------------------------------------------
    // Numerically permute matrix A, that is apply the row/column ordering from
    // the symbolic analysis step to get the permuted matrix PAP.
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_symmetric_permute_A(&PAP, A, true, S));

    //--------------------------------------------------------------------------
    // Factorization: Perform the REF LDL factorization of
    // A. By default, up-looking factorization is done; however,
    // the left looking factorization is done if option->algo=SPEX_LDL_LEFT
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_symmetric_factor(&F, S, PAP, false, option));

    //--------------------------------------------------------------------------
    // Set F_handle = F, free all workspace and return success
    //--------------------------------------------------------------------------

    (*F_handle) = F ;
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
