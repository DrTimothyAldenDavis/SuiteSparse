//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_analyze: Perform the symbolic analysis of A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: perform symmetric analysis to obtain row/column permutation for
 * Cholesky and LDL.  That is, preordering A, computing the elimination tree,
 * getting the column counts of A, setting the column pointers and exact number
 * of non zeros of L.
 *
 * Input arguments of the function:
 *
 * S:      Symbolic analysis struct
 *         On input it's NULL
 *         On output it contains the row/column permutation, the elimination
 *         tree, and the number of nonzeros in L.
 *
 * A:      User's input matrix (Must be SPEX_MPZ and SPEX_CSC)
 *
 * option: Command options (Default if NULL)
 *
 */

#define SPEX_FREE_WORKSPACE         \
{                                   \
    SPEX_matrix_free(&PAP, NULL);   \
}

#define SPEX_FREE_ALL                               \
{                                                   \
    SPEX_FREE_WORKSPACE ;                           \
    SPEX_symbolic_analysis_free (&S, option);      \
}

#include "spex_cholesky_internal.h"

SPEX_info spex_symmetric_analyze
(
    // Output
    SPEX_symbolic_analysis *S_handle, // Symbolic analysis data structure
    // Input
    const SPEX_matrix A,        // Input matrix. Must be SPEX_MPZ and SPEX_CSC
    const SPEX_options option   // Command options (Default if NULL)
)
{

    SPEX_info info;
    // SPEX must be initialized
    if (!spex_initialized())
    {
        return SPEX_PANIC;
    }

    // Check inputs
    if ( !S_handle || !A)
    {
        return SPEX_INCORRECT_INPUT;
    }

    // SPEX must be CSC
    SPEX_REQUIRE_KIND(A, SPEX_CSC);

    // Declare permuted matrix and S
    SPEX_matrix PAP = NULL;
    SPEX_symbolic_analysis S = NULL;

    //--------------------------------------------------------------------------
    // Determine if A is indeed symmetric. If so, we try Cholesky.
    // This symmetry check checks for both the nonzero pattern and values.
    //--------------------------------------------------------------------------

    bool is_symmetric ;
    SPEX_CHECK( SPEX_determine_symmetry(&is_symmetric, A, option) );
    if (!is_symmetric)
    {
        SPEX_FREE_WORKSPACE ;
        return SPEX_UNSYMMETRIC ;
    }

    //--------------------------------------------------------------------------
    // Preorder: obtain the row/column ordering of A (Default is AMD)
    //--------------------------------------------------------------------------

    SPEX_CHECK( spex_symmetric_preorder(&S, A, option) );

    //--------------------------------------------------------------------------
    // Permute matrix A, that is apply the row/column ordering from the
    // symbolic analysis step to get the permuted matrix PAP.
    //--------------------------------------------------------------------------

    SPEX_CHECK( spex_symmetric_permute_A(&PAP, A, false, S) );

    //--------------------------------------------------------------------------
    // Symbolic Analysis: compute the elimination tree of PAP
    //--------------------------------------------------------------------------

    SPEX_CHECK( spex_symmetric_symbolic_analysis(S, PAP, option) );

    //--------------------------------------------------------------------------
    // Set output, free all workspace and return success
    //--------------------------------------------------------------------------

    (*S_handle) = S ;
    SPEX_FREE_WORKSPACE ;
    return (SPEX_OK);
}

