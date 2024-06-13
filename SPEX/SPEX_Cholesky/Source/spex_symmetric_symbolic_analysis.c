//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_symbolic_analysis: Symbolic analysis for Cholesky
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE         \
{                                   \
    SPEX_FREE(post);                \
    SPEX_FREE(c);                   \
}

# define SPEX_FREE_ALL               \
{                                    \
    SPEX_FREE_WORKSPACE              \
}

#include "spex_cholesky_internal.h"

/* Purpose: perform the symbolic analysis for the SPEX Cholesky or LDL
 * factorization, that is, computing and postordering the elimination tree,
 * getting the column counts of the symmetric matrix A, setting the column
 * pointers and exact number of non zeros of L.
 *
 * IMPORTANT: This function assumes that A has already been permuted.
 *
 * Input arguments of the function:
 *
 * S:           Symbolic analysis struct for Cholesky factorization.
 *              On input it contains information that is not used in this
 *              function such as the row/column permutation
 *              On output it contains the elimination tree and
 *              the number of nonzeros in L.
 *
 * A:           The user's permuted input matrix
 *
 * option:      Command options
 */

SPEX_info spex_symmetric_symbolic_analysis
(
    //Output
    SPEX_symbolic_analysis S,  // Symbolic analysis
    //Input
    const SPEX_matrix A,       // Matrix to be factored
    const SPEX_options option  // Command options
)
{

    SPEX_info info;

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    // Inputs are checked by the caller, asserts are here as a reminder of the
    // format
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);

    // Declare local variables
    int64_t n = A->n;
    int64_t *post = NULL;
    int64_t *c = NULL;

    // Obtain elimination tree of A
    SPEX_CHECK( spex_symmetric_etree(&S->parent, A) );

    // Postorder the elimination tree of A
    SPEX_CHECK( spex_symmetric_post(&post, S->parent, n) );

    // Get the column counts of A
    SPEX_CHECK( spex_symmetric_counts(&c, A, S->parent, post) );

    // Set the column pointers of L
    S->cp = (int64_t*) SPEX_malloc( (n+1)*sizeof(int64_t));
    if (S->cp == NULL)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    SPEX_CHECK( spex_cumsum(S->cp, c, n));

    // Set the exact number of nonzeros in L
    S->lnz = S->cp[n];
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
