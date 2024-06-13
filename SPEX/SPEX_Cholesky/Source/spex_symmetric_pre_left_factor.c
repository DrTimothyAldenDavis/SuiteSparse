//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_pre_left_factor: Symbolic left-looking Cholesky
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE         \
{                                   \
    SPEX_FREE(c);                   \
}

# define SPEX_FREE_ALL               \
{                                    \
    SPEX_FREE_WORKSPACE              \
    SPEX_matrix_free(&L, NULL);      \
}

#include "spex_cholesky_internal.h"

/* Purpose: This function performs a symbolic left-looking factorization.
 * On input, A is the matrix to be factored, parent contains the elimination
 * tree and S contains the row/column permutations and number of nonzeros in L.
 * On output, L_handle is allocated to contain the nonzero pattern of L and
 * memory for the values.
 *
 * Importantly, this function assumes that A has already been permuted.
 *
 * Input arguments of the function:
 *
 * L_handle:    A handle to the L matrix. Null on input.
 *              On output, contains a pointer to the partial L matrix.
 *
 * xi:          Workspace nonzero pattern vector. It stores the pattern of
 *              nonzeros of the kth column of L for the triangular solve.
 *
 * A:           The user's permuted input matrix
 *
 * S:            Symbolic analysis struct for Cholesky or LDL factorization.
 *               On input it contains information that is not used in this
 *               function such as the row/column permutation
 *               On output it contains the number of nonzeros in L.
 */

SPEX_info spex_symmetric_pre_left_factor
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
)
{

    // All inputs have been checked by the caller, thus asserts are used here
    // as a reminder of the expected data types
    SPEX_info info;
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(A->type == SPEX_MPZ);

    int64_t  top, k, j, jnew, n = A->n, p = 0;
    int64_t *c = NULL;
    SPEX_matrix L = NULL;
    ASSERT(n >= 0);

    //--------------------------------------------------------------------------
    // Declare memory for L and c
    //--------------------------------------------------------------------------

    // Allocate L
    SPEX_CHECK(SPEX_matrix_allocate(&L, SPEX_CSC, SPEX_MPZ, n, n, S->lnz,
        false, false, NULL));

    // Allocate c
    c = (int64_t*) SPEX_malloc(n* sizeof (int64_t));
    if (!c)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }

    // Set the column pointers of L and c
    for (k = 0; k < n; k++)
    {
        L->p[k] = c[k] = S->cp[k];
    }

    L->i[0] = 0;
    c[0]++;

    //--------------------------------------------------------------------------
    // Iterations 1:n-1
    //--------------------------------------------------------------------------
    for (k = 1; k < n; k++)
    {
        // Obtain nonzero pattern in xi[top..n]
        SPEX_CHECK(spex_symmetric_ereach(&top, xi, A, k, S->parent, c));

        //----------------------------------------------------------------------
        // Iterate accross the nonzeros in x
        //----------------------------------------------------------------------
        for (j = top; j < n; j++)
        {
            jnew = xi[j];
            if (jnew == k) continue;
            p = c[jnew]++;
            // Place the i location of the next nonzero
            L->i[p] = k;
        }
        p = c[k]++;
        L->i[p] = k;
    }
    // Finalize L->p
    L->p[n] = S->lnz;
    (*L_handle) = L;

    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
