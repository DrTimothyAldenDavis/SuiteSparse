//-----------------------------------------------------------------------------
// SPEX_LU/spex_lu_rank_factorize: find rank(A) if A is square
//------------------------------------------------------------------------------

// SPEX_LU: (c) 2019-2026, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This internal function performs the SPEX Left LU factorization,
 * but does not keep the L and U factors because the only purpose is for
 * calculating the rank of a square matrix A. Columns of A are essentially
 * skipped if they are linearly dependent, but otherwise the standard spex LU
 * factorization is computed. If A is full rank, this is equivalent to just
 * a simplified LU factorization but U is never stored because L is only needed
 * for rank computation.
 *
 *  rank: undefined on input, contains rank on output
 *
 *  A: input only, not modified
 *  S: input only, not modified
 *  option: input only, not modified
 */

#define SPEX_FREE_WORKSPACE         \
    SPEX_matrix_free(&x, NULL);     \
    SPEX_FREE(xi);                  \
    SPEX_FREE(h);                   \
    SPEX_FREE(pivs);                \

#define SPEX_FREE_ALL               \
    SPEX_FREE_WORKSPACE;            \
    SPEX_factorization_free(&F, option);

#include "spex_lu_internal.h"

SPEX_info spex_lu_rank_factorize
(
    // output:
    int64_t *rank,                  // rank of A
    // input:
    const SPEX_matrix A,            // matrix to be factored
    const SPEX_symbolic_analysis S, // symbolic analysis
    const SPEX_options option       // command options
)
{

    //--------------------------------------------------------------------------
    // All inputs are checked by the only caller, so no need to check here
    //--------------------------------------------------------------------------
    SPEX_info info;
    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    SPEX_factorization F = NULL ;
    int64_t *xi = NULL ;
    int64_t *h = NULL ;
    int64_t *pivs = NULL ;
    SPEX_matrix x = NULL ;

    int64_t n = A->n ;
    int64_t exact_rank = 0;

    int64_t k = 0, top, i, j, col, loc, lnz = 0, pivot, jnew ;
    size_t size ;

    // allocate memory space for the factorization
    F = (SPEX_factorization) SPEX_calloc(1, sizeof(SPEX_factorization_struct));
    if (F == NULL)
    {
        return SPEX_OUT_OF_MEMORY;
    }
    // set factorization kind
    F->kind = SPEX_LU_FACTORIZATION;

    // Inverse pivot ordering
    F->Pinv_perm = (int64_t*) SPEX_malloc (n * sizeof(int64_t));
    // Actual row permutation, the inverse of pinv. This
    // is used for sorting
    F->P_perm =    (int64_t*) SPEX_malloc (n * sizeof(int64_t));
    // column permutation, to be copied from S->Q_perm
    F->Q_perm =    (int64_t*) SPEX_malloc (n * sizeof(int64_t));

    // Indicator of which rows have been pivotal
    // pivs[i] = 1 if row i has been selected as a pivot
    // row, otherwise, pivs[i] < 0
    pivs = (int64_t*) SPEX_malloc(n* sizeof(int64_t));

    // h is the history vector utilized for the sparse REF
    // triangular solve algorithm. h serves as a global
    // vector which is repeatedly passed into the triangular
    // solve algorithm
    h = (int64_t*) SPEX_malloc(n* sizeof(int64_t));

    // xi is the global nonzero pattern vector. It stores
    // the pattern of nonzeros of the kth column of L and U
    // for the triangular solve.
    xi = (int64_t*) SPEX_malloc(2*n* sizeof(int64_t));

    if (!(F->Pinv_perm) || !(F->P_perm) || !(F->Q_perm) ||
        !pivs || !h || !xi)
    {
        // out of memory: free everything and return
        SPEX_FREE_ALL  ;
        return SPEX_OUT_OF_MEMORY;
    }

    // copy column permutation from symbolic analysis to factorization
    memcpy(F->Q_perm, S->Q_perm, n * sizeof(int64_t));

    // initialize workspace and pivot status
    for (i = 0; i < n; i++)
    {
        h[i] = -1;
        pivs[i] = -1;
        // Initialize location based vectors
        F->Pinv_perm[i] = i;
        F->P_perm[i]    = i;
    }

    //--------------------------------------------------------------------------
    // Declare memory for rhos, L, and U
    //--------------------------------------------------------------------------

    // Create rhos, a global dense mpz_t matrix of dimension n*1
    SPEX_CHECK (SPEX_matrix_allocate(&(F->rhos), SPEX_DENSE, SPEX_MPZ, n, 1, n,
        false, false, option));

    // Allocate L without initializing each entry.
    // L is allocated to have nnz(L) which is estimated by the symbolic
    // analysis. However, unlike traditional matrix allocation, the second
    // boolean parameter here is set to false, so the individual values of
    // L are not allocated. Instead, a more efficient method to
    // allocate these values is done in the factorization to reduce
    // memory usage.
    // Note that U is never allocated because it's not needed for rank computation
    SPEX_CHECK (SPEX_matrix_allocate(&(F->L), SPEX_CSC, SPEX_MPZ, n, n, S->lnz,
        false, false, option));

    //--------------------------------------------------------------------------
    // allocate and initialize the workspace x
    //--------------------------------------------------------------------------

    // SPEX Left LU utilizes arbitrary sized integers which can grow beyond the
    // default 64 bits allocated by GMP. If the integers frequently grow, GMP
    // can get bogged down by performing intermediate reallocations. Instead,
    // we utilize a larger estimate on the workspace x vector so that computing
    // the values in L and U do not require too many extra intemediate calls to
    // realloc.
    //
    // Note that the estimate presented here is not an upper bound nor a lower
    // bound.  It is still possible that more bits will be required which is
    // correctly handled internally.
    int64_t estimate = 64 * SPEX_MAX (2, ceil (log2 ((double) n)));

    // Create x, a global dense mpz_t matrix of dimension n*1. Unlike rhos, the
    // second boolean parameter is set to false to avoid initializing
    // each mpz entry of x with default size.  It is intialized below.
    SPEX_CHECK (SPEX_matrix_allocate(&x, SPEX_DENSE, SPEX_MPZ, n, 1, n,
        false, /* do not initialize the entries of x: */ false, option));

    // initialize the entries of x
    for (i = 0; i < n; i++)
    {
        // Allocate memory for entries of x
        SPEX_MPZ_INIT2(x->x.mpz[i], estimate);
    }

    //--------------------------------------------------------------------------
    // Iterations 0:n-1 (1:n in standard)
    //--------------------------------------------------------------------------

    // Iterations 0:n-1
    for (k = 0; k < n; k++)
    {
        // grab the current column of A
        col = S->Q_perm[k];

        // We overwrite Q_perm so triangular_solve fetches the right column,
        // even though we are telling it we are only on iteration 'exact_rank'.
        // Because exact_rank <= k, we never overwrite future columns.
        F->Q_perm[exact_rank] = col;

        // Start the L column pointer
        F->L->p[exact_rank] = lnz;

        if (lnz + n > F->L->nzmax)
        {
            SPEX_CHECK(spex_sparse_realloc(F->L));
        }

        // Call triangular solve with exact_rank instead of k
        SPEX_CHECK(spex_left_lu_ref_triangular_solve(&top, F->L, A, exact_rank, xi,
            (const int64_t *) (F->Q_perm), F->rhos,
            (const int64_t *) (F->Pinv_perm), (const int64_t *) (F->P_perm),
            h, x));

        // Get pivot using exact_rank
        info = spex_left_lu_get_pivot(&pivot, x, pivs, n, top, xi,
            col, exact_rank, F->rhos, F->Pinv_perm, F->P_perm, option);

        if (info == SPEX_SINGULAR)
        {
            // If singular we do nothing and dont populate L. Column k is thrown
            // away and we move on to the next column.
            continue;
        }

        // Populate L using exact_rank for the boundary
        for (j = top; j < n; j++)
        {
            jnew = xi[j];
            loc = F->Pinv_perm[jnew];

            if (loc >= exact_rank)
            {
                F->L->i[lnz] = jnew;
                SPEX_MPZ_SIZEINBASE(&size, x->x.mpz[jnew], 2);
                SPEX_MPZ_INIT2(F->L->x.mpz[lnz], size+2);
                SPEX_MPZ_SET(F->L->x.mpz[lnz], x->x.mpz[jnew]);
                lnz++;
            }
        }

        exact_rank++;
    }

    // Finalize the last pointer up to exact_rank
    F->L->p[exact_rank] = lnz;
    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    // free everything
    SPEX_FREE_ALL ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*rank) = exact_rank ;
    return (SPEX_OK);
}

