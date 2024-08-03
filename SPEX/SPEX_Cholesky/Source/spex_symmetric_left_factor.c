//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_left_factor: Left-looking REF Chol. factorization
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE         \
{                                   \
    SPEX_matrix_free(&x, NULL);     \
    SPEX_FREE(xi);                  \
    SPEX_FREE(h);                   \
    SPEX_FREE(c);                   \
}

#define SPEX_FREE_ALL               \
{                                   \
    SPEX_matrix_free(&L, NULL);     \
    SPEX_matrix_free(&rhos, NULL);  \
    SPEX_FREE_WORKSPACE             \
}

#include "spex_cholesky_internal.h"

/* Purpose: Perform the left-looking Cholesky or LDL factorization.
 * In order to compute the L matrix, it performs n iterations of a sparse REF
 * symmetric triangular solve function which, at each iteration, computes the
 * kth column of L.
 *
 * Importantly, this function assumes that A has already been permuted.
 *
 * Input arguments of the function:
 *
 * L_handle:    A handle to the L matrix. Null on input.
 *              On output, contains a pointer to the L matrix.
 *
 * rhos_handle: A handle to the sequence of pivots. NULL on input.
 *              On output it contains a pointer to the pivots matrix.
 *
 * S:           Symbolic analysis struct for Cholesky or LDL factorization.
 *              On input it contains information that is not used in this
 *              function such as the row/column permutation
 *              On output it contains the elimination tree and
 *              the number of nonzeros in L.
 *
 * A:           The user's permuted input matrix
 *
 * chol:        True if we are performing a Cholesky factorization and
 *              false if we are performing an LDL factorization
 *
 * option:      Command options
 *
 */


SPEX_info spex_symmetric_left_factor
(
    // Output
    SPEX_matrix *L_handle,    // Lower triangular matrix. NULL on input.
    SPEX_matrix *rhos_handle, // Sequence of pivots. NULL on input.
    // Input
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                               // elimination tree of A, the column pointers of
                               // L, and the exact number of nonzeros of L.
    const SPEX_matrix A,       // Matrix to be factored
    bool chol,                 // If true we are attempting a Cholesky
                               // factorization only and thus the pivot
                               // elements must be >0 If false, we try a
                               // general LDL factorization with the pivot
                               // element strictly != 0.
    const SPEX_options option  // command options
)
{

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    SPEX_info info;
    ASSERT (A != NULL);
    ASSERT (A->type == SPEX_MPZ);
    ASSERT (A->kind == SPEX_CSC);
    ASSERT (L_handle != NULL);
    ASSERT (rhos_handle != NULL);
    (*L_handle) = NULL ;
    (*rhos_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    SPEX_matrix L = NULL ;
    SPEX_matrix rhos = NULL ;
    int64_t *xi = NULL ;
    int64_t *h = NULL ;
    int64_t *c;
    SPEX_matrix x = NULL ;

    // Declare variables
    int64_t n = A->n, top, i, j, lnz = 0, jnew, k;
    int sgn;
    size_t size;

    c = (int64_t*) SPEX_malloc(n* sizeof (int64_t));

    // h is the history vector utilized for the sparse REF
    // triangular solve algorithm. h serves as a global
    // vector which is repeatedly passed into the triangular
    // solve algorithm
    h = (int64_t*) SPEX_malloc(n* sizeof(int64_t));

    // xi serves as a global nonzero pattern vector. It stores
    // the pattern of nonzeros of the kth column of L
    // for the triangular solve.
    xi = (int64_t*) SPEX_malloc(2*n* sizeof(int64_t));

    if (!h || !xi || !c)
    {
        SPEX_FREE_WORKSPACE;
        return SPEX_OUT_OF_MEMORY;
    }
    // initialize workspace history array
    for (i = 0; i < n; i++)
    {
        h[i] = -1;
    }

    //--------------------------------------------------------------------------
    // Allocate and initialize the workspace x
    //--------------------------------------------------------------------------

    // SPEX utilizes arbitrary sized integers which can grow beyond the
    // default 64 bits allocated by GMP. If the integers frequently grow, GMP
    // can get bogged down by performing intermediate reallocations. Instead,
    // we utilize a larger estimate on the workspace x vector so that computing
    // the values in L and U do not require too many extra intermediate calls to
    // realloc.
    //
    // The bound given in the paper is that the number of bits is <= n log sigma
    // where sigma is the largest entry in A. Because this bound is extremely
    // pessimistic, instead of using this bound, we use a very rough estimate:
    // 64*max(2, log (n))
    //
    // Note that the estimate presented here is not an upper bound nor a lower
    // bound.  It is still possible that more bits will be required which is
    // correctly handled internally.
    int64_t estimate = 64 * SPEX_MAX (2, ceil (log2 ((double) n)));

    // Create x, a "global" dense mpz_t matrix of dimension n*1 (i.e., it is
    // used as workspace re-used at each iteration). The second boolean
    // parameter is set to false, indicating that the size of each mpz entry
    // will be initialized afterwards (and should not be initialized with the
    // default size).
    SPEX_CHECK (SPEX_matrix_allocate(&x, SPEX_DENSE, SPEX_MPZ, n, 1, n,
        false, /* do not initialize the entries of x: */ false, option));

    // Create rhos, a "global" dense mpz_t matrix of dimension n*1.
    // As inidicated with the second boolean parameter true, the mpz entries in
    // rhos are initialized to the default size (unlike x).

    SPEX_CHECK (SPEX_matrix_allocate(&(rhos), SPEX_DENSE, SPEX_MPZ, n, 1, n,
        false, true, option));

    // initialize the entries of x
    for (i = 0; i < n; i++)
    {
        // Allocate memory for entries of x
        SPEX_MPZ_INIT2(x->x.mpz[i], estimate);
    }

    //--------------------------------------------------------------------------
    // Declare memory for L
    //--------------------------------------------------------------------------

    // Since we are performing a left-looking factorization, we pre-allocate L
    // by performing a symbolic version of the factorization and obtaining the
    // exact nonzero pattern of L.
    // That said, the individual (x) values of L are not allocated. Instead,
    // a more efficient method to allocate these values is done inside the
    // factorization to reduce memory usage.

    SPEX_CHECK(spex_symmetric_pre_left_factor(&(L), xi, A, S));

    // Set the column pointers of L
    for (k = 0; k < n; k++)
    {
        L->p[k] = c[k] = (S->cp)[k];
    }

    //--------------------------------------------------------------------------
    // Perform the factorization
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Iterations 0:n-1 (1:n in standard)
    //--------------------------------------------------------------------------
    for (k = 0; k < n; k++)
    {
        // LDx = A(:,k)
        SPEX_CHECK(spex_symmetric_left_triangular_solve(&top, x, xi, L, A, k,
            rhos, h, S->parent, c));

        // Set the pivot element If this element is less than or equal to zero,
        // either no pivot element exists or the matrix is not SPD.
        SPEX_MPZ_SGN(&sgn, x->x.mpz[k]);
        // If we're attempting a Cholesky factorization the diagonal must be
        // > 0
        if (chol)
        {
            if (sgn > 0)
            {
                SPEX_MPZ_SET(rhos->x.mpz[k], x->x.mpz[k]);
            }
            else
            {
                // A is not symmetric positive definite
                SPEX_FREE_ALL;
                return SPEX_NOTSPD;
            }
        }
        // If we're attempting an LDL factorization the diagonal must be != 0
        else
        {
            if (sgn != 0)
            {
                SPEX_MPZ_SET(rhos->x.mpz[k], x->x.mpz[k]);
            }
            else
            {
                // A has a zero along the diagonal
                SPEX_FREE_ALL;
                return SPEX_ZERODIAG;
            }
        }
        //----------------------------------------------------------------------
        // Add the nonzeros to the L matrix
        //----------------------------------------------------------------------
        for (j = top; j < n; j++)
        {
            // Index of x[i]
            jnew = xi[j];
            if (jnew >= k)
            {
                // Find the size of x[j]
                size = mpz_sizeinbase(x->x.mpz[jnew],2);

                // GMP manual: Allocated size should be size+2
                SPEX_MPZ_INIT2(L->x.mpz[lnz], size+2);

                // Place the x value of this nonzero in row jnew of L
                SPEX_MPZ_SET(L->x.mpz[lnz],x->x.mpz[jnew]);

                // Increment lnz
                lnz += 1;
            }
        }
    }
    // Finalize L->p
    L->p[n] = S->lnz;

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------
    (*L_handle) = L;
    (*rhos_handle) = rhos;
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
