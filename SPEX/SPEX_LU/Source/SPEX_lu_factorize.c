//-----------------------------------------------------------------------------
// SPEX_LU/SPEX_lu_factorize: exact sparse LU factorization
//------------------------------------------------------------------------------

// SPEX_LU: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This internal function performs the SPEX Left LU factorization, but
 * does not free the row permutation when finished. This factorization is done
 * via n iterations of the sparse REF triangular solve function. The overall
 * factorization is PAQ = LDU The determinant of A can be obtained as
 * determinant = rhos[n-1]
 *
 *  F: undefined on input, created on output
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

SPEX_info SPEX_lu_factorize
(
    // output:
    SPEX_factorization *F_handle,   // LU factorization
    // input:
    const SPEX_matrix A,            // matrix to be factored
    const SPEX_symbolic_analysis S, // symbolic analysis
    const SPEX_options option       // command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!spex_initialized ( )) return (SPEX_PANIC);
    
    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_LU_LEFT)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    SPEX_REQUIRE (A, SPEX_CSC, SPEX_MPZ);
    int64_t anz;
    // SPEX enviroment is checked to be init'ed and A is a SPEX_CSC matrix that
    // is not NULL, so SPEX_matrix_nnz must return SPEX_OK
    SPEX_info info = SPEX_matrix_nnz (&anz, A, option);
    ASSERT(info == SPEX_OK);

    if (!F_handle || !S || anz < 0)
    {
        return SPEX_INCORRECT_INPUT;
    }

    (*F_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    SPEX_factorization F = NULL ;
    int64_t *xi = NULL ;
    int64_t *h = NULL ;
    int64_t *pivs = NULL ;
    SPEX_matrix x = NULL ;

    int64_t n = A->n ;

    int64_t k = 0, top, i, j, col, loc, lnz = 0, unz = 0, pivot, jnew ;
    size_t size ;

    // allocate memory space for the factorization
    F = (SPEX_factorization) SPEX_calloc(1, sizeof(SPEX_factorization_struct));
    if (F == NULL)
    {
        return SPEX_OUT_OF_MEMORY;
    }
    // set factorization kind
    F->kind = SPEX_LU_FACTORIZATION;
    // Allocate and set scale_for_A
    SPEX_MPQ_INIT(F->scale_for_A);
    SPEX_MPQ_SET (F->scale_for_A, A->scale);

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

    // Allocate L and U without initializing each entry.
    // L and U are allocated to have nnz(L) which is estimated by the symbolic
    // analysis. However, unlike traditional matrix allocation, the second
    // boolean parameter here is set to false, so the individual values of
    // L and U are not allocated. Instead, a more efficient method to
    // allocate these values is done in the factorization to reduce
    // memory usage.
    SPEX_CHECK (SPEX_matrix_allocate(&(F->L), SPEX_CSC, SPEX_MPZ, n, n, S->lnz,
        false, false, option));
    SPEX_CHECK (SPEX_matrix_allocate(&(F->U), SPEX_CSC, SPEX_MPZ, n, n, S->unz,
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

    for (k = 0; k < n; k++)
    {
        // Column pointers for column k of L and U
        F->L->p[k] = lnz;
        F->U->p[k] = unz;
        col = F->Q_perm[k];

        //----------------------------------------------------------------------
        // Reallocate memory if necessary
        // if lnz+n > L->nzmax, L needs to expand to accomodate new nonzeros.
        // To do so, we double the size of the L and U matrices.
        //----------------------------------------------------------------------
        if (lnz + n > F->L->nzmax)
        {
            // Double the size of L
            SPEX_CHECK(spex_sparse_realloc(F->L));
        }
        if (unz + n > F->U->nzmax)
        {
            // Double the size of U
            SPEX_CHECK(spex_sparse_realloc(F->U));
        }

        //----------------------------------------------------------------------
        // Triangular solve to compute LDx = A(:,k)
        //----------------------------------------------------------------------
        SPEX_CHECK(spex_left_lu_ref_triangular_solve(&top, F->L, A, k, xi,
            (const int64_t *) (F->Q_perm),
            F->rhos,
            (const int64_t *) (F->Pinv_perm),
            (const int64_t *) (F->P_perm),
            h, x));

        //----------------------------------------------------------------------
        // Obtain pivot
        //----------------------------------------------------------------------
        SPEX_CHECK(spex_left_lu_get_pivot(&pivot, x, pivs, n, top, xi,
            col, k, F->rhos, F->Pinv_perm, F->P_perm, option));

        //----------------------------------------------------------------------
        // Populate L and U. We iterate across all nonzeros in x
        //----------------------------------------------------------------------
        for (j = top; j < n; j++)
        {
            jnew = xi[j];
            // Location of x[j] in final matrix
            loc = F->Pinv_perm[jnew];

            //------------------------------------------------------------------
            // loc <= k are rows above k, thus go to U
            //------------------------------------------------------------------
            if (loc <= k)
            {
                // Place the i location of the unz nonzero
                F->U->i[unz] = jnew;
                // Find the size in bits of x[j]
                SPEX_MPZ_SIZEINBASE(&size, x->x.mpz[jnew], 2);
                // GMP manual: Allocated size should be size+2
                SPEX_MPZ_INIT2(F->U->x.mpz[unz], size+2);
                // Place the x value of the unz nonzero
                SPEX_MPZ_SET(F->U->x.mpz[unz], x->x.mpz[jnew]);
                // Increment unz
                unz++;
            }

            //------------------------------------------------------------------
            // loc >= k are rows below k, thus go to L
            //------------------------------------------------------------------
            if (loc >= k)
            {
                // Place the i location of the lnz nonzero
                F->L->i[lnz] = jnew;
                // Set the size of x[j]
                SPEX_MPZ_SIZEINBASE(&size, x->x.mpz[jnew], 2);
                // GMP manual: Allocated size should be size+2
                SPEX_MPZ_INIT2(F->L->x.mpz[lnz], size+2);
                // Place the x value of the lnz nonzero
                SPEX_MPZ_SET(F->L->x.mpz[lnz], x->x.mpz[jnew]);
                // Increment lnz
                lnz++;
            }
        }
    }

    // Finalize L->p, U->p
    F->L->p[n] = lnz;
    F->U->p[n] = unz;

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    // free everything, but keep F
    SPEX_FREE_WORKSPACE ;

    // This cannot fail since the size of L and U are shrinking.
    // Collapse L
    spex_sparse_collapse(F->L);
    // Collapse U
    spex_sparse_collapse(F->U);

    //--------------------------------------------------------------------------
    // finalize the row indices in L and U
    //--------------------------------------------------------------------------

    // Permute entries in L
    for (i = 0; i < lnz; i++)
    {
        F->L->i[i] = F->Pinv_perm[F->L->i[i]];
    }
    // Permute entries in U
    for (i = 0; i < unz; i++)
    {
        F->U->i[i] = F->Pinv_perm[F->U->i[i]];
    }

    //--------------------------------------------------------------------------
    // check the LU factorization (debugging only)
    //--------------------------------------------------------------------------

    #if 0
    SPEX_CHECK (SPEX_matrix_check (F->L, option));
    SPEX_CHECK (SPEX_matrix_check (F->U, option));
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*F_handle) = F ;
    return (SPEX_OK);
}

