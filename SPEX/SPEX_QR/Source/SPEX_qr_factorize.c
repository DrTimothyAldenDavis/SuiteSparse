//------------------------------------------------------------------------------
// SPEX_QR/Source/SPEX_qr_factorize.c: QR factorization
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the REF QR factorization via
 * Integer-preserving Gram-Schmidt
 *
 * Input arguments of the function:
 *
 * F_handle:    Handle to the factorization struct. Null on input.
 *              On output, contains a pointer to the factorization.
 *
 * A:           User's input matrix. Must be SPEX_MPZ and SPEX_CSC.
 *
 * S:           Symbolic analysis struct for QR factorization.
 *              On input it contains the column elimination tree and
 *              the number of nonzeros in R.
 *
 * option:      Command options.
 */

#define SPEX_FREE_WORKSPACE              \
    {                                    \
        SPEX_matrix_free(&(RT), option); \
        SPEX_free(h);                    \
        SPEX_free(Qk);                   \
        SPEX_free(ldCols);               \
        SPEX_matrix_free(&RTPi, option); \
        SPEX_matrix_free(&RPi, option);  \
        SPEX_free(Pi_perm);              \
        SPEX_free(Piinv_perm);           \
    }

#define SPEX_FREE_ALL                      \
    {                                      \
        SPEX_FREE_WORKSPACE                \
        SPEX_matrix_free(&Q, option);      \
        SPEX_matrix_free(&rhos, option);   \
        SPEX_matrix_free(&rhosPi, option); \
        SPEX_factorization_free(&F, NULL); \
    }

#include "spex_qr_internal.h"

SPEX_info SPEX_qr_factorize(
    // Output
    SPEX_factorization *F_handle, // QR factorization struct
    // Input
    const SPEX_matrix A,      // Matrix to be factored. Must be SPEX_MPZ
                              // and SPEX_CSC
    SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                              // column elimination tree of A, the column
                              // permutation, and number of nonzeros in R
    const SPEX_options option // command options.
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
    if (A->kind != SPEX_CSC || A->type != SPEX_MPZ || S->kind != SPEX_QR_FACTORIZATION)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_QR_GS)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }
    // Declare variables
    int64_t n = A->n, m = A->m, k, i, pQ, p, iQ, pR;
    SPEX_factorization F = NULL;
    SPEX_matrix RT = NULL, Q = NULL, rhos = NULL;
    SPEX_matrix RTPi = NULL;
    SPEX_matrix RPi = NULL;
    SPEX_matrix rhosPi = NULL;
    int64_t *h = NULL, *Qk = NULL;
    int64_t *Pi_perm = NULL;    // Column permutation for rank deficient matrices
    int64_t *Piinv_perm = NULL; // Inverse row permutation for rank deficient matrices

    // Varibles needed to compute the rank of a matrix.
    // assume matrix is full rank. isZeros is true if a column is linearly
    // dependent, ldCols keeps track of linearly dependent columns
    bool isZeros = true, *ldCols = NULL;
    int64_t rank = n;
    int sgn;

    // Allocate memory for the factorization
    F = (SPEX_factorization)SPEX_calloc(1, sizeof(SPEX_factorization_struct));
    if (F == NULL)
        return SPEX_OUT_OF_MEMORY;

    // set factorization kind
    F->kind = SPEX_QR_FACTORIZATION;

    // Allocate and set scale_for_A
    SPEX_MPQ_INIT(F->scale_for_A);
    SPEX_MPQ_SET(F->scale_for_A, A->scale);

    //--------------------------------------------------------------------------
    // Allocate and compute the nonzero structure of Q and R
    //--------------------------------------------------------------------------
    SPEX_CHECK(spex_qr_nonzero_structure(&RT, &Q, A, S, option));

    SPEX_CHECK(SPEX_matrix_allocate(&rhos, SPEX_DENSE, SPEX_MPZ, n, 1, n,
                                    false, true, option));

    //--------------------------------------------------------------------------
    // Allocate and initialize supporting vectors
    //--------------------------------------------------------------------------
    h = (int64_t *)SPEX_calloc((Q->nz), sizeof(int64_t)); // history matrix
    // Qk contains the position of the nonzero elements in each row of the
    // latest column of Q to be finalized (or -1 if the corresponding element
    // in that row is symbolically zero*
    Qk = (int64_t *)SPEX_malloc((m + 1) * sizeof(int64_t));
    ldCols = (bool *)SPEX_calloc((n), sizeof(bool));
    if (!h || !Qk || !ldCols || !(rhos))
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    for (k = 0; k < m; k++)
    {
        Qk[k] = -1;
    }
    for (k = Q->p[0]; k < Q->p[1]; k++) // Q(:,0)=A(:,0) first column of Q is finalized
    {
        Qk[Q->i[k]] = k;
    }

    //--------------------------------------------------------------------------
    // Perform IPGS to get Q and R
    //--------------------------------------------------------------------------
    // If the first column is full of numerical zeros
    for (pQ = Q->p[0]; pQ < Q->p[1]; pQ++)
    {

        SPEX_MPZ_SGN(&sgn, Q->x.mpz[pQ]);
        if (sgn != 0)
        {
            isZeros = false;
        }
    }

    for (k = 0; k < n - 1; k++)
    {
        // when the kth column of Q is all zeros (it is linearly dependent)
        // then the kth row of R is all zeros too and you skip operations on k
        if (isZeros)
        {
            ldCols[k] = true; // kth pivot of R is zeros, kth column of Q is ld

            // Set the kth pivot to be equal to the k-1th pivot for computations
            if (k == 0)
            {
                SPEX_MPZ_SET_UI(rhos->x.mpz[k], 1); // rho[0]=1
            }
            else
            {
                SPEX_MPZ_SET(rhos->x.mpz[k], rhos->x.mpz[k - 1]);
            }

            // Finalize Q k+1 (it keeps its previous values)
            for (pQ = Q->p[k + 1]; pQ < Q->p[k + 2]; pQ++)
            {
                // History update
                // This ensures updates are performed when a matrix is rank deficient but
                // an entry hasn't been updated since initialization.
                if (h[pQ] < k + 1)
                {
                    // Entry has been updated before, do a history update
                    if (h[pQ] > 0)
                    {
                        SPEX_CHECK(spex_history_update(Q, rhos, pQ, k - 1, h[pQ], h[pQ] - 1, 0, option));
                    }
                    // h[pq] = 0 but we are outside of column 1. In this case, we need to bring the entry up
                    // to iteration k+1
                    else if (k > 0)
                    {
                        SPEX_MPZ_MUL(Q->x.mpz[pQ], Q->x.mpz[pQ], rhos->x.mpz[k - 1]);
                    }
                }
                // Update the history
                h[pQ] = k + 1;
                iQ = Q->i[pQ];
                Qk[iQ] = pQ;

                // Check for linear dependency
                SPEX_MPZ_SGN(&sgn, Q->x.mpz[pQ]);
                if (sgn != 0)
                {
                    isZeros = false;
                }
            }

            rank--;
        }
        else
        {
            // Integer-preserving Gram-Schmidt
            SPEX_CHECK(spex_qr_ipgs(RT, Q, rhos, Qk, h, &isZeros, k, A,
                                    S->Q_perm, option));
        }
    }

    // Finalize R (get the last element/pivot)
    if (isZeros)
    {
        ldCols[k] = true;
        rank--;
        SPEX_MPZ_SET_UI(RT->x.mpz[RT->p[n] - 1], 0);
        SPEX_MPZ_SET(rhos->x.mpz[n - 1], rhos->x.mpz[n - 2]);
    }
    else
    {
        SPEX_CHECK(spex_dot_product(RT->x.mpz[RT->p[n] - 1], Q, n - 1, A,
                                    S->Q_perm[n - 1], option));
        SPEX_MPZ_SET(rhos->x.mpz[n - 1], RT->x.mpz[RT->p[n] - 1]);
    }

    //--------------------------------------------------------------------------
    // Get rank revealing permutation
    //--------------------------------------------------------------------------
    if (rank != n)
    {
        // If A is rank deficient then Q and RT have at least one 0 column
        // Here those columns will be permuted to the right side of the
        // respective matrix. And as such the column permutation of A (Q_perm)
        // will be updated.

        // Indices for modifying Q_perm (and creating Pi_perm) according to
        // whether a column of Q is linearly dependent or linearly independent
        // of the previous columns
        int64_t iLD = n - 1, iLI = 0, index;
        F->rank = rank;

        Pi_perm = (int64_t *)SPEX_malloc(n * sizeof(int64_t));
        Piinv_perm = (int64_t *)SPEX_malloc(n * sizeof(int64_t));
        F->Q_perm = (int64_t *)SPEX_malloc(n * sizeof(int64_t));

        if (!(F->Q_perm) || !Piinv_perm || !Pi_perm)
        {
            // out of memory: free everything and return
            SPEX_FREE_ALL;
            return SPEX_OUT_OF_MEMORY;
        }

        info = SPEX_matrix_allocate(&rhosPi, SPEX_DENSE, SPEX_MPZ, n, 1, n,
                                        false, true, option);
        if (info != SPEX_OK)
        {
            SPEX_FREE_ALL;
            return info;
        }

        // If the k-th column of Q is linearly dependent (ldCols[k]=true), then
        // k needs to be towards the end of Pi_perm and the original Q_perm[k]
        // needs to be towards the end of the updated column permutation
        for (k = 0; k < n; k++)
        {
            if (ldCols[k]) // ldCols[k] is true when the k col is linearly dependent
            {
                Pi_perm[iLD] = k;
                // Piinv_perm[n+iLD]=k;
                F->Q_perm[iLD] = S->Q_perm[k];
                iLD--;
            }
            else
            {
                Pi_perm[iLI] = k;
                // Piinv_perm[n-iLI]=k;
                F->Q_perm[iLI] = S->Q_perm[k];
                iLI++;
            }
        }

        // Populate pinv
        for (k = 0; k < n; k++)
        {
            index = Pi_perm[k];
            Piinv_perm[index] = k;
        }

        // Permute Q and RT
        // Zero columns of Q will be on the right
        // Zero rows of R will be on the bottom (Zero columns of RT will be on
        // the right)
        SPEX_CHECK(spex_qr_permute_A(&F->Q, Q, true, Pi_perm, NULL, option));
        SPEX_CHECK(spex_qr_permute_A(&RTPi, RT, true, Pi_perm, Piinv_perm, option));
        SPEX_CHECK(SPEX_transpose(&F->R, RTPi, true, option));
        F->R->nz = RT->p[n] - 1;

        for (k = 0; k < n; k++)
        {
            SPEX_MPZ_SET(rhosPi->x.mpz[k], rhos->x.mpz[Pi_perm[k]]);
        }

        F->rhos = rhosPi;
        // Free the original workspaces since F now has the permuted copies
        SPEX_matrix_free(&Q, option);
        SPEX_matrix_free(&rhos, option);

    }
    else
    {
        // Q has full rank, no cleanup has to happen here

        F->rank = n; // matrix has full rank

        // column permutation, to be copied from S->Q_perm
        F->Q_perm = (int64_t *)SPEX_malloc(n * sizeof(int64_t));
        if (!(F->Q_perm))
        {
            // out of memory: free everything and return
            SPEX_FREE_ALL;
            return SPEX_OUT_OF_MEMORY;
        }

        // Copy column permutation from symbolic analysis to factorization
        memcpy(F->Q_perm, S->Q_perm, n * sizeof(int64_t));

        SPEX_CHECK(SPEX_transpose(&F->R, RT, true, option));
        F->R->nz = RT->p[n] - 1;

        F->rhos = rhos;
        F->Q = Q;
    }

    //--------------------------------------------------------------------------
    // Return result and free workspace
    //--------------------------------------------------------------------------
    (*F_handle) = F;

    SPEX_FREE_WORKSPACE;

    return SPEX_OK;
}
