//------------------------------------------------------------------------------
// SPEX_QR/Source/SPEX_qr_solve.c: Solve exactly x = R \ Q^T b
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the sparse R \ Q^T b. First it computes Q^T b
 * then it solves R \ (Q^T b) using backward substitution.
 *
 * Input/output arguments:
 *
 * x_handle: A pointer to the solution vectors. Unitialized on input.
 *           on output, contains the exact rational solution of the system
 *
 * b:        Set of RHS vectors
 *
 * F:        QR factorization of A.
 *
 * option:   command options
 */

#define SPEX_FREE_WORKSPACE               \
    {                                     \
        SPEX_matrix_free(&b_new, option); \
        SPEX_free(Qinv_perm);             \
    }

#define SPEX_FREE_ALL                 \
    {                                 \
        SPEX_FREE_WORKSPACE           \
        SPEX_matrix_free(&x, option); \
    }

#include "spex_qr_internal.h"
#include "spex_lu_internal.h"

SPEX_info SPEX_qr_solve
(
    // Output
    SPEX_matrix *x_handle, // On input: undefined.
                           // On output: Rational solution (SPEX_MPQ)
                           // to the system.
    // input
    const SPEX_factorization F, // The QR factorization.
    const SPEX_matrix b,        // Right hand side vector
    const SPEX_options option   // command options
)
{
    SPEX_info info;
    // Check inputs, the number of columns of Q' must equal the number of rows of b
    // Also, both matrices must be mpz and dense
    // Ensure SPEX is initialized
    if (!spex_initialized())
    {
        return SPEX_PANIC;
    }

    // Check the inputs
    if (!x_handle || b->type != SPEX_MPZ || b->kind != SPEX_DENSE || F->kind != SPEX_QR_FACTORIZATION)
    {
        return SPEX_INCORRECT_INPUT;
    }

    SPEX_matrix b_new = NULL, x = NULL;
    int64_t k, p, i, j, qi, qj;
    int64_t rank = F->rank; // when matrix is full rank, rank=n
    int64_t index;
    int64_t *Qinv_perm = NULL;
    int64_t n = F->Q->n;
    int sgn;
    // b->new has Q->n rows and b->n columns
    SPEX_CHECK(SPEX_matrix_allocate(&b_new, SPEX_DENSE, SPEX_MPZ, b->m, b->n, 0,
                                    false, true, NULL));

    Qinv_perm = (int64_t *)SPEX_malloc(n * sizeof(int64_t));
    if (!Qinv_perm)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    for (k = 0; k < n; k++)
    {
        index = F->Q_perm[k];
        Qinv_perm[index] = k;
    }

    //--------------------------------------------------------------------------
    // Need to compute b_new[i] = R(n,n)* Q'[i,:] dot b[i]
    // This is equivalent to b_new[i] = R(n,n)* Q[:,i] dot b[i]
    //--------------------------------------------------------------------------
    // Iterate across every RHS vector
    for (k = 0; k < b->n; k++) // if b is a vector this will only be once
    {
        // Compute b[j,k]
        for (j = 0; j < F->Q->n; j++)
        {
            qj = j;
            for (p = F->Q->p[qj]; p < F->Q->p[qj + 1]; p++)
            {
                i = F->Q->i[p];
                SPEX_MPZ_ADDMUL(SPEX_2D(b_new, qj, k, mpz), F->Q->x.mpz[p],
                                SPEX_2D(b, i, k, mpz));
            }
            // F->rhos->x.mpz[F->R->n-1] is the determinant
            SPEX_MPZ_MUL(SPEX_2D(b_new, qj, k, mpz), SPEX_2D(b_new, qj, k, mpz),
                         F->rhos->x.mpz[F->rank - 1]);
        }
    }

    // Force free variables to zero for the basic solution if necessary
    for (k = 0; k < b->n; k++)
    {
        for (i = rank; i < F->Q->n; i++)
        {
            SPEX_MPZ_SET_UI(SPEX_2D(b_new, i, k, mpz), 0);
        }
    }

    //--------------------------------------------------------------------------
    // backwards substitution
    //--------------------------------------------------------------------------
    // Solves Rx=b_new (overwrites b_new into x)
    SPEX_CHECK(spex_qr_back_sub(b_new, F->R, rank, F->rhos, option));
    //--------------------------------------------------------------------------
    // x = Q*b_new/scale
    //--------------------------------------------------------------------------
    // set scale = b->scale * rhos[n-1] / A_scale
    SPEX_MPQ_SET_Z(b_new->scale, F->rhos->x.mpz[F->rank - 1]);
    SPEX_MPQ_MUL(b_new->scale, b_new->scale, b->scale);
    SPEX_MPQ_DIV(b_new->scale, b_new->scale, F->scale_for_A);

    // allocate space for x as dense MPQ matrix
    SPEX_CHECK(SPEX_matrix_allocate(&x, SPEX_DENSE, SPEX_MPQ, F->R->n, b->n,
                                    0, false, true, option));

    // obtain x from permuted b_new with scale applied
    for (i = 0; i < F->Q->n; i++)
    {
        qi = F->Q_perm[i];
        for (j = 0; j < b->n; j++)
        {
            SPEX_MPQ_SET_Z(SPEX_2D(x, qi, j, mpq),
                           SPEX_2D(b_new, i, j, mpz));
            SPEX_MPQ_DIV(SPEX_2D(x, qi, j, mpq),
                         SPEX_2D(x, qi, j, mpq), b_new->scale);
        }
    }

    //--------------------------------------------------------------------------
    // Return result and free workspace
    //--------------------------------------------------------------------------
    (*x_handle) = x;

    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
