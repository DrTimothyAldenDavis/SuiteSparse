//------------------------------------------------------------------------------
// SPEX_QR/Source/spex_qr_transpose_solve.c: Solve exactly x = Q D * (R^T D \ b)
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function solves the sparse x = Q D ( R^T D \ b).
 * Q D R are the REF QR factorization of A^T and thus this function is intended
 * for use when solving Ax = b and A is rectangular with more columns than rows.
 *
 * This function first solves y = R^T D \b and then calculates x as
 * x = Q D y
 *
 * A must have full row rank, otherwise the solve is undefined
 *
 * Input/output arguments:
 *
 * x_handle: A pointer to the solution vectors. Unitialized on input.
 *           on output, contains the exact rational solution of the system
 *
 * b:        Set of RHS vectors
 *
 * F:        QR factorization of A^T.
 *
 * option:   command options
 */

#define SPEX_FREE_WORKSPACE               \
    {                                     \
        SPEX_matrix_free(&b_new, option); \
        SPEX_matrix_free(&b2, option);    \
        SPEX_mpq_clear(temp);             \
    }

#define SPEX_FREE_ALL                 \
    {                                 \
        SPEX_FREE_WORKSPACE           \
        SPEX_matrix_free(&x, option); \
    }

#include "spex_qr_internal.h"
#include "spex_lu_internal.h"



SPEX_info spex_qr_transpose_solve
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
    // All inputs are checked by the caller, we can assert what
    // must be true.
    ASSERT(spex_initialized());
    ASSERT(x_handle != NULL);
    ASSERT(b->type == SPEX_MPZ);
    ASSERT(b->kind == SPEX_DENSE);
    ASSERT(F->kind == SPEX_QR_FACTORIZATION);

   // Declare matrices and a temporary mpq_t variable
    SPEX_matrix x = NULL, b_new = NULL, b2 = NULL;
    mpq_t temp;
    SPEX_MPQ_SET_NULL(temp);
    SPEX_CHECK( SPEX_mpq_init(temp));

    int64_t i, j, p, k;

    // Allocate memory for b_new
    SPEX_CHECK(SPEX_matrix_allocate(&b_new, SPEX_DENSE, SPEX_MPZ, b->m, b->n, 0, false, true, NULL));

    // Apply P^T to the right-hand side
    for (k = 0; k < b->n; k++)
    {
        for (i = 0; i < b->m; i++)
        {
            SPEX_MPZ_SET(SPEX_2D(b_new, i, k, mpz), SPEX_2D(b, F->Q_perm[i], k, mpz));
        }
    }

    // Transpose forward solve. Set b_new = (R^T D) \ b_new
    SPEX_CHECK( spex_qr_transpose_forward_sub( F->R, F->rank, b_new, F->rhos));

    // Now we have b_new = (R' D) \ b. The next step is to calculate x
    // x = Q D b_new
    // Note that D itself is rational with entries D[j,j] = 1/ (rhos[j]*rhos[j-1])
    // We will do a scaling with D first and then do the dot products with Q

    // We need b2 because we now switch from integer to rational
    SPEX_CHECK(SPEX_matrix_allocate(&b2, SPEX_DENSE, SPEX_MPQ, b->m, b->n, 0,
                                    false, true, NULL));

    // Loop through each RHS vector
    for (k = 0; k < b_new->n; k++)
    {
        // Need to multiply each entry by the associated entry in D
        // Recall that D[j,j] = 1/rhos[j]*rhos[j-1]
        // In order to avoid an if in the inner loop, we will do b[0]
        // here and then the rest in the for
        SPEX_CHECK( SPEX_mpq_set_num( SPEX_2D(b2, 0, k, mpq),
                                      SPEX_2D(b_new, 0, k, mpz)));
        SPEX_CHECK( SPEX_mpq_set_den( SPEX_2D(b2, 0, k, mpq),
                                      F->rhos->x.mpz[0]));
        SPEX_CHECK( SPEX_mpq_canonicalize( SPEX_2D(b2, 0, k, mpq)));

        // Only the entries in b_new[0..rank] are nonzero. Loop through
        // what's left
        // Since A must be full row rank for the solve to work, we could
        // equivalently change j < b_new->m but we leave it as rank
        // in case future development directly handles the rank deficient
        // case
        for (j = 1; j < F->rank; j++)
        {
            // Compute D[j,j] * b_new[j]
            // Start by initializing b2[j] = b_new[j]
            SPEX_CHECK( SPEX_mpq_set_num( SPEX_2D(b2, j, k, mpq),
                                      SPEX_2D(b_new, j, k, mpz)));

            // First we need to calculate b2[j] / rhos[j-1]
            // Since rhos is mpz_t, we first convert it to mpq_t
            SPEX_CHECK( SPEX_mpq_set_z( temp, F->rhos->x.mpz[j-1]));
            SPEX_CHECK( SPEX_mpq_div( SPEX_2D(b2, j, k, mpq), SPEX_2D(b2, j, k, mpq), temp));

            // Same process for b2[j] / rhos[j]
            SPEX_CHECK( SPEX_mpq_set_z( temp, F->rhos->x.mpz[j]));
            SPEX_CHECK( SPEX_mpq_div( SPEX_2D(b2, j, k, mpq), SPEX_2D(b2, j, k, mpq), temp));
        }
    }

    // Now, b2 = D*b_new
    // b2 is also rational at this point.
    // All that's left is to calculate Q*b2
    // We will use x directly. Note that x is of size Q->m by b->n
    SPEX_CHECK(SPEX_matrix_allocate(&x, SPEX_DENSE, SPEX_MPQ, F->Q->m, b->n, 0,
                                    false, true, NULL));

    // Loop through each RHS vector
    for (k = 0; k < b_new->n; k++)
    {
        // Loop through columns 1:rank of Q
        // If A is rank deficient Q will contain
        // n-rank columns of zeros
        // Though in order for the transpose solve to work correctly
        // A must have full row rank
        for (j = 0; j < F->rank; j++)
        {
            // Loop through the nonzeros in each column
            // b3[i] += Q[i,j]*b2[i]
            for (p = F->Q->p[j]; p < F->Q->p[j+1]; p++)
            {
                i = F->Q->i[p];
                SPEX_CHECK( SPEX_mpq_set_ui(temp, 0, 1));
                SPEX_CHECK( SPEX_mpq_set_num(temp, F->Q->x.mpz[p]));
                SPEX_CHECK( SPEX_mpq_mul(temp, temp, SPEX_2D(b2, j, k, mpq)));
                SPEX_CHECK( SPEX_mpq_add( SPEX_2D(x, i, k, mpq), SPEX_2D(x, i, k, mpq), temp));
            }
        }
    }
    //--------------------------------------------------------------------------
    // x = x/scale
    //--------------------------------------------------------------------------
    // set scale = b->scale / A_scale
    SPEX_MPQ_SET(temp, b->scale);
    SPEX_MPQ_DIV(temp, temp, F->scale_for_A);

    // obtain x from permuted b2 with scale applied
    for (i = 0; i < F->Q->m; i++)
    {
        for (j = 0; j < b->n; j++)
        {
            SPEX_MPQ_DIV(SPEX_2D(x, i, j, mpq),
                         SPEX_2D(x, i, j, mpq), temp);
        }
    }

    //--------------------------------------------------------------------------
    // Return result and free workspace
    //--------------------------------------------------------------------------
    (*x_handle) = x;

    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
