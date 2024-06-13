//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_backward_sub: Solve L' x = b for Cholesky
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_ALL ;

#include "spex_cholesky_internal.h"

/* Purpose: This solves the system L'x = b for Cholesky or LDL factorization.
 * On input, x contains the scaled solution of L D x = b and L is the REF
 * Cholesky or LDL factor of A.  On output, x is the solution to the linear
 * system Ax = (det A)b.
 */

SPEX_info spex_symmetric_backward_sub
(
    // Output
    SPEX_matrix x,          // Solution vector to A x = det(A) * b
    // Input
    const SPEX_matrix L     // The lower triangular matrix
)
{

    SPEX_info info;
    // All inputs have been checked by the caller, asserts are
    // here as a reminder
    ASSERT(L->type == SPEX_MPZ);
    ASSERT(L->kind == SPEX_CSC);
    ASSERT(x->type == SPEX_MPZ);
    ASSERT(x->kind == SPEX_DENSE);

    int64_t k, p, j, n = L->n;
    int sgn, sgn2;

    // Iterate across the RHS vectors
    for (k = 0; k < x->n; k++)
    {
        // Iterate across the rows of x
        for (j = n-1; j >= 0; j--)
        {
            // Iterate across column j of L
            for (p = L->p[j]+1; p < L->p[j+1]; p++)
            {
                // If either x[p,k] or L[p,k] is 0, skip the operation
                SPEX_MPZ_SGN(&sgn, SPEX_2D(x, L->i[p], k, mpz));
                SPEX_MPZ_SGN(&sgn2, L->x.mpz[p]);
                if (sgn == 0 || sgn2 ==0 ) continue;

                // Compute x[j,k] = x[j,k] - L[p,k]*x[p,k]
                SPEX_MPZ_SUBMUL(SPEX_2D(x, j, k, mpz),
                                L->x.mpz[p], SPEX_2D(x, L->i[p], k, mpz));
            }

            // Compute x[j,k] = x[j,k] / L[j,j]
            SPEX_MPZ_DIVEXACT(SPEX_2D(x, j, k, mpz),
                              SPEX_2D(x, j, k, mpz), L->x.mpz[ L->p[j]]);

        }
    }
    return SPEX_OK;
}
