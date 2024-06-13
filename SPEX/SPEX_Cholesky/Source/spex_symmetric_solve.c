//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_solve: Solve the system after factorization
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE        \
{                                  \
    SPEX_matrix_free(&b2, option); \
}

# define SPEX_FREE_ALL             \
{                                  \
    SPEX_FREE_WORKSPACE            \
    SPEX_matrix_free(&x, NULL);    \
}

#include "spex_cholesky_internal.h"

/* Purpose: solve the system A x = b using the Cholesky or LDL factorization.
 *
 * Input arguments:
 *
 * x_handle:        A handle to the solution matrix. On input this is NULL,
 *                  on output x_handle contains a pointer to the solution
 *                  vector(s)
 *
 * F:               The factorization struct containing the REF Cholesky
 *                  or LDL factorization of A, permutation, etc
 *
 * b:               Right hand side vector
 *
 * option:          Command options
 */

SPEX_info spex_symmetric_solve
(
    // Output
    SPEX_matrix *x_handle,      // On input: undefined.
                                // On output: Rational solution (SPEX_MPQ)
                                // to the system.
    // input/output:
    SPEX_factorization F,       // The Cholesky or LDL factorization of A
    // input:
    const SPEX_matrix b,        // Right hand side vector
    const SPEX_options option   // command options
)
{

    SPEX_info info;

    SPEX_REQUIRE(b, SPEX_DENSE, SPEX_MPZ);

    // det is the determinant of the PAP matrix. It is obtained for free
    // from the SPEX Cholesky factorization det = rhos[n-1] = L[n,n]
    mpz_t *det = NULL;

    //--------------------------------------------------------------------------
    // Declare workspace and output
    //--------------------------------------------------------------------------
    // x is the permuted final solution vector returned to the user
    SPEX_matrix x = NULL;
    // b2 is the permuted right hand side vector(s)
    SPEX_matrix b2 = NULL;

    //--------------------------------------------------------------------------
    // get b2 = Pinv*b
    //--------------------------------------------------------------------------

    SPEX_CHECK (spex_permute_dense_matrix (&b2, b, F->Pinv_perm, option));

    //--------------------------------------------------------------------------
    // Forward substitution, b2 = L \ b2. Note that b2 is overwritten
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_symmetric_forward_sub(b2, F->L, F->rhos));

    //--------------------------------------------------------------------------
    // Apply the determinant to b2, b2 = det*b2
    //--------------------------------------------------------------------------

    // Set the value of the determinant det = rhos[n-1]
    det = &(F->rhos->x.mpz[F->L->n-1]);

    // Multiply b2 by the determinant. This multiplication ensures that the next
    // backsolve is integral
    SPEX_CHECK(spex_matrix_mul(b2, (*det) ));

    //--------------------------------------------------------------------------
    // Backsolve, b2 = L' \ b2. Note that, again, b2 is overwritten
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_symmetric_backward_sub(b2, F->L));

    //--------------------------------------------------------------------------
    // get real solution x by applying both permutation and scale
    // x = P*b2/scale
    //--------------------------------------------------------------------------

    // Scale is the scaling factor for the solution vectors.
    // When the forward/backsolve is complete, the entries in
    // x/det are rational, but are solving the scaled linear system
    // A' x = b' (that is if A had input which was rational or floating point
    // and had to be converted to integers). Thus, the scale here is used
    // to convert x into into the actual solution of A x = b.
    // Mathematically, set scale = b->scale * rhos[n-1] / PAP->scale

    SPEX_MPQ_SET_Z(b2->scale, (*det));
    SPEX_MPQ_MUL(b2->scale, b2->scale, b->scale);
    SPEX_MPQ_DIV(b2->scale, b2->scale, F->scale_for_A);

    // allocate space for x as dense MPQ matrix
    SPEX_CHECK (SPEX_matrix_allocate (&x, SPEX_DENSE, SPEX_MPQ, b->m, b->n,
        0, false, true, option));

    // obtain x from permuted b2 with scale applied
    for (int64_t i = 0 ; i < b->m ; i++)
    {
        int64_t pi = F->P_perm[i];
        for (int64_t j = 0 ; j < b->n ; j++)
        {

            SPEX_MPQ_SET_Z(SPEX_2D(x,  pi, j, mpq),
                                      SPEX_2D(b2,  i, j, mpz));
            SPEX_MPQ_DIV(SPEX_2D(x,  pi, j, mpq),
                                    SPEX_2D(x,  pi, j, mpq), b2->scale);
        }
    }

    // Set output, free memory
    (*x_handle) = x;
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
