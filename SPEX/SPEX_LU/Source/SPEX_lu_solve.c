//------------------------------------------------------------------------------
// SPEX_LU/SPEX_lu_solve: exact solution of Ax=b
//------------------------------------------------------------------------------

// SPEX_LU: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function solves the linear system LD^(-1)U x = b. It
 * essnetially serves as a wrapper for all forward and backward substitution
 * routines. This function always returns the solution matrix x as a mpq_t
 * matrix. If a user desires to have double or mpfr output, they must create
 * a matrix copy.
 *
 * Input/output arguments:
 *
 * x_handle: A pointer to the solution vectors. Unitialized on input.
 *           on output, contains the exact rational solution of the system
 *
 * b:        Set of RHS vectors
 *
 * F:        LU factorization of A. Mathematically, F is unchanged.
 *           However, if F is updatable on input, it is converted to
 *           non-updatable.  If F is already non-updatable,
 *           it is not modified.
 *
 * option:   command options
 */

#define SPEX_FREE_WORKSPACE             \
    SPEX_matrix_free (&b2, NULL);

#define SPEX_FREE_ALL                   \
    SPEX_FREE_WORKSPACE                 \
    SPEX_matrix_free (&x, NULL);

#include "spex_lu_internal.h"

SPEX_info SPEX_lu_solve     // solves the linear system LD^(-1)U x = b
(
    // Output
    SPEX_matrix *x_handle,  // rational solution to the system
    // input/output:
    SPEX_factorization F,   // The non-updatable LU factorization.
                            // Mathematically, F is unchanged.  However, if F
                            // is updatable on input, it is converted to
                            // non-updatable.  If F is already non-updatable,
                            // it is not modified.
    // input:
    const SPEX_matrix b,    // right hand side vector
    const SPEX_options option // Command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    if (!spex_initialized ( )) return (SPEX_PANIC);

    SPEX_REQUIRE (b, SPEX_DENSE, SPEX_MPZ);

    if (!x_handle || !F || F->kind != SPEX_LU_FACTORIZATION)
    {
        return SPEX_INCORRECT_INPUT;
    }

    // check components of F in debug mode
    ASSERT_MATRIX (F->L,    SPEX_CSC,   SPEX_MPZ);
    ASSERT_MATRIX (F->U,    SPEX_CSC,   SPEX_MPZ);
    ASSERT_MATRIX (F->rhos, SPEX_DENSE, SPEX_MPZ);

    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    (*x_handle) = NULL;
    int64_t n = F->L->n;

    SPEX_matrix x = NULL;   // final solution
    SPEX_matrix b2 = NULL;  // permuted b

    //--------------------------------------------------------------------------
    // b2 (Pinv_perm) = b
    //--------------------------------------------------------------------------

    SPEX_CHECK (spex_permute_dense_matrix (&b2, b, F->Pinv_perm, option));

    //--------------------------------------------------------------------------
    // b2 = L\b2, via forward substitution
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_left_lu_forward_sub(F->L, b2, F->rhos));

    //--------------------------------------------------------------------------
    // b2 = b2 * det, where det=rhos[n-1]
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_matrix_mul(b2, F->rhos->x.mpz[n-1]));

    //--------------------------------------------------------------------------
    // b2 = U\b2, via back substitution
    //--------------------------------------------------------------------------
    SPEX_CHECK(spex_left_lu_back_sub(F->U, b2));

    //--------------------------------------------------------------------------
    // x = Q*b2/scale
    //--------------------------------------------------------------------------
    // set scale = b->scale * rhos[n-1] / A_scale
    SPEX_MPQ_SET_Z(b2->scale, F->rhos->x.mpz[n-1]);
    SPEX_MPQ_MUL(b2->scale, b2->scale, b->scale);
    SPEX_MPQ_DIV(b2->scale, b2->scale, F->scale_for_A);

    // allocate space for x as dense MPQ matrix
    SPEX_CHECK (SPEX_matrix_allocate (&x, SPEX_DENSE, SPEX_MPQ, b->m, b->n,
        0, false, true, option));

    // obtain x from permuted b2 with scale applied
    for (int64_t i = 0 ; i < b->m ; i++)
    {
        int64_t qi = F->Q_perm[i];
        for (int64_t j = 0 ; j < b->n ; j++)
        {
            SPEX_MPQ_SET_Z(SPEX_2D(x,  qi, j, mpq),
                                      SPEX_2D(b2,  i, j, mpz));
            SPEX_MPQ_DIV(SPEX_2D(x,  qi, j, mpq),
                                    SPEX_2D(x,  qi, j, mpq), b2->scale);
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    SPEX_FREE_WORKSPACE ;
    (*x_handle) = x ;
    return (SPEX_OK);
}

