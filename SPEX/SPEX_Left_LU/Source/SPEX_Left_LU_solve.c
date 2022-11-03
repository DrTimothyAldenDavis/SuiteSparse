//------------------------------------------------------------------------------
// SPEX_Left_LU/SPEX_Left_LU_solve: exact solution of Ax=b
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function solves the linear system LD^(-1)U x = b. It
 * essnetially serves as a wrapper for all forward and backward substitution
 * routines. This function always returns the solution matrix x as a rational
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
 * A:        Input matrix. Unmodified on input/output
 *
 * L:        Lower triangular matrix. Unmodified on input/output
 *
 * U:        Upper triangular matrix. Unmodified on input/output
 *
 * rhos:     dense mpz_t matrix of pivots. Contains the sequence of pivots
 *           encountered during factorization and is used for forward/back
 *           substitution. Unmodified on input/output.
 *
 * S:        symbolic analysis struct, used for vector permutation
 *
 * pinv:     inverse row permutation vector, used to permute the b vectors.
 *           unmodified on input/output.
 *
 * option:   command options
 */

#define SPEX_FREE_WORK                  \
    SPEX_matrix_free (&b2, NULL) ;      \
    SPEX_matrix_free (&x2, NULL) ;      \
    SPEX_MPQ_CLEAR (scale) ;

#define SPEX_FREE_ALL                   \
    SPEX_FREE_WORK                      \
    SPEX_matrix_free (&x, NULL) ;

#include "spex_left_lu_internal.h"

SPEX_info SPEX_Left_LU_solve     // solves the linear system LD^(-1)U x = b
(
    // Output
    SPEX_matrix **x_handle,  // rational solution to the system
    // input:
    const SPEX_matrix *b,   // right hand side vector
    const SPEX_matrix *A,   // Input matrix
    const SPEX_matrix *L,   // lower triangular matrix
    const SPEX_matrix *U,   // upper triangular matrix
    const SPEX_matrix *rhos,// sequence of pivots
    const SPEX_LU_analysis *S,// symbolic analysis struct
    const int64_t *pinv,    // inverse row permutation
    const SPEX_options* option // Command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    SPEX_REQUIRE (b,    SPEX_DENSE, SPEX_MPZ) ;
    SPEX_REQUIRE (A,    SPEX_CSC,   SPEX_MPZ) ;
    SPEX_REQUIRE (L,    SPEX_CSC,   SPEX_MPZ) ;
    SPEX_REQUIRE (U,    SPEX_CSC,   SPEX_MPZ) ;
    SPEX_REQUIRE (rhos, SPEX_DENSE, SPEX_MPZ) ;

    if (!x_handle || !S || !pinv || L->m != A->m || L->n != U->m ||
        U->n != A->n || A->n != A->m || A->m != b->m )
    {
        return SPEX_INCORRECT_INPUT;
    }
    *x_handle = NULL;

    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    int64_t i, n = L->n;
    mpq_t scale ;
    SPEX_MPQ_SET_NULL (scale) ;

    SPEX_matrix *x = NULL;   // final solution
    SPEX_matrix *x2 = NULL;  // unpermuted solution
    SPEX_matrix *b2 = NULL;  // permuted b

    //--------------------------------------------------------------------------
    // b2 (pinv) = b
    //--------------------------------------------------------------------------

    SPEX_CHECK (spex_left_lu_permute_b (&b2, b, pinv, option)) ;

    //--------------------------------------------------------------------------
    // b2 = L\b2, via forward substitution
    //--------------------------------------------------------------------------

    SPEX_CHECK(spex_left_lu_forward_sub(L, b2, (SPEX_matrix*) rhos));

    //--------------------------------------------------------------------------
    // b2 = b2 * det, where det=rhos[n-1]
    //--------------------------------------------------------------------------

    SPEX_CHECK(SPEX_matrix_mul(b2, rhos->x.mpz[n-1]));

    //--------------------------------------------------------------------------
    // b2 = U\b2, via back substitution
    //--------------------------------------------------------------------------
    SPEX_CHECK(spex_left_lu_back_sub(U, b2));

    //--------------------------------------------------------------------------
    // x2 = b2/det, where det=rhos[n-1]
    //--------------------------------------------------------------------------

    SPEX_CHECK (SPEX_matrix_div (&x2, b2, rhos->x.mpz[n-1], option)) ;

    //--------------------------------------------------------------------------
    // x = Q*x2
    //--------------------------------------------------------------------------

    SPEX_CHECK (spex_left_lu_permute_x (&x, x2, (SPEX_LU_analysis *) S, option)) ;
    SPEX_matrix_free (&x2, NULL) ;

    //--------------------------------------------------------------------------
    // Check the solution if desired (debugging only)
    //--------------------------------------------------------------------------

    bool check = SPEX_OPTION_CHECK (option) ;
    if (check)
    {
        SPEX_CHECK (SPEX_check_solution (A, x, b, option)) ;
    }

    //--------------------------------------------------------------------------
    // Scale the solution if necessary.
    //--------------------------------------------------------------------------

    SPEX_CHECK(SPEX_mpq_init(scale));

    // set the scaling factor scale = A->scale / b->scale
    SPEX_CHECK( SPEX_mpq_div(scale, A->scale, b->scale));

    // Determine if the scaling factor is 1
    int r;
    SPEX_CHECK(SPEX_mpq_cmp_ui(&r, scale, 1, 1));
    int64_t nz = x->m * x->n;
    if (r != 0 )
    {
        for (i = 0; i < nz; i++)
        {
            SPEX_CHECK(SPEX_mpq_mul(x->x.mpq[i], x->x.mpq[i], scale));
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    SPEX_FREE_WORK ;
    (*x_handle) = x ;
    return (SPEX_OK) ;
}

