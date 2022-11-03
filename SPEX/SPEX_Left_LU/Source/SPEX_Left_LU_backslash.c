//------------------------------------------------------------------------------
// SPEX_Left_LU/SPEX_Left_LU_backslash: solve Ax=b, returning solution as desired data type
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SPEX Left LU factorization to exactly solve the
 *          linear system Ax = b. This is essentially an exact version of
 *          MATLAB sparse backslash
 *
 * Input/Output arguments:
 *
 * X_handle:    A pointer to the solution of the linear system. The output is
 *              allowed to be returned in either double precision, mpfr_t, or
 *              rational mpq_t
 *
 * type:        Data structure of output desired. Must be either SPEX_MPQ,
 *              SPEX_FP64, or SPEX_MPFR
 *
 * A:           User's input matrix. It must be populated prior to calling this
 *              function.
 *
 * b:           Collection of right hand side vectors. Must be populated prior to
 *              factorization.
 *
 * option:      Struct containing various command parameters for the factorization. If
 *              NULL on input, default values are used.
 */

# define SPEX_FREE_WORK             \
    SPEX_matrix_free(&L, NULL);     \
    SPEX_matrix_free(&U, NULL);     \
    SPEX_FREE(pinv);                \
    SPEX_matrix_free(&rhos, NULL);  \
    SPEX_LU_analysis_free (&S, NULL);

# define SPEX_FREE_ALL              \
    SPEX_FREE_WORK                  \
    SPEX_matrix_free(&x, NULL);     \

#include "spex_left_lu_internal.h"

SPEX_info SPEX_Left_LU_backslash
(
    // Output
    SPEX_matrix **X_handle,       // Final solution vector
    // Input
    SPEX_type type,               // Type of output desired
                                  // Must be SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    const SPEX_matrix *A,         // Input matrix
    const SPEX_matrix *b,         // Right hand side vector(s)
    const SPEX_options* option    // Command options
)
{

    //-------------------------------------------------------------------------
    // check inputs
    //-------------------------------------------------------------------------

    SPEX_info info ;
    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    if (X_handle == NULL)
    {
        return SPEX_INCORRECT_INPUT;
    }
    (*X_handle) = NULL;

    if (type != SPEX_MPQ && type != SPEX_FP64 && type != SPEX_MPFR)
    {
        return SPEX_INCORRECT_INPUT;
    }

    SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ) ;
    SPEX_REQUIRE (b, SPEX_DENSE, SPEX_MPZ) ;

    SPEX_matrix *L = NULL ;
    SPEX_matrix *U = NULL ;
    SPEX_matrix *x = NULL;
    int64_t *pinv = NULL ;
    SPEX_matrix *rhos = NULL ;
    SPEX_LU_analysis *S = NULL;

    //--------------------------------------------------------------------------
    // Symbolic Analysis
    //--------------------------------------------------------------------------

    SPEX_CHECK(SPEX_LU_analyze(&S, A, option));

    //--------------------------------------------------------------------------
    // LU Factorization
    //--------------------------------------------------------------------------

    SPEX_CHECK(SPEX_Left_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));

    //--------------------------------------------------------------------------
    // Solve
    //--------------------------------------------------------------------------

    SPEX_CHECK (SPEX_Left_LU_solve (&x, b, A,
        (const SPEX_matrix *) L,
        (const SPEX_matrix *) U,
        (const SPEX_matrix *) rhos,
        S,
        (const int64_t *) pinv,
        option)) ;

    //--------------------------------------------------------------------------
    // Now, x contains the exact solution of the linear system in mpq_t
    // precision set the output.
    //--------------------------------------------------------------------------

    if (type == SPEX_MPQ)
    {
        (*X_handle) = x ;
    }
    else
    {
        SPEX_matrix* x2 = NULL ;
        SPEX_CHECK (SPEX_matrix_copy (&x2, SPEX_DENSE, type, x, option)) ;
        (*X_handle) = x2 ;
        SPEX_matrix_free (&x, NULL) ;
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    SPEX_FREE_WORK ;
    return (SPEX_OK) ;
}

