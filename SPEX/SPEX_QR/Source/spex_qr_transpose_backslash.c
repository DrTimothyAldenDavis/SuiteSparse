//------------------------------------------------------------------------------
// SPEX_QR/spex_qr_transpose_backslash: solve Ax=b when A is rectangular with
//                                      more columns than rows
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SPEX QR factorization to exactly solve
 * the linear system Ax = b when A is rectangular and contains more columns than
 * rows. In this case we compute the factorization of A^T, A^T = Q D R.
 * Then, x is found by solving R^T D y = b and x = Q D y
 *
 * Input/Output arguments:
 *
 * x_handle:    A pointer to the solution of the linear system. The output is
 *              allowed to be returned in either double precision, mpfr_t, or
 *              rational mpq_t
 *
 * type:        Data structure of output desired. Must be either SPEX_MPQ,
 *              SPEX_FP64, or SPEX_MPFR
 *
 * A:           User's input matrix. It must be populated prior to calling this
 *              function.
 *
 * b:           Collection of right hand side vectors. Must be populated prior
 *              to factorization.
 *
 * option:      Struct containing various command parameters for the
 *              factorization. If NULL on input, default values are used.
 */

#define SPEX_FREE_WORKSPACE              \
    SPEX_factorization_free(&F, option); \
    SPEX_symbolic_analysis_free(&S, option); \
    SPEX_matrix_free(&AT, NULL);

#define SPEX_FREE_ALL   \
    SPEX_FREE_WORKSPACE \
    SPEX_matrix_free(&x, NULL); \

#include "spex_qr_internal.h"

SPEX_info spex_qr_transpose_backslash
(
    // Output
    SPEX_matrix *x_handle, // Final solution vector
    // Input
    SPEX_type type,           // Type of output desired. Must be
                              // SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    const SPEX_matrix A,      // Input matrix
    const SPEX_matrix b,      // Right hand side vector(s)
    const SPEX_options option // Command options
)
{
    //-------------------------------------------------------------------------
    // All inputs are checked by the caller, no need to check here
    //-------------------------------------------------------------------------
    SPEX_info info;

    SPEX_REQUIRE(A, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(b, SPEX_DENSE, SPEX_MPZ);

    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL;
    SPEX_matrix x = NULL;
    SPEX_matrix AT = NULL;

    //--------------------------------------------------------------------------
    // Perform a full transpose of A
    //--------------------------------------------------------------------------

    SPEX_CHECK( SPEX_transpose(&AT, A, true, option));

    //--------------------------------------------------------------------------
    // Symbolic Analysis of AT
    //--------------------------------------------------------------------------
    SPEX_CHECK(SPEX_qr_analyze(&S, AT, option));

    //--------------------------------------------------------------------------
    // QR Factorization of AT.
    //--------------------------------------------------------------------------
    SPEX_CHECK(SPEX_qr_factorize(&F, AT, S, option));

    if (F->rank != A->m)
    {
        // If A is rank deficient, then an error code is returned and the
        // function terminates. This is due to a limitation in thin QR itself
        // in that there is no straightforward way to return a basic or
        // minimum norm solution for a wide A with rank deficiency
        SPEX_FREE_ALL;
        return SPEX_SINGULAR;
    }

    //--------------------------------------------------------------------------
    // Solve
    //--------------------------------------------------------------------------
    SPEX_CHECK(spex_qr_transpose_solve(&x, F, b, option));

    //--------------------------------------------------------------------------
    // Now, x contains the exact solution of the linear system in mpq_t
    // precision set the output.
    //--------------------------------------------------------------------------

    if (type == SPEX_MPQ)
    {
        (*x_handle) = x;
    }
    else
    {
        SPEX_matrix x2 = NULL;
        SPEX_CHECK(SPEX_matrix_copy(&x2, SPEX_DENSE, type, x, option));
        (*x_handle) = x2;
        SPEX_matrix_free(&x, NULL);
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    SPEX_FREE_WORKSPACE;
    return (SPEX_OK);
}
