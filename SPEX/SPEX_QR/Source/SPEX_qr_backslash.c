//------------------------------------------------------------------------------
// SPEX_QR/SPEX_qr_backslash: solve Ax=b, return solution as desired data type
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SPEX QR factorization to exactly solve
 * the linear system Ax = b. It serves as a caller for either
 * the standard backslash (A x = b when A has more rows than columns)
 * or the transposed backslash (Ax = b when A has more columns than rows)
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

#define SPEX_FREE_ALL   \
    SPEX_FREE_WORKSPACE \
    SPEX_matrix_free(&x, NULL);

#include "spex_qr_internal.h"

SPEX_info SPEX_qr_backslash
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
    // check inputs
    //-------------------------------------------------------------------------
    SPEX_info info;
    if (!spex_initialized())
        return (SPEX_PANIC);

    if (x_handle == NULL)
    {
        return SPEX_INCORRECT_INPUT;
    }
    (*x_handle) = NULL;

    if (type != SPEX_MPQ && type != SPEX_FP64 && type != SPEX_MPFR)
    {
        return SPEX_INCORRECT_INPUT;
    }

    if (A->n <= 0 || A-> m <= 0)
    {
        return SPEX_INCORRECT_INPUT;
    }

    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_QR_GS)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    SPEX_REQUIRE(A, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(b, SPEX_DENSE, SPEX_MPZ);

    // Declare output
    SPEX_matrix x = NULL;

    //-------------------------------------------------------------------------
    // Select the appropriate algorithm based on the size of A
    //-------------------------------------------------------------------------

    if (A->m >= A->n)
    {
        // A is a tall skinny matrix. We factorize A directly and use the REF QR
        // of A to solve Ax = b.
        // If A has full column rank, the least squares solution is returned.
        // If A is rank deficient, a basic solution is returned

        info = spex_qr_standard_backslash(&x, type, A, b, option);
    }
    else
    {
        // A is a short wide matrix. We factorize A^T and use the REF QR
        // of A^T to solve Ax = b.
        // If A has full row rank, the minimum norm solution is returned.
        // If A is rank deficient, SPEX_SINGULAR is returned
        info = spex_qr_transpose_backslash(&x, type, A, b, option);
    }

    // x contains either the exact solution of the system or is NULL
    (*x_handle) = x;
    // returns SPEX_OK if the algorithm is successful or the appropriate error.
    return info;
}
