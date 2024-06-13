//------------------------------------------------------------------------------
// SPEX_Cholesky/SPEX_cholesky_backslash: solve Ax=b
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SPEX Cholesky factorization to exactly solve
 *          the linear system Ax = b.
 *
 * Input/Output arguments:
 *
 * x_handle:    A pointer to the solution of the linear system. The output
 *              can be returned in double precision,
 *              mpfr_t (user-specified precision floating point), or
 *              mpq_t (rational)
 *
 * type:        Type of output desired.
 *              Must be SPEX_MPQ, SPEX_FP64 or SPEX_MPFR
 *
 * A:           User's input matrix.
 *              Must be populated prior to calling this function.
 *
 * b:           Collection of right hand side vector(s).
 *              Must be populated prior to calling this function.
 *
 * option:      Struct containing various command parameters for the
 *              factorization. If NULL on input, default values are used.
 */

#include "spex_cholesky_internal.h"

SPEX_info SPEX_cholesky_backslash
(
    // Output
    SPEX_matrix *x_handle,      // On input: undefined.
                                // On output: solution vector(s)
    // Input
    SPEX_type type,             // Type of output desired
                                // Must be SPEX_FP64, SPEX_MPFR, or SPEX_MPQ
    const SPEX_matrix A,        // Input matrix. Must be SPEX_MPZ and SPEX_CSC
    const SPEX_matrix b,        // Right hand side vector(s). Must be
                                // SPEX_MPZ and SPEX_DENSE
    const SPEX_options option   // Command options (Default if NULL)
)
{
    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_CHOL_LEFT
        && algo != SPEX_CHOL_UP)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }
    // The work is done in the spex_symmetric_backslash code
    // All we have to do is wrap it with chol = true
    SPEX_info info;

    info = spex_symmetric_backslash(x_handle, type, A, b, true, option);

    return info;
}
