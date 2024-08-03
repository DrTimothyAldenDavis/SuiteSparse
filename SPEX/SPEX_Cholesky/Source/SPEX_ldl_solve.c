//------------------------------------------------------------------------------
// SPEX_Cholesky/SPEX_ldl_solve: Solve the linear system after LDL
// factorization
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_cholesky_internal.h"

/* Purpose: This function solves the linear system LDL' x = b.
 *
 * Input arguments:
 *
 * x_handle:        A handle to the solution matrix. On input this is NULL,
 *                  on output x_handle contains a pointer to the solution
 *                  vector(s)
 *
 * F:               The factorization struct containing the REF LDL
 *                  factorization of A, permutation, etc
 *
 * b:               Right hand side vector(s)
 *
 * option:          Command options *
 */

SPEX_info SPEX_ldl_solve
(
    // Output
    SPEX_matrix *x_handle,      // On input: undefined.
                                // On output: Rational solution (SPEX_MPQ)
                                // to the system.
    // input/output:
    SPEX_factorization F,       // The non-updatable LDL factorization.
                                // Mathematically, F is unchanged.  However, if
                                // F is updatable on input, it is converted to
                                // non-updatable.  If F is already
                                // non-updatable, it is not modified.
    // input:
    const SPEX_matrix b,        // Right hand side vector
    const SPEX_options option   // command options
)
{
    // Just need to call the symmetric solve
    SPEX_info info;

    // Ensure SPEX is initialized
    if (!spex_initialized())
    {
        return SPEX_PANIC;
    }

    // Check the inputs
    if (!x_handle || b->type != SPEX_MPZ || b->kind != SPEX_DENSE)
    {
        return SPEX_INCORRECT_INPUT;
    }

    if (F->kind != SPEX_LDL_FACTORIZATION)
    {
        return SPEX_INCORRECT_INPUT;
    }

    info = spex_symmetric_solve(x_handle, F, b, option);

    return info;
}
