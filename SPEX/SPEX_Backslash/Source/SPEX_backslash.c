//------------------------------------------------------------------------------
// SPEX_Backslash/SPEX_backslash.c: Solve a system Ax=b
//------------------------------------------------------------------------------

// SPEX_Backslash: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: Exactly solve sparse linear systems using SPEX Software package,
 * automatically determining the most appropriate factorization method.
 *
 * Input/Output arguments:
 *
 * x_handle:    A pointer to the solution of the linear system.
 *              Null on input. The output can be in double precision, mpfr_t, or
 *              rational mpq_t
 *
 * type:        Type of output desired, must be either SPEX_MPQ,
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

#include "spex_util_internal.h"
#include "SPEX.h"

SPEX_info SPEX_backslash
(
    // Output
    SPEX_matrix *x_handle,      // On output: Final solution vector(s)
                                // On input: undefined
    // Input
    const SPEX_type type,       // Type of output desired
                                // Must be SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    const SPEX_matrix A,        // Input matrix
    const SPEX_matrix b,        // Right hand side vector(s)
    SPEX_options option         // Command options (NULL: means use defaults)
)
{

    SPEX_info info;
    // Check inputs
    if (!spex_initialized()) return SPEX_PANIC;

    // Check for NULL pointers
    if (!x_handle || !A || !b )
    {
        return SPEX_INCORRECT_INPUT;
    }

    // Check for data types and dimension of A and b
    if (A->m != A->n || A->type != SPEX_MPZ || A->kind != SPEX_CSC
        || b->type != SPEX_MPZ || b->kind != SPEX_DENSE)
    {
        return SPEX_INCORRECT_INPUT;
    }

    // Check that output type is correct
    if (type != SPEX_MPQ && type != SPEX_FP64 && type != SPEX_MPFR)
    {
        return SPEX_INCORRECT_INPUT;
    }


    SPEX_options backslash_options = NULL;
    info = SPEX_create_default_options(&backslash_options);
    if (info != SPEX_OK)
    {
        return SPEX_OUT_OF_MEMORY;
    }

    if (option != NULL)
    {
        // IF the options are not NULL, copy the important parts.
        // Otherwise do nothing
        backslash_options->print_level = option->print_level; // print level
        backslash_options->prec = option->prec;               // MPFR precision
        backslash_options->round = option->round;             // MPFR rounding
    }

    // Declare output
    SPEX_matrix x = NULL;

    // Attempt a Cholesky factorization of A.
    // If Cholesky is occuring, we update the option
    // struct to do AMD and diagonal pivoting
    backslash_options->order = SPEX_AMD;
    backslash_options->pivot = SPEX_DIAGONAL;

    // Try SPEX Cholesky. The output for this function
    // is either:
    // SPEX_OK:       Cholesky success, x is the exact solution
    // SPEX_NOTSPD:   Cholesky failed. This means
    //                A is not SPD. In this case, we try LU
    // Other error code: Some error. Return the error code and exit
    info = SPEX_cholesky_backslash(&x, type, A, b, backslash_options);
    if (info == SPEX_OK)
    {
        // Cholesky was successful. Set x_handle = x
        (*x_handle) = x;

        // x_handle contains the exact solution of Ax = b and is
        // stored in the user desired type. Now, we exit and return ok
        SPEX_FREE(backslash_options);
        return SPEX_OK;
    }
    else if (info == SPEX_NOTSPD)
    {
        // Cholesky factorization failed. Must try
        // LU factorization now

        // Since LU is occuring, we update the option
        // struct to do COLAMD and small pivoting
        backslash_options->order = SPEX_COLAMD;
        backslash_options->pivot = SPEX_SMALLEST;

        // The LU factorization can return either:
        // SPEX_OK: LU success, x is the exact solution
        // Other error code: Some error. Return the error
        //                   code and exit
        info = SPEX_lu_backslash(&x, type, A, b, backslash_options);
        if (info == SPEX_OK)
        {
            // LU success, set x_handle = x
            (*x_handle) = x;

            // x_handle contains the exact solution of Ax = b and is
            // stored in the user desired type. Now, we exit and return ok
            SPEX_FREE(backslash_options);
            return SPEX_OK;
        }
        else
        {
            // Both Cholesky and LU have failed, info contains
            // the problem, most likely that A is singular
            // Note that, because LU failed, x_handle is still
            // NULL so there is no potential for a memory leak here
            SPEX_FREE(backslash_options);
            return info;
        }
    }
    else
    {
        // Cholesky failed, but not due to a SPEX_NOTSPD
        // error code. Most likely invalid input or out of
        // memory condition.
        // Note that since Cholesky failed, x_handle is still NULL
        // so there is no potential for a memory leak here
        SPEX_FREE(backslash_options);
        return info;
    }

}
