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

    (*x_handle) = NULL ;

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

    // Declare output
    SPEX_matrix x = NULL;

    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    switch (algo)
    {
        // Left-looking LU factorization is desired. Call lu backslash
        // with user-specified options
        case SPEX_LU_LEFT:
            info = SPEX_lu_backslash (&x, type, A, b, option);
            break ;

        // Some type of Cholesky factorization is desired. Call
        // Cholesky backslash with user-specified options
        case SPEX_CHOL_UP:
        case SPEX_CHOL_LEFT:
            info = SPEX_cholesky_backslash (&x, type, A, b, option);
            break ;

        // Some type of LDL factorization is desired. Call
        // LDL backslash with user-specified options
        case SPEX_LDL_UP:
        case SPEX_LDL_LEFT:
            info = SPEX_ldl_backslash (&x, type, A, b, option);
            break ;

        // Default algorithm is utilized. In this case, SPEX Backslash
        // attempts to find the appropriate algorithm. First, up-looking
        // LDL factorization is attempted. If LDL is successful, return x
        // and exit. If LDL fails, LU factorization is attempted.
        default:
        case SPEX_ALGORITHM_DEFAULT:

            // Try SPEX ldl. The output for this function
            // is either:
            // SPEX_OK:          LDL success, x is the exact solution
            // SPEX_UNSYMMETRIC: Matrix is unsymmetric and not a candidate for LDL
            // SPEX_ZERODIAG:    A is symmetric but does not have a nonzero diagonal.
            //                   not a candidate for LDL.
            // Other error code: Some error. Return the error code and exit
            info = SPEX_ldl_backslash(&x, type, A, b, option);

            if (info == SPEX_ZERODIAG || info == SPEX_UNSYMMETRIC)
            {
                // ldl factorization failed but matrix is a candidate 
                // for LU factorization.

                // The LU factorization can return either:
                // SPEX_OK: LU success, x is the exact solution
                // Other error code: Some error. Return the error
                //                   code and exit
                info = SPEX_lu_backslash(&x, type, A, b, option);
            }
    }
    // x contains either the exact solution of the system or is NULL
    (*x_handle) = x;
    // returns SPEX_OK if the algorithm is successful or the appropriate error.
    return info;
}
