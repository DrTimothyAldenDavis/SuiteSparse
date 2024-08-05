//------------------------------------------------------------------------------
// SPEX_LU/SPEX_lu_backslash: solve Ax=b, return solution as desired data type
//------------------------------------------------------------------------------

// SPEX_LU: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SPEX Left LU factorization to exactly solve
 * the linear system Ax = b. This is essentially an exact version of MATLAB
 * sparse backslash
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

# define SPEX_FREE_WORKSPACE                    \
    SPEX_factorization_free(&F, option);        \
    SPEX_symbolic_analysis_free (&S, option);

# define SPEX_FREE_ALL              \
    SPEX_FREE_WORKSPACE             \
    SPEX_matrix_free(&x, NULL);     \

#include "spex_lu_internal.h"

SPEX_info SPEX_lu_backslash
(
    // Output
    SPEX_matrix *x_handle,        // Final solution vector
    // Input
    SPEX_type type,               // Type of output desired. Must be
                                  // SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    const SPEX_matrix A,          // Input matrix
    const SPEX_matrix b,          // Right hand side vector(s)
    const SPEX_options option     // Command options
)
{

    //-------------------------------------------------------------------------
    // check inputs
    //-------------------------------------------------------------------------

    SPEX_info info ;
    if (!spex_initialized ( )) return (SPEX_PANIC);
    
    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_LU_LEFT)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    if (x_handle == NULL)
    {
        return SPEX_INCORRECT_INPUT;
    }
    (*x_handle) = NULL;

    if (type != SPEX_MPQ && type != SPEX_FP64 && type != SPEX_MPFR)
    {
        return SPEX_INCORRECT_INPUT;
    }

    SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ);
    SPEX_REQUIRE (b, SPEX_DENSE, SPEX_MPZ);

    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL ;
    SPEX_matrix x = NULL;

    //--------------------------------------------------------------------------
    // Symbolic Analysis
    //--------------------------------------------------------------------------

    SPEX_CHECK(SPEX_lu_analyze(&S, A, option));

    //--------------------------------------------------------------------------
    // LU Factorization
    //--------------------------------------------------------------------------

    SPEX_CHECK(SPEX_lu_factorize(&F, A, S, option));
    
    //--------------------------------------------------------------------------
    // Solve
    //--------------------------------------------------------------------------

    SPEX_CHECK (SPEX_lu_solve (&x, F, b, option));

    //--------------------------------------------------------------------------
    // Now, x contains the exact solution of the linear system in mpq_t
    // precision set the output.
    //--------------------------------------------------------------------------

    if (type == SPEX_MPQ)
    {
        (*x_handle) = x ;
    }
    else
    {
        SPEX_matrix x2 = NULL ;
        SPEX_CHECK (SPEX_matrix_copy (&x2, SPEX_DENSE, type, x, option));
        (*x_handle) = x2 ;
        SPEX_matrix_free (&x, NULL);
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    SPEX_FREE_WORKSPACE ;
    return (SPEX_OK);
}

