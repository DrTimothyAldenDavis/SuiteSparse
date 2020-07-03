//------------------------------------------------------------------------------
// SLIP_LU/SLIP_backslash: solve Ax=b, returning solution as desired data type
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This code utilizes the SLIP LU factorization to exactly solve the
 *          linear system Ax = b. This is essentially an exact version of
 *          MATLAB sparse backslash
 *
 * Input/Output arguments:
 *
 * X_handle:    A pointer to the solution of the linear system. The output is
 *              allowed to be returned in either double precision, mpfr_t, or
 *              rational mpq_t
 *
 * type:        Data structure of output desired. Must be either SLIP_MPQ,
 *              SLIP_FP64, or SLIP_MPFR
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

# define SLIP_FREE_WORK             \
    SLIP_matrix_free(&L, NULL);     \
    SLIP_matrix_free(&U, NULL);     \
    SLIP_FREE(pinv);                \
    SLIP_matrix_free(&rhos, NULL);  \
    SLIP_LU_analysis_free (&S, NULL);

# define SLIP_FREE_ALL              \
    SLIP_FREE_WORK                  \
    SLIP_matrix_free(&x, NULL);     \

#include "slip_internal.h"

SLIP_info SLIP_backslash
(
    // Output
    SLIP_matrix **X_handle,       // Final solution vector
    // Input
    SLIP_type type,               // Type of output desired
                                  // Must be SLIP_MPQ, SLIP_MPFR, or SLIP_FP64
    const SLIP_matrix *A,         // Input matrix
    const SLIP_matrix *b,         // Right hand side vector(s)
    const SLIP_options* option    // Command options
)
{

    //-------------------------------------------------------------------------
    // check inputs
    //-------------------------------------------------------------------------

    SLIP_info info ;
    if (!slip_initialized ( )) return (SLIP_PANIC) ;

    if (X_handle == NULL)
    {
        return SLIP_INCORRECT_INPUT;
    }
    (*X_handle) = NULL;

    if (type != SLIP_MPQ && type != SLIP_FP64 && type != SLIP_MPFR)
    {
        return SLIP_INCORRECT_INPUT;
    }

    SLIP_REQUIRE (A, SLIP_CSC,   SLIP_MPZ) ;
    SLIP_REQUIRE (b, SLIP_DENSE, SLIP_MPZ) ;

    SLIP_matrix *L = NULL ;
    SLIP_matrix *U = NULL ;
    SLIP_matrix *x = NULL;
    int64_t *pinv = NULL ;
    SLIP_matrix *rhos = NULL ;
    SLIP_LU_analysis *S = NULL;

    //--------------------------------------------------------------------------
    // Symbolic Analysis
    //--------------------------------------------------------------------------

    SLIP_CHECK(SLIP_LU_analyze(&S, A, option));

    //--------------------------------------------------------------------------
    // LU Factorization
    //--------------------------------------------------------------------------

    SLIP_CHECK(SLIP_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));

    //--------------------------------------------------------------------------
    // Solve
    //--------------------------------------------------------------------------

    SLIP_CHECK (SLIP_LU_solve (&x, b, A,
        (const SLIP_matrix *) L,
        (const SLIP_matrix *) U,
        (const SLIP_matrix *) rhos,
        S,
        (const int64_t *) pinv,
        option)) ;

    //--------------------------------------------------------------------------
    // Now, x contains the exact solution of the linear system in mpq_t
    // precision set the output.
    //--------------------------------------------------------------------------

    if (type == SLIP_MPQ)
    {
        (*X_handle) = x ;
    }
    else
    {
        SLIP_matrix* x2 = NULL ;
        SLIP_CHECK (SLIP_matrix_copy (&x2, SLIP_DENSE, type, x, option)) ;
        (*X_handle) = x2 ;
        SLIP_matrix_free (&x, NULL) ;
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    SLIP_FREE_WORK ;
    return (SLIP_OK) ;
}

