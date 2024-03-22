//------------------------------------------------------------------------------
// SPEX/MATLAB/SPEX_mex.h: include file for MATLAB functions
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// This file contains the routines for SPEX's MATLAB interface.
// The MATLAB interface can be used to solve linear systems Ax = b
// using either the left LU factorization, Cholesky factorization,
// or allowing SPEX to choose for the user the correct algorithm.

// Note that the MATLAB interface of SPEX is intended to provide
// accurate solutions to linear systems within MATLAB. Note that there
// are two major limitations to the MATLAB interface:

//   1) The MATLAB interface does not give access to the factors L and/or U.
//      thus, one can not reuse the factors after a solution is computed (nor
//      can they be updated). If a user desires this functionality, they should
//      utilize the C interface.

//
//   2) The MATLAB interface preserves 16 decimal digits of precision from
//      MATLAB.  That is, since the input is floating point, all numbers are
//      assumed to be correct to 16 digits. This is due to the fact that
//      epsilon is about 2e-16 thus, floating point numbers, such as fl(0.9)
//      can not be expressed exactly.  With an exact (floating-point)
//      conversion, fl(0.9) is sent to SPEX as 45000000000000001 /
//      50000000000000000. Using 16 decimal digits of precision, this number is
//      sent to spex as 9/10.  If one wishes to preserve full floating point
//      precision, they may either scale the matrix themselves within MATLAB,
//      or if this is not possible, utilize the C interface, and copy the
//      matrix twice. First it should be copied from SPEX_FP64 (double) to
//      SPEX_MPFR and then from SPEX_MPFR to SPEX_MPQ. Either of these options
//      will preserve as much floating-point precision as the user desires.

#ifndef SPEX_BACKSLASH_MEX_H
#define SPEX_BACKSLASH_MEX_H

#include "SPEX.h"
#include "matrix.h"
#include "mex.h"

#define SPEX_MEX_OK(method)             \
{                                       \
    status = method ;                   \
    if (status != SPEX_OK)              \
    {                                   \
        spex_mex_error (status, "") ;   \
    }                                   \
}


/* These enums determine the type of output desired by the user
 */
typedef enum
{
    SPEX_SOLUTION_DOUBLE = 0,   // x as double
    SPEX_SOLUTION_VPA = 1,      // return x as vpa
    SPEX_SOLUTION_CHAR = 2      // x as cell strings
}
SPEX_solution ;


typedef struct spex_mex_options
{
    SPEX_solution solution ;    // how should x be returned to MATLAB
    int32_t digits ;            // # of digits to use for vpa
}
spex_mex_options ;

/* Purpose: A GMP reallocation function
 * This allows GMP to use MATLAB's default realloc function
 */
void *SPEX_gmp_mex_realloc
(
    void *x,    // void * to be reallocated
    size_t a,   // Previous size
    size_t b    // New size
);

/* Purpose: A GMP free function. This allows GMP to use
 * MATLAB's mxFree instead of free
 */
void SPEX_gmp_mex_free
(
    void *x,    // void * to be freed
    size_t a    // Size
);

/* Purpose: get command options from MATLAB
 */
void spex_mex_get_matlab_options
(
    SPEX_options option,           // Control parameters
    spex_mex_options *mexoptions,   // MATLAB-specific options
    const mxArray* input            // options struct, may be NULL
) ;

// Purpose: This function checks if the array x contains Inf's, NaN's, or
// if its values can be represented as int64_t values.

bool spex_mex_check_for_inf     // true if x can be represented as int64_t
(
    double *x, // The array of numeric values
    mwSize n   // size of array
) ;

/* Purpose: This function reads in the A matrix and right hand side vectors. */
void spex_mex_get_A_and_b
(
    SPEX_matrix *A_handle,     // Internal SPEX Mat stored in CSC
    SPEX_matrix *b_handle,     // mpz matrix used internally
    const mxArray* pargin[],    // The input A matrix and options
    SPEX_options option
) ;

/* Purpose: Report errors if they arise
 */
void spex_mex_error
(
    SPEX_info status,
    char *message
) ;

#endif

