//------------------------------------------------------------------------------
// SLIP_LU/MATLAB/SLIP_LU_mex.h: include file for MATLAB functions
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#ifndef SLIP_MEX_H
#define SLIP_MEX_H

#include "slip_internal.h"
#include "matrix.h"

#define SLIP_MEX_OK(method)             \
{                                       \
    status = method ;                   \
    if (status != SLIP_OK)              \
    {                                   \
        slip_mex_error (status, "") ;   \
    }                                   \
}

typedef enum
{
    SLIP_SOLUTION_DOUBLE = 0,   // x as double
    SLIP_SOLUTION_VPA = 1,      // return x as vpa
    SLIP_SOLUTION_CHAR = 2      // x as cell strings
}
SLIP_solution ;

typedef struct slip_mex_options
{
    SLIP_solution solution ;    // how should x be returned to MATLAB
    int32_t digits ;            // # of digits to use for vpa
}
slip_mex_options ;

/* Purpose: A GMP reallocation function
 * This allows GMP to use MATLAB's default realloc function
 */
void* SLIP_gmp_mex_realloc
(
    void* x,    // void* to be reallocated
    size_t a,   // Previous size
    size_t b    // New size
);

/* Purpose: A GMP free function. This allows GMP to use
 * MATLAB's mxFree instead of free
 */
void SLIP_gmp_mex_free
(
    void* x,    // void* to be freed
    size_t a    // Size
);

void slip_get_matlab_options
(
    SLIP_options* option,           // Control parameters
    slip_mex_options *mexoptions,   // MATLAB-specific options
    const mxArray* input            // options struct, may be NULL
) ;

// Purpose: This function checks if the array x contains Inf's, NaN's, or
// if its values can be represented as int64_t values.

bool slip_mex_check_for_inf     // true if x can be represented as int64_t
(
    double* x, // The array of numeric values
    mwSize n   // size of array
) ;

/* Purpose: This function reads in the A matrix and right hand side vectors. */
void slip_mex_get_A_and_b
(
    SLIP_matrix **A_handle,     // Internal SLIP Mat stored in CSC
    SLIP_matrix **b_handle,     // mpz matrix used internally
    const mxArray* pargin[],    // The input A matrix and options
    SLIP_options* option
) ;

/* Purpose: Report errors if they arise
 */
void slip_mex_error
(
    SLIP_info status,
    char *message
) ;

#endif

