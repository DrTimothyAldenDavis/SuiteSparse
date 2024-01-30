//------------------------------------------------------------------------------
// Demo/spex_demos.h: #include file the demo programs
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


#include "SPEX.h"
#include "spex_util_internal.h"
#include "spex_gmp.h"

#define DEMO_OK(method)                         \
{                                               \
    ok = method ;                               \
    if (ok != SPEX_OK)                          \
    {                                           \
        spex_demo_determine_error (ok) ;        \
        FREE_WORKSPACE ;                        \
        return 0 ;                              \
    }                                           \
}

#define SPEX_MIN(a,b) (((a) < (b)) ? (a) : (b))

/* Purpose: This processes the command line for user specified options */
SPEX_info spex_demo_process_command_line //processes the command line
(
    int64_t argc,           // number of command line arguments
    char *argv[],           // set of command line arguments
    SPEX_options option,   // struct containing the command options
    char **mat_name,        // Name of the matrix to be read in
    char **rhs_name,        // Name of the RHS vector to be read in
    int64_t *rat            // data type of output solution.
                            // 1: mpz, 2: double, 3: mpfr
);

/* Purpose: This function reads in a matrix stored in a triplet format.
 * This format used can be seen in any of the example mat files.
 */
SPEX_info spex_demo_tripread
(
    SPEX_matrix *A_handle,     // Matrix to be constructed
    FILE *file,                 // file to read from (must already be open)
    SPEX_type C_type,       // C->type: mpz_t, mpq_t, mpfr_t, int64_t, or double
    SPEX_options option
) ;

/* Purpose: This function reads in a double matrix stored in a triplet format.
 * This format used can be seen in any of the example mat files.
 */
SPEX_info spex_demo_tripread_double
(
    SPEX_matrix *A_handle,     // Matrix to be constructed
    FILE *file,                 // file to read from (must already be open)
    SPEX_options option
) ;

/* Purpose: This function reads in a matrix stored in a triplet format.
 * This format used can be seen in any of the example mat files.
 */
SPEX_info spex_demo_tripread_mpz
(
    SPEX_matrix *A_handle,     // Matrix to be constructed
    FILE *file,                 // file to read from (must already be open)
    SPEX_options option
) ;

/* Purpose: SPEX_read_dense: read a dense matrix. */
SPEX_info spex_demo_read_dense
(
    SPEX_matrix *b_handle,      // Matrix to be constructed
    FILE *file,                  // file to read from (must already be open)
    SPEX_options option
) ;

/* Purpose: Determine why a SPEX_Chol function failed
 */
void spex_demo_determine_error
(
    SPEX_info ok
);


SPEX_info spex_demo_check_solution
(
    const SPEX_matrix A,         // Input matrix
    const SPEX_matrix x,         // Solution vectors
    const SPEX_matrix b,         // Right hand side vectors
    const SPEX_options option    // Command options
);
