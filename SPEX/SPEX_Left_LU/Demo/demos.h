//------------------------------------------------------------------------------
// SPEX_Left_LU/Demo/demos.h: #include file the demo programs
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "SPEX.h"
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

#define OK(method)                      \
{                                       \
    ok = method ;                       \
    if (ok != SPEX_OK)                  \
    {                                   \
        printf ("Error: %d line %d file %s\n", ok, __LINE__, __FILE__) ; \
        FREE_WORKSPACE ;                \
        return 0 ;                      \
    }                                   \
}

#define SPEX_MIN(a,b) (((a) < (b)) ? (a) : (b))

/* Purpose: This processes the command line for user specified options */
SPEX_info SPEX_process_command_line //processes the command line
(
    int argc,               // number of command line arguments
    char* argv[],           // set of command line arguments
    SPEX_options* option,   // struct containing the command options
    char** mat_name,        // Name of the matrix to be read in
    char** rhs_name,        // Name of the RHS vector to be read in
    SPEX_type *rat          // data type of output solution:
                            // 1:SPEX_MPZ (default), 2:SPEX_FP64, 3:SPEX_MPFR
);

/* Purpose: This function prints out the user specified/default options*/
void SPEX_print_options     // display specified/default options to user
(
    SPEX_options* option // struct containing all of the options
);

/* Purpose: This function shows the usage of the code.*/
void SPEX_show_usage(void);

/* Purpose: This function reads in a matrix stored in a triplet format.
 * This format used can be seen in any of the example mat files.
 */
SPEX_info SPEX_tripread
(
    SPEX_matrix **A_handle,     // Matrix to be constructed
    FILE* file,                 // file to read from (must already be open)
    SPEX_options* option
) ;

/* Purpose: This function reads in a double matrix stored in a triplet format.
 * This format used can be seen in any of the example mat files.
 */
SPEX_info SPEX_tripread_double
(
    SPEX_matrix **A_handle,     // Matrix to be constructed
    FILE* file,                 // file to read from (must already be open)
    SPEX_options* option
) ;

/* Purpose: SPEX_read_dense: read a dense matrix. */
SPEX_info SPEX_read_dense
(
    SPEX_matrix **b_handle,      // Matrix to be constructed
    FILE* file,                  // file to read from (must already be open)
    SPEX_options* option
) ;
