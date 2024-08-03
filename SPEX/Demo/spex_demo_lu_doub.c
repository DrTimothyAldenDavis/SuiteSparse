//------------------------------------------------------------------------------
// Demo/spex_demo_lu_doub.c: example of simple SPEX_LU call on a double matrix
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// usage:
// spex_lu_demo_doub Followed by the listed args:
//
// f (or file) Filename. e.g., spex_lu_demo_doub f MATRIX_NAME RHS_NAME, which
// indicates spex_lu_demo_doub will read matrix from MATRIX_NAME and right hand
// side from RHS_NAME.  The matrix must be stored in Matrix Market format.
// Refer to http://math.nist.gov/MatrixMarket/formats.html for information on
// Matrix Market format.  The right hand side vector must be stored as a dense
// vector.
//
// p (or piv) Pivot_param. e.g., spex_lu_demo_doub p 0, which indicates SPEX_LU
// will use smallest pivot for pivot scheme. Other available options are listed
// as follows:
//
//        0: Smallest pivot: Default and recommended
//        1: Diagonal pivoting
//        2: First nonzero per column chosen as pivot
//        3: Diagonal pivoting with tolerance for smallest pivot
//        4: Diagonal pivoting with tolerance for largest pivot
//        5: Largest pivot
//
// q (or col) Column_order_param. e.g., spex_lu_demo_doub q 1, which indicates
// SPEX_LU will use no column ordering. Other available options are:
//
//        1: None: Not recommended for sparse matrices 2: COLAMD: Default 3:
//        AMD
//
// t (or tol) tolerance_param. e.g., spex_lu_demo_doub t 1e-10, which indicates
// SPEX_LU will use 1e-10 as the tolerance for pivot scheme 3 and 4 mentioned
// above.  Therefore, it is only necessary if pivot scheme 3 or 4 is used.
//
// o (or out). e.g., spex_lu_demo_doub o 1, which indicates SPEX_LU will output
// the errors and warnings during the process. Other available options are: 0:
// print nothing 1: just errors and warnings: Default 2: terse, with basic
// stats from COLAMD/AMD and SPEX and solution

#include "spex_demos.h"

#define FREE_WORKSPACE                          \
{                                               \
    if (mat_file != NULL)                       \
    {                                           \
        fclose(mat_file);                       \
    }                                           \
    mat_file = NULL ;                           \
    SPEX_matrix_free(&A, option);               \
    SPEX_factorization_free(&F, option);        \
    SPEX_symbolic_analysis_free(&S, option);    \
    SPEX_FREE(option);                          \
    SPEX_finalize();                            \
}

int main (int argc, char *argv[])
{

    SPEX_matrix A = NULL;
    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL;
    FILE *mat_file = NULL;
    SPEX_options option = NULL;
    char *mat_name = NULL, *rhs_name = NULL;
    int64_t rat = 1;

    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This
    // is done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    SPEX_TRY (SPEX_initialize ( )) ;

    // Initialize option, command options for the factorization
    SPEX_TRY (SPEX_create_default_options(&option));

    //--------------------------------------------------------------------------
    // After initializing memory, we process the command line for this function.
    // Such a step is optional, a user can also manually set these parameters.
    // For example, if one wished to use the AMD ordering, they can just set
    // option->order = SPEX_AMD.
    //--------------------------------------------------------------------------

    SPEX_TRY (spex_demo_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // Read in A
    //--------------------------------------------------------------------------

    mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return (1) ;
    }

    SPEX_TRY (spex_demo_tripread(&A, mat_file, SPEX_FP64, option));
    fclose(mat_file);
    mat_file = NULL ;

    //--------------------------------------------------------------------------
    // We now perform symbolic analysis by getting the column preordering of
    // the matrix A. This is done via the SPEX_lu_analyze function. The output
    // of this function is a column permutation Q where we factor the matrix AQ
    // and an estimate of the number of nonzeros in L and U.
    //--------------------------------------------------------------------------

    double start_col = SUITESPARSE_TIME;

    // Column ordering using either AMD, COLAMD or nothing
    SPEX_TRY (SPEX_lu_analyze(&S, A, option));

    double end_col = SUITESPARSE_TIME;

    //--------------------------------------------------------------------------
    // Now we perform the SPEX Left LU factorization to obtain matrices L and U
    // and a row permutation P such that PAQ = LDU. Note that the D matrix is
    // never explicitly constructed or used.
    //--------------------------------------------------------------------------

    double start_factor = SUITESPARSE_TIME;

    SPEX_TRY (SPEX_lu_factorize(&F, A, S, option));

    double end_factor = SUITESPARSE_TIME;

    //--------------------------------------------------------------------------
    // print results
    //--------------------------------------------------------------------------

    // Timing stats
    double t_sym = (end_col-start_col) ;
    double t_factor = (end_factor - start_factor) ;

    printf("\nNumber of L+U nonzeros: \t\t%"PRId64,
        (F->L->p[F->L->n]) + (F->U->p[F->U->n]) - (F->L->m));
    printf("\nSymbolic analysis time: \t\t%lf", t_sym);
    printf("\nSPEX Left LU Factorization time: \t%lf", t_factor);

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    printf ("\n%s: all tests passed\n\n", __FILE__);
    return (0) ;
}

