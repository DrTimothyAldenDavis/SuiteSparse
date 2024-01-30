//------------------------------------------------------------------------------
// Demo/spex_demo_lu_doub.c: example of simple SPEX_LU call on a double matrix
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* This program will exactly solve the sparse linear system Ax = b by
 * performing the SPEX Left LU factorization. This is intended to be a
 * demonstration of the "advanced interface" of SPEX Left LU. Refer to
 * README.txt for information on how to properly use this code
 */

// usage:
// spex_lu_demo Followed by the listed args:
//
// f (or file) Filename. e.g., spex_lu_demo f MATRIX_NAME RHS_NAME, which
// indicates spex_lu_demo will read matrix from MATRIX_NAME and right hand side
// from RHS_NAME.  The matrix must be stored in Matrix Market format. Refer to
// http://math.nist.gov/MatrixMarket/formats.html for information on Matrix
// Market format.  The right hand side vector must be stored as a dense vector.
//
// p (or piv) Pivot_param. e.g., spex_lu_demo p 0, which indicates SPEX_LU will
// use smallest pivot for pivot scheme. Other available options are listed as
// follows:
//
//        0: Smallest pivot: Default and recommended
//        1: Diagonal pivoting
//        2: First nonzero per column chosen as pivot
//        3: Diagonal pivoting with tolerance for smallest pivot
//        4: Diagonal pivoting with tolerance for largest pivot
//        5: Largest pivot
//
// q (or col) Column_order_param. e.g., spex_lu_demo q 1, which indicates
// SPEX_LU will use COLAMD for column ordering. Other available options are:
//
//        0: None: Not recommended for sparse matrices 1: COLAMD: Default 2:
//        AMD
//
// t (or tol) tolerance_param. e.g., spex_lu_demo t 1e-10, which indicates
// SPEX_LU will use 1e-10 as the tolerance for pivot scheme 3 and 4 mentioned
// above.  Therefore, it is only necessary if pivot scheme 3 or 4 is used.
//
// o (or out). e.g., spex_lu_demo o 1, which indicates SPEX_LU will output the
// errors and warnings during the process. Other available options are: 0:
// print nothing 1: just errors and warnings: Default 2: terse, with basic
// stats from COLAMD/AMD and SPEX and solution
//
// If none of the above args is given, they are set to the following default:
//

#include "spex_demos.h"

#define FREE_WORKSPACE                           \
{                                                \
    SPEX_matrix_free(&A, option);                \
    SPEX_factorization_free(&F, option);         \
    SPEX_symbolic_analysis_free(&S, option);     \
    SPEX_FREE(option);                           \
    SPEX_finalize();                             \
}

int main (int argc, char *argv[])
{

    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This
    // is done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    SPEX_initialize();

    //--------------------------------------------------------------------------
    // We first initialize the default parameters. These parameters are modified
    // either via command line arguments or when reading in data. The important
    // initializations are in the block below.
    //
    // First, we initialize 6 SPEX_matrices. Note that these matrices must
    // simply be declared, they will be created and allocated within their
    // respective functions. These matrices are:
    //
    //  A:  User input matrix. Must be SPEX_CSC and SPEX_MPZ for routines
    //
    //  L:  Lower triangular matrix. Will be output as SPEX_CSC and SPEX_MPZ
    //
    //  U:  Upper triangular matrix. Will be output as SPEX_CSC and SPEX_MPZ
    //
    //  x:  Solution to the linear system. Will be output as SPEX_DENSE and
    //      SPEX_MPQ
    //
    //  b:  Set of right hand side vectors. Must be SPEX_DENSE and SPEX_MPZ
    //
    // Additionally, two other data structures are declared here:
    //
    //  pinv:   Inverse row permutation used for LDU factorization and
    //          permutation
    //
    //  S:      Symbolic analysis struct.
    //
    // Lastly, the following parameter is created:
    //
    //  option: Command options for the factorization. In general, this can be
    //          set to default values and is almost always the last input
    //          argument for SPEX Left LU functions (except SPEX_malloc and
    //          such)
    //--------------------------------------------------------------------------
    SPEX_matrix A = NULL;
    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL;
    SPEX_info ok ;

    // Initialize option, command options for the factorization
    SPEX_options option = NULL;
    DEMO_OK(SPEX_create_default_options(&option));

    // Extra parameters used to obtain A, b, etc
    char *mat_name, *rhs_name;
    int64_t rat = 1;

    //--------------------------------------------------------------------------
    // After initializing memory, we process the command line for this function.
    // Such a step is optional, a user can also manually set these parameters.
    // For example, if one wished to use the AMD ordering, they can just set
    // option->order = SPEX_AMD.
    //--------------------------------------------------------------------------

    DEMO_OK(spex_demo_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // In this demo file, we now read in the A and b matrices from external
    // files.  Refer to the example.c file or the user guide for other
    // methods of creating the input matrix. In general, the user can create
    // his/her matrix (say in double form) and then create a copy of it with
    // SPEX_matrix_copy
    //--------------------------------------------------------------------------

    // Read in A
    FILE *mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }

    DEMO_OK(spex_demo_tripread(&A, mat_file, SPEX_FP64, option));
    fclose(mat_file);

#if 0
    // Read in right hand side
    FILE *rhs_file = fopen(rhs_name,"r");
    if( rhs_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    DEMO_OK(SPEX_read_dense(&b, rhs_file, option));
    fclose(rhs_file);

    // Check if the size of A matches b
    if (A->n != b->m)
    {
        fprintf (stderr, "Error! Size of A and b do not match!\n");
        FREE_WORKSPACE;
        return 0;
    }
#endif

    //--------------------------------------------------------------------------
    // We now perform symbolic analysis by getting the column preordering of
    // the matrix A. This is done via the SPEX_lu_analyze function. The output
    // of this function is a column permutation Q where we factor the matrix AQ
    // and an estimate of the number of nonzeros in L and U.
    //
    // Note that in the simple interface demostrated in the example*.c files,
    // all of the following code is condensed into the single SPEX_backslash
    // function.
    //--------------------------------------------------------------------------

    double start_col = SuiteSparse_time ();

    // Column ordering using either AMD, COLAMD or nothing
    DEMO_OK(SPEX_lu_analyze(&S, A, option));

    double end_col = SuiteSparse_time ();

    //--------------------------------------------------------------------------
    // Now we perform the SPEX Left LU factorization to obtain matrices L and U
    // and a row permutation P such that PAQ = LDU. Note that the D matrix is
    // never explicitly constructed or used.
    //--------------------------------------------------------------------------

    double start_factor = SuiteSparse_time ();

    ok = SPEX_lu_factorize(&F, A, S, option);
    if (ok != SPEX_OK)
    {
        if (ok == SPEX_SINGULAR)
        {
            printf("\nSingular");
        }
        return 0;
    }

    double end_factor = SuiteSparse_time ();

    //--------------------------------------------------------------------------
    // We now solve the system Ax=b using the L and U factors computed above.
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
    return 0;
}

