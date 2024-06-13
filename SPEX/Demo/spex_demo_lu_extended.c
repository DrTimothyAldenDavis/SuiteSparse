//------------------------------------------------------------------------------
// Demo/spex_demo_lu_extended.c: extended SPEX_LU example for a double matrix
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// This program will exactly solve the sparse linear system Ax = b by
// performing the SPEX Left LU factorization. This is intended to be a
// demonstration of the "advanced interface" of SPEX Left LU. Refer to
// README.txt for information on how to properly use this code

// usage:
// spex_demo_lu_extended Followed by the listed args:
//
// f (or file) Filename. e.g., spex_demo_lu_extended f MATRIX_NAME RHS_NAME,
// which indicates spex_demo_lu_extended will read matrix from MATRIX_NAME and
// right hand side from RHS_NAME.  The matrix must be stored in Matrix Market
// format. Refer to http://math.nist.gov/MatrixMarket/formats.html for
// information on Matrix Market format.  The right hand side vector must be
// stored as a dense vector.
//
// p (or piv) Pivot_param. e.g., spex_demo_lu_extended p 0, which indicates
// SPEX_LU will use smallest pivot for pivot scheme. Other available options
// are listed as follows:
//
//        0: Smallest pivot: Default and recommended
//        1: Diagonal pivoting
//        2: First nonzero per column chosen as pivot
//        3: Diagonal pivoting with tolerance for smallest pivot
//        4: Diagonal pivoting with tolerance for largest pivot
//        5: Largest pivot
//
// q (or col) Column_order_param. e.g., spex_demo_lu_extended q 1, which
// indicates SPEX_LU will use COLAMD for column ordering. Other available
// options are:
//
//        1: None: Not recommended for sparse matrices
//        2: Default: COLAMD
//        3: AMD
//
// t (or tol) tolerance_param. e.g., spex_demo_lu_extended t 1e-10, which
// indicates SPEX_LU will use 1e-10 as the tolerance for pivot scheme 3 and 4
// mentioned above.  Therefore, it is only necessary if pivot scheme 3 or 4 is
// used.
//
// o (or out). e.g., spex_demo_lu_extended o 1, which indicates SPEX_LU will
// output the errors and warnings during the process. Other available options
// are:
//
//        0: print nothing
//        1: just errors and warnings: Default
//        2: terse, with basic stats from COLAMD/AMD and SPEX and solution
//
//
// If none of the above args is given, they are set to the following default:
//
//  p = 0, i.e., using smallest pivot
//  q = 2, i.e., using COLAMD
//  t = 0.1, not being using since p != 3 or 4

#include "spex_demos.h"

#define FREE_WORKSPACE                          \
{                                               \
    if (mat_file != NULL)                       \
    {                                           \
        fclose(mat_file);                       \
    }                                           \
    mat_file = NULL ;                           \
    if (rhs_file != NULL)                       \
    {                                           \
        fclose(rhs_file);                       \
    }                                           \
    rhs_file = NULL ;                           \
    SPEX_matrix_free(&A, option);               \
    SPEX_symbolic_analysis_free(&S, option);    \
    SPEX_factorization_free(&F, option);        \
    SPEX_matrix_free(&x, option);               \
    SPEX_matrix_free(&b, option);               \
    SPEX_FREE(option);                          \
    SPEX_finalize();                            \
}

int main (int argc, char *argv[])
{

    SPEX_matrix A = NULL;
    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL;
    SPEX_matrix x = NULL;
    SPEX_matrix b = NULL;
    FILE *rhs_file = NULL;
    FILE *mat_file = NULL ;
    SPEX_options option = NULL;

    // Extra parameters used to obtain A, b, etc
    char *mat_name = NULL, *rhs_name = NULL;
    int64_t rat=1;

    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This
    // is done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    SPEX_TRY (SPEX_initialize ( )) ;

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

    // Initialize option, command options for the factorization
    SPEX_TRY (SPEX_create_default_options(&option));
    option->order=SPEX_NO_ORDERING;

    //--------------------------------------------------------------------------
    // After initializing memory, we process the command line for this function.
    // Such a step is optional, a user can also manually set these parameters.
    // For example, if one wished to use the AMD ordering, they can just set
    // option->order = SPEX_AMD.
    //--------------------------------------------------------------------------

    SPEX_TRY (spex_demo_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // In this demo file, we now read in the A and b matrices from external
    // files.  Refer to the example.c file or the user guide for other
    // methods of creating the input matrix. In general, the user can create
    // his/her matrix (say in double form) and then create a copy of it with
    // SPEX_matrix_copy
    //--------------------------------------------------------------------------

    // Read in A
    mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return (1) ;
    }
    SPEX_TRY (spex_demo_tripread(&A, mat_file, SPEX_MPZ, option));
    fclose(mat_file);
    mat_file = NULL ;

    // Read in right hand side
    rhs_file = fopen(rhs_name,"r");
    if( rhs_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return (1) ;
    }
    SPEX_TRY (spex_demo_read_dense(&b, rhs_file, option));
    fclose(rhs_file);
    rhs_file = NULL ;

    // Check if the size of A matches b
    if (A->n != b->m)
    {
        fprintf (stderr, "Error! Size of A and b do not match!\n");
        FREE_WORKSPACE;
        return (1) ;
    }

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
    // We now solve the system Ax=b using the L and U factors computed above.
    //--------------------------------------------------------------------------

    double start_solve = SUITESPARSE_TIME;

    // Solve LDU x = b
    SPEX_TRY (SPEX_lu_solve(&x, F, b, option));

    double end_solve = SUITESPARSE_TIME;

    // Done, x now contains the exact solution of the linear system Ax=b in
    // dense rational form. There is an optional final step here where the user
    // can cast their solution to a different data type or matrix format.
    // Below, we have a block of code which illustrates how one would do this.

    // Example of how to transform output. Uncomment if desired
    //
    // SPEX_kind my_kind = SPEX_DENSE;  // SPEX_CSC, SPEX_TRIPLET or SPEX_DENSE
    // SPEX_type my_type = SPEX_FP64;   // SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    //
    // SPEX_matrix my_x = NULL;        // New output
    // Create copy which is stored as my_kind and my_type:
    // SPEX_matrix_copy( &my_x, my_kind, my_type, x, option);

    // Timing stats
    double t_sym = (end_col-start_col) ;
    double t_factor = (end_factor - start_factor) ;
    double t_solve =  (end_solve - start_solve) ;

    printf("\nNumber of L+U nonzeros: \t\t%"PRId64,
        (F->L->p[F->L->n]) + (F->U->p[F->U->n]) - (F->L->m));
    printf("\nSymbolic analysis time: \t\t%lf", t_sym);
    printf("\nSPEX Left LU Factorization time: \t%lf", t_factor);
    printf("\nFB Substitution time: \t\t\t%lf\n\n", t_solve);

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    fprintf (stderr, "%s: all tests passed\n\n", __FILE__);
    return (0) ;
}

