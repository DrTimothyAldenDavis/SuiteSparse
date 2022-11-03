//------------------------------------------------------------------------------
// SPEX_Left_LU/Demo/spex_lu_demo.c: example main program for SPEX
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "demos.h"

/* This program will exactly solve the sparse linear system Ax = b by
 * performing the SPEX Left LU factorization. This is intended to be a demonstration
 * of the "advanced interface" of SPEX Left LU. Refer to README.txt for
 * information on how to properly use this code
 */

// usage:
// spex_lu_demo Followed by the listed args:
//
// f (or file) Filename. e.g., spex_lu_demo f MATRIX_NAME RHS_NAME, which indicates
// spex_lu_demo will read matrix from MATRIX_NAME and right hand side from RHS_NAME.
// The matrix must be stored in Matrix Market format. Refer to
// http://math.nist.gov/MatrixMarket/formats.html for information on
// Matrix Market format.
// The right hand side vector must be stored as a dense vector.
//
// p (or piv) Pivot_param. e.g., spex_lu_demo p 0, which indicates SPEX_Left_LU will use
// smallest pivot for pivot scheme. Other available options are listed
// as follows:
//        0: Smallest pivot: Default and recommended
//        1: Diagonal pivoting
//        2: First nonzero per column chosen as pivot
//        3: Diagonal pivoting with tolerance for smallest pivot
//        4: Diagonal pivoting with tolerance for largest pivot
//        5: Largest pivot
//
// q (or col) Column_order_param. e.g., spex_lu_demo q 1, which indicates SPEX_Left_LU
// will use COLAMD for column ordering. Other available options are:
//        0: None: Not recommended for sparse matrices
//        1: COLAMD: Default
//        2: AMD
//
// t (or tol) tolerance_param. e.g., spex_lu_demo t 1e-10, which indicates SPEX_Left_LU
// will use 1e-10 as the tolerance for pivot scheme 3 and 4 mentioned above.
// Therefore, it is only necessary if pivot scheme 3 or 4 is used.
//
// o (or out). e.g., spex_lu_demo o 1, which indicates SPEX_Left_LU will output the
// errors and warnings during the process. Other available options are:
//        0: print nothing
//        1: just errors and warnings: Default
//        2: terse, with basic stats from COLAMD/AMD and SPEX and solution
//
//
// If none of the above args is given, they are set to the following default:
//
//  mat_name = "../ExampleMats/10teams_mat.txt"
//  rhs_name = "../ExampleMats/10teams_v.txt"
//  p = 0, i.e., using smallest pivot
//  q = 1, i.e., using COLAMD
//  t = 0.1, not being using since p != 3 or 4


#define FREE_WORKSPACE                           \
    SPEX_matrix_free(&A, option);                \
    SPEX_matrix_free(&L, option);                \
    SPEX_matrix_free(&U, option);                \
    SPEX_matrix_free(&x, option);                \
    SPEX_matrix_free(&b, option);                \
    SPEX_matrix_free(&rhos, option);             \
    SPEX_FREE(pinv);                             \
    SPEX_LU_analysis_free(&S, option);           \
    SPEX_FREE(option);                           \
    SPEX_finalize( ) ;

int main (int argc, char* argv[])
{

    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This is done
    // by calling the SPEX_initialize() function.
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
    //  x:  Solution to the linear system. Will be output as SPEX_DENSE and SPEX_MPQ
    //
    //  b:  Set of right hand side vectors. Must be SPEX_DENSE and SPEX_MPZ
    //
    // Additionally, two other data structures are declared here:
    //
    //  pinv:   Inverse row permutation used for LDU factorization and permutation
    //
    //  S:      Symbolic analysis struct.
    //
    // Lastly, the following parameter is created:
    //
    //  option: Command options for the factorization. In general, this can be 
    //          set to default values and is almost always the last input argument
    //          for SPEX Left LU functions (except SPEX_malloc and such)
    //--------------------------------------------------------------------------
    SPEX_matrix *A = NULL;
    SPEX_matrix *L = NULL;
    SPEX_matrix *U = NULL;
    SPEX_matrix *x = NULL;
    SPEX_matrix *b = NULL;
    SPEX_matrix *rhos = NULL;
    int64_t* pinv = NULL;
    SPEX_LU_analysis* S = NULL;
    SPEX_info ok ;
    
    // Initialize option, command options for the factorization
    SPEX_options *option = NULL;
    OK(SPEX_create_default_options(&option));
    
    // Extra parameters used to obtain A, b, etc
    char *mat_name, *rhs_name;
    SPEX_type rat;
    mat_name = "../ExampleMats/10teams_mat.txt";// Set demo matrix and RHS name
    rhs_name = "../ExampleMats/10teams_v.txt";
    
    //--------------------------------------------------------------------------
    // After initializing memory, we process the command line for this function.
    // Such a step is optional, a user can also manually set these parameters.
    // For example, if one wished to use the AMD ordering, they can just set
    // option->order = SPEX_AMD.
    //--------------------------------------------------------------------------

    OK(SPEX_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // In this demo file, we now read in the A and b matrices from external
    // files.  Refer to the example.c file or the user guide for other
    // methods of creating the input matrix. In general, the user can create 
    // his/her matrix (say in double form) and then create a copy of it with
    // SPEX_matrix_copy
    //--------------------------------------------------------------------------

    // Read in A
    FILE* mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    OK(SPEX_tripread(&A, mat_file, option));
    fclose(mat_file);

    // Read in right hand side
    FILE* rhs_file = fopen(rhs_name,"r");
    if( rhs_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    OK(SPEX_read_dense(&b, rhs_file, option));
    fclose(rhs_file);

    // Check if the size of A matches b
    if (A->n != b->m)
    {
        fprintf (stderr, "Error! Size of A and b do not match!\n");
        FREE_WORKSPACE;
        return 0;
    }

    //--------------------------------------------------------------------------
    // We now perform symbolic analysis by getting the column preordering of
    // the matrix A. This is done via the SPEX_LU_analyze function. The output
    // of this function is a column permutation Q where we factor the matrix AQ
    // and an estimate of the number of nonzeros in L and U.
    //
    // Note that in the simple interface demostrated in the example*.c files,
    // all of the following code is condensed into the single SPEX_backslash
    // function.
    //--------------------------------------------------------------------------

    clock_t start_col = clock();

    // Column ordering using either AMD, COLAMD or nothing
    OK(SPEX_LU_analyze(&S, A, option));
    if (option->print_level > 0)
    {
        SPEX_print_options(option);
    }

    clock_t end_col = clock();

    //--------------------------------------------------------------------------
    // Now we perform the SPEX Left LU factorization to obtain matrices L and U and a
    // row permutation P such that PAQ = LDU. Note that the D matrix is never
    // explicitly constructed or used.
    //--------------------------------------------------------------------------

    clock_t start_factor = clock();

    OK(SPEX_Left_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));

    clock_t end_factor = clock();

    //--------------------------------------------------------------------------
    // We now solve the system Ax=b using the L and U factors computed above.
    //--------------------------------------------------------------------------

    clock_t start_solve = clock();

    // SPEX Left LU has an optional check step which can verify that the solution
    // vector x satisfies Ax=b in perfect precision intended for debugging.
    //
    // Note that this is entirely optional and not necessary. The solution
    // returned is guaranteed to be exact.   It appears here just as a
    // verification that SPEX Left LU is computing its expected result.  This test
    // can fail only if it runs out of memory, or if there is a bug in the
    // code.  Also, note that this function can be quite time consuming; thus
    // it is not recommended to be used in general.
    // 
    // To enable said check, the following bool is set to true
    //
    option->check = true; 
    
    // Solve LDU x = b
    OK(SPEX_Left_LU_solve(&x, b,
        (const SPEX_matrix *) A,
        (const SPEX_matrix *) L,
        (const SPEX_matrix *) U,
        (const SPEX_matrix *) rhos,
                     S,
        (const int64_t *) pinv,
                     option));    

    clock_t end_solve = clock();

    // Done, x now contains the exact solution of the linear system Ax=b in 
    // dense rational form. There is an optional final step here where the user
    // can cast their solution to a different data type or matrix format.
    // Below, we have a block of code which illustrates how one would do this.

    // Example of how to transform output. Uncomment if desired
    //
    // SPEX_kind my_kind = SPEX_DENSE;  // SPEX_CSC, SPEX_TRIPLET or SPEX_DENSE
    // SPEX_type my_type = SPEX_FP64;   // SPEX_MPQ, SPEX_MPFR, or SPEX_FP64
    //
    // SPEX_matrix* my_x = NULL;        // New output
    // Create copy which is stored as my_kind and my_type:
    // SPEX_matrix_copy( &my_x, my_kind, my_type, x, option);

    // Timing stats
    double t_sym = (double) (end_col-start_col)/CLOCKS_PER_SEC;
    double t_factor = (double) (end_factor - start_factor) / CLOCKS_PER_SEC;
    double t_solve =  (double) (end_solve - start_solve) / CLOCKS_PER_SEC;

    printf("\nNumber of L+U nonzeros: \t\t%"PRId64,
        (L->p[L->n]) + (U->p[U->n]) - (L->m));
    printf("\nSymbolic analysis time: \t\t%lf", t_sym);
    printf("\nSPEX Left LU Factorization time: \t%lf", t_factor);
    printf("\nFB Substitution time: \t\t\t%lf\n\n", t_solve);

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    printf ("\n%s: all tests passed\n\n", __FILE__) ;
    return 0;
}

