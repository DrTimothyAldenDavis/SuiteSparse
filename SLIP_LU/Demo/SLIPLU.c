//------------------------------------------------------------------------------
// SLIP_LU/Demo/SLIPLU.c: example main program for SLIP_LU
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#include "demos.h"

/* This program will exactly solve the sparse linear system Ax = b by
 * performing the SLIP LU factorization. This is intended to be a demonstration
 * of the "advanced interface" of SLIP LU. Refer to README.txt for
 * information on how to properly use this code
 */

// usage:
// SLIPLU Followed by the listed args:
//
// help. e.g., SLIPLU help, which indicates SLIPLU to print to guideline
// for using this function.
//
// f (or file) Filename. e.g., SLIPLU f MATRIX_NAME RHS_NAME, which indicates
// SLIPLU will read matrix from MATRIX_NAME and right hand side from RHS_NAME.
// The matrix must be stored in Matrix Market format. Refer to
// http://math.nist.gov/MatrixMarket/formats.html for information on
// Matrix Market format.
// The right hand side vector must be stored as a dense vector.
//
// p (or piv) Pivot_param. e.g., SLIPLU p 0, which inidcates SLIPLU will use
// smallest pivot for pivot scheme. Other available options are listed
// as follows:
//        0: Smallest pivot: Default and recommended
//        1: Diagonal pivoting
//        2: First nonzero per column chosen as pivot
//        3: Diagonal pivoting with tolerance for smallest pivot
//        4: Diagonal pivoting with tolerance for largest pivot
//        5: Largest pivot
//
// q (or col) Column_order_param. e.g., SLIPLU q 1, which indicates SLIPLU
// will use COLAMD for column ordering. Other available options are:
//        0: None: Not recommended for sparse matrices
//        1: COLAMD: Default
//        2: AMD
//
// t (or tol) tolerance_param. e.g., SLIPLU t 1e-10, which indicates SLIPLU
// will use 1e-10 as the tolerance for pivot scheme 3 and 4 mentioned above.
// Therefore, it is only necessary if pivot scheme 3 or 4 is used.
//
// o (or out). e.g., SLIPLU o 1, which indicates SLIPLU will output the
// errors and warnings during the process. Other available options are:
//        0: print nothing
//        1: just errors and warnings: Default
//        2: terse, with basic stats from COLAMD/AMD and SLIP and solution
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
    SLIP_matrix_free(&A, option);                \
    SLIP_matrix_free(&L, option);                \
    SLIP_matrix_free(&U, option);                \
    SLIP_matrix_free(&x, option);                \
    SLIP_matrix_free(&b, option);                \
    SLIP_matrix_free(&rhos, option);             \
    SLIP_FREE(pinv);                             \
    SLIP_LU_analysis_free(&S, option);           \
    SLIP_FREE(option);                           \
    SLIP_finalize( ) ;

int main (int argc, char* argv[])
{

    //--------------------------------------------------------------------------
    // Prior to using SLIP LU, its environment must be initialized. This is done
    // by calling the SLIP_initialize() function.
    //--------------------------------------------------------------------------

    SLIP_initialize();

    //--------------------------------------------------------------------------
    // We first initialize the default parameters. These parameters are modified
    // either via command line arguments or when reading in data. The important
    // initializations are in the block below.
    //
    // First, we initialize 6 SLIP_matrices. Note that these matrices must
    // simply be declared, they will be created and allocated within their
    // respective functions. These matrices are:
    //
    //  A:  User input matrix. Must be SLIP_CSC and SLIP_MPZ for routines
    //  
    //  L:  Lower triangular matrix. Will be output as SLIP_CSC and SLIP_MPZ
    //
    //  U:  Upper triangular matrix. Will be output as SLIP_CSC and SLIP_MPZ 
    //
    //  x:  Solution to the linear system. Will be output as SLIP_DENSE and SLIP_MPQ
    //
    //  b:  Set of right hand side vectors. Must be SLIP_DENSE and SLIP_MPZ
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
    //          for SLIP LU functions (except SLIP_malloc and such)
    //--------------------------------------------------------------------------
    SLIP_matrix *A = NULL;
    SLIP_matrix *L = NULL;
    SLIP_matrix *U = NULL;
    SLIP_matrix *x = NULL;
    SLIP_matrix *b = NULL;
    SLIP_matrix *rhos = NULL;
    int64_t* pinv = NULL;
    SLIP_LU_analysis* S = NULL;
    
    // Initialize option, command options for the factorization
    SLIP_options *option = SLIP_create_default_options();
    
    // Extra parameters used to obtain A, b, etc
    SLIP_info ok ;
    char *mat_name, *rhs_name;
    SLIP_type rat;
    mat_name = "../ExampleMats/10teams_mat.txt";// Set demo matrix and RHS name
    rhs_name = "../ExampleMats/10teams_v.txt";
    
    if (!option)
    {
        fprintf (stderr, "Error! OUT of MEMORY!\n");
        SLIP_finalize();
        return 0;
    }

    //--------------------------------------------------------------------------
    // After initializing memory, we process the command line for this function.
    // Such a step is optional, a user can also manually set these parameters.
    // For example, if one wished to use the AMD ordering, they can just set
    // option->order = SLIP_AMD.
    //--------------------------------------------------------------------------

    bool help ;
    OK(SLIP_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat, &help));
    if (help) return (0) ;

    //--------------------------------------------------------------------------
    // In this demo file, we now read in the A and b matrices from external
    // files.  Refer to the example.c file or the user guide for other
    // methods of creating the input matrix. In general, the user can create 
    // his/her matrix (say in double form) and then create a copy of it with
    // SLIP_matrix_copy
    //--------------------------------------------------------------------------

    // Read in A
    FILE* mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    OK(SLIP_tripread(&A, mat_file, option));
    fclose(mat_file);

    // Read in right hand side
    FILE* rhs_file = fopen(rhs_name,"r");
    if( rhs_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    OK(SLIP_read_dense(&b, rhs_file, option));
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
    // the matrix A. This is done via the SLIP_LU_analyze function. The output
    // of this function is a column permutation Q where we factor the matrix AQ
    // and an estimate of the number of nonzeros in L and U.
    //
    // Note that in the simple interface demostrated in the example*.c files,
    // all of the following code is condensed into the single SLIP_backslash
    // function.
    //--------------------------------------------------------------------------

    clock_t start_col = clock();

    // Column ordering using either AMD, COLAMD or nothing
    OK(SLIP_LU_analyze(&S, A, option));
    if (option->print_level > 0)
    {
        SLIP_print_options(option);
    }

    clock_t end_col = clock();

    //--------------------------------------------------------------------------
    // Now we perform the SLIP LU factorization to obtain matrices L and U and a
    // row permutation P such that PAQ = LDU. Note that the D matrix is never
    // explicitly constructed or used.
    //--------------------------------------------------------------------------

    clock_t start_factor = clock();

    OK(SLIP_LU_factorize(&L, &U, &rhos, &pinv, A, S, option));

    clock_t end_factor = clock();

    //--------------------------------------------------------------------------
    // We now solve the system Ax=b using the L and U factors computed above.
    //--------------------------------------------------------------------------

    clock_t start_solve = clock();

    // SLIP LU has an optional check step which can verify that the solution
    // vector x satisfies Ax=b in perfect precision intended for debugging.
    //
    // Note that this is entirely optional and not necessary. The solution
    // returned is guaranteed to be exact.   It appears here just as a
    // verification that SLIP LU is computing its expected result.  This test
    // can fail only if it runs out of memory, or if there is a bug in the
    // code.  Also, note that this function can be quite time consuming; thus
    // it is not recommended to be used in general.
    // 
    // To enable said check, the following bool is set to true
    //
    option->check = true; 
    
    // Solve LDU x = b
    OK(SLIP_LU_solve(&x, b,
        (const SLIP_matrix *) A,
        (const SLIP_matrix *) L,
        (const SLIP_matrix *) U,
        (const SLIP_matrix *) rhos,
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
    // SLIP_kind my_kind = SLIP_DENSE;  // SLIP_CSC, SLIP_TRIPLET or SLIP_DENSE
    // SLIP_type my_type = SLIP_FP64;   // SLIP_MPQ, SLIP_MPFR, or SLIP_FP64
    //
    // SLIP_matrix* my_x = NULL;        // New output
    // Create copy which is stored as my_kind and my_type:
    // SLIP_matrix_copy( &my_x, my_kind, my_type, x, option);

    // Timing stats
    double t_sym = (double) (end_col-start_col)/CLOCKS_PER_SEC;
    double t_factor = (double) (end_factor - start_factor) / CLOCKS_PER_SEC;
    double t_solve =  (double) (end_solve - start_solve) / CLOCKS_PER_SEC;

    printf("\nNumber of L+U nonzeros: \t\t%"PRId64,
        (L->p[L->n]) + (U->p[U->n]) - (L->m));
    printf("\nSymbolic analysis time: \t\t%lf", t_sym);
    printf("\nSLIP LU Factorization time: \t\t%lf", t_factor);
    printf("\nFB Substitution time: \t\t\t%lf\n\n", t_solve);

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    printf ("\n%s: all tests passed\n\n", __FILE__) ;
    return 0;
}

