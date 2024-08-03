//------------------------------------------------------------------------------
// Demo/spex_demo_lu_simple2.c: example of simple SPEX_LU call for triplet format
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// This example shows how to use SPEX Left LU within your code.  Unlike
// example1, the input matrix here is directly read in from the triplet
// format. Also, differs from example1 in that the output solution is given in
// mpq_t precision

// usage:
// spex_demo_lu_simple2 f mat_file rhs_file > out
// mat_file is the Matrix Market file containing the A matrix
// rhs_file is a list of entries for right hand side dense matrix
// if input file names are not specified, they are defaulted to
// ../ExampleMats/10teams.mat and ../ExampleMats/10teams.v, respectively.
// out is the file for the output with the calculated result

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
    SPEX_symbolic_analysis_free(&S, option);    \
    SPEX_matrix_free(&A, option);               \
    SPEX_FREE(option);                          \
    SPEX_matrix_free(&b, option);               \
    SPEX_matrix_free(&x, option);               \
    SPEX_finalize();                            \
}

int main (int argc, char **argv)
{

    SPEX_matrix A = NULL ;                     // input matrix
    SPEX_matrix b = NULL ;                     // Right hand side vector
    SPEX_matrix x = NULL ;                     // Solution vectors
    SPEX_symbolic_analysis S = NULL ;          // Column permutation
    SPEX_options option = NULL;
    FILE *mat_file = NULL;
    FILE *rhs_file = NULL;
    char *mat_name = NULL, *rhs_name = NULL;

    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This is
    // done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    SPEX_TRY (SPEX_initialize ( )) ;

    //--------------------------------------------------------------------------
    // Get matrix and right hand side file names
    //--------------------------------------------------------------------------

    if (argc < 3)
    { 
        perror ("usage: spex_demo_lu_simple2 matfile rhsfile");
        return (1) ;
    }
    mat_name = argv[1];
    rhs_name = argv[2];

    //--------------------------------------------------------------------------
    // Get default options
    //--------------------------------------------------------------------------

    SPEX_TRY (SPEX_create_default_options(&option));

    //--------------------------------------------------------------------------
    // Allocate memory, read in A and b
    //--------------------------------------------------------------------------

    // Read in A. The output of this demo function is A in CSC format with
    // mpz_t entries.
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

    // Read in b. The output of this demo function is b in dense format with
    // mpz_t entries
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
        printf("%"PRId64" %"PRId64" \n", A->m,b->m);
        fprintf (stderr, "Error! Size of A and b do not match!\n");
        FREE_WORKSPACE;
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // solve
    //--------------------------------------------------------------------------

    double start_s = SUITESPARSE_TIME;

    // Solve the system and give MPQ solution
    // Will utilize default COLAMD ordering and small pivoting
    SPEX_TRY (SPEX_lu_backslash( &x, SPEX_MPQ, A, b, option));

    double end_s = SUITESPARSE_TIME;

    double t_s = (end_s - start_s) ;

    printf("\nSPEX LU Factor & Solve time: %lf\n", t_s);

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    printf ("\n%s: all tests passed\n\n", __FILE__);
    return (0) ;
}

