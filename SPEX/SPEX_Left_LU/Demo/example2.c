//------------------------------------------------------------------------------
// SPEX_Left_LU/Demo/example2.c: example main program for SPEX_Left_LU
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "demos.h"

// This example shows how to use SPEX Left LU within your code 
// Unlike example1, the input matrix here is directly read in from the 
// triplet formmat. Also, differs from example1 in that the output solution
// is given in mpq_t precision

// usage:
// example2 mat_file rhs_file > out
// mat_file is the Matrix Market file containing the A matrix
// rhs_file is a list of entries for right hand side dense matrix
// if input file names are not specified, they are defaulted to
// ../ExampleMats/10teams.mat and ../ExampleMats/10teams.v, respectively.
// out is file for output calculated result

#define FREE_WORKSPACE              \
    SPEX_LU_analysis_free(&S, option);\
    SPEX_matrix_free(&A, option);   \
    SPEX_FREE(option);              \
    SPEX_matrix_free(&b, option);   \
    SPEX_matrix_free(&x, option);   \
    SPEX_finalize();

int main (int argc, char **argv)
{
    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This is
    // done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------
    SPEX_initialize();

    //--------------------------------------------------------------------------
    // Get matrix and right hand side file names
    //--------------------------------------------------------------------------
    char *mat_name, *rhs_name;
    mat_name = "../ExampleMats/10teams_mat.txt";
    rhs_name = "../ExampleMats/10teams_v.txt";
    if (argc > 2)
    {
        mat_name = argv[1];
        rhs_name = argv[2];
    }

    //--------------------------------------------------------------------------
    // Declare our data structures
    //--------------------------------------------------------------------------
    SPEX_info ok;
    SPEX_matrix *A = NULL ;                     // input matrix
    SPEX_matrix *b = NULL ;                     // Right hand side vector
    SPEX_matrix *x = NULL ;                     // Solution vectors
    SPEX_LU_analysis *S = NULL ;                // Column permutation
    SPEX_options *option = NULL;
    OK(SPEX_create_default_options(&option));

    //--------------------------------------------------------------------------
    // Allocate memory, read in A and b
    //--------------------------------------------------------------------------

    // Read in A. The output of this demo function is A in CSC format with
    // mpz_t entries.
    FILE* mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    OK(SPEX_tripread(&A, mat_file, option));
    fclose(mat_file);

    // Read in b. The output of this demo function is b in dense format with
    // mpz_t entries
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
        printf("%"PRId64" %"PRId64" \n", A->m,b->m);
        fprintf (stderr, "Error! Size of A and b do not match!\n");
        FREE_WORKSPACE;
        return 0;
    }

    //--------------------------------------------------------------------------
    // solve
    //--------------------------------------------------------------------------

    clock_t start_s = clock();
    
    // SPEX Left LU has an optional check, to enable it, one can set the following
    // parameter to be true.
    option->check = true;
   
    // Solve the system and give MPQ solution
    OK(SPEX_Left_LU_backslash( &x, SPEX_MPQ, A, b, option));
    
    clock_t end_s = clock();

    double t_s = (double) (end_s - start_s) / CLOCKS_PER_SEC;

    printf("\nSPEX Left LU Factor & Solve time: %lf\n", t_s);

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;

    printf ("\n%s: all tests passed\n\n", __FILE__) ;
    return 0;
}

