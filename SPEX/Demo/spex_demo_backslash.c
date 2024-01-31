//------------------------------------------------------------------------------
// Demo/spex_demo_backslash: example of SPEX_Blackslash
//------------------------------------------------------------------------------

// SPEX: (c) 2021-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


/* A demo of SPEX_Backslash in C
 */
 
# include "spex_demos.h"

#define FREE_WORKSPACE                  \
{                                       \
    SPEX_matrix_free(&A,NULL);          \
    SPEX_matrix_free(&b,NULL);          \
    SPEX_matrix_free(&x,NULL);          \
    SPEX_matrix_free(&x2, NULL);        \
    SPEX_FREE(option);                  \
    SPEX_finalize();                    \
}                                       \


int main( int argc, char *argv[] )
{

    //--------------------------------------------------------------------------
    // Prior to using SPEX, its environment must be initialized. This is done
    // by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    DEMO_INIT (ok) ;

    //--------------------------------------------------------------------------
    // Declare memory & Process Command Line
    //--------------------------------------------------------------------------
    int64_t n = 0 ;

    SPEX_matrix A = NULL;
    SPEX_matrix b = NULL;
    SPEX_matrix x = NULL;
    SPEX_matrix x2 = NULL;

    // Set default options
    SPEX_options option = NULL;
    DEMO_OK(SPEX_create_default_options(&option));

    char *mat_name = NULL, *rhs_name = NULL;
    int64_t rat = 1;

    // Process the command line
    DEMO_OK(spex_demo_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // Allocate memory
    //--------------------------------------------------------------------------

    // Read in A
    FILE *mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return (1) ;
    }

    // Note, there are a few matrices in BasisLIB that dont fit in double
    // Need to use the other tripread for those.
    DEMO_OK(spex_demo_tripread(&A, mat_file, SPEX_MPZ, option));
    fclose(mat_file);
    n = A->n;

    // Read in b. The output of this demo function is b in dense format with
    // mpz_t entries
    FILE *rhs_file = fopen(rhs_name,"r");
    if( rhs_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return (1) ;
    }
    DEMO_OK(spex_demo_read_dense(&b, rhs_file, option));
    fclose(rhs_file);

    //--------------------------------------------------------------------------
    // Solve Ax = b
    //--------------------------------------------------------------------------

    printf("solving Ax=b ...\n");
    fflush (stdout);
    fflush (stderr);
    double start = SuiteSparse_time ();

    option->print_level = 0;

    DEMO_OK( SPEX_backslash(&x, SPEX_MPQ, A, b, option));

    double end = SuiteSparse_time ();

    double t_tot = (end - start) ;

    printf("\nSPEX Backslash Factor & Solve time: %lf\n", t_tot);

    option->print_level=1;
    DEMO_OK( spex_demo_check_solution(A,x,b,option));


    // x2 is a copy of the solution. x2 is a dense matrix with double entries
    // Note: roundoff will have occured in converting the exact solution
    // to the double x.
    DEMO_OK ( SPEX_matrix_copy(&x2, SPEX_DENSE, SPEX_FP64, x, option));

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------
    FREE_WORKSPACE;
    return (0) ;
}

