//------------------------------------------------------------------------------
// Demo/spex_demo_cholesky_extended: example of extended call of SPEX_Cholesky
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


#include "spex_demos.h"

#define FREE_WORKSPACE                          \
{                                               \
    SPEX_matrix_free(&A,NULL);                  \
    SPEX_matrix_free(&b,NULL);                  \
    SPEX_matrix_free(&x,NULL);                  \
    SPEX_symbolic_analysis_free (&S, option);   \
    SPEX_factorization_free(&F, option);        \
    SPEX_FREE(option);                          \
    SPEX_finalize();                            \
}

int main( int argc, char *argv[] )
{

    //--------------------------------------------------------------------------
    // Prior to using SPEX-Chol, its environment must be initialized. This is
    // done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    DEMO_INIT (ok) ;

    //--------------------------------------------------------------------------
    // Declare memory & Process Command Line
    //--------------------------------------------------------------------------
    int64_t n = 0 ;

    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL ;
    SPEX_matrix A = NULL;
    SPEX_matrix b = NULL;
    SPEX_matrix x = NULL;
   
    char *mat_name, *rhs_name;
    int64_t rat = 1;

    // Default options.
    SPEX_options option = NULL;
    DEMO_OK(SPEX_create_default_options(&option));

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

    DEMO_OK(spex_demo_tripread(&A, mat_file, SPEX_FP64, option));
    fclose(mat_file);
    n = A->n;
    // For this code, we utilize a vector of all ones as the RHS vector
    SPEX_matrix_allocate(&b, SPEX_DENSE, SPEX_MPZ, n, 1, n, false, true,
        option);
    
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
    // Perform Analysis of A
    //--------------------------------------------------------------------------

    double start_col = SuiteSparse_time ();

    // Symmetric ordering of A. Uncomment the desired one, AMD is recommended
    //option->order = SPEX_NO_ORDERING;  // No ordering
    option->order = SPEX_AMD;  // AMD
    //option->order = SPEX_COLAMD; // COLAMD
    DEMO_OK(SPEX_cholesky_analyze(&S, A, option));

    double end_col = SuiteSparse_time ();

    //--------------------------------------------------------------------------
    // Factorize PAP
    //--------------------------------------------------------------------------

    //option->algo=SPEX_CHOL_LEFT;
    double start_factor = SuiteSparse_time ();

    DEMO_OK( SPEX_cholesky_factorize(&F, A, S, option));

    double end_factor = SuiteSparse_time ();

    option->print_level=3;
    //DEMO_OK(SPEX_matrix_check(F->L,option));

    //--------------------------------------------------------------------------
    // Solve linear system
    //--------------------------------------------------------------------------

    double start_solve = SuiteSparse_time ();

    DEMO_OK( SPEX_cholesky_solve(&x, F, b, option));

    double end_solve = SuiteSparse_time ();

    //--------------------------------------------------------------------------
    // Output & Timing Stats
    //--------------------------------------------------------------------------

    double t_col = (end_col-start_col) ;
    double t_factor = (end_factor - start_factor) ;
    double t_solve =  (end_solve - start_solve) ;

    printf("\nNumber of L nonzeros: \t\t\t%ld",
        (F->L->p[F->L->n]) );
    printf("\nSymbolic Analysis Check time: \t\t%lf", t_col);
    printf("\nIP Chol Factorization time: \t\t%lf", t_factor);
    printf("\nFB Substitution time: \t\t\t%lf\n\n", t_solve);

    // Check solution
    option->print_level=1;
    //DEMO_OK ( SPEX_check_solution(A,x,b,option));

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    return (0) ;
}

