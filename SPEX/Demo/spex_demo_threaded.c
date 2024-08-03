//------------------------------------------------------------------------------
// Demo/spex_demo_threaded: example of SPEX_backslash with multiple threads
//------------------------------------------------------------------------------

// SPEX: (c) 2021-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// A demo of SPEX_backslash in C: solving the same system with many different
// user threads, just to test user multithreading.

# include "spex_demos.h"

#define FREE_WORKSPACE                  \
{                                       \
    if (mat_file != NULL)               \
    {                                   \
        fclose(mat_file);               \
    }                                   \
    mat_file = NULL ;                   \
    if (rhs_file != NULL)               \
    {                                   \
        fclose(rhs_file);               \
    }                                   \
    rhs_file = NULL ;                   \
    SPEX_matrix_free(&A,NULL);          \
    SPEX_matrix_free(&b,NULL);          \
    SPEX_FREE(option);                  \
    SPEX_finalize();                    \
}                                       \

#ifdef _OPENMP
#include <omp.h>
#endif

int main( int argc, char *argv[] )
{

    int64_t n = 0 ;
    SPEX_matrix A = NULL;
    SPEX_matrix b = NULL;
    SPEX_options option = NULL;
    FILE *mat_file = NULL ;
    FILE *rhs_file = NULL;
    char *mat_name = NULL, *rhs_name = NULL;
    int64_t rat = 1;

    //--------------------------------------------------------------------------
    // Prior to using SPEX, its environment must be initialized. This is done
    // by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    #ifdef _OPENMP
    printf ("spex_demo_threaded: with OpenMP\n") ;
    #else
    printf ("spex_demo_threaded: without OpenMP\n") ;
    #endif

    SPEX_TRY (SPEX_initialize ( )) ;

    //--------------------------------------------------------------------------
    // Declare memory & Process Command Line
    //--------------------------------------------------------------------------

    // Set default options
    SPEX_TRY (SPEX_create_default_options(&option));

    // Process the command line
    SPEX_TRY (spex_demo_process_command_line(argc, argv, option,
        &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // Allocate memory
    //--------------------------------------------------------------------------

    // Read in A
    mat_file = fopen(mat_name,"r");
    if( mat_file == NULL )
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return (1) ;
    }

    // Read in the matrix, assuming that it is stored in MPZ format.
    SPEX_TRY (spex_demo_tripread(&A, mat_file, SPEX_MPZ, option));
    fclose(mat_file);
    mat_file = NULL ;

    n = A->n;

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

    //--------------------------------------------------------------------------
    // Solve Ax = b
    //--------------------------------------------------------------------------

    // This demo solves the same system many times, which isn't very useful,
    // since each thread should compute the same solution each time.  However,
    // it serves as a useful test for the thread safety feature of SPEX.  If
    // there is a race condition, then it's likely that one thread will fail
    // to properly solve one of its systems.

    fflush (stdout);
    fflush (stderr);

    option->print_level = 0;

    int nthreads = 1 ;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads ( ) ;
    #endif

    #define NTRIALS 10

    printf("solving Ax=b with nthreads: %d, with %d trials per thread\n"
        "Please wait...\n", nthreads, NTRIALS);

    bool test_pass = true ;

    int id ;
    #pragma omp parallel for num_threads(nthreads) schedule(static,1) \
        reduction(&&:test_pass)
    for (id = 0 ; id < nthreads ; id++)
    {
        SPEX_info info = SPEX_thread_initialize ( ) ;
        if (info != SPEX_OK)
        {
            printf ("SPEX thread %d: failed to initialize its context\n", id) ;
            test_pass = false ;
            SPEX_thread_finalize ( ) ;
            continue ;
        }

        // this thread solves the same system many times, just to
        // hammer the thread-safety aspect of SPEX
        bool my_test_pass = true ;
        for (int ntrials = 0 ; ntrials < NTRIALS ; ntrials++)
        {
            SPEX_matrix myx = NULL ;
            info = SPEX_backslash (&myx, SPEX_MPQ, A, b, option) ;
            if (info != SPEX_OK)
            {
                printf ("SPEX thread %d: backslash failed\n", id) ;
                my_test_pass = false ;
                test_pass = false ;
                break ;
            }
            info = spex_demo_check_solution (A,myx,b,option) ;
            SPEX_matrix_free (&myx, NULL) ;
            if (info != SPEX_OK)
            {
                printf ("SPEX thread %d: wrong solution\n", id) ;
                my_test_pass = false ;
                test_pass = false ;
                break ;
            }
        }

        info = SPEX_thread_finalize ( ) ;
        if (info != SPEX_OK)
        {
            printf ("SPEX thread %d: failed to finalize its context\n", id) ;
            my_test_pass = false ;
            test_pass = false ;
        }

        if (my_test_pass)
        {
            printf ("SPEX thread %d: ok.  All systems solved exactly.\n", id) ;
        }
    }

    if (!test_pass)
    {
        printf ("SPEX thread test failed\n\n") ;
    }
    else
    {
        printf ("SPEX thread test passed\n\n") ;
    }

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    return (test_pass ? 0 : 1) ;
}

