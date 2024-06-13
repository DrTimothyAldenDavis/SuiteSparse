// ----------------------------------------------------------------------------
// SPEX/Tcov/tcov_for_lu2.c: test coverage for SPEX_Cholesky
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

/* This program will cover what isn't covered by tcov_for_lu
 */

#include "tcov_utilities.h"
#include "spex_demos.h"

// test wrapper for SPEX_* function when expected error would produce
#define ERR(method,expected_error)                                      \
{                                                                       \
    SPEX_info info5 = (method) ;                                        \
    if (info5 != expected_error)                                        \
    {                                                                   \
        printf ("SPEX method was expected to fail, but succeeded!\n") ; \
        printf ("this error was expected:\n") ;                         \
        SPEX_PRINT_INFO (expected_error) ;                              \
        printf ("but this error was obtained:\n") ;                     \
        TEST_ABORT (info5) ;                                            \
    }                                                                   \
}
//------------------------------------------------------------------------------
// read_test_matrix: read in a matrix from a file
//------------------------------------------------------------------------------

void read_test_matrix (SPEX_matrix *A_handle, char *filename);

void read_test_matrix (SPEX_matrix *A_handle, char *filename)
{
    FILE *f = fopen (filename, "r");
    OK (f == NULL ? SPEX_PANIC : SPEX_OK);
    OK (spex_demo_tripread (A_handle, f, SPEX_FP64, NULL));
    fclose (f);
}

//------------------------------------------------------------------------------
// create_test_rhs: create a right-hand-side vector
//------------------------------------------------------------------------------

void create_test_rhs (SPEX_matrix *b_handle, int64_t n);

void create_test_rhs (SPEX_matrix *b_handle, int64_t n)
{
    OK (SPEX_matrix_allocate (b_handle, SPEX_DENSE, SPEX_MPZ, n, 1, n, false,
        true, NULL));
    SPEX_matrix b = *(b_handle);
    // b(0)=0
    OK (SPEX_mpz_set_ui (b->x.mpz [0], 0));
    for (int64_t k = 1 ; k < n ; k++)
    {
        // b(k) = 1
        OK (SPEX_mpz_set_ui (b->x.mpz [k], 1));
    }
}


//------------------------------------------------------------------------------
// spex_test_lu_backslash: test SPEX_lu_backslash
//------------------------------------------------------------------------------

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL                           \
{                                               \
    OK (SPEX_matrix_free (&x, option));         \
}

SPEX_info spex_test_lu_backslash (SPEX_matrix A, SPEX_matrix b,
    SPEX_options option);

SPEX_info spex_test_lu_backslash (SPEX_matrix A, SPEX_matrix b,
    SPEX_options option)
{
    SPEX_matrix x = NULL ;
    // solve Ax=b
    OK2 (SPEX_lu_backslash (&x, SPEX_MPQ, A, b, option));
    // disable memory testing when checking the solution
    int64_t save = malloc_count ; malloc_count = INT64_MAX ;
    OK (spex_demo_check_solution (A, x, b, option));
    // re-enable memory testing
    malloc_count = save ;
    SPEX_FREE_ALL;
    return (SPEX_OK) ;
}

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL                                   \
{                                                       \
    OK (SPEX_matrix_free (&A, option));                 \
    OK (SPEX_matrix_free (&b, option));                 \
    OK (SPEX_matrix_free (&x, option));                 \
    OK (SPEX_symbolic_analysis_free(&S, option))        \
}


int main (int argc, char *argv [])
{

    //--------------------------------------------------------------------------
    // start SPEX
    //--------------------------------------------------------------------------

    SPEX_matrix A = NULL, b = NULL, x = NULL ;
    SPEX_symbolic_analysis S = NULL ;
    //SPEX_factorization F = NULL, F2 = NULL ;
    SPEX_options option = NULL ;

    if (argc < 2)
    {
        printf ("usage: tcov_for_cholesky matrixfilename\n");
        TEST_ABORT (SPEX_INCORRECT_INPUT);
    }

    OK (SPEX_initialize_expert (tcov_malloc, tcov_calloc, tcov_realloc,
        tcov_free));

    // disable malloc testing for the first part of the test
    spex_set_gmp_ntrials (INT64_MAX) ;
    malloc_count = INT64_MAX ;

    OK (SPEX_create_default_options (&option));

    //--------------------------------------------------------------------------
    // load the test matrix and create the right-hand-side
    //--------------------------------------------------------------------------

    read_test_matrix (&A, argv [1]);
    int64_t n = A->n ;
    int64_t m = A->m ;
    int64_t anz = -1 ;
    OK (SPEX_matrix_nnz (&anz, A, option));
    printf ("\nInput matrix: %ld-by-%ld with %ld entries\n", n, m, anz);
    OK ((n != m) ? SPEX_PANIC : SPEX_OK);
    create_test_rhs (&b, A->n);

    //TESTS
    option->pivot = SPEX_TOL_LARGEST;
    option->order = SPEX_AMD ;
    option->print_level = 3 ;
    printf ("LU backslash, AMD ordering, no malloc testing:\n");
    OK (spex_test_lu_backslash (A, b, option));
    option->print_level = 0 ;

    option->pivot = SPEX_FIRST_NONZERO ;
    option->order = SPEX_COLAMD ;
    option->print_level = 3 ;
    printf ("LU backslash, AMD ordering, no malloc testing:\n");
    OK (spex_test_lu_backslash (A, b, option));
    option->print_level = 0 ;

    option->pivot = SPEX_TOL_SMALLEST ;
    option->tol = 0;
    option->order = SPEX_COLAMD ;
    option->print_level = 3 ;
    printf ("LU backslash, AMD ordering, no malloc testing:\n");
    OK (spex_test_lu_backslash (A, b, option));
    option->print_level = 0 ;

    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));

    option->order = SPEX_AMD ;
    read_test_matrix (&A, "../ExampleMats/test1.mat.txt");
    OK (SPEX_lu_analyze( &S, A, option));
    OK (SPEX_symbolic_analysis_free(&S, option));
    OK (SPEX_matrix_free (&A, option));

    read_test_matrix (&A, "../ExampleMats/test5.mat.txt");
    SPEX_lu_analyze( &S, A, option);
    OK (SPEX_symbolic_analysis_free (&S, option));
    OK (SPEX_matrix_free (&A, option));
    
    // Give an incorrect algorithm to SPEX Backslash
    read_test_matrix (&A, "../ExampleMats/10teams.mat.txt");
    create_test_rhs (&b, A->n);
    option->algo = 99;
    ERR( SPEX_lu_backslash(&x, SPEX_MPQ, A, b, option), SPEX_INCORRECT_ALGORITHM);
    
    // Give an incorrect algorithm to SPEX_lu_analyze
    ERR( SPEX_lu_analyze( &S, A, option), SPEX_INCORRECT_ALGORITHM);
    
    // Give an incorrect algorithm to spex lu factorize
    SPEX_factorization F;
    option->algo = SPEX_ALGORITHM_DEFAULT;
    OK( SPEX_lu_analyze( &S, A, option));
    option->algo = 99;
    ERR( SPEX_lu_factorize( &F, A, S, option), SPEX_INCORRECT_ALGORITHM);
    OK (SPEX_symbolic_analysis_free (&S, option));

    SPEX_FREE_ALL;
    OK (SPEX_finalize ( )) ;
    SPEX_FREE (option) ;

    printf ("%s: all tests passed\n\n", __FILE__);
    fprintf (stderr, "%s: all tests passed\n\n", __FILE__);
    return (0) ;
}
