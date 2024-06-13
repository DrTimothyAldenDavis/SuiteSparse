// ----------------------------------------------------------------------------
// SPEX/Tcov/tcov_for_cholesky.c: test coverage for SPEX_Cholesky
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

/* This program will exactly solve the sparse linear system Ax = b by performing
 * the SPEX Cholesky factorization.
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
// BRUTAL: test a method with debug malloc, until it succeeds
//------------------------------------------------------------------------------

#define NTRIAL_MAX 10000

#define BRUTAL(method)                                                      \
{                                                                           \
    int64_t trial = 0 ;                                                     \
    SPEX_info info2 = SPEX_OK ;                                             \
    for (trial = 0 ; trial <= NTRIAL_MAX ; trial++)                         \
    {                                                                       \
        malloc_count = trial ;                                              \
        info2 = (method) ;                                                  \
        if (info2 != SPEX_OUT_OF_MEMORY) break ;                            \
    }                                                                       \
    if (info2 != SPEX_OK) TEST_ABORT (info2) ;                              \
    malloc_count = INT64_MAX ;                                              \
    printf ("\nBrutal Cholesky trials %ld: tests passed\n", trial);         \
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
// spex_test_chol_backslash: test SPEX_cholesky_backslash
//------------------------------------------------------------------------------

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL                           \
{                                               \
    OK (SPEX_matrix_free (&x, option));         \
}

SPEX_info spex_test_chol_backslash (SPEX_matrix A, SPEX_matrix b,
    SPEX_options option);

SPEX_info spex_test_chol_backslash (SPEX_matrix A, SPEX_matrix b,
    SPEX_options option)
{
    SPEX_matrix x = NULL ;
    // solve Ax=b
    OK2 (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option));
    // disable memory testing when checking the solution
    int64_t save = malloc_count ; malloc_count = INT64_MAX ;
    OK (spex_demo_check_solution (A, x, b, option));
    // re-enable memory testing
    malloc_count = save ;
    SPEX_FREE_ALL;
    return (SPEX_OK) ;
}

SPEX_info spex_test_ldl_backslash (SPEX_matrix A, SPEX_matrix b,
    SPEX_options option);

SPEX_info spex_test_ldl_backslash (SPEX_matrix A, SPEX_matrix b,
    SPEX_options option)
{
    SPEX_matrix x = NULL ;
    // solve Ax=b
    OK2 (SPEX_ldl_backslash (&x, SPEX_MPQ, A, b, option));
    // disable memory testing when checking the solution
    int64_t save = malloc_count ; malloc_count = INT64_MAX ;
    OK (spex_demo_check_solution (A, x, b, option));
    // re-enable memory testing
    malloc_count = save ;
    SPEX_FREE_ALL;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// spex_test_cdiv_qr: test SPEX_cdiv_qr
//------------------------------------------------------------------------------

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL       \
{                           \
    SPEX_mpz_clear (q1);  \
    SPEX_mpz_clear (r1);  \
}

SPEX_info spex_test_cdiv_qr (mpz_t n, mpz_t d) ;

SPEX_info spex_test_cdiv_qr (mpz_t n, mpz_t d)
{
    //SPEX_info info ;
    mpz_t q1, r1;
    SPEX_mpz_set_null (q1);
    SPEX_mpz_set_null (r1);
    OK2 (SPEX_mpz_init2(q1,1));
    OK2 (SPEX_mpz_init2(r1,1));

    OK2 (SPEX_mpz_cdiv_qr(q1,r1,n,d));

    SPEX_FREE_ALL ; 
    return (SPEX_OK) ;
}

//  BRUTAL ( spex_test_cdiv_qr ()) ;


//------------------------------------------------------------------------------
// spex_test_chol_afs: test SPEX_cholesky_[analyze,factorize,solve]
//------------------------------------------------------------------------------

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL                                   \
{                                                       \
    OK (SPEX_symbolic_analysis_free (&S, option));      \
    OK (SPEX_factorization_free (&F, option));          \
    OK (SPEX_matrix_free (&x, option));                 \
}

SPEX_info spex_test_chol_afs
(
    SPEX_matrix A,
    SPEX_matrix b,
    SPEX_options option
) ;

SPEX_info spex_test_chol_afs (SPEX_matrix A, SPEX_matrix b, SPEX_options option)
{
    SPEX_symbolic_analysis S = NULL ;
    SPEX_factorization F = NULL ;
    SPEX_matrix x = NULL ;
    // solve Ax=b
    OK2 (SPEX_cholesky_analyze (&S, A, option));
    OK2 (SPEX_cholesky_factorize (&F, A, S, option));
    OK2 (SPEX_cholesky_solve (&x, F, b, option));
    // disable memory testing when checking the solution
    int64_t save = malloc_count ; malloc_count = INT64_MAX ;
    OK (spex_demo_check_solution (A, x, b, option));
    // re-enable memory testing
    malloc_count = save ;
    SPEX_FREE_ALL;
    return (SPEX_OK);
}

SPEX_info spex_test_ldl_afs
(
    SPEX_matrix A,
    SPEX_matrix b,
    SPEX_options option
) ;

SPEX_info spex_test_ldl_afs (SPEX_matrix A, SPEX_matrix b, SPEX_options option)
{
    SPEX_symbolic_analysis S = NULL ;
    SPEX_factorization F = NULL ;
    SPEX_matrix x = NULL ;
    // solve Ax=b
    OK2 (SPEX_ldl_analyze (&S, A, option));
    OK2 (SPEX_ldl_factorize (&F, A, S, option));
    OK2 (SPEX_ldl_solve (&x, F, b, option));
    // disable memory testing when checking the solution
    int64_t save = malloc_count ; malloc_count = INT64_MAX ;
    OK (spex_demo_check_solution (A, x, b, option));
    // re-enable memory testing
    malloc_count = save ;
    SPEX_FREE_ALL;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// tcov_for_cholesky: main program
//------------------------------------------------------------------------------

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL                                   \
{                                                       \
    OK (SPEX_symbolic_analysis_free (&S, option));      \
    OK (SPEX_factorization_free (&F, option));          \
    OK (SPEX_matrix_free (&x, option));                 \
    OK (SPEX_matrix_free (&A, option));                 \
    OK (SPEX_matrix_free (&b, option));                 \
    SPEX_FREE (option);                                 \
}

int main (int argc, char *argv [])
{

    //--------------------------------------------------------------------------
    // start SPEX
    //--------------------------------------------------------------------------

    SPEX_matrix A = NULL, b = NULL, x = NULL ;
    SPEX_symbolic_analysis S = NULL ;
    SPEX_factorization F = NULL, F2 = NULL ;
    SPEX_options option = NULL ;

    if (argc < 2)
    {
        printf ("usage: tcov_for_cholesky matrixfilename\n");
        TEST_ABORT (SPEX_INCORRECT_INPUT);
    }

    SPEX_info info ;
    OK (SPEX_initialize_expert (tcov_malloc, tcov_calloc, tcov_realloc,
        tcov_free));

    // disable malloc testing for the first part of the test
    spex_set_gmp_ntrials (INT64_MAX) ;
    malloc_count = INT64_MAX ;

    OK (SPEX_create_default_options (&option));

    //--------------------------------------------------------------------------
    // test a few small invalid matrices
    //--------------------------------------------------------------------------

    // unsymmetric matrix (row counts != col counts)
    printf ("Cholesky: error handling for unsymmetric matrix (1)\n");
    read_test_matrix (&A, "../ExampleMats/test1.mat.txt");
    create_test_rhs (&b, A->n);
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option), SPEX_UNSYMMETRIC);
    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));

    // unsymmetric matrix (unsymmetric pattern)
    printf ("Cholesky: error handling for unsymmetric matrix (2)\n");
    read_test_matrix (&A, "../ExampleMats/test2.mat.txt");
    create_test_rhs (&b, A->n);
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option), SPEX_UNSYMMETRIC);
    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));

    // unsymmetric matrix (unsymmetric values)
    printf ("Cholesky: error handling for unsymmetric matrix (3)\n");
    read_test_matrix (&A, "../ExampleMats/test3.mat.txt");
    create_test_rhs (&b, A->n);
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option), SPEX_UNSYMMETRIC);
    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));

    // symmetric indefinite matrix
    printf ("Cholesky: error handling for symmetric indefinite matrix (4)\n");
    read_test_matrix (&A, "../ExampleMats/test4.mat.txt");
    create_test_rhs (&b, A->n);
    option->algo = SPEX_CHOL_UP ;
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option), SPEX_NOTSPD);
    option->algo = SPEX_CHOL_LEFT ;
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option), SPEX_NOTSPD);
    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));
    
    // symmetric indefinite matrix
    printf ("Cholesky: error handling for symmetric matrix with zero in diagonal\n");
    read_test_matrix (&A, "../ExampleMats/test4.mat.txt");
    create_test_rhs (&b, A->n);
    option->algo = SPEX_LDL_UP ;
    ERR (SPEX_ldl_backslash (&x, SPEX_MPQ, A, b, option), SPEX_ZERODIAG);
    option->algo = SPEX_LDL_LEFT ;
    ERR (SPEX_ldl_backslash (&x, SPEX_MPQ, A, b, option), SPEX_ZERODIAG);
    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));

    // reset option->algo
    option->algo = SPEX_ALGORITHM_DEFAULT;

    //--------------------------------------------------------------------------
    // symetry check
    //--------------------------------------------------------------------------
    read_test_matrix (&A, "../ExampleMats/test5.mat.txt");
    SPEX_ldl_analyze( &S, A, option);
    OK (SPEX_matrix_free (&A, option));
    OK (SPEX_matrix_free (&b, option));
    
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

    //--------------------------------------------------------------------------
    // test SPEX_transpose
    //--------------------------------------------------------------------------

    printf("\n Test SPEX_transpose \n");
    SPEX_matrix A_mpq = NULL, A_mpfr = NULL, A_int = NULL, A_fp = NULL;
    SPEX_matrix T_mpq = NULL, T_mpfr = NULL, T_int = NULL, T_fp = NULL;
    // T = A'
    OK ( SPEX_matrix_copy(&A_mpq, SPEX_CSC, SPEX_MPQ, A, option));
    OK ( SPEX_transpose(&T_mpq, A_mpq, option) );

    OK ( SPEX_matrix_copy(&A_mpfr, SPEX_CSC, SPEX_MPFR, A, option));
    OK ( SPEX_transpose(&T_mpfr, A_mpfr, option) );

    OK ( SPEX_matrix_copy(&A_int, SPEX_CSC, SPEX_INT64, A, option));
    OK ( SPEX_transpose(&T_int, A_int, option) );

    OK ( SPEX_matrix_copy(&A_fp, SPEX_CSC, SPEX_FP64, A, option));
    OK ( SPEX_transpose(&T_fp, A_fp, option));

    SPEX_matrix_free(&A_mpq,option);
    SPEX_matrix_free(&A_mpfr,option);
    SPEX_matrix_free(&A_int,option);
    SPEX_matrix_free(&A_fp,option);
    SPEX_matrix_free(&T_mpq,option);
    SPEX_matrix_free(&T_mpfr,option);
    SPEX_matrix_free(&T_int,option);
    SPEX_matrix_free(&T_fp,option);
    
    //--------------------------------------------------------------------------
    // test spex_expand_mpfr_array
    //--------------------------------------------------------------------------

    //create mpfr array where all elements are multiples of 220
    mpfr_rnd_t round = SPEX_OPTION_ROUND (option);
    mpfr_t* x_mpfr = spex_create_mpfr_array (3, option);
    mpz_t* x_mpz = spex_create_mpz_array (3);
    mpq_t x_scale;
    SPEX_MPQ_INIT (x_scale) ;

    SPEX_MPQ_SET_UI (x_scale, 1, 10);
    for (int64_t k = 0 ; k < 3 ; k++)
    {
        SPEX_MPFR_SET_SI( x_mpfr[k],(k+2)*220, round);
    }

    OK ( spex_expand_mpfr_array (x_mpz, x_mpfr, x_scale, 3, option));

    // free x_mpz, x_mpfr, and x_scale
    spex_free_mpz_array (&x_mpz, 3) ;
    spex_free_mpfr_array (&x_mpfr, 3) ;
    SPEX_mpq_clear (x_scale) ;

    //--------------------------------------------------------------------------
    // missing gmp coverage
    //--------------------------------------------------------------------------
    mpz_t gmp_x, gmp_y, gmp_0;
    mpq_t gmp_a, gmp_b, gmp_c;
    mpfr_t gmp_e, gmp_f, gmp_g, gmp_h ;
    uint64_t num1=2;
    uint64_t num2=3;
    int r;

    //Initialization   
    SPEX_MPZ_INIT(gmp_x);
    SPEX_MPZ_INIT(gmp_y);
    SPEX_MPZ_INIT(gmp_0);
        
    SPEX_MPQ_INIT(gmp_a);
    SPEX_MPQ_INIT(gmp_b);
    SPEX_MPQ_INIT(gmp_c);

    SPEX_MPFR_INIT2(gmp_e, 128);
    SPEX_MPFR_INIT2(gmp_f, 128);
    SPEX_MPFR_INIT2(gmp_g, 128);
    SPEX_MPFR_INIT2(gmp_h, 128);

    //set values
    SPEX_MPZ_SET_SI(gmp_y, -4);
    SPEX_MPQ_SET_UI(gmp_b, 2, 7);
    SPEX_MPFR_SET_SI(gmp_f, 10, round);
    SPEX_MPFR_SET_SI(gmp_g, 7, round);

    SPEX_MPZ_SET_SI(gmp_0, 0);

    //Test
    FILE *fil = fopen ("../ExampleMats/test4.mat.txt", "r");
    SPEX_gmp_fscanf(fil,"c");
    fclose (fil);
    
    SPEX_MPZ_ABS(gmp_x,gmp_y);

    SPEX_MPQ_SET_NUM(gmp_c,gmp_x);

    SPEX_MPQ_ABS(gmp_a,gmp_b);

    SPEX_MPQ_CMP(&r,gmp_b,gmp_a);

    SPEX_MPFR_MUL(gmp_e,gmp_f,gmp_g,round);

    SPEX_MPFR_UI_POW_UI(gmp_h,num1,num2,round);
    
    SPEX_MPQ_NEG(gmp_a,gmp_b);

    printf("Brutal test of SPEX_cdiv_qr: \n");
    mpz_t gmp_n,gmp_d,tmpz;
    SPEX_MPZ_INIT2(gmp_n,1);
    SPEX_MPZ_INIT(gmp_d);
    SPEX_MPZ_INIT(tmpz);
    SPEX_MPZ_SET_SI(tmpz, 1); //tmpz=1
    SPEX_MPZ_SET_SI(gmp_n, INT64_MAX);
    for (int ii = 0; ii<3;ii++){
    SPEX_MPZ_MUL(gmp_n, gmp_n, gmp_n);
    }// gmp_n = INT64_MAX^8
    SPEX_MPZ_SET(gmp_d, gmp_n); //gmp_d = gmp_n = INT64_MAX^8
    SPEX_MPZ_MUL(gmp_n, gmp_n, gmp_n);
    SPEX_MPZ_SUB(gmp_n, gmp_n, tmpz);// gmp_n = (INT64_MAX^8)^2-1
    // we should get q = r = INT64_MAX^8-1
    BRUTAL(spex_test_cdiv_qr (gmp_n,gmp_d));

    ERR(SPEX_mpz_cdiv_qr(gmp_x,gmp_y,gmp_n,gmp_0),SPEX_PANIC);
    ERR(SPEX_mpz_divexact(gmp_x,gmp_y,gmp_0),SPEX_PANIC);
    
    //Free
    SPEX_mpz_clear (gmp_x);
    SPEX_mpz_clear (gmp_0);
    SPEX_mpz_clear (gmp_y);
    SPEX_mpq_clear (gmp_a);
    SPEX_mpq_clear (gmp_b);
    SPEX_mpq_clear (gmp_c);
    SPEX_mpfr_clear (gmp_e);
    SPEX_mpfr_clear (gmp_f);
    SPEX_mpfr_clear (gmp_g);
    SPEX_mpfr_clear (gmp_h);
    SPEX_mpz_clear (gmp_n);
    SPEX_mpz_clear (gmp_d);
    SPEX_mpz_clear (tmpz);
    
    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    // inputs cannot be NULL
    ERR (SPEX_matrix_nnz (NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR (SPEX_matrix_nnz (NULL, A, NULL),
        SPEX_INCORRECT_INPUT);
    ERR (SPEX_matrix_nnz (&anz, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR (SPEX_cholesky_analyze (NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR (SPEX_cholesky_backslash (NULL, SPEX_MPQ, NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR (SPEX_cholesky_factorize (NULL, NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR (SPEX_determine_symmetry (NULL, NULL, NULL ),
        SPEX_INCORRECT_INPUT);

    // type cannot be int64
    ERR (SPEX_cholesky_backslash (&x, SPEX_INT64, A, b, option),
        SPEX_INCORRECT_INPUT);

    // mangle the matrix: invalid dimensions
    A->n = 0 ;
    A->m = 0 ;
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_INPUT);
    A->n = n ;
    A->m = n ;

    // mangle the matrix: invalid type
    A->type = SPEX_INT64 ;
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_INPUT);
    bool is_symmetric ;
    ERR (SPEX_determine_symmetry (&is_symmetric, A, option),
        SPEX_INCORRECT_INPUT);
    A->type = SPEX_MPZ ;

    // valid analysis, but break the factorization
    OK (SPEX_cholesky_analyze (&S, A, option));
    A->type = SPEX_INT64 ;
    ERR (SPEX_cholesky_factorize (&F, A, S, option),
        SPEX_INCORRECT_INPUT);
    A->type = SPEX_MPZ ;
    OK (SPEX_symbolic_analysis_free (&S, option));

    // valid analysis and factorization, but break the solve
    OK (SPEX_cholesky_analyze (&S, A, option));
    OK (SPEX_cholesky_factorize (&F, A, S, option));
    b->type = SPEX_INT64 ;
    ERR (SPEX_cholesky_solve (&x, F, b, option),
        SPEX_INCORRECT_INPUT);
    b->type = SPEX_MPZ ;
    F->kind = SPEX_LDL_FACTORIZATION;
    ERR (SPEX_cholesky_solve (&x, F, b, option),
        SPEX_INCORRECT_INPUT);
    F->kind = SPEX_CHOLESKY_FACTORIZATION;
    OK (SPEX_symbolic_analysis_free (&S, option));
    OK (SPEX_factorization_free (&F, option));

    // invalid algorithm for Chol/ldl backslash
    option->algo = 99 ;
    ERR (SPEX_cholesky_backslash (&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_ALGORITHM);
    ERR (SPEX_ldl_backslash (&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_ALGORITHM);

    // invalid algorithm for Chol/ldl analyze
    ERR (SPEX_cholesky_analyze( &S, A, option),
         SPEX_INCORRECT_ALGORITHM);
    ERR (SPEX_ldl_analyze( &S, A, option),
         SPEX_INCORRECT_ALGORITHM);

    // invalid algorithm for Chol/ldl factorize
    option->algo = SPEX_ALGORITHM_DEFAULT;
    OK (SPEX_cholesky_analyze (&S, A, option));
    option->algo = 99;
    ERR (SPEX_cholesky_factorize( &F, A, S, option),
         SPEX_INCORRECT_ALGORITHM);
    OK (SPEX_symbolic_analysis_free (&S, option));

    option->algo = SPEX_ALGORITHM_DEFAULT;
    OK (SPEX_ldl_analyze (&S, A, option));
    option->algo = 99;
    ERR (SPEX_ldl_factorize( &F, A, S, option),
         SPEX_INCORRECT_ALGORITHM);
    OK (SPEX_symbolic_analysis_free (&S, option));

    ERR (SPEX_ldl_factorize (NULL, NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);

    // valid analysis, but break the factorization
    option->algo = SPEX_ALGORITHM_DEFAULT;
    OK (SPEX_ldl_analyze (&S, A, option));
    A->type = SPEX_INT64 ;
    ERR (SPEX_ldl_factorize (&F, A, S, option),
        SPEX_INCORRECT_INPUT);
    A->type = SPEX_MPZ ;
    OK (SPEX_symbolic_analysis_free (&S, option));

    // valid analysis and factorization, but break the solve
    OK (SPEX_ldl_analyze (&S, A, option));
    OK (SPEX_ldl_factorize (&F, A, S, option));
    b->type = SPEX_INT64 ;
    ERR (SPEX_ldl_solve (&x, F, b, option),
        SPEX_INCORRECT_INPUT);
    b->type = SPEX_MPZ ;
    F->kind = SPEX_CHOLESKY_FACTORIZATION;
    ERR (SPEX_ldl_solve (&x, F, b, option),
        SPEX_INCORRECT_INPUT);
    F->kind = SPEX_LDL_FACTORIZATION;
    OK (SPEX_symbolic_analysis_free (&S, option));
    OK (SPEX_factorization_free (&F, option));

    //--------------------------------------------------------------------------
    // solve Ax=b with SPEX_cholesky_backslash and check the solution
    //--------------------------------------------------------------------------

    option->order = SPEX_AMD ;
    option->algo = SPEX_CHOL_UP ;
    option->print_level = 3 ;
    printf ("Cholesky backslash, up-looking, no malloc testing:\n");
    OK (spex_test_chol_backslash (A, b, option));
    option->print_level = 0 ;

    printf ("Cholesky backslash, up-looking, no malloc testing, colamd:\n");
    option->order = SPEX_COLAMD ;
    option->print_level = 3 ;
    OK (spex_test_chol_backslash (A, b, option));
    option->order = SPEX_AMD ;
    option->print_level = 0 ;

    printf ("Cholesky backslash, no malloc testing, natural ordering:\n");
    option->order = SPEX_NO_ORDERING ;
    OK (spex_test_chol_backslash (A, b, option));

    printf ("Cholesky backslash, no malloc testing, return x as MPFR:\n");
    OK (SPEX_cholesky_backslash (&x, SPEX_MPFR, A, b, option));
    //NOTE: mpfr solution can't be checked because mpfr->mpz isn't guaranteed
    //      to be exact
    OK (SPEX_matrix_free (&x, option));

    printf ("Cholesky backslash, up-looking with malloc testing, colamd:\n");
    option->order = SPEX_COLAMD ;
    BRUTAL (spex_test_chol_backslash (A, b, option));

    printf ("Cholesky backslash, up-looking with malloc testing, amd:\n");
    option->order = SPEX_AMD ;
    BRUTAL (spex_test_chol_backslash (A, b, option));

    printf ("Cholesky backslash, up-looking with malloc testing, "
        " no ordering:\n");
    option->order = SPEX_NO_ORDERING ;
    BRUTAL (spex_test_chol_backslash (A, b, option));

    printf ("Cholesky backslash, left-looking with malloc testing:\n");
    option->algo = SPEX_CHOL_LEFT ;
    BRUTAL (spex_test_chol_backslash (A, b, option));

    //--------------------------------------------------------------------------
    // solve Ax=b with SPEX_cholesky_[analyze,factorize,solve]; check solution
    //--------------------------------------------------------------------------

    option->algo = SPEX_CHOL_UP ;

    printf ("Cholesky analyze/factorize/solve, no malloc testing:\n");
    spex_set_gmp_ntrials (INT64_MAX) ;
    malloc_count = INT64_MAX ;
    OK (spex_test_chol_afs (A, b, option));

    printf ("Cholesky analyze/factorize/solve, with malloc testing:\n");
    // also check a different RHS, with b(0) = 0
    OK (SPEX_mpz_set_ui (b->x.mpz [0], 0));
    BRUTAL (spex_test_chol_afs (A, b, option));


    //--------------------------------------------------------------------------
    // solve Ax=b with SPEX_ldl_backslash and check the solution
    //--------------------------------------------------------------------------

    option->order = SPEX_AMD ;
    option->algo = SPEX_LDL_UP ;
    option->print_level = 3 ;
    printf ("LDL backslash, up-looking, no malloc testing:\n");
    OK (spex_test_ldl_backslash (A, b, option));
    option->print_level = 0 ;

    printf ("LDL backslash, left-looking with malloc testing:\n");
    option->algo = SPEX_LDL_LEFT ;
    BRUTAL (spex_test_ldl_backslash (A, b, option));

    //--------------------------------------------------------------------------
    // solve Ax=b with SPEX_cholesky_[analyze,factorize,solve]; check solution
    //--------------------------------------------------------------------------

    option->algo = SPEX_LDL_UP ;

    printf ("LDL analyze/factorize/solve, no malloc testing:\n");
    spex_set_gmp_ntrials (INT64_MAX) ;
    malloc_count = INT64_MAX ;
    OK (spex_test_ldl_afs (A, b, option));

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    // SPEX not initialized
    spex_set_initialized (false);
    ERR (SPEX_cholesky_factorize (&F2, A, S, option), SPEX_PANIC);
    ERR (SPEX_cholesky_analyze (NULL, NULL, NULL), SPEX_PANIC);
    ERR (SPEX_cholesky_solve (NULL, NULL, NULL, NULL), SPEX_PANIC);
    ERR (SPEX_cholesky_backslash (NULL, SPEX_MPQ, NULL, NULL, NULL),
        SPEX_PANIC);

    ERR (SPEX_ldl_factorize (&F2, A, S, option), SPEX_PANIC);
    ERR (SPEX_ldl_solve (NULL, NULL, NULL, NULL), SPEX_PANIC);

    spex_set_initialized (true);
    SPEX_FREE_ALL;
    SPEX_finalize ( ) ;

    printf ("%s: all tests passed\n\n", __FILE__);
    fprintf (stderr, "%s: all tests passed\n\n", __FILE__);
    return 0;
}

