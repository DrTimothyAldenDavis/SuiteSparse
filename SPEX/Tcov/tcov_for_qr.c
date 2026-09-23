// ----------------------------------------------------------------------------
// SPEX/Tcov/tcov_for_qr.c: test coverage for SPEX_QR
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2023, Chris Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

/* Test coverage for QR and some additional routines introduced with SPEX QR
 */

#include "tcov_utilities.h"
#include "spex_demos.h"
#include "../SPEX_QR/Source/spex_qr_internal.h"

// test wrapper for SPEX_* function when expected error would produce
#define ERR(method, expected_error)                                       \
    {                                                                     \
        SPEX_info info5 = (method);                                       \
        if (info5 != expected_error)                                      \
        {                                                                 \
            printf("SPEX method was expected to fail, but succeeded!\n"); \
            printf("this error was expected:\n");                         \
            SPEX_PRINT_INFO(expected_error);                              \
            printf("but this error was obtained:\n");                     \
            TEST_ABORT(info5);                                            \
        }                                                                 \
    }

//------------------------------------------------------------------------------
// BRUTAL: test a method with debug malloc, until it succeeds
//------------------------------------------------------------------------------

// The method must return a bool (true if successful, false if failure).

#define NTRIAL_MAX 100000 // needs to be at least 149362 (this takes an hour)

#define BRUTAL(method)                                           \
    {                                                            \
        int64_t trial = 0;                                       \
        SPEX_info info2 = SPEX_OUT_OF_MEMORY;                    \
        while (info2 != SPEX_OK)                                 \
        {                                                        \
            trial++;                                             \
            malloc_count = trial;                                \
            info2 = (method);                                    \
            if (info2 != SPEX_OUT_OF_MEMORY)                     \
                break;                                           \
        }                                                        \
        if (info2 != SPEX_OK)                                    \
            TEST_ABORT(info2);                                   \
        malloc_count = INT64_MAX;                               \
        printf("\nBrutal QR trials %ld: tests passed\n", trial); \
    }

//------------------------------------------------------------------------------
// read_test_matrix: read in a matrix from a file
//------------------------------------------------------------------------------

void read_test_matrix(SPEX_matrix *A_handle, char *filename);

void read_test_matrix(SPEX_matrix *A_handle, char *filename)
{
    FILE *f = fopen(filename, "r");
    OK(f == NULL ? SPEX_PANIC : SPEX_OK);
    OK(spex_demo_tripread(A_handle, f, SPEX_FP64, NULL));
    fclose(f);
}

// Helper to instantly build a matrix from a string in memory
void generate_test_matrix(SPEX_matrix *A_handle, const char *triplets)
{
    // tmpfile() creates a temporary file in RAM that auto-deletes when closed
    FILE *f = tmpfile();
    fprintf(f, "%s", triplets);
    rewind(f); // send the reader back to the beginning of the file

    // Read it using your existing tripread function
    OK(spex_demo_tripread(A_handle, f, SPEX_FP64, NULL));
    fclose(f);
}

//------------------------------------------------------------------------------
// create_test_rhs: create a right-hand-side vector
//------------------------------------------------------------------------------

void create_test_rhs(SPEX_matrix *b_handle, int64_t n);

void create_test_rhs(SPEX_matrix *b_handle, int64_t n)
{
    OK(SPEX_matrix_allocate(b_handle, SPEX_DENSE, SPEX_MPZ, n, 1, n, false,
                            true, NULL));
    SPEX_matrix b = *(b_handle);
    // b(0)=0
    OK(SPEX_mpz_set_ui(b->x.mpz[0], 0));
    for (int64_t k = 1; k < n; k++)
    {
        // b(k) = 1
        OK(SPEX_mpz_set_ui(b->x.mpz[k], 1));
    }
}

//------------------------------------------------------------------------------
// spex_test_qr_backslash: test SPEX_qr_backslash
//------------------------------------------------------------------------------

#undef SPEX_FREE_ALL
#define SPEX_FREE_ALL                     \
    {                                     \
        OK(SPEX_matrix_free(&x, option)); \
    }

SPEX_info spex_test_qr_backslash(SPEX_matrix A, SPEX_matrix b, SPEX_options option);

SPEX_info spex_test_qr_backslash(SPEX_matrix A, SPEX_matrix b, SPEX_options option)
{
    SPEX_matrix x = NULL;
    SPEX_info info;

    // solve Ax=b
    info = SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option);
    if (info != SPEX_OK) { SPEX_FREE_ALL; return info; } // Leak-proof return!

    int64_t save = malloc_count;
    malloc_count = INT64_MAX;
    malloc_count = save;
    SPEX_FREE_ALL;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// spex_test_qr_afs: test SPEX_qr_[analyze,factorize,solve]
//------------------------------------------------------------------------------

#undef SPEX_FREE_ALL
#define SPEX_FREE_ALL                                \
    {                                                \
        OK(SPEX_symbolic_analysis_free(&S, option)); \
        OK(SPEX_factorization_free(&F, option));     \
        OK(SPEX_matrix_free(&x, option));            \
    }

SPEX_info spex_test_qr_afs(SPEX_matrix A, SPEX_matrix b, SPEX_options option);

SPEX_info spex_test_qr_afs(SPEX_matrix A, SPEX_matrix b, SPEX_options option)
{
    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL;
    SPEX_matrix x = NULL;
    SPEX_info info;

    // solve Ax=b with leak-proof manual checks
    info = SPEX_qr_analyze(&S, A, option);
    if (info != SPEX_OK) { SPEX_FREE_ALL; return info; }

    info = SPEX_qr_factorize(&F, A, S, option);
    if (info != SPEX_OK) { SPEX_FREE_ALL; return info; }

    info = SPEX_qr_solve(&x, F, b, option);
    if (info != SPEX_OK) { SPEX_FREE_ALL; return info; }

    int64_t save = malloc_count;
    malloc_count = INT64_MAX;
    malloc_count = save;
    SPEX_FREE_ALL;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// tcov_for_qr: main program
//------------------------------------------------------------------------------

#undef SPEX_FREE_ALL
#define SPEX_FREE_ALL                                \
    {                                                \
        OK(SPEX_symbolic_analysis_free(&S, option)); \
        OK(SPEX_factorization_free(&F, option));     \
        OK(SPEX_matrix_free(&x, option));            \
        OK(SPEX_matrix_free(&A, option));            \
        OK(SPEX_matrix_free(&b, option));            \
        SPEX_FREE(option);                           \
    }

int main(int argc, char *argv[])
{

    //--------------------------------------------------------------------------
    // start SPEX
    //--------------------------------------------------------------------------

    SPEX_matrix A = NULL, b = NULL, x = NULL;
    SPEX_symbolic_analysis S = NULL;
    SPEX_factorization F = NULL, F2 = NULL;
    SPEX_options option = NULL;

    if (argc < 2)
    {
        printf("usage: tcov_for_qr matrixfilename\n");
        TEST_ABORT(SPEX_INCORRECT_INPUT);
    }

    SPEX_info info;
    OK(SPEX_initialize_expert(tcov_malloc, tcov_calloc, tcov_realloc,
                              tcov_free));

    // disable malloc testing for the first part of the test
    spex_set_gmp_ntrials(INT64_MAX);
    malloc_count = INT64_MAX;

    OK(SPEX_create_default_options(&option));

    //--------------------------------------------------------------------------
    // load the test matrix and create the right-hand-side
    //--------------------------------------------------------------------------

    read_test_matrix(&A, argv[1]);
    int64_t n = A->n;
    int64_t m = A->m;
    int64_t anz = -1;
    OK(SPEX_matrix_nnz(&anz, A, option));
    printf("\nInput matrix: %lld-by-%lld with %lld entries\n", n, m, anz);
    OK((n != m) ? SPEX_PANIC : SPEX_OK);
    create_test_rhs(&b, A->m);
    option->algo = SPEX_QR_GS;

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    // inputs cannot be NULL
    ERR(SPEX_matrix_nnz(NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR(SPEX_matrix_nnz(NULL, A, NULL),
        SPEX_INCORRECT_INPUT);
    ERR(SPEX_matrix_nnz(&anz, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR(SPEX_qr_analyze(NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR(SPEX_qr_backslash(NULL, SPEX_MPQ, NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);
    ERR(SPEX_qr_factorize(NULL, NULL, NULL, NULL),
        SPEX_INCORRECT_INPUT);

    // type cannot be int64
    ERR(SPEX_qr_backslash(&x, SPEX_INT64, A, b, option),
        SPEX_INCORRECT_INPUT);

    // mangle the matrix: invalid dimensions
    A->n = 0;
    A->m = 0;
    ERR(SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_INPUT);
    ERR(SPEX_qr_analyze(&S, A, option),
        SPEX_INCORRECT_INPUT);
    A->n = n;
    A->m = m;

    // mangle the matrix: invalid type
    A->type = SPEX_INT64;
    ERR(SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_INPUT);

    // valid analysis, but break the factorization
    OK(SPEX_qr_analyze(&S, A, option));
    A->type = SPEX_INT64;
    ERR(SPEX_qr_factorize(&F, A, S, option),
        SPEX_INCORRECT_INPUT);
    A->type = SPEX_MPZ;
    OK(SPEX_symbolic_analysis_free(&S, option));

    // valid analysis and factorization, but break the solve
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));
    b->type = SPEX_INT64;
    ERR(SPEX_qr_solve(&x, F, b, option),
        SPEX_INCORRECT_INPUT);
    b->type = SPEX_MPZ;
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    // invalid algorithm
    option->algo = 99;
    ERR(SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option),
        SPEX_INCORRECT_ALGORITHM);

    OK(SPEX_qr_analyze(&S, A, option));
    ERR(SPEX_qr_factorize(&F, A, S, option),
        SPEX_INCORRECT_ALGORITHM);
    option->algo = SPEX_QR_GS;
    OK(SPEX_symbolic_analysis_free(&S, option));

    //--------------------------------------------------------------------------
    // solve Ax=b with SPEX_qr_backslash and check the solution
    //--------------------------------------------------------------------------

    option->order = SPEX_COLAMD;
    option->print_level = 3;
    printf("QR backslash, no malloc testing:\n");
    OK(spex_test_qr_backslash(A, b, option));
    option->print_level = 0;

    printf("QR backslash, no malloc testing, amd:\n");
    option->order = SPEX_AMD;
    option->print_level = 3;
    OK(spex_test_qr_backslash(A, b, option));
    option->order = SPEX_AMD;
    option->print_level = 0;

    printf("QR backslash, no malloc testing, natural ordering:\n");
    option->order = SPEX_NO_ORDERING;
    OK(spex_test_qr_backslash(A, b, option));

    printf("QR backslash, no malloc testing, return x as MPFR:\n");
    OK(SPEX_qr_backslash(&x, SPEX_MPFR, A, b, option));
    // NOTE: mpfr solution can't be checked because mpfr->mpz isn't guaranteed
    //       to be exact

    OK(SPEX_matrix_free(&x, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));

    /// symmetric input
    read_test_matrix(&A, "../ExampleMats/mesh1e1.mat.txt");
    create_test_rhs(&b, A->n);
    option->algo = SPEX_QR_GS;
    OK(SPEX_qr_backslash(&x, SPEX_MPFR, A, b, option));

    OK(SPEX_matrix_free(&x, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));

    // Transpose Forward Sub History Update (hx > -1 trap)
    // 3x4 Wide Matrix. Col 0 and Col 2 overlap, but Col 1 and 2 are orthogonal.
    // This forces R[1,2] to be a numerical zero, but R[0,2] to be nonzero,
    // leaving hx = 0 when i = 2.
    const char *hx_test_str =
        "3 4 4\n"
        "1 1 1\n"
        "2 2 1\n"
        "3 1 1\n"
        "3 3 1\n";

    generate_test_matrix(&A, hx_test_str);
    create_test_rhs(&b, A->m);

    // We MUST use NO_ORDERING to ensure the columns stay in this exact order
    option->order = SPEX_NO_ORDERING;

    OK(SPEX_qr_backslash(&x, SPEX_MPFR, A, b, option));

    option->order = SPEX_DEFAULT_ORDERING; // Reset ordering
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));
    OK(SPEX_matrix_free(&x, option));

    //--------------------------------------------------------------------------
    // rank deficient & special cases
    //--------------------------------------------------------------------------

    // 1. Structurally Rank Deficient (Empty Column)
    // 3x3 matrix, but column 1 is completely empty.
    const char *srd_str =
        "3 3 2\n"
        "1 1 1\n"
        "3 3 1\n";
    generate_test_matrix(&A, srd_str);
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_analyze(&S, A, option));
    BRUTAL(SPEX_qr_factorize(&F, A, S, option));

    // Test SPEX_qr_rank since we are here!
    int64_t qr_rank;
    OK(SPEX_qr_rank(&qr_rank, A, option));

    // Mangle A and trigger an error for qr_rank
    A->n = 0;
    ERR(SPEX_qr_rank(&qr_rank, A, option),SPEX_INCORRECT_INPUT);
    A->n = 3;

    // Give an incorrect algorithm for qr_rank
    option->algo = SPEX_LU_LEFT;
    ERR(SPEX_qr_rank(&qr_rank, A, option), SPEX_INCORRECT_ALGORITHM);
    option->algo = SPEX_ALGORITHM_DEFAULT;

    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    // 2. Numerical zero in IPGS (Linearly dependent columns)
    // Col 2 is exactly Col 0 + Col 1, which causes Gram-Schmidt to hit a numerical zero
    const char *ipgs_zero_str =
        "3 3 4\n"
        "1 1 1\n"
        "2 2 1\n"
        "1 3 1\n"
        "2 3 1\n";
    generate_test_matrix(&A, ipgs_zero_str);
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));

    // Test the same thing with a right hand side so we
    // can hit the rank deficient case in solve
    create_test_rhs(&b, A->m);
    OK(SPEX_qr_backslash(&x, SPEX_MPFR, A, b, option));

    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&x, option));
    OK(SPEX_matrix_free(&b, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    // 3. Wide Matrix Backslash (m < n)
    // 2 rows, 3 columns
    const char *wide_str =
        "2 3 3\n"
        "1 1 1\n"
        "2 2 1\n"
        "1 3 1\n";
    generate_test_matrix(&A, wide_str);
    create_test_rhs(&b, A->m);
    option->order = SPEX_NO_ORDERING;
    BRUTAL(spex_test_qr_backslash(A, b, option));

    // test rank on wide matrices while we are here
    OK(SPEX_qr_rank(&qr_rank, A, option));

    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));
    OK(SPEX_matrix_free(&x, option));

    // 4. Transpose Solver
    // 3x3 square matrix, testing A^T x = b
    const char *sq_str =
        "3 3 3\n"
        "1 1 1\n"
        "2 2 1\n"
        "3 3 1\n";
    generate_test_matrix(&A, sq_str);
    create_test_rhs(&b, A->n);
    BRUTAL(spex_test_qr_backslash(A, b, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));
    OK(SPEX_matrix_free(&x, option));

    read_test_matrix(&A, "../ExampleMats/srd_test1.mat.txt");
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    read_test_matrix(&A, "../ExampleMats/srd_test2.mat.txt");
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    read_test_matrix(&A, "../ExampleMats/srd_test3.mat.txt");
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    read_test_matrix(&A, "../ExampleMats/srd_test4.mat.txt");
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    // checks for numerical zero in ipgs
    read_test_matrix(&A, "../ExampleMats/test7.mat.txt");
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_qr_factorize(&F, A, S, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_factorization_free(&F, option));

    // checks for numerical zero in back sub
    read_test_matrix(&A, "../ExampleMats/test8.mat.txt");
    create_test_rhs(&b, A->n);
    option->order = SPEX_NO_ORDERING;
    OK(SPEX_qr_backslash(&x, SPEX_MPFR, A, b, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));
    OK(SPEX_matrix_free(&x, option));

    //--------------------------------------------------------------------------
    // solve Ax=b with SPEX_qr_[analyze,factorize,solve]; check solution
    //--------------------------------------------------------------------------

    read_test_matrix(&A, "../ExampleMats/LF10.mat.txt");
    create_test_rhs(&b, A->m);
    option->algo = SPEX_QR_GS;
    printf("QR analyze/factorize/solve, no malloc testing:\n");
    spex_set_gmp_ntrials(INT64_MAX);
    malloc_count = INT64_MAX;
    OK(spex_test_qr_afs(A, b, option));

    printf("QR analyze/factorize/solve, with malloc testing:\n");
    // also check a different RHS, with b(n-1) = 0
    OK(SPEX_mpz_set_ui(b->x.mpz[A->n - 1], 0));
    OK(spex_test_qr_afs(A, b, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));

    // Brutal test of the transpose solver
    // 8x10 Wide Matrix (8 rows, 10 columns, 31 non-zeros)
    const char *wide_8x10_dense_str =
        "8 10 31\n"
        "1 1 1\n"
        "1 5 2\n"
        "1 9 1\n"
        "2 2 1\n"
        "2 3 3\n"
        "2 8 1\n"
        "2 10 2\n"
        "3 1 1\n"
        "3 3 1\n"
        "3 4 2\n"
        "3 9 1\n"
        "4 2 2\n"
        "4 4 1\n"
        "4 7 1\n"
        "4 10 3\n"
        "5 3 1\n"
        "5 5 1\n"
        "5 6 2\n"
        "5 8 1\n"
        "6 4 1\n"
        "6 6 1\n"
        "6 9 2\n"
        "7 1 2\n"
        "7 5 3\n"
        "7 7 1\n"
        "7 10 1\n"
        "8 2 1\n"
        "8 6 1\n"
        "8 8 1\n"
        "8 9 3\n"
        "8 10 1\n";

    spex_set_gmp_ntrials(INT64_MAX);
    malloc_count = INT64_MAX;
    generate_test_matrix(&A, wide_8x10_dense_str);
    create_test_rhs(&b, A->m);
    OK(spex_test_qr_backslash(A, b, option));
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_matrix_free(&b, option));

    spex_set_gmp_ntrials(INT64_MAX);
    malloc_count = INT64_MAX;

    // This hits the specific "return SPEX_SINGULAR" in spex_qr_transpose_backslash.c
    const char *wide_rd_str =
        "2 3 6\n"
        "1 1 1\n"
        "1 2 1\n"
        "1 3 1\n"
        "2 1 1\n"
        "2 2 1\n"
        "2 3 1\n"; // Col 3 is a duplicate of Col 1
    generate_test_matrix(&A, wide_rd_str);
    create_test_rhs(&b, A->m);
    ERR(SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option), SPEX_SINGULAR);
    // Hit an error in analyze
    option->order = 99;
    option->algo = 99;
    ERR(SPEX_qr_analyze(&S, A, option), SPEX_INCORRECT_INPUT);
    option->order = SPEX_DEFAULT_ORDERING;
    ERR(SPEX_qr_analyze(&S, A, option), SPEX_INCORRECT_INPUT);
    option->algo = SPEX_ALGORITHM_DEFAULT;
    OK(SPEX_matrix_free(&A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_matrix_free(&b, option));

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    generate_test_matrix(&A, sq_str);
    BRUTAL(SPEX_qr_analyze(&S, A, option));
    OK(SPEX_symbolic_analysis_free(&S, option));
    OK(SPEX_matrix_free(&A, option));

    // NULL String Free
    SPEX_mpfr_free_str(NULL);

    // The GMP NULL Allocator Panics
    // We must destroy the global SPEX environment to make spex_gmp = NULL
    SPEX_finalize();

    // Call the internal GMP hooks directly
    spex_gmp_allocate(10);
    spex_gmp_reallocate(NULL, 10, 20);

    // SPEX not initialized Panics
    ERR(SPEX_qr_factorize(&F2, A, S, option), SPEX_PANIC);
    ERR(SPEX_qr_analyze(NULL, NULL, NULL), SPEX_PANIC);
    ERR(SPEX_qr_solve(NULL, NULL, NULL, NULL), SPEX_PANIC);
    ERR(SPEX_qr_backslash(NULL, SPEX_MPQ, NULL, NULL, NULL), SPEX_PANIC);
    ERR(SPEX_qr_rank(NULL, NULL, NULL), SPEX_PANIC);

    // Properly revive the environment before freeing everything
    SPEX_initialize_expert(tcov_malloc, tcov_calloc, tcov_realloc, tcov_free);

    SPEX_FREE_ALL;
    printf("%s: all tests passed\n\n", __FILE__);
    fprintf(stderr, "%s: all tests passed\n\n", __FILE__);
    SPEX_finalize();
    return 0;
}
