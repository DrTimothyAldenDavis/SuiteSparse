// ==========================================================================  /
// =======================  paru_brutal_test.cpp  ===========================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief    to test all the allocations: malloc, calloc, realloc
 *              They fail one by one; very slow
 *
 * @author Aznaveh
 * */
#include <math.h>

#define TEST_FREE_ALL                       \
{                                           \
    ParU_FreeNumeric(&Num, Control);        \
    ParU_FreeSymbolic(&Sym, Control);       \
    ParU_FreeControl(&Control);             \
    cholmod_l_free_sparse(&A, cc);          \
    cholmod_l_finish(cc);                   \
    if (B  != NULL) free(B);                \
    B   = NULL ;                            \
    if (X  != NULL) free(X);                \
    X  = NULL ;                             \
    if (b  != NULL) free(b);                \
    b  = NULL ;                             \
    if (xx != NULL) free(xx) ;              \
    xx = NULL ;                             \
}

#include "paru_cov.hpp"

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A = NULL;
    ParU_Symbolic Sym = NULL;
    ParU_Numeric Num = NULL;
    ParU_Control Control = NULL ;
    double *b = NULL, *B = NULL, *X = NULL, *xx = NULL ;
    ParU_Info info;

    // default log10 of expected residual.  +1 means failure is expected
    double expected_log10_resid = -16 ;
    if (argc > 1)
    {
        expected_log10_resid = (double) atoi (argv [1]) ;
    }

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);

    // puting Control lines to work
    info = ParU_InitControl (NULL) ;
    TEST_ASSERT (info == PARU_INVALID) ;
    BRUTAL_ALLOC_TEST (info, ParU_InitControl (&Control)) ;
    TEST_ASSERT (info == PARU_SUCCESS) ;
    ParU_Set (PARU_CONTROL_ORDERING, PARU_ORDERING_AMD, Control) ;

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        TEST_ASSERT (expected_log10_resid == 101) ;
        TEST_PASSES ;
    }

    if (mtype != CHOLMOD_SPARSE)
    {
        TEST_ASSERT (expected_log10_resid == 102) ;
        TEST_PASSES ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    BRUTAL_ALLOC_TEST(info, ParU_Analyze(A, &Sym, Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 104) ;
        TEST_PASSES ;
    }

    TEST_ASSERT (Sym != NULL) ;

    BRUTAL_ALLOC_TEST(info, ParU_Factorize(A, Sym, &Num, Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 105) ;
        TEST_PASSES ;
    }

    //~~~~~~~~~~~~~~~~~~~Test the results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int64_t n ;
    info = ParU_Get (Sym, Num, PARU_GET_N, &n, Control) ;
    TEST_ASSERT (info == PARU_SUCCESS) ;

    b = (double *)malloc(n * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    xx = (double *)malloc(n * sizeof(double));
    TEST_ASSERT (xx != NULL) ;

    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;

    BRUTAL_ALLOC_TEST(info, ParU_Solve(Sym, Num, b, xx, Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 106) ;
        TEST_PASSES ;
    }

    double resid, anorm, xnorm;
    BRUTAL_ALLOC_TEST(info, ParU_Residual(A, xx, b,
                resid, anorm, xnorm, Control));
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 107) ;
        TEST_PASSES ;
    }

    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;

    BRUTAL_ALLOC_TEST(
        info, paru_backward(b, resid, anorm, xnorm, A, Sym, Num, Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        TEST_PASSES ;
    }
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));

    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    if (b  != NULL) free(b );
    b  = NULL ;
    if (xx != NULL) free(xx);
    xx = NULL ;

    const int64_t nrhs = 16;  // number of right hand sides

    B = (double *)malloc(n * nrhs * sizeof(double));
    TEST_ASSERT (B != NULL) ;

    X = (double *)malloc(n * nrhs * sizeof(double));
    TEST_ASSERT (X != NULL) ;

    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = 0; j < nrhs; ++j)
        {
            B[j * n + i] = (double)(i + j + 1);
        }
    }

    BRUTAL_ALLOC_TEST(info, ParU_Solve(Sym, Num, nrhs, B, X, Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 109) ;
        TEST_PASSES ;
    }

    BRUTAL_ALLOC_TEST(
        info, ParU_Residual(A, X, B, nrhs, resid, anorm, xnorm, Control));
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 110) ;
        TEST_PASSES ;
    }

    printf("mRhs Residual is |%.2e|\n", resid) ;
    if (anorm > 0) resid = resid / anorm ;
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_PASSES ;
}

