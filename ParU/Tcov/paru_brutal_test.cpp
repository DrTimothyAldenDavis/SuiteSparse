// ==========================================================================  /
// =======================  paru_brutal_test.cpp  ===========================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*
 * @brief    to test all the allocations: malloc, calloc, realloc
 *              They fail one by one; very slow
 *
 * @author Aznaveh
 * */
#include <math.h>

#define TEST_FREE_ALL                       \
{                                           \
    ParU_Freenum(&Num, &Control);           \
    ParU_Freesym(&Sym, &Control);           \
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
    ParU_Symbolic *Sym = NULL;
    ParU_Numeric *Num = NULL;
    double *b = NULL, *B = NULL, *X = NULL, *xx = NULL ;

    // default log10 of expected residual.  +1 means failure is expected
    double expected_log10_resid = -16 ;
    if (argc > 1)  
    {
        expected_log10_resid = (double) atoi (argv [1]) ;
    }
    printf ("expected log10 of resid: %g\n", expected_log10_resid) ;

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);

    ParU_Control Control;
    // puting Control lines to work
    Control.mem_chunk = 0;
    Control.umfpack_ordering = 23;
    Control.umfpack_strategy = 23;
    Control.paru_max_threads = -1;
    Control.relaxed_amalgamation_threshold = -1;
    Control.paru_strategy = 23;
    Control.scale = -1;
    Control.panel_width = -1;
    Control.piv_toler = -1;
    Control.diag_toler = -1;
    Control.trivial = -1;
    Control.worthwhile_dgemm = -2;
    Control.worthwhile_trsm = -1;

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
    int ver[3];
    char date[128];
    ParU_Version(ver, date);
    printf("ParU %d.%d.%d", ver[0], ver[1], ver[2]);
    printf(" %s\n", date);

    ParU_Ret info;

    // info = ParU_Analyze(A, &Sym, &Control);
    BRUTAL_ALLOC_TEST(info, ParU_Analyze(A, &Sym, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 104) ;
        TEST_PASSES ;
    }

    TEST_ASSERT (Sym != NULL) ;

    // printf("In: %ldx%ld nnz = %ld \n", Sym->m, Sym->n, Sym->anz);

    // info = ParU_Factorize(A, Sym, &Num, &Control);
    BRUTAL_ALLOC_TEST(info, ParU_Factorize(A, Sym, &Num, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 105) ;
        TEST_PASSES ;
    }

    //~~~~~~~~~~~~~~~~~~~Test the results~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int64_t m = 0;
    m = Sym->m;

    b = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    xx = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (xx != NULL) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;
    // info = ParU_Solve(Sym, Num, b, xx, &Control);

    BRUTAL_ALLOC_TEST(info, ParU_Solve(Sym, Num, b, xx, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 106) ;
        TEST_PASSES ;
    }

    double resid, anorm, xnorm;
    // info = ParU_Residual(A, xx, b, m, resid, anorm, xnorm, &Control);
    BRUTAL_ALLOC_TEST(info, ParU_Residual(A, xx, b, m, 
                resid, anorm, xnorm, &Control));
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 107) ;
        TEST_PASSES ;
    }

    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;

    BRUTAL_ALLOC_TEST(
        info, paru_backward(b, resid, anorm, xnorm, A, Sym, Num, &Control));
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

    const int64_t nrhs = 16;  // number of right handsides

    B = (double *)malloc(m * nrhs * sizeof(double));
    TEST_ASSERT (B != NULL) ;

    X = (double *)malloc(m * nrhs * sizeof(double));
    TEST_ASSERT (X != NULL) ;

    for (int64_t i = 0; i < m; ++i)
    {
        for (int64_t j = 0; j < nrhs; ++j)
        {
            B[j * m + i] = (double)(i + j + 1);
        }
    }

    // info = ParU_Solve(Sym, Num, nrhs, B, X, &Control);

    BRUTAL_ALLOC_TEST(info, ParU_Solve(Sym, Num, nrhs, B, X, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 109) ;
        TEST_PASSES ;
    }

    BRUTAL_ALLOC_TEST(
        info, ParU_Residual(A, X, B, m, nrhs, resid, anorm, xnorm, &Control));
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

