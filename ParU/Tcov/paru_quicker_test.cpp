// ==========================================================================  /
// =======================  paru_quicker_test.cpp  ==========================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*
 * @brief    for coverage test of bigger matrices
 *
 * @author Aznaveh
 * */
#include <math.h>

#include "paru_cov.hpp"

#define TEST_FREE_ALL                       \
{                                           \
    ParU_Freenum(&Num, &Control);           \
    ParU_Freesym(&Sym, &Control);           \
    cholmod_l_free_sparse(&A, cc);          \
    cholmod_l_finish(cc);                   \
    if (B  != NULL) { free(B);  B  = NULL; } \
    if (X  != NULL) { free(X);  X  = NULL; } \
    if (b  != NULL) { free(b);  b  = NULL; } \
    if (x  != NULL) { free(x);  x  = NULL; } \
    if (xx != NULL) { free(xx); xx = NULL; } \
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_Symbolic *Sym = NULL;
    ParU_Numeric *Num = NULL ;
    double *b = NULL, *B = NULL, *X = NULL, *xx = NULL, *x = NULL ;

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
    Control.mem_chunk = 1024;
    Control.umfpack_ordering = 23;
    Control.umfpack_strategy = 23;
    Control.paru_max_threads = 4;
    Control.relaxed_amalgamation_threshold = -1;
    Control.paru_strategy = 23;
    Control.scale = -1;
    Control.panel_width = -1;
    Control.piv_toler = -1;
    Control.diag_toler = -1;
    Control.trivial = -1;
    Control.worthwhile_dgemm = -2;
    Control.worthwhile_trsm = -1;
    Control.umfpack_default_singleton= -1;
    Control.paru_strategy = PARU_STRATEGY_SYMMETRIC;

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);

    if (mtype != CHOLMOD_SPARSE)
    {
        // matrix is not sparse
        TEST_ASSERT (expected_log10_resid == 101) ;
        TEST_PASSES ;
    }

    /////This part is for covering the codes that cannot be covered through
    ///// factorizing 
    // covering alloc lines
    int64_t *t = NULL;

    t = (int64_t *)paru_alloc(1, sizeof(int64_t) * 0);
    TEST_ASSERT (t == NULL) ;

    t = (int64_t *)paru_alloc(Size_max, sizeof(int64_t));
    TEST_ASSERT (t == NULL) ;

    size_t size = 0;
    t = (int64_t *)paru_realloc(10, sizeof(int64_t) * 0, t, &size);
    TEST_ASSERT (t == NULL) ;

    t = (int64_t *)paru_realloc(10, sizeof(int64_t), t, &size);
    TEST_ASSERT (t != NULL) ;
    paru_free(10, sizeof(int64_t), t);
    t = NULL ;

    int64_t *test_new = new int64_t[0];
    delete[] test_new;
    // covering elementList
    paru_element* elementList[] = {NULL};
    int64_t lac_0 = lac_el(elementList, 0);
    TEST_ASSERT (lac_0 == LONG_MAX) ;

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int ver[3] = {0, 0, 0} ;
    char date[128];
    date [0] = '\0' ;
    ParU_Version(ver, date);
    printf("ParU %d.%d.%d", ver[0], ver[1], ver[2]);
    printf(" %s\n", date);
    TEST_ASSERT (date [0] != '\0') ;

    ParU_Ret info;

    info = ParU_Analyze(A, &Sym, &Control);
    if (info != PARU_SUCCESS)
    {
        // analysis failed
        TEST_ASSERT (expected_log10_resid == 102) ;
        TEST_ASSERT_INFO (info == PARU_INVALID || info == PARU_SINGULAR, info) ;
        printf ("matrix invalid, or singular (expected result)\n") ;
        TEST_PASSES ;
    }

    Control.paru_strategy = PARU_STRATEGY_AUTO;
    info = ParU_Analyze(A, &Sym, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 103) ;
        TEST_PASSES ;
    }

    // printf("In: %ldx%ld nnz = %ld \n", Sym->m, Sym->n, Sym->anz);

    info = ParU_Factorize(NULL, Sym, &Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Factorize(A, NULL, &Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Factorize(A, Sym, &Num, &Control);
    if (info != PARU_SUCCESS)
    {
        // matrix is singular
        TEST_ASSERT (expected_log10_resid == 104) ;
        TEST_ASSERT_INFO (info == PARU_SINGULAR, info) ;
        printf ("matrix is numerically singular (expected result)\n") ;
        TEST_PASSES ;
    }

    //~~~~~~~~~~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int64_t m = Sym->m;
    double resid = 0, anorm = 0 , xnorm = 0 ;

    b = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    xx = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (xx != NULL) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;

    info = ParU_Solve(Sym, NULL, b, xx, &Control);  // coverage
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, NULL, b, &Control);      // coverage
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, b, xx, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 105) ;
        TEST_PASSES ;
    }

    info =
        ParU_Residual(A, xx, NULL, m, resid, anorm, xnorm, &Control);// coverage
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Residual(A, xx, b, m, resid, anorm, xnorm, &Control);
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 106) ;
        TEST_PASSES ;
    }

    printf("Residual is |%.2e|, anorm is %.2e, xnorm is %.2e "
            "and rcond is %.2e.\n", resid, anorm, xnorm, Num->rcond);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;
    info = paru_backward(b, resid, anorm, xnorm, NULL, Sym, Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    if (b  != NULL) free(b);
    b = NULL ;
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

    info = ParU_Solve(Sym, NULL, nrhs, B, X, &Control);  // for coverage
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, NULL, nrhs, B, &Control);     // for coverage
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, nrhs, B, X, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 107) ;
        TEST_PASSES ;
    }

    // This one is just for more coverage
    info = ParU_Residual(A, X, NULL, m, nrhs, resid, anorm, xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Residual(A, X, B, m, nrhs, resid, anorm, xnorm, &Control);
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        TEST_PASSES ;
    }

    printf("mRhs Residual is |%.2e|\n", resid);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_PASSES ;
}

