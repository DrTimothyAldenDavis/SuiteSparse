// ==========================================================================  /
// =======================  paru_quick_test.cpp  ============================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief    for coverage test of bigger matrices
 *
 * @author Aznaveh
 * */
#include <math.h>

#include "paru_cov.hpp"

#define TEST_FREE_ALL                                   \
{                                                       \
    ParU_FreeNumeric(&Num, &Control);                   \
    ParU_FreeSymbolic(&Sym, &Control);                  \
    cholmod_l_free_sparse(&A, cc);                      \
    cholmod_l_finish(cc);                               \
    if (B  != NULL) { free(B);  B  = NULL; }            \
    if (P  != NULL) { free(P);  P  = NULL; }            \
    if (Q  != NULL) { free(Q);  Q  = NULL; }            \
    if (R  != NULL) { free(R);  R  = NULL; }            \
    if (Perm != NULL) { free(Perm);  Perm = NULL; }     \
    if (Scale != NULL) { free(Scale);  Scale = NULL; }  \
    if (X  != NULL) { free(X);  X  = NULL; }            \
    if (b  != NULL) { free(b);  b  = NULL; }            \
    if (x  != NULL) { free(x);  x  = NULL; }            \
    if (xx != NULL) { free(xx); xx = NULL; }            \
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_Symbolic Sym = NULL;
    ParU_Numeric Num = NULL ;
    double *b = NULL, *B = NULL, *X = NULL, *xx = NULL, *x = NULL,
        *Scale = NULL, *R = NULL ;
    int64_t *Perm = NULL, *P = NULL, *Q = NULL ;
    double err = 0 ;

    // default log10 of expected residual.  +1 means failure is expected
    double expected_log10_resid = -16 ;
    if (argc > 1)
    {
        expected_log10_resid = (double) atoi (argv [1]) ;
    }

    // ordering to use
    int umfpack_ordering = 23 ;
    if (argc > 2)
    {

#if 0
/* Control [UMFPACK_ORDERING] and Info [UMFPACK_ORDERING_USED] are one of: */
#define UMFPACK_ORDERING_CHOLMOD 0      /* use CHOLMOD (AMD/COLAMD then METIS)*/
#define UMFPACK_ORDERING_AMD 1          /* use AMD/COLAMD */
#define UMFPACK_ORDERING_GIVEN 2        /* user-provided Qinit */
#define UMFPACK_ORDERING_METIS 3        /* use METIS */
#define UMFPACK_ORDERING_BEST 4         /* try many orderings, pick best */
#define UMFPACK_ORDERING_NONE 5         /* natural ordering */
#define UMFPACK_ORDERING_USER 6         /* user-provided function */
#define UMFPACK_ORDERING_METIS_GUARD 7  // Use METIS, AMD, or COLAMD.
    // Symmetric strategy: always use METIS on A+A'.  Unsymmetric strategy: use
    // METIS on A'A, unless A has one or more very dense rows.  In that case,
    // A'A is very costly to form.  In this case, COLAMD is used instead of
    // METIS.
#endif

        umfpack_ordering = (int) atoi (argv [2]) ;
        printf ("umfpack ordering to use: %d\n", umfpack_ordering) ;
    }

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype = -1 ;
    cholmod_l_start(cc);

    ParU_Control Control;
    // puting Control lines to work
    Control.mem_chunk = 1024;
    Control.umfpack_ordering = umfpack_ordering ;
    Control.umfpack_strategy = 23;
    Control.paru_max_threads = 4;
    Control.relaxed_amalgamation = -1;
    Control.paru_strategy = 23;
    Control.prescale = -1;
    Control.panel_width = -1;
    Control.piv_toler = -1;
    Control.diag_toler = -1;
    Control.trivial = -1;
    Control.worthwhile_dgemm = -2;
    Control.worthwhile_trsm = -1;
    Control.filter_singletons = 2 ;
    Control.paru_strategy = PARU_STRATEGY_SYMMETRIC;

    // read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);

    if (mtype != CHOLMOD_SPARSE)
    {
        // matrix is not sparse
        TEST_ASSERT (expected_log10_resid == 101) ;
        TEST_PASSES ;
    }

    // to test upper bound on this option (Matrix/xenon1.mtx)
    Control.relaxed_amalgamation = (A->nrow == 48600) ? 1024 : (-1) ;

    /////This part is for covering the codes that cannot be covered through
    ///// factorizing
    // covering alloc lines
    int64_t *t = NULL;

    t = (int64_t *)paru_malloc (1, sizeof(int64_t) * 0);
    TEST_ASSERT (t == NULL) ;

    t = (int64_t *)paru_malloc (Size_max, sizeof(int64_t));
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

    ParU_Info info;

    info = ParU_Analyze(nullptr, &Sym, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Analyze(A, nullptr, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Analyze(A, &Sym, nullptr);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

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

    info = ParU_Factorize(nullptr, Sym, &Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Factorize(A, nullptr, &Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Factorize(A, Sym, nullptr, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Factorize(A, Sym, &Num, nullptr);
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

    const char *blas_library, *tasking ;
    double resid = 0, anorm = 0 , xnorm = 0, rcond,
        min_udiag, max_udiag, flops ;
    int64_t n, anz, rs1, cs1, paru_strategy, umfpack_strategy,
        umf_ordering, lnz, unz ;

    info = ParU_Get (Sym, Num, PARU_GET_BLAS_LIBRARY_NAME, &blas_library,
        &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    printf ("blas_library: %s\n", blas_library) ;

    info = ParU_Get (Sym, Num, PARU_GET_FRONT_TREE_TASKING, &tasking,
        &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    printf ("tasking: %s\n", tasking) ;

    info = ParU_Get (Sym, Num, PARU_GET_N, &tasking, &Control) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_BLAS_LIBRARY_NAME,
        (const char **) NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_N, (int64_t *) NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_N, &n, (ParU_Control *) NULL) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_BLAS_LIBRARY_NAME, &n, &Control) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_BLAS_LIBRARY_NAME, &flops, &Control) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_N, &n, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_ANZ, &anz, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_NROW_SINGLETONS, &rs1, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_NCOL_SINGLETONS, &cs1, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_PARU_STRATEGY, &paru_strategy,
        &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_UMFPACK_STRATEGY, &umfpack_strategy,
        &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_UMFPACK_ORDERING, &umf_ordering,
        &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_LNZ, &lnz, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_UNZ, &unz, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    printf ("n: %ld anz: %ld rs1: %ld cs1: %ld \n"
        "paru strategy: %ld umfpack_strategy: %ld umfpack_odering: %ld\n"
        "lnz: %ld unz: %ld\n", n, anz, rs1, cs1, paru_strategy,
        umfpack_strategy, umf_ordering, lnz, unz) ;

    Q = (int64_t *) malloc (n * sizeof (int64_t)) ;
    TEST_ASSERT (Q != NULL) ;
    P = (int64_t *) malloc (n * sizeof (int64_t)) ;
    TEST_ASSERT (P != NULL) ;
    R = (double *) malloc (n * sizeof (double)) ;
    TEST_ASSERT (R != NULL) ;

    info = ParU_Get (Sym, Num, PARU_GET_P, P, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_Q, Q, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_ROW_SCALE_FACTORS, R, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_RCOND_ESTIMATE, &rcond, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_MIN_UDIAG, &min_udiag, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_MAX_UDIAG, &max_udiag, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_FLOP_COUNT, &flops, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_FLOP_COUNT, &flops, (ParU_Control *) NULL) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Get (Sym, Num, PARU_GET_FLOP_COUNT, (double *) NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    printf ("min_udiag: %g max_udiag: %g rcond: %g flops: %g\n",
        min_udiag, max_udiag, rcond, flops) ;

    b = (double *)malloc(n * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    xx = (double *)malloc(n * sizeof(double));
    TEST_ASSERT (xx != NULL) ;

    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;

    info = ParU_LSolve(NULL, Num, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_LSolve(Sym, NULL, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_LSolve(Sym, Num, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_LSolve(Sym, Num, b, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(NULL, Num, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(Sym, NULL, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(Sym, Num, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(Sym, Num, b, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (NULL, R, b, n, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (P, R, NULL, n, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (P, R, b, n, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (P, R, b, n, xx, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (NULL, NULL, b, n, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Q, NULL, NULL, n, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Q, NULL, b, n, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Q, NULL, b, n, xx, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    Perm = (int64_t *) malloc (n * sizeof(int64_t)) ;
    TEST_ASSERT (Perm != NULL) ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        Perm [i] = (i + 2) % n ;
    }
    Scale = (double *) malloc (n * sizeof (double)) ;
    TEST_ASSERT (Scale != NULL) ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        Scale [i] = 2 + i / 100.0 ;
    }

    info = ParU_InvPerm (Perm, Scale, b, n, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    err = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        err = fabs (xx [Perm [i]] * Scale [Perm [i]] - b [i]) ;
    }
    TEST_ASSERT (err < 1e-10) ;

    info = ParU_Perm (Perm, NULL, b, n, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    err = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        err = fabs (xx [i] - b [Perm [i]]) ;
    }
    TEST_ASSERT (err < 1e-10) ;

    info = ParU_Solve(NULL, Num, b, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, NULL, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, b, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, b, xx, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 105) ;
        TEST_PASSES ;
    }

    info =
        ParU_Residual(A, xx, NULL, resid, anorm, xnorm, &Control);// coverage
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Residual(A, xx, b, resid, anorm, xnorm, &Control);
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 106) ;
        TEST_PASSES ;
    }

    printf("Residual is |%.2e|, anorm is %.2e, xnorm is %.2e "
            "and rcond is %.2e.\n", resid, anorm, xnorm, rcond);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;
    info = paru_backward(b, resid, anorm, xnorm, NULL, Sym, Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    if (b != NULL) free(b);
    b = NULL ;
    if (xx != NULL) free(xx);
    xx = NULL ;

    const int64_t nrhs = 16;  // number of right handsides

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

    info = ParU_LSolve(NULL, Num, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_LSolve(Sym, NULL, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_LSolve(Sym, Num, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_LSolve(Sym, Num, nrhs, X, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(NULL, Num, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(Sym, NULL, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(Sym, Num, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_USolve(Sym, Num, nrhs, X, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (NULL, R, B, n, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (P, R, NULL, n, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (P, R, B, n, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Perm (P, R, B, n, nrhs, X, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (NULL, NULL, B, n, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Q, NULL, NULL, n, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Q, NULL, B, n, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Q, NULL, B, n, nrhs, X, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_InvPerm (Perm, Scale, B, n, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    err = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        for (int64_t j = 0 ; j < nrhs ; j++)
        {
            err = fabs (X [Perm [i] + j*n] * Scale [Perm [i]] - B [i + j*n]) ;
        }
    }
    TEST_ASSERT (err < 1e-10) ;

    info = ParU_Perm (Perm, NULL, B, n, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    err = 0 ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        for (int64_t j = 0 ; j < nrhs ; j++)
        {
            err = fabs (X [i + j*n] - B [Perm [i] + j*n]) ;
        }
    }
    TEST_ASSERT (err < 1e-10) ;

    info = ParU_Solve(NULL, Num, nrhs, B, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, NULL, nrhs, B, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, nrhs, NULL, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, nrhs, B, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, nrhs, B, X, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Solve(Sym, Num, nrhs, B, X, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 107) ;
        TEST_PASSES ;
    }

    // This one is just for more coverage
    info = ParU_Residual(A, X, NULL, nrhs, resid, anorm, xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_Residual(A, X, B, nrhs, resid, anorm, xnorm, &Control);
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        TEST_PASSES ;
    }

    printf("mRhs Residual is |%.2e|\n", resid);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    // solve again, overwriting B with solution X
    info = ParU_Solve(Sym, Num, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    // compare X and B
    err = 0 ;
    for (int64_t i = 0 ; i < n*nrhs ; i++)
    {
        err += fabs (B [i] - X [i]) ;
    }
    err = err / xnorm ;
    TEST_ASSERT (err < 1e-10) ;

    info = ParU_FreeNumeric (NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_FreeNumeric (&Num, NULL) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    info = ParU_FreeSymbolic (NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    info = ParU_FreeSymbolic (&Sym, NULL) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // test paru_bin_src_col
    int64_t sorted_list [5] = {1, 4, 9, 99} ;
    int64_t result = paru_bin_srch_col (sorted_list, 0, 4, 100) ;
    TEST_ASSERT (result == -1) ;
    sorted_list [1] = flip (4) ;
    result = paru_bin_srch_col (sorted_list, 0, 4, 4) ;
    TEST_ASSERT (result == 1) ;

    info = paru_umfpack_info (UMFPACK_WARNING_singular_matrix) ;
    TEST_ASSERT (info == PARU_SINGULAR) ;

    info = paru_umfpack_info (UMFPACK_ERROR_out_of_memory) ;
    TEST_ASSERT (info == PARU_OUT_OF_MEMORY) ;

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_PASSES ;
}

