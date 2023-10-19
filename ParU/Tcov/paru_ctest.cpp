// ==========================================================================  /
// =======================  paru_ctest.cpp ==================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*
 * @brief    test to see how to call umfpack symbolic analysis
 *
 * @author Aznaveh
 * */
#include <math.h>

#define TEST_FREE_ALL                       \
{                                           \
    umfpack_dl_free_symbolic(&Symbolic);    \
    umfpack_dl_free_numeric(&Numeric);      \
    ParU_C_Freenum(&Num, &Control);         \
    ParU_C_Freesym(&Sym, &Control);         \
    cholmod_l_free_sparse(&A, cc);          \
    cholmod_l_finish(cc);                   \
    if (B  != NULL) free(B);                \
    B  = NULL ;                             \
    if (X  != NULL) free(X);                \
    X  = NULL ;                             \
    if (b  != NULL) free(b);                \
    b  = NULL ;                             \
    if (x  != NULL) free(x);                \
    x  = NULL ;                             \
    if (xx != NULL) free(xx) ;              \
    xx = NULL ;                             \
}

#include "paru_cov.hpp"
extern "C"
{
#include "ParU_C.h"
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_C_Symbolic *Sym = NULL;
    ParU_C_Numeric *Num = NULL ;
    double *b = NULL, *B = NULL, *X = NULL, *xx = NULL, *x = NULL ;
    void *Symbolic = NULL, *Numeric = NULL;

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

    ParU_C_Control Control;
    ParU_C_Init_Control(&Control); //initialize the Control in C

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        TEST_ASSERT (expected_log10_resid == 101) ;
        return (0) ;
    }

    if (mtype != CHOLMOD_SPARSE)
    {
        TEST_ASSERT (expected_log10_resid == 102) ;
        return (0) ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int ver[3];
    char date[128];
    ParU_C_Version(ver, date);


    ParU_Ret info;

    Control.umfpack_ordering = UMFPACK_ORDERING_AMD;
    // Control.umfpack_strategy = UMFPACK_STRATEGY_UNSYMMETRIC;
    // Control.umfpack_strategy = UMFPACK_STRATEGY_SYMMETRIC;
    // Control.umfpack_default_singleton = 0;
    // Control.paru_max_threads = 6;
    Control.umfpack_ordering = UMFPACK_ORDERING_METIS_GUARD;
    //info = ParU_C_Analyze(A, &Sym, &Control);
    BRUTAL_ALLOC_TEST(info, ParU_C_Analyze(A, &Sym, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 104) ;
        return (0) ;
    }

    //info = ParU_C_Factorize(A, Sym, &Num, &Control);
    BRUTAL_ALLOC_TEST(info, ParU_C_Factorize(A, Sym, &Num, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 105) ;
        return (0) ;
    }

    //~~~~~~~~~~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int64_t m = Sym->m;
    // double my_time, my_solve_time;
    double resid = 0, anorm = 0, xnorm = 0;

    b = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    xx = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (xx != NULL) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;
    //info = ParU_C_Solve_Axb(Sym, Num, b, xx, &Control);

    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_Axb(Sym, Num, b, xx, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 106) ;
        return (0) ;
    }

    BRUTAL_ALLOC_TEST(info, ParU_C_Residual_bAx(A, xx, b, m, &resid, 
                &anorm, &xnorm, &Control));
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        return (0) ;
    }

    printf("Residual is |%.2e| anorm is %.2e, xnorm is %.2e and rcond"
            " is %.2e.\n", resid, anorm, xnorm, Num->rcond);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

   BRUTAL_ALLOC_TEST(info, ParU_C_Solve_Axx(Sym, Num, b, &Control));
   if (info != PARU_SUCCESS)
   {
       TEST_ASSERT (expected_log10_resid == 107) ;
       return (0) ;
   }


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

    //info = ParU_C_Solve_AXB(Sym, Num, nrhs, B, X, &Control);
    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_AXB(Sym, Num, nrhs, B, X, 
                &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 109) ;
        return (0) ;
    }

    BRUTAL_ALLOC_TEST (info, ParU_C_Residual_BAX(A, X, B, m, nrhs, &resid,
                &anorm, &xnorm, &Control));
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 110) ;
        return (0) ;
    }

    printf("mRhs Residual is |%.2e|\n", resid);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_AXX(Sym, Num, nrhs, B, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 111) ;
        return (0) ;
    }

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int64_t max_threads = omp_get_max_threads();
    omp_set_num_threads (max_threads);

    //~~~~~~~~~~~~~~~~~~~Calling umfpack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // double umf_time = 0;

    int status ;           // Info [UMFPACK_STATUS]
    double Info[UMFPACK_INFO],  // Contains statistics about the symbolic analysis

        umf_Control[UMFPACK_CONTROL];  // it is set in umfpack_dl_defaults and
    // is used in umfpack_dl_symbolic; if
    // passed NULL it will use the defaults
    umfpack_dl_defaults(umf_Control);
    // umf_Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
    // umf_Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    // umf_Control [UMFPACK_STRATEGY] =   UMFPACK_STRATEGY_UNSYMMETRIC;
    // umf_Control [UMFPACK_STRATEGY] =   UMFPACK_STRATEGY_SYMMETRIC;
    // umf_Control[UMFPACK_SINGLETONS] = Control.umfpack_default_singleton;
    umf_Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS_GUARD;

    int64_t *Ap = (int64_t *)A->p;
    int64_t *Ai = (int64_t *)A->i;
    double *Ax = (double *)A->x;
    // int64_t m = A->nrow;
    int64_t n = A->ncol;

    status =
        umfpack_dl_symbolic(n, n, Ap, Ai, Ax, &Symbolic, umf_Control, Info);
    // umf_Control[UMFPACK_PRL] = 0;
    // umfpack_dl_report_info(umf_Control, Info);
    TEST_ASSERT_INFO (status == UMFPACK_OK, status) ;


    status =
        umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, umf_Control, Info);
    TEST_ASSERT_INFO (status == UMFPACK_OK, status) ;

    // umf_Control[UMFPACK_PRL] = 2;
    // umfpack_dl_report_info(umf_Control, Info);
    // umfpack_dl_report_status(umf_Control, status);


    b = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    x = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (x != NULL) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;

    status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, umf_Control,
                              Info);
    TEST_ASSERT_INFO (status == UMFPACK_OK, status) ;

    double umf_resid, umf_anorm, umf_xnorm;
    info = ParU_C_Residual_bAx(A, x, b, m, &umf_resid, 
            &umf_anorm, &umf_xnorm, &Control);
    umf_resid = (umf_anorm == 0 || umf_xnorm == 0 ) ? 0 : 
        (umf_resid/(umf_anorm*umf_xnorm));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 112) ;
        return (0) ;
    }

    printf("UMFPACK Residual is |%.2e| and anorm is %.2e and rcond is %.2e.\n",
           umf_resid, umf_anorm, Num->rcond);
    TEST_ASSERT (umf_resid == 0 || log10 (umf_resid) <= expected_log10_resid) ;

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_PASSES ;

}
