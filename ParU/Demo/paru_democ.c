// ==========================================================================  /
// =======================  paru_demo.c  ====================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief    test to see how to call umfpack symbolic analysis
 *
 * @author Aznaveh
 * */

#include <stdint.h>
#include <math.h>

#include "ParU.h"

#define FREE_ALL_AND_RETURN(info)               \
{                                               \
    if (b != NULL) free(b);                     \
    if (xx != NULL) free(xx);                   \
    if (x != NULL) free(x);                     \
    if (B != NULL) free(B);                     \
    if (X != NULL) free(X);                     \
    umfpack_dl_free_symbolic(&Symbolic);        \
    umfpack_dl_free_numeric(&Numeric);          \
    ParU_C_FreeNumeric(&Num, Control);          \
    ParU_C_FreeSymbolic(&Sym, Control);         \
    ParU_C_FreeControl(&Control);               \
    cholmod_l_free_sparse(&A, cc);              \
    cholmod_l_finish(cc);                       \
    return (info) ;                             \
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc ;
    cholmod_sparse *A = NULL ;
    ParU_C_Symbolic Sym = NULL ;
    ParU_C_Numeric Num = NULL ;
    ParU_C_Control Control = NULL ;
    double *b = NULL, *xx = NULL, *B = NULL, *X = NULL, *x = NULL ;
    void *Symbolic = NULL, *Numeric = NULL ;  // UMFPACK factorization

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        printf("ParU: input matrix is invalid\n");
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    if (mtype != CHOLMOD_SPARSE)
    {
        printf("ParU: input matrix must be sparse\n");
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    if (A->xtype != CHOLMOD_REAL)
    {
        printf("ParU: input matrix must be real\n");
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int ver[3];
    char date[128];
    ParU_C_Version(ver, date);
    printf("================= ParU %d.%d.%d C demo ===========================",
           ver[0], ver[1], ver[2]);
    printf(" %s\n", date);

    double my_start_time = SUITESPARSE_TIME;

    ParU_Info info;
    info = ParU_C_InitControl(&Control);  // initialize the Control in C

    ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING, PARU_ORDERING_METIS_GUARD,
        Control) ;
    printf ("\n--------- ParU_C_Analyze:\n") ;
    info = ParU_C_Analyze(A, &Sym, Control);
    double my_time_analyze = SUITESPARSE_TIME - my_start_time;
    if (info != PARU_SUCCESS)
    {
        FREE_ALL_AND_RETURN (info) ;
    }

    int64_t n, anz ;
    ParU_C_Get_INT64 (Sym, Num, PARU_GET_N, &n, Control) ;
    ParU_C_Get_INT64 (Sym, Num, PARU_GET_ANZ, &anz, Control) ;
    printf("In: %" PRId64 "x%" PRId64 " nnz = %" PRId64 " \n", n, n, anz);
    printf("ParU: Symbolic factorization: %lf seconds\n", my_time_analyze);
    printf ("\n--------- ParU_C_Factorize:\n") ;
    double my_start_time_fac = SUITESPARSE_TIME;
    info = ParU_C_Factorize(A, Sym, &Num, Control);
    double my_time_fac = SUITESPARSE_TIME - my_start_time_fac;
    if (info != PARU_SUCCESS)
    {
        printf("ParU: factorization was NOT successful in %lf seconds!",
               my_time_fac);
        if (info == PARU_OUT_OF_MEMORY) printf("\nOut of memory\n");
        if (info == PARU_INVALID) printf("\nInvalid!\n");
        if (info == PARU_SINGULAR) printf("\nSingular!\n");
        FREE_ALL_AND_RETURN (info) ;
    }

    printf("ParU: factorization was successful in %lf seconds.\n",
           my_time_fac);

    //~~~~~~~~~~~~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    double my_time, my_solve_time;

    b = (double *)malloc(n * sizeof(double));
    xx = (double *)malloc(n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;
    printf ("\n--------- ParU_C_Solve_Axb:\n") ;
    double my_solve_time_start = SUITESPARSE_TIME;
    info = ParU_C_Solve_Axb(Sym, Num, b, xx, Control);
    if (info != PARU_SUCCESS)
    {
        printf ("ParU: solve failed.\n");
        FREE_ALL_AND_RETURN (info) ;
    }
    my_solve_time = SUITESPARSE_TIME - my_solve_time_start;
    my_time = SUITESPARSE_TIME - my_start_time;
    printf("Solve time is %lf seconds.\n", my_solve_time);

    // printing out x
#if 0
    printf("x = [");
    for (int64_t i = 0; i < n; ++i)
        printf (" %.2lf, ", xx[i]);
    printf("]\n");
#endif

    double resid, anorm, xnorm;
    printf ("\n--------- ParU_C_Residual_bAx:\n") ;
    info =
        ParU_C_Residual_bAx(A, xx, b, &resid, &anorm, &xnorm, Control);
    if (info != PARU_SUCCESS)
    {
        printf("ParU: resid failed.\n");
        FREE_ALL_AND_RETURN (info) ;
    }
    double rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));

    double rcond ;
    ParU_C_Get_FP64 (Sym, Num, PARU_GET_RCOND_ESTIMATE, &rcond, Control) ;
    printf(
        "Residual is |%.2e|, anorm is %.2e, xnorm is %.2e, and rcond is"
        " %.2e.\n", rresid, anorm, xnorm, rcond);

    const int64_t nrhs = 16;  // number of right handsides
    B = (double *)malloc(n * nrhs * sizeof(double));
    X = (double *)malloc(n * nrhs * sizeof(double));
    for (int64_t i = 0; i < n; ++i)
    {
        for (int64_t j = 0; j < nrhs; ++j)
        {
            B[j * n + i] = (double)(i + j + 1);
        }
    }

    printf ("\n--------- ParU_C_Solve_AXB:\n") ;
    info = ParU_C_Solve_AXB(Sym, Num, nrhs, B, X, Control);
    if (info != PARU_SUCCESS)
    {
        printf("ParU: mRhs Solve has a problem.\n");
        FREE_ALL_AND_RETURN (info) ;
    }
    printf ("\n--------- ParU_C_Residual_BAX:\n") ;
    info = ParU_C_Residual_BAX(A, X, B, nrhs, &resid, &anorm, &xnorm,
                               Control);
    if (info != PARU_SUCCESS)
    {
        printf("ParU: mRhs Residual has a problem.\n");
        FREE_ALL_AND_RETURN (info) ;
    }

    rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    printf("Multiple right hand side: relative residual is |%.2e|.\n",
            rresid);

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //~~~~~~~~~~~~~~~~~~~Calling umfpack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    double umf_time = 0;
    double umf_start_time = SUITESPARSE_TIME;
    double status,           // Info [UMFPACK_STATUS]
        Info[UMFPACK_INFO],  // statistics about the symbolic analysis
        umf_Control[UMFPACK_CONTROL];  // it is set in umfpack_dl_defaults and
    // is used in umfpack_dl_symbolic; if
    // passed NULL it will use the defaults
    umfpack_dl_defaults(umf_Control);
    umf_Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS_GUARD;

    int64_t *Ap = (int64_t *)A->p;
    int64_t *Ai = (int64_t *)A->i;
    double *Ax = (double *)A->x;

    status =
        umfpack_dl_symbolic(n, n, Ap, Ai, Ax, &Symbolic, umf_Control, Info);
    // umf_Control[UMFPACK_PRL] = 0;
    // umfpack_dl_report_info(umf_Control, Info);
    if (status < 0)
    {
        umfpack_dl_report_info(umf_Control, Info);
        umfpack_dl_report_status(umf_Control, status);
        printf("umfpack_dl_symbolic failed\n");
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }
    double umf_symbolic = SUITESPARSE_TIME - umf_start_time;
    double umf_fac_start = SUITESPARSE_TIME;
    status =
        umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, umf_Control, Info);
    // umf_Control[UMFPACK_PRL] = 2;
    // umfpack_dl_report_info(umf_Control, Info);
    // umfpack_dl_report_status(umf_Control, status);
    if (status < 0)
    {
        umfpack_dl_report_info(umf_Control, Info);
        umfpack_dl_report_status(umf_Control, status);
        printf("umfpack_dl_numeric failed\n");
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    double umf_time_fac = SUITESPARSE_TIME - umf_fac_start;

    x = (double *)malloc(n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;

    double solve_start = SUITESPARSE_TIME;
    status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, umf_Control,
                              Info);
    double umf_solve_time = SUITESPARSE_TIME - solve_start;
    umf_time = SUITESPARSE_TIME - umf_start_time;
    double umf_resid, umf_anorm, umf_xnorm;
    info = ParU_C_Residual_bAx(A, x, b, &umf_resid, &umf_anorm, &umf_xnorm,
                               Control);
    double umf_rresid = (umf_anorm == 0 || umf_xnorm == 0 )
        ? 0 : (umf_resid/(umf_anorm*umf_xnorm));
    printf(
        "UMFPACK Residual is |%.2e|, anorm is %.2e, xnorm is %.2e, and rcond"
        " is %.2e.\n",
        umf_rresid, umf_anorm, umf_xnorm, rcond);

    // Writing results to a file
#if 0
    if (info == PARU_SUCCESS)
    {
        FILE *res_file;
        char res_name[] = "res.txt";
        res_file = fopen(res_name, "a");
        if (res_file == NULL)
        {
            printf("Par: error in making %s to write the results!\n", res_name);
        }
        fprintf(res_file, "%ld %ld %lf %lf %lf %lf %lf %lf %lf %lf\n", n,
                anz, my_time_analyze, my_time_fac, my_solve_time, my_time,
                umf_symbolic, umf_time_fac, umf_solve_time, umf_time);
        fclose(res_file);
    }
    printf("my_time = %lf umf_time=%lf umf_solv_t = %lf ratio = %lf\n", my_time,
           umf_time, umf_solve_time, my_time / umf_time);
#endif  // writing to a file

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    FREE_ALL_AND_RETURN (PARU_SUCCESS) ;
}
