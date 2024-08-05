// ==========================================================================  /
// =======================  paru_demo.cpp  ==================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief    test to see how to call umfpack symbolic analysis
 *
 * @author Aznaveh
 * */

#include <iostream>
#include <iomanip>
#include <ios>
#include <cmath>

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
    ParU_FreeNumeric(&Num, Control);            \
    ParU_FreeSymbolic(&Sym, Control);           \
    ParU_FreeControl(&Control);                 \
    cholmod_l_free_sparse(&A, cc);              \
    cholmod_l_finish(cc);                       \
    return (info) ;                             \
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_Symbolic Sym = NULL;
    ParU_Numeric Num = NULL ;
    ParU_Control Control = NULL ;
    ParU_Info info;
    double *b = NULL, *xx = NULL, *B = NULL, *X = NULL, *x = NULL ;
    void *Symbolic = NULL, *Numeric = NULL ;

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        std::cout << "ParU: input matrix is invalid" << std::endl;
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    if (mtype != CHOLMOD_SPARSE)
    {
        std::cout << "ParU: input matrix must be sparse" << std::endl;
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    if (A->xtype != CHOLMOD_REAL)
    {
        std::cout << "ParU: input matrix must be real" << std::endl;
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int ver[3];
    char date[128];
    ParU_Version(ver, date);
    std::cout << "================= ParU "
        << ver[0] << "." << ver[1] << "."  << ver[2]
        << " ================================== " << date << std::endl;

    info = ParU_InitControl (&Control) ;
    if (info != PARU_SUCCESS)
    {
        FREE_ALL_AND_RETURN (info) ;
    }

    const char *blas_name ;
    info = ParU_Get (PARU_CONTROL_BLAS_LIBRARY_NAME, &blas_name, Control) ;
    if (info != PARU_SUCCESS)
    {
        FREE_ALL_AND_RETURN (info) ;
    }
    std::cout << "BLAS: " << blas_name << std::endl ;

    const char *tasking ;
    info = ParU_Get (PARU_CONTROL_FRONT_TREE_TASKING, &tasking, Control) ;
    if (info != PARU_SUCCESS)
    {
        FREE_ALL_AND_RETURN (info) ;
    }
    std::cout << "frontal tree tasking: " << tasking << std::endl ;

    int64_t using_openmp ;
    info = ParU_Get (PARU_CONTROL_OPENMP, &using_openmp, Control) ;
    if (info != PARU_SUCCESS)
    {
        FREE_ALL_AND_RETURN (info) ;
    }
    std::cout << "OpenMP in ParU: " << (using_openmp ? "yes" : "no" )
        << std::endl ;

    double my_start_time = SUITESPARSE_TIME ;

    ParU_Set (PARU_CONTROL_ORDERING, PARU_ORDERING_METIS_GUARD, Control) ;

    std::cout << "\n--------- ParU_Analyze:\n";
    info = ParU_Analyze(A, &Sym, Control);
    double my_time_analyze = SUITESPARSE_TIME - my_start_time;
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: analyze failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }

    int64_t n, anz ;
    info = ParU_Get (Sym, Num, PARU_GET_N, &n, Control) ;
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: stats failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }

    info = ParU_Get (Sym, Num, PARU_GET_ANZ, &anz, Control) ;
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: stats failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }

    std::cout << "In: " << n << "x" << n
        << " nnz = " << anz << std::endl;
    std::cout << std::scientific << std::setprecision(1)
        << "ParU: Symbolic factorization: " << my_time_analyze
        << " seconds\n";
    std::cout << "\n--------- ParU_Factorize:" << std::endl;
    double my_start_time_fac = SUITESPARSE_TIME;
    info = ParU_Factorize(A, Sym, &Num, Control);
    double my_time_fac = SUITESPARSE_TIME - my_start_time_fac;
    if (info != PARU_SUCCESS)
    {
        std::cout << std::scientific << std::setprecision(1)
            << "ParU: factorization was NOT successful: "
            << my_time_fac << " seconds\n";
        if (info == PARU_OUT_OF_MEMORY)
            std::cout << "Out of memory\n";
        if (info == PARU_INVALID)
            std::cout << "Invalid!\n";
        if (info == PARU_SINGULAR)
            std::cout << "Singular!\n";
        FREE_ALL_AND_RETURN (info) ;
    }
    else
    {
        std::cout << std::scientific << std::setprecision(1)
            << "ParU: factorization was successful in " << my_time_fac
            << " seconds.\n";
    }

    double rcond ;
    info = ParU_Get (Sym, Num, PARU_GET_RCOND_ESTIMATE, &rcond, Control) ;
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: stats failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }

    //~~~~~~~~~~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    double my_time, my_solve_time;

    b = (double *)malloc(n * sizeof(double));
    xx = (double *)malloc(n * sizeof(double));

    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;
    std::cout << "\n--------- ParU_Solve:\n";
    double my_solve_time_start = SUITESPARSE_TIME;
    info = ParU_Solve(Sym, Num, b, xx, Control);
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: solve failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }
    my_solve_time = SUITESPARSE_TIME - my_solve_time_start;
    my_time = SUITESPARSE_TIME - my_start_time;
    std::cout << std::defaultfloat << std::setprecision(1)
        << "Solve time is " << my_solve_time << " seconds.\n";

    // printing out xx
#if 0
    std::cout << "x = [";
    std::cout << std::defaultfloat  << std::setprecision(22);
    for (int64_t i = 0; i < n; ++i)
        std::cout << " " << xx[i] << "', ";
    std::cout << "]\n";
#endif

    double resid, anorm, xnorm;
    std::cout << "\n--------- ParU_Residual:\n";
    info = ParU_Residual(A, xx, b, resid, anorm, xnorm, Control);
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: resid failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }
    double rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));

    std::cout << std::scientific << std::setprecision(2)
        << "Relative residual is |" << rresid << "|, anorm is " << anorm
        << ", xnorm is " << xnorm << " and rcond is " << rcond << "."
        << std::endl;

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

    std::cout << "\n--------- ParU_Solve:\n";
    info = ParU_Solve(Sym, Num, nrhs, B, X, Control);
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: solve failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }
    std::cout << "\n--------- ParU_Residual:\n";
    info = ParU_Residual(A, X, B, nrhs, resid, anorm, xnorm, Control);
    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: solve failed" << std::endl;
        FREE_ALL_AND_RETURN (info) ;
    }
    rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));

    std::cout << std::scientific << std::setprecision(2)
        << "Multiple right hand side: relative residual is |"
        << rresid << "|." << std::endl;

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //~~~~~~~~~~~~~~~~~~~Calling umfpack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    double umf_time = 0;
    double umf_start_time = SUITESPARSE_TIME;
    double status,           // Info [UMFPACK_STATUS]
        Info[UMFPACK_INFO],  // Contains statistics about the symbolic analysis
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
        std::cout << "umfpack_dl_symbolic failed\n";
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
        std::cout << "umfpack_dl_numeric failed\n";
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
    info = ParU_Residual(A, x, b, umf_resid, umf_anorm, umf_xnorm, Control);
    double umf_rresid = (umf_anorm == 0 || umf_xnorm == 0 )
        ? 0 : (umf_resid/(umf_anorm*umf_xnorm));
    std::cout << std::scientific << std::setprecision(2)
        << "UMFPACK relative residual is |" << umf_rresid << "|, anorm is "
        << umf_anorm << ", xnorm is " << umf_xnorm << " and rcond is "
        << rcond << "." << std::endl;

    // Writing results to a file
#if 0
    if (info == PARU_SUCCESS)
    {
        FILE *res_file;
        char res_name[] = "res.txt";
        res_file = fopen(res_name, "a");
        if (res_file == NULL)
        {
            std::cout << "Par: error in making " << res_name << " to write the results!\n";
        }
        fprintf(res_file, "%" PRId64" %" PRId64 " %lf %lf %lf %lf %lf %lf %lf %lf\n",
                n, anz, my_time_analyze, my_time_fac, my_solve_time, my_time,
                umf_symbolic, umf_time_fac, umf_solve_time, umf_time);
        fclose(res_file);
    }
    std::cout << std::scientific << std::setprecision(1)
        << "my_time = " << my_time << " umf_time = " << umf_time
        << " umf_solv_t = " << umf_solve_time << " ratio = "
        << my_time / umf_time << std::endl;
#endif  // writing to a file

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FREE_ALL_AND_RETURN (PARU_SUCCESS) ;
}

