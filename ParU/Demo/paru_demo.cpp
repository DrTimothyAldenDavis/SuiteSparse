// ==========================================================================  /
// =======================  paru_demo.cpp  ==================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*
 * @brief    test to see how to call umfpack symbolic analysis
 *
 * @author Aznaveh
 * */
#include <iostream>
#include <iomanip>
#include <ios>
#include <cmath>
#include <omp.h>

#include "ParU.hpp"

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_Symbolic *Sym = NULL;

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
        exit(1);
    }

    if (mtype != CHOLMOD_SPARSE)
    {
        std::cout << "ParU: input matrix must be sparse" << std::endl;
        exit(1);
    }

    if (A->xtype != CHOLMOD_REAL)
    {
        std::cout << "ParU: input matrix must be real" << std::endl;
        exit(1);
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int ver[3];
    char date[128];
    ParU_Version(ver, date);
    std::cout << "================= ParU "
        << ver[0] << "." << ver[1] << "."  << ver[2]
        << " ================================== " << date << std::endl;

    double my_start_time = omp_get_wtime();

    ParU_Control Control;
    ParU_Ret info;

    Control.umfpack_ordering = UMFPACK_ORDERING_AMD;
    // Control.umfpack_strategy = UMFPACK_STRATEGY_UNSYMMETRIC;
    // Control.umfpack_strategy = UMFPACK_STRATEGY_SYMMETRIC;
    // Control.umfpack_default_singleton = 0;
    // Control.paru_max_threads = 6;
    Control.umfpack_ordering = UMFPACK_ORDERING_METIS_GUARD;
    std::cout << "\n--------- ParU_Analyze:\n";
    info = ParU_Analyze(A, &Sym, &Control);
    double my_time_analyze = omp_get_wtime() - my_start_time;
    if (info != PARU_SUCCESS)
    {
        cholmod_l_free_sparse(&A, cc);
        cholmod_l_finish(cc);
        return info;
    }
    std::cout << "In: " << Sym->m << "x" << Sym->n
        << " nnz = " << Sym->anz << std::endl;
    std::cout << std::scientific << std::setprecision(1)
        << "ParU: Symbolic factorization is done in " << my_time_analyze << "s!\n";
    ParU_Numeric *Num;
    std::cout << "\n--------- ParU_Factorize:" << std::endl;
    double my_start_time_fac = omp_get_wtime();
    info = ParU_Factorize(A, Sym, &Num, &Control);
    double my_time_fac = omp_get_wtime() - my_start_time_fac;
    if (info != PARU_SUCCESS)
    {
        std::cout << std::scientific << std::setprecision(1)
            << "ParU: factorization was NOT successful in " << my_time_fac
            << " seconds!\n";
        if (info == PARU_OUT_OF_MEMORY)
            std::cout << "Out of memory\n";
        if (info == PARU_INVALID)
            std::cout << "Invalid!\n";
        if (info == PARU_SINGULAR)
            std::cout << "Singular!\n";
        cholmod_l_free_sparse(&A, cc);
        cholmod_l_finish(cc);
        ParU_Freesym(&Sym, &Control);
        return info;
    }
    else
    {
        std::cout << std::scientific << std::setprecision(1)
            << "ParU: factorization was successful in " << my_time_fac
            << " seconds.\n";
    }

    //~~~~~~~~~~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int64_t m = Sym->m;
    double my_time, my_solve_time;
#if 1
    if (info == PARU_SUCCESS)
    {
        double *b = (double *)malloc(m * sizeof(double));
        double *xx = (double *)malloc(m * sizeof(double));
        for (int64_t i = 0; i < m; ++i) b[i] = i + 1;
        std::cout << "\n--------- ParU_Solve:\n";
        double my_solve_time_start = omp_get_wtime();
        info = ParU_Solve(Sym, Num, b, xx, &Control);
        if (info != PARU_SUCCESS)
        {
            std::cout << "ParU: Solve has a problem.\n";
            free(b);
            free(xx);
            cholmod_l_free_sparse(&A, cc);
            cholmod_l_finish(cc);
            ParU_Freesym(&Sym, &Control);
            return info;
        }
        my_solve_time = omp_get_wtime() - my_solve_time_start;
        my_time = omp_get_wtime() - my_start_time;
        std::cout << std::defaultfloat << std::setprecision(1)
            << "Solve time is " << my_solve_time << " seconds.\n";

        // printing out x
    #if 0
        std::cout << "x = [";
        std::cout << std::defaultfloat  << std::setprecision(22);
        for (int64_t i = 0; i < m; ++i)
            std::cout << " " << xx[i] << "', ";
        std::cout << "]\n";
    #endif

        double resid, anorm, xnorm;
        std::cout << "\n--------- ParU_Residual:\n";
        info = ParU_Residual(A, xx, b, m, resid, anorm, xnorm, &Control);
        if (info != PARU_SUCCESS)
        {
            std::cout << "ParU: Residual has a problem.\n";
            free(b);
            free(xx);
            cholmod_l_free_sparse(&A, cc);
            cholmod_l_finish(cc);
            ParU_Freesym(&Sym, &Control);
            return info;
        }
        double rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));

        std::cout << std::scientific << std::setprecision(2)
            << "Relative residual is |" << rresid << "|, anorm is " << anorm
            << ", xnorm is " << xnorm << " and rcond is " << Num->rcond << "."
            << std::endl;

        free(b);
        free(xx);
        const int64_t nrhs = 16;  // number of right handsides
        double *B = (double *)malloc(m * nrhs * sizeof(double));
        double *X = (double *)malloc(m * nrhs * sizeof(double));
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < nrhs; ++j)
                B[j * m + i] = (double)(i + j + 1);

        std::cout << "\n--------- ParU_Solve:\n";
        info = ParU_Solve(Sym, Num, nrhs, B, X, &Control);
        if (info != PARU_SUCCESS)
        {
            std::cout << "ParU: mRhs Solve has a problem.\n";
            free(B);
            free(X);
            cholmod_l_free_sparse(&A, cc);
            cholmod_l_finish(cc);
            ParU_Freesym(&Sym, &Control);
            return info;
        }
        std::cout << "\n--------- ParU_Residual:\n";
        info = ParU_Residual(A, X, B, m, nrhs, resid, anorm, xnorm, &Control);
        if (info != PARU_SUCCESS)
        {
            std::cout << "ParU: mRhs Residual has a problem.\n";
            free(B);
            free(X);
            cholmod_l_free_sparse(&A, cc);
            cholmod_l_finish(cc);
            ParU_Freesym(&Sym, &Control);
            return info;
        }
        rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));

        std::cout << std::scientific << std::setprecision(2)
            << "Multiple right hand side: relative residual is |"
            << rresid << "|." << std::endl;

        free(B);
        free(X);
    }
#endif

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int64_t max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    //~~~~~~~~~~~~~~~~~~~Calling umfpack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    double umf_time = 0;

#if 1
    double umf_start_time = omp_get_wtime();
    double status,           // Info [UMFPACK_STATUS]
        Info[UMFPACK_INFO],  // Contains statistics about the symbolic analysis

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
    void *Symbolic, *Numeric;  // Output argument in umf_dl_symbolc;

    status =
        umfpack_dl_symbolic(n, n, Ap, Ai, Ax, &Symbolic, umf_Control, Info);
    // umf_Control[UMFPACK_PRL] = 0;
    // umfpack_dl_report_info(umf_Control, Info);
    if (status < 0)
    {
        umfpack_dl_report_info(umf_Control, Info);
        umfpack_dl_report_status(umf_Control, status);
        std::cout << "umfpack_dl_symbolic failed\n";
        exit(0);
    }
    double umf_symbolic = omp_get_wtime() - umf_start_time;
    double umf_fac_start = omp_get_wtime();
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
    }

    double umf_time_fac = omp_get_wtime() - umf_fac_start;

    double *b = (double *)malloc(m * sizeof(double));
    double *x = (double *)malloc(m * sizeof(double));
    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;

    double solve_start = omp_get_wtime();
    status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, umf_Control,
                              Info);
    double umf_solve_time = omp_get_wtime() - solve_start;
    umf_time = omp_get_wtime() - umf_start_time;
    double umf_resid, umf_anorm, umf_xnorm;
    info = ParU_Residual(A, x, b, m, umf_resid, umf_anorm, umf_xnorm, &Control);
    double umf_rresid = (umf_anorm == 0 || umf_xnorm == 0 ) 
        ? 0 : (umf_resid/(umf_anorm*umf_xnorm));
    std::cout << std::scientific << std::setprecision(2)
        << "UMFPACK relative residual is |" << umf_rresid << "|, anorm is "
        << umf_anorm << ", xnorm is " << umf_xnorm << " and rcond is "
        << Num->rcond << "." << std::endl;

    free(x);
    free(b);

    umfpack_dl_free_symbolic(&Symbolic);
    umfpack_dl_free_numeric(&Numeric);
#endif  // calling umfpack

    // Writing results to a file
#if 0
    if (info == PARU_SUCCESS)
    {
        FILE *res_file;
        char res_name[] = "../build/Res/res.txt";
        res_file = fopen(res_name, "a");
        if (res_file == NULL)
        {
            std::cout << "Par: error in making " << res_name << " to write the results!\n";
        }
        fprintf(res_file, "%" PRId64" %" PRId64 " %lf %lf %lf %lf %lf %lf %lf %lf\n",
                Sym->m, Sym->anz, my_time_analyze, my_time_fac, my_solve_time, my_time,
                umf_symbolic, umf_time_fac, umf_solve_time, umf_time);
        fclose(res_file);
    }
    std::cout << std::scientific << std::setprecision(1)
        << "my_time = " << my_time << " umf_time = " << umf_time
        << " umf_solv_t = " << umf_solve_time << " ratio = "
        << my_time / umf_time << std::endl;
#endif  // writing to a file

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ParU_Freenum(&Num, &Control);
    ParU_Freesym(&Sym, &Control);

    cholmod_l_free_sparse(&A, cc);
    cholmod_l_finish(cc);
}
