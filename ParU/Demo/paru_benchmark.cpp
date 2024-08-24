// ==========================================================================  /
// =======================  paru_benchmark.cpp ==============================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief    benchmark ParU and UMFPACK
 *
 * @author Aznaveh
 * */

#include <iostream>
#include <iomanip>
#include <ios>
#include <cmath>

#include "ParU.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
#error "OpenMP not available!"
#endif

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
    if (argc > 1 && fp != NULL) fclose (fp) ;   \
    fp = NULL ;                                 \
    return (info) ;                             \
}

static int compar (const void *p1, const void *p2)
{
    double x1 = *((double *) p1) ;
    double x2 = *((double *) p2) ;
    return (x1 < x2 ? -1 : ((x1 > x2) ? 1 : 0)) ;
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
    FILE *fp = stdin ;

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);

    if (argc > 1)
    {
        std::cout << "Matrix: " << argv [1] << std::endl ;
        fp = fopen (argv [1], "r") ;
    }
    else
    {
        std::cout << "Matrix: stdin" << std::endl ;
    }

    // A = mread (fp) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(fp, 1, &mtype, cc);
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

    int64_t max_nthreads ;
    #ifdef _OPENMP
    max_nthreads = omp_get_max_threads ( ) ;
    #else
    max_ntreads = 1 ;
    #endif
    std::cout << "max # threads: " << max_nthreads << std::endl ;

    // allocate workspace
    int64_t n, anz ;
    n = A->nrow ;
    const int64_t nrhs = 16;  // number of right handsides
    b = (double *)malloc(n * sizeof(double));
    xx = (double *)malloc(n * sizeof(double));
    B = (double *)malloc(n * nrhs * sizeof(double));
    X = (double *)malloc(n * nrhs * sizeof(double));
    x = (double *)malloc(n * sizeof(double));
    double rcond ;

    #define NTRIALS 5
    int middle = NTRIALS / 2 ;

    for (int ord = 0 ; ord <= 1 ; ord++)
    {
        int ordering = (ord == 0) ? PARU_ORDERING_AMD :
            PARU_ORDERING_METIS_GUARD ;
        printf ("===== ParU ordering: %d\n", ordering) ;
        ParU_Set (PARU_CONTROL_ORDERING, ordering, Control) ;
        for (int nthreads = max_nthreads ; nthreads > 0 ; nthreads = nthreads/2)
        {
            printf ("# threads: %d\n", nthreads) ;
            ParU_Set (PARU_CONTROL_MAX_THREADS, (int64_t) nthreads, Control) ;
            int64_t nthreads2 = 0 ;
            ParU_Get (PARU_CONTROL_NUM_THREADS, &nthreads2, Control) ;
            if (nthreads != (int32_t) nthreads2)
            {
                std::cout << "ParU: invalid # of threads!" << std::endl;
                FREE_ALL_AND_RETURN (PARU_INVALID) ;
            }

            double ParU_sym_times [NTRIALS] ;
            double ParU_num_times [NTRIALS] ;
            double ParU_sol_times [NTRIALS] ;
            for (int trial = 0 ; trial < NTRIALS  ; trial++)
            {
                printf ("Trial: %d\n", trial) ;

                double my_start_time = SUITESPARSE_TIME ;
                info = ParU_Analyze(A, &Sym, Control);
                double my_time_analyze = SUITESPARSE_TIME - my_start_time;
                if (info != PARU_SUCCESS)
                {
                    std::cout << "ParU: analyze failed" << std::endl;
                    FREE_ALL_AND_RETURN (info) ;
                }

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

                if (trial == 0)
                {
                    std::cout << "n: " << n << " anz: " << anz << std::endl ;
                }

                double my_start_time_fac = SUITESPARSE_TIME;
                info = ParU_Factorize(A, Sym, &Num, Control);
                double my_time_fac = SUITESPARSE_TIME - my_start_time_fac;
                if (info != PARU_SUCCESS)
                {
                    std::cout << std::scientific << std::setprecision(6)
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

                info = ParU_Get (Sym, Num, PARU_GET_RCOND_ESTIMATE, &rcond,
                    Control) ;
                if (info != PARU_SUCCESS)
                {
                    std::cout << "ParU: stats failed" << std::endl;
                    FREE_ALL_AND_RETURN (info) ;
                }

                //~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                double my_time, my_solve_time;

                for (int64_t i = 0; i < n; ++i) b[i] = i + 1;
                double my_solve_time_start = SUITESPARSE_TIME;
                info = ParU_Solve(Sym, Num, b, xx, Control);
                if (info != PARU_SUCCESS)
                {
                    std::cout << "ParU: solve failed" << std::endl;
                    FREE_ALL_AND_RETURN (info) ;
                }
                my_solve_time = SUITESPARSE_TIME - my_solve_time_start;
                my_time = SUITESPARSE_TIME - my_start_time;

                double resid, anorm, xnorm;
                info = ParU_Residual(A, xx, b, resid, anorm, xnorm, Control);
                if (info != PARU_SUCCESS)
                {
                    std::cout << "ParU: resid failed" << std::endl;
                    FREE_ALL_AND_RETURN (info) ;
                }
                double rresid = (anorm == 0 || xnorm == 0 ) ? 0 :
                    (resid/(anorm*xnorm));

                for (int64_t i = 0; i < n; ++i)
                {
                    for (int64_t j = 0; j < nrhs; ++j)
                    {
                        B[j * n + i] = (double)(i + j + 1);
                    }
                }

                my_solve_time_start = SUITESPARSE_TIME;
                info = ParU_Solve(Sym, Num, nrhs, B, X, Control);
                if (info != PARU_SUCCESS)
                {
                    std::cout << "ParU: solve failed" << std::endl;
                    FREE_ALL_AND_RETURN (info) ;
                }
                double my_solve_time2 = SUITESPARSE_TIME - my_solve_time_start;

                info = ParU_Residual(A, X, B, nrhs, resid, anorm, xnorm,
                    Control);
                if (info != PARU_SUCCESS)
                {
                    std::cout << "ParU: solve failed" << std::endl;
                    FREE_ALL_AND_RETURN (info) ;
                }
                double rresid2 = (anorm == 0 || xnorm == 0 ) ? 0 :
                    (resid/(anorm*xnorm));

                if (trial == 0)
                {
                    std::cout << std::scientific << std::setprecision(6)
                        << "Relative residual: " << rresid << " rcond: "
                        << rcond << std::endl;
                    std::cout << std::scientific << std::setprecision(6)
                        << "Multiple right hand side: relative residual is |"
                        << rresid2 << "|." << std::endl;
                }

                std::cout << std::scientific << std::setprecision(6)
                    << "ParU: time: sym: " << my_time_analyze
                    << " num: " << my_time_fac
                    << " solve (1 rhs): " << my_solve_time
                    << " solve (16 rhs): " << my_solve_time2 << std::endl ;

                ParU_sym_times [trial] = my_time_analyze ;
                ParU_num_times [trial] = my_time_fac ;
                ParU_sol_times [trial] = my_solve_time ;

                ParU_FreeNumeric(&Num, Control);
                ParU_FreeSymbolic(&Sym, Control);
            }

            qsort (ParU_sym_times, NTRIALS, sizeof (double), compar) ;
            qsort (ParU_num_times, NTRIALS, sizeof (double), compar) ;
            qsort (ParU_sol_times, NTRIALS, sizeof (double), compar) ;

            std::cout << std::scientific << std::setprecision(6)
                << "\nParU ordering " << ordering
                << " threads " << nthreads
                << " median sym: " << ParU_sym_times [middle]
                << " num: " << ParU_num_times [middle]
                << " sol: " << ParU_sol_times [middle]
                << " total: " << ParU_sym_times [middle] +
                ParU_num_times [middle] + ParU_sol_times [middle]
                << std::endl << std::endl ;
        }
    }

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //~~~~~~~~~~~~~~~~~~~Calling umfpack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    double umf_time = 0;
    double status,           // Info [UMFPACK_STATUS]
        Info[UMFPACK_INFO],  // Contains statistics about the symbolic analysis
        umf_Control[UMFPACK_CONTROL];  // it is set in umfpack_dl_defaults and
    // is used in umfpack_dl_symbolic; if
    // passed NULL it will use the defaults
    umfpack_dl_defaults(umf_Control);

    int64_t *Ap = (int64_t *)A->p;
    int64_t *Ai = (int64_t *)A->i;
    double *Ax = (double *)A->x;

    for (int ord = 0 ; ord <= 1 ; ord++)
    {
        int ordering = (ord == 0) ?  UMFPACK_ORDERING_AMD :
            UMFPACK_ORDERING_METIS_GUARD ;
        printf ("\n===== UMFPACK ordering: %d\n", ordering) ;
        for (int nthreads = max_nthreads ; nthreads > 0 ; nthreads = nthreads/2)
        {
            printf ("# threads: %d\n", nthreads) ;
            #ifdef _OPENMP
            omp_set_num_threads (nthreads) ;
            #endif

            double UMF_sym_times [NTRIALS] ;
            double UMF_num_times [NTRIALS] ;
            double UMF_sol_times [NTRIALS] ;
            umf_Control[UMFPACK_ORDERING] = ordering ;
            for (int trial = 0 ; trial < NTRIALS ; trial++)
            {
                printf ("Trial: %d\n", trial) ;

                double umf_start_time = SUITESPARSE_TIME;
                status = umfpack_dl_symbolic(n, n, Ap, Ai, Ax, &Symbolic,
                    umf_Control, Info);
                if (status < 0)
                {
                    umfpack_dl_report_info(umf_Control, Info);
                    umfpack_dl_report_status(umf_Control, status);
                    std::cout << "umfpack_dl_symbolic failed\n";
                    FREE_ALL_AND_RETURN (PARU_INVALID) ;
                }
                double umf_symbolic = SUITESPARSE_TIME - umf_start_time;
                double umf_fac_start = SUITESPARSE_TIME;
                status = umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric,
                    umf_Control, Info);
                if (status < 0)
                {
                    umfpack_dl_report_info(umf_Control, Info);
                    umfpack_dl_report_status(umf_Control, status);
                    std::cout << "umfpack_dl_numeric failed\n";
                    FREE_ALL_AND_RETURN (PARU_INVALID) ;
                }

                double umf_time_fac = SUITESPARSE_TIME - umf_fac_start;

                for (int64_t i = 0; i < n; ++i) b[i] = i + 1;

                double solve_start = SUITESPARSE_TIME;
                status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x, b,
                    Numeric, umf_Control, Info);
                double umf_solve_time = SUITESPARSE_TIME - solve_start;
                umf_time = SUITESPARSE_TIME - umf_start_time;

                double umf_resid, umf_anorm, umf_xnorm;
                info = ParU_Residual(A, x, b, umf_resid, umf_anorm, umf_xnorm,
                    Control);
                double umf_rresid = (umf_anorm == 0 || umf_xnorm == 0 )
                    ? 0 : (umf_resid/(umf_anorm*umf_xnorm));
                if (trial == 0)
                {
                    std::cout << std::scientific << std::setprecision(6)
                        << "UMFPACK relative residual: " << umf_rresid
                        << std::endl;
                }

                std::cout << std::scientific << std::setprecision(6)
                    << "UMFPACK time: sym " << umf_symbolic
                    << " num: " << umf_time_fac
                    << " solve: " << umf_solve_time << std::endl ;

                UMF_sym_times [trial] = umf_symbolic ;
                UMF_num_times [trial] = umf_time_fac ;
                UMF_sol_times [trial] = umf_solve_time ;

                umfpack_dl_free_symbolic(&Symbolic);
                umfpack_dl_free_numeric(&Numeric);
            }

            qsort (UMF_sym_times, NTRIALS, sizeof (double), compar) ;
            qsort (UMF_num_times, NTRIALS, sizeof (double), compar) ;
            qsort (UMF_sol_times, NTRIALS, sizeof (double), compar) ;

            std::cout << std::scientific << std::setprecision(6)
                << "\nUMF  ordering " << ordering
                << " threads " << nthreads
                << " median sym: " << UMF_sym_times [middle]
                << " num: " << UMF_num_times [middle]
                << " sol: " << UMF_sol_times [middle]
                << " total: " << UMF_sym_times [middle] +
                UMF_num_times [middle] + UMF_sol_times [middle]
                << std::endl << std::endl ;
        }
    }

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FREE_ALL_AND_RETURN (PARU_SUCCESS) ;
}

