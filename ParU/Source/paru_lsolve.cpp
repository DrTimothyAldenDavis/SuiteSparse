////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// ParU_Lsolve //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  In this file I have lsolve x = L\x
 *
 *
 ********    The final result is something like this (nf = 4)
 *     col1    col2
 *      ___________________________________________
 *     |\       |                                 |       c
 *     |*\      |         U1                      |       c
 *     |****\   |                                 |       c  DTRSV on here
 *     |*DTRSV*\|_________________________________|       c
 *     |******* |\         |                      |       c
 *     |        |****\     |       U2             |       x
 *     |   LU1  |*DTRSV**\ |______________________|       x DGEMV updates
 *     |        |          |**\         |         |       x   here
 *     |        | LU2      |*****\      |   U3    |       x
 *     | DGEMV  |          |*DTRSV**\   |_________|       x
 *     |        |          |   LU3      |* LU4    |       x
 *     |        | DGEMV    |  DGEMV     |****     |       x
 *     |        |          |            |DTRSV*   |       x
 *     |________|__________|____________|_________|       x
 *
 *     This function just goes through LUs in the data structure and does a
 *     TRSV on triangular part
 *     Then does DGEMV on the rest for 0 to nf
 *
 *           BLAS_DTRSV  is used here but I do not use BLAS_DGEMV explicitly
 *           while it needs space for each thread doing this computation.
 *           I guess using this way can have a good performance.
 *
 * @author Aznaveh
 * */
#include "paru_internal.hpp"

ParU_Ret ParU_Lsolve(ParU_Symbolic *Sym, ParU_Numeric *Num, double *x, ParU_Control *Control)
{
    DEBUGLEVEL(0);
    if (!x) return PARU_INVALID;
    PARU_DEFINE_PRLEVEL;
    int64_t nf = Sym->nf;
    int64_t blas_ok = TRUE;
#ifndef NDEBUG
    int64_t m = Sym->m;
    PRLEVEL(1, ("%%inside lsolve x is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, (" %.2lf, ", x[k]));
    }
    PRLEVEL(PR, (" \n"));
#endif
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    int64_t n1 = Sym->n1;   // row+col singletons
    int64_t *Ps = Num->Ps;  // row permutation S->LU

    PRLEVEL(1, ("%%Working on singletons if any\n%%"));
    // singletons
    int64_t rs1 = Sym->rs1;
    if (rs1 > 0)
    {
        int64_t cs1 = Sym->cs1;

        for (int64_t j = cs1; j < n1; j++)
        {
            PRLEVEL(PR, ("j = " LD "\n", j));
            int64_t *Slp = Sym->lstons.Slp;
            int64_t *Sli = Sym->lstons.Sli;
            double *Slx = Num->Slx;
            ASSERT(Sli != NULL && Slx != NULL && Slp != NULL);
            int64_t diag = Slp[j - cs1];
            PRLEVEL(PR, (" x[" LD "]=%.2lf Slx[" LD "]=%.2lf\n", j, x[j], diag,
                         Slx[diag]));
            x[j] /= Slx[diag];
            PRLEVEL(PR, (" After x[" LD "]=%.2lf \n", j, x[j]));

            for (int64_t p = Slp[j - cs1] + 1; p < Slp[j - cs1 + 1]; p++)
            {
                int64_t r = Sli[p] < n1 ? Sli[p] : Ps[Sli[p] - n1] + n1;
                PRLEVEL(PR, (" r=" LD "\n", r));
                x[r] -= Slx[p] * x[j];
                PRLEVEL(PR, ("A x[" LD "]=%.2lf\n", Sli[p], x[Sli[p]]));
            }
        }
    }
#ifndef NDEBUG
    PRLEVEL(PR, ("%%lsove singletons finished and  x is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
#endif

    BLAS_set_num_threads(control_nthreads (Control)) ;

    // gather scatter space for dgemm
    double *work = static_cast<double*>(paru_alloc((Num->max_row_count), sizeof(double)));
    if (work == NULL)
    {
        PRLEVEL(1, ("ParU: out of memory lsolve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    ParU_Factors *LUs = Num->partial_LUs;
    int64_t *Super = Sym->Super;

    for (int64_t f = 0; f < nf; f++)
    {
        int64_t rowCount = Num->frowCount[f];
        int64_t *frowList = Num->frowList[f];
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;
        double *A = LUs[f].p;
        double *X = x + col1 + n1;

        PRLEVEL(2, ("%% Working on DTRSV\n"));
        SUITESPARSE_BLAS_dtrsv("L", "N", "U", fp, A, rowCount, X, 1, blas_ok);
        PRLEVEL(2, ("%% DTRSV is just finished\n"));

#ifndef NDEBUG
        PR = 2;
        PRLEVEL(PR, ("%% LUs:\n%%"));
        for (int64_t r = 0; r < rowCount; r++)
        {
            PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
            for (int64_t c = col1; c < col2; c++)
                PRLEVEL(PR, (" %2.5lf\t", A[(c - col1) * rowCount + r]));
            PRLEVEL(PR, ("\n"));
        }

        PRLEVEL(PR, ("%% lda = " LD "\n%%", rowCount));
        PRLEVEL(PR, ("%% during lsolve x [" LD "-" LD ")is:\n%%", col1, col2));
        // for (int64_t k = col1; k < col2; k++)
        int64_t m = Sym->m;
        for (int64_t k = 0; k < m; k++)
        {
            PRLEVEL(PR, (" %.2lf, ", x[k]));
        }
        PRLEVEL(PR, (" \n"));
#endif

        if (rowCount > fp)
        {
            PRLEVEL(2, ("%% lsolve: Working on DGEMV\n%%"));
            PRLEVEL(2, ("fp=" LD "  rowCount=" LD "\n", fp, rowCount));
            double alpha = 1;
            double beta = 0;
            SUITESPARSE_BLAS_dgemv("N", rowCount - fp, fp, &alpha, A + fp,
                                   rowCount, x + n1 + col1, 1, &beta, work, 1,
                                   blas_ok);
        }

        // don't use parallel loop if using dgemv
        // pragma omp parallel for
        for (int64_t i = fp; i < rowCount; i++)
        {
            // alternative to dgemv; do not need work if using this
            // computing the inner product
            // double i_prod = 0.0;  // inner product
            // for (int64_t j = col1; j < col2; j++)
            //{
            //    i_prod += A[(j - col1) * rowCount + i] * x[j + n1];
            //}

            double i_prod = work[i - fp];
            int64_t r = Ps[frowList[i]] + n1;
            PRLEVEL(2, ("i_prod[" LD "]=%lf  work=%lf r=" LD "\n", i, i_prod,
                        work[i - fp], r));
            x[r] -= i_prod;
        }
    }
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%% lsolve took %1.1lf\n", time));
#endif
#ifndef NDEBUG
    PRLEVEL(1, ("%%after lsolve x is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
#endif
    paru_free(Num->max_row_count, sizeof(double), work);
    return (blas_ok ? PARU_SUCCESS : PARU_TOO_LARGE);
}

//////////////// ParU_Lsolve ///multiple right hand side mRHS///////////////////
ParU_Ret ParU_Lsolve( ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *X, int64_t nrhs,   // X is m-by-nrhs, where A is m-by-m
    ParU_Control *Control)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
    if (!X) return PARU_INVALID;
    int64_t m = Sym->m;
    int64_t nf = Sym->nf;
    int64_t blas_ok = TRUE;

#ifndef NDEBUG
    PR = 2;
    PRLEVEL(1, ("%% mRHS inside LSolve X is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, ("%%"));
        for (int64_t l = 0; l < nrhs; l++)
        {
            PRLEVEL(PR, (" %.2lf, ", X[l * m + k]));
        }
        PRLEVEL(PR, (" \n"));
    }
    PRLEVEL(PR, (" \n"));
#endif
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif

    int64_t n1 = Sym->n1;   // row+col singletons
    int64_t *Ps = Num->Ps;  // row permutation S->LU

    // singletons
    int64_t rs1 = Sym->rs1;
    if (rs1 > 0)
    {
        int64_t cs1 = Sym->cs1;

        for (int64_t j = cs1; j < n1; j++)
        {
            PRLEVEL(PR, ("j = " LD "\n", j));
            int64_t *Slp = Sym->lstons.Slp;
            int64_t *Sli = Sym->lstons.Sli;
            double *Slx = Num->Slx;
            ASSERT(Sli != NULL && Slx != NULL && Slp != NULL);
            int64_t diag = Slp[j - cs1];
            PRLEVEL(PR, (" X[" LD "]=%.2lf Slx[" LD "]=%.2lf\n", j, X[j * nrhs], diag,
                         Slx[diag]));
            // pragma omp simd
            for (int64_t l = 0; l < nrhs; l++)
            {
                X[l * m + j] /= Slx[diag];
            }
            PRLEVEL(PR, (" After X[" LD "]=%.2lf \n", j, X[j * nrhs]));

            for (int64_t p = Slp[j - cs1] + 1; p < Slp[j - cs1 + 1]; p++)
            {
                int64_t r = Sli[p] < n1 ? Sli[p] : Ps[Sli[p] - n1] + n1;
                PRLEVEL(PR, (" r=" LD "\n", r));
                // pragma omp simd
                for (int64_t l = 0; l < nrhs; l++)
                {
                    X[l * m + r] -= Slx[p] * X[l * m + j];
                }
                PRLEVEL(PR, ("A X[" LD "]=%.2lf\n", Sli[p], X[Sli[p]] * nrhs));
            }
        }
    }
#ifndef NDEBUG
    PRLEVEL(1, ("%% mRHS lsolve singletons finished and X is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, ("%%"));
        for (int64_t l = 0; l < nrhs; l++)
        {
            PRLEVEL(PR, (" %.2lf, ", X[l * m + k]));
        }
        PRLEVEL(PR, (" \n"));
    }
    PRLEVEL(PR, (" \n"));
#endif

    BLAS_set_num_threads(control_nthreads (Control)) ;

    // gather scatter space for dgemm
    double *work =
        static_cast<double*>(paru_alloc((Num->max_row_count * nrhs), sizeof(double)));
    if (work == NULL)
    {
        PRLEVEL(1, ("ParU: out of memory lsolve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    ParU_Factors *LUs = Num->partial_LUs;
    int64_t *Super = Sym->Super;

    for (int64_t f = 0; f < nf; f++)
    {
        int64_t rowCount = Num->frowCount[f];
        int64_t *frowList = Num->frowList[f];
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;
        double *A = LUs[f].p;
        PRLEVEL(2, ("%% mRHS Working on DTRSM f=" LD "\n", f));
        double alpha = 1;
        SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", fp, nrhs, &alpha, A, rowCount,
                               X + n1 + col1, m, blas_ok);
        PRLEVEL(2, ("%% mRHS DTRSM is just finished f=" LD "\n", f));
#ifndef NDEBUG
        PR = 2;
        PRLEVEL(PR, ("%% LUs:\n%%"));
        for (int64_t r = 0; r < rowCount; r++)
        {
            PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
            for (int64_t c = col1; c < col2; c++)
                PRLEVEL(PR, (" %2.5lf\t", A[(c - col1) * rowCount + r]));
            PRLEVEL(PR, ("\n"));
        }

        PRLEVEL(PR, ("%% lda = " LD "\n%%", rowCount));
        PR = 1;
        PRLEVEL(PR,
                ("%% during lsolve X f=" LD "[" LD "-" LD ")is:\n%%", f, col1, col2));
        for (int64_t k = 0; k < m; k++)
        {
            PRLEVEL(1, ("%%"));
            for (int64_t l = 0; l < nrhs; l++)
            {
                PRLEVEL(1, (" %.2lf, ", X[l * m + k]));
            }
            PRLEVEL(1, (" \n"));
        }
        PRLEVEL(1, (" \n"));
#endif

        if (rowCount > fp)
        {
            PRLEVEL(2, ("%% mRHS lsolve: Working on DGEMM\n%%"));
            PRLEVEL(2, ("fp=" LD "  rowCount=" LD "\n", fp, rowCount));
            double beta = 0;
            SUITESPARSE_BLAS_dgemm("N", "N", rowCount - fp, nrhs, fp, &alpha,
                                   A + fp, rowCount, X + n1 + col1, m, &beta,
                                   work, rowCount - fp, blas_ok);
        }

        // don't use parallel loop if using dgemm
        // pragma omp parallel for
        for (int64_t i = fp; i < rowCount; i++)
        {
            // alternative to dgemm; do not need work if using this
            // computing the inner product
            // double i_prod[nrhs] = {0.0};  // inner product
            // for (int64_t j = col1; j < col2; j++)
            //{
            //    for (int64_t l = 0; l < nrhs; l++)
            //        i_prod[l] += A[(j - col1) * rowCount + i] * X[l*m +j+n1];
            //}

            // double* i_prod = work+i-fp;
            int64_t r = Ps[frowList[i]] + n1;
            for (int64_t l = 0; l < nrhs; l++)
            {
                X[l * m + r] -= work[i - fp + l * (rowCount - fp)];
                // X[l*m+r] -= i_prod[l];
            }
        }
    }
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%% mRHS lsolve took %1.1lfs\n", time));
#endif
#ifndef NDEBUG
    PRLEVEL(1, ("%% after lsolve X is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < nrhs; l++)
        {
            PRLEVEL(1, (" %.2lf, ", X[l * m + k]));
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));
#endif
    paru_free(Num->max_row_count * nrhs, sizeof(double), work);
    return (blas_ok ? PARU_SUCCESS : PARU_TOO_LARGE);
}
