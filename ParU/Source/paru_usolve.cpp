////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// ParU_Usolve //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  In this file I have usolve x = U\x
 *
 *
 ********       The final result is something like this (nf = 4)
 *         ___________________________________________
 *        |\*******|                                 |       x
 *        | \*DTRSV|         U1   DGEMV              |       x
 *        |    \***|                                 |       x
 *        |       \|_________________________________|       x
 *        |        |\**DTRSV**|                      |       x
 *        |        |    \*****|       U2    DGEMV    |       x
 *        |   LU1  |        \*|______________________|       o
 *        |        |          |  \**DTRSV**|  DGEMV  |       o
 *        |        | LU2      |     \******|   U3    |       o
 *        |        |          |        \***|_________|       o DGEMV updates
 *        |        |          |   LU3      |*********|       c  here
 *        |        |          |            |  *******|       c   DTRSV on here
 *        |        |          |            |LU4  ****|       c
 *        |________|__________|____________|________*|       c
 *
 *        This function just goes through LUs and US in the data structure and
 *        does a TRSV on triangular part  Then does DGEMV on the rest
 *       for nf down to 0
 *
 *              BLAS_DTRSV  is used here but I do not use BLAS_DGEMV explicitly
 *              while it needs space for each thread doing this computation.
 *              I guess using this way can have a good performance.
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"

ParU_Ret ParU_Usolve(ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *x, ParU_Control *Control)
{
    DEBUGLEVEL(0);
    // check if input is read
    if (!x) return PARU_INVALID;
    int64_t blas_ok = TRUE;
    PARU_DEFINE_PRLEVEL;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    int64_t nf = Sym->nf;

    int64_t n1 = Sym->n1;   // row+col singletons
    int64_t *Ps = Num->Ps;  // row permutation

    ParU_Factors *LUs = Num->partial_LUs;
    ParU_Factors *Us = Num->partial_Us;
    int64_t *Super = Sym->Super;

    BLAS_set_num_threads(control_nthreads (Control)) ;
    double *work = static_cast<double*>(paru_alloc((Num->max_col_count), sizeof(double)));
    if (work == NULL)
    {
        PRLEVEL(1, ("ParU: out of memory usolve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    for (int64_t f = nf - 1; f >= 0; --f)
    {
        int64_t *frowList = Num->frowList[f];
        int64_t *fcolList = Num->fcolList[f];
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;
        int64_t colCount = Num->fcolCount[f];

        // do dgemv
        // performed on Us
        // I am not calling BLAS_DGEMV while the column permutation is different

        double *A2 = Us[f].p;
        if (A2 != NULL)
        {
            PRLEVEL(2, ("%% usolve: Working on DGEMV\n%%"));

            double *xg = &work[0] + fp;         // size Xg is colCount
            for (int64_t j = 0; j < colCount; j++)  // gathering x in Xg
            {
                xg[j] = x[fcolList[j] + n1];
            }
            double alpha = 1;
            double beta = 0;
            SUITESPARSE_BLAS_dgemv("N", fp, colCount, &alpha, A2, fp, xg, 1,
                                   &beta, work, 1, blas_ok);

            for (int64_t i = 0; i < fp; i++)  // scattering the back in to x
            {
                int64_t r = Ps[frowList[i]] + n1;
                x[r] -= work[i];
            }

            // pragma omp parallel for
            // for (int64_t i = 0; i < fp; i++)
            //{
            //    PRLEVEL(2, ("%% Usolve: Working on DGEMV\n"));
            //    // computing the inner product
            //    double i_prod = 0.0;  // innter product
            //    for (int64_t j = 0; j < colCount; j++)
            //    {
            //        i_prod += A2[fp * j + i] * x[fcolList[j] + n1];
            //    }
            //    int64_t r = Ps[frowList[i]] + n1;
            //    PRLEVEL(2, ("i_prod[" LD "]=%lf  r=" LD "\n", i, i_prod,  r));
            //    x[r] -= i_prod;
            //}
        }

        int64_t rowCount = Num->frowCount[f];

        double *A1 = LUs[f].p;
        PRLEVEL(2, ("%% Usolve: Working on DTRSV\n"));
        SUITESPARSE_BLAS_dtrsv("U", "N", "N", fp, A1, rowCount, x + col1 + n1,
                               1, blas_ok);
        PRLEVEL(2, ("%% DTRSV is just finished\n"));
    }

#ifndef NDEBUG
    int64_t m = Sym->m;
    PRLEVEL(1, ("%% before singleton x is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
    PR = 1;
#endif
    int64_t cs1 = Sym->cs1;
    if (cs1 > 0)
    {
        for (int64_t i = cs1 - 1; i >= 0; i--)
        {
            PRLEVEL(PR, ("i = " LD "\n", i));
            int64_t *Sup = Sym->ustons.Sup;
            int64_t *Suj = Sym->ustons.Suj;
            double *Sux = Num->Sux;
            ASSERT(Suj != NULL && Sux != NULL && Sup != NULL);
            PRLEVEL(PR, (" Before computation x[" LD "]=%.2lf \n", i, x[i]))
            for (int64_t p = Sup[i] + 1; p < Sup[i + 1]; p++)
            {
                int64_t r = Suj[p];
                PRLEVEL(PR, (" r=" LD "\n", r));
                x[i] -= Sux[p] * x[r];
                PRLEVEL(PR, ("Suj[" LD "]=" LD "\n", p, Suj[p]));
                PRLEVEL(PR, (" x[" LD "]=%.2lf x[" LD "]=%.2lf\n", r, x[r], i, x[i]));
            }
            int64_t diag = Sup[i];
            x[i] /= Sux[diag];
            PRLEVEL(PR, (" After computation x[" LD "]=%.2lf \n", i, x[i]))
            PRLEVEL(PR, ("\n"));
        }
    }

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%% usolve took %1.1lf\n", time));
#endif
#ifndef NDEBUG
    PRLEVEL(1, ("%%after usolve x is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
#endif
    paru_free(Num->max_col_count, sizeof(double), work);
    return (blas_ok ? PARU_SUCCESS : PARU_TOO_LARGE);
}

//////////////// ParU_Usolve ///multiple right hand side mRHS///////////////////
ParU_Ret ParU_Usolve(ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *X, int64_t nrhs, ParU_Control *Control)
{
    DEBUGLEVEL(0);
    // check if input is read
    if (!X) return PARU_INVALID;
    int64_t blas_ok = TRUE;
    PARU_DEFINE_PRLEVEL;
    int64_t m = Sym->m;
    int64_t nf = Sym->nf;
#ifndef NDEBUG
    PRLEVEL(1, ("%% mRHS inside USolve X is:\n"));
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
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    int64_t n1 = Sym->n1;   // row+col singletons
    int64_t *Ps = Num->Ps;  // row permutation

    ParU_Factors *LUs = Num->partial_LUs;
    ParU_Factors *Us = Num->partial_Us;
    int64_t *Super = Sym->Super;

    BLAS_set_num_threads(control_nthreads (Control)) ;
    double *work =
        static_cast<double*>(paru_alloc((Num->max_col_count * nrhs), sizeof(double)));
    if (work == NULL)
    {
        PRLEVEL(1, ("ParU: out of memory Usolve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    for (int64_t f = nf - 1; f >= 0; --f)
    {
        int64_t *frowList = Num->frowList[f];
        int64_t *fcolList = Num->fcolList[f];
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;
        int64_t colCount = Num->fcolCount[f];

        // do dgemm
        // performed on Us

        double *A2 = Us[f].p;
        if (A2 != NULL)
        {
            PRLEVEL(2, ("%% mRHS usolve: Working on DGEMM f=" LD "\n%%", f));
            double *Xg = &work[0] + fp * nrhs;     // size Xg is colCount x nrhs
            for (int64_t j = 0; j < colCount; j++)  // gathering X in Xg
            {
                for (int64_t l = 0; l < nrhs; l++)
                {
                    Xg[l * colCount + j] = X[l * m + fcolList[j] + n1];
                }
            }

            double alpha = 1;
            double beta = 0;
            SUITESPARSE_BLAS_dgemm("N", "N", fp, nrhs, colCount, &alpha, A2, fp,
                                   Xg, colCount, &beta, work, fp, blas_ok);

            for (int64_t i = 0; i < fp; i++)  // scattering the back in to X
            {
                int64_t r = Ps[frowList[i]] + n1;
                for (int64_t l = 0; l < nrhs; l++)
                {
                    X[l * m + r] -= work[l * fp + i];
                }
            }

            // algternative way for dgemm
            // pragma omp parallel for schedule(static)
            // for (int64_t i = 0; i < fp; i++)
            //{
            //    // computing the inner product
            //    double i_prod[nrhs] = {0.0};  // inner product
            //    for (int64_t j = 0; j < colCount; j++)
            //    {
            //        for (int64_t l = 0; l < nrhs; l++)
            //            i_prod[l] += A2[fp * j + i] * X[l*m + fcolList[j] +
            //            n1];
            //    }
            //    int64_t r = Ps[frowList[i]] + n1;
            //    for (int64_t l = 0; l < nrhs; l++)
            //    {
            //        PRLEVEL(2, ("i_prod[" LD "]=%lf  r=" LD "\nrhs", i, i_prod[i], r));
            //        X[l*m+r] -= i_prod[l];
            //    }
            //}
        }

        int64_t rowCount = Num->frowCount[f];

        PRLEVEL(2, ("%% mRHS Usolve: Working on DTRSM\n"));
        double *A1 = LUs[f].p;
        double alpha = 1;
        SUITESPARSE_BLAS_dtrsm("L", "U", "N", "N", fp, nrhs, &alpha, A1, rowCount,
                               X + n1 + col1, m, blas_ok);
        PRLEVEL(2, ("%% mRHS DTRSM is just finished\n"));
    }

    PRLEVEL(1, ("%% mRHS Usolve working on singletons \n"));
    int64_t cs1 = Sym->cs1;
    if (cs1 > 0)
    {
        for (int64_t i = cs1 - 1; i >= 0; i--)
        {
            PRLEVEL(PR, ("i = " LD "\n", i));
            int64_t *Sup = Sym->ustons.Sup;
            int64_t *Suj = Sym->ustons.Suj;
            double *Sux = Num->Sux;
            ASSERT(Suj != NULL && Sux != NULL && Sup != NULL);
            for (int64_t p = Sup[i] + 1; p < Sup[i + 1]; p++)
            {
                int64_t r = Suj[p];
                PRLEVEL(PR, (" r=" LD "\n", r));
#pragma omp simd
                for (int64_t l = 0; l < nrhs; l++)
                {
                    X[l * m + i] -= Sux[p] * X[l * m + r];
                }
            }
            int64_t diag = Sup[i];
#pragma omp simd
            for (int64_t l = 0; l < nrhs; l++)
            {
                X[l * m + i] /= Sux[diag];
            }
        }
    }
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%% mRHS usolve took %1.1lfs\n", time));
#endif
#ifndef NDEBUG
    PRLEVEL(1, ("%%after usolve X is:\n"));
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
    paru_free(Num->max_col_count * nrhs, sizeof(double), work);
    return (blas_ok ? PARU_SUCCESS : PARU_TOO_LARGE);
}
