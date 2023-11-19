////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_tasked_dgemm //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief      a wrapper around BLAS_DGEMM for tasked base dgemmed
 *
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"

int64_t paru_tasked_dgemm(int64_t f, int64_t M, int64_t N, int64_t K,
    double *A, int64_t lda, double *B, int64_t ldb, double beta, double *C,
    int64_t ldc, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    // alpha is always -1  in my DGEMMs
    double alpha = -1;
    int64_t naft;

    int64_t blas_ok = TRUE;
    ParU_Control *Control = Num->Control;
    int64_t trivial = Control->trivial;
    int64_t L = Control->worthwhile_dgemm;
#pragma omp atomic read
    naft = Work->naft;
    const int32_t max_threads = Control->paru_max_threads;
    if (naft == 1)
    {
        BLAS_set_num_threads(max_threads);
    }
    else
    {
        BLAS_set_num_threads(1);
    }
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    if (M < trivial && N < trivial && K < trivial)
    // if(0)
    {
        PRLEVEL(1, ("%% SMALL DGEMM (" LD "," LD "," LD ") in " LD "\n", M, N, K, f));
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
            {
                if (beta == 0) C[i + j * ldc] = 0;
                for (int64_t k = 0; k < K; k++)
                {
                    C[i + j * ldc] -= A[i + k * lda] * B[k + j * ldb];
                }
            }
    }
    else if ((M < L && N < L) || (naft == 1) || (naft >= max_threads))
    // if small or no other tasks competing or there are lots of other tasks
    // if(1)
    {
#ifndef NDEBUG
        if (naft == 1)
        {
            PRLEVEL(1, ("%% A max_threads DGEMM (" LD "x" LD ") in " LD "\n", M, N, f));
        }
        else if (M < L && N < L)
        {
            PRLEVEL(1, ("%% Single call DGEMM (" LD "x" LD ") in " LD "\n", M, N, f));
        }
#endif
        SUITESPARSE_BLAS_dgemm("N", "N", M, N, K, &alpha, A, lda, B, ldb, &beta,
                               C, ldc, blas_ok);
    }
    else
    {
#if ( defined ( BLAS_Intel10_64ilp ) || defined ( BLAS_Intel10_64lp ) )
        int my_share = max_threads / naft;
        if (my_share == 0) my_share = 1;
        PRLEVEL(1, ("%% MKL local threads for DGEMM (" LD "x" LD ") in " LD " [[%d]]\n",
                    M, N, f, my_share));
        // using my share of threads
        mkl_set_num_threads_local(my_share);
        // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, -1,
        // A,
        //            lda, B, ldb, beta, C, ldc);
        // BLAS_DGEMM (transa, transb,
        //   M, N, K, alpha, A, lda, ldb, beta, C, ldc, blas_ok);
        SUITESPARSE_BLAS_dgemm("N", "N", M, N, K, &alpha, A, lda, B, ldb, &beta,
                               C, ldc, blas_ok);

        mkl_set_num_threads_local(0);
#else
        PRLEVEL(1, ("%%YES tasking for DGEMM (" LD "x" LD ") in " LD " \n", M, N, f));
        int64_t num_col_blocks = N / L + 1;
        int64_t num_row_blocks = M / L + 1;

        int64_t len_col = N / num_col_blocks;
        int64_t len_row = M / num_row_blocks;

        PRLEVEL(1, ("%% col-blocks=" LD ",row-blocks=" LD " [" LD "]\n", num_col_blocks,
                    num_row_blocks, num_col_blocks * num_row_blocks));
    #pragma omp parallel proc_bind(close)
    #pragma omp single nowait
        {
            for (int64_t I = 0; I < num_row_blocks; I++)
            {
                int64_t m = (I + 1) == num_row_blocks ? (M - I * len_row) : len_row;

                for (int64_t J = 0; J < num_col_blocks; J++)
                {
                    int64_t n =
                        (J + 1) == num_col_blocks ? (N - J * len_col) : len_col;
                    PRLEVEL(1, ("%% I=" LD " J=" LD " m=" LD " n=" LD " in " LD "\n", I, J, m, n,
                                f));
    #pragma omp task
                    {
                        int64_t my_blas_ok = TRUE;
                        SUITESPARSE_BLAS_dgemm(
                            "N", "N", m, n, K, &alpha, A + (I * len_row), lda,
                            B + (J * len_col * ldb), ldb, &beta,
                            C + (J * ldc * len_col + I * len_row), ldc,
                            my_blas_ok);
                        if (!my_blas_ok)
                        {
    #pragma omp atomic write
                            blas_ok = my_blas_ok;
                        }
                    }
                }
            }
        }
#endif
    }

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("%% DGEMM (" LD "," LD "," LD ")%1.1f in " LD " {" LD "} in %lf seconds\n", M, N,
                K, beta, f, naft, time));
#endif

#ifdef COUNT_FLOPS
    #pragma omp atomic update
    Work->flp_cnt_dgemm += (double)2 * M * N * K;
#endif
    return blas_ok;
}
