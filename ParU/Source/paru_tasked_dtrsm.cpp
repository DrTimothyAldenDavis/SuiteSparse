////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_tasked_dtrsm //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief      a wrapper around  BLAS_TRSM for tasking
 *
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

bool paru_tasked_dtrsm
(
    int64_t f,
    int64_t m,
    int64_t n,
    double alpha,
    double *a,
    int64_t lda,
    double *b,
    int64_t ldb,
    paru_work *Work,
    ParU_Numeric Num
)
{

    // get Control
    int32_t nthreads = Work->nthreads ;
    int64_t worthwhile_dtrsm = Work->worthwhile_dtrsm ;

    DEBUGLEVEL(0);
    int64_t naft;

    bool blas_ok = true ;
#ifdef PARU_COVERAGE
    worthwhile_dtrsm = 32;
#endif
    #pragma omp atomic read
    naft = Work->naft;
    if (naft == 1)
    {
        BLAS_set_num_threads(nthreads);
    }
    else
    {
        BLAS_set_num_threads(1);
    }
    if (n < worthwhile_dtrsm || (naft == 1) || (naft >= nthreads))
    {
#ifndef NDEBUG
        if (n < worthwhile_dtrsm) PRLEVEL(1, ("%% Small TRSM (" LD "x" LD ") in " LD "\n", m, n, f));
        if (naft == 1)
            PRLEVEL(1, ("%% All threads for trsm(" LD "x" LD ") in " LD "\n", m, n, f));
#endif
        SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", m, n, &alpha, a, lda, b, ldb,
                               blas_ok);
    }
    else
    {
#if ( defined ( BLAS_Intel10_64ilp ) || defined ( BLAS_Intel10_64lp ) )
        int my_share = nthreads / naft;
        if (my_share == 0) my_share = 1;
        PRLEVEL(1, ("%% MKL local threads for trsm(" LD "x" LD ") in " LD " [[%d]]\n", m,
                    n, f, my_share));
        mkl_set_num_threads_local(my_share);
        SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", m, n, &alpha, a, lda, b, ldb,
                               blas_ok);

        mkl_set_num_threads_local(0);
#else
        PRLEVEL(1, ("%%YES tasksingt for trsm(" LD "x" LD ") in " LD " \n", m, n, f));
        int64_t num_blocks = n / worthwhile_dtrsm + 1;
        int64_t len_bloc = n / num_blocks;
        PRLEVEL(1, ("%%  num_blocks = " LD "\n", num_blocks));
        #pragma omp parallel proc_bind(close)
        #pragma omp single nowait
        {
            for (int64_t J = 0; J < num_blocks; J++)
            {
                int64_t n_b = (J + 1) == num_blocks ? (n - J * len_bloc) : len_bloc;
                PRLEVEL(1, ("%%  n_b= " LD "\n", n_b));
                #pragma omp task
                {
                    bool my_blas_ok = true ;
                    SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", m, n_b, &alpha,
                                           a, lda, (b + J * len_bloc * ldb),
                                           ldb, my_blas_ok);
                    if (!my_blas_ok)
                    {
                        #pragma omp atomic write
                        blas_ok = false ;
                    }
                }
            }
        }
#endif
    }

#ifdef COUNT_FLOPS
    #pragma omp atomic update
    Work->flp_cnt_trsm += (double)(m + 1) * m * n;
#endif
    return blas_ok;
}
