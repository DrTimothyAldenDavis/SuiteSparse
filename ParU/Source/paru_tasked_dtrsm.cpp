////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_tasked_dtrsm //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2025, Mohsen Aznaveh and Timothy A. Davis,
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
#ifdef PARU_COVERAGE
    worthwhile_dtrsm = 32;
#else
    int64_t worthwhile_dtrsm = Work->worthwhile_dtrsm ;
#endif

    DEBUGLEVEL(0);
    int64_t naft;

    bool blas_ok = true ;

    #define CHUNK ((double) 5e8)
    double work = ((double) m) * ((double) m) * ((double) n) ;
    int nth = paru_nthreads_to_use (work, CHUNK, nthreads) ;
    int my_share = 1 ;

    #pragma omp atomic read
    naft = Work->naft;
    bool small = (n < worthwhile_dtrsm) ;

    if (small || (naft >= nth))
    {

        //----------------------------------------------------------------------
        // single-threaded call to dtrsm
        //----------------------------------------------------------------------

#ifndef NDEBUG
        if (small) PRLEVEL(1, ("%% small, for DTRSM (" LD "x" LD ") in " LD "\n", m, n, f));
#endif
        int prior = BLAS_set_num_threads_local (1) ;
        SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", m, n, &alpha, a, lda, b, ldb,
                               blas_ok);
        BLAS_set_num_threads_local (prior) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // parallel dtrsm
        //----------------------------------------------------------------------

        // using my share of threads
        my_share = std::max ((int) 1, (int) (nth / naft)) ;

        if (Work->nthreads_for_blas > 1)
        {

            //------------------------------------------------------------------
            // parallel dtrsm with multi-threaded BLAS (MKL or OpenBLAS)
            //------------------------------------------------------------------

            my_share = std::min (my_share, Work->nthreads_for_blas) ;
            PRLEVEL(1, ("MKL local threads for DTRSM(" LD "x" LD ") in " LD " [[%d]]\n", m,
                        n, f, my_share));
            int prior = BLAS_set_num_threads_local (my_share) ;
            SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", m, n, &alpha, a, lda, b, ldb,
                                   blas_ok);
            BLAS_set_num_threads_local (prior) ;

        }
        else
        {

            //------------------------------------------------------------------
            // tasked dtrsm with any BLAS
            //------------------------------------------------------------------

            PRLEVEL(1, ("taking for DTRSM(" LD "x" LD ") in " LD " \n", m, n, f));
            int64_t num_blocks = n / worthwhile_dtrsm + 1;
            int64_t len_bloc = n / num_blocks;
            PRLEVEL(2, ("num_blocks = " LD "\n", num_blocks));

            #pragma omp parallel proc_bind(close) num_threads(my_share)
            #pragma omp single nowait
            {
                for (int64_t J = 0; J < num_blocks; J++)
                {
                    int64_t n_b = (J + 1) == num_blocks ? (n - J * len_bloc) : len_bloc;
                    PRLEVEL(2, ("%%  n_b= " LD "\n", n_b));
                    #pragma omp task
                    {
                        bool my_blas_ok = true ;
                        int prior = BLAS_set_num_threads_local (1) ;
                        SUITESPARSE_BLAS_dtrsm("L", "L", "N", "U", m, n_b, &alpha,
                                               a, lda, (b + J * len_bloc * ldb),
                                               ldb, my_blas_ok);
                        BLAS_set_num_threads_local (prior) ;
                        if (!my_blas_ok)
                        {
                            #pragma omp atomic write
                            blas_ok = false ;
                        }
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

#ifdef COUNT_FLOPS
    #pragma omp atomic update
    Work->flp_cnt_trsm += (double)(m + 1) * m * n;
#endif
    return (blas_ok) ;
}

