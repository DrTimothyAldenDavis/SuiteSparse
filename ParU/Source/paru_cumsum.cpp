////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_cumsum ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*!
 * @brief   Overwrite a vector of length n with its cumulative sum of length
 *          n+1.
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"

int64_t paru_cumsum
(
    int64_t n,
    int64_t *X,
    size_t mem_chunk,
    int32_t nthreads
)
{

    // n is size, X is size n and in/out
    int64_t tot = 0;
    if (X == NULL) return tot;

    if (n < (int64_t) mem_chunk)
    {
        for (int64_t k = 0; k < n; k++)
        {
            X[k] += tot;  // tot = sum (X[0:k-1])
            tot = X[k];
        }
        return tot;
    }
    int64_t mid = n / 2;
    int64_t sum = 0;

    #pragma omp parallel shared(sum, n, X, mem_chunk, nthreads) \
        firstprivate(mid) num_threads(nthreads)
    {
        #pragma omp single
        {
            #pragma omp task
            sum = paru_cumsum(mid, X, mem_chunk, nthreads) ;
            #pragma omp task
            paru_cumsum(n - mid, X + mid, mem_chunk, nthreads) ;
            #pragma omp taskwait
            #pragma omp taskloop
            for (int64_t i = mid; i < n; i++)
            {
                X[i] += sum;
            }
        }
    }
    return X[n - 1];
}
