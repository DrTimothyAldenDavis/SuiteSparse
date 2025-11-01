////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// paru_nthreads ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2025, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "paru_internal.hpp"

int32_t paru_nthreads (ParU_Control Control)
{
    if (Control == NULL)
    {
        // default # of threads
        return (PARU_omp_get_max_threads ( )) ;
    }
    int32_t nthreads = Control->paru_max_threads ;
    if (nthreads == PARU_DEFAULT_MAX_THREADS)
    {
        // default # of threads
        return (PARU_omp_get_max_threads ( )) ;
    }
    else
    {
        return std::min (nthreads, PARU_omp_get_max_threads ( )) ;
    }
}

