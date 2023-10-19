////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_omp.hpp ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

#ifndef PARU_OMP_H
#define PARU_OMP_H
//!
// definitions of using OpenMP inside ParU 
//  @author Aznaveh
//

#if defined ( _OPENMP )
    #include <omp.h>
    #define PARU_OPENMP_MAX_THREADS       omp_get_max_threads ( )
    #define PARU_OPENMP_GET_NUM_THREADS   omp_get_num_threads ( )
    #define PARU_OPENMP_GET_WTIME         omp_get_wtime ( )
    #define PARU_OPENMP_GET_THREAD_ID     omp_get_thread_num ( )
    #define PARU_OPENMP_SET_DYNAMIC(d)   omp_set_dynamic(d)
    #define PARU_OPENMP_SET_MAX_ACTIVE_LEVELS(l)   omp_set_max_active_levels(l)
    #define PARU_OPENMP_GET_ACTIVE_LEVEL   omp_get_active_level()
    #define PARU_OPENMP_GET_THREAD_NUM   omp_get_thread_num ( )

#else

    #define PARU_OPENMP_MAX_THREADS       (1)
    #define PARU_OPENMP_GET_NUM_THREADS   (1)
    #define PARU_OPENMP_GET_WTIME         (0)
    #define PARU_OPENMP_GET_THREAD_ID     (0)
    #define PARU_OPENMP_SET_DYNAMIC       (0)
    #define PARU_OPENMP_SET_MAX_ACTIVE_LEVELS(l)
    #define PARU_OPENMP_GET_ACTIVE_LEVEL   (0)
    #define PARU_OPENMP_GET_THREAD_NUM     (0)
#endif

#endif
