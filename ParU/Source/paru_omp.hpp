////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_omp.hpp ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2025, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PARU_OMP_H
#define PARU_OMP_H
//!
// definitions of using OpenMP inside ParU
//  @author Aznaveh
//

#if defined ( _OPENMP )

    #include <omp.h>

    static inline double PARU_omp_get_wtime (void)
    {
        // get the current wallclock time
        return (omp_get_wtime ( )) ;
    }

    static inline int PARU_omp_get_max_threads (void)
    {
        // get the max # of threads used by OpenMP
        return (omp_get_max_threads ( )) ;
    }

    static inline int PARU_omp_get_num_threads (void)
    {
        // get the current # of threads used by OpenMP
        return (omp_get_num_threads ( )) ;
    }

    static inline int PARU_omp_set_num_threads (int nthreads)
    {
        // get the current # of threads used by OpenMP, and return prior setting
        int prior = omp_get_num_threads ( ) ;
        omp_set_num_threads (nthreads) ;
        return (prior) ;
    }

    static inline int PARU_omp_get_dynamic (void)
    {
        // get the current OpenMP dynamic setting
        return (omp_get_dynamic ( )) ;
    }

    static inline int PARU_omp_set_dynamic (int dynamic)
    {
        // set the OpenMP dynamic threading option, and return prior setting
        int prior = omp_get_dynamic ( ) ;
        omp_set_dynamic (dynamic) ;
        return (prior) ;
    }

    static inline int PARU_omp_get_active_level (void)
    {
        // get the current OpenMP active level
        return (omp_get_active_level ( )) ;
    }

    static inline int PARU_omp_get_max_active_levels (void)
    {
        // get the max # of OpenMP active levels
        return (omp_get_max_active_levels ( )) ;
    }

    static inline void PARU_omp_set_max_active_levels (int nlevels)
    {
        // set the max # of OpenMP active levels
        omp_set_max_active_levels (nlevels) ;
    }

    static inline int PARU_omp_get_thread_num (void)
    {
        // get the id of this thread
        return (omp_get_thread_num ( )) ;
    }

#else

    // no OpenMP, so use sequential frontal tree tasking
    #ifndef PARU_1TASK
    #define PARU_1TASK
    #endif

    static inline double PARU_omp_get_wtime (void)
    {
        return (0) ;
    }

    static inline int PARU_omp_get_max_threads (void)
    {
        return (1) ;
    }

    static inline int PARU_omp_get_num_threads (void)
    {
        return (1) ;
    }

    static inline int PARU_omp_set_num_threads (int nthreads)
    {
        return (1) ;
    }

    static inline int PARU_omp_get_dynamic (void)
    {
        return (0) ;
    }

    static inline int PARU_omp_set_dynamic (int dynamic)
    {
        return (0) ;
    }

    static inline int PARU_omp_get_active_level (void)
    {
        return (0) ;
    }

    static inline int PARU_omp_get_max_active_levels (void)
    {
        return (1) ;
    }

    static inline void PARU_omp_set_max_active_levels (int nlevels)
    {
        ;
    }

    static inline int PARU_omp_get_thread_num (void)
    {
        return (0) ;
    }

#endif

#endif
