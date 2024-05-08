////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Set ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

// ParU_Set: set a parameter in the Control object

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_Set: set an int32_t or int64_t parameter in the Control object
//------------------------------------------------------------------------------

ParU_Info ParU_Set
(
    // input
    ParU_Control_enum parameter,    // parameter to set
    int32_t c,                      // value to set it to
    // control:
    ParU_Control Control
)
{
    return (ParU_Set (parameter, (int64_t) c, Control)) ;
}

ParU_Info ParU_Set
(
    // input
    ParU_Control_enum parameter,    // parameter to set
    int64_t c,                      // value to set it to
    // control:
    ParU_Control Control
)
{

    if (!Control)
    {
        return (PARU_INVALID) ;
    }

    switch (parameter)
    {

        case PARU_CONTROL_MAX_THREADS:            // max number of threads
            Control->paru_max_threads =
                (c <= 0) ? PARU_DEFAULT_MAX_THREADS : c ;
            break ;

        case PARU_CONTROL_STRATEGY:               // ParU strategy
            Control->strategy =
                (c == PARU_STRATEGY_AUTO ||
                 c == PARU_STRATEGY_UNSYMMETRIC ||
                 c == PARU_STRATEGY_SYMMETRIC) ? c : PARU_DEFAULT_STRATEGY ;
            break ;

        case PARU_CONTROL_UMFPACK_STRATEGY:       // UMFPACK strategy
            Control->umfpack_strategy =
                (c == UMFPACK_STRATEGY_AUTO ||
                 c == UMFPACK_STRATEGY_UNSYMMETRIC ||
                 c == UMFPACK_STRATEGY_SYMMETRIC) ? c :
                 PARU_DEFAULT_UMFPACK_STRATEGY ;
            break ;

        case PARU_CONTROL_ORDERING:               // ParU ordering, via UMFPACK
            Control->ordering =
                (c == PARU_ORDERING_AMD ||
                 c == PARU_ORDERING_METIS ||
                 c == PARU_ORDERING_METIS_GUARD ||
                 c == PARU_ORDERING_CHOLMOD ||
                 c == PARU_ORDERING_BEST ||
                 c == PARU_ORDERING_NONE) ? c : PARU_DEFAULT_ORDERING ;
            break ;

        case PARU_CONTROL_RELAXED_AMALGAMATION:   // goal # pivots in each front
            Control->relaxed_amalgamation =
                (c < 0) ? PARU_DEFAULT_RELAXED_AMALGAMATION :
                std::min (c, (int64_t) 512) ;
            break ;

        case PARU_CONTROL_PANEL_WIDTH:            // # of pivots in a panel
            Control->panel_width = (c <= 0) ? PARU_DEFAULT_PANEL_WIDTH : c ;
            break ;

        case PARU_CONTROL_DGEMM_TINY:             // dimension of tiny dgemm's
            Control->trivial = (c < 0) ? PARU_DEFAULT_DGEMM_TINY : c ;
            break ;

        case PARU_CONTROL_DGEMM_TASKED:           // dimension of tasked dgemm's
            Control->worthwhile_dgemm =
                (c < 0) ? PARU_DEFAULT_DGEMM_TASKED : c ;
            break ;

        case PARU_CONTROL_DTRSM_TASKED:           // dimension of tasked dtrsm's
            Control->worthwhile_dtrsm =
                (c < 0) ? PARU_DEFAULT_DTRSM_TASKED : c ;
            break ;

        case PARU_CONTROL_PRESCALE:               // prescale input matrix
            Control->prescale =
                (c < 0 || c > PARU_PRESCALE_MAX) ? PARU_DEFAULT_PRESCALE : c ;
            break ;

        case PARU_CONTROL_SINGLETONS:             // filter singletons, or not
            Control->filter_singletons =
                (c < 0) ? PARU_DEFAULT_SINGLETONS : std::min (c, (int64_t) 1) ;
            break ;

        case PARU_CONTROL_MEM_CHUNK:              // chunk size memset, memcpy
            Control->mem_chunk = (c <= 0) ? PARU_DEFAULT_MEM_CHUNK : c ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}


//------------------------------------------------------------------------------
// ParU_Set: set a float or double parameter in the Control object
//------------------------------------------------------------------------------

ParU_Info ParU_Set
(
    // input
    ParU_Control_enum parameter,    // parameter to set
    float c,                        // value to set it to
    // control:
    ParU_Control Control
)
{
    return (ParU_Set (parameter, (double) c, Control)) ;
}

ParU_Info ParU_Set
(
    // input
    ParU_Control_enum parameter,    // parameter to set
    double c,                       // value to set it to
    // control:
    ParU_Control Control
)
{

    if (!Control)
    {
        return (PARU_INVALID) ;
    }

    switch (parameter)
    {

        case PARU_CONTROL_PIVOT_TOLERANCE:          // pivot tolerance
            Control->piv_toler =
                (c < 0) ? PARU_DEFAULT_PIVOT_TOLERANCE :
                std::min (c, (double) 1) ;
            break ;

        case PARU_CONTROL_DIAG_PIVOT_TOLERANCE:     // diagonal pivot tolerance
            Control->diag_toler =
                (c < 0) ? PARU_DEFAULT_DIAG_PIVOT_TOLERANCE :
                std::min (c, (double) 1) ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}

