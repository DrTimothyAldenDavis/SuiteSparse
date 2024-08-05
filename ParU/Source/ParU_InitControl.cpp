////////////////////////////////////////////////////////////////////////////////
////////////////////////// ParU_InitControl.cpp ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  Create the Control object and set its parameters to their defaults.
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_InitControl
//------------------------------------------------------------------------------

ParU_Info ParU_InitControl
(
    // output:
    ParU_Control *Control_handle    // Control object to create
)
{

    if (Control_handle == NULL)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = PARU_CALLOC (1, ParU_Control_struct) ;
    if (!Control)
    {
        return (PARU_OUT_OF_MEMORY) ;
    }

    // set the defaults for all Control parameters
    Control->paru_max_threads       = PARU_DEFAULT_MAX_THREADS ;
    Control->strategy               = PARU_DEFAULT_STRATEGY ;
    Control->umfpack_strategy       = PARU_DEFAULT_UMFPACK_STRATEGY ;
    Control->ordering               = PARU_DEFAULT_ORDERING ;
    Control->relaxed_amalgamation   = PARU_DEFAULT_RELAXED_AMALGAMATION ;
    Control->panel_width            = PARU_DEFAULT_PANEL_WIDTH ;
    Control->trivial                = PARU_DEFAULT_DGEMM_TINY ;
    Control->worthwhile_dgemm       = PARU_DEFAULT_DGEMM_TASKED ;
    Control->worthwhile_dtrsm       = PARU_DEFAULT_DTRSM_TASKED ;
    Control->prescale               = PARU_DEFAULT_PRESCALE ;
    Control->filter_singletons      = PARU_DEFAULT_SINGLETONS ;
    Control->mem_chunk              = PARU_DEFAULT_MEM_CHUNK ;
    Control->piv_toler              = PARU_DEFAULT_PIVOT_TOLERANCE ;
    Control->diag_toler             = PARU_DEFAULT_DIAG_PIVOT_TOLERANCE ;

    (*Control_handle) = Control ;
    return (PARU_SUCCESS) ;
}

