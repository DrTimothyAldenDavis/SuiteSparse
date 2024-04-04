////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// paru_cp_control.cpp //////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//
/*! @brief  Copy the C control structure into the C++ control structure
 *  program can call ParU.
 *
 *  @author Aznaveh
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// paru_cp_control: copy the C control structure into the C++ structure
//------------------------------------------------------------------------------

void paru_cp_control (ParU_Control *Control, ParU_C_Control *Control_C)
{
    Control->mem_chunk = Control_C->mem_chunk;

    Control->umfpack_ordering = Control_C->umfpack_ordering;
    Control->umfpack_strategy = Control_C->umfpack_strategy;
    Control->umfpack_default_singleton = Control_C->umfpack_default_singleton;

    Control->relaxed_amalgamation_threshold = 
        Control_C->relaxed_amalgamation_threshold;

    Control->scale = Control_C->scale;
    Control->panel_width = Control_C->panel_width;
    Control->paru_strategy = Control_C->paru_strategy;


    Control->piv_toler = Control_C->piv_toler;
    Control->diag_toler = Control_C->diag_toler;
    Control->trivial = Control_C->trivial;
    Control->worthwhile_dgemm = Control_C->worthwhile_dgemm;
    Control->worthwhile_trsm = Control_C->worthwhile_trsm;
    Control->paru_max_threads = Control_C->paru_max_threads;
}

