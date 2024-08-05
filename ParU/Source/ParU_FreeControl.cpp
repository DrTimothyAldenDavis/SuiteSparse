////////////////////////////////////////////////////////////////////////////////
////////////////////////// ParU_FreeControl.cpp ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  Free the Control object.
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_FreeControl
//------------------------------------------------------------------------------

ParU_Info ParU_FreeControl
(
    // input/output:
    ParU_Control *Control_handle    // Control object to free
)
{

    if (Control_handle == NULL || *Control_handle == NULL)
    {
        // nothing to do
        return (PARU_SUCCESS) ;
    }
    ParU_Control Control = *Control_handle ;
    PARU_FREE (1, ParU_Control_struct, Control) ;
    (*Control_handle) = NULL ;
    return (PARU_SUCCESS) ;
}

