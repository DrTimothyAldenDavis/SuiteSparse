//------------------------------------------------------------------------------
// SPEX_Util/SPEX_LU_analysis_free: Free memory from symbolic analysis struct
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function frees the memory of the SPEX_LU_analysis struct
 *
 * Input is the SPEX_LU_analysis structure, it is destroyed on function
 * termination.
 */

#include "spex_util_internal.h"

SPEX_info SPEX_LU_analysis_free
(
    SPEX_LU_analysis **S, // Structure to be deleted
    const SPEX_options *option
)
{
    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    if ((S != NULL) && (*S != NULL))
    {
        SPEX_FREE ((*S)->q) ;
        SPEX_FREE (*S) ;
    }

    return (SPEX_OK) ;
}

