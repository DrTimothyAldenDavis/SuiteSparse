//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_symbolic_analysis_free: Free memory for the
// SPEX_symbolic_analysis data type.
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function frees the memory of the SPEX_symbolic_analysis struct
 *
 * Input is the SPEX_symbolic_analysis structure, it is destroyed on function
 * termination.
 */

#include "spex_util_internal.h"

SPEX_info SPEX_symbolic_analysis_free
(
    SPEX_symbolic_analysis *S_handle,   // Structure to be deleted
    const SPEX_options option
)
{

    if (!spex_initialized ( )) return (SPEX_PANIC);

    if ((S_handle != NULL) && (*S_handle != NULL))
    {

        SPEX_FREE((*S_handle)->P_perm);
        SPEX_FREE((*S_handle)->Pinv_perm);
        SPEX_FREE((*S_handle)->Q_perm);
        SPEX_FREE((*S_handle)->Qinv_perm);

        SPEX_FREE((*S_handle)->parent);
        SPEX_FREE((*S_handle)->cp);
        SPEX_FREE (*S_handle);
    }

    return (SPEX_OK);
}

