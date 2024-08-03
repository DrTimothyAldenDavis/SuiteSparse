//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_create_default_options: set defaults
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: Create SPEX_options pointer with default parameters
 * upon successful allocation, which are defined in spex_internal.h
 */

#include "spex_util_internal.h"

SPEX_info SPEX_create_default_options (SPEX_options *option_handle)
{

    if (!spex_initialized ( )) return (SPEX_PANIC);

    //--------------------------------------------------------------------------
    // allocate the option struct
    //--------------------------------------------------------------------------

    (*option_handle) = SPEX_calloc(1, sizeof(SPEX_options_struct));
    if (!(*option_handle))
    {
        // out of memory
        return (SPEX_OUT_OF_MEMORY);
    }

    //--------------------------------------------------------------------------
    // set defaults
    //--------------------------------------------------------------------------

    (*option_handle)->pivot       = SPEX_DEFAULT_PIVOT ;
    (*option_handle)->order       = SPEX_DEFAULT_ORDER ;
    (*option_handle)->print_level = SPEX_DEFAULT_PRINT_LEVEL ;
    (*option_handle)->prec        = SPEX_DEFAULT_PRECISION ;
    (*option_handle)->tol         = SPEX_DEFAULT_TOL ;
    (*option_handle)->round       = SPEX_DEFAULT_MPFR_ROUND ;
    (*option_handle)->algo        = SPEX_DEFAULT_ALGORITHM ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return SPEX_OK ;
}

