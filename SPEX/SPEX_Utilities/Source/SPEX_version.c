//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_version: report SPEX version information
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Returns the library version and date.

#include "spex_util_internal.h"

SPEX_info SPEX_version
(
    // output
    int version [3],            // SPEX major, minor, and sub version
    char date [128]             // date of this version
)
{

    if (version != NULL)
    {
        version [0] = SPEX_VERSION_MAJOR ;
        version [1] = SPEX_VERSION_MINOR ;
        version [2] = SPEX_VERSION_SUB ;
    }

    if (date != NULL)
    {
        strncpy (date, SPEX_DATE, 127);
    }

    return (SPEX_OK);
}

