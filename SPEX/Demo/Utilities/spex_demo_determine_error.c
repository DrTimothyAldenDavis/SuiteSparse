//------------------------------------------------------------------------------
// Demo/spex_determine_error: auxiliary file for user demos
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: Determine why a SPEX function failed
 */

#include "spex_demos.h"

void spex_demo_determine_error
(
    SPEX_info info,
    int line,
    char *file
)
{
    printf("\nError code: %d", info);
    if (info == SPEX_OUT_OF_MEMORY)
    {
        printf("\nSPEX: Out of memory\n");
    }
    else if (info == SPEX_SINGULAR)
    {
        printf("\nSPEX: Input matrix is singular OR no diagonal pivot. Please ensure input is Correct\n");
    }
    else if (info == SPEX_INCORRECT_INPUT)
    {
        printf("\nSPEX: Incorrect input for a SPEX_Chol Function\n");
    }
    printf ("line %d, file: %s\n", line, file) ;
}
