//------------------------------------------------------------------------------
// GB_macrofy_defn: construct defn for an operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_defn
(
    FILE *fp,
    int kind,
    const char *name,
    const char *defn
)
{

    if (name != NULL && defn != NULL)
    {
        if (kind == 2)
        {
            // built-in macro only need for CUDA
            fprintf (fp, "#if defined ( __NVCC__)\n") ;
        }

        // construct the guard to prevent duplicate definitions
        fprintf (fp, "#ifndef GB_GUARD_%s_DEFINED\n", name) ;
        fprintf (fp, "#define GB_GUARD_%s_DEFINED\n", name) ;

        if (kind == 0)
        {
            // built-in operator defined by a function,
            // or a user-defined operator
            fprintf (fp, "GB_STATIC_INLINE\n%s\n", defn) ;
        }
        else // kind is 1 or 2
        {
            // built-in operator defined by a macro
            fprintf (fp, "#define %s\n", defn) ;
        }

        // end the guard
        fprintf (fp, "#endif\n") ;

        if (kind == 2)
        {
            // end the NVCC guard
            fprintf (fp, "#endif\n") ;
        }
    }
}

