//------------------------------------------------------------------------------
// GB_macrofy_copyright: print the copyright and license
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_copyright
(
    FILE *fp                // target file to write, already open
)
{

    const char *date = GxB_IMPLEMENTATION_DATE ;
    int len = strlen (date) ;

    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n"
        "// SuiteSparse:GraphBLAS v%d.%d.%d, Timothy A. Davis, (c) 2017-%s,"
        " All Rights Reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// The above copyright and license do not apply to any\n"
        "// user-defined types and operators defined below.\n"
        "//--------------------------------------"
        "----------------------------------------\n\n",
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB,
        date + GB_IMAX (0, len - 4)) ;
}

