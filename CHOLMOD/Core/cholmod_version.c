//------------------------------------------------------------------------------
// CHOLMOD/Core/cholmod_version: current version of CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Core Module.  Copyright (C) 2005-2022, University of Florida.
// All Rights Reserved. Author:  Timothy A. Davis
// SPDX-License-Identifier: LGPL-2.1+
//------------------------------------------------------------------------------

/* Return the current version of CHOLMOD.  Unlike all other functions in
   CHOLMOD, this function does not require the CHOLMOD Common. */

#include "cholmod_internal.h"

int CHOLMOD(version)        /* returns CHOLMOD_VERSION */
(
    /* output, contents not defined on input.  Not used if NULL.
        version [0] = CHOLMOD_MAIN_VERSION ;
        version [1] = CHOLMOD_SUB_VERSION ;
        version [2] = CHOLMOD_SUBSUB_VERSION ;
    */
    int version [3]
)
{
    if (version != NULL)
    {
        version [0] = CHOLMOD_MAIN_VERSION ;
        version [1] = CHOLMOD_SUB_VERSION ;
        version [2] = CHOLMOD_SUBSUB_VERSION ;
    }
    return (CHOLMOD_VERSION) ;
}

