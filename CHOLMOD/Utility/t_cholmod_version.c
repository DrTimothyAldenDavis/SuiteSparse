//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_version: CHOLMOD version
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(version)    // returns CHOLMOD_VERSION
(
    // if version is not NULL, then cholmod_version returns its contents as:
    // version [0] = CHOLMOD_MAIN_VERSION
    // version [1] = CHOLMOD_SUB_VERSION
    // version [2] = CHOLMOD_SUBSUB_VERSION
    int version [3]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (version == NULL)
    {
        return (CHOLMOD_VERSION) ;
    }

    //--------------------------------------------------------------------------
    // return the full version of CHOLMOD
    //--------------------------------------------------------------------------

    version [0] = CHOLMOD_MAIN_VERSION ;
    version [1] = CHOLMOD_SUB_VERSION ;
    version [2] = CHOLMOD_SUBSUB_VERSION ;
    return (CHOLMOD_VERSION) ;
}

