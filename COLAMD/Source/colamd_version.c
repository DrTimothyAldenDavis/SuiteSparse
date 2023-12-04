//------------------------------------------------------------------------------
// COLAMD/Source/colamd_version.c: return COLAMD version
//------------------------------------------------------------------------------

// COLAMD, Copyright (c) 1998-2022, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

#include "colamd.h"

void colamd_version (int version [3])
{
    version [0] = COLAMD_MAIN_VERSION ;
    version [1] = COLAMD_SUB_VERSION ;
    version [2] = COLAMD_SUBSUB_VERSION ;
}

