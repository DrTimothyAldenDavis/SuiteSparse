//------------------------------------------------------------------------------
// CAMD/Source/camd_version: return CAMD version
//------------------------------------------------------------------------------

// CAMD, Copyright (c) 2007-2023, Timothy A. Davis, Yanqing Chen, Patrick R.
// Amestoy, and Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

#include "camd_internal.h"

void camd_version (int version [3])
{
    version [0] = CAMD_MAIN_VERSION ;
    version [1] = CAMD_SUB_VERSION ;
    version [2] = CAMD_SUBSUB_VERSION ;
}

