//------------------------------------------------------------------------------
// AMD/Source/amd_version: return AMD version
//------------------------------------------------------------------------------

// AMD, Copyright (c) 1996-2023, Timothy A. Davis, Patrick R. Amestoy, and
// Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

#include "amd_internal.h"

void amd_version (int version [3])
{
    version [0] = AMD_MAIN_VERSION ;
    version [1] = AMD_SUB_VERSION ;
    version [2] = AMD_SUBSUB_VERSION ;
}

