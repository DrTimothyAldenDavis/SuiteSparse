//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_version: return UMFPACK version
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "umf_internal.h"

void umfpack_version (int version [3])
{
    version [0] = UMFPACK_MAIN_VERSION ;
    version [1] = UMFPACK_SUB_VERSION ;
    version [2] = UMFPACK_SUBSUB_VERSION ;
}

