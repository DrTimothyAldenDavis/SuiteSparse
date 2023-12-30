//------------------------------------------------------------------------------
// BTF/Source/btf_version: return BTF version
//------------------------------------------------------------------------------

// BTF, Copyright (c) 2004-2023, University of Florida.  All Rights Reserved.
// Author: Timothy A. Davis.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "btf.h"

void btf_version (int version [3])
{
    version [0] = BTF_MAIN_VERSION ;
    version [1] = BTF_SUB_VERSION ;
    version [2] = BTF_SUBSUB_VERSION ;
}

