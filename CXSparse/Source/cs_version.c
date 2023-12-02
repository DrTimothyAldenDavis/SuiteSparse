// CXSparse/Source/cxsparse_version: return CXSparse version
// CXSparse, Copyright (c) 2006-2023, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
void cxsparse_version (int version [3])
{
    version [0] = CS_VER ;
    version [1] = CS_SUBVER ;
    version [2] = CS_SUBSUB ;
}
