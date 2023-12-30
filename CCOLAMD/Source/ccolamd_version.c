//------------------------------------------------------------------------------
// CCOLAMD/Source/ccolamd_version.c: return CCOLAMD version
//------------------------------------------------------------------------------

// CCOLAMD, Copyright (c) 2005-2022, Univ. of Florida, All Rights Reserved.
// Authors: Timothy A. Davis, Sivasankaran Rajamanickam, and Stefan Larimore.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

#include "ccolamd.h"

void ccolamd_version (int version [3])
{
    version [0] = CCOLAMD_MAIN_VERSION ;
    version [1] = CCOLAMD_SUB_VERSION ;
    version [2] = CCOLAMD_SUBSUB_VERSION ;
}

