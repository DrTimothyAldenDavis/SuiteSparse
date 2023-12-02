//------------------------------------------------------------------------------
// SPEX/Source/amd_version: return SPEX version
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2023, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_util_internal.h"

void SPEX_version (int version [3])
{
    version [0] = SPEX_VERSION_MAJOR ;
    version [1] = SPEX_VERSION_MINOR ;
    version [2] = SPEX_VERSION_SUB ;
}

