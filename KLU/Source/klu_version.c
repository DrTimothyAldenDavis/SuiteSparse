//------------------------------------------------------------------------------
// KLU/Source/klu_version: return KLU version
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2023, University of Florida.  All Rights Reserved.
// Authors: Timothy A. Davis and Ekanathan Palamadai.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "klu_internal.h"

void klu_version (int version [3])
{
    version [0] = KLU_MAIN_VERSION ;
    version [1] = KLU_SUB_VERSION ;
    version [2] = KLU_SUBSUB_VERSION ;
}

