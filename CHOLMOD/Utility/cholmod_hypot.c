//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_hypot: complex hypot
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

double CHOLMOD(hypot) (double x, double y)
{
    return (SuiteSparse_config_hypot (x, y)) ;
}

