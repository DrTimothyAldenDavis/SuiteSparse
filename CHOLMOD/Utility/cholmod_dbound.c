//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_dbound: bound diagonal of LDL (double, int32)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#define CHOLMOD_BOUND_FUNCTION  cholmod_dbound
#define COMMON_BOUND            (Common->dbound)
#define COMMON_BOUNDS_HIT       (Common->ndbounds_hit)
#define Real                    double

#define CHOLMOD_INT32
#include "t_cholmod_bound.c"

