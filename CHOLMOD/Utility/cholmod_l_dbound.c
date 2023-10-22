//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_l_dbound: bound diagonal of LDL (double, int64)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#define CHOLMOD_BOUND_FUNCTION  cholmod_l_dbound
#define COMMON_BOUND            (Common->dbound)
#define COMMON_BOUNDS_HIT       (Common->ndbounds_hit)
#define Real                    double

#define CHOLMOD_INT64
#include "t_cholmod_bound.c"

