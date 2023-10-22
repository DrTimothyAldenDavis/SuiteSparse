//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_l_sbound: bound diagonal of LDL (single, int64)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#define CHOLMOD_BOUND_FUNCTION  cholmod_l_sbound
#define COMMON_BOUND            (Common->sbound)
#define COMMON_BOUNDS_HIT       (Common->nsbounds_hit)
#define Real                    float

#define CHOLMOD_INT64
#include "t_cholmod_bound.c"

