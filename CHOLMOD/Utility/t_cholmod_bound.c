//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_bound: bound diagonal of LDL
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// Template for creating the following functions:
// cholmod_dbound       double, int32, using Common->dbound and ndbounds_hit
// cholmod_l_dbound     double, int64, using Common->dbound and ndbounds_hit
// cholmod_sbound       single, int32, using Common->sbound and nsbounds_hit
// cholmod_l_sbound     single, int64, using Common->sbound and nsbounds_hit

// This method ensures that the absolute value of D(j,j) is greater than dbound
// (for double) or sbound (for single), for LDL' factorization and
// update/downdate.  It is not used for supernodal factorization.

Real CHOLMOD_BOUND_FUNCTION     // returns modified diagonal entry D(j,j)
(
    // input:
    Real djj,                   // input diagonal entry D(j,j)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (0) ;
    if (isnan (djj)) return (djj) ;    // no change if D(j,j) is NaN

    //--------------------------------------------------------------------------
    // check the bound
    //--------------------------------------------------------------------------

    bool hit ;
    Real bound ;
    if (djj >= 0)
    {
        // D(j,j) is positive: check if djj in range [0,Common->bound]
        bound = COMMON_BOUND ;
        hit = (djj < bound) ;
    }
    else
    {
        // D(j,j) is negative: check if djj in range [-Common->bound,0]
        bound = -COMMON_BOUND ;
        hit = (djj > bound) ;
    }

    //--------------------------------------------------------------------------
    // record the hit
    //--------------------------------------------------------------------------

    if (hit)
    {
        // bound the diagonal entry
        djj = bound ;
        // record the # of times the bound was hit
        COMMON_BOUNDS_HIT++ ;
        // set an error flag, if not already set
        if (Common->status == CHOLMOD_OK)
        {
            ERROR (CHOLMOD_DSMALL, "diagonal entry is below threshold") ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (djj) ;
}

