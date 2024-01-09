//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_clear_flag: clear Common->Flag
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// cholmod_clear_flag increments Common->mark to ensure
// Common->Flag [0..Common->nrow-1] < mark holds.  If incrementing
// Common->mark results in integer overflow, Flag is set to EMPTY and
// Common->mark is set to zero.

int64_t CHOLMOD(clear_flag)     // returns the new Common->mark
(
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (EMPTY) ;

    //--------------------------------------------------------------------------
    // increment the mark
    //--------------------------------------------------------------------------

    Common->mark++ ;

    //--------------------------------------------------------------------------
    // check for int32 or int64 overflow
    //--------------------------------------------------------------------------

    bool overflow = (Common->mark <= 0) ;
    #if defined ( CHOLMOD_INT32 )
    overflow = overflow || (Common->mark > INT32_MAX) ;
    #endif

    if (overflow)
    {
        Common->mark = 0 ;
        CHOLMOD(set_empty) (Common->Flag, Common->nrow) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (check_flag (Common)) ;
    return (Common->mark) ;
}

