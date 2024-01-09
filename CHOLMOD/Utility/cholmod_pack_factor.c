//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_pack_factor: pack a simplicial factorization
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// The columns of simplicial factor L can have gaps between them, with empty
// space.  This method removes the empty space, leaving all the empty space at
// the tail end of L->i and L->x.  Each column of L is reduced in is size so
// that it has at most Common->grow2 empty space at the end of each column.

// L must be simplicial and numerical (not symbolic).  If L is supernodal, or
// symbolic, this method does nothing.

// This method can be followed by a call to cholmod_reallocate_factor, to
// reduce the size of L->i and L->x.  Or, the space can be left to accomodate
// future growth from updates/downdates.

// The columns of L are not made to appear in monotonic order.  For that,
// use cholmod_change_factor which can both pack the columns and make them
// monotonic.

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// t_cholmod_pack_factor_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_pack_factor_worker.c"
#define COMPLEX
#include "t_cholmod_pack_factor_worker.c"
#define ZOMPLEX
#include "t_cholmod_pack_factor_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_pack_factor_worker.c"
#define COMPLEX
#include "t_cholmod_pack_factor_worker.c"
#define ZOMPLEX
#include "t_cholmod_pack_factor_worker.c"

//------------------------------------------------------------------------------
// cholmod_pack_factor
//------------------------------------------------------------------------------

int CHOLMOD(pack_factor)
(
    // input/output:
    cholmod_factor *L,      // factor to pack
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_FACTOR_INVALID (L, FALSE) ;
    Common->status = CHOLMOD_OK ;

    DEBUG (CHOLMOD(dump_factor) (L, "pack:L input", Common)) ;

    if (L->xtype == CHOLMOD_PATTERN || L->is_super)
    {
        // nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // pack
    //--------------------------------------------------------------------------

    switch ((L->xtype + L->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_pack_factor_worker (L, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_pack_factor_worker (L, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_pack_factor_worker (L, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_pack_factor_worker (L, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_pack_factor_worker (L, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_pack_factor_worker (L, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    DEBUG (CHOLMOD(dump_factor) (L, "done pack", Common)) ;
    return (TRUE) ;
}

