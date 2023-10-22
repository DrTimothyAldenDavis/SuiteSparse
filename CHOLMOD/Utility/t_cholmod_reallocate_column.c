//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_reallocate_column: reallocate a column of a factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Expand the space for a single column L(:,j) of a simplicial factor.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                                                 \
    if (Common->status != CHOLMOD_OK)                                   \
    {                                                                   \
        /* out of memory; change L to simplicial symbolic */            \
        CHOLMOD(change_factor) (CHOLMOD_PATTERN + L->dtype, L->is_ll,   \
            /* make L simplicial: */ FALSE,                             \
            /* make L packed */ TRUE,                                   \
            /* make L monotonic: */ TRUE, L, Common) ;                  \
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;                \
        return (FALSE) ;                                                \
    }

//------------------------------------------------------------------------------
// t_cholmod_reallocate_column_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_reallocate_column_worker.c"
#define COMPLEX
#include "t_cholmod_reallocate_column_worker.c"
#define ZOMPLEX
#include "t_cholmod_reallocate_column_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_reallocate_column_worker.c"
#define COMPLEX
#include "t_cholmod_reallocate_column_worker.c"
#define ZOMPLEX
#include "t_cholmod_reallocate_column_worker.c"

//------------------------------------------------------------------------------
// cholmod_reallocate_column
//------------------------------------------------------------------------------

int CHOLMOD(reallocate_column)
(
    size_t j,                   // reallocate L(:,j)
    size_t need,                // space in L(:,j) for this # of entries
    cholmod_factor *L,          // L factor modified, L(:,j) resized
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_FACTOR_INVALID (L, FALSE) ;
    Common->status = CHOLMOD_OK ;

    DEBUG (CHOLMOD(dump_factor) (L, "realloc col:L input", Common)) ;

    Int n = L->n ;
    if (L->xtype == CHOLMOD_PATTERN || L->is_super || j >= n)
    {
        ERROR (CHOLMOD_INVALID, "L not simplicial or j out of range") ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // ensure need is in range 1:(n-j) and add slack space
    //--------------------------------------------------------------------------

    need = MAX (need, 1) ;
    double slack = MAX (Common->grow1, 1.0) * ((double) need) + Common->grow2 ;
    slack = MIN (slack, (double) (n-j)) ;
    size_t nslack = (size_t) floor (slack) ;
    need = MAX (need, slack) ;
    need = MAX (need, 1) ;
    need = MIN (need, n-j) ;

    //--------------------------------------------------------------------------
    // quick return if L(:,j) already big enough
    //--------------------------------------------------------------------------

    Int *Lp    = (Int *) L->p ;
    Int *Lnext = (Int *) L->next ;
    Int *Lprev = (Int *) L->prev ;

    size_t already_have = ((size_t) Lp [Lnext [j]] - (size_t) Lp [j]) ;
    if (already_have >= need)
    {
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // check if enough space at the end of L->i, L->x, and L->z
    //--------------------------------------------------------------------------

    Int tail = n ;
    Int new_nzmax_required = need + Lp [tail] ;
    if (new_nzmax_required > L->nzmax)
    {

        //----------------------------------------------------------------------
        // out of space in L, so grow the entire factor to lnznew space
        //----------------------------------------------------------------------

        double grow0 = Common->grow0 ;
        grow0 = (isnan (grow0) || grow0 < 1.2) ? 1.2 : grow0 ;
        double xnz = grow0 * (((double) L->nzmax) + ((double) need) + 1) ;
        size_t lnznew = (xnz > (double) SIZE_MAX) ? SIZE_MAX : (size_t) xnz  ;

        CHOLMOD(reallocate_factor) (lnznew, L, Common) ;
        RETURN_IF_ERROR ;

        //----------------------------------------------------------------------
        // count # of times any factor has been reallocated
        //----------------------------------------------------------------------

        Common->nrealloc_factor++ ;

        //----------------------------------------------------------------------
        // repack all columns so each column has some slack spce
        //----------------------------------------------------------------------

        CHOLMOD(pack_factor) (L, Common) ;
        RETURN_IF_ERROR ;
    }

    //--------------------------------------------------------------------------
    // move j to the end of the list
    //--------------------------------------------------------------------------

    L->is_monotonic = FALSE ;           // L is no longer monotonic
    Lnext [Lprev [j]] = Lnext [j] ;     // remove j from is current place
    Lprev [Lnext [j]] = Lprev [j] ;
    Lnext [Lprev [tail]] = j ;          // place it at the end of the list
    Lprev [j] = Lprev [tail] ;
    Lnext [j] = tail ;
    Lprev [tail] = j ;

    //--------------------------------------------------------------------------
    // add space to L(:,j), now at the end of L
    //--------------------------------------------------------------------------

    Int psrc = Lp [j] ;
    Int pdest = Lp [tail] ;
    Lp [j] = pdest  ;
    Lp [tail] += need ;

    //--------------------------------------------------------------------------
    // move L(:,j) to its new space at the end of L
    //--------------------------------------------------------------------------

    switch ((L->xtype + L->dtype) % 8)
    {
        default:
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_reallocate_column_worker (L, j, pdest, psrc) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_reallocate_column_worker (L, j, pdest, psrc) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_reallocate_column_worker (L, j, pdest, psrc) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_reallocate_column_worker (L, j, pdest, psrc) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_reallocate_column_worker (L, j, pdest, psrc) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_reallocate_column_worker (L, j, pdest, psrc) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // count # of times any L(:,j) has been reallocated
    //--------------------------------------------------------------------------

    Common->nrealloc_col++ ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    DEBUG (CHOLMOD(dump_factor) (L, "realloc col:L output", Common)) ;
    return (TRUE) ;
}

