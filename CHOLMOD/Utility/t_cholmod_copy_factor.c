//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy_factor: copy a factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Creates an exact copy of a sparse factorization object.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_factor) (&H, Common) ;         \
        return (NULL) ;                             \
    }

//------------------------------------------------------------------------------
// t_cholmod_copy_factor_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_copy_factor_worker.c"
#define COMPLEX
#include "t_cholmod_copy_factor_worker.c"
#define ZOMPLEX
#include "t_cholmod_copy_factor_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_copy_factor_worker.c"
#define COMPLEX
#include "t_cholmod_copy_factor_worker.c"
#define ZOMPLEX
#include "t_cholmod_copy_factor_worker.c"

//------------------------------------------------------------------------------

cholmod_factor *CHOLMOD(copy_factor)    // return a copy of the factor
(
    cholmod_factor *L,      // factor to copy (not modified)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_FACTOR_INVALID (L, FALSE) ;
    Common->status = CHOLMOD_OK ;

    DEBUG (CHOLMOD(dump_factor) (L, "copy_factor:L", Common)) ;

    //--------------------------------------------------------------------------
    // get inputs and sizes of entries
    //--------------------------------------------------------------------------

    size_t n = L->n ;
    size_t ei = sizeof (Int) ;
    size_t e = (L->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((L->xtype == CHOLMOD_PATTERN) ? 0 :
                    ((L->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * ((L->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // allocate the new factor H, H->Perm, and H->ColCount
    //--------------------------------------------------------------------------

    cholmod_factor *H = CHOLMOD(allocate_factor) (n, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // copy the symbolic contents (same for simplicial or supernodal)
    //--------------------------------------------------------------------------

    memcpy (H->Perm,     L->Perm,     n * ei) ;
    memcpy (H->ColCount, L->ColCount, n * ei) ;
    H->ordering = L->ordering ;
    H->is_ll = L->is_ll ;

    //--------------------------------------------------------------------------
    // copy the rest of the factor
    //--------------------------------------------------------------------------

    if (L->is_super)
    {

        //----------------------------------------------------------------------
        // L is a numerical supernodal factor; change H to supernodal
        //----------------------------------------------------------------------

        H->xsize  = L->xsize ;
        H->ssize  = L->ssize ;
        H->nsuper = L->nsuper ;

        CHOLMOD(change_factor) (L->xtype + L->dtype, /* to LL': */ TRUE,
            /* to supernodal: */ TRUE, /* to packed: */ TRUE,
            /* to monotonic: */ TRUE, H, Common) ;
        RETURN_IF_ERROR ;

        //----------------------------------------------------------------------
        // copy the supernodal contents
        //----------------------------------------------------------------------

        H->maxcsize = L->maxcsize ;
        H->maxesize = L->maxesize ;

        memcpy (H->super, L->super, (L->nsuper + 1) * ei) ;
        memcpy (H->pi,    L->pi,    (L->nsuper + 1) * ei) ;
        memcpy (H->px,    L->px,    (L->nsuper + 1) * ei) ;
        memset (H->s,     0,        ei) ;
        memcpy (H->s,     L->s,     (L->ssize) * ei) ;

        if (L->xtype == CHOLMOD_REAL || L->xtype == CHOLMOD_COMPLEX)
        {
            memcpy (H->x, L->x,     (L->xsize) * ex) ;
        }

    }
    else if (L->xtype != CHOLMOD_PATTERN)
    {

        //----------------------------------------------------------------------
        // L is a numerical simplicial factor; change H to the same
        //----------------------------------------------------------------------

        H->nzmax = L->nzmax ;
        CHOLMOD(change_factor) (L->xtype + L->dtype, L->is_ll,
            /* to supernodal: */ FALSE, /* to packed: */ -1,
            /* to monotonic: */ TRUE, H, Common) ;
        RETURN_IF_ERROR ;

        //----------------------------------------------------------------------
        // copy the simplicial contents
        //----------------------------------------------------------------------

        H->xtype = L->xtype ;
        H->dtype = L->dtype ;

        memcpy (H->p,    L->p,    (n+1) * ei) ;
        memcpy (H->prev, L->prev, (n+2) * ei) ;
        memcpy (H->next, L->next, (n+2) * ei) ;
        memcpy (H->nz,   L->nz,   n * ei) ;

        switch ((L->xtype + L->dtype) % 8)
        {

            case CHOLMOD_SINGLE + CHOLMOD_REAL:
                r_s_cholmod_copy_factor_worker (L, H) ;
                break ;

            case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
                c_s_cholmod_copy_factor_worker (L, H) ;
                break ;

            case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
                z_s_cholmod_copy_factor_worker (L, H) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_REAL:
                r_cholmod_copy_factor_worker (L, H) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
                c_cholmod_copy_factor_worker (L, H) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
                z_cholmod_copy_factor_worker (L, H) ;
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the copy and return result
    //--------------------------------------------------------------------------

    H->minor = L->minor ;
    H->is_monotonic = L->is_monotonic ;

    DEBUG (CHOLMOD(dump_factor) (H, "copy_factor:H", Common)) ;
    ASSERT (H->xtype == L->xtype && H->is_super == L->is_super) ;
    return (H) ;
}

