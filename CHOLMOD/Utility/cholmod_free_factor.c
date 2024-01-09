//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_free_factor: free a sparse factorization
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// cholmod_to_simplicial_sym
//------------------------------------------------------------------------------

// L is converted into a valid simplicial symbolic object, containing just
// L->Perm and L->ColCount.  This method is used by cholmod_change_factor.  L
// itself is not freed.  This method is for internal use, not for the end-user
// (who should use cholmod_change_factor to access this functionality).

void CHOLMOD(to_simplicial_sym)
(
    cholmod_factor *L,          // sparse factorization to modify
    int to_ll,                  // change L to hold a LL' or LDL' factorization
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Common != NULL) ;
    ASSERT (L != NULL) ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = (L->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((L->xtype == CHOLMOD_PATTERN) ? 0 :
                    ((L->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * ((L->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    size_t nzmax = L->nzmax ;
    size_t n     = L->n ;
    size_t s     = L->nsuper + 1 ;
    size_t xs    = (L->is_super) ? L->xsize : nzmax ;
    size_t ss    = (L->ssize) ;

    //--------------------------------------------------------------------------
    // free the components of L
    //--------------------------------------------------------------------------

    // symbolic part of L (except for L->Perm and L->ColCount)
    L->IPerm = CHOLMOD(free) (n,     ei, L->IPerm,    Common) ;

    // simplicial form of L
    L->p     = CHOLMOD(free) (n+1,   ei, L->p,        Common) ;
    L->i     = CHOLMOD(free) (nzmax, ei, L->i,        Common) ;
    L->nz    = CHOLMOD(free) (n,     ei, L->nz,       Common) ;
    L->next  = CHOLMOD(free) (n+2,   ei, L->next,     Common) ;
    L->prev  = CHOLMOD(free) (n+2,   ei, L->prev,     Common) ;

    // supernodal form of L
    L->pi    = CHOLMOD(free) (s,     ei, L->pi,       Common) ;
    L->px    = CHOLMOD(free) (s,     ei, L->px,       Common) ;
    L->super = CHOLMOD(free) (s,     ei, L->super,    Common) ;
    L->s     = CHOLMOD(free) (ss,    ei, L->s,        Common) ;

    // numerical part of L
    L->x     = CHOLMOD(free) (xs,    ex, L->x,        Common) ;
    L->z     = CHOLMOD(free) (xs,    ez, L->z,        Common) ;

    //--------------------------------------------------------------------------
    // change the header contents to reflect the simplicial symbolic status
    //--------------------------------------------------------------------------

    L->nzmax = 0 ;                  // no entries
    L->is_super = FALSE ;           // L is simplicial
    L->xtype = CHOLMOD_PATTERN ;    // L is symbolic
    L->minor = n ;                  // see cholmod.h
    L->is_ll = to_ll ? 1: 0 ;       // L represents an LL' or LDL' factorization
    L->ssize = 0 ;                  // L->s is not present
    L->xsize = 0 ;                  // L->x is not present
    L->nsuper = 0 ;                 // no supernodes
    L->maxesize = 0 ;               // no rows in any supernodes
    L->maxcsize = 0 ;               // largest update matrix is size zero
}

//------------------------------------------------------------------------------
// cholmod_free_factor
//------------------------------------------------------------------------------

int CHOLMOD(free_factor)
(
    // input/output:
    cholmod_factor **L,         // handle of sparse factorization to free
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    if (L == NULL || (*L) == NULL)
    {
        // L is already freed; nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // convert L to a simplicial symbolic LL' factorization
    //--------------------------------------------------------------------------

    CHOLMOD(to_simplicial_sym) (*L, 1, Common) ;

    //--------------------------------------------------------------------------
    // free the rest of L and return result
    //--------------------------------------------------------------------------

    size_t n = (*L)->n ;
    size_t ei = sizeof (Int) ;
    CHOLMOD(free) (n, ei, (*L)->Perm,     Common) ;
    CHOLMOD(free) (n, ei, (*L)->ColCount, Common) ;
    (*L) = CHOLMOD(free) (1, sizeof (cholmod_factor), (*L), Common) ;
    return (TRUE) ;
}

