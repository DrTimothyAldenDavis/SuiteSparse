//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_alloc_factor: allocate a simplicial factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Allocates a simplicial symbolic factor, with only L->Perm and L->ColCount
// created and set to empty values (L->Perm is the identity permutation, and
// ColCount is all 1's).  L is pattern.  Unlike cholmod_allocate_factor, the
// factor can be either single or double precision.  L->xtype is
// CHOLMOD_PATTERN.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_factor) (&L, Common) ;     \
        return (NULL) ;                         \
    }

cholmod_factor *CHOLMOD(alloc_factor)       // return the new factor L
(
    // input:
    size_t n,               // L is factorization of an n-by-n matrix
    int dtype,              // CHOLMOD_SINGLE or CHOLMOD_DOUBLE
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    Common->status = CHOLMOD_OK ;

    if ((int64_t) n >= Int_max)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // get the dtype
    //--------------------------------------------------------------------------

    dtype = dtype & 4 ;     // double or single

    //--------------------------------------------------------------------------
    // allocate the header for L
    //--------------------------------------------------------------------------

    cholmod_factor *L = CHOLMOD(calloc) (1, sizeof (cholmod_factor), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the header
    //--------------------------------------------------------------------------

    L->n = n ;                      // # of rows and columns
    L->itype = ITYPE ;              // integer type
    L->dtype = dtype ;              // double or single
    L->is_monotonic = TRUE ;        // columns of L appear in order 0..n-1
    L->minor = n ;                  // L has not been factorized

    //--------------------------------------------------------------------------
    // allocate Perm and ColCount
    //--------------------------------------------------------------------------

    L->Perm     = CHOLMOD(malloc) (n, sizeof (Int), Common) ;
    L->ColCount = CHOLMOD(malloc) (n, sizeof (Int), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // initialize Perm and and ColCount
    //--------------------------------------------------------------------------

    Int *Perm     = (Int *) L->Perm ;
    Int *ColCount = (Int *) L->ColCount ;

    for (Int j = 0 ; j < n ; j++)
    {
        Perm [j] = j ;
        ColCount [j] = 1 ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (L) ;
}

