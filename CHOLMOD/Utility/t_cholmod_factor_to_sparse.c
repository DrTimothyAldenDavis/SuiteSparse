//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_factor_to_sparse: convert factor to sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Creates a sparse matrix A from a factor L.  The contents of L are
// transferred into A, and L is returned as a simplicial symbolic factor.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                     \
    if (Common->status < CHOLMOD_OK)        \
    {                                       \
        CHOLMOD(free_sparse) (&A, Common) ; \
        return (NULL) ;                     \
    }

cholmod_sparse *CHOLMOD(factor_to_sparse)
(
    // input/output:
    cholmod_factor *L,  // input: factor to convert; output: L is converted
                        // to a simplicial symbolic factor
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_FACTOR_INVALID (L, NULL) ;
    Common->status = CHOLMOD_OK ;

    DEBUG (CHOLMOD(dump_factor) (L, "factor_to_sparse:L input", Common)) ;

    if (L->xtype == CHOLMOD_PATTERN)
    {
        ERROR (CHOLMOD_INVALID, "L must be numerical on input") ;
        return (NULL) ;
    }

    cholmod_sparse *A = NULL ;

    //--------------------------------------------------------------------------
    // convert L in place
    //--------------------------------------------------------------------------

    CHOLMOD(change_factor) (L->xtype, L->is_ll,
        /* L becomes simplicial: */ FALSE,
        /* L becomes packed: */ TRUE,
        /* L becomes monotonic: */ TRUE, L, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // allocate the sparse matrix A
    //--------------------------------------------------------------------------

    A = CHOLMOD(calloc) (1, sizeof (cholmod_sparse), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // move the contents of L into A, and make L pattern
    //--------------------------------------------------------------------------

    A->nrow = L->n ;
    A->ncol = L->n ;
    A->p    = L->p ;        L->p = NULL ;
    A->i    = L->i ;        L->i = NULL ;
    A->x    = L->x ;        L->x = NULL ;
    A->z    = L->z ;        L->z = NULL ;

    A->stype = 0 ;
    A->itype = L->itype ;
    A->xtype = L->xtype ;   L->xtype = CHOLMOD_PATTERN ;
    A->dtype = L->dtype ;

    A->sorted = TRUE ;
    A->packed = TRUE ;
    A->nzmax = L->nzmax ;

    CHOLMOD(change_factor) (CHOLMOD_PATTERN, FALSE, FALSE, TRUE, TRUE, L,
            Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (A, "factor to sparse:A", Common) >= 0) ;
    DEBUG (CHOLMOD(dump_factor) (L, "factor_to_sparse:L input", Common)) ;
    return (A) ;
}

