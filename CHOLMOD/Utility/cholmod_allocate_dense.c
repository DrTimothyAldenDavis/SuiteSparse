//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_allocate_dense: allocate dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Allocate a dense matrix.  The space is not initialized.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_dense) (&X, Common) ;      \
        return (NULL) ;                         \
    }

cholmod_dense *CHOLMOD(allocate_dense)
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t d,       // leading dimension
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // get the xtype and dtype
    //--------------------------------------------------------------------------

    int xtype = xdtype & 3 ;    // real, complex, or zomplex (not pattern)
    int dtype = xdtype & 4 ;    // double or single

    if (xtype == CHOLMOD_PATTERN)
    {
        ERROR (CHOLMOD_INVALID, "xtype invalid") ;
        return (NULL) ;
    }

    d = MAX (d, nrow) ;         // leading dimension d must be >= nrow

    int ok = TRUE ;
    size_t nzmax = CHOLMOD(mult_size_t) (d, ncol, &ok) ;
    if (!ok || nzmax >= Int_max)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // allocate the header
    //--------------------------------------------------------------------------

    cholmod_dense *X = CHOLMOD(calloc) (1, sizeof (cholmod_dense), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the header
    //--------------------------------------------------------------------------

    X->nrow = nrow ;            // # rows
    X->ncol = ncol ;            // # columns
    X->xtype = xtype ;          // real, complex, or zomplex
    X->dtype = dtype ;          // double or single
    X->d = d ;                  // leading dimension

    //--------------------------------------------------------------------------
    // reallocate the dense matrix to change X->nzmax from 0 to nzmax
    //--------------------------------------------------------------------------

    CHOLMOD(realloc_multiple) (nzmax, /* nint: */ 0, xtype + dtype,
        /* I not used: */ NULL, /* J not used: */ NULL, &(X->x), &(X->z),
        &(X->nzmax), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

//  ASSERT (CHOLMOD(dump_dense) (X, "allocate_dense:X", Common) >= 0) ;
    return (X) ;
}

