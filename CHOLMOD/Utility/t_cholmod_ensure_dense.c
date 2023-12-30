//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_ensure_dense: ensure dense matrix has a given size
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Ensure a dense matrix has a given size, xtype, and dtype.  If not, it is
// freed and reallocated.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_dense) (X, Common) ;       \
        return (NULL) ;                         \
    }

cholmod_dense *CHOLMOD(ensure_dense)
(
    // input/output:
    cholmod_dense **X,  // matrix to resize as needed (*X may be NULL)
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
    RETURN_IF_NULL (X, NULL) ;
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

    //--------------------------------------------------------------------------
    // get the dimensions
    //--------------------------------------------------------------------------

    d = MAX (d, nrow) ;         // leading dimension d must be >= nrow
    int ok = TRUE ;
    size_t nzmax_required = CHOLMOD(mult_size_t) (d, ncol, &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // reshape or reallocate the matrix
    //--------------------------------------------------------------------------

    if ((*X) != NULL && nzmax_required <= (*X)->nzmax && xtype == (*X)->xtype
        && dtype == (*X)->dtype)
    {
        // The required total size (nzmax_required) is ok, but the dimensions
        // might not be.  This allows an n-by-m matrix to be reconfigured in
        // O(1) time into an m-by-n matrix.  X->nzmax is not changed, so a
        // matrix can be reduced in size in O(1) time, and then enlarged again
        // back to the original size, also in O(1) time.
        (*X)->nrow = nrow ;
        (*X)->ncol = ncol ;
        (*X)->d = d ;
        RETURN_IF_DENSE_MATRIX_INVALID (*X, NULL) ;
    }
    else
    {
        // free the matrix and reallocate it with the right properties
        CHOLMOD(free_dense) (X, Common) ;
        (*X) = CHOLMOD(allocate_dense) (nrow, ncol, d, xdtype, Common) ;

    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (*X) ;
}

