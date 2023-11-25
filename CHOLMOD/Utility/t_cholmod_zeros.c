//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_zeros: allocate an all-zero dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Allocate a dense matrix.  The space is set to zero.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_dense) (&X, Common) ;      \
        return (NULL) ;                         \
    }

cholmod_dense *CHOLMOD(zeros)
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
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
    // allocate a dense matrix
    //--------------------------------------------------------------------------

    cholmod_dense *X = CHOLMOD(allocate_dense) (nrow, ncol, nrow, xdtype,
        Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // get the xtype and dtype
    //--------------------------------------------------------------------------

    int xtype = xdtype & 3 ;    // real, complex, or zomplex (not pattern)
    int dtype = xdtype & 4 ;    // double or single

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t e = (dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;
    size_t ez = e * ((xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // clear the contents
    //--------------------------------------------------------------------------

    if (X->x != NULL) memset (X->x, 0, X->nzmax * ex) ;
    if (X->z != NULL) memset (X->z, 0, X->nzmax * ez) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_dense) (X, "zeros:X", Common) >= 0) ;
    return (X) ;
}

