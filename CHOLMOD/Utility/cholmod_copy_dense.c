//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense: copy a dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Copies a dense matrix X into a new dense matrix Y, with the same leading
// dimensions.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_dense) (&Y, Common) ;          \
        return (NULL) ;                             \
    }

cholmod_dense *CHOLMOD(copy_dense)  // returns new dense matrix
(
    // input:
    cholmod_dense *X,   // input dense matrix
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_DENSE_MATRIX_INVALID (X, FALSE) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate the output matrix Y with same properties as X
    //--------------------------------------------------------------------------

    cholmod_dense *Y = CHOLMOD(allocate_dense) (X->nrow, X->ncol, X->d,
        X->xtype + X->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // copy X = Y
    //--------------------------------------------------------------------------

    CHOLMOD(copy_dense2) (X, Y, Common) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (Y) ;
}

