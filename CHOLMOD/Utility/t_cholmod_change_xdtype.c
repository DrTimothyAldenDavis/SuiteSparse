//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_xtype: change xtype and/or dtype of an object
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// t_change_xdtype_template
//------------------------------------------------------------------------------

#define CHANGE_XDTYPE2 change_xdtype_d2d
#define Real_IN  double
#define Real_OUT double
#include "t_cholmod_change_xdtype_template.c"

#define CHANGE_XDTYPE2 change_xdtype_d2s
#define Real_IN  double
#define Real_OUT float
#include "t_cholmod_change_xdtype_template.c"

#define CHANGE_XDTYPE2 change_xdtype_s2d
#define Real_IN  float
#define Real_OUT double
#include "t_cholmod_change_xdtype_template.c"

#define CHANGE_XDTYPE2 change_xdtype_s2s
#define Real_IN  float
#define Real_OUT float
#include "t_cholmod_change_xdtype_template.c"

//------------------------------------------------------------------------------
// change_xdtype: change the xtype and/or dtype of an array
//------------------------------------------------------------------------------

static int change_xdtype
(
    Int nz,                 // # of entries
    int *input_xtype,       // xtype of input:  pattern, real, complex, zomplex
    int output_xtype,       // xtype of output: pattern, real, complex, zomplex
    int *input_dtype,       // dtype of input:  double or single
    int output_dtype,       // dtype of output: double or single
    void **X,               // X array for real, complex, zomplex cases
    void **Z,               // Z array for zomplex case
    cholmod_common *Common
)
{
    if (*input_dtype == CHOLMOD_DOUBLE)
    {
        if (output_dtype == CHOLMOD_DOUBLE)
        {
            // double to double
            return (change_xdtype_d2d (nz, input_xtype, output_xtype,
                input_dtype, output_dtype, X, Z, Common)) ;
        }
        else
        {
            // double to single
            return (change_xdtype_d2s (nz, input_xtype, output_xtype,
                input_dtype, output_dtype, X, Z, Common)) ;
        }
    }
    else
    {
        if (output_dtype == CHOLMOD_DOUBLE)
        {
            // single to double
            return (change_xdtype_s2d (nz, input_xtype, output_xtype,
                input_dtype, output_dtype, X, Z, Common)) ;
        }
        else
        {
            // single to single
            return (change_xdtype_s2s (nz, input_xtype, output_xtype,
                input_dtype, output_dtype, X, Z, Common)) ;
        }
    }
}

//------------------------------------------------------------------------------
// cholmod_sparse_xtype: change xtype and/or dtype of a sparse matrix
//------------------------------------------------------------------------------

int CHOLMOD(sparse_xtype)
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_sparse *A,  // sparse matrix to change
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, FALSE) ;

    //--------------------------------------------------------------------------
    // change the xtype and/or dtype
    //--------------------------------------------------------------------------

    int output_xtype = to_xdtype & 3 ;  // pattern, real, complex, or zomplex
    int output_dtype = to_xdtype & 4 ;  // double or single

    return (change_xdtype (A->nzmax, &(A->xtype), output_xtype,
        &(A->dtype), output_dtype, &(A->x), &(A->z), Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_triplet_xtype: change xtype and/or dtype of a triplet matrix
//------------------------------------------------------------------------------

int CHOLMOD(triplet_xtype)
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_triplet *T, // triplet matrix to change
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_TRIPLET_MATRIX_INVALID (T, FALSE) ;

    //--------------------------------------------------------------------------
    // change the xtype and/or dtype
    //--------------------------------------------------------------------------

    int output_xtype = to_xdtype & 3 ;  // pattern, real, complex, or zomplex
    int output_dtype = to_xdtype & 4 ;  // double or single

    return (change_xdtype (T->nzmax, &(T->xtype), output_xtype,
        &(T->dtype), output_dtype, &(T->x), &(T->z), Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_dense_xtype: change xtype and/or dtype of a dense matrix
//------------------------------------------------------------------------------

int CHOLMOD(dense_xtype)
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_dense *X,   // dense matrix to change
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_DENSE_MATRIX_INVALID (X, FALSE) ;

    //--------------------------------------------------------------------------
    // change the xtype and/or dtype
    //--------------------------------------------------------------------------

    int output_xtype = to_xdtype & 3 ;  // real, complex, or zomplex
    int output_dtype = to_xdtype & 4 ;  // double or single

    if (output_xtype <= CHOLMOD_PATTERN)
    {
        // output_xtype not supported
        ERROR (CHOLMOD_INVALID, "invalid xtype") ;
        return (FALSE) ; 
    }

    return (change_xdtype (X->nzmax, &(X->xtype), output_xtype,
        &(X->dtype), output_dtype, &(X->x), &(X->z), Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_factor_xtype: change xtype and/or dtype of a factor
//------------------------------------------------------------------------------

int CHOLMOD(factor_xtype)
(
    int to_xdtype,      // requested xtype and dtype
    cholmod_factor *L,  // factor to change
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_FACTOR_INVALID (L, FALSE) ;

    //--------------------------------------------------------------------------
    // change the xtype and/or dtype
    //--------------------------------------------------------------------------

    int output_xtype = to_xdtype & 3 ;  // real, complex, or zomplex
    int output_dtype = to_xdtype & 4 ;  // double or single

    if (output_xtype <= CHOLMOD_PATTERN || 
        L->is_super && output_xtype == CHOLMOD_ZOMPLEX)
    {
        // output_xtype not supported
        ERROR (CHOLMOD_INVALID, "invalid xtype") ;
        return (FALSE) ; 
    }

    Int nzmax = L->is_super ? L->xsize : L->nzmax ;

    return (change_xdtype (nzmax, &(L->xtype), output_xtype,
        &(L->dtype), output_dtype, &(L->x), &(L->z), Common)) ;
}

