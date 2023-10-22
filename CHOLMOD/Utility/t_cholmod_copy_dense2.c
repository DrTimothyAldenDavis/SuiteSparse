//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense2: copy a dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Copies a dense matrix X into another dense matrix Y, which must already
// be allocated on input.  The dimensions, xtype, and dtype of X and Y must
// match, but X->d and Y->d can differ.

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// t_cholmod_copy_dense2_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_copy_dense2_worker.c"
#define COMPLEX
#include "t_cholmod_copy_dense2_worker.c"
#define ZOMPLEX
#include "t_cholmod_copy_dense2_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_copy_dense2_worker.c"
#define COMPLEX
#include "t_cholmod_copy_dense2_worker.c"
#define ZOMPLEX
#include "t_cholmod_copy_dense2_worker.c"

//------------------------------------------------------------------------------

int CHOLMOD(copy_dense2)
(
    cholmod_dense *X,   // input dense matrix
    cholmod_dense *Y,   // output dense matrix (already allocated on input)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_DENSE_MATRIX_INVALID (X, FALSE) ;
    RETURN_IF_DENSE_MATRIX_INVALID (Y, FALSE) ;
    Common->status = CHOLMOD_OK ;

    if (X->nrow  != Y->nrow  ||
        X->ncol  != Y->ncol  ||
        X->xtype != Y->xtype ||
        X->dtype != Y->dtype)
    {
        ERROR (CHOLMOD_INVALID, "X and Y: wrong dimensions or type") ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t e = (X->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((X->xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;
    size_t ez = e * ((X->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // copy X = Y
    //--------------------------------------------------------------------------

    if (X->d == Y->d)
    {

        //----------------------------------------------------------------------
        // no change of leading dimension: copy all of X and Y as-is
        //----------------------------------------------------------------------

        size_t nz = X->d * X->ncol ;
        if (X->x != NULL) memcpy (Y->x, X->x, nz * ex) ;
        if (X->z != NULL) memcpy (Y->z, X->z, nz * ez) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // copy Y into X, with possible change of leading dimension
        //----------------------------------------------------------------------

        switch ((X->xtype + X->dtype) % 8)
        {
            case CHOLMOD_SINGLE + CHOLMOD_REAL:
                r_s_cholmod_copy_dense2_worker (X, Y) ;
                break ;

            case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
                c_s_cholmod_copy_dense2_worker (X, Y) ;
                break ;

            case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
                z_s_cholmod_copy_dense2_worker (X, Y) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_REAL:
                r_cholmod_copy_dense2_worker (X, Y) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
                c_cholmod_copy_dense2_worker (X, Y) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
                z_cholmod_copy_dense2_worker (X, Y) ;
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (TRUE) ;
}

