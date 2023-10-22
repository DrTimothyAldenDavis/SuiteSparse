//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense_nnz: # of nonzeros in a dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Returns the # of nonzero entries in a dense matrix.

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// t_cholmod_dense_nnz_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_dense_nnz_worker.c"
#define COMPLEX
#include "t_cholmod_dense_nnz_worker.c"
#define ZOMPLEX
#include "t_cholmod_dense_nnz_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_dense_nnz_worker.c"
#define COMPLEX
#include "t_cholmod_dense_nnz_worker.c"
#define ZOMPLEX
#include "t_cholmod_dense_nnz_worker.c"

//------------------------------------------------------------------------------
// cholmod_dense_nnz
//------------------------------------------------------------------------------

int64_t CHOLMOD(dense_nnz)      // return # of entries in the dense matrix
(
    cholmod_dense *X,       // input matrix
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_DENSE_MATRIX_INVALID (X, EMPTY) ;
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_dense) (X, "dense_nnz:X", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // count the # of nonzero entries in X
    //--------------------------------------------------------------------------

    int xnz = 0 ;
    switch ((X->xtype + X->dtype) % 8)
    {

        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            xnz = r_s_cholmod_dense_nnz_worker (X) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            xnz = c_s_cholmod_dense_nnz_worker (X) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            xnz = z_s_cholmod_dense_nnz_worker (X) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            xnz = r_cholmod_dense_nnz_worker (X) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            xnz = c_cholmod_dense_nnz_worker (X) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            xnz = z_cholmod_dense_nnz_worker (X) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (xnz) ;
}

