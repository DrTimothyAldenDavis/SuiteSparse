//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_sparse_to_dense: convert a sparse matrix to dense
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Converts a sparse matrix (as input) to a new dense matrix (as output).
// The xtype and dtype are preserved, except if A->xtype is pattern.  In that
// case, the output matrix X has an xtype of real, and consists of 1's and 0's.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_dense) (&X, Common) ;      \
        return (NULL) ;                         \
    }

//------------------------------------------------------------------------------
// t_cholmod_sparse_to_dense_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define PATTERN
#include "t_cholmod_sparse_to_dense_worker.c"
#define REAL
#include "t_cholmod_sparse_to_dense_worker.c"
#define COMPLEX
#include "t_cholmod_sparse_to_dense_worker.c"
#define ZOMPLEX
#include "t_cholmod_sparse_to_dense_worker.c"

#undef  DOUBLE
#define SINGLE
#define PATTERN
#include "t_cholmod_sparse_to_dense_worker.c"
#define REAL
#include "t_cholmod_sparse_to_dense_worker.c"
#define COMPLEX
#include "t_cholmod_sparse_to_dense_worker.c"
#define ZOMPLEX
#include "t_cholmod_sparse_to_dense_worker.c"

//------------------------------------------------------------------------------
// cholmod_sparse_to_dense
//------------------------------------------------------------------------------

cholmod_dense *CHOLMOD(sparse_to_dense)     // return a dense matrix
(
    cholmod_sparse *A,      // input matrix
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_sparse) (A, "sparse_to_dense:A", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // allocate an all-zero dense matrix
    //--------------------------------------------------------------------------

    int xxtype = (A->xtype == CHOLMOD_PATTERN)? CHOLMOD_REAL : A->xtype ;
    cholmod_dense *X = CHOLMOD(zeros) (A->nrow, A->ncol, xxtype + A->dtype,
        Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // copy A into X
    //--------------------------------------------------------------------------

    switch ((A->xtype + A->dtype) % 8)
    {

        case CHOLMOD_SINGLE + CHOLMOD_PATTERN:
            p_s_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_PATTERN:
            p_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_sparse_to_dense_worker (X, A) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_sparse_to_dense_worker (X, A) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_dense) (X, "sparse_to_dense:X", Common) >= 0) ;
    return (X) ;
}

