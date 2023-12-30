//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense_to_sparse: convert a dense matrix to sparse
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Converts a dense matrix X (as input) to a new sparse matrix C (as output).
// The xtype and dtype are preserved, except if mode is 0 in which case C is
// returned as a pattern sparse matrix.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_sparse) (&C, Common) ;     \
        return (NULL) ;                         \
    }

//------------------------------------------------------------------------------
// t_cholmod_dense_to_sparse_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_dense_to_sparse_worker.c"
#define COMPLEX
#include "t_cholmod_dense_to_sparse_worker.c"
#define ZOMPLEX
#include "t_cholmod_dense_to_sparse_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_dense_to_sparse_worker.c"
#define COMPLEX
#include "t_cholmod_dense_to_sparse_worker.c"
#define ZOMPLEX
#include "t_cholmod_dense_to_sparse_worker.c"

//------------------------------------------------------------------------------
// cholmod_dense_to_sparse
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(dense_to_sparse)        // return a sparse matrix C
(
    // input:
    cholmod_dense *X,       // input matrix
    int mode,               // 1: copy the values
                            // 0: C is pattern
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_DENSE_MATRIX_INVALID (X, NULL) ;
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_dense) (X, "dense_to_sparse:X", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // allocate the sparse matrix result C
    //--------------------------------------------------------------------------

    mode = RANGE (mode, 0, 1) ;
    int cnz = CHOLMOD(dense_nnz) (X, Common) ;
    int cxtype = (mode == 0) ? CHOLMOD_PATTERN : X->xtype ;
    cholmod_sparse *C = CHOLMOD(allocate_sparse) (X->nrow, X->ncol, cnz,
        /* C is sorted: */ TRUE, /* C is packed: */ TRUE, /* C->stype: */ 0,
        cxtype + X->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // copy the nonzeros (or just their pattern) from X into C
    //--------------------------------------------------------------------------

    switch ((X->xtype + X->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_dense_to_sparse_worker (C, X) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_dense_to_sparse_worker (C, X) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_dense_to_sparse_worker (C, X) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_dense_to_sparse_worker (C, X) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_dense_to_sparse_worker (C, X) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_dense_to_sparse_worker (C, X) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (C, "sparse_to_dense:C", Common) >= 0) ;
    return (C) ;
}

