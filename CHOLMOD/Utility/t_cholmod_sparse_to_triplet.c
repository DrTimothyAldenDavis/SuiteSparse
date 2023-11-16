//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_sparse_to_triplet: convert sparse to triplet
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// t_cholmod_sparse_to_triplet_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_sparse_to_triplet_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_sparse_to_triplet_worker.c"
#define COMPLEX
#include "t_cholmod_sparse_to_triplet_worker.c"
#define ZOMPLEX
#include "t_cholmod_sparse_to_triplet_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_sparse_to_triplet_worker.c"
#define COMPLEX
#include "t_cholmod_sparse_to_triplet_worker.c"
#define ZOMPLEX
#include "t_cholmod_sparse_to_triplet_worker.c"

//------------------------------------------------------------------------------
// cholmod_sparse_to_triplet: convert sparse matrix to triplet form
//------------------------------------------------------------------------------

cholmod_triplet *CHOLMOD(sparse_to_triplet)
(
    // input:
    cholmod_sparse *A,      // matrix to copy into triplet form T
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_sparse) (A, "sparse_triplet:A", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // allocate triplet matrix
    //--------------------------------------------------------------------------

    size_t nz = (size_t) CHOLMOD(nnz) (A, Common) ;
    cholmod_triplet *T = CHOLMOD(allocate_triplet) (A->nrow, A->ncol, nz,
        A->stype, A->xtype + A->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // copy a sparse matrix A to a triplet matrix T
    //--------------------------------------------------------------------------

    switch ((A->xtype + A->dtype) % 8)
    {
        default:
            p_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_sparse_to_triplet_worker (T, A) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_triplet) (T, "sparse_to_triplet:T", Common)) ;
    return (T) ;
}

