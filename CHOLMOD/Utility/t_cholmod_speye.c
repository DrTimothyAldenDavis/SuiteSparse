//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_speye: sparse identity matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Create a sparse identity matrix, possibly rectangular, of any xtype or
// dtype.  The A->stype is zero (unsymmetric) but this can be modified by the
// caller to either +1 or -1, and the matrix will still be valid.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_sparse) (&A, Common) ;     \
        return (NULL) ;                         \
    }

//------------------------------------------------------------------------------
// t_cholmod_speye_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_speye_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_speye_worker.c"
#define COMPLEX
#include "t_cholmod_speye_worker.c"
#define ZOMPLEX
#include "t_cholmod_speye_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_speye_worker.c"
#define COMPLEX
#include "t_cholmod_speye_worker.c"
#define ZOMPLEX
#include "t_cholmod_speye_worker.c"

//------------------------------------------------------------------------------
// cholmod_speye: create a sparse identity matrix
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(speye)
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate the matrix
    //--------------------------------------------------------------------------

    cholmod_sparse *A = CHOLMOD(allocate_sparse) (nrow, ncol, MIN (nrow, ncol),
        /* A is sorted: */ TRUE, /* A is packed: */ TRUE, /* stype: */ 0,
        xdtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the matrix with all 1's on the diagonal
    //--------------------------------------------------------------------------

    switch (xdtype % 8)
    {
        default:
            p_cholmod_speye_worker (A) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_speye_worker (A) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_speye_worker (A) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_speye_worker (A) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_speye_worker (A) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_speye_worker (A) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_speye_worker (A) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (A, "speye:A", Common) >= 0) ;
    return (A) ;
}

