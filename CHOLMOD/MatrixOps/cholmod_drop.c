//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/cholmod_drop: drop small entries from a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Drop small entries from A, and entries in the ignored part of A if A
// is symmetric.  None of the matrix operations drop small numerical entries
// from a matrix, except for this one.  NaN's and Inf's are kept.
//
// workspace: none
//
// Supports any xtype (pattern, real, complex, zomplex) and any dtype
// (single or double).

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMATRIXOPS

//------------------------------------------------------------------------------
// t_cholmod_drop_worker
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_drop_worker.c"
#define COMPLEX
#include "t_cholmod_drop_worker.c"
#define ZOMPLEX
#include "t_cholmod_drop_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_drop_worker.c"
#define COMPLEX
#include "t_cholmod_drop_worker.c"
#define ZOMPLEX
#include "t_cholmod_drop_worker.c"

//------------------------------------------------------------------------------
// cholmod_drop
//------------------------------------------------------------------------------

int CHOLMOD(drop)
(
    // input:
    double tol,         // keep entries with absolute value > tol
    // input/output:
    cholmod_sparse *A,  // matrix to drop entries from
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_sparse) (A, "A predrop", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // drop small entries from A
    //--------------------------------------------------------------------------

    switch ((A->xtype + A->dtype) % 8)
    {

        default:
            // pattern only: just drop entries outside the lower/upper part,
            // if A is symmetric
            if (A->stype > 0)
            {
                CHOLMOD(band_inplace) (0, A->ncol, 0, A, Common) ;
            }
            else if (A->stype < 0)
            {
                CHOLMOD(band_inplace) (-(A->nrow), 0, 0, A, Common) ;
            }
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_drop_worker (tol, A, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_drop_worker (tol, A, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_drop_worker (tol, A, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_drop_worker (tol, A, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_drop_worker (tol, A, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_drop_worker (tol, A, Common) ;
            break ;
    }

    ASSERT (CHOLMOD(dump_sparse) (A, "A dropped", Common) >= 0) ;
    return (TRUE) ;
}

#endif
#endif

