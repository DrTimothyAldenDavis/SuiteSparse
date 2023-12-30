//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/cholmod_vertcat: vertical concatenation: C=[A;B]
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Vertical concatenation, C = [A ; B] in MATLAB notation.
//
// A and B can be up/lo/unsym; C is unsymmetric and packed.
// A and B must have the same number of columns.
// C is sorted if both A and B are sorted.
//
// workspace: Iwork (max (A->nrow, A->ncol, B->nrow, B->ncol)).
//      allocates temporary copies of A and B if they are symmetric.
//
// A and B must have the same xtype and dtype, unless mode is 0.

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMATRIXOPS

//------------------------------------------------------------------------------
// t_cholmod_vertcat_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_vertcat_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_vertcat_worker.c"
#define COMPLEX
#include "t_cholmod_vertcat_worker.c"
#define ZOMPLEX
#include "t_cholmod_vertcat_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_vertcat_worker.c"
#define COMPLEX
#include "t_cholmod_vertcat_worker.c"
#define ZOMPLEX
#include "t_cholmod_vertcat_worker.c"

//------------------------------------------------------------------------------
// cholmod_vertcat
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(vertcat)
(
    // input:
    cholmod_sparse *A,  // top matrix to concatenate
    cholmod_sparse *B,  // bottom matrix to concatenate
    int mode,           // 2: numerical (conj) if A and/or B are symmetric
                        // 1: numerical (non-conj.) if A and/or B are symmetric
                        // 0: pattern
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_sparse *C = NULL, *A2 = NULL, *B2 = NULL ;

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_NULL (B, NULL) ;

    mode = RANGE (mode, 0, 2) ;
    if (A->xtype == CHOLMOD_PATTERN || B->xtype == CHOLMOD_PATTERN)
    {
        mode = 0 ;
    }
    bool values = (mode != 0) ;

    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;
    RETURN_IF_XTYPE_INVALID (B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;
    if (A->ncol != B->ncol)
    {
        // A and B must have the same number of columns
        ERROR (CHOLMOD_INVALID, "A and B must have same # of columns") ;
        return (NULL) ;
    }
    if (mode != 0 && (A->xtype != B->xtype || A->dtype != B->dtype))
    {
        // A and B must have the same xtype and dtype if mode is 0
        ERROR (CHOLMOD_INVALID, "A and B must have same xtype and dtype") ;
        return (NULL) ;
    }

    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    Int anrow = A->nrow ;
    Int bnrow = B->nrow ;
    Int ncol = A->ncol ;
    CHOLMOD(allocate_work) (0, MAX3 (anrow, bnrow, ncol), 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // convert A and/or B if necessary
    //--------------------------------------------------------------------------

    // convert A to unsymmetric, if necessary
    A2 = NULL ;
    if (A->stype != 0)
    {
        // workspace: Iwork (max (A->nrow,A->ncol))
        A2 = CHOLMOD(copy) (A, 0, mode, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (NULL) ;
        }
        A = A2 ;
    }

    // convert B to unsymmetric, if necessary
    B2 = NULL ;
    if (B->stype != 0)
    {
        // workspace: Iwork (max (B->nrow,B->ncol))
        B2 = CHOLMOD(copy) (B, 0, mode, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            CHOLMOD(free_sparse) (&A2, Common) ;
            return (NULL) ;
        }
        B = B2 ;
    }

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    Int anz = CHOLMOD(nnz) (A, Common) ;
    Int bnz = CHOLMOD(nnz) (B, Common) ;
    Int nrow = anrow + bnrow ;
    Int nz = anz + bnz ;

    C = CHOLMOD(allocate_sparse) (nrow, ncol, nz, A->sorted && B->sorted, TRUE,
            0, (values ? A->xtype : CHOLMOD_PATTERN) + A->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        CHOLMOD(free_sparse) (&A2, Common) ;
        CHOLMOD(free_sparse) (&B2, Common) ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // C = [A ; B]
    //--------------------------------------------------------------------------

    switch ((C->xtype + C->dtype) % 8)
    {
        default:
            p_cholmod_vertcat_worker (C, A, B) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_vertcat_worker (C, A, B) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_vertcat_worker (C, A, B) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_vertcat_worker (C, A, B) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_vertcat_worker (C, A, B) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_vertcat_worker (C, A, B) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_vertcat_worker (C, A, B) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // free the unsymmetric copies of A and B, and return C
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&A2, Common) ;
    CHOLMOD(free_sparse) (&B2, Common) ;
    return (C) ;
}

#endif
#endif

