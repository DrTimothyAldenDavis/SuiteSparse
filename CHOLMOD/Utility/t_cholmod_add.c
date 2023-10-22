//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_add: C = alpha*A + beta*B
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// The C matrix is returned as packed and sorted.  A and B must have the same
// xtype, unless one of them is pattern, in which case only the patterns of A
// and B are used, and C has xtype of pattern.  A, B, and C must all have the
// same dtype.  If A->stype and B->stype differ, then C is computed as
// unsymmetric (stype of 0).

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_sparse) (&C, Common) ;         \
        CHOLMOD(free_sparse) (&A2, Common) ;        \
        CHOLMOD(free_sparse) (&B2, Common) ;        \
        return (NULL) ;                             \
    }

//------------------------------------------------------------------------------
// t_cholmod_add_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_add_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_add_worker.c"
#define COMPLEX
#include "t_cholmod_add_worker.c"
#define ZOMPLEX
#include "t_cholmod_add_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_add_worker.c"
#define COMPLEX
#include "t_cholmod_add_worker.c"
#define ZOMPLEX
#include "t_cholmod_add_worker.c"

//------------------------------------------------------------------------------
// cholmod_add: C = alpha*A + beta*B
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(add)    // return C = alpha*A + beta*B
(
    cholmod_sparse *A,  // input matrix
    cholmod_sparse *B,  // input matrix
    double alpha [2],   // scale factor for A (two entires used if complex)
    double beta [2],    // scale factor for B (two entries used if complex)
    int values,         // if TRUE compute the numerical values of C
    int sorted,         // ignored; C is now always returned as sorted
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs and determine the xtype and dtype of C
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (B, NULL) ;
    ASSERT (CHOLMOD(dump_sparse) (A, "add:A", Common) >= 0) ;
    ASSERT (CHOLMOD(dump_sparse) (B, "add:B", Common) >= 0) ;
    Common->status = CHOLMOD_OK ;
    cholmod_sparse *A2 = NULL, *B2 = NULL, *C = NULL ;

    if (A->nrow != B->nrow || A->ncol != B->ncol)
    {
        ERROR (CHOLMOD_INVALID, "A and B dimensions do not match") ;
        return (NULL) ;
    }

    int axtype = A->xtype ;
    int bxtype = B->xtype ;
    if (!values || axtype == CHOLMOD_PATTERN || bxtype == CHOLMOD_PATTERN)
    {
        // treat A and B as if they are pattern-only matrices; C is pattern
        values = FALSE ;
        axtype = CHOLMOD_PATTERN ;
        bxtype = CHOLMOD_PATTERN ;
    }

    if (axtype != bxtype)
    {
        ERROR (CHOLMOD_INVALID, "A and B xtypes do not match") ;
        return (NULL) ;
    }

    if (values && A->dtype != B->dtype)
    {
        ERROR (CHOLMOD_INVALID, "A and B dtypes do not match") ;
        return (NULL) ;
    }

    int xtype = axtype ;
    int dtype = A->dtype ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries of C, A, and B
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = (dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((xtype == CHOLMOD_PATTERN) ? 0 :
                    ((xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * ((xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // convert/sort A and/or B, if needed
    //--------------------------------------------------------------------------

    int mode = values ? 1 : 0 ;

    if (A->stype == B->stype)
    {

        //----------------------------------------------------------------------
        // A and B have the same stype, but make sure they are both sorted
        //----------------------------------------------------------------------

        // A2 = sorted copy of A, if A is not sorted, with same stype as A
        if (!A->sorted)
        {
            A2 = CHOLMOD(copy) (A, A->stype, mode, Common) ;
            RETURN_IF_ERROR ;
            CHOLMOD(sort) (A2, Common) ;
            RETURN_IF_ERROR ;
            A = A2 ;
        }

        // B2 = sorted copy of B, if B is not sorted, with same stype as B
        if (!B->sorted)
        {
            B2 = CHOLMOD(copy) (B, B->stype, mode, Common) ;
            RETURN_IF_ERROR ;
            CHOLMOD(sort) (B2, Common) ;
            RETURN_IF_ERROR ;
            B = B2 ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // the stype of A and B differ, so make both unsymmetric and sorted
        //----------------------------------------------------------------------

        if (!(A->stype == 0 && A->sorted))
        {
            // A2 = sorted unsymmetric copy of A with stype of zero
            A2 = CHOLMOD(copy) (A, 0, mode, Common) ;
            RETURN_IF_ERROR ;
            if (!A2->sorted)
            {
                CHOLMOD(sort) (A2, Common) ;
                RETURN_IF_ERROR ;
            }
            A = A2 ;
        }

        if (!(B->stype == 0 && B->sorted))
        {
            // B2 = sorted unsymmetric copy of A with stype of zero
            B2 = CHOLMOD(copy) (B, 0, mode, Common) ;
            RETURN_IF_ERROR ;
            if (!B2->sorted)
            {
                CHOLMOD(sort) (B2, Common) ;
                RETURN_IF_ERROR ;
            }
            B = B2 ;
        }
    }

    // C, A, and B now all have the same stype, and are sorted
    ASSERT (A->stype == B->stype) ;
    ASSERT (A->sorted) ;
    ASSERT (B->sorted) ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    size_t nzmax = CHOLMOD(nnz) (A, Common) + CHOLMOD(nnz) (B, Common) ;
    C = CHOLMOD(allocate_sparse) (A->nrow, A->ncol, nzmax,
        /* C is sorted: */ TRUE, /* C is packed: */ TRUE,
        A->stype, xtype + dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // C = alpha*A + beta*B
    //--------------------------------------------------------------------------

    switch ((xtype + dtype) % 8)
    {
        default:
            p_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_add_worker (C, A, B, alpha, beta, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // free temporary copies of A and B, if created
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&A2, Common) ;
    CHOLMOD(free_sparse) (&B2, Common) ;

    //--------------------------------------------------------------------------
    // compact the space of C
    //--------------------------------------------------------------------------

    size_t cnz = CHOLMOD(nnz) (C, Common) ;
    CHOLMOD(reallocate_sparse) (cnz, C, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (C, "add:C", Common) >= 0) ;
    return (C) ;
}

