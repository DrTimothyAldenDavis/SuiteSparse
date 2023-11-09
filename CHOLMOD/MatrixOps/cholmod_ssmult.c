//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/cholmod_ssmult: sparse-times-sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// C = A*B.  Multiply two sparse matrices.
//
// A and B can be packed or unpacked, sorted or unsorted, and of any stype.
// If A or B are symmetric, an internal unsymmetric copy is made first, however.
// C is computed as if A and B are unsymmetric, and then if the stype input
// parameter requests a symmetric form (upper or lower) the matrix is converted
// into that form.
//
// C is returned as packed, and either unsorted or sorted, depending on the
// "sorted" input parameter.  If C is returned sorted, then either C = (B'*A')'
// or C = (A*B)'' is computed, depending on the number of nonzeros in A, B, and
// C.
//
// workspace:
//      if C unsorted: Flag (A->nrow), W (A->nrow) if values
//      if C sorted:   Flag (B->ncol), W (B->ncol) if values
//      Iwork (max (A->ncol, A->nrow, B->nrow, B->ncol))
//      allocates temporary copies for A, B, and C, if required.
//
// Matrices of any xtype and dtype supported, but the xtype and dtype of
// A and B must match (unless values is FALSE).

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMATRIXOPS

//------------------------------------------------------------------------------
// t_cholmod_ssmult_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_ssmult_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_ssmult_worker.c"
#define COMPLEX
#include "t_cholmod_ssmult_worker.c"
#define ZOMPLEX
#include "t_cholmod_ssmult_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_ssmult_worker.c"
#define COMPLEX
#include "t_cholmod_ssmult_worker.c"
#define ZOMPLEX
#include "t_cholmod_ssmult_worker.c"

//------------------------------------------------------------------------------
// cholmod_ssmult
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(ssmult)     // return C=A*B
(
    // input:
    cholmod_sparse *A,  // left matrix to multiply
    cholmod_sparse *B,  // right matrix to multiply
    int stype,          // requested stype of C
    int values,         // TRUE: do numerical values, FALSE: pattern only
    int sorted,         // if TRUE then return C with sorted columns
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_sparse *C = NULL, *A2 = NULL, *B2 = NULL, *C2 = NULL ;

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    values = values &&
        (A->xtype != CHOLMOD_PATTERN) && (B->xtype != CHOLMOD_PATTERN) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;
    RETURN_IF_XTYPE_INVALID (B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;
    if (A->ncol != B->nrow)
    {
        // inner dimensions must agree
        ERROR (CHOLMOD_INVALID, "A and B inner dimensions must match") ;
        return (NULL) ;
    }

    if (values && (A->xtype != B->xtype || A->dtype != B->dtype))
    {
        // A and B must have the same numerical type if values is TRUE
        ERROR (CHOLMOD_INVALID, "A and B must have the same xtype and dtype") ;
        return (NULL) ;
    }

    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    if (A->nrow <= 1)
    {
        // C will be implicitly sorted, so no need to sort it here
        sorted = FALSE ;
    }
    size_t n1 ;
    if (sorted)
    {
        n1 = MAX (A->nrow, B->ncol) ;
    }
    else
    {
        n1 = A->nrow ;
    }
    size_t n2 = (size_t) MAX4 (A->ncol, A->nrow, B->nrow, B->ncol) ;
    size_t nw = ((A->xtype >= CHOLMOD_COMPLEX) ? 2 : 1) * (values ? n1 : 0) ;
    CHOLMOD(alloc_work) (n1, n2, nw, A->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    // convert A to unsymmetric, if necessary
    A2 = NULL ;
    B2 = NULL ;
    if (A->stype)
    {
        // workspace: Iwork (max (A->nrow,A->ncol))
        A2 = CHOLMOD(copy) (A, 0, values, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;
            return (NULL) ;
        }
        A = A2 ;
    }

    // convert B to unsymmetric, if necessary
    if (B->stype)
    {
        // workspace: Iwork (max (B->nrow,B->ncol))
        B2 = CHOLMOD(copy) (B, 0, values, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            CHOLMOD(free_sparse) (&A2, Common) ;
            ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;
            return (NULL) ;
        }
        B = B2 ;
    }

    ASSERT (CHOLMOD(dump_sparse) (A, "A", Common) >= 0) ;
    ASSERT (CHOLMOD(dump_sparse) (B, "B", Common) >= 0) ;

    // get the A matrix
    Int *Ap  = A->p ;
    Int *Anz = A->nz ;
    Int *Ai  = A->i ;
    bool apacked = A->packed ;

    // get the B matrix
    Int *Bp  = B->p ;
    Int *Bnz = B->nz ;
    Int *Bi  = B->i ;
    bool bpacked = B->packed ;

    // get the size of C
    Int nrow = A->nrow ;
    Int ncol = B->ncol ;

    // get workspace
    void *W = Common->Xwork ;   // size nrow, unused if values is FALSE
    Int *Flag = Common->Flag ;  // size nrow, Flag [0..nrow-1] < mark on input

    //--------------------------------------------------------------------------
    // count the number of entries in the result C
    //--------------------------------------------------------------------------

    Int cnz = 0 ;
    for (Int j = 0 ; j < ncol ; j++)
    {
        // clear the Flag array
        CLEAR_FLAG (Common) ;
        Int mark = Common->mark ;

        // for each nonzero B(k,j) in column j, do:
        Int pb = Bp [j] ;
        Int pbend = (bpacked) ? (Bp [j+1]) : (pb + Bnz [j]) ;
        for ( ; pb < pbend ; pb++)
        {
            // B(k,j) is nonzero
            Int k = Bi [pb] ;

            // add the nonzero pattern of A(:,k) to the pattern of C(:,j)
            Int pa = Ap [k] ;
            Int paend = (apacked) ? (Ap [k+1]) : (pa + Anz [k]) ;
            for ( ; pa < paend ; pa++)
            {
                Int i = Ai [pa] ;
                if (Flag [i] != mark)
                {
                    Flag [i] = mark ;
                    cnz++ ;
                }
            }
        }
        if (cnz < 0)
        {
            break ;         // integer overflow case
        }
    }

    CLEAR_FLAG (Common) ;
    ASSERT (check_flag (Common)) ;

    //--------------------------------------------------------------------------
    // check for integer overflow
    //--------------------------------------------------------------------------

    if (cnz < 0)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        CHOLMOD(free_sparse) (&A2, Common) ;
        CHOLMOD(free_sparse) (&B2, Common) ;
        ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    C = CHOLMOD(allocate_sparse) (nrow, ncol, cnz, FALSE, TRUE, 0,
            values ? A->xtype : CHOLMOD_PATTERN, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        CHOLMOD(free_sparse) (&A2, Common) ;
        CHOLMOD(free_sparse) (&B2, Common) ;
        ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;
        return (NULL) ;
    }

    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;

    //--------------------------------------------------------------------------
    // C = A*B
    //--------------------------------------------------------------------------

    switch ((C->xtype + C->dtype) % 8)
    {

        default:
            p_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // clear workspace and free temporary matrices
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&A2, Common) ;
    CHOLMOD(free_sparse) (&B2, Common) ;
    CLEAR_FLAG (Common) ;
    ASSERT (check_flag (Common)) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;

    //--------------------------------------------------------------------------
    // convert C to a symmetric upper/lower matrix if requested
    //--------------------------------------------------------------------------

    // convert C in place, which cannot fail since no memory is allocated
    if (stype > 0)
    {
        // C = triu (C), in place
        (void) CHOLMOD(band_inplace) (0, ncol, values, C, Common) ;
        C->stype = 1 ;
    }
    else if (stype < 0)
    {
        // C = tril (C), in place
        (void) CHOLMOD(band_inplace) (-nrow, 0, values, C, Common) ;
        C->stype = -1 ;
    }
    ASSERT (Common->status >= CHOLMOD_OK) ;

    //--------------------------------------------------------------------------
    // sort C, if requested
    //--------------------------------------------------------------------------

    if (sorted)
    {
        // workspace: Iwork (max (C->nrow,C->ncol))
        if (!CHOLMOD(sort) (C, Common))
        {
            // out of memory
            CHOLMOD(free_sparse) (&C, Common) ;
            ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;
            return (NULL) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (C, "ssmult", Common) >= 0) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, nw, A->dtype, Common)) ;
    return (C) ;
}

#endif
#endif

