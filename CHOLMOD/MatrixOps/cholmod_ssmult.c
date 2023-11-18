//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/cholmod_ssmult: sparse-times-sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// C = A*B.  Multiply two sparse matrices.
//
// A and B can be packed or unpacked, sorted or unsorted, and of any stype.  If
// A or B are symmetric, an internal unsymmetric copy is made first, however.
// For the complex case, if A and/or B are symmetric with just their lower or
// upper part stored, they are assumed to be Hermitian when converted.
//
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
// A and B must match (unless mode is zero).

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
    int mode,           // 2: numerical (conj) if A and/or B are symmetric
                        // 1: numerical (non-conj.) if A and/or B are symmetric
                        // 0: pattern
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
    mode = RANGE (mode, 0, 2) ;
    if (A->xtype == CHOLMOD_PATTERN || B->xtype == CHOLMOD_PATTERN)
    {
        mode = 0 ;
    }
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;
    RETURN_IF_XTYPE_INVALID (B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;
    if (A->ncol != B->nrow)
    {
        // inner dimensions must agree
        ERROR (CHOLMOD_INVALID, "A and B inner dimensions must match") ;
        return (NULL) ;
    }

    bool values = (mode != 0) ;
    if (values && (A->xtype != B->xtype || A->dtype != B->dtype))
    {
        // A and B must have the same numerical type if mode != 0
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
        A2 = CHOLMOD(copy) (A, 0, mode, Common) ;
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
        B2 = CHOLMOD(copy) (B, 0, mode, Common) ;
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
    void *W = Common->Xwork ;   // size nrow, unused if values is false
    Int *Flag = Common->Flag ;  // size nrow, Flag [0..nrow-1] < mark on input

    //--------------------------------------------------------------------------
    // count the number of entries in the result C
    //--------------------------------------------------------------------------

    int ok = TRUE ;
    size_t cnz = 0 ;
    size_t cnzmax = SIZE_MAX - A->nrow ;
    for (Int j = 0 ; ok && (j < ncol) ; j++)
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
        ok = (cnz < cnzmax) ;
    }

    CLEAR_FLAG (Common) ;
    ASSERT (check_flag (Common)) ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    C = CHOLMOD(allocate_sparse) (nrow, ncol, ok ? cnz : SIZE_MAX, FALSE, TRUE,
        0, (values ? A->xtype : CHOLMOD_PATTERN) + A->dtype, Common) ;
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

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_ssmult_worker (C, A, B, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_ssmult_worker (C, A, B, Common) ;
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
        // this cannot fail (no workspace; sort is done in-place)
        CHOLMOD(sort) (C, Common) ;
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

