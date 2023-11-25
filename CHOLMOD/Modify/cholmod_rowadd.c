//------------------------------------------------------------------------------
// CHOLMOD/Modify/cholmod_rowadd: add row/column to an LDL' factorization
//------------------------------------------------------------------------------

// CHOLMOD/Modify Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// and William W. Hager. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Adds a row and column to an LDL' factorization, and optionally updates the
// solution to Lx=b.
//
// workspace: Flag (nrow), Head (nrow+1), W (2*nrow), Iwork (2*nrow)
//
// Only real matrices are supported (single or double).  A symbolic L is
// converted into a numeric identity matrix before the row is added.
// The dtypes of all matrices must match, except when L is symbolic (in which
// case it is converted to the dtype of R).

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMODIFY

//------------------------------------------------------------------------------
// icomp: for sorting by qsort
//------------------------------------------------------------------------------

static int icomp (Int *i, Int *j)
{
    if (*i < *j)
    {
        return (-1) ;
    }
    else
    {
        return (1) ;
    }
}

//------------------------------------------------------------------------------
// t_cholmod_rowadd_worker
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_rowadd_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_rowadd_worker.c"

//------------------------------------------------------------------------------
// cholmod_rowadd
//------------------------------------------------------------------------------

// cholmod_rowadd adds a row to the LDL' factorization.  It computes the kth
// row and kth column of L, and then updates the submatrix L (k+1:n,k+1:n)
// accordingly.  The kth row and column of L should originally be equal to the
// kth row and column of the identity matrix (they are treated as such, if they
// are not).  The kth row/column of L is computed as the factorization of the
// kth row/column of the matrix to factorize, which is provided as a single
// n-by-1 sparse matrix R.  The sparse vector R need not be sorted.

int CHOLMOD(rowadd)
(
    // input:
    size_t k,           // row/column index to add
    cholmod_sparse *R,  // row/column of matrix to factorize (n-by-1)
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_common *Common
)
{
    double bk [2] ;
    bk [0] = 0. ;
    bk [1] = 0. ;
    return (CHOLMOD(rowadd_mark) (k, R, bk, NULL, L, NULL, NULL, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_rowadd_solve
//------------------------------------------------------------------------------

// Does the same as cholmod_rowadd, and also updates the solution to Lx=b
// See cholmod_updown for a description of how Lx=b is updated.  There is on
// additional parameter:  bk specifies the new kth entry of b.

int CHOLMOD(rowadd_solve)
(
    // input:
    size_t k,           // row/column index to add
    cholmod_sparse *R,  // row/column of matrix to factorize (n-by-1)
    double bk [2],      // kth entry of the right-hand-side b
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{
    return (CHOLMOD(rowadd_mark) (k, R, bk, NULL, L, X, DeltaB, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_rowadd_mark
//------------------------------------------------------------------------------

// Does the same as cholmod_rowadd_solve, except only part of L is used in
// the update/downdate of the solution to Lx=b.  This routine is an "expert"
// routine.  It is meant for use in LPDASA only.

int CHOLMOD(rowadd_mark)
(
    // input:
    size_t kadd,        // row/column index to add
    cholmod_sparse *R,  // row/column of matrix to factorize (n-by-1)
    double bk [2],      // kth entry of the right hand side, b
    Int *colmark,       // Int array of size 1.  See cholmod_updown.c
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_NULL (R, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_PATTERN, CHOLMOD_REAL, FALSE) ;
    RETURN_IF_XTYPE_INVALID (R, CHOLMOD_REAL, CHOLMOD_REAL, FALSE) ;
    Int n = L->n ;
    Int k = kadd ;
    if (kadd >= L->n || k < 0)
    {
        ERROR (CHOLMOD_INVALID, "k invalid") ;
        return (FALSE) ;
    }
    if (R->ncol != 1 || R->nrow != L->n)
    {
        ERROR (CHOLMOD_INVALID, "R invalid") ;
        return (FALSE) ;
    }
    if (L->xtype != CHOLMOD_PATTERN && L->dtype != R->dtype)
    {
        ERROR (CHOLMOD_INVALID, "R and L must have the same dtype") ;
        return (FALSE) ;
    }

    if ((X != NULL) && (DeltaB != NULL))
    {
        // also update the solution to Lx=b
        RETURN_IF_XTYPE_INVALID (X, CHOLMOD_REAL, CHOLMOD_REAL, FALSE) ;
        RETURN_IF_XTYPE_INVALID (DeltaB, CHOLMOD_REAL, CHOLMOD_REAL, FALSE) ;
        if (X->nrow != L->n || X->ncol != 1 ||
            DeltaB->nrow != L->n || DeltaB->ncol != 1 ||
            X->dtype != R->dtype || DeltaB->dtype != R->dtype)
        {
            ERROR (CHOLMOD_INVALID, "X and/or DeltaB invalid") ;
            return (FALSE) ;
        }
    }

    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // s = 2*n
    int ok = TRUE ;
    size_t s = CHOLMOD(mult_size_t) (L->n, 2, &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (FALSE) ;
    }

    CHOLMOD(alloc_work) (L->n, s, s, R->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, s, R->dtype, Common)) ;

    //--------------------------------------------------------------------------
    // convert to simplicial numeric LDL' factor, if not already
    //--------------------------------------------------------------------------

    if (L->xtype == CHOLMOD_PATTERN || L->is_super || L->is_ll)
    {
        // can only update/downdate a simplicial LDL' factorization
        if (L->xtype == CHOLMOD_PATTERN)
        {
            // L is symbolic; convert it to R->dtype
            L->dtype = R->dtype ;
        }
        CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE, FALSE, FALSE, L,
                Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory, L is returned unchanged
            return (FALSE) ;
        }
    }

    ASSERT (L->dtype == R->dtype) ;

    //--------------------------------------------------------------------------
    // update L and X
    //--------------------------------------------------------------------------

    float s_bk [2] ;
    s_bk [0] = (float) bk [0] ;
    s_bk [1] = (float) bk [1] ;

    switch (L->dtype & 4)
    {
        case CHOLMOD_SINGLE:
            ok = rs_cholmod_rowadd_worker (k, R, s_bk, colmark, L, X, DeltaB,
                Common) ;
            break ;

        case CHOLMOD_DOUBLE:
            ok = rd_cholmod_rowadd_worker (k, R, bk, colmark, L, X, DeltaB,
                Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    DEBUG (CHOLMOD(dump_factor) (L, "LDL factorization, L:", Common)) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, s, R->dtype, Common)) ;
    return (ok) ;
}

#endif
#endif

