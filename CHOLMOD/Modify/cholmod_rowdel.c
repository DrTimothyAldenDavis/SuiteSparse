//------------------------------------------------------------------------------
// CHOLMOD/Modify/cholmod_rowdel: delete row/column from an LDL' factorization
//------------------------------------------------------------------------------

// CHOLMOD/Modify Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// and William W. Hager.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Deletes a row and column from an LDL' factorization.  The row and column k
// is set to the kth row and column of the identity matrix.  Optionally
// downdates the solution to Lx=b.
//
// workspace: Flag (nrow), Head (nrow+1), W (nrow*2), Iwork (2*nrow)
//
// Only real matrices (single or double) are supported (exception: since only
// the pattern of R is used, it can have any valid xtype).

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMODIFY

//------------------------------------------------------------------------------
// t_cholmod_rowdel_worker
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_rowdel_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_rowdel_worker.c"

//------------------------------------------------------------------------------
// cholmod_rowdel
//------------------------------------------------------------------------------

// Sets the kth row and column of L to be the kth row and column of the identity
// matrix, and updates L(k+1:n,k+1:n) accordingly.   To reduce the running time,
// the caller can optionally provide the nonzero pattern (or an upper bound) of
// kth row of L, as the sparse n-by-1 vector R.  Provide R as NULL if you want
// CHOLMOD to determine this itself, which is easier for the caller, but takes
// a little more time.

int CHOLMOD(rowdel)
(
    // input:
    size_t k,           // row/column index to delete
    cholmod_sparse *R,  // NULL, or the nonzero pattern of kth row of L
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_common *Common
)
{
    double yk [2] ;
    yk [0] = 0. ;
    yk [1] = 0. ;
    return (CHOLMOD(rowdel_mark) (k, R, yk, NULL, L, NULL, NULL, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_rowdel_solve
//------------------------------------------------------------------------------

// Does the same as cholmod_rowdel, but also downdates the solution to Lx=b.
// When row/column k of A is "deleted" from the system A*y=b, this can induce
// a change to x, in addition to changes arising when L and b are modified.
// If this is the case, the kth entry of y is required as input (yk).

int CHOLMOD(rowdel_solve)
(
    // input:
    size_t k,           // row/column index to delete
    cholmod_sparse *R,  // NULL, or the nonzero pattern of kth row of L
    double yk [2],      // kth entry in the solution to A*y=b
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{
    return (CHOLMOD(rowdel_mark) (k, R, yk, NULL, L, X, DeltaB, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_rowdel_mark
//------------------------------------------------------------------------------

// Does the same as cholmod_rowdel_solve, except only part of L is used in
// the update/downdate of the solution to Lx=b.  This routine is an "expert"
// routine.  It is meant for use in LPDASA only.
//
// if R == NULL then columns 0:k-1 of L are searched for row k.  Otherwise, it
// searches columns in the set defined by the pattern of the first column of R.
// This is meant to be the pattern of row k of L (a superset of that pattern is
// OK too).  R must be a permutation of a subset of 0:k-1.

int CHOLMOD(rowdel_mark)
(
    // input:
    size_t kdel,        // row/column index to delete
    cholmod_sparse *R,  // NULL, or the nonzero pattern of kth row of L
    double yk [2],      // kth entry in the solution to A*y=b
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
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_PATTERN, CHOLMOD_REAL, FALSE) ;
    Int n = L->n ;
    Int k = kdel ;
    if (kdel >= L->n || k < 0)
    {
        ERROR (CHOLMOD_INVALID, "k invalid") ;
        return (FALSE) ;
    }
    if (R != NULL)
    {
        RETURN_IF_XTYPE_INVALID (R, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
        if (R->ncol != 1 || R->nrow != L->n)
        {
            ERROR (CHOLMOD_INVALID, "R invalid") ;
            return (FALSE) ;
        }
    }

    if ((X != NULL) && (DeltaB != NULL))
    {
        RETURN_IF_XTYPE_INVALID (X, CHOLMOD_REAL, CHOLMOD_REAL, FALSE) ;
        RETURN_IF_XTYPE_INVALID (DeltaB, CHOLMOD_REAL, CHOLMOD_REAL, FALSE) ;
        if (X->nrow != L->n || X->ncol != 1 ||
            DeltaB->nrow != L->n || DeltaB->ncol != 1 ||
            X->dtype != L->dtype || DeltaB->dtype != L->dtype)
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

    CHOLMOD(alloc_work) (L->n, s, s, L->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 2*n, L->dtype, Common)) ;

    //--------------------------------------------------------------------------
    // convert to simplicial numeric LDL' factor, if not already
    //--------------------------------------------------------------------------

    if (L->xtype == CHOLMOD_PATTERN || L->is_super || L->is_ll)
    {
        // can only update/downdate a simplicial LDL' factorization
        CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE, FALSE, FALSE, L,
                Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory, L is returned unchanged
            return (FALSE) ;
        }
    }

    //--------------------------------------------------------------------------
    // update L and X
    //--------------------------------------------------------------------------

    float s_yk [2] ;
    s_yk [0] = (float) yk [0] ;
    s_yk [1] = (float) yk [1] ;

    switch (L->dtype & 4)
    {
        case CHOLMOD_SINGLE:
            ok = rs_cholmod_rowdel_worker (k, R, s_yk, colmark, L, X, DeltaB,
                Common) ;
            break ;

        case CHOLMOD_DOUBLE:
            ok = rd_cholmod_rowdel_worker (k, R, yk, colmark, L, X, DeltaB,
                Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    DEBUG (CHOLMOD(dump_factor) (L, "LDL factorization, L:", Common)) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, s, L->dtype, Common)) ;
    return (ok) ;
}

#endif
#endif

