// =============================================================================
// === spqr_1fixed =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

//  Find column singletons, but do not permute the columns of A.  If there are
//  no column singletons, and no right-hand side B, then this takes only
//  O(m+n) time and memory.
//
//  Returns a sparse matrix Y with column pointers allocated and initialized,
//  but no values (nzmax(Y) is zero).  Y has n-n1cols+bncols columns, and
//  m-n1rows rows.  B is empty and no singletons are found, Y is NULL.

#include "spqr.hpp"

template <typename Entry, typename Int> int spqr_1fixed
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    Int bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    Int **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    Int **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    Int *p_n1cols,         // number of column singletons found
    Int *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
)
{
    cholmod_sparse *Y ;
    Int *P1inv, *R1p, *Yp, *Qrows, *Ap, *Ai ;
    char *Mark ;
    Entry *Ax ;
    Int i, j, k, p, d, row, n1rows, n1cols, ynz, iold, inew, kk, m, n, xtype ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    xtype = spqr_type <Entry> ( ) ;

    m = A->nrow ;
    n = A->ncol ;
    Ap = (Int *) A->p ;
    Ai = (Int *) A->i ;
    Ax = (Entry *) A->x ;

    // set outputs to NULL in case of early return
    *p_R1p    = NULL ;
    *p_P1inv  = NULL ;
    *p_Y      = NULL ;
    *p_n1cols = EMPTY ;
    *p_n1rows = EMPTY ;

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    Mark = (char *) spqr_calloc <Int> (m, sizeof (char), cc) ;
    Qrows = (Int *) spqr_malloc <Int> (n, sizeof (Int), cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_free <Int> (m, sizeof (char), Mark, cc) ;
        spqr_free <Int> (n, sizeof (Int), Qrows, cc) ;
        return (FALSE) ;
    }

    // -------------------------------------------------------------------------
    // find singletons; no column permutations allowed
    // -------------------------------------------------------------------------

    n1cols = 0 ;        // number of column singletons found
    n1rows = 0 ;        // number of corresponding singleton rows

    for (j = 0 ; j < n ; j++)
    {
        // count the number of unmarked rows in column j
        Entry aij = 0 ;
        d = 0 ;
        row = EMPTY ;
        for (p = Ap [j] ; d < 2 && p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            if (!Mark [i])
            {
                // row i is not taken by a prior column singleton.  If this
                // is the only unflagged row and the value is large enough,
                // it will become the row for this column singleton. 
                aij = Ax [p] ;
                row = i ;
                d++ ;
            }
        }
        if (d == 0)
        {
            // j is a dead column singleton
            Qrows [n1cols++] = EMPTY ;
        }
        else if (d == 1 && spqr_abs (aij) > tol)
        {
            // j is a live column singleton
            Qrows [n1cols++] = row ;
            // flag row i as taken
            Mark [row] = TRUE ;
            n1rows++ ;
        }
        else
        {
            // j is not a singleton; quit searching
            break ;
        }
    }

    // -------------------------------------------------------------------------
    // construct P1inv permutation, row counts R1p, and col pointers Yp
    // -------------------------------------------------------------------------

    if (n1cols == 0 && bncols == 0)
    {

        // ---------------------------------------------------------------------
        // no singletons, and B empty; Y=A will be done via pointer alias
        // ---------------------------------------------------------------------

        Y = NULL ;
        Yp = NULL ;
        P1inv = NULL ;
        R1p = NULL ;

    }
    else if (n1cols == 0)
    {

        // ---------------------------------------------------------------------
        // no singletons in the matrix; no R1 matrix, no P1inv permutation
        // ---------------------------------------------------------------------

        // Y has no entries yet; nnz(Y) will be determined later
        Y = spqr_allocate_sparse <Int> (m, n+bncols, 0,
            FALSE, TRUE, 0, xtype, cc) ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_free <Int> (m, sizeof (char), Mark, cc) ;
            spqr_free <Int> (n, sizeof (Int), Qrows, cc) ;
            return (FALSE) ;
        }

        Yp = (Int *) Y->p ;

        ASSERT (n1rows == 0) ;
        P1inv = NULL ;
        R1p = NULL ;

        // ---------------------------------------------------------------------
        // copy the column pointers of A for the first part of Y = [A B]
        // ---------------------------------------------------------------------

        ynz = Ap [n] ;
        for (k = 0 ; k <= n ; k++)
        {
            Yp [k] = Ap [k] ;
        }

    }
    else
    {

        // ---------------------------------------------------------------------
        // construct the row singleton permutation
        // ---------------------------------------------------------------------

        // Y has no entries yet; nnz(Y) will be determined later
        Y = spqr_allocate_sparse <Int> (m-n1rows, n-n1cols+bncols, 0,
            TRUE, TRUE, 0, xtype, cc) ;
        P1inv = (Int *) spqr_malloc <Int> (m, sizeof (Int), cc) ;
        R1p   = (Int *) spqr_calloc <Int> (n1rows+1, sizeof (Int), cc) ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_free_sparse <Int> (&Y, cc) ;
            spqr_free <Int> (m, sizeof (Int), P1inv, cc) ;
            spqr_free <Int> (n1rows+1, sizeof (Int), R1p, cc) ;
            spqr_free <Int> (m, sizeof (char), Mark, cc) ;
            spqr_free <Int> (n, sizeof (Int), Qrows, cc) ;
            return (FALSE) ;
        }

        Yp = (Int *) Y->p ;

#ifndef NDEBUG
        for (i = 0 ; i < m ; i++) P1inv [i] = EMPTY ;
#endif

        kk = 0 ;
        for (k = 0 ; k < n1cols ; k++)
        {
            i = Qrows [k] ;
            if (i != EMPTY)
            {
                // row i is the kk-th singleton row
                ASSERT (Mark [i]) ;
                ASSERT (P1inv [i] == EMPTY) ;
                P1inv [i] = kk ;
                kk++ ;
            }
        }
        for (i = 0 ; i < m ; i++)
        {
            if (!Mark [i])
            {
                // row i is not a singleton row
                ASSERT (P1inv [i] == EMPTY) ;
                P1inv [i] = kk ;
                kk++ ;
            }
        }
        ASSERT (kk == m) ;

        // ---------------------------------------------------------------------
        // find row counts for R11
        // ---------------------------------------------------------------------

        for (k = 0 ; k < n1cols ; k++)
        {
            for (p = Ap [k] ; p < Ap [k+1] ; p++)
            {
                iold = Ai [p] ;
                inew = P1inv [iold] ;
                ASSERT (inew < n1rows) ;
                R1p [inew]++ ;              // a singleton row; in R1
            }
        }

        // ---------------------------------------------------------------------
        // find row counts for R12 and column pointers for A2 part of Y
        // ---------------------------------------------------------------------

        ynz = 0 ;
        for ( ; k < n ; k++)
        {
            Yp [k-n1cols] = ynz ;
            for (p = Ap [k] ; p < Ap [k+1] ; p++)
            {
                iold = Ai [p] ;
                inew = P1inv [iold] ;
                if (inew < n1rows)
                {
                    R1p [inew]++ ;          // a singleton row; in R1
                }
                else
                {
                    ynz++ ;                 // not a singleton row; in A2
                }
            }
        }
        Yp [n-n1cols] = ynz ;

#ifndef NDEBUG
        PR (("n1cols: %ld\n", n1cols)) ;
        for (i = 0 ; i < n1rows ; i++)
        {
            PR (("R1p [%ld] is %ld\n", i, R1p [i])) ;
            ASSERT (R1p [i] > 0) ;
        }
#endif
    }

    // -------------------------------------------------------------------------
    // free workspace and return results
    // -------------------------------------------------------------------------

    spqr_free <Int> (n, sizeof (Int), Qrows, cc) ;
    spqr_free <Int> (m, sizeof (char), Mark, cc) ;

    *p_R1p    = R1p ;
    *p_P1inv  = P1inv ;
    *p_Y      = Y ;
    *p_n1cols = n1cols ;
    *p_n1rows = n1rows ;

    return (TRUE) ;
}


// =============================================================================

template int spqr_1fixed <Complex, int32_t>
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    int32_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    int32_t **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    int32_t **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    int32_t *p_n1cols,         // number of column singletons found
    int32_t *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;
template int spqr_1fixed <Complex, int64_t>
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    int64_t **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    int64_t **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    int64_t *p_n1cols,         // number of column singletons found
    int64_t *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template int spqr_1fixed <double, int32_t>
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    int32_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    int32_t **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    int32_t **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    int32_t *p_n1cols,         // number of column singletons found
    int32_t *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;
template int spqr_1fixed <double, int64_t>
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    int64_t **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    int64_t **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    int64_t *p_n1cols,         // number of column singletons found
    int64_t *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;
