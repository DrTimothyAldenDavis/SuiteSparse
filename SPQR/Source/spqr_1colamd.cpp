// =============================================================================
// === spqr_1colamd ============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

//  Find column singletons, with column permutations allowed.  After column
//  singletons are found (and ordered first in Q1fill), the remaining columns
//  are optionally permuted via COLAMD or CHOLMOD's internal ordering method(s).
//  This function handles the natural, COLAMD, and CHOLMOD ordering options
//  (not the fixed ordering option).
//
//  Returns a sparse matrix Y with column pointers allocated and initialized,
//  but no values.  Y has n-n1cols+bncols columns, and m-n1rows rows.  B is
//  empty and no singletons are found, Y is NULL.

#include "spqr.hpp"

template <typename Entry, typename Int> int spqr_1colamd  // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    Int bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // outputs, neither allocated nor defined on input

    Int **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

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
    Int *Q1fill, *Degree, *Qrows, *W, *Winv, *ATp, *ATj, *R1p, *P1inv, *Yp,
        *Ap, *Ai, *Work ;
    Entry *Ax ;
    Int p, d, j, i, k, n1cols, n1rows, row, pend, n2rows, n2cols = EMPTY,
        nz2, kk, p2, col2, ynz, fill_reducing_ordering, m, n, xtype, worksize ;
    cholmod_sparse *AT, *Y ;

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
    *p_Q1fill = NULL ;
    *p_R1p = NULL ;
    *p_P1inv = NULL ;
    *p_Y = NULL ;
    *p_n1cols = EMPTY ;
    *p_n1rows = EMPTY ;

    // -------------------------------------------------------------------------
    // allocate result Q1fill (Y, R1p, P1inv allocated later)
    // -------------------------------------------------------------------------

    Q1fill = (Int *) spqr_malloc <Int> (n+bncols, sizeof (Int), cc) ;

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    fill_reducing_ordering = !
        ((ordering == SPQR_ORDERING_FIXED) ||
         (ordering == SPQR_ORDERING_GIVEN) ||
         (ordering == SPQR_ORDERING_NATURAL)) ;

    worksize = ((fill_reducing_ordering) ? 3:2) * n ;

    Work = (Int *) spqr_malloc <Int> (worksize, sizeof (Int), cc) ;
    Degree = Work ;         // size n
    Qrows  = Work + n ;     // size n
    Winv   = Qrows ;        // Winv and Qrows not needed at the same time
    W      = Qrows + n ;    // size n if fill-reducing ordering, else size 0

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory; free everything and return
        spqr_free <Int> (worksize, sizeof (Int), Work, cc) ;
        spqr_free <Int> (n+bncols, sizeof (Int), Q1fill, cc) ;
        return (FALSE) ;
    }

    // -------------------------------------------------------------------------
    // initialze queue with empty columns, and columns with just one entry
    // -------------------------------------------------------------------------

    n1cols = 0 ;
    n1rows = 0 ;

    for (j = 0 ; j < n ; j++)
    {
        p = Ap [j] ;
        d = Ap [j+1] - p ;
        if (d == 0)
        {
            // j is a dead column singleton
            PR (("initial dead %ld\n", j)) ;
            Q1fill [n1cols] = j ;
            Qrows [n1cols] = EMPTY ;
            n1cols++ ;
            Degree [j] = EMPTY ;
        }
        else if (d == 1 && spqr_abs (Ax [p]) > tol)
        {
            // j is a column singleton, live or dead
            PR (("initial live %ld %ld\n", j, Ai [p])) ;
            Q1fill [n1cols] = j ;
            Qrows [n1cols] = Ai [p] ;       // this might be a duplicate
            n1cols++ ;
            Degree [j] = EMPTY ;
        }
        else
        {
            // j has degree > 1, it is not (yet) a singleton
            Degree [j] = d ;
        }
    }

    // Degree [j] = EMPTY if j is in the singleton queue, or the Degree [j] > 1
    // is the degree of column j otherwise

    // -------------------------------------------------------------------------
    // create AT = spones (A')
    // -------------------------------------------------------------------------

    AT = spqr_transpose <Int> (A, 0, cc) ;       // [

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory; free everything and return
        spqr_free <Int> (worksize, sizeof (Int), Work, cc) ;
        spqr_free <Int> (n+bncols, sizeof (Int), Q1fill, cc) ;
        return (FALSE) ;
    }

    ATp = (Int *) AT->p ;
    ATj = (Int *) AT->i ;

    // -------------------------------------------------------------------------
    // remove column singletons via breadth-first-search
    // -------------------------------------------------------------------------

    for (k = 0 ; k < n1cols ; k++)
    {

        // ---------------------------------------------------------------------
        // get a new singleton from the queue
        // ---------------------------------------------------------------------

        // Int col = Q1fill [k] ;   unused variable, for debugging
        #define col (Q1fill [k])

        row = Qrows [k] ;
        PR (("\n---- singleton col %ld row %ld\n", col, row)) ;
        ASSERT (Degree [col] == EMPTY) ;

        if (row == EMPTY || ATp [row] < 0)
        {

            // -----------------------------------------------------------------
            // col is a dead column singleton; remove duplicate row index
            // -----------------------------------------------------------------

            Qrows [k] = EMPTY ;
            row = EMPTY ;
            PR (("dead: %ld\n", col)) ;

        }
        else
        {

            // -----------------------------------------------------------------
            // col is a live col singleton; remove its row from matrix
            // -----------------------------------------------------------------

            n1rows++ ;
            p = ATp [row] ;
            ATp [row] = FLIP (p) ;          // flag the singleton row
            pend = UNFLIP (ATp [row+1]) ;
            PR (("live: %ld row %ld\n", col, row)) ;
            for ( ; p < pend ; p++)
            {
                // look for new column singletons after row is removed
                j = ATj [p] ;
                d = Degree [j] ;
                if (d == EMPTY)
                {
                    // j is already in the singleton queue
                    continue ;
                }
                ASSERT (d >= 1) ;
                ASSERT2 (spqrDebug_listcount (j, Q1fill, n1cols, 0, cc) == 0) ;
                d-- ;
                Degree [j] = d ;
                if (d == 0)
                {
                    // a new dead col singleton
                    PR (("newly dead %ld\n", j)) ;
                    Q1fill [n1cols] = j ;
                    Qrows [n1cols] = EMPTY ;
                    n1cols++ ;
                    Degree [j] = EMPTY ;
                }
                else if (d == 1)
                {
                    // a new live col singleton; find its single live row
                    for (p2 = Ap [j] ; p2 < Ap [j+1] ; p2++)
                    {
                        i = Ai [p2] ;
                        if (ATp [i] >= 0 && spqr_abs (Ax [p2]) > tol)
                        {
                            // i might appear in Qrows [k+1:n1cols-1]
                            PR (("newly live %ld\n", j)) ;
                            ASSERT2 (spqrDebug_listcount (i,Qrows,k+1,1,cc)==0);
                            Q1fill [n1cols] = j ;
                            Qrows [n1cols] = i ;
                            n1cols++ ;
                            Degree [j] = EMPTY ;
                            break ;
                        }
                    }
                }
            }
        }
        // Q1fill [0:k] and Qrows [0:k] have no duplicates
        ASSERT2 (spqrDebug_listcount (col, Q1fill, n1cols, 0, cc) == 1) ; 
        ASSERT2 (IMPLIES (row >= 0, spqrDebug_listcount 
            (row, Qrows, k+1, 1, cc) == 1)) ; 

        // used for debugging only
        #undef col
    }

    // -------------------------------------------------------------------------
    // Degree flags the column singletons, ATp flags their rows
    // -------------------------------------------------------------------------

#ifndef NDEBUG
    k = 0 ;
    for (j = 0 ; j < n ; j++)
    {
        PR (("j %ld Degree[j] %ld\n", j, Degree [j])) ;
        if (Degree [j] > 0) k++ ;        // j is not a column singleton
    }
    PR (("k %ld n %ld n1cols %ld\n", k, n, n1cols)) ;
    ASSERT (k == n - n1cols) ;
    for (k = 0 ; k < n1cols ; k++)
    {
        Int col = Q1fill [k] ;
        ASSERT (Degree [col] <= 0) ;
    }
    k = 0 ;
    for (i = 0 ; i < m ; i++)
    {
        if (ATp [i] >= 0) k++ ;     // i is not a row of a col singleton
    }
    ASSERT (k == m - n1rows) ;
    for (k = 0 ; k < n1cols ; k++)
    {
        row = Qrows [k] ;
        ASSERT (IMPLIES (row != EMPTY, ATp [row] < 0)) ;
    }
#endif

    // -------------------------------------------------------------------------
    // find the row ordering
    // -------------------------------------------------------------------------

    if (n1cols == 0)
    {

        // ---------------------------------------------------------------------
        // no singletons in the matrix; no R1 matrix, no P1inv permutation
        // ---------------------------------------------------------------------

        ASSERT (n1rows == 0) ;
        R1p = NULL ;
        P1inv = NULL ;

    }
    else
    {

        // ---------------------------------------------------------------------
        // construct the row singleton permutation
        // ---------------------------------------------------------------------

        // allocate result arrays R1p and P1inv
        R1p   = (Int *) spqr_malloc <Int> (n1rows+1, sizeof (Int), cc) ;
        P1inv = (Int *) spqr_malloc <Int> (m,        sizeof (Int), cc) ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory; free everything and return
            spqr_free_sparse <Int> (&AT, cc) ;
            spqr_free <Int> (worksize, sizeof (Int), Work, cc) ;
            spqr_free <Int> (n+bncols, sizeof (Int), Q1fill, cc) ;
            spqr_free <Int> (n1rows+1, sizeof (Int), R1p, cc) ;
            spqr_free <Int> (m,        sizeof (Int), P1inv, cc) ;
            return (FALSE) ;
        }

#ifndef NDEBUG
        for (i = 0 ; i < m ; i++) P1inv [i] = EMPTY ;
#endif

        kk = 0 ;
        for (k = 0 ; k < n1cols ; k++)
        {
            i = Qrows [k] ;
            PR (("singleton col %ld row %ld\n", Q1fill [k], i)) ;
            if (i != EMPTY)
            {
                // row i is the kk-th singleton row
                ASSERT (ATp [i] < 0) ;
                ASSERT (P1inv [i] == EMPTY) ;
                P1inv [i] = kk ;
                // also find # of entries in row kk of R1
                R1p [kk] = UNFLIP (ATp [i+1]) - UNFLIP (ATp [i]) ;
                kk++ ;
            }
        }
        ASSERT (kk == n1rows) ;
        for (i = 0 ; i < m ; i++)
        {
            if (ATp [i] >= 0)
            {
                // row i is not a singleton row
                ASSERT (P1inv [i] == EMPTY) ;
                P1inv [i] = kk ;
                kk++ ;
            }
        }
        ASSERT (kk == m) ;

    }

    // Qrows is no longer needed.

    // -------------------------------------------------------------------------
    // complete the column ordering
    // -------------------------------------------------------------------------

    if (!fill_reducing_ordering)
    {

        // ---------------------------------------------------------------------
        // natural ordering
        // ---------------------------------------------------------------------

        if (n1cols == 0)
        {

            // no singletons, so natural ordering is 0:n-1 for now
            for (k = 0 ; k < n ; k++)
            {
                Q1fill [k] = k ;
            }

        }
        else
        {

            // singleton columns appear first, then non column singletons
            k = n1cols ;
            for (j = 0 ; j < n ; j++)
            {
                if (Degree [j] > 0)
                {
                    // column j is not a column singleton
                    Q1fill [k++] = j ;
                }
            }
            ASSERT (k == n) ;
        }

    }
    else
    {

        // ---------------------------------------------------------------------
        // fill-reducing ordering of pruned submatrix
        // ---------------------------------------------------------------------

        if (n1cols == 0)
        {

            // -----------------------------------------------------------------
            // no singletons found; do fill-reducing on entire matrix
            // -----------------------------------------------------------------

            n2cols = n ;
            n2rows = m ;

        }
        else
        {

            // -----------------------------------------------------------------
            // create the pruned matrix for fill-reducing by removing singletons
            // -----------------------------------------------------------------

            // find the mapping of original columns to pruned columns
            n2cols = 0 ;
            for (j = 0 ; j < n ; j++)
            {
                if (Degree [j] > 0)
                {
                    // column j is not a column singleton
                    W [j] = n2cols++ ;
                    PR (("W [%ld] = %ld\n", j, W [j])) ;
                }
                else
                {
                    // column j is a column singleton
                    W [j] = EMPTY ;
                    PR (("W [%ld] = %ld (j is col singleton)\n", j, W [j])) ;
                }
            }
            ASSERT (n2cols == n - n1cols) ;

            // W is now a mapping of the original columns to the columns in the
            // pruned matrix.  W [col] == EMPTY if col is a column singleton.
            // Otherwise col2 = W [j] is a column of the pruned matrix.

            // -----------------------------------------------------------------
            // delete row and column singletons from A'
            // -----------------------------------------------------------------

            // compact A' by removing row and column singletons
            nz2 = 0 ;
            n2rows = 0 ;
            for (i = 0 ; i < m ; i++)
            {
                p = ATp [i] ;
                if (p >= 0)
                {
                    // row i is not a row of a column singleton
                    ATp [n2rows++] = nz2 ;
                    pend = UNFLIP (ATp [i+1]) ;
                    for (p = ATp [i] ; p < pend ; p++)
                    {
                        j = ATj [p] ;
                        ASSERT (W [j] >= 0 && W [j] < n-n1cols) ;
                        ATj [nz2++] = W [j] ;
                    }
                }
            }
            ATp [n2rows] = nz2 ;
            ASSERT (n2rows == m - n1rows) ;
        }

        // ---------------------------------------------------------------------
        // fill-reducing ordering of the transpose of the pruned A' matrix
        // ---------------------------------------------------------------------

        PR (("n1cols %ld n1rows %ld n2cols %ld n2rows %ld\n",
            n1cols, n1rows, n2cols, n2rows)) ;
        ASSERT ((Int) AT->nrow == n) ;
        ASSERT ((Int) AT->ncol == m) ;

        AT->nrow = n2cols ;
        AT->ncol = n2rows ;

        // save the current CHOLMOD settings
        Int save [6] ;
        save [0] = cc->supernodal ;
        save [1] = cc->nmethods ;
        save [2] = cc->postorder ;
        save [3] = cc->method [0].ordering ;
        save [4] = cc->method [1].ordering ;
        save [5] = cc->method [2].ordering ;

        // follow the ordering with a postordering of the column etree
        cc->postorder = TRUE ;

        // 8:best: best of COLAMD(A), AMD(A'A), and METIS (if available)
        if (ordering == SPQR_ORDERING_BEST)
        {
            ordering = SPQR_ORDERING_CHOLMOD ;
            cc->nmethods = 3 ;
            cc->method [0].ordering = CHOLMOD_COLAMD ;
            cc->method [1].ordering = CHOLMOD_AMD ;
            cc->method [2].ordering = CHOLMOD_METIS ;
        }

        // 9:bestamd: best of COLAMD(A) and AMD(A'A)
        if (ordering == SPQR_ORDERING_BESTAMD)
        {
            // if METIS is not installed, this option is the same as 8:best
            ordering = SPQR_ORDERING_CHOLMOD ;
            cc->nmethods = 2 ;
            cc->method [0].ordering = CHOLMOD_COLAMD ;
            cc->method [1].ordering = CHOLMOD_AMD ;
        }

        if (ordering == SPQR_ORDERING_DEFAULT)
        {
            // Version 1.2.0:  just use COLAMD
            ordering = SPQR_ORDERING_COLAMD ;
        }

        if (ordering == SPQR_ORDERING_AMD)
        {
            // use CHOLMOD's interface to AMD to order A'*A
            spqr_amd <Int> (AT, NULL, 0, (Int *) (Q1fill + n1cols), cc) ;
        }
        else if (ordering == SPQR_ORDERING_METIS)
        {
            // use CHOLMOD's interface to METIS to order A'*A (if installed)
            TEST_COVERAGE_PAUSE ;
            #ifndef NPARTITION
            spqr_metis <Int> (AT, NULL, 0, TRUE,
                (Int *) (Q1fill + n1cols), cc) ;
            #else
            cc->status = CHOLMOD_NOT_INSTALLED ;
            #endif
            TEST_COVERAGE_RESUME ;
        }
        else if (ordering == SPQR_ORDERING_CHOLMOD)
        {
            // use CHOLMOD's internal ordering (defined by cc) to order AT
            PR (("Using CHOLMOD, nmethods %d\n", cc->nmethods)) ;
            cc->supernodal = CHOLMOD_SIMPLICIAL ;
            cc->postorder = TRUE ;
            cholmod_factor *Sc ;
            TEST_COVERAGE_PAUSE ;
            Sc = spqr_analyze_p2 <Int> (FALSE, AT, NULL, NULL, 0, cc) ;
            TEST_COVERAGE_RESUME ;
            if (Sc != NULL)
            {
                // copy perm from Sc->Perm [0:n2cols-1] to Q1fill (n1cols:n)
                Int *Sc_perm = (Int *) Sc->Perm ;
                for (k = 0 ; k < n2cols ; k++)
                {
                    Q1fill [k + n1cols] = Sc_perm [k] ;
                }
                // CHOLMOD selected an ordering; determine the ordering used
                switch (Sc->ordering)
                {
                    case CHOLMOD_AMD:    ordering = SPQR_ORDERING_AMD    ;break;
                    case CHOLMOD_COLAMD: ordering = SPQR_ORDERING_COLAMD ;break;
                    case CHOLMOD_METIS:  ordering = SPQR_ORDERING_METIS  ;break;
                }
            }
            spqr_free_factor <Int> (&Sc, cc) ;
            PR (("CHOLMOD used method %d : ordering: %d\n", cc->selected,
                cc->method [cc->selected].ordering)) ;
        }
        else // SPQR_ORDERING_DEFAULT or SPQR_ORDERING_COLAMD
        {
            // use CHOLMOD's interface to COLAMD to order AT
            ordering = SPQR_ORDERING_COLAMD ;
            spqr_colamd <Int> (AT, NULL, 0, TRUE,
                (Int *) (Q1fill + n1cols), cc) ;
        }

        cc->SPQR_istat [7] = ordering ;

        // restore the CHOLMOD settings
        cc->supernodal              = save [0] ;
        cc->nmethods                = save [1] ;
        cc->postorder               = save [2] ;
        cc->method [0].ordering     = save [3] ;
        cc->method [1].ordering     = save [4] ;
        cc->method [2].ordering     = save [5] ;

        AT->nrow = n ;
        AT->ncol = m ;
    }

    // -------------------------------------------------------------------------
    // free AT
    // -------------------------------------------------------------------------

    spqr_free_sparse <Int> (&AT, cc) ;   // ]

    // -------------------------------------------------------------------------
    // check if the method succeeded
    // -------------------------------------------------------------------------

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory; free everything and return
        spqr_free <Int> (worksize, sizeof (Int), Work, cc) ;
        spqr_free <Int> (n+bncols, sizeof (Int), Q1fill, cc) ;
        spqr_free <Int> (n1rows+1, sizeof (Int), R1p, cc) ;
        spqr_free <Int> (m,        sizeof (Int), P1inv, cc) ;
        return (FALSE) ;
    }

    // -------------------------------------------------------------------------
    // map the fill-reducing ordering ordering back to A
    // -------------------------------------------------------------------------

    if (n1cols > 0 && fill_reducing_ordering)
    {
        // Winv is workspace of size n2cols <= n

        #ifndef NDEBUG
        for (j = 0 ; j < n2cols ; j++) Winv [j] = EMPTY ;
        #endif

        for (j = 0 ; j < n ; j++)
        {
            // j is a column of A.  col2 = W [j] is either EMPTY, or it is 
            // the corresponding column of the pruned matrix
            col2 = W [j] ;
            if (col2 != EMPTY)
            {
                ASSERT (col2 >= 0 && col2 < n2cols) ;
                Winv [col2] = j ;
            }
        }

        for (k = n1cols ; k < n ; k++)
        {
            // col2 is a column of the pruned matrix
            col2 = Q1fill [k] ;
            // j is the corresonding column of the A matrix
            j = Winv [col2] ;
            ASSERT (j >= 0 && j < n) ;
            Q1fill [k] = j ;
        }
    }

    // -------------------------------------------------------------------------
    // identity permutation of the columns of B
    // -------------------------------------------------------------------------

    for (k = n ; k < n+bncols ; k++)
    {
        // tack on the identity permutation for columns of B
        Q1fill [k] = k ;
    }

    // -------------------------------------------------------------------------
    // find column pointers for Y = [A2 B2]; columns of A2
    // -------------------------------------------------------------------------

    if (n1cols == 0 && bncols == 0)
    {
        // A will be factorized instead of Y
        Y = NULL ;
    }
    else
    {
        // Y has no entries yet; nnz(Y) will be determined later
        Y = spqr_allocate_sparse <Int> (m-n1rows, n-n1cols+bncols, 0,
            FALSE, TRUE, 0, xtype, cc) ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory; free everything and return
            spqr_free <Int> (worksize, sizeof (Int), Work, cc) ;
            spqr_free <Int> (n+bncols, sizeof (Int), Q1fill, cc) ;
            spqr_free <Int> (n1rows+1, sizeof (Int), R1p, cc) ;
            spqr_free <Int> (m,        sizeof (Int), P1inv, cc) ;
            return (FALSE) ;
        }

        Yp = (Int *) Y->p ; 

        ynz = 0 ;
        PR (("1c wrapup: n1cols %ld n %ld\n", n1cols, n)) ;
        for (k = n1cols ; k < n ; k++)
        {
            j = Q1fill [k] ;
            d = Degree [j] ;
            ASSERT (d >= 1 && d <= m) ;
            Yp [k-n1cols] = ynz ;
            ynz += d ;
        }
        Yp [n-n1cols] = ynz ;
    }

    // -------------------------------------------------------------------------
    // free workspace and return results
    // -------------------------------------------------------------------------

    spqr_free <Int> (worksize, sizeof (Int), Work, cc) ;

    *p_Q1fill = Q1fill ;
    *p_R1p    = R1p ;
    *p_P1inv  = P1inv ;
    *p_Y      = Y ;
    *p_n1cols = n1cols ;
    *p_n1rows = n1rows ;
    return (TRUE) ;
}

template int spqr_1colamd <double, int32_t>  // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    int32_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // outputs, neither allocated nor defined on input

    int32_t **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

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

template int spqr_1colamd <Complex, int32_t>  // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    int32_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // outputs, neither allocated nor defined on input

    int32_t **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

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

template  int spqr_1colamd <double, int64_t> // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // outputs, neither allocated nor defined on input

    int64_t **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

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

template int spqr_1colamd <Complex, int64_t> // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // outputs, neither allocated nor defined on input

    int64_t **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

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
