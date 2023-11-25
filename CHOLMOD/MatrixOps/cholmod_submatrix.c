//------------------------------------------------------------------------------
// CHOLMOD/MatrixOps/cholmod_submatrix: extract submatrix from a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/MatrixOps Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// C = A (rset,cset), where C becomes length(rset)-by-length(cset) in
// dimension.  rset and cset can have duplicate entries.  A can be symmetric;
// C is returned as unsymmetric.   C is packed.  If the sorted flag is TRUE on
// input, or rset is sorted and A is sorted, then C is sorted; otherwise C is
// unsorted.
//
// A NULL rset or cset means "[ ]" in MATLAB notation.
// If the length of rset or cset is negative, it denotes ":" in MATLAB notation.
//
// For permuting a matrix, this routine is an alternative to cholmod_ptranspose
// (which permutes and transposes a matrix and can work on symmetric matrices).
//
// The time taken by this routine is O(A->nrow) if the Common workspace needs
// to be initialized, plus O(C->nrow + C->ncol + nnz (A (:,cset))).  Thus, if C
// is small and the workspace is not initialized, the time can be dominated by
// the call to cholmod_allocate_work.  However, once the workspace is
// allocated, subsequent calls take less time.
//
// workspace:  Iwork (max (A->nrow + length (rset), length (cset))).
//      allocates temporary copy of C if it is to be returned sorted.
//
// Matrices of any xtype and dtype are supported.

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMATRIXOPS

//------------------------------------------------------------------------------
// t_cholmod_submatrix_worker
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_submatrix_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_submatrix_worker.c"
#define COMPLEX
#include "t_cholmod_submatrix_worker.c"
#define ZOMPLEX
#include "t_cholmod_submatrix_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_submatrix_worker.c"
#define COMPLEX
#include "t_cholmod_submatrix_worker.c"
#define ZOMPLEX
#include "t_cholmod_submatrix_worker.c"

//------------------------------------------------------------------------------
// check_subset
//------------------------------------------------------------------------------

// Check the rset or cset, and return TRUE if valid, FALSE if invalid

static int check_subset (Int *set, Int len, Int n)
{
    Int k ;
    if (set == NULL)
    {
        return (TRUE) ;
    }
    for (k = 0 ; k < len ; k++)
    {
        if (set [k] < 0 || set [k] >= n)
        {
            return (FALSE) ;
        }
    }
    return (TRUE) ;
}

//------------------------------------------------------------------------------
// cholmod_submatrix
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(submatrix)  // return C = A (rset,cset)
(
    // input:
    cholmod_sparse *A,  // matrix to subreference
    Int *rset,          // set of row indices, duplicates OK
    int64_t rsize,      // size of rset, or -1 for ":"
    Int *cset,          // set of column indices, duplicates OK
    int64_t csize,      // size of cset, or -1 for ":"
    int mode,           // 2: numerical (conj) if A is symmetric,
                        // 1: numerical (non-conj.) if A is symmetric
                        // 0: pattern
    int sorted,         // if TRUE then return C with sorted columns
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_sparse *C = NULL, *A2 = NULL ;
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, NULL) ;

    mode = RANGE (mode, 0, 2) ;
    if (A->xtype == CHOLMOD_PATTERN)
    {
        mode = 0 ;
    }
    bool values = (mode != 0) ;

    if (rsize > Int_max || csize > Int_max)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (NULL) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // get problem size
    //--------------------------------------------------------------------------

    Int ancol = A->ncol ;
    Int anrow = A->nrow ;
    Int nr = (Int) rsize ;
    Int nc = (Int) csize ;

    if (rset == NULL)
    {
        // nr = 0 denotes rset = [ ], nr < 0 denotes rset = 0:anrow-1
        nr = (nr < 0) ? (-1) : 0 ;
    }
    if (cset == NULL)
    {
        // nr = 0 denotes cset = [ ], nr < 0 denotes cset = 0:ancol-1
        nc = (nc < 0) ? (-1) : 0 ;
    }
    Int cnrow = (nr < 0) ? anrow : nr ;  // negative rset means rset = 0:anrow-1
    Int cncol = (nc < 0) ? ancol : nc ;  // negative cset means cset = 0:ancol-1

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if (nr < 0 && nc < 0)
    {

        //----------------------------------------------------------------------
        // C = A (:,:), use cholmod_copy instead
        //----------------------------------------------------------------------

        // workspace: Iwork (max (C->nrow,C->ncol))
        PRINT1 (("submatrix C = A (:,:)\n")) ;
        C = CHOLMOD(copy) (A, 0, mode, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (NULL) ;
        }
        return (C) ;
    }

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    PRINT1 (("submatrix nr "ID" nc "ID" Cnrow "ID" Cncol "ID""
            "  Anrow "ID" Ancol "ID"\n", nr, nc, cnrow, cncol, anrow, ancol)) ;

    // s = MAX3 (anrow+MAX(0,nr), cncol, cnrow) ;
    int ok = TRUE ;
    size_t nr_size = (size_t) MAX (0, nr) ;
    size_t s = CHOLMOD(add_size_t) (A->nrow, MAX (0, nr_size), &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (NULL) ;
    }
    s = MAX3 (s, ((size_t) cncol), ((size_t) cnrow)) ;

    CHOLMOD(allocate_work) (anrow, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }

    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;

    //--------------------------------------------------------------------------
    // check rset and cset
    //--------------------------------------------------------------------------

    PRINT1 (("nr "ID" nc "ID"\n", nr, nc)) ;
    PRINT1 (("anrow "ID" ancol "ID"\n", anrow, ancol)) ;
    PRINT1 (("cnrow "ID" cncol "ID"\n", cnrow, cncol)) ;
    DEBUG (for (Int i = 0 ; i < nr ; i++)
        PRINT2 (("rset ["ID"] = "ID"\n", i, rset [i])));
    DEBUG (for (Int i = 0 ; i < nc ; i++)
        PRINT2 (("cset ["ID"] = "ID"\n", i, cset [i])));

    if (!check_subset (rset, nr, anrow))
    {
        ERROR (CHOLMOD_INVALID, "invalid rset") ;
        return (NULL) ;
    }

    if (!check_subset (cset, nc, ancol))
    {
        ERROR (CHOLMOD_INVALID, "invalid cset") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // convert A if necessary
    //--------------------------------------------------------------------------

    // convert A to unsymmetric, if necessary
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

    ASSERT (A->stype == 0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ap  = A->p ;
    Int *Anz = A->nz ;
    Int *Ai  = A->i ;
    bool packed = A->packed ;

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    Int *Head  = Common->Head ;          // size anrow
    Int *Iwork = Common->Iwork ;
    Int *Rlen  = Iwork ;                 // size anrow
    Int *Rnext = Iwork + anrow ;         // size nr, not used if nr < 0

    //--------------------------------------------------------------------------
    // construct inverse of rset and compute nnz (C)
    //--------------------------------------------------------------------------

    // C is sorted if A and rset are sorted, or if C has one row or less
    bool csorted = A->sorted || (cnrow <= 1) ;

    Int nnz = 0 ;
    if (nr < 0)
    {
        // C = A (:,cset) where cset = [ ] or cset is not empty
        ASSERT (IMPLIES (cncol > 0, cset != NULL)) ;
        for (Int cj = 0 ; cj < cncol ; cj++)
        {
            // construct column cj of C, which is column j of A
            Int j = cset [cj] ;
            nnz += (packed) ? (Ap [j+1] - Ap [j]) : MAX (0, Anz [j]) ;
        }
    }
    else
    {
        // C = A (rset,cset), where rset is not empty but cset might be empty
        // create link lists in reverse order to preserve natural order
        Int ilast = anrow ;
        for (Int ci = nr-1 ; ci >= 0 ; ci--)
        {
            // row i of A becomes row ci of C; add ci to ith link list
            Int i = rset [ci] ;
            Int head = Head [i] ;
            Rlen [i] = (head == EMPTY) ? 1 : (Rlen [i] + 1) ;
            Rnext [ci] = head ;
            Head [i] = ci ;
            if (i > ilast)
            {
                // row indices in columns of C will not be sorted
                csorted = FALSE ;
            }
            ilast = i ;
        }

        #ifndef NDEBUG
        for (Int i = 0 ; i < anrow ; i++)
        {
            Int k = 0 ;
            Int rlen = (Head [i] != EMPTY) ? Rlen [i] : -1 ;
            PRINT1 (("Row "ID" Rlen "ID": ", i, rlen)) ;
            for (Int ci = Head [i] ; ci != EMPTY ; ci = Rnext [ci])
            {
                k++ ;
                PRINT2 ((""ID" ", ci)) ;
            }
            PRINT1 (("\n")) ;
            ASSERT (IMPLIES (Head [i] != EMPTY, k == Rlen [i])) ;
        }
        #endif

        // count entries in C
        for (Int cj = 0 ; cj < cncol ; cj++)
        {
            // count rows in column cj of C, which is column j of A
            Int j = (nc < 0) ? cj : (cset [cj]) ;
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                // row i of A becomes multiple rows (ci) of C
                Int i = Ai [p] ;
                ASSERT (i >= 0 && i < anrow) ;
                if (Head [i] != EMPTY)
                {
                    nnz += Rlen [i] ;
                }
            }
        }
    }
    PRINT1 (("nnz (C) "ID"\n", nnz)) ;

    // rset and cset are now valid
    DEBUG (CHOLMOD(dump_subset) (rset, rsize, anrow, "rset", Common)) ;
    DEBUG (CHOLMOD(dump_subset) (cset, csize, ancol, "cset", Common)) ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    C = CHOLMOD(allocate_sparse) (cnrow, cncol, nnz, csorted, TRUE, 0,
            (values ? A->xtype : CHOLMOD_PATTERN) + A->dtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        for (Int i = 0 ; i < anrow ; i++)
        {
            Head [i] = EMPTY ;
        }
        ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;
        CHOLMOD(free_sparse) (&A2, Common) ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // C = A (rset,cset)
    //--------------------------------------------------------------------------

    if (nnz == 0)
    {
        // C has no entries
        Int *Cp = C->p ;
        for (Int cj = 0 ; cj <= cncol ; cj++)
        {
            Cp [cj] = 0 ;
        }
    }
    else
    {
        switch ((C->xtype + C->dtype) % 8)
        {
            default:
                p_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                rs_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                cs_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                zs_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                rd_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                cd_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                zd_cholmod_submatrix_worker (C, A, nr, nc, cset, Head, Rnext) ;
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // clear workspace
    //--------------------------------------------------------------------------

    for (Int ci = 0 ; ci < nr ; ci++)
    {
        Head [rset [ci]] = EMPTY ;
    }

    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;

    //--------------------------------------------------------------------------
    // sort C, if requested
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (C , "C before sort", Common) >= 0) ;
    if (sorted && !csorted)
    {
        CHOLMOD(sort) (C, Common) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&A2, Common) ;
    ASSERT (CHOLMOD(dump_sparse) (C , "Final C", Common) >= 0) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;
    return (C) ;
}

#endif
#endif

