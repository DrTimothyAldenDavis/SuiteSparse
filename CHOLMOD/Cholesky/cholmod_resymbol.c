//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/cholmod_resymbol: recompute symbolic pattern of L
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Recompute the symbolic pattern of L.  Entries not in the symbolic pattern
// are dropped.  L->Perm can be used (or not) to permute the input matrix A.
//
// These routines are used after a supernodal factorization is converted into
// a simplicial one, to remove zero entries that were added due to relaxed
// supernode amalgamation.  They can also be used after a series of downdates
// to remove entries that would no longer be present if the matrix were
// factorized from scratch.  A downdate (cholmod_updown) does not remove any
// entries from L.
//
// workspace: Flag (nrow), Head (nrow+1),
//      if symmetric:   Iwork (2*nrow)
//      if unsymmetric: Iwork (2*nrow+ncol).
//      Allocates up to 2 copies of its input matrix A (pattern only).

#include "cholmod_internal.h"

#ifndef NCHOLESKY

//------------------------------------------------------------------------------
// t_cholmod_resymbol_worker
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_resymbol_worker.c"
#define COMPLEX
#include "t_cholmod_resymbol_worker.c"
#define ZOMPLEX
#include "t_cholmod_resymbol_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_resymbol_worker.c"
#define COMPLEX
#include "t_cholmod_resymbol_worker.c"
#define ZOMPLEX
#include "t_cholmod_resymbol_worker.c"

//------------------------------------------------------------------------------
// cholmod_resymbol
//------------------------------------------------------------------------------

// Remove entries from L that are not in the factorization of P*A*P', P*A*A'*P',
// or P*F*F'*P' (depending on A->stype and whether fset is NULL or not).
//
// cholmod_resymbol is the same as cholmod_resymbol_noperm, except that it
// first permutes A according to L->Perm.  A can be upper/lower/unsymmetric,
// in contrast to cholmod_resymbol_noperm (which can be lower or unsym).

int CHOLMOD(resymbol)   // recompute symbolic pattern of L
(
    // input:
    cholmod_sparse *A,  // matrix to analyze
    Int *fset,          // subset of 0:(A->ncol)-1
    size_t fsize,       // size of fset
    int pack,           // if TRUE, pack the columns of L
    // input/output:
    cholmod_factor *L,  // factorization, entries pruned on output
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_sparse *H = NULL, *F = NULL, *G = NULL ;

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    Common->status = CHOLMOD_OK ;
    if (L->is_super)
    {
        // cannot operate on a supernodal factorization
        ERROR (CHOLMOD_INVALID, "cannot operate on supernodal L") ;
        return (FALSE) ;
    }
    if (L->n != A->nrow)
    {
        // dimensions must agree
        ERROR (CHOLMOD_INVALID, "A and L dimensions do not match") ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    Int stype = A->stype ;
    Int nrow = A->nrow ;
    Int ncol = A->ncol ;

    // s = 2*nrow + (stype ? 0 : ncol)
    int ok = TRUE ;
    size_t s = CHOLMOD(mult_size_t) (A->nrow, 2, &ok) ;
    s = CHOLMOD(add_size_t) (s, (stype ? 0 : A->ncol), &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (FALSE) ;
    }

    CHOLMOD(allocate_work) (A->nrow, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // permute the input matrix if necessary
    //--------------------------------------------------------------------------

    H = NULL ;
    G = NULL ;

    if (stype > 0)
    {
        if (L->ordering == CHOLMOD_NATURAL)
        {
            // F = triu(A)'
            // workspace: Iwork (nrow)
            G = CHOLMOD(ptranspose) (A, 0, NULL, NULL, 0, Common) ;
        }
        else
        {
            // F = triu(A(p,p))'
            // workspace: Iwork (2*nrow)
            G = CHOLMOD(ptranspose) (A, 0, L->Perm, NULL, 0, Common) ;
        }
        F = G ;
    }
    else if (stype < 0)
    {
        if (L->ordering == CHOLMOD_NATURAL)
        {
            F = A ;
        }
        else
        {
            // G = triu(A(p,p))'
            // workspace: Iwork (2*nrow)
            G = CHOLMOD(ptranspose) (A, 0, L->Perm, NULL, 0, Common) ;
            // H = G'
            // workspace: Iwork (nrow)
            H = CHOLMOD(ptranspose) (G, 0, NULL, NULL, 0, Common) ;
            F = H ;
        }
    }
    else
    {
        if (L->ordering == CHOLMOD_NATURAL)
        {
            F = A ;
        }
        else
        {
            // G = A(p,f)'
            // workspace: Iwork (nrow if no fset; MAX (nrow,ncol) if fset)
            G = CHOLMOD(ptranspose) (A, 0, L->Perm, fset, fsize, Common) ;
            // H = G'
            // workspace: Iwork (ncol)
            H = CHOLMOD(ptranspose) (G, 0, NULL, NULL, 0, Common) ;
            F = H ;
        }
    }

    // No need to check for failure here.  cholmod_resymbol_noperm will return
    // FALSE if F is NULL.

    //--------------------------------------------------------------------------
    // resymbol
    //--------------------------------------------------------------------------

    ok = CHOLMOD(resymbol_noperm) (F, fset, fsize, pack, L, Common) ;

    //--------------------------------------------------------------------------
    // free the temporary matrices, if they exist
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&H, Common) ;
    CHOLMOD(free_sparse) (&G, Common) ;
    return (ok) ;
}

//------------------------------------------------------------------------------
// cholmod_resymbol_noperm
//------------------------------------------------------------------------------

// Redo symbolic LDL' or LL' factorization of I + F*F' or I+A, where F=A(:,f).
//
// L already exists, but is a superset of the true dynamic pattern (simple
// column downdates and row deletions haven't pruned anything).  Just redo the
// symbolic factorization and drop entries that are no longer there.  The
// diagonal is not modified.  The number of nonzeros in column j of L
// (L->nz[j]) can decrease.  The column pointers (L->p[j]) remain unchanged if
// pack is FALSE or if L is not monotonic.  Otherwise, the columns of L are
// packed in place.
//
// For the symmetric case, the columns of the lower triangular part of A
// are accessed by column.  NOTE that this the transpose of the general case.
//
// For the unsymmetric case, F=A(:,f) is accessed by column.
//
// A need not be sorted, and can be packed or unpacked.  If L->Perm is not
// identity, then A must already be permuted according to the permutation used
// to factorize L.  The advantage of using this routine is that it does not
// need to create permuted copies of A first.
//
// This routine can be called if L is only partially factored via cholmod_rowfac
// since all it does is prune.  If an entry is in F*F' or A, but not in L, it
// isn't added to L.
//
// L must be simplicial LDL' or LL'; it cannot be supernodal or symbolic.
//
// The set f is held in fset and fsize.
//      fset = NULL means ":" in MATLAB. fset is ignored.
//      fset != NULL means f = fset [0..fset-1].
//      fset != NULL and fsize = 0 means f is the empty set.
//      There can be no duplicates in fset.
//      Common->status is set to CHOLMOD_INVALID if fset is invalid.
//
// workspace: Flag (nrow), Head (nrow+1),
//      if symmetric:   Iwork (2*nrow)
//      if unsymmetric: Iwork (2*nrow+ncol).
//      Unlike cholmod_resymbol, this routine does not allocate any temporary
//      copies of its input matrix.

int CHOLMOD(resymbol_noperm)    // recompute symbolic pattern of L
(
    // input:
    cholmod_sparse *A,  // matrix to analyze
    Int *fset,          // subset of 0:(A->ncol)-1
    size_t fsize,       // size of fset
    int pack,           // if TRUE, pack the columns of L
    // input/output:
    cholmod_factor *L,  // factorization, entries pruned on output
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    Int ncol = A->ncol ;
    Int nrow = A->nrow ;
    Int stype = A->stype ;
    ASSERT (IMPLIES (stype != 0, nrow == ncol)) ;
    if (stype > 0)
    {
        // symmetric, with upper triangular part, not supported
        ERROR (CHOLMOD_INVALID, "symmetric upper not supported ") ;
        return (FALSE) ;
    }
    if (L->is_super)
    {
        // cannot operate on a supernodal or symbolic factorization
        ERROR (CHOLMOD_INVALID, "cannot operate on supernodal L") ;
        return (FALSE) ;
    }
    if (L->n != A->nrow)
    {
        // dimensions must agree
        ERROR (CHOLMOD_INVALID, "A and L dimensions do not match") ;
        return (FALSE) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // s = nrow + (stype ? 0 : ncol)
    size_t s = A->nrow ;
    int ok = TRUE ;
    if (stype != 0)
    {
        s = CHOLMOD(add_size_t) (s, A->ncol, &ok) ;
    }
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (FALSE) ;
    }

    CHOLMOD(allocate_work) (A->nrow, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;        // out of memory
    }
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ai = A->i ;
    Int *Ap = A->p ;
    Int *Anz = A->nz ;
    bool apacked = A->packed ;
    bool sorted = A->sorted ;

    Int *Lp = L->p ;

    // If L is monotonic on input, then it can be packed or
    // unpacked on output, depending on the pack input parameter.

    // cannot pack a non-monotonic matrix
    if (!(L->is_monotonic))
    {
        pack = FALSE ;
    }

    ASSERT (L->nzmax >= (size_t) (Lp [L->n])) ;

    PRINT1 (("\n\n===================== Resymbol pack %d Apacked %d\n",
        pack, A->packed)) ;
    ASSERT (CHOLMOD(dump_sparse) (A, "ReSymbol A:", Common) >= 0) ;
    DEBUG (CHOLMOD(dump_factor) (L, "ReSymbol initial L (i, x):", Common)) ;

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    Int *Head  = Common->Head ;      // size nrow+1
    Int *Iwork = Common->Iwork ;
    Int *Link  = Iwork ;             // size nrow [
    Int *Anext = Iwork + nrow ;      // size ncol, unsym. only
    for (Int j = 0 ; j < nrow ; j++)
    {
        Link [j] = EMPTY ;
    }

    //--------------------------------------------------------------------------
    // for the unsymmetric case, queue each column of A (:,f)
    //--------------------------------------------------------------------------

    // place each column of the basis set on the link list corresponding to
    // the smallest row index in that column

    if (stype == 0)
    {
        Int nf ;
        bool use_fset = (fset != NULL) ;
        if (use_fset)
        {
            nf = fsize ;
            // This is the only O(ncol) loop in cholmod_resymbol.
            // It is required only to check the fset.
            for (Int j = 0 ; j < ncol ; j++)
            {
                Anext [j] = -2 ;
            }
            for (Int jj = 0 ; jj < nf ; jj++)
            {
                Int j = fset [jj] ;
                if (j < 0 || j > ncol || Anext [j] != -2)
                {
                    // out-of-range or duplicate entry in fset
                    ERROR (CHOLMOD_INVALID, "fset invalid") ;
                    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;
                    return (FALSE) ;
                }
                // flag column j as having been seen
                Anext [j] = EMPTY ;
            }
            // the fset is now valid
            ASSERT (CHOLMOD(dump_perm) (fset, nf, ncol, "fset", Common)) ;
        }
        else
        {
            nf = ncol ;
        }
        for (Int jj = 0 ; jj < nf ; jj++)
        {
            Int j = (use_fset) ? (fset [jj]) : jj ;
            // column j is the fset; find the smallest row (if any)
            Int p = Ap [j] ;
            Int pend = (apacked) ? (Ap [j+1]) : (p + Anz [j]) ;
            if (pend > p)
            {
                Int k = Ai [p] ;
                if (!sorted)
                {
                    for ( ; p < pend ; p++)
                    {
                        k = MIN (k, Ai [p]) ;
                    }
                }
                // place column j on link list k
                ASSERT (k >= 0 && k < nrow) ;
                Anext [j] = Head [k] ;
                Head [k] = j ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // recompute symbolic LDL' factorization
    //--------------------------------------------------------------------------

    switch ((L->xtype + L->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_resymbol_worker (A, pack, L, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_resymbol_worker (A, pack, L, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_resymbol_worker (A, pack, L, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_resymbol_worker (A, pack, L, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_resymbol_worker (A, pack, L, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_resymbol_worker (A, pack, L, Common) ;
            break ;
    }

    // done using Iwork for Link and Anext ]

    //--------------------------------------------------------------------------
    // convert L to packed, if requested
    //--------------------------------------------------------------------------

    if (pack)
    {
        // Shrink L to be just large enough.  It cannot fail.
        // workspace: none
        ASSERT ((size_t) (Lp [nrow]) <= L->nzmax) ;
        CHOLMOD(reallocate_factor) (Lp [nrow], L, Common) ;
        ASSERT (Common->status >= CHOLMOD_OK) ;
    }

    //--------------------------------------------------------------------------
    // clear workspace and return result
    //--------------------------------------------------------------------------

    CLEAR_FLAG (Common) ;
    ASSERT (check_flag (Common)) ;
    DEBUG (CHOLMOD(dump_factor) (L, "ReSymbol final L (i, x):", Common)) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;
    return (TRUE) ;
}
#endif

