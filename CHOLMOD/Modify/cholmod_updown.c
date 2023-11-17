//------------------------------------------------------------------------------
// CHOLMOD/Modify/cholmod_updown: sparse Cholesky update/downdate
//------------------------------------------------------------------------------

// CHOLMOD/Modify Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// and William W. Hager. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Updates/downdates the LDL' factorization (symbolic, then numeric), by
// computing a new factorization of
//
//      Lnew * Dnew * Lnew' = Lold * Dold * Lold' +/- C*C'
//
// C must be sorted.  It can be either packed or unpacked.  As in all CHOLMOD
// routines, the columns of L are sorted on input, and also on output.
//
// If the factor is not an unpacked LDL' or dynamic LDL', it is converted
// to an LDL' dynamic factor.  An unpacked LDL' factor may be updated, but if
// any one column runs out of space, the factor is converted to an LDL'
// dynamic one.  If the initial conversion fails, the factor is returned
// unchanged.
//
// If memory runs out during the update, the factor is returned as a simplicial
// symbolic factor.  That is, everything is freed except for the fill-reducing
// ordering and its corresponding column counts (typically computed by
// cholmod_analyze).
//
// Note that the fill-reducing permutation L->Perm is NOT used.  The row
// indices of C refer to the rows of L, not A.  If your original system is
// LDL' = PAP' (where P = L->Perm), and you want to compute the LDL'
// factorization of A+CC', then you must permute C first.  That is:
//
//      PAP' = LDL'
//      P(A+CC')P' = PAP'+PCC'P' = LDL' + (PC)(PC)' = LDL' + Cnew*Cnew'
//      where Cnew = P*C.
//
// You can use the cholmod_submatrix routine in the MatrixOps module
// to permute C, with:
//
// Cnew = cholmod_submatrix (C, L->Perm, L->n, NULL, -1, TRUE, TRUE, Common) ;
//
// Note that the sorted input parameter to cholmod_submatrix must be TRUE,
// because cholmod_updown requires C with sorted columns.
//
// The system Lx=b can also be updated/downdated.  The old system was Lold*x=b.
// The new system is Lnew*xnew = b + deltab.  The old solution x is overwritten
// with xnew.  Note that as in the update/downdate of L itself, the fill-
// reducing permutation L->Perm is not used.  x and b are in the permuted
// ordering, not your original ordering.  x and b are n-by-1; this routine
// does not handle multiple right-hand-sides.
//
// workspace: Flag (nrow), Head (nrow+1), W (maxrank*nrow), Iwork (nrow),
// where maxrank is 2, 4, or 8.
//
// Only real matrices are supported (single and double).  A symbolic L is
// converted into a numeric identity matrix.

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NMODIFY

//------------------------------------------------------------------------------
// cholmod_updown
//------------------------------------------------------------------------------

// Compute the new LDL' factorization of LDL'+CC' (an update) or LDL'-CC'
// (a downdate).  The factor object L need not be an LDL' factorization; it
// is converted to one if it isn't.

int CHOLMOD(updown)
(
    // input:
    int update,         // TRUE for update, FALSE for downdate
    cholmod_sparse *C,  // the incoming sparse update
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_common *Common
)
{
    return (CHOLMOD(updown_mask2) (update, C, /* colmark: */ NULL,
        /* mask: */ NULL, /* maskmark: */ 0,
        L, /* X: */ NULL, /* DeltaB: */ NULL, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_updown_solve
//------------------------------------------------------------------------------

// Does the same as cholmod_updown, except that it also updates/downdates the
// solution to Lx=b+DeltaB.  x and b must be n-by-1 dense matrices.  b is not
// need as input to this routine, but a sparse change to b is (DeltaB).  Only
// entries in DeltaB corresponding to columns modified in L are accessed; the
// rest are ignored.

int CHOLMOD(updown_solve)
(
    // input:
    int update,         // TRUE for update, FALSE for downdate
    cholmod_sparse *C,  // the incoming sparse update
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{
    return (CHOLMOD(updown_mask2) (update, C, /* colmark: */ NULL,
        /* mask: */ NULL, /* maskmark: */ 0,
        L, X, DeltaB, Common)) ;
}

//------------------------------------------------------------------------------
// Power2
//------------------------------------------------------------------------------

// Power2 [i] is smallest power of 2 that is >= i (for i in range 0 to 8)

static size_t Power2 [ ] =
{
//  0  1  2  3  4  5  6  7  8
    0, 1, 2, 4, 4, 8, 8, 8, 8
} ;

//------------------------------------------------------------------------------
// debug routines
//------------------------------------------------------------------------------

#ifndef NDEBUG

static void dump_set (Int s, Int **Set_ps1, Int **Set_ps2, Int j, Int n,
        cholmod_common *Common)
{
    Int *p, len, i, ilast ;

    if (CHOLMOD(dump) < -1)
    {
        // no checks if debug level is -2 or less
        return ;
    }

    len = Set_ps2 [s] - Set_ps1 [s] ;
    PRINT2 (("Set s: "ID" len: "ID":", s, len)) ;
    ASSERT (len > 0) ;
    ilast = j ;
    for (p = Set_ps1 [s] ; p < Set_ps2 [s] ; p++)
    {
        i = *p ;
        PRINT3 ((" "ID"", i)) ;
        ASSERT (i > ilast && i < n) ;
        ilast = i ;
    }
    PRINT3 (("\n")) ;
}

#endif

//------------------------------------------------------------------------------
// Path_type
//------------------------------------------------------------------------------

// A path is a set of nodes of the etree which are all affected by the same
// columns of C.

typedef struct Path_struct
{
    Int start ;         // column at which to start, or EMPTY if initial
    Int end ;           // column at which to end, or EMPTY if initial
    Int ccol ;          // column of C to which path refers
    Int parent ;        // parent path
    Int c ;             // child of j along this path
    Int next ;          // next path in link list
    Int rank ;          // number of rank-1 paths merged onto this path
    Int order ;         // dfs order of this path
    Int wfirst ;        // first column of W to affect this path
    Int pending ;       // column at which the path is pending
    Int botrow ;        // for partial update/downdate of solution to Lx=b

} Path_type ;


//------------------------------------------------------------------------------
// dfs
//------------------------------------------------------------------------------

// Compute the DFS order of the set of paths.  This can be recursive because
// there are at most 23 paths to sort: one for each column of C (8 at most),
// and one for each node in a balanced binary tree with 8 leaves (15).
// Stack overflow is thus not a problem.

static void dfs
(
    Path_type *Path,    // the set of Paths
    Int k,              // the rank of the update/downdate
    Int path,           // which path to work on
    Int *path_order,    // the current path order
    Int *w_order,       // the current order of the columns of W
    Int depth,
    Int npaths          // total number of paths
)
{
    Int c ;             // child path

    ASSERT (path >= 0 && path < npaths) ;
    if (path < k)
    {
        // this is a leaf node, corresponding to column W (:,path)
        // and column C (:, Path [path].ccol)
        ASSERT (Path [path].ccol >= 0) ;
        Path [path].wfirst = *w_order ;
        Path [path].order = *w_order ;
        (*w_order)++ ;
    }
    else
    {
        // this is a non-leaf path, within the tree
        ASSERT (Path [path].c != EMPTY) ;
        ASSERT (Path [path].ccol == EMPTY) ;
        // order each child path
        for (c = Path [path].c ; c != EMPTY ; c = Path [c].next)
        {
            dfs (Path, k, c, path_order, w_order, depth+1, npaths) ;
            if (Path [path].wfirst == EMPTY)
            {
                Path [path].wfirst = Path [c].wfirst ;
            }
        }
        // order this path next
        Path [path].order = (*path_order)++ ;
    }
}

//------------------------------------------------------------------------------
// numeric update/downdate routines
//------------------------------------------------------------------------------

// naming scheme for the update/downdate worker methods:
//
// single case:  s_updown_k_rank
// double case:  d_updown_k_rank
//
// where k is 1, 2, 4, or 8, and rank is r for the t_cholmod_updown_wdim
// method, or 1 to 8 for the lowest level kernels.  See t_cholmod_updown_wdim.c
// for details.

#define UPDOWN_METHOD(prefix,k,rank) prefix ## updown_ ## k ## _ ## rank

#define DOUBLE
#define REAL
#include "t_cholmod_updown_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_updown_worker.c"

//------------------------------------------------------------------------------
// cholmod_updown_mark
//------------------------------------------------------------------------------

// Update/downdate LDL' +/- C*C', and update/downdate selected portions of the
// solution to Lx=b.
//
// The original system is L*x = b.  The new system is Lnew*xnew = b + deltab.
// deltab(i) can be nonzero only if column i of L is modified by the update/
// downdate.  If column i is not modified, the deltab(i) is not accessed.
//
// The solution to Lx=b is not modified if either X or DeltaB are NULL.
//
// Rowmark and colmark:
// --------------------
//
// rowmark and colmark affect which portions of L take part in the update/
// downdate of the solution to Lx=b.  They do not affect how L itself is
// updated/downdated.  They are both ignored if X or DeltaB are NULL.
//
// If not NULL, rowmark is an integer array of size n where L is n-by-n.
// rowmark [j] defines the part of column j of L that takes part in the update/
// downdate of the forward solve, Lx=b.  Specifically, if i = rowmark [j],
// then L(j:i-1,j) is used, and L(i:end,j) is ignored.
//
// If not NULL, colmark is an integer array of size C->ncol.  colmark [ccol]
// for a column C(:,ccol) redefines those parts of L that take part in the
// update/downdate of Lx=b.  Each column of C affects a set of columns of L.
// If column ccol of C affects column j of L, then the new rowmark [j] of
// column j of L is defined as colmark [ccol].  In a multiple-rank update/
// downdate, if two or more columns of C affect column j, its new rowmark [j]
// is the colmark of the least-numbered column of C.  colmark is ignored if
// it is NULL, in which case rowmark is not modified.  If colmark [ccol] is
// EMPTY (-1), then rowmark is not modified for that particular column of C.
// colmark is ignored if it is NULL, or rowmark, X, or DeltaB are NULL.
//
// The algorithm for modifying the solution to Lx=b when rowmark and colmark
// are NULL is as follows:
//
//      for each column j of L that is modified:
//          deltab (j:end) += L (j:end,j) * x(j)
//      modify L
//      for each column j of L that is modified:
//          x (j) = deltab (j)
//          deltab (j) = 0
//          deltab (j+1:end) -= L (j+1:end,j) * x(j)
//
// If rowmark is non-NULL but colmark is NULL:
//
//      for each column j of L that is modified:
//          deltab (j:rowmark(j)-1) += L (j:rowmark(j)-1,j) * x(j)
//      modify L
//      for each column j of L that is modified:
//          x (j) = deltab (j)
//          deltab (j) = 0
//          deltab (j+1:rowmark(j)-1) -= L (j+1:rowmark(j)-1,j) * x(j)
//
// If both rowmark and colmark are non-NULL:
//
//      for each column j of L that is modified:
//          deltab (j:rowmark(j)-1) += L (j:rowmark(j)-1,j) * x(j)
//      modify L
//      for each column j of L that is modified:
//          modify rowmark (j) according to colmark
//      for each column j of L that is modified:
//          x (j) = deltab (j)
//          deltab (j) = 0
//          deltab (j+1:rowmark(j)-1) -= L (j+1:rowmark(j)-1,j) * x(j)
//
// Note that if the rank of C exceeds k = Common->maxrank (which is 2, 4, or 8),
// then the update/downdate is done as a series of rank-k updates.  In this
// case, the above algorithm is repeated for each block of k columns of C.
//
// Unless it leads to no changes in rowmark, colmark should be used only if
// C->ncol <= Common->maxrank, because the update/downdate is done with maxrank
// columns at a time.  Otherwise, the results are undefined.
//
// This routine is an "expert" routine.  It is meant for use in LPDASA only.

int CHOLMOD(updown_mark)
(
    // input:
    int update,         // TRUE for update, FALSE for downdate
    cholmod_sparse *C,  // the incoming sparse update
    Int *colmark,       // array of size n.  See cholmod_updown.c for details
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{
    return (CHOLMOD(updown_mask2) (update, C, colmark,
        /* mask: */ NULL, /* maskmark: */ 0,
        L, X, DeltaB, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_updown_mask
//------------------------------------------------------------------------------

int CHOLMOD(updown_mask)
(
    // input:
    int update,         // TRUE for update, FALSE for downdate
    cholmod_sparse *C,  // the incoming sparse update
    Int *colmark,       // array of size n.  See cholmod_updown.c for details
    Int *mask,          // size n
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{
    return (CHOLMOD(updown_mask2) (update, C, colmark,
        mask, /* maskmark: */ 0,
        L, X, DeltaB, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_updown_mask2
//------------------------------------------------------------------------------

int CHOLMOD(updown_mask2)
(
    // input:
    int update,         // TRUE for update, FALSE for downdate
    cholmod_sparse *C,  // the incoming sparse update
    Int *colmark,       // array of size n.  See cholmod_updown.c for details
    Int *mask,          // size n
    Int maskmark,
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
    RETURN_IF_NULL (C, FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_PATTERN, CHOLMOD_REAL, FALSE) ;
    RETURN_IF_XTYPE_INVALID (C, CHOLMOD_REAL, CHOLMOD_REAL, FALSE) ;
    Int n = L->n ;
    Int cncol = C->ncol ;
    if (!(C->sorted))
    {
        ERROR (CHOLMOD_INVALID, "C must have sorted columns") ;
        return (FALSE) ;
    }
    if (L->n != C->nrow)
    {
        ERROR (CHOLMOD_INVALID, "C and L dimensions do not match") ;
        return (FALSE) ;
    }
    if (L->dtype != C->dtype)
    {
        ERROR (CHOLMOD_INVALID, "C and L must have the same dtype") ;
        return (FALSE) ;
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
    Common->modfl = 0 ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // Note: cholmod_rowadd and cholmod_rowdel use the second n doubles in
    // Common->Xwork for Cx, and then perform a rank-1 update here, which uses
    // the first n doubles in Common->Xwork.   Both the rowadd and rowdel
    // routines allocate enough workspace so that Common->Xwork isn't destroyed
    // below.  Also, both cholmod_rowadd and cholmod_rowdel use the second n
    // ints in Common->Iwork for C->i.

    // make sure maxrank is in the proper range
    size_t maxrank = CHOLMOD(maxrank) (n, Common) ;
    Int k = MIN (cncol, (Int) maxrank) ;    // maximum k is wdim
    size_t wdim = Power2 [k] ;              // number of columns needed in W
    ASSERT (wdim <= maxrank) ;
    PRINT1 (("updown wdim final "ID" k "ID"\n", (Int) wdim, k)) ;

    // w = wdim * n
    int ok = TRUE ;
    size_t w = CHOLMOD(mult_size_t) (L->n, wdim, &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (FALSE) ;
    }

    CHOLMOD(alloc_work) (L->n, L->n, w, L->dtype, Common) ;
    if (Common->status < CHOLMOD_OK || maxrank == 0)
    {
        // out of memory, L is returned unchanged
        return (FALSE) ;
    }

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

    CLEAR_FLAG (Common) ;
    ASSERT (check_flag (Common)) ;

    PRINT1 (("updown, rank %g update %d\n", (double) C->ncol, update)) ;
    DEBUG (CHOLMOD(dump_factor) (L, "input L for updown", Common)) ;
    ASSERT (CHOLMOD(dump_sparse) (C, "input C for updown", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // quick return
    //--------------------------------------------------------------------------

    if (cncol <= 0 || n == 0)
    {
        // nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // update/downdate
    //--------------------------------------------------------------------------

    switch (L->dtype & 4)
    {
        case CHOLMOD_SINGLE:
            ok = rs_cholmod_updown_worker (k, update, C, colmark, mask,
                maskmark, L, X, DeltaB, Common) ;
            break ;

        case CHOLMOD_DOUBLE:
            ok = rd_cholmod_updown_worker (k, update, C, colmark, mask,
                maskmark, L, X, DeltaB, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;
    DEBUG (CHOLMOD(dump_factor) (L, "output L for updown", Common)) ;
    return (ok) ;
}

#endif
#endif

