//------------------------------------------------------------------------------
// CHOLMOD/Supernodal/cholmod_super_numeric: supernodal Cholesky factorization
//------------------------------------------------------------------------------

// CHOLMOD/Supernodal Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Computes the Cholesky factorization of A+beta*I or A*F+beta*I.  Only the
// the lower triangular part of A+beta*I or A*F+beta*I is accessed.  The
// matrices A and F must already be permuted according to the fill-reduction
// permutation L->Perm.  cholmod_factorize is an "easy" wrapper for this code
// which applies that permutation.  beta is real.
//
// Symmetric case: A is a symmetric (lower) matrix.  F is not accessed.
// With a fill-reducing permutation, A(p,p) should be passed instead, where is
// p is L->Perm.
//
// Unsymmetric case: A is unsymmetric, and F must be present.  Normally, F=A'.
// With a fill-reducing permutation, A(p,f) and A(p,f)' should be passed as A
// and F, respectively, where f is a list of the subset of the columns of A.
//
// The input factorization L must be supernodal (L->is_super is TRUE).  It can
// either be symbolic or numeric.  In the first case, L has been analyzed by
// cholmod_analyze or cholmod_super_symbolic, but the matrix has not yet been
// numerically factorized.  The numerical values are allocated here and the
// factorization is computed.  In the second case, a prior matrix has been
// analyzed and numerically factorized, and a new matrix is being factorized.
// The numerical values of L are replaced with the new numerical factorization.
//
// L->is_ll is ignored, and set to TRUE.  This routine always computes an LL'
// factorization.  Supernodal LDL' factorization is not (yet) supported.
// FUTURE WORK: perform a supernodal LDL' factorization if L->is_ll is FALSE.
//
// If the matrix is not positive definite the routine returns TRUE, but sets
// Common->status to CHOLMOD_NOT_POSDEF and L->minor is set to the column at
// which the failure occurred.  The supernode containing the non-positive
// diagonal entry is set to zero (this includes columns to the left of L->minor
// in the same supernode), as are all subsequent supernodes.
//
// workspace: Flag (nrow), Head (nrow+1), Iwork (2*nrow + 5*nsuper).
//      Allocates temporary space of size L->maxcsize * sizeof(double)
//      (twice that for the complex/zomplex case).
//
// If L is supernodal symbolic on input, it is converted to a supernodal numeric
// factor on output, with an xtype of real if A is real, or complex if A is
// complex or zomplex.  If L is supernodal numeric on input, its xtype must
// match A (except that L can be complex and A zomplex).  The xtype of A and F
// must match.  The dtypes of all matrices must match.

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NSUPERNODAL

//------------------------------------------------------------------------------
// GPU templates: double and double complex cases only
//------------------------------------------------------------------------------

#define DOUBLE
#ifdef CHOLMOD_INT64
#ifdef CHOLMOD_HAS_CUDA
#include "cholmod_gpu_kernels.h"
#define REAL
#include "../GPU/t_cholmod_gpu.c"
#define COMPLEX
#include "../GPU/t_cholmod_gpu.c"
#endif
#endif

//------------------------------------------------------------------------------
// CPU templates:  all cases
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_super_numeric_worker.c"
#define COMPLEX
#include "t_cholmod_super_numeric_worker.c"
#define ZOMPLEX
#include "t_cholmod_super_numeric_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_super_numeric_worker.c"
#define COMPLEX
#include "t_cholmod_super_numeric_worker.c"
#define ZOMPLEX
#include "t_cholmod_super_numeric_worker.c"

//------------------------------------------------------------------------------
// cholmod_super_numeric
//------------------------------------------------------------------------------

// Returns TRUE if successful, or if the matrix is not positive definite.
// Returns FALSE if out of memory, inputs are invalid, or other fatal error
// occurs.

int CHOLMOD(super_numeric)
(
    // input:
    cholmod_sparse *A,  // matrix to factorize
    cholmod_sparse *F,  // F = A' or A(:,f)'
    double beta [2],    // beta*I is added to diagonal of matrix to factorize
    // input/output:
    cholmod_factor *L,  // factorization
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_dense *C ;
    Int *Super, *Map, *SuperMap ;
    size_t maxcsize ;
    Int nsuper, n, i, k, s, nrow ;
    int symbolic ;

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_PATTERN, CHOLMOD_COMPLEX, FALSE) ;
    if (A->stype < 0)
    {
        if (A->nrow != A->ncol || A->nrow != L->n)
        {
            ERROR (CHOLMOD_INVALID, "invalid dimensions") ;
            return (FALSE) ;
        }
    }
    else if (A->stype == 0)
    {
        if (A->nrow != L->n)
        {
            ERROR (CHOLMOD_INVALID, "invalid dimensions") ;
            return (FALSE) ;
        }
        RETURN_IF_NULL (F, FALSE) ;
        RETURN_IF_XTYPE_INVALID (F, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
        if (A->nrow != F->ncol || A->ncol != F->nrow || F->stype != 0)
        {
            ERROR (CHOLMOD_INVALID, "F invalid") ;
            return (FALSE) ;
        }
        if (A->xtype != F->xtype || A->dtype != F->dtype)
        {
            ERROR (CHOLMOD_INVALID, "A and F must have same xtype and dtype") ;
            return (FALSE) ;
        }
    }
    else
    {
        // symmetric upper case not supported
        ERROR (CHOLMOD_INVALID, "symmetric upper case not supported") ;
        return (FALSE) ;
    }
    if (!(L->is_super))
    {
        ERROR (CHOLMOD_INVALID, "L not supernodal") ;
        return (FALSE) ;
    }
    if (L->xtype != CHOLMOD_PATTERN)
    {
        if (! ((A->xtype == CHOLMOD_REAL    && L->xtype == CHOLMOD_REAL)
            || (A->xtype == CHOLMOD_COMPLEX && L->xtype == CHOLMOD_COMPLEX)
            || (A->xtype == CHOLMOD_ZOMPLEX && L->xtype == CHOLMOD_COMPLEX)))
        {
            ERROR (CHOLMOD_INVALID, "complex type mismatch") ;
            return (FALSE) ;
        }
        if (A->dtype != L->dtype)
        {
            ERROR (CHOLMOD_INVALID, "A and L must have the same dtype") ;
            return (FALSE) ;
        }
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate workspace in Common
    //--------------------------------------------------------------------------

    nsuper = L->nsuper ;
    maxcsize = L->maxcsize ;
    nrow = A->nrow ;
    n = nrow ;

    PRINT1 (("nsuper "ID" maxcsize %g\n", nsuper, (double) maxcsize)) ;
    ASSERT (nsuper >= 0 && maxcsize > 0) ;

    // w = 2*nrow + 5*nsuper
    int ok = TRUE ;
    size_t w = CHOLMOD(mult_size_t) (A->nrow, 2, &ok) ;
    size_t t = CHOLMOD(mult_size_t) (L->nsuper, 5, &ok) ;
    w = CHOLMOD(add_size_t) (w, t, &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (FALSE) ;
    }

    CHOLMOD(allocate_work) (A->nrow, w, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;

    //--------------------------------------------------------------------------
    // get the current factor L and allocate numerical part, if needed
    //--------------------------------------------------------------------------

    Super = L->super ;
    symbolic = (L->xtype == CHOLMOD_PATTERN) ;
    if (symbolic)
    {
        // convert to supernodal numeric by allocating L->x
        L->dtype = A->dtype ;       // ensure L has the same dtype as A
        CHOLMOD(change_factor) (
                (A->xtype == CHOLMOD_REAL) ? CHOLMOD_REAL : CHOLMOD_COMPLEX,
                TRUE, TRUE, TRUE, TRUE, L, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // the factor L remains in symbolic supernodal form
            return (FALSE) ;
        }
    }
    ASSERT (L->dtype == A->dtype) ;
    ASSERT (L->dtype == CHOLMOD_DOUBLE || L->dtype == CHOLMOD_SINGLE) ;
    ASSERT (L->xtype == CHOLMOD_REAL   || L->xtype == CHOLMOD_COMPLEX) ;

    // supernodal LDL' is not supported
    L->is_ll = TRUE ;

    //--------------------------------------------------------------------------
    // get more workspace
    //--------------------------------------------------------------------------

    C = CHOLMOD(allocate_dense) (maxcsize, 1, maxcsize, L->xtype + L->dtype,
        Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        int status = Common->status ;
        if (symbolic)
        {
            // Change L back to symbolic, since the numeric values are not
            // initialized.  This cannot fail.
            CHOLMOD(change_factor) (CHOLMOD_PATTERN, TRUE, TRUE, TRUE, TRUE, L,
                Common) ;
        }
        // the factor L is now back to the form it had on input
        Common->status = status ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    SuperMap = Common->Iwork ;  // size n
    Map = Common->Flag ;        // size n, use Flag as workspace for Map array
    for (i = 0 ; i < n ; i++)
    {
        Map [i] = EMPTY ;
    }

    //--------------------------------------------------------------------------
    // find the mapping of nodes to relaxed supernodes
    //--------------------------------------------------------------------------

    // SuperMap [k] = s if column k is contained in supernode s
    for (s = 0 ; s < nsuper ; s++)
    {
        PRINT1 (("Super ["ID"] "ID" ncols "ID"\n",
                    s, Super[s], Super[s+1]-Super[s])) ;
        for (k = Super [s] ; k < Super [s+1] ; k++)
        {
            SuperMap [k] = s ;
            PRINT2 (("relaxed SuperMap ["ID"] = "ID"\n", k, SuperMap [k])) ;
        }
    }

    //--------------------------------------------------------------------------
    // supernodal numerical factorization, using template routine
    //--------------------------------------------------------------------------

    float s_beta [2] ;
    s_beta [0] = (float) beta [0] ;
    s_beta [1] = (float) beta [1] ;

    switch ((A->xtype + A->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            ok = rs_cholmod_super_numeric_worker (A, F, s_beta, L, C, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            ok = cs_cholmod_super_numeric_worker (A, F, s_beta, L, C, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            // A is zomplex, but L is complex
            ok = zs_cholmod_super_numeric_worker (A, F, s_beta, L, C, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            ok = rd_cholmod_super_numeric_worker (A, F, beta, L, C, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            ok = cd_cholmod_super_numeric_worker (A, F, beta, L, C, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            // A is zomplex, but L is complex
            ok = zd_cholmod_super_numeric_worker (A, F, beta, L, C, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // clear Common workspace, free temp workspace C, and return
    //--------------------------------------------------------------------------

    // Flag array was used as workspace, clear it
    Common->mark = EMPTY ;
    CLEAR_FLAG (Common) ;
    ASSERT (check_flag (Common)) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;
    CHOLMOD(free_dense) (&C, Common) ;
    return (ok) ;
}

#endif
#endif

