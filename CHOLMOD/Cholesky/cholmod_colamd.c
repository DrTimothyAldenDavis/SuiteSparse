//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/cholmod_colamd: COLAMD interface for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// CHOLMOD interface to the COLAMD ordering routine (version 2.4 or later).
// Finds a permutation p such that the Cholesky factorization of PAA'P' is
// sparser than AA' using colamd.  If the postorder input parameter is TRUE,
// the column etree is found and postordered, and the colamd ordering is then
// combined with its postordering.  A must be unsymmetric.
//
// There can be no duplicate entries in f.
// f can be length 0 to n if A is m-by-n.
//
// workspace: Iwork (4*nrow+ncol), Head (nrow+1), Flag (nrow)
//      Allocates a copy of its input matrix, which
//      is then used as CCOLAMD's workspace.
//
// Supports any xtype (pattern, real, complex, or zomplex)

#include "cholmod_internal.h"

#ifndef NCHOLESKY

#include "colamd.h"
#if (!defined (COLAMD_VERSION) || (COLAMD_VERSION < COLAMD_VERSION_CODE (3,3)))
#error "CHOLMOD:Cholesky requires COLAMD 3.3.1 or later"
#endif

//------------------------------------------------------------------------------
// cholmod_colamd
//------------------------------------------------------------------------------

int CHOLMOD(colamd)
(
    // input:
    cholmod_sparse *A,  // matrix to order
    Int *fset,          // subset of 0:(A->ncol)-1
    size_t fsize,       // size of fset
    int postorder,      // if TRUE, follow with a coletree postorder
    // output:
    Int *Perm,          // size A->nrow, output permutation
    cholmod_common *Common
)
{

    double knobs [COLAMD_KNOBS] ;
    cholmod_sparse *C ;
    Int *NewPerm, *Parent, *Post, *Work2n ;
    Int k, nrow, ncol ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (Perm, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    if (A->stype != 0)
    {
        ERROR (CHOLMOD_INVALID, "matrix must be unsymmetric") ;
        return (FALSE) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    nrow = A->nrow ;
    ncol = A->ncol ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // Note: this is less than the space used in cholmod_analyze, so if
    // cholmod_colamd is being called by that routine, no space will be
    // allocated.

    // s = 4*nrow + ncol
    int ok = TRUE ;
    size_t s = CHOLMOD(mult_size_t) (A->nrow, 4, &ok) ;
    s = CHOLMOD(add_size_t) (s, A->ncol, &ok) ;

    #if defined ( CHOLMOD_INT64 )
    size_t alen = colamd_l_recommended (A->nzmax, ncol, nrow) ;
    colamd_l_set_defaults (knobs) ;
    #else
    size_t alen = colamd_recommended (A->nzmax, ncol, nrow) ;
    colamd_set_defaults (knobs) ;
    #endif

    if (!ok || alen == 0)
    {
        ERROR (CHOLMOD_TOO_LARGE, "matrix invalid or too large") ;
        return (FALSE) ;
    }

    CHOLMOD(allocate_work) (0, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // allocate COLAMD workspace
    //--------------------------------------------------------------------------

    // C is purely symbolic, so C->dtype doesn't actually matter, but it must
    // match A->dtype for the call to cholmod_transpose_unsym below.
    C = CHOLMOD(allocate_sparse) (ncol, nrow, alen, TRUE, TRUE, 0,
            CHOLMOD_PATTERN + A->dtype, Common) ;

    //--------------------------------------------------------------------------
    // copy (and transpose) the input matrix A into the colamd workspace
    //--------------------------------------------------------------------------

    // C = A (:,f)', which also packs A if needed.
    // workspace: Iwork (nrow if no fset; MAX (nrow,ncol) if fset)
    ok = CHOLMOD(transpose_unsym) (A, 0, NULL, fset, fsize, C, Common) ;

    //--------------------------------------------------------------------------
    // order the matrix (destroys the contents of C->i and C->p)
    //--------------------------------------------------------------------------

    // get parameters
    if (Common->current < 0 || Common->current >= CHOLMOD_MAXMETHODS)
    {
        // this is the CHOLMOD default, not the COLAMD default
        knobs [COLAMD_DENSE_ROW] = -1 ;
    }
    else
    {
        // get the knobs from the Common parameters
        knobs [COLAMD_DENSE_COL] = Common->method[Common->current].prune_dense ;
        knobs [COLAMD_DENSE_ROW] = Common->method[Common->current].prune_dense2;
        knobs [COLAMD_AGGRESSIVE] = Common->method[Common->current].aggressive ;
    }

    if (ok)
    {
        Int *Cp ;
        Int stats [COLAMD_STATS] ;
        Cp = C->p ;

        #if defined ( CHOLMOD_INT64 )
        colamd_l (ncol, nrow, alen, C->i, Cp, knobs, stats) ;
        #else
        colamd (ncol, nrow, alen, C->i, Cp, knobs, stats) ;
        #endif

        ok = stats [COLAMD_STATUS] ;
        ok = (ok == COLAMD_OK || ok == COLAMD_OK_BUT_JUMBLED) ;
        // permutation returned in C->p, if the ordering succeeded
        for (k = 0 ; k < nrow ; k++)
        {
            Perm [k] = Cp [k] ;
        }
    }

    CHOLMOD(free_sparse) (&C, Common) ;

    //--------------------------------------------------------------------------
    // column etree postordering
    //--------------------------------------------------------------------------

    if (postorder)
    {
        // use the last 2*n space in Iwork for Parent and Post
        Work2n = Common->Iwork ;
        Work2n += 2*((size_t) nrow) + ncol ;
        Parent = Work2n ;               // size nrow
        Post   = Work2n + nrow ;        // size nrow

        // workspace: Iwork (2*nrow+ncol), Flag (nrow), Head (nrow+1)
        ok = ok && CHOLMOD(analyze_ordering) (A, CHOLMOD_COLAMD, Perm, fset,
                fsize, Parent, Post, NULL, NULL, NULL, Common) ;

        // combine the colamd permutation with its postordering
        if (ok)
        {
            NewPerm = Common->Iwork ;   // size nrow
            for (k = 0 ; k < nrow ; k++)
            {
                NewPerm [k] = Perm [Post [k]] ;
            }
            for (k = 0 ; k < nrow ; k++)
            {
                Perm [k] = NewPerm [k] ;
            }
        }
    }

    return (ok) ;
}
#endif

