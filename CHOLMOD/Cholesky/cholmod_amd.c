//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/cholmod_amd: AMD interface for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// CHOLMOD interface to the AMD ordering routine.  Orders A if the matrix is
// symmetric.  On output, Perm [k] = i if row/column i of A is the kth
// row/column of P*A*P'.  This corresponds to A(p,p) in MATLAB notation.
//
// If A is unsymmetric, cholmod_amd orders A*A'.  On output, Perm [k] = i if
// row/column i of A*A' is the kth row/column of P*A*A'*P'.  This corresponds to
// A(p,:)*A(p,:)' in MATLAB notation.  If f is present, A(p,f)*A(p,f)' is
// ordered.
//
// Computes the flop count for a subsequent LL' factorization, the number
// of nonzeros in L, and the number of nonzeros in the matrix ordered (A,
// A*A' or A(:,f)*A(:,f)').
//
// workspace: Iwork (6*nrow). Head (nrow).
//
// Allocates a temporary copy of A+A' or A*A' (with
// both upper and lower triangular parts) as input to AMD.
//
// Supports any xtype (pattern, real, complex, or zomplex) and any dtype.

#include "cholmod_internal.h"

#ifndef NCHOLESKY

#include "amd.h"
#if (!defined (AMD_VERSION) || (AMD_VERSION < AMD_VERSION_CODE (3,3)))
#error "CHOLMOD:Cholesky requires AMD 3.3.1 or later"
#endif

//------------------------------------------------------------------------------
// cholmod_amd
//------------------------------------------------------------------------------

int CHOLMOD(amd)
(
    // input:
    cholmod_sparse *A,  // matrix to order
    Int *fset,          // subset of 0:(A->ncol)-1
    size_t fsize,       // size of fset
    // output:
    Int *Perm,          // size A->nrow, output permutation
    cholmod_common *Common
)
{

    double Info [AMD_INFO], Control2 [AMD_CONTROL], *Control ;
    Int *Cp, *Len, *Nv, *Head, *Elen, *Degree, *Wi, *Iwork, *Next ;
    cholmod_sparse *C ;
    Int j, n, cnz ;
    int ok = TRUE ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    n = A->nrow ;

    RETURN_IF_NULL (Perm, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    Common->status = CHOLMOD_OK ;
    if (n == 0)
    {
        // nothing to do
        Common->fl = 0 ;
        Common->lnz = 0 ;
        Common->anz = 0 ;
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    // Note: this is less than the space used in cholmod_analyze, so if
    // cholmod_amd is being called by that routine, no space will be
    // allocated.

    // s = MAX (6*nrow, ncol)
    size_t s = CHOLMOD(mult_size_t) (A->nrow, 6, &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (FALSE) ;
    }
    s = MAX (s, A->ncol) ;

    CHOLMOD(allocate_work) (A->nrow, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }

    Iwork  = Common->Iwork ;
    Degree = Iwork ;                    // size n
    Wi     = Iwork + n ;                // size n
    Len    = Iwork + 2*((size_t) n) ;   // size n
    Nv     = Iwork + 3*((size_t) n) ;   // size n
    Next   = Iwork + 4*((size_t) n) ;   // size n
    Elen   = Iwork + 5*((size_t) n) ;   // size n

    Head = Common->Head ;   // size n+1, but only n is used

    //--------------------------------------------------------------------------
    // construct the input matrix for AMD
    //--------------------------------------------------------------------------

    if (A->stype == 0)
    {
        // C = A*A' or A(:,f)*A(:,f)', add extra space of nnz(C)/2+n to C
        C = CHOLMOD(aat) (A, fset, fsize, -2, Common) ;
    }
    else
    {
        // C = A+A', but use only the upper triangular part of A if A->stype = 1
        // and only the lower part of A if A->stype = -1.  Add extra space of
        // nnz(C)/2+n to C.
        C = CHOLMOD(copy) (A, 0, -2, Common) ;
    }

    if (Common->status < CHOLMOD_OK)
    {
        // out of memory, fset invalid, or other error
        return (FALSE) ;
    }

    Cp = C->p ;
    for (j = 0 ; j < n ; j++)
    {
        Len [j] = Cp [j+1] - Cp [j] ;
    }

    // C does not include the diagonal, and both upper and lower parts.
    // Common->anz includes the diagonal, and just the lower part of C
    cnz = Cp [n] ;
    Common->anz = cnz / 2 + n ;

    //--------------------------------------------------------------------------
    // order C using AMD
    //--------------------------------------------------------------------------

    // get parameters
    if (Common->current < 0 || Common->current >= CHOLMOD_MAXMETHODS)
    {
        // use AMD defaults
        Control = NULL ;
    }
    else
    {
        Control = Control2 ;
        Control [AMD_DENSE] = Common->method [Common->current].prune_dense ;
        Control [AMD_AGGRESSIVE] = Common->method [Common->current].aggressive ;
    }

    #if defined ( CHOLMOD_INT64 )
    amd_l2 (n, C->p,  C->i, Len, C->nzmax, cnz, Nv, Next, Perm, Head, Elen,
            Degree, Wi, Control, Info) ;
    #else
    amd_2 (n, C->p,  C->i, Len, C->nzmax, cnz, Nv, Next, Perm, Head, Elen,
            Degree, Wi, Control, Info) ;
    #endif

    // LL' flop count.  Need to subtract n for LL' flop count.  Note that this
    // is a slight upper bound which is often exact (see AMD/Source/amd_2.c for
    // details).  cholmod_analyze computes an exact flop count and fill-in.
    Common->fl = Info [AMD_NDIV] + 2 * Info [AMD_NMULTSUBS_LDL] + n ;

    // Info [AMD_LNZ] excludes the diagonal
    Common->lnz = n + Info [AMD_LNZ] ;

    //--------------------------------------------------------------------------
    // free the AMD workspace and clear the persistent workspace in Common
    //--------------------------------------------------------------------------

    ASSERT (IMPLIES (Common->status == CHOLMOD_OK,
                CHOLMOD(dump_perm) (Perm, n, n, "AMD2 perm", Common))) ;
    CHOLMOD(free_sparse) (&C, Common) ;
    for (j = 0 ; j <= n ; j++)
    {
        Head [j] = EMPTY ;
    }
    return (TRUE) ;
}
#endif

