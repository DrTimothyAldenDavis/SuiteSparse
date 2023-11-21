//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/cholmod_solve: solve a linear system
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Solve one of the following systems.  D is identity for an LL' factorization,
// in which the D operation is skipped:
//
//      Ax=b        0: CHOLMOD_A     x = P' * (L' \ (D \ (L \ (P * b))))
//      LDL'x=b     1: CHOLMOD_LDLt  x =      (L' \ (D \ (L \ (    b))))
//      LDx=b       2: CHOLMOD_LD    x =      (     (D \ (L \ (    b))))
//      DL'x=b      3: CHOLMOD_DLt   x =      (L' \ (D \ (    (    b))))
//      Lx=b        4: CHOLMOD_L     x =      (     (    (L \ (    b))))
//      L'x=b       5: CHOLMOD_Lt    x =      (L' \ (    (    (    b))))
//      Dx=b        6: CHOLMOD_D     x =      (     (D \ (    (    b))))
//      x=Pb        7: CHOLMOD_P     x =      (     (    (    (P * b))))
//      x=P'b       8: CHOLMOD_Pt    x = P' * (     (    (    (    b))))
//
// The factorization can be simplicial LDL', simplicial LL', or supernodal LL'.
// For an LL' factorization, D is the identity matrix.  Thus CHOLMOD_LD and
// CHOLMOD_L solve the same system if an LL' factorization was performed,
// for example.
//
// The supernodal solver uses BLAS routines:
//
//      dtrsv, dgemv, dtrsm, dgemm: double real
//      ztrsv, zgemv, ztrsm, zgemm: double complex
//      strsv, sgemv, strsm, sgemm: float real
//      ctrsv, cgemv, ctrsm, cgemm: float complex
//
// If both L and B are real, then X is returned real.  If either is complex
// or zomplex, X is returned as either complex or zomplex, depending on the
// Common->prefer_zomplex parameter.
//
// Supports any numeric xtype (real, complex, or zomplex; pattern-only matrices
// not supported), and any dtype.
//
// This routine does not check to see if the diagonal of L or D is zero,
// because sometimes a partial solve can be done with indefinite or singular
// matrix.  If you wish to check in your own code, test L->minor.  If L->minor
// == L->n, then the matrix has no zero diagonal entries.  If k = L->minor <
// L->n, then L(k,k) is zero for an LL' factorization, or D(k,k) is zero for an
// LDL' factorization.
//
// This routine returns X as NULL only if it runs out of memory.  If L is
// indefinite or singular, then X may contain Inf's or NaN's, but it will exist
// on output.

#include "cholmod_internal.h"

#ifndef NCHOLESKY

//------------------------------------------------------------------------------
// P(k): permutation macro for t_cholmod_*_worker methods
//------------------------------------------------------------------------------

// If Perm is NULL, it is interpretted as the identity permutation

#define P(k) ((Perm == NULL) ? (k) : Perm [k])

//------------------------------------------------------------------------------
// t_cholmod_solve_worker
//------------------------------------------------------------------------------

#define DOUBLE
#include "t_cholmod_psolve_worker.c"
#define REAL
#include "t_cholmod_solve_worker.c"
#define COMPLEX
#include "t_cholmod_solve_worker.c"
#define ZOMPLEX
#include "t_cholmod_solve_worker.c"

#undef  DOUBLE
#define SINGLE
#include "t_cholmod_psolve_worker.c"
#define REAL
#include "t_cholmod_solve_worker.c"
#define COMPLEX
#include "t_cholmod_solve_worker.c"
#define ZOMPLEX
#include "t_cholmod_solve_worker.c"

//------------------------------------------------------------------------------
// cholmod_solve
//------------------------------------------------------------------------------

// Solve a linear system.
//
// The factorization can be simplicial LDL', simplicial LL', or supernodal LL'.
// The Dx=b solve returns silently for the LL' factorizations (it is implicitly
// identity).

cholmod_dense *CHOLMOD(solve)   // returns solution X
(
    // input:
    int sys,            // system to solve
    cholmod_factor *L,  // factorization to use
    cholmod_dense *B,   // right-hand-side
    cholmod_common *Common
)
{

    cholmod_dense *Y = NULL, *X = NULL, *E = NULL ;

    // do the solve, allocating workspaces as needed
    int ok = CHOLMOD (solve2) (sys, L, B, NULL, &X, NULL, &Y, &E, Common) ;

    // free workspaces if allocated, and free result if an error occured
    CHOLMOD(free_dense) (&Y, Common) ;
    CHOLMOD(free_dense) (&E, Common) ;
    if (!ok)
    {
        CHOLMOD(free_dense) (&X, Common) ;
    }
    return (X) ;
}


//------------------------------------------------------------------------------
// cholmod_solve2
//------------------------------------------------------------------------------

// This function acts just like cholmod_solve, except that the solution X and
// the internal workspace (Y and E) can be passed in preallocated.  If the
// solution X or any required workspaces are not allocated on input, or if they
// are the wrong size or type, then this function frees them and reallocates
// them as the proper size and type.  Thus, if you have a sequence of solves to
// do, you can let this function allocate X, Y, and E on the first call.
// Subsequent calls to cholmod_solve2 can then reuse this space.  You must then
// free the workspaces Y and E (and X if desired) when you are finished.  For
// example, the first call to cholmod_l_solve2, below, will solve the requested
// system.  The next 2 calls (with different right-hand-sides but the same
// value of "sys") will resuse the workspace and solution X from the first
// call.  Finally, when all solves are done, you must free the workspaces Y and
// E (otherwise you will have a memory leak), and you should also free X when
// you are done with it.  Note that on input, X, Y, and E must be either valid
// cholmod_dense matrices, or initialized to NULL.  You cannot pass in an
// uninitialized X, Y, or E.
//
//      cholmod_dense *X = NULL, *Y = NULL, *E = NULL ;
//      ...
//      cholmod_l_solve2 (sys, L, B1, NULL, &X, NULL, &Y, &E, Common) ;
//      cholmod_l_solve2 (sys, L, B2, NULL, &X, NULL, &Y, &E, Common) ;
//      cholmod_l_solve2 (sys, L, B3, NULL, &X, NULL, &Y, &E, Common) ;
//      cholmod_l_free_dense (&X, Common) ;
//      cholmod_l_free_dense (&Y, Common) ;
//      cholmod_l_free_dense (&E, Common) ;
//
// The equivalent when using cholmod_l_solve is:
//
//      cholmod_dense *X = NULL, *Y = NULL, *E = NULL ;
//      ...
//      X = cholmod_l_solve (sys, L, B1, Common) ;
//      cholmod_l_free_dense (&X, Common) ;
//      X = cholmod_l_solve (sys, L, B2, Common) ;
//      cholmod_l_free_dense (&X, Common) ;
//      X = cholmod_l_solve (sys, L, B3, Common) ;
//      cholmod_l_free_dense (&X, Common) ;
//
// Both methods work fine, but in the 2nd method with cholmod_solve, the
// internal workspaces (Y and E) are allocated and freed on each call.
//
// Bset is an optional sparse column (pattern only) that specifies a set of row
// indices.  It is ignored if NULL, or if sys is CHOLMOD_P or CHOLMOD_Pt.  If
// it is present and not ignored, B must be a dense column vector, and only
// entries B(i) where i is in the pattern of Bset are considered.  All others
// are treated as if they were zero (they are not accessed).  L must be a
// simplicial factorization, not supernodal.  L is converted from supernodal to
// simplicial if necessary.  The solution X is defined only for entries in the
// output sparse pattern of Xset.  The xtype (real/complex/zomplex) of L and B
// must match.
//
// NOTE: If Bset is present and L is supernodal, it is converted to simplicial
// on output.

int CHOLMOD(solve2)         // returns TRUE on success, FALSE on failure
(
    // input:
    int sys,                        // system to solve
    cholmod_factor *L,              // factorization to use
    cholmod_dense *B,               // right-hand-side
    cholmod_sparse *Bset,
    // output:
    cholmod_dense **X_Handle,       // solution, allocated if need be
    cholmod_sparse **Xset_Handle,
    // workspace:
    cholmod_dense **Y_Handle,       // workspace, or NULL
    cholmod_dense **E_Handle,       // workspace, or NULL
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    cholmod_dense *Y = NULL, *X = NULL ;
    cholmod_sparse *C, *Yset, C_header, Yset_header, *Xset ;
    Int *Perm = NULL, *IPerm = NULL ;
    Int n, nrhs, ncols, k1, nr, ytype, k, blen, p, i, d, nrow ;
    Int Cp [2], Ysetp [2], *Ci, *Yseti, ysetlen ;

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_NULL (B, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (B, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
    if (sys < CHOLMOD_A || sys > CHOLMOD_Pt)
    {
        ERROR (CHOLMOD_INVALID, "invalid system") ;
        return (FALSE) ;
    }
    if (L->dtype != B->dtype)
    {
        ERROR (CHOLMOD_INVALID, "dtype of L and B must match") ;
        return (FALSE) ;
    }
    DEBUG (CHOLMOD(dump_factor) (L, "L", Common)) ;
    DEBUG (CHOLMOD(dump_dense) (B, "B", Common)) ;
    nrhs = B->ncol ;
    n = (Int) L->n ;
    d = (Int) B->d ;
    nrow = (Int) B->nrow ;
    if (d < n || nrow != n)
    {
        ERROR (CHOLMOD_INVALID, "dimensions of L and B do not match") ;
        return (FALSE) ;
    }

    if (sys == CHOLMOD_P || sys == CHOLMOD_Pt)
    {
        // Bset is ignored when solving Px=b or P'x=b
        Bset = NULL ;
    }

    if (Bset)
    {
        if (nrhs != 1)
        {
            ERROR (CHOLMOD_INVALID, "Bset requires a single right-hand side") ;
            return (FALSE) ;
        }
        if (L->xtype != B->xtype)
        {
            ERROR (CHOLMOD_INVALID, "Bset requires xtype of L and B to match") ;
            return (FALSE) ;
        }
        DEBUG (CHOLMOD(dump_sparse) (Bset, "Bset", Common)) ;
    }
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if ((sys == CHOLMOD_P || sys == CHOLMOD_Pt || sys == CHOLMOD_A)
            && L->ordering != CHOLMOD_NATURAL)
    {
        // otherwise, Perm is NULL, and the identity permutation is used
        Perm = L->Perm ;
    }

    //--------------------------------------------------------------------------
    // allocate the result X (or resuse the space from a prior call)
    //--------------------------------------------------------------------------

    int ctype = (Common->prefer_zomplex) ? CHOLMOD_ZOMPLEX : CHOLMOD_COMPLEX ;

    int X_xtype ;
    if (Bset)
    {
        X_xtype = L->xtype ;
    }
    else if (sys == CHOLMOD_P || sys == CHOLMOD_Pt)
    {
        // x=Pb and x=P'b return X real if B is real; X is the preferred
        // complex/zcomplex type if B is complex or zomplex
        X_xtype = (B->xtype == CHOLMOD_REAL) ? CHOLMOD_REAL : ctype ;
    }
    else if (L->xtype == CHOLMOD_REAL && B->xtype == CHOLMOD_REAL)
    {
        // X is real if both L and B are real
        X_xtype = CHOLMOD_REAL ;
    }
    else
    {
        // X is complex, use the preferred complex/zomplex type
        X_xtype = ctype ;
    }

    // ensure X has the right size and type
    X = CHOLMOD(ensure_dense) (X_Handle, n, nrhs, n, X_xtype + L->dtype,
        Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // solve using L, D, L', P, or some combination
    //--------------------------------------------------------------------------

    if (Bset)
    {

        //----------------------------------------------------------------------
        // solve for a subset of x, with a sparse b
        //----------------------------------------------------------------------

        #ifndef NSUPERNODAL
        // convert a supernodal L to simplicial when using Bset
        if (L->is_super)
        {
            // Can only use Bset on a simplicial factorization.  The supernodal
            // factor L is converted to simplicial, leaving the xtype unchanged
            // (real, complex, or zomplex).  Since the supernodal factorization
            // is already LL', it is left in that form.   This conversion uses
            // the super_num_to_simplicial_num function in
            // cholmod_change_factor.
            CHOLMOD(change_factor) (
                CHOLMOD_REAL,   // ignored, since L is already numeric
                TRUE,           // convert to LL' (no change to num. values)
                FALSE,          // convert to simplicial
                FALSE,          // do not pack the columns of L
                FALSE,          // (ignored)
                L, Common) ;
            if (Common->status < CHOLMOD_OK)
            {
                // out of memory, L is returned unchanged
                return (FALSE) ;
            }
        }
        #endif

        // L, X, and B are all the same xtype
        // ensure Y is the the right size
        Y = CHOLMOD(ensure_dense) (Y_Handle, 1, n, 1, L->xtype + L->dtype,
            Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }

        //----------------------------------------------------------------------
        // get the inverse permutation, constructing it if needed
        //----------------------------------------------------------------------

        DEBUG (CHOLMOD (dump_perm) (Perm,  n,n, "Perm",  Common)) ;

        if ((sys == CHOLMOD_A || sys == CHOLMOD_P) && Perm != NULL)
        {
            // The inverse permutation IPerm is used for the c=Pb step,
            // which is needed only for solving Ax=b or x=Pb.  No other
            // steps should use IPerm
            if (L->IPerm == NULL)
            {
                // construct the inverse permutation.  This is done only once
                // and then stored in L permanently.
                L->IPerm = CHOLMOD(malloc) (n, sizeof (Int), Common) ;
                if (Common->status < CHOLMOD_OK)
                {
                    // out of memory
                    return (FALSE) ;
                }
                IPerm = L->IPerm ;
                for (k = 0 ; k < n ; k++)
                {
                    IPerm [Perm [k]] = k ;
                }
            }
            // x=A\b and x=Pb both need IPerm
            IPerm = L->IPerm ;
        }

        DEBUG (CHOLMOD (dump_perm) (Perm,  n,n, "Perm",  Common)) ;
        DEBUG (CHOLMOD (dump_perm) (IPerm, n,n, "IPerm", Common)) ;

        //----------------------------------------------------------------------
        // ensure Xset is the right size and type
        //----------------------------------------------------------------------

        // Xset is n-by-1, nzmax >= n, pattern-only, packed, unsorted
        Xset = *Xset_Handle ;
        if (Xset == NULL || (Int) Xset->nrow != n || (Int) Xset->ncol != 1 ||
            (Int) Xset->nzmax < n || Xset->itype != CHOLMOD_PATTERN)
        {
            // this is done only once, for the 1st call to cholmod_solve
            CHOLMOD(free_sparse) (Xset_Handle, Common) ;
            Xset = CHOLMOD(allocate_sparse) (n, 1, n, FALSE, TRUE, 0,
                CHOLMOD_PATTERN + L->dtype, Common) ;
            (*Xset_Handle) = Xset ;
        }
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }
        Xset->sorted = FALSE ;
        Xset->stype = 0 ;

        //----------------------------------------------------------------------
        // ensure Flag of size n, and 3*n Int workspace is available
        //----------------------------------------------------------------------

        // does no work if prior calls already allocated enough space
        CHOLMOD(allocate_work) (n, 3*n, 0, Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }

        // [ use Iwork (n:3n-1) for Ci and Yseti
        Int *Iwork = Common->Iwork ;
        // Iwork (0:n-1) is not used because it is used by check_perm,
        // print_perm, check_sparse, and print_sparse
        Ci = Iwork + n ;
        Yseti = Ci + n ;

        // reallocating workspace would break Ci and Yseti
        int save_realloc_state = Common->no_workspace_reallocate ;
        Common->no_workspace_reallocate = TRUE ;

        //----------------------------------------------------------------------
        // C = permuted Bset, to correspond to the permutation of L
        //----------------------------------------------------------------------

        // C = IPerm (Bset)
        DEBUG (CHOLMOD(dump_sparse) (Bset, "Bset", Common)) ;

        Int *Bsetp = Bset->p ;
        Int *Bseti = Bset->i ;
        Int *Bsetnz = Bset->nz ;
        blen = (Bset->packed) ? Bsetp [1] : Bsetnz [0] ;

        // C = spones (P*B) or C = spones (B) if IPerm is NULL
        C = &C_header ;
        C->nrow = n ;
        C->ncol = 1 ;
        C->nzmax = n ;
        C->packed = TRUE ;
        C->stype = 0 ;
        C->itype = ITYPE ;
        C->xtype = CHOLMOD_PATTERN ;
        C->dtype = L->dtype ;
        C->nz = NULL ;
        C->p = Cp ;
        C->i = Ci ;
        C->x = NULL ;
        C->z = NULL ;
        C->sorted = FALSE ;
        Cp [0] = 0 ;
        Cp [1] = blen ;
        for (p = 0 ; p < blen ; p++)
        {
            Int iold = Bseti [p] ;
            Ci [p] = IPerm ? IPerm [iold] : iold ;
        }
        DEBUG (CHOLMOD (dump_sparse) (C, "C", Common)) ;

        // create a sparse column Yset from Iwork (n:2n-1)
        Yset = &Yset_header ;
        Yset->nrow = n ;
        Yset->ncol = 1 ;
        Yset->nzmax = n ;
        Yset->packed = TRUE ;
        Yset->stype = 0 ;
        Yset->itype = ITYPE ;
        Yset->xtype = CHOLMOD_PATTERN ;
        Yset->dtype = L->dtype ;
        Yset->nz = NULL ;
        Yset->p = Ysetp ;
        Yset->i = Yseti ;
        Yset->x = NULL ;
        Yset->z = NULL ;
        Yset->sorted = FALSE ;
        Ysetp [0] = 0 ;
        Ysetp [1] = 0 ;
        DEBUG (CHOLMOD (dump_sparse) (Yset, "Yset empty", Common)) ;

        //----------------------------------------------------------------------
        // Yset = nonzero pattern of L\C, or just C itself
        //----------------------------------------------------------------------

        // this takes O(ysetlen) time
        int ok = true ;
        if (sys == CHOLMOD_P || sys == CHOLMOD_Pt || sys == CHOLMOD_D)
        {
            Ysetp [1] = blen ;
            for (p = 0 ; p < blen ; p++)
            {
                Yseti [p] = Ci [p] ;
            }
        }
        else
        {
            ok = CHOLMOD(lsolve_pattern) (C, L, Yset, Common) ;
        }

        if (ok)
        {

            DEBUG (CHOLMOD (dump_sparse) (Yset, "Yset", Common)) ;

            //------------------------------------------------------------------
            // Y (C) = B (Bset)
            //------------------------------------------------------------------

            switch ((L->xtype + L->dtype) % 8)
            {
                case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                    rs_bset_perm (B, Bset, Yset, C, Y) ;
                    break ;

                case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                    cs_bset_perm (B, Bset, Yset, C, Y) ;
                    break ;

                case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                    zs_bset_perm (B, Bset, Yset, C, Y) ;
                    break ;

                case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                    rd_bset_perm (B, Bset, Yset, C, Y) ;
                    break ;

                case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                    cd_bset_perm (B, Bset, Yset, C, Y) ;
                    break ;

                case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                    zd_bset_perm (B, Bset, Yset, C, Y) ;
                    break ;
            }

            DEBUG (CHOLMOD (dump_dense) (Y, "Y (C) = B (Bset)", Common)) ;

            //------------------------------------------------------------------
            // solve Y = (L' \ (L \ Y'))', or other system, with template
            //------------------------------------------------------------------

            // the solve only iterates over columns in Yseti [0...ysetlen-1]

            if (! (sys == CHOLMOD_P || sys == CHOLMOD_Pt))
            {
                switch ((L->xtype + L->dtype) % 8)
                {
                    case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                        rs_simplicial_solver (sys, L, Y, Yset) ;
                        break ;

                    case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                        cs_simplicial_solver (sys, L, Y, Yset) ;
                        break ;

                    case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                        zs_simplicial_solver (sys, L, Y, Yset) ;
                        break ;

                    case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                        rd_simplicial_solver (sys, L, Y, Yset) ;
                        break ;

                    case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                        cd_simplicial_solver (sys, L, Y, Yset) ;
                        break ;

                    case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                        zd_simplicial_solver (sys, L, Y, Yset) ;
                        break ;
                }
            }

            DEBUG (CHOLMOD (dump_dense) (Y, "Y after solve", Common)) ;

            //------------------------------------------------------------------
            // X = P'*Y, but only for rows in Yset, and create Xset
            //------------------------------------------------------------------

            switch ((L->xtype + L->dtype) % 8)
            {
                case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                    rs_bset_iperm (Y, Yset, Perm, X, Xset) ;
                    break ;

                case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                    cs_bset_iperm (Y, Yset, Perm, X, Xset) ;
                    break ;

                case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                    zs_bset_iperm (Y, Yset, Perm, X, Xset) ;
                    break ;

                case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                    rd_bset_iperm (Y, Yset, Perm, X, Xset) ;
                    break ;

                case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                    cd_bset_iperm (Y, Yset, Perm, X, Xset) ;
                    break ;

                case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                    zd_bset_iperm (Y, Yset, Perm, X, Xset) ;
                    break ;
            }

            DEBUG (CHOLMOD(dump_sparse) (Xset, "Xset", Common)) ;
            DEBUG (CHOLMOD(dump_dense) (X, "X", Common)) ;
        }

        Common->no_workspace_reallocate = save_realloc_state ;
        // done using Iwork (n:3n-1) for Ci and Yseti ]

    }
    else if (sys == CHOLMOD_P)
    {

        //----------------------------------------------------------------------
        // x = P*b
        //----------------------------------------------------------------------

        if (L->dtype == CHOLMOD_DOUBLE)
        {
            d_perm (B, Perm, 0, nrhs, X) ;
        }
        else
        {
            s_perm (B, Perm, 0, nrhs, X) ;
        }

    }
    else if (sys == CHOLMOD_Pt)
    {

        //----------------------------------------------------------------------
        // x = P'*b
        //----------------------------------------------------------------------

        if (L->dtype == CHOLMOD_DOUBLE)
        {
            d_iperm (B, Perm, 0, nrhs, X) ;
        }
        else
        {
            s_iperm (B, Perm, 0, nrhs, X) ;
        }

    }
    else if (L->is_super)
    {

        //----------------------------------------------------------------------
        // solve using a supernodal LL' factorization
        //----------------------------------------------------------------------

        #ifndef NSUPERNODAL
        // allocate workspace
        cholmod_dense *E ;
        size_t dual ;
        Common->blas_ok = TRUE ;
        dual = (L->xtype == CHOLMOD_REAL && B->xtype != CHOLMOD_REAL) ? 2 : 1 ;
        Y = CHOLMOD(ensure_dense) (Y_Handle, n, dual*nrhs, n,
            L->xtype + L->dtype, Common) ;

        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }

        E = CHOLMOD(ensure_dense) (E_Handle, dual*nrhs, L->maxesize, dual*nrhs,
                L->xtype + L->dtype, Common) ;

        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }

        if (L->dtype == CHOLMOD_DOUBLE)
        {
            d_perm (B, Perm, 0, nrhs, Y) ;                  // Y = P*B
        }
        else
        {
            s_perm (B, Perm, 0, nrhs, Y) ;                  // Y = P*B
        }

        if (sys == CHOLMOD_A || sys == CHOLMOD_LDLt)
        {
            CHOLMOD(super_lsolve) (L, Y, E, Common) ;       // Y = L\Y
            CHOLMOD(super_ltsolve) (L, Y, E, Common) ;      // Y = L'\Y
        }
        else if (sys == CHOLMOD_L || sys == CHOLMOD_LD)
        {
            CHOLMOD(super_lsolve) (L, Y, E, Common) ;       // Y = L\Y
        }
        else if (sys == CHOLMOD_Lt || sys == CHOLMOD_DLt)
        {
            CHOLMOD(super_ltsolve) (L, Y, E, Common) ;      // Y = L'\Y
        }

        if (L->dtype == CHOLMOD_DOUBLE)
        {
            d_iperm (Y, Perm, 0, nrhs, X) ;                 // X = P'*Y
        }
        else
        {
            s_iperm (Y, Perm, 0, nrhs, X) ;                 // X = P'*Y
        }

        #else
        // CHOLMOD Supernodal module not installed
        ERROR (CHOLMOD_NOT_INSTALLED,"Supernodal module not installed") ;
        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // solve using a simplicial LL' or LDL' factorization
        //----------------------------------------------------------------------

        if (L->xtype == CHOLMOD_REAL && B->xtype == CHOLMOD_REAL)
        {
            // L, B, and Y are all real
            // solve with up to 4 columns of B at a time
            ncols = 4 ;
            nr = MAX (4, nrhs) ;
            ytype = CHOLMOD_REAL ;
        }
        else if (L->xtype == CHOLMOD_REAL)
        {
            // L is real and B is complex or zomplex
            // solve with one column of B (real/imag), at a time
            ncols = 1 ;
            nr = 2 ;
            ytype = CHOLMOD_REAL ;
        }
        else
        {
            // L is complex or zomplex, B is real/complex/zomplex, Y has the
            // same complexity as L.  Solve with one column of B at a time.
            ncols = 1 ;
            nr = 1 ;
            ytype = L->xtype ;
        }

        Y = CHOLMOD(ensure_dense) (Y_Handle, nr, n, nr, ytype + L->dtype,
            Common) ;
        if (Common->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }

        for (k1 = 0 ; k1 < nrhs ; k1 += ncols)
        {

            //------------------------------------------------------------------
            // Y = B (P, k1:k1+ncols-1)' = (P * B (:,...))'
            //------------------------------------------------------------------

            if (L->dtype == CHOLMOD_DOUBLE)
            {
                d_ptrans (B, Perm, k1, ncols, Y) ;
            }
            else
            {
                s_ptrans (B, Perm, k1, ncols, Y) ;
            }

            //------------------------------------------------------------------
            // solve Y = (L' \ (L \ Y'))', or other system, with template
            //------------------------------------------------------------------

            switch ((L->xtype + L->dtype) % 8)
            {

                case CHOLMOD_SINGLE + CHOLMOD_REAL:
                    rs_simplicial_solver (sys, L, Y, NULL) ;
                    break ;

                case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
                    cs_simplicial_solver (sys, L, Y, NULL) ;
                    break ;

                case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
                    zs_simplicial_solver (sys, L, Y, NULL) ;
                    break ;

                case CHOLMOD_DOUBLE + CHOLMOD_REAL:
                    rd_simplicial_solver (sys, L, Y, NULL) ;
                    break ;

                case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
                    cd_simplicial_solver (sys, L, Y, NULL) ;
                    break ;

                case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
                    zd_simplicial_solver (sys, L, Y, NULL) ;
                    break ;
            }

            //------------------------------------------------------------------
            // X (P, k1:k2+ncols-1) = Y'
            //------------------------------------------------------------------

            if (L->dtype == CHOLMOD_DOUBLE)
            {
                d_iptrans (Y, Perm, k1, ncols, X) ;
            }
            else
            {
                s_iptrans (Y, Perm, k1, ncols, X) ;
            }
        }
    }

    DEBUG (CHOLMOD(dump_dense) (X, "X result", Common)) ;
    return (Common->status == CHOLMOD_OK) ;
}
#endif

