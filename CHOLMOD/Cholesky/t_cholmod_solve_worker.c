//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/t_cholmod_solve: template for cholmod_solve
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Template routine for cholmod_solve.  Supports any numeric xtype (real,
// complex, or zomplex).  The xtypes of all matrices (L and Y) must match.

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// simplicial template solvers
//------------------------------------------------------------------------------

// LL': solve Lx=b with non-unit diagonal
#define LL
#include "t_cholmod_lsolve_template.c"

// LDL': solve LDx=b
#define LD
#include "t_cholmod_lsolve_template.c"

// LDL': solve Lx=b with unit diagonal
#include "t_cholmod_lsolve_template.c"

// LL': solve L'x=b with non-unit diagonal
#define LL
#include "t_cholmod_ltsolve_template.c"

// LDL': solve DL'x=b
#define LD
#include "t_cholmod_ltsolve_template.c"

// LDL': solve L'x=b with unit diagonal
#include "t_cholmod_ltsolve_template.c"

//------------------------------------------------------------------------------
// t_ldl_dsolve
//------------------------------------------------------------------------------

// Solve Dx=b for an LDL' factorization, where Y holds b' on input and x' on
// output.
//
// The number of right-hand-sides (nrhs) is not restricted, even if Yset
// is present.

static void TEMPLATE (ldl_dsolve)
(
    cholmod_factor *L,
    cholmod_dense *Y,           // nr-by-n with leading dimension nr
    cholmod_sparse *Yset        // input pattern Yset
)
{

    ASSERT (L->xtype == Y->xtype) ; // L and Y must have the same xtype
    ASSERT (L->dtype == Y->dtype) ; // L and Y must have the same dtype
    ASSERT (L->n == Y->ncol) ;      // dimensions must match
    ASSERT (Y->nrow == Y->d) ;      // leading dimension of Y = # rows of Y
    ASSERT (L->xtype != CHOLMOD_PATTERN) ;      // L is not symbolic
    ASSERT (!(L->is_super) && !(L->is_ll)) ;    // L is simplicial LDL'

    Real d [1] ;
    Int nrhs = Y->nrow ;
    Int n = L->n ;
    Int *Lp = L->p ;
    Real *Lx = L->x ;
    Real *Yx = Y->x ;
    Real *Yz = Y->z ;

    if (Yset)
    {
        Int *Yseti = Yset->i ;
        Int *Ysetp = Yset->p ;
        Int ysetlen = Ysetp [1] ;
        for (Int kk = 0 ; kk < ysetlen ; kk++)
        {
            Int k = Yseti [kk] ;
            Int k1 = k*nrhs ;
            Int k2 = (k+1)*nrhs ;
            // d = L (k,k)
            ASSIGN_REAL (d,0, Lx,Lp[k]) ;
            for (Int p = k1 ; p < k2 ; p++)
            {
                // Y (p) = Y (p) / d
                DIV_REAL (Yx,Yz,p, Yx,Yz,p, d,0) ;
            }
        }
    }
    else
    {
        for (Int k = 0 ; k < n ; k++)
        {
            Int k1 = k*nrhs ;
            Int k2 = (k+1)*nrhs ;
            // d = L (k,k)
            ASSIGN_REAL (d,0, Lx,Lp[k]) ;
            for (Int p = k1 ; p < k2 ; p++)
            {
                // Y (p) = Y (p) / d
                DIV_REAL (Yx,Yz,p, Yx,Yz,p, d,0) ;
            }
        }
    }
}

//------------------------------------------------------------------------------
// t_simplicial_solver
//------------------------------------------------------------------------------

// Solve a linear system, where Y' contains the (array-transposed) right-hand
// side on input, and the solution on output.  No permutations are applied;
// these must have already been applied to Y on input.
//
// Yset is an optional list of indices from cholmod_lsolve_pattern.  The solve
// is performed only on the columns of L corresponding to entries in Yset.
// Ignored if NULL.  If present, most functions require that Y' consist of a
// single dense column.

static void TEMPLATE (simplicial_solver)
(
    int sys,                // system to solve
    cholmod_factor *L,      // factor to use, a simplicial LL' or LDL'
    cholmod_dense *Y,       // right-hand-side on input, solution on output
    cholmod_sparse *Yset    // input pattern Yset
)
{
    if (L->is_ll)
    {
        // The factorization is LL'
        if (sys == CHOLMOD_A || sys == CHOLMOD_LDLt)
        {
            // Solve Ax=b or LL'x=b
            TEMPLATE (ll_lsolve_k) (L, Y, Yset) ;
            TEMPLATE (ll_ltsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_L || sys == CHOLMOD_LD)
        {
            // Solve Lx=b
            TEMPLATE (ll_lsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_Lt || sys == CHOLMOD_DLt)
        {
            // Solve L'x=b
            TEMPLATE (ll_ltsolve_k) (L, Y, Yset) ;
        }
    }
    else
    {
        // The factorization is LDL'
        if (sys == CHOLMOD_A || sys == CHOLMOD_LDLt)
        {
            // Solve Ax=b or LDL'x=b
            TEMPLATE (ldl_lsolve_k) (L, Y, Yset) ;
            TEMPLATE (ldl_dltsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_LD)
        {
            // Solve LDx=b
            TEMPLATE (ldl_ldsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_L)
        {
            // Solve Lx=b
            TEMPLATE (ldl_lsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_Lt)
        {
            // Solve L'x=b
            TEMPLATE (ldl_ltsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_DLt)
        {
            // Solve DL'x=b
            TEMPLATE (ldl_dltsolve_k) (L, Y, Yset) ;
        }
        else if (sys == CHOLMOD_D)
        {
            // Solve Dx=b
            TEMPLATE (ldl_dsolve) (L, Y, Yset) ;
        }
    }
}

//------------------------------------------------------------------------------
// bset_perm
//------------------------------------------------------------------------------

// Y (C) = B (Bset)

static void TEMPLATE (bset_perm)
(
    // input:
    cholmod_dense *B,       // input matrix B
    cholmod_sparse *Bset,   // input pattern Bset
    cholmod_sparse *Yset,   // input pattern Yset
    cholmod_sparse *C,      // input pattern C
    // input/output:
    cholmod_dense *Y        // output matrix Y, already allocated
)
{

    //--------------------------------------------------------------------------
    // clear the parts of Y that we will use in the solve
    //--------------------------------------------------------------------------

    Real *Yx = Y->x ;
    Real *Yz = Y->z ;
    Int *Ysetp = Yset->p ;
    Int *Yseti = Yset->i ;
    Int ysetlen = Ysetp [1] ;

    for (Int p = 0 ; p < ysetlen ; p++)
    {
        Int i = Yseti [p] ;
        // Y (i) = 0
        CLEAR (Yx, Yz, i) ;
    }

    //--------------------------------------------------------------------------
    // scatter and permute B into Y
    //--------------------------------------------------------------------------

    // Y (C) = B (Bset)
    Real *Bx = B->x ;
    Real *Bz = B->z ;
    Int *Bsetp = Bset->p ;
    Int *Bseti = Bset->i ;
    Int *Bsetnz = Bset->nz ;
    Int blen = (Bset->packed) ? Bsetp [1] : Bsetnz [0] ;
    Int *Ci = C->i ;

    for (Int p = 0 ; p < blen ; p++)
    {
        Int iold = Bseti [p] ;
        Int inew = Ci [p] ;
        // Y (inew) = B (iold)
        ASSIGN (Yx, Yz, inew, Bx, Bz, iold) ;
    }
}

//------------------------------------------------------------------------------
// bset_iperm
//------------------------------------------------------------------------------

// X (P (Yset)) = Y (Yset)

static void TEMPLATE (bset_iperm)
(
    // input:
    cholmod_dense *Y,       // input matrix Y
    cholmod_sparse *Yset,   // input pattern Yset
    Int *Perm,              // optional input permutation (can be NULL)
    // input/output:
    cholmod_dense *X,       // output matrix X, already allocated
    cholmod_sparse *Xset    // output pattern Xset
)
{

    Real *Xx = X->x ;
    Real *Xz = X->z ;
    Int *Xseti = Xset->i ;
    Int *Xsetp = Xset->p ;

    Real *Yx = Y->x ;
    Real *Yz = Y->z ;

    Int *Ysetp = Yset->p ;
    Int *Yseti = Yset->i ;
    Int ysetlen = Ysetp [1] ;

    for (Int p = 0 ; p < ysetlen ; p++)
    {
        Int inew = Yseti [p] ;
        Int iold = P(inew) ;        // was: (Perm ? Perm [inew] : inew) ;
        // X (iold) = Y (inew)
        ASSIGN (Xx, Xz, iold, Yx, Yz, inew) ;
        Xseti [p] = iold ;
    }

    Xsetp [0] = 0 ;
    Xsetp [1] = ysetlen ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

