//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/t_cholmod_rcond_worker: estimate rcond of a factorization
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Return a rough estimate of the reciprocal of the condition number.

//------------------------------------------------------------------------------
// LMINMAX macros
//------------------------------------------------------------------------------

// Update lmin and lmax for one entry L(j,j)

#define FIRST_LMINMAX(Ljj,lmin,lmax)    \
{                                       \
    Real ljj = Ljj ;                    \
    if (isnan (ljj))                    \
    {                                   \
        return (0) ;                    \
    }                                   \
    lmin = ljj ;                        \
    lmax = ljj ;                        \
}

#define LMINMAX(Ljj,lmin,lmax)          \
{                                       \
    Real ljj = Ljj ;                    \
    if (isnan (ljj))                    \
    {                                   \
        return (0) ;                    \
    }                                   \
    if (ljj < lmin)                     \
    {                                   \
        lmin = ljj ;                    \
    }                                   \
    else if (ljj > lmax)                \
    {                                   \
        lmax = ljj ;                    \
    }                                   \
}

//------------------------------------------------------------------------------
// t_cholmod_rcond_worker
//------------------------------------------------------------------------------

#include "cholmod_template.h"

static double TEMPLATE (cholmod_rcond_worker)
(
    cholmod_factor *L
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real lmin = 0 ;
    Real lmax = 0 ;
    Real *Lx = L->x ;
    Int n = L->n ;
    Int e = (L->xtype == CHOLMOD_COMPLEX) ? 2 : 1 ;

    //--------------------------------------------------------------------------
    // compute the approximate rcond of L
    //--------------------------------------------------------------------------

    if (L->is_super)
    {

        //----------------------------------------------------------------------
        // L is supernodal
        //----------------------------------------------------------------------

        Int nsuper = L->nsuper ;    // number of supernodes in L
        Int *Lpi = L->pi ;          // column pointers for integer pattern
        Int *Lpx = L->px ;          // column pointers for numeric values
        Int *Super = L->super ;     // supernode sizes
        FIRST_LMINMAX (Lx [0], lmin, lmax) ;    // first diagonal entry of L
        for (Int s = 0 ; s < nsuper ; s++)
        {
            Int k1 = Super [s] ;    // first column in supernode s
            Int k2 = Super [s+1] ;  // last column in supernode is k2-1
            Int psi = Lpi [s] ;     // first row index is L->s [psi]
            Int psend = Lpi [s+1] ; // last row index is L->s [psend-1]
            Int psx = Lpx [s] ;     // first numeric entry is Lx [psx]
            Int nsrow = psend - psi ;   // supernode is nsrow-by-nscol
            Int nscol = k2 - k1 ;
            for (Int jj = 0 ; jj < nscol ; jj++)
            {
                LMINMAX (Lx [e * (psx + jj + jj*nsrow)], lmin, lmax) ;
            }
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // L is simplicial
        //----------------------------------------------------------------------

        Int  *Lp = L->p ;
        if (L->is_ll)
        {
            // LL' factorization
            FIRST_LMINMAX (Lx [Lp [0]], lmin, lmax) ;
            for (Int j = 1 ; j < n ; j++)
            {
                LMINMAX (Lx [e * Lp [j]], lmin, lmax) ;
            }
        }
        else
        {
            // LDL' factorization, the diagonal might be negative
            FIRST_LMINMAX (fabs (Lx [Lp [0]]), lmin, lmax) ;
            for (Int j = 1 ; j < n ; j++)
            {
                LMINMAX (fabs (Lx [e * Lp [j]]), lmin, lmax) ;
            }
        }
    }

    double rcond = ((double) lmin) / ((double) lmax) ;
    if (L->is_ll)
    {
        rcond = rcond*rcond ;
    }
    return (rcond) ;
}

#undef LMINMAX
#undef FIRST_LMINMAX

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

