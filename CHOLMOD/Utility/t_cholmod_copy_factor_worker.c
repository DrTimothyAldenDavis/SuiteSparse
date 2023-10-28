//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy_factor_worker: copy a factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Columns are copied individually, to avoid copying uninitialized space

#include "cholmod_template.h"

static void TEMPLATE (cholmod_copy_factor_worker)
(
    cholmod_factor *L,      // input factor to copy (not modified)
    cholmod_factor *H       // output factor
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    size_t n = L->n ;

    Int  *Lp  = (Int  *) L->p ;
    Int  *Li  = (Int  *) L->i ;
    Int  *Lnz = (Int  *) L->nz ;
    Real *Lx  = (Real *) L->x ;
    Real *Lz  = (Real *) L->z ;

    Int  *Hi  = (Int  *) H->i ;
    Real *Hx  = (Real *) H->x ;
    Real *Hz  = (Real *) H->z ;

    //--------------------------------------------------------------------------
    // copy each column
    //--------------------------------------------------------------------------

    for (Int j = 0 ; j < n ; j++)
    {
        Int p = Lp [j] ;
        Int pend = p + Lnz [j] ;
        for ( ; p < pend ; p++)
        {
            Hi [p] = Li [p] ;
            ASSIGN (Hx, Hz, p, Lx, Lz, p) ;
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

