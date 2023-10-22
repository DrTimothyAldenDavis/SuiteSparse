//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_factor_1_worker: change factor to identity
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// t_cholmod_change_factor_1_worker:  set L to the identity matrix
//------------------------------------------------------------------------------

// L is simplicial numeric.

static void TEMPLATE (cholmod_change_factor_1_worker)
(
    cholmod_factor *L
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L->xtype != CHOLMOD_PATTERN) ;
    ASSERT (!L->is_super) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int  *Lp = (Int  *) L->p ;
    Int  *Li = (Int  *) L->i ;
    Real *Lx = (Real *) L->x ;
    Real *Lz = (Real *) L->z ;
    Int n = L->n ;

    //--------------------------------------------------------------------------
    // set L to the identity matrix
    //--------------------------------------------------------------------------

    Real onex [2] = {1,0} ;
    Real onez [1] = {0} ;

    for (Int j = 0 ; j < n ; j++)
    {
        Int p = Lp [j] ;
        ASSERT (p < Lp [j+1]) ;
        Li [p] = j ;
        ASSIGN (Lx, Lz, p, onex, onez, 0) ;
    }
}

