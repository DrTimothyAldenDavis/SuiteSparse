//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_reallocate_column_worker
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_reallocate_column_worker)
(
    cholmod_factor *L,          // L factor modified, L(:,j) resized
    Int j,                      // column L(:,j) to move
    Int pdest,                  // place to move it to
    Int psrc                    // place to move it from
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int  *Lnz = (Int  *) L->nz ;
    Int  *Li  = (Int  *) L->i ;
    Real *Lx  = (Real *) L->x ;
    Real *Lz  = (Real *) L->z ;
    Int len = Lnz [j] ;

    //--------------------------------------------------------------------------
    // move L(:,j) to its new position
    //--------------------------------------------------------------------------

    for (Int k = 0 ; k < len ; k++, pdest++, psrc++)
    {
        // move L(i,j) from position psrc to position pdest
        Li [pdest] = Li [psrc] ;
        ASSIGN (Lx, Lz, pdest, Lx, Lz, psrc) ;
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

