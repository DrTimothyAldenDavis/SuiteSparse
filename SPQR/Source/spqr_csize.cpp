// =============================================================================
// === spqr_csize ==============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "spqr.hpp"

int64_t spqr_csize     // returns # of entries in C of a child
(
    // input, not modified
    int64_t c,                 // child c
    int64_t *Rp,               // size nf+1, pointers for pattern of R
    int64_t *Cm,               // size nf, Cm [c] = # of rows in child C
    int64_t *Super             // size nf, pivotal columns in each front
)
{
    int64_t pc, cm, fnc, fpc, cn, csize ;

    pc = Rp [c] ;                   // get the pattern of child R
    cm = Cm [c] ;                   // # of rows in child C
    fnc = Rp [c+1] - pc ;           // total # cols in child F
    fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
    cn = fnc - fpc ;                // # of cols in child C
    ASSERT (cm >= 0 && cm <= cn) ;
    ASSERT (pc + cm <= Rp [c+1]) ;
    // Note that this is safe from int64_t overflow
    csize = (cm * (cm+1)) / 2 + cm * (cn - cm) ;
    return (csize) ;
}

