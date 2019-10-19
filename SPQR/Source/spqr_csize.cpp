// =============================================================================
// === spqr_csize ==============================================================
// =============================================================================

#include "spqr.hpp"

Int spqr_csize     // returns # of entries in C of a child
(
    // input, not modified
    Int c,                  // child c
    Int *Rp,                // size nf+1, pointers for pattern of R
    Int *Cm,                // size nf, Cm [c] = # of rows in child C
    Int *Super              // size nf, pivotal columns in each front
)
{
    Int pc, cm, fnc, fpc, cn, csize ;

    pc = Rp [c] ;                   // get the pattern of child R
    cm = Cm [c] ;                   // # of rows in child C
    fnc = Rp [c+1] - pc ;           // total # cols in child F
    fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
    cn = fnc - fpc ;                // # of cols in child C
    ASSERT (cm >= 0 && cm <= cn) ;
    ASSERT (pc + cm <= Rp [c+1]) ;
    // Note that this is safe from Int overflow
    csize = (cm * (cm+1)) / 2 + cm * (cn - cm) ;
    return (csize) ;
}

