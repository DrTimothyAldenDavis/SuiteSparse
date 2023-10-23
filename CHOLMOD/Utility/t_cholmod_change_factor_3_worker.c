//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_factor_3_worker: change format of a factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Converts a supernodal numeric L into a simplicial numeric L.  The factor is
// modified in place in Lx.

// If to_packed is false, entries in Lx stay in the same position they had in
// the supernode, and Li is created to hold the pattern of all those entries.
// Gaps will naturally appear because of the unused space in each supernode.
//
// Consider the following example:
//
// Each supernode is lower trapezoidal, with a top part that is lower
// triangular (with diagonal present) and a bottom part that is rectangular.
// In this example below, nscol = 5, so the supernode represents 5 columns of
// L, and nsrow = 8, which means that the first column of the supernode has 8
// entries, including the diagonal.
//
//      x 1 2 3 4
//      x x 2 3 4
//      x x x 3 4
//      x x x x 4
//      x x x x x
//      x x x x x
//      x x x x x
//      x x x x x
//
// If the entries in the supernode are not moved, the first column will have a
// single entry of slack space at the end of the space (the "1" above).  The
// 2nd column has 2 entries of slack space ("2").  The 3rd column has 3 entries
// of slack space ("3"), and so on.  The last column (5th) has no slack space
// since it is followed by the next supenode.

// If to_packed is true, the space is compressed and the unused entries above
// the diagonal of each supernode are removed.  That is, in this example, the
// supernode takes up 40 entries in Lx in supernodal form.  In the simplicial
// form, there is no slack space, and the total space becomes 8 + 7 + 6 + 5 + 4
// = 30.  The first column starts at position Lpx [s] in Lx for supernode s,
// and is moved to the 'left', starting at position Lp [j] if the first column
// is j = L->super [s].  Since space is being compacted, Lp [j] <= Lpx [s]
// holds, and thus Lx can be compacted in place when to_packed is true.
//
//      x
//      x x
//      x x x
//      x x x x
//      x x x x x
//      x x x x x
//      x x x x x
//      x x x x x

// The factor is either real or complex (not pattern nor complex).  Thus,
// L->x is always non-NULL, and L->z is NULL and not used.  Instead of Lz,
// a blank appears in the macros that access the values of L below.

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// t_cholmod_change_factor_3_worker: convert supernodal numeric L to simplicial
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_change_factor_3_worker)
(
    cholmod_factor *L,  // factor to modify
    int to_packed,      // if true: convert L to packed
    int to_ll,          // if true, convert to LL. if false: to LDL'
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L->is_super) ;
    ASSERT (L->xtype == CHOLMOD_REAL || L->xtype == CHOLMOD_COMPLEX) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = L->n ;                  // L is n-by-n
    Int nsuper = L->nsuper ;        // # of supernodes
    Int *Lpi   = (Int *) L->pi ;    // index into L->s for supernode pattern
    Int *Lpx   = (Int *) L->px ;    // index into L->x for supernode values
    Int *Ls    = (Int *) L->s ;     // pattern of supernodes
    Int *Super = (Int *) L->super ; // 1st column in each supernode

    // Lx holds supernode values on input; simplicial values on output:
    Real *Lx = (Real *) L->x ;     // numerical values (real or complex)

    // the simplicial space is allocated but not initialized on input:
    Int  *Lp  = (Int  *) L->p ;     // simplicial col pointers
    Int  *Li  = (Int  *) L->i ;     // simplicial row indices
    Int  *Lnz = (Int  *) L->nz ;    // simplicial column counts
    Int lnz = L->nzmax ;            // size of Li, Lp, and Lx

    //----------------------------------------------------------------------
    // convert supernodal LL' to simplicial LL' or LDL' (packed/unpacked)
    //----------------------------------------------------------------------

    if (to_packed)
    {
        if (to_ll)
        {
            // convert to simplicial packed LL'
            #define TO_PACKED true
            #define TO_LL     true
            #include "t_cholmod_change_factor_3_template.c"
        }
        else
        {
            // convert to simplicial packed LDL'
            #define TO_PACKED true
            #define TO_LL     false
            #include "t_cholmod_change_factor_3_template.c"
        }
    }
    else
    {
        if (to_ll)
        {
            // convert to simplicial unpacked LL'
            #define TO_PACKED false
            #define TO_LL     true
            #include "t_cholmod_change_factor_3_template.c"
        }
        else
        {
            // convert to simplicial unpacked LDL'
            #define TO_PACKED false
            #define TO_LL     false
            #include "t_cholmod_change_factor_3_template.c"
        }
    }
}

