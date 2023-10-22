//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_factor_2_worker: change format of a factor
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// t_cholmod_change_factor_2_worker: changes a simplicial numeric factor
//------------------------------------------------------------------------------

// The contents of the old L->(i,x,z) is copied/converted into Li2, Lx2, and
// Lz2.

static void TEMPLATE (cholmod_change_factor_2_worker)
(
    cholmod_factor *L,  // factor to modify
    int to_packed,      // if true: convert L to packed
    Int *Li2,           // new space for L->i (if out_of_place is true)
    Real *Lx2,          // new space for L->x (if out_of_place is true)
    Real *Lz2,          // new space for L->z (if out_of_place is true)
    Int lnz,            // max # of entries that Li2, Lx2, Lz2 can hold
    bool grow,          // if true: add slack space to the new columns of L
    double grow1,       // growth factor for each column
    double grow2,       // growth factor for each column
    bool make_ll,       // if true: convert LDL' to LL'
    bool out_of_place,  // if true: convert L out-of-place using Li2, Lx2, Lz2
    bool make_ldl,      // if true: convert LL' to LDL'
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!L->is_super) ;
    ASSERT (L->xtype != CHOLMOD_PATTERN) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = L->n ;
    Int  *Lp  = (Int  *) L->p ;
    Int  *Li  = (Int  *) L->i ;
    Real *Lx  = (Real *) L->x ;
    Real *Lz  = (Real *) L->z ;
    Int  *Lnz = (Int  *) L->nz ;

    //--------------------------------------------------------------------------
    // initialize L->minor; will be set below to the min j where D(j,j) <= 0
    //--------------------------------------------------------------------------

    if (make_ll)
    {
        L->minor = n ;
    }

    //--------------------------------------------------------------------------
    // convert the simplicial numeric L
    //--------------------------------------------------------------------------

    if (out_of_place)
    {

        //----------------------------------------------------------------------
        // L must be converted out of place
        //----------------------------------------------------------------------

        // L is converted out-of-place, into the newly allocate space Li2, Lx2,
        // and Lz2.  This occurs if L is to be packed and/or made monotonic,
        // but L is not already monotonic.

        ASSERT (Li2 != NULL) ;
        ASSERT (Lx2 != NULL) ;
        ASSERT (IMPLIES (L->xtype == CHOLMOD_ZOMPLEX, Lz2 != NULL)) ;

        #define OUT_OF_PLACE
        #include "t_cholmod_change_factor_2_template.c"

    }
    else if (to_packed)
    {

        //----------------------------------------------------------------------
        // pack L, removing all slack space, in existing Li, Lx, and Lz space
        //----------------------------------------------------------------------

        ASSERT (Li2 == NULL) ;
        ASSERT (Lx2 == NULL) ;
        ASSERT (Lz2 == NULL) ;

        #define TO_PACKED
        #include "t_cholmod_change_factor_2_template.c"

    }
    else
    {

        //----------------------------------------------------------------------
        // in-place conversion of L: no entries are moved
        //----------------------------------------------------------------------

        ASSERT (Li2 == NULL) ;
        ASSERT (Lx2 == NULL) ;
        ASSERT (Lz2 == NULL) ;

        #define IN_PLACE
        #include "t_cholmod_change_factor_2_template.c"
    }
}

