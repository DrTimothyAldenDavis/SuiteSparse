// =============================================================================
// === spqr_happly_work ========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Determines the workspace workspace needed by spqr-happly

#include "spqr.hpp"

template <typename Int> int spqr_happly_work
(
    // input
    int method,     // 0,1,2,3 

    Int m,         // X is m-by-n
    Int n,

    // FUTURE : make H cholmod_sparse:
    Int nh,        // number of Householder vectors
    Int *Hp,       // size nh+1, column pointers for H
    Int hchunk,

    // outputs; sizes of workspaces needed
    Int *p_vmax, 
    Int *p_vsize, 
    Int *p_csize
)
{
    Int maxhlen, h, hlen, vmax, mh, vsize, csize, vsize1, vsize2 ;
    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    *p_vmax = 0 ;
    *p_vsize = 0 ;
    *p_csize = 0 ;

    if (m == 0 || n == 0 || nh == 0)
    {
        // nothing to do
        return (TRUE) ;
    }

    // -------------------------------------------------------------------------
    // determine the length of the longest Householder vector
    // -------------------------------------------------------------------------

    maxhlen = 1 ;
    for (h = 0 ; h < nh ; h++)
    {
        hlen = Hp [h+1] - Hp [h] ;
        maxhlen = MAX (maxhlen, hlen) ;
    }

    // number of rows of H
    mh = (method == 0 || method == 1) ? m : n ;

    // -------------------------------------------------------------------------
    // determine workspace sizes
    // -------------------------------------------------------------------------

    // Int overflow cannot occur with vmax since H is already allocated
    if (method == 0 || method == 3)
    {
        // apply H in the forward direction; H(0) first, H(nh-1) last
        vmax = 2 * maxhlen + 8 ;
    }
    else
    {
        // apply H in the backward direction; H(nh-1) first, H(0) last
        vmax = maxhlen + hchunk ;
    }

    vmax = MIN (vmax, mh) ;
    vmax = MAX (vmax, 2) ;

    // csize = vmax * ((method <= 1) ? n : m) ;
    csize = spqr_mult (vmax, (method <= 1) ? n : m, &ok) ;

    // vsize = (hchunk*hchunk + ((method <= 1) ? n : m)*hchunk + vmax*hchunk) ;
    vsize  = spqr_mult (hchunk, hchunk, &ok) ;
    vsize1 = spqr_mult ((method <= 1) ? n : m, hchunk, &ok) ;
    vsize2 = spqr_mult (vmax, hchunk, &ok) ;
    vsize = spqr_add (vsize, vsize1, &ok) ;
    vsize = spqr_add (vsize, vsize2, &ok) ;

    // -------------------------------------------------------------------------
    // return workspace sizes
    // -------------------------------------------------------------------------

    *p_vmax = vmax ;
    *p_vsize = vsize ;
    *p_csize = csize ;
    return (ok) ;
}

template int spqr_happly_work <int32_t>
(
    // input
    int method,     // 0,1,2,3 

    int32_t m,         // X is m-by-n
    int32_t n,

    // FUTURE : make H cholmod_sparse:
    int32_t nh,        // number of Householder vectors
    int32_t *Hp,       // size nh+1, column pointers for H
    int32_t hchunk,

    // outputs; sizes of workspaces needed
    int32_t *p_vmax, 
    int32_t *p_vsize, 
    int32_t *p_csize
) ;
template int spqr_happly_work <int64_t>
(
    // input
    int method,     // 0,1,2,3 

    int64_t m,         // X is m-by-n
    int64_t n,

    // FUTURE : make H cholmod_sparse:
    int64_t nh,        // number of Householder vectors
    int64_t *Hp,       // size nh+1, column pointers for H
    int64_t hchunk,

    // outputs; sizes of workspaces needed
    int64_t *p_vmax, 
    int64_t *p_vsize, 
    int64_t *p_csize
) ;
