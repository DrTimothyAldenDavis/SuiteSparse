// =============================================================================
// === spqr_fcsize =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "spqr.hpp"

template <typename Int> Int spqr_fcsize    // returns # of entries in C of current front F
(
    // input, not modified
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int rank               // the C block starts at F (rank,npiv)
)
{
    Int cm, cn, csize ;
    ASSERT (m >= 0 && n >= 0 && npiv >= 0 && npiv <= n) ;
    ASSERT (rank >= 0 && rank <= MIN (m,npiv)) ;
    cn = n - npiv ;                         // number of columns of C
    cm = MIN (m-rank, cn) ;                 // number of rows of C
    ASSERT (cm <= cn) ;
    // Note that this is safe from int64_t overflow:
    csize = (cm * (cm+1)) / 2 + cm * (cn - cm) ;
    return (csize) ;                        // return # of entries in C
}