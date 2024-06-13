//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_leaf: Subroutine for column counts of Cholesky
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_cholesky_internal.h"

/* Purpose: consider A(i,j), node j in ith row subtree and return lca(jprev,j)
   Used to determine Column counts of Cholesky or LDL factor */

SPEX_info spex_symmetric_leaf
(
    int64_t *lca_handle,    // Least common ancestor (jprev,j)
    const int64_t i,        // Index (subtree i)
    const int64_t j,        // Index (node j)
    const int64_t *first,   // first[j] is the first descendant of node j
    int64_t *maxfirst,      // maxfirst[j] is the maximum first descendant of
                            // node j
    int64_t *prevleaf,      // prevleaf[i] is the previous leaf of ith subtree
    int64_t *ancestor,      // ancestor[i] is the ancestor of ith subtree
    int64_t *jleaf          // indicates whether j is the first leaf (value of
                            // 1) or not (value of 2)
)
{

    *jleaf = 0 ;
    if (i <= j || first [j] <= maxfirst [i])
    {
        (*lca_handle) = -1;
        return (SPEX_OK);  // j not a leaf
    }

    // Declare variables
    int64_t q, s, sparent, jprev ;

    maxfirst [i] = first [j] ;      // update max first[j] seen so far
    jprev = prevleaf [i] ;          // jprev = previous leaf of ith subtree
    prevleaf [i] = j ;
    (*jleaf) = (jprev == -1) ? 1:2 ; // j is first or subsequent leaf

    if ((*jleaf) == 1)
    {
        (*lca_handle) = i;
        return SPEX_OK ;   // if 1st leaf, q = root of ith subtree
    }
    for (q = jprev ; q != ancestor [q] ; q = ancestor [q]);
    for (s = jprev ; s != q ; s = sparent)
    {
        sparent = ancestor [s] ;    // path compression
        ancestor [s] = q ;
    }
    (*lca_handle) = q;
    return SPEX_OK ;                    // q = least common ancestor (jprev,j)
}
