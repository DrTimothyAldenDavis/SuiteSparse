//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_etree: Compute the elimination tree of a matrix A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE \
{                           \
    SPEX_FREE(w);           \
}

#define SPEX_FREE_ALL       \
{                           \
    SPEX_FREE_WORKSPACE ;   \
    SPEX_FREE(parent);      \
}

#include "spex_cholesky_internal.h"


/* Purpose: Compute the elimination tree of A */

SPEX_info spex_symmetric_etree
(
    // Output
    int64_t **tree_handle,      // On output: contains the elimination tree of A
                                // On input: undefined.
    // Input
    const SPEX_matrix A         // Input matrix (must be symmetric with
                                // nonzero diagonal).
)
{

    // All inputs are checked by the caller so asserts are used here as a
    // reminder of the appropriate formats
    ASSERT (A != NULL);
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->n == A->m);
    ASSERT (tree_handle != NULL);
    (*tree_handle) = NULL ;

    // Declare variables
    int64_t i, k, p, n, inext, *w = NULL, *parent = NULL, *ancestor ;
    n = A->n ;

    // Allocate parent
    parent = (int64_t*) SPEX_malloc( n * sizeof(int64_t));
    // Allocate workspace
    w = (int64_t*) SPEX_malloc( n * sizeof(int64_t) );
    if (!parent || !w)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    ancestor = w ;
    for (k = 0 ; k < n ; k++)
    {
        parent [k] = -1 ;                       // node k has no parent yet
        ancestor [k] = -1 ;                     // nor does k have an ancestor
        for (p = A->p [k] ; p < A->p [k+1] ; p++)
        {
            i = A->i [p] ;
            for ( ; i != -1 && i < k ; i = inext)   // traverse from i to k
            {
                inext = ancestor [i] ;              // inext = ancestor of i
                ancestor [i] = k ;                  // path compression
                if (inext == -1) parent [i] = k ;   // no anc., parent is k
            }
        }
    }
    SPEX_FREE_WORKSPACE ;
    (*tree_handle) = parent;
    return SPEX_OK;
}
