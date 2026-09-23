//------------------------------------------------------------------------------
// SPEX_QR/spex_qr_etree: Compute the elimination tree of a matrix A
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2026, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
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

#include "spex_qr_internal.h"


/* Purpose: Compute the elimination tree of ATA without forming ATA */

SPEX_info spex_qr_etree
(
    // Output
    int64_t **tree_handle,      // On output: contains the column elimination 
                                // tree of A
                                // On input: undefined.
    // Input
    const SPEX_matrix A         // Input matrix.
)
{

    // All inputs are checked by the caller so asserts are used here as a
    // reminder of the appropriate formats
    ASSERT (A != NULL);
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->n <= A->m);
    ASSERT (tree_handle != NULL);
    (*tree_handle) = NULL ;

    // Declare variables
    int64_t i, k, p, n, m, inext, *w = NULL, *parent = NULL, *ancestor, *prev ;
    n = A->n ; m=A->m;

    // Allocate parent
    parent = (int64_t*) SPEX_malloc( n * sizeof(int64_t));
    // Allocate workspace
    w = (int64_t*) SPEX_malloc( (m+n) * sizeof(int64_t) );
    if (!parent || !w)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    ancestor = w ;
    prev=w+n;
    for(i=0; i<m; i++)
    {
        prev[i]=-1;
    }

    for (k = 0 ; k < n ; k++)
    {
        parent [k] = -1 ;                       // node k has no parent yet
        ancestor [k] = -1 ;                     // nor does k have an ancestor
        for (p = A->p [k] ; p < A->p [k+1] ; p++)
        {
            i = prev[A->i[p]];
            for ( ; i != -1 && i < k ; i = inext)   // traverse from i to k
            {
                inext = ancestor [i] ;              // inext = ancestor of i
                ancestor [i] = k ;                  // path compression
                if (inext == -1) parent [i] = k ;   // no anc., parent is k
            }
            prev[A->i[p]]=k;
        }
    }
    SPEX_FREE_WORKSPACE ;
    (*tree_handle) = parent;
    return SPEX_OK;
}
