//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_ereach: Compute reach of an elimination tree
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_cholesky_internal.h"

/* Purpose: This function computes the reach of the kth row of A on the
 * elimination tree of A.  On input, k is the iteration of the algorithm,
 * parent contains the elimination tree and w is workspace.  On output,
 * xi[top_handle..n-1] contains the nonzero pattern of the kth row of L (or the
 * kth column of L').
 */

SPEX_info spex_symmetric_ereach
(
    // Output
    int64_t *top_handle,    // On output: starting point of nonzero pattern
                            // On input: undefined
    int64_t *xi,            // On output: contains the nonzero pattern in
                            // xi[top..n-1]
                            // On input: undefined
    // Input
    const SPEX_matrix A,    // Matrix to be analyzed
    const int64_t k,        // Node to start at
    const int64_t *parent,  // Elimination tree of A
    int64_t *w              // Workspace array
)
{

    // Check inputs
    ASSERT(A->n >= 0);
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(A->type == SPEX_MPZ);

    // Declare variables
    int64_t i, p, n, len, top ;
    top = n = A->n ;

    // Mark node k as visited
    SPEX_MARK(w, k);

    // Iterate across nonzeros in A(:,k)
    for (p = A->p[k] ; p < A->p[k+1] ; p++)
    {
        // A(i,k) is nonzero
        i = A->i[p] ;
        if (i > k)
        {
            continue ;  // only use upper triangular part of A
        }
        for (len = 0 ; !SPEX_MARKED(w,i); i = parent[i]) // traverse up etree
        {
            ASSERT (i >= 0 && i < n);
            xi[len++] = i ;           // L(k,i) is nonzero
            SPEX_MARK(w, i);        // mark i as visited
        }
        while (len > 0) xi[--top] = xi[--len] ; // push path onto stack
    }
    for (p = top ; p < n ; p++) SPEX_MARK(w, xi[p]);    // unmark all nodes
    SPEX_MARK(w, k);                // unmark node k
    (*top_handle) = top ;
    return SPEX_OK ;                 // xi [top..n-1] contains pattern of L(k,:)
}
