//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_reach: compute the set of nodes reachable from an input set
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_left_lu_internal.h"

/* Purpose: This function computes the reach of column k of A on the graph of L
 * mathematically that is: xi = Reach(A(:,k))_G_L
 *
 * This function is derived from CSparse/cs_reach.c
 */

void spex_left_lu_reach    // compute the reach of column k of A on the graph of L
(
    int64_t *top,
    SPEX_matrix* L,         // matrix representing graph of L
    const SPEX_matrix* A,   // input matrix
    int64_t k,              // column of A of interest
    int64_t* xi,            // nonzero pattern
    const int64_t* pinv     // row permutation
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------
    if (top == NULL) { return ;}
    // inputs have been checked in spex_ref_triangular_solve
    int64_t p, n = L->n;
    *top = n;

    //--------------------------------------------------------------------------
    // Iterating across number of nonzero in column k
    //--------------------------------------------------------------------------

    for (p = A->p[k]; p < A->p[k + 1]; p++)
    {
        // DFS at unmarked node i
        if (!SPEX_MARKED(L->p, A->i[p]))
        {
            spex_left_lu_dfs(top, A->i[p], L, xi, xi+n, pinv);
        }
    }

    // Restore L
    for ( p = *top; p < n; p++)
    {
        SPEX_MARK(L->p, xi[p]);
    }
}

