//------------------------------------------------------------------------------
// SLIP_LU/slip_reach: compute the set of nodes reachable from an input set
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#include "slip_internal.h"

/* Purpose: This function computes the reach of column k of A on the graph of L
 * mathematically that is: xi = Reach(A(:,k))_G_L
 *
 * This function is derived from CSparse/cs_reach.c
 */

void slip_reach    // compute the reach of column k of A on the graph of L
(
    int64_t *top,
    SLIP_matrix* L,         // matrix representing graph of L
    const SLIP_matrix* A,   // input matrix
    int64_t k,              // column of A of interest
    int64_t* xi,            // nonzero pattern
    const int64_t* pinv     // row permutation
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------
    if (top == NULL) { return ;}
    // inputs have been checked in slip_ref_triangular_solve
    int64_t p, n = L->n;
    *top = n;

    //--------------------------------------------------------------------------
    // Iterating across number of nonzero in column k
    //--------------------------------------------------------------------------

    for (p = A->p[k]; p < A->p[k + 1]; p++)
    {
        // DFS at unmarked node i
        if (!SLIP_MARKED(L->p, A->i[p]))
        {
            slip_dfs(top, A->i[p], L, xi, xi+n, pinv);
        }
    }

    // Restore L
    for ( p = *top; p < n; p++)
    {
        SLIP_MARK(L->p, xi[p]);
    }
}

