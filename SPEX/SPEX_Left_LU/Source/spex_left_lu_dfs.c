//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_dfs: depth-first search
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs a depth first search of the graph of the
 * matrix starting at node j. The output of this function is the set of nonzero
 * indices in the xi vector.  This function is modified from CSparse/cs_dfs.
 */

#include "spex_left_lu_internal.h"

void spex_left_lu_dfs // performs a dfs of the graph of the matrix starting at node j
(
    int64_t *top,          // beginning of stack
    int64_t j,             // What node to start DFS at
    SPEX_matrix* L,        // matrix which represents the Graph of L
    int64_t* xi,           // the nonzero pattern
    int64_t* pstack,       // workspace vector
    const int64_t* pinv    // row permutation
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_KIND (L, SPEX_CSC) ;

    // top xi etc already checked in the caller function

    //--------------------------------------------------------------------------

    int64_t i, p, p2, done, jnew, head = 0;

    // Initialize the recursion stack
    xi[0] = j;

    while (head >= 0)
    {
        // The j value of the nonzero
        j = xi[head];
        // The relative j value
        jnew = pinv[j];

        //----------------------------------------------------------------------
        // Mark L->p[j] if not marked yet
        //----------------------------------------------------------------------
        if (!SPEX_MARKED (L->p,j))
        {
            SPEX_MARK(L->p,j);
            pstack[head] = (jnew < 0) ? 0 : SPEX_UNFLIP(L->p[jnew]);
        }
        // Node j is done if no unvisited neighbors
        done = 1;

        p2 = (jnew < 0) ? 0 : SPEX_UNFLIP(L->p[jnew+1]);

        //----------------------------------------------------------------------
        // Examine all neighbors of j
        //----------------------------------------------------------------------
        for (p = pstack[head]; p < p2; p++)
        {
            // Looking at neighbor node i
            i = L->i[p];
            // Skip already visited node
            if (SPEX_MARKED(L->p,i))  {continue;}

            // pause DFS of node j
            pstack[head] = p;
            // Start DFS at node i
            xi[++head] = i;
            // node j is not done
            done = 0;
            // break to start dfs i
            break;
        }
        if (done != 0)
        {
            head--;
            xi[--(*top)] = j;
        }
    }
    return;
}
