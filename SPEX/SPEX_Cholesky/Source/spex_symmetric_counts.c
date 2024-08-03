//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_counts: Column counts for Cholesky factorization
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE    \
{                              \
    SPEX_FREE(w);              \
}

#define SPEX_FREE_ALL          \
{                              \
    SPEX_FREE_WORKSPACE;       \
    SPEX_FREE(colcount);       \
}

#include "spex_cholesky_internal.h"

#define HEAD(k,j) ( j)
#define NEXT(J)   (-1)

/* Purpose: Obtain the column counts of an SPD matrix for Cholesky
 * factorization or a symmetrix matrix for LDL factorization.  This is a
 * modified version of Csparse's cs_chol_counts function.
 */

SPEX_info spex_symmetric_counts
(
    // Output
    int64_t **c_handle,     // On ouptut: column counts
                            // On input: undefined
    // Input
    const SPEX_matrix A,    // Input matrix
    const int64_t *parent,  // Elimination tree
    const int64_t *post     // Post-order of the tree
)
{

    SPEX_info info;
    int64_t i, j, k, n, J, s, p, q, jleaf, *colcount = NULL, *w = NULL;
    // Auxiliary variables
    int64_t  *maxfirst, *prevleaf, *ancestor, *first, *delta ;
    n = A->n ;
    // Can not have negative n
    ASSERT(n >= 0);
    // Size of workspace
    s = 4*n ;
    // Allocate result in delta
    colcount = (int64_t*) SPEX_malloc(n* sizeof (int64_t));
    // Create a workspace of size s
    w = (int64_t*) SPEX_malloc (s* sizeof (int64_t));
    if (colcount == NULL || w == NULL)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    delta = colcount;
    ancestor = w ; maxfirst = w+n ; prevleaf = w+2*n ; first = w+3*n ;
    // Clear workspace
    for (k = 0 ; k < s ; k++)
    {
        w [k] = -1 ;
    }
    // Find first j
    for (k = 0 ; k < n ; k++)
    {
        j = post[k] ;
        delta[j] = (first[j] == -1) ? 1 : 0 ;  /* delta[j]=1 if j is a leaf */
        for ( ; j != -1 && first[j] == -1 ; j = parent[j])
        {
            first [j] = k ;
        }
    }
    // Initialize ancestor of each node
    for (i = 0 ; i < n ; i++)
    {
        ancestor[i] = i ;
    }
    for (k = 0 ; k < n ; k++)
    {
        j = post[k] ;          /* j is the kth node in postordered etree */
        if (parent[j] != -1)
        {
            delta[parent[j]]-- ;    /* j is not a root */
        }
        for (J = HEAD(k,j); J != -1 ; J = NEXT(J))   /* J=j for LL'=A case */
        {
            for (p = A->p[J] ; p < A->p[J+1] ; p++)
            {
                i = A->i[p] ;
                SPEX_CHECK(spex_symmetric_leaf(&q, i, j, first, maxfirst,
                    prevleaf, ancestor, &jleaf));
                if (jleaf >= 1)
                {
                    delta[j]++ ;   /* A(i,j) is in skeleton */
                }
                if (jleaf == 2)
                {
                    delta[q]-- ;   /* account for overlap in q */
                }
            }
        }
        if (parent[j] != -1)
        {
            ancestor[j] = parent[j] ;
        }
    }
    for (j = 0 ; j < n ; j++)           /* sum up delta's of each child */
    {
        if (parent[j] != -1)
        {
            colcount[parent[j]] += colcount[j] ;
        }
    }
    (*c_handle) = colcount;
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
