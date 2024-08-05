//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_post: Postorder a forest
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE     \
{                               \
    SPEX_FREE (w);             \
}

#define SPEX_FREE_ALL           \
{                               \
    SPEX_FREE (post);          \
    SPEX_FREE_WORKSPACE ;       \
}

#include "spex_cholesky_internal.h"

/* Purpose: post order a forest */

SPEX_info spex_symmetric_post
(
    // Output
    int64_t **post_handle, // On output: post-order of the forest
                           // On input: undefied
    // Input
    const int64_t *parent, // Parent[j] is parent of node j in forest
    const int64_t n        // Number of nodes in the forest
)
{

    SPEX_info info ;

    // All inputs have been checked by the caller
    ASSERT( n >= 0);

    // Declare variables
    int64_t j, k = 0, *post = NULL, *w = NULL, *head, *next, *stack ;

    // Allocate the postordering result
    post = (int64_t*) SPEX_malloc(n* sizeof(int64_t));

    // Create a workspace
    w = (int64_t*) SPEX_malloc (3*n* sizeof (int64_t));
    if ((w == NULL) || (post == NULL))
    {
        SPEX_FREE_ALL;
        return (SPEX_OUT_OF_MEMORY);
    }
    head = w ; next = w + n ; stack = w + 2*n ;

    // Empty linked lists
    for (j = 0 ; j < n ; j++)
    {
        head [j] = -1 ;
    }
    for (j = n-1 ; j >= 0 ; j--)            // traverse nodes in reverse order
    {
        if (parent [j] == -1) continue ;    // j is a root
        next [j] = head [parent [j]] ;      // add j to list of its parent
        head [parent [j]] = j ;
    }
    for (j = 0 ; j < n ; j++)
    {
        if (parent [j] != -1) continue ;    // skip j if it is not a root
        SPEX_CHECK(spex_symmetric_tdfs (&k, j, head, next, post, stack));
    }
    SPEX_FREE_WORKSPACE ;
    (*post_handle) = post;
    return (SPEX_OK);
}
