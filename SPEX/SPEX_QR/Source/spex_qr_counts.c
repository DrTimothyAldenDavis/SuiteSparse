//------------------------------------------------------------------------------
// SPEX_QR/spex_qr_counts: Column counts for QR factorization
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2026, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORKSPACE          \
    {                                \
        SPEX_FREE(w);                \
        SPEX_matrix_free(&AT, NULL); \
    }

#define SPEX_FREE_ALL        \
    {                        \
        SPEX_FREE_WORKSPACE; \
        SPEX_FREE(colcount); \
    }

#include "spex_qr_internal.h"

#define HEAD(k, j) (head[k])
#define NEXT(J) (next[J])

// This is a modified version of Csparse's cs_init_ata function
static void spex_qr_init_ata(
    SPEX_matrix AT,
    const int64_t *post,
    int64_t *w,
    int64_t **head,
    int64_t **next)
{
    int64_t i, k, p, m = AT->n, n = AT->m;
    *head = w + 4 * n, *next = w + 5 * n + 1;
    for (k = 0; k < n; k++)
        w[post[k]] = k; /* invert post */
    for (i = 0; i < m; i++)
    {
        for (k = n, p = AT->p[i]; p < AT->p[i + 1]; p++)
        {
            k = SPEX_MIN(k, w[AT->i[p]]);
        }
        (*next)[i] = (*head)[k]; /* place row i in linked list k */
        (*head)[k] = i;
    }
}

/* Purpose: Obtain the column counts of a matrix for QR
 * factorization. This is a modified version of Csparse's cs_counts
 * function
 */

SPEX_info spex_qr_counts
(
    // Output
    int64_t **c_handle, // On ouptut: column counts
                        // On input: undefined
    int64_t *rnz,       // On output: number of nonzeros in R
    // Input
    const SPEX_matrix A,   // Input matrix
    const int64_t *parent, // Elimination tree
    const int64_t *post    // Post-order of the tree
)
{

    SPEX_info info;
    int64_t i, j, k, n, m, J, s, p, q, jleaf, nz;
    int64_t *colcount = NULL, *w = NULL;
    int64_t *head = NULL, *next = NULL;
    SPEX_matrix AT = NULL;
    // Auxiliary variables
    int64_t *maxfirst, *prevleaf, *ancestor, *first, *delta;
    n = A->n;
    m = A->m;
    // Can not have negative n
    ASSERT(n >= 0);
    // Size of workspace
    s = 4 * n + m + n + 1;
    // Allocate result in delta
    colcount = (int64_t *)SPEX_malloc(n * sizeof(int64_t));
    // Create a workspace of size s
    w = (int64_t *)SPEX_malloc(s * sizeof(int64_t));
    if (colcount == NULL || w == NULL)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }

    // Transpose matrix A
    SPEX_CHECK(SPEX_transpose(&AT, A, false, NULL));
    delta = colcount;
    ancestor = w;
    maxfirst = w + n;
    prevleaf = w + 2 * n;
    first = w + 3 * n;
    // Clear workspace
    for (k = 0; k < s; k++)
    {
        w[k] = -1;
    }
    // Find first j
    for (k = 0; k < n; k++)
    {
        j = post[k];
        delta[j] = (first[j] == -1) ? 1 : 0; /* delta[j]=1 if j is a leaf */
        for (; j != -1 && first[j] == -1; j = parent[j])
        {
            first[j] = k;
        }
    }
    // Init ATA
    spex_qr_init_ata(AT, post, w, &head, &next);

    // Initialize ancestor of each node
    for (i = 0; i < n; i++)
    {
        ancestor[i] = i;
    }
    for (k = 0; k < n; k++)
    {
        j = post[k]; /* j is the kth node in postordered etree */
        if (parent[j] != -1)
        {
            delta[parent[j]]--; /* j is not a root */
        }
        for (J = HEAD(k, j); J != -1; J = NEXT(J)) /* J=j for LL'=A case */
        {
            for (p = AT->p[J]; p < AT->p[J + 1]; p++)
            {
                i = AT->i[p];
                SPEX_CHECK(spex_symmetric_leaf(&q, i, j, first, maxfirst,
                                               prevleaf, ancestor, &jleaf));
                if (jleaf >= 1)
                {
                    delta[j]++; /* A(i,j) is in skeleton */
                }
                if (jleaf == 2)
                {
                    delta[q]--; /* account for overlap in q */
                }
            }
        }
        if (parent[j] != -1)
        {
            ancestor[j] = parent[j];
        }
    }
    for (j = 0; j < n; j++) /* sum up delta's of each child */
    {
        if (parent[j] != -1)
        {
            colcount[parent[j]] += colcount[j];
        }
    }

    nz = 0;
    for (i = 0; i < n; i++)
    {
        nz += colcount[i];
    }
    *rnz = nz;

    (*c_handle) = colcount;
    SPEX_FREE_WORKSPACE;
    return SPEX_OK;
}
