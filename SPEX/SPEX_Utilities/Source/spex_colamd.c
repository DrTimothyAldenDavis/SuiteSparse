//------------------------------------------------------------------------------
// SPEX_Utilities/spex_colamd: Call COLAMD for matrix ordering
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


#define SPEX_FREE_ALL       \
{                           \
    SPEX_FREE (perm);      \
    SPEX_FREE (A2);        \
}

#include "spex_util_internal.h"

/* Purpose:  SPEX interface to COLAMD
 */

SPEX_info spex_colamd
(
    int64_t **perm_handle,
    int64_t *nnz,
    const SPEX_matrix A,
    const SPEX_options option
)
{

    SPEX_info info;
    (*nnz) = 0 ;
    (*perm_handle) = NULL ;
    int64_t *A2 = NULL, *perm = NULL ;

    int64_t anz; // Number of nonzeros in A
    SPEX_CHECK (SPEX_matrix_nnz(&anz, A, option));
    int64_t i, n = A->n;

    int pr = SPEX_OPTION_PRINT_LEVEL(option);

    // Allocate memory for permutation
    perm = (int64_t*)SPEX_malloc( (n+1)*sizeof(int64_t) );
    if (perm == NULL)
    {
        SPEX_FREE_ALL;
        return (SPEX_OUT_OF_MEMORY);
    }

    // determine workspace required for COLAMD
    int64_t Alen = colamd_l_recommended (anz, n, n) + 2*n ;
    A2 = (int64_t*) SPEX_malloc (Alen*sizeof(int64_t));
    if (!A2)
    {
        // out of memory
        SPEX_FREE_ALL;
        return (SPEX_OUT_OF_MEMORY);
    }

    // Initialize S->p as per COLAMD documentation
    for (i = 0; i < n+1; i++)
    {
        perm[i] = A->p[i];
    }

    // Initialize A2 per COLAMD documentation
    for (i = 0; i < anz; i++)
    {
        A2[i] = A->i[i];
    }
    
    // find the colamd ordering
    int64_t stats[COLAMD_STATS];
    int64_t colamd_result = colamd_l (n, n, Alen, A2, perm,
        (double *) NULL, stats);
    if (!colamd_result)
    {
        printf("fail\n");
        // COLAMD failed: matrix is invalid
        SPEX_FREE_ALL;
        return (SPEX_INCORRECT_INPUT);
    }

    // very rough estimate for lnz and unz
    (*nnz) = 10*anz;

    // Print stats if desired
    if (pr > 0)
    {
        SPEX_PRINTF ("\n****Ordering Information****\n");
        colamd_l_report ((int64_t *) stats);
    }

    // free workspace and return result
    SPEX_FREE (A2);
    (*perm_handle) = perm ;
    return SPEX_OK;
}

