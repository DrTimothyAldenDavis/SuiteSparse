//------------------------------------------------------------------------------
// SPEX_Utilities/spex_amd: Call AMD for matrix ordering
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_ALL       \
{                           \
    SPEX_free (perm);      \
}

#include "spex_util_internal.h"

/* Purpose: SPEX interface to AMD
 */

SPEX_info spex_amd
(
    int64_t **perm_handle,
    int64_t *nnz,
    const SPEX_matrix A,
    const SPEX_options option
)
{

    (*nnz) = 0 ;
    (*perm_handle) = NULL ;

    int pr = SPEX_OPTION_PRINT_LEVEL(option);
    int64_t n = A->n;
    int64_t *perm = NULL ;

    // Allocate memory for permutation
    perm = (int64_t*)SPEX_malloc( (n+1)*sizeof(int64_t) );
    if (perm == NULL)
    {
        SPEX_FREE_ALL;
        return (SPEX_OUT_OF_MEMORY);
    }

    double Control[AMD_CONTROL];           // Declare AMD control
    amd_l_defaults(Control);              // Set AMD defaults
    double Info [AMD_INFO];
    // Perform AMD
    int64_t amd_result = amd_l_order(n,
                (int64_t *)A->p, (int64_t *)A->i,
                (int64_t *)perm, Control, Info);
    if (pr > 0)   // Output AMD info if desired
    {
        SPEX_PRINTF("\n****Ordering Information****\n");
        amd_l_control(Control);
        amd_l_info(Info);
    }
    if (!(amd_result == AMD_OK || amd_result == AMD_OK_BUT_JUMBLED))
    {
        // AMD failed: either out of memory, or bad input
        SPEX_FREE_ALL;
        if (amd_result == AMD_OUT_OF_MEMORY)
        {
            // AMD ran out of memory
            return (SPEX_OUT_OF_MEMORY);
        }
        // input matrix is invalid
        return (SPEX_INCORRECT_INPUT);
    }

    (*nnz) = Info[AMD_LNZ];  // Exact number of nonzeros for Cholesky
    (*perm_handle)=perm;
    return SPEX_OK;
}

