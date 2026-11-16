//------------------------------------------------------------------------------
// SPEX_Utilities/spex_history_update: Performs history update
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_util_internal.h"

/*
 * History Update
 *
 * A[a]=(rhos[i]*A[a])/rhos[k]
 *
 * Used in LU triangular solve, Cholesky up and left triangular solve and QR IPGS
 * In LU and Cholesky A is L
 * In QR A is Q
 */

SPEX_info spex_history_update
(
    SPEX_matrix A,              // Matrix to be updated
    const SPEX_matrix rhos,     // sequence of pivots
    const int64_t a,            // Index of element in A to be updated   
    const int64_t i,            // Index of current pivot
    const int64_t j,            // History value
    const int64_t k,            // Index of last updated pivot
    const int64_t thres,        // Threshold after which division must be performed
    const SPEX_options option   // Command options
)
{
    
    SPEX_info info;
    
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(rhos->type == SPEX_MPZ);
    ASSERT(rhos->kind == SPEX_DENSE);

    // A[a] = A[a]*rhos[i]
    SPEX_MPZ_MUL(A->x.mpz[a], A->x.mpz[a], rhos->x.mpz[i]);
    if(j>thres)
    {
        // If necessary, divide by the previous pivot. This is necessary
        // if the history of A[a] (j) is larger than the threshold
        // which in most cases is 0 or 1
        // A[a] = A[a] / rhos[k]
        SPEX_MPZ_DIVEXACT(A->x.mpz[a], A->x.mpz[a], rhos->x.mpz[k]);
    }
    
    return SPEX_OK;
}


