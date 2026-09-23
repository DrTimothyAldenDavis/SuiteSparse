//------------------------------------------------------------------------------
// SPEX_Utilities/spex_ipge_update: Performs IPGE update
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_util_internal.h"

/*
 * IPGE Update
 *
 * A[a]=(rhos[j]*A[a]-B[b]*A[k])/rhos[i]
 *
 * Used in LU triangular solve, Cholesky up and left triangular solve and QR IPGS
 * In LU and Cholesky A is L and B is x
 * In QR A is Q and B is R
 */

SPEX_info spex_ipge_update
(
    SPEX_matrix A,            // Matrix to be updated
    SPEX_matrix B,            // Matrix B
    const SPEX_matrix rhos,   // Sequence of pivots
    const int64_t a,          // Index for the element in A to be updated
    const int64_t b,          // Index for matrix B
    const int64_t i,          // Index for the dividing pivot
    const int64_t j,          // Index for the multiplying pivot
    const int64_t k,          // Index for matrix A
    const SPEX_options option // Command options
)
{
    
    SPEX_info info;
    
    ASSERT(B->type == SPEX_MPZ);
    ASSERT(B->kind == SPEX_CSC);
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(rhos->type == SPEX_MPZ);
    ASSERT(rhos->kind == SPEX_DENSE);
    
    int sgn;
    
    SPEX_MPZ_SGN(&sgn, A->x.mpz[a]);
    if(sgn==0)
    {
        SPEX_MPZ_SUBMUL(A->x.mpz[a], B->x.mpz[b], A->x.mpz[k]);
        if(j>=1)
        {
            SPEX_MPZ_SGN(&sgn, rhos->x.mpz[i]);
            SPEX_MPZ_DIVEXACT(A->x.mpz[a], A->x.mpz[a], rhos->x.mpz[i]);
        }
    }
    else
    {
        
        SPEX_MPZ_MUL(A->x.mpz[a], A->x.mpz[a], rhos->x.mpz[j]);
        SPEX_MPZ_SUBMUL(A->x.mpz[a], B->x.mpz[b], A->x.mpz[k]); 
        if(j>=1)
        {
            SPEX_MPZ_DIVEXACT(A->x.mpz[a], A->x.mpz[a], rhos->x.mpz[i]);
        }
    }
    
    
    return SPEX_OK;
}
