//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_permute_A: Symmetric permutation of matrix A
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_cholesky_internal.h"

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL { SPEX_matrix_free (&PAP, NULL); }

/* Purpose: Permute the matrix A and return PAP = P*A*P'.  On input PAP is
 * undefined and A contains the input matrix.  On output PAP contains the
 * permuted matrix (P*A*P').
 *
 * Input arguments:
 *
 * PAP_handle:   The user's permuted input matrix.
 *
 * A:            The user's input matrix
 *
 * S:            Symbolic analysis struct for Cholesky or LDL factorization.
 *               Contains row/column permutation of A
 */

SPEX_info spex_symmetric_permute_A
(
    //Output
    SPEX_matrix* PAP_handle,   // On input: undefined
                               // On output: contains the permuted matrix
    //Input
    const SPEX_matrix A,       // Input matrix
    const bool numeric,        // True if user wants to permute pattern and
                               // numbers, false if only pattern
    const SPEX_symbolic_analysis S  // Symbolic analysis struct that contains
                                // row/column permutations
)
{

    SPEX_info info;

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    ASSERT(A != NULL);
    ASSERT(S != NULL);
    ASSERT(PAP_handle != NULL);
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);

    // Create indices and pinv, the inverse row permutation
    int64_t j, k, t, nz = 0, n = A->n;
    (*PAP_handle) = NULL ;
    //int64_t *pinv = NULL;

    // Allocate memory for PAP which is a permuted copy of A
    SPEX_matrix PAP = NULL ;
    SPEX_CHECK(SPEX_matrix_allocate(&PAP, SPEX_CSC, SPEX_MPZ, n, n, A->p[n],
        false, true, NULL));

    if(numeric)
    {

        //----------------------------------------------------------------------
        // construct PAP with numerical values
        //----------------------------------------------------------------------

        // Set PAP scale
        SPEX_MPQ_SET(PAP->scale, A->scale);

        // Populate the entries in PAP
        for (k = 0; k < n; k++)
        {
            // Set the number of nonzeros in the kth column of PAP
            PAP->p[k] = nz;
            // Column k of PAP is equal to column S->P_perm[k] of A. j is the
            // starting point for nonzeros and indices for column S->P_perm[k]
            // of A
            j = S->P_perm[k];
            // Iterate across the nonzeros in column S->P_perm[k]
            for (t = A->p[j]; t < A->p[j+1]; t++)
            {
                // Set the nonzero value and location of the entries in column
                // k of PAP
                SPEX_MPZ_SET(PAP->x.mpz[nz], A->x.mpz[t]);
                // Row i of this nonzero is equal to pinv[A->i[t]]
                PAP->i[nz] = S->Pinv_perm[ A->i[t] ];
                // Move to the next nonzero element of PAP
                nz++;
            }
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // construct PAP with just its pattern, not the values
        //----------------------------------------------------------------------

        // FUTURE: tell SPEX_matrix_allocate not to allocate PAP->x at all.
        SPEX_FREE (PAP->x.mpz);
        ASSERT (PAP->x.mpz == NULL);
        PAP->x_shallow = true ;

        // Populate the entries in PAP
        for (k = 0; k < n; k++)
        {
            // Set the number of nonzeros in the kth column of PAP
            PAP->p[k] = nz;
            // Column k of PAP is equal to column S->p[k] of A. j is the
            // starting point for nonzeros and indices for column S->p[k] of A
            j = S->P_perm[k];
            // Iterate across the nonzeros in column S->p[k]
            for (t = A->p[j]; t < A->p[j+1]; t++)
            {
                // Row i of this nonzero is equal to pinv[A->i[t]]
                PAP->i[nz] = S->Pinv_perm[ A->i[t] ];
                // Move to the next nonzero element of PAP
                nz++;
            }
        }
    }

    // Finalize the last column of PAP
    PAP->p[n] = nz;
    // Set output, return success
    (*PAP_handle) = PAP;
    return SPEX_OK;
}

