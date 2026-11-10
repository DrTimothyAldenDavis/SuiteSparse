//------------------------------------------------------------------------------
// SPEX_QR/spex_qr_permute_A: Permutation of matrix A
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2026, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_qr_internal.h"

#undef SPEX_FREE_ALL
#define SPEX_FREE_ALL                 \
    {                                 \
        SPEX_matrix_free(&PAQ, NULL); \
    }

/* Purpose: Given the column permutation Q permute the matrix A and return PAQ'
 * Input arguments:
 *
 * PAQ_handle:   The user's permuted input matrix.
 *
 * A:            The user's input matrix
 *
 * S:            Symbolic analysis struct for QR factorization.
 *               Contains column permutation of A
 */

SPEX_info spex_qr_permute_A
(
    // Output
    SPEX_matrix *PAQ_handle, // On input: undefined
                             // On output: contains the permuted matrix
    // Input
    const SPEX_matrix A,      // Input matrix
    const bool numeric,       // True if user wants to permute pattern and
                              // numbers, false if only pattern
    const int64_t *Q_perm,    // column permutation
    const int64_t *P_perm,    // row permutation
    const SPEX_options option // Command options (Default if NULL)
)
{

    SPEX_info info;

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    ASSERT(A != NULL);
    ASSERT(S != NULL);
    ASSERT(PAQ_handle != NULL);
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);

    // Create indices and pinv, the inverse row permutation
    int64_t j, k, t, nz = 0, n = A->n, m = A->m;
    (*PAQ_handle) = NULL;

    // Allocate memory for PAQ which is a permuted copy of A
    SPEX_matrix PAQ = NULL;
    SPEX_CHECK(SPEX_matrix_allocate(&PAQ, SPEX_CSC, SPEX_MPZ, m, n, A->p[n],
                                    false, true, NULL));

    if (numeric)
    {

        //----------------------------------------------------------------------
        // construct PAQ with numerical values
        //----------------------------------------------------------------------

        // Set PAQ scale
        SPEX_MPQ_SET(PAQ->scale, A->scale);

        // Populate the entries in PAQ
        for (k = 0; k < n; k++)
        {
            // Set the number of nonzeros in the kth column of PAQ
            PAQ->p[k] = nz;
            // Column k of PAQ is equal to column Q_perm[k] of A. j is the
            // starting point for nonzeros and indices for column Q_perm[k]
            // of A
            j = Q_perm[k];
            // Iterate across the nonzeros in column Q_perm[k]
            for (t = A->p[j]; t < A->p[j + 1]; t++)
            {
                // Set the nonzero value and location of the entries in column
                // k of
                SPEX_MPZ_SET(PAQ->x.mpz[nz], A->x.mpz[t]);
                // Row i of this nonzero is equal to A->i[t] if P_perm doesn't exist
                PAQ->i[nz] = P_perm ? P_perm[A->i[t]] : A->i[t];
                // Move to the next nonzero element of PAQ
                nz++;
            }
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // construct PAQ with just its pattern, not the values
        //----------------------------------------------------------------------

        SPEX_FREE(PAQ->x.mpz);
        ASSERT(PAQ->x.mpz == NULL);
        PAQ->x_shallow = true;

        // Populate the entries in PAQ
        for (k = 0; k < n; k++)
        {
            // Set the number of nonzeros in the kth column of PAQ
            PAQ->p[k] = nz;
            // Column k of PAQ is equal to column Q_perm[k] of A. j is the
            // starting point for nonzeros and indices for column Q_perm[k] of A
            j = Q_perm[k];
            // Iterate across the nonzeros in column Q_perm[k]]
            for (t = A->p[j]; t < A->p[j + 1]; t++)
            {
                // Row i of this nonzero is equal to pinv[A->i[t]]
                PAQ->i[nz] = P_perm ? P_perm[A->i[t]] : A->i[t];
                // Move to the next nonzero element of PAQ
                nz++;
            }
        }
    }

    // Finalize the last column of PAQ
    PAQ->p[n] = nz;
    // Set output, return success
    (*PAQ_handle) = PAQ;
    return SPEX_OK;
}
