//------------------------------------------------------------------------------
// SPEX_QR/Source/spex_qr_basic_solve.c: Basic solution back solve
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs sparse REF backward substitution for
 * underdetermined SLEs, solving the system Rx = b. Where the last n-rank rows of
 * R are 0.
 *
 * R is a sparse mpz matrix, and bx is a dense mpz matrix.
 * If A is full rank, the diagonal entry of U will appear as the last entry in
 * each column. If A is rank deficient, this is not true and the diagonal entry
 * needs to be searched for.
 *
 * The input argument bx contains b on input, and it is overwritten on output
 * by the solution x.
 */

#include "spex_qr_internal.h"

SPEX_info spex_qr_back_sub // performs sparse REF backward substitution
    (
        SPEX_matrix bx,           // right hand side matrix
        const SPEX_matrix R,      // input upper triangular matrix
        const int64_t rank,       // rank of right triangular matrix
        const SPEX_matrix rhos,   // sequence of pivots
        const SPEX_options option // command options
    )
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info;
    SPEX_REQUIRE(R, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(bx, SPEX_DENSE, SPEX_MPZ);

    //--------------------------------------------------------------------------

    int sgn;
    mpz_t *Rx = R->x.mpz;
    int64_t *Ri = R->i;
    int64_t *Rp = R->p;
    int64_t n = R->n;
    int64_t extra;

    // Loop through all RHS vectors
    for (int64_t k = 0; k < bx->n; k++)
    {
        // Start at bx[n]
        for (int64_t j = rank - 1; j >= 0; j--)
        {
            // If bx[j] is zero skip this iteration
            SPEX_MPZ_SGN(&sgn, SPEX_2D(bx, j, k, mpz));
            if (sgn == 0)
            {
                continue;
            }

            // If A is rank deficient, the diagonal entry is not at position
            // R->p[j+1] so we will search for it instead.
            // This is still O(1) time if A is full rank because it starts at the
            // end of the column.
            int64_t diag_idx = -1;
            for (int64_t i = Rp[j + 1] - 1; i >= Rp[j]; i--)
            {
                if (Ri[i] == j)
                {
                    diag_idx = i;
                    break;
                }
            }

            // Divide by the found diagonal element
            SPEX_MPZ_DIVEXACT(SPEX_2D(bx, j, k, mpz),
                              SPEX_2D(bx, j, k, mpz),
                              Rx[diag_idx]);

            // Back-substitute strictly into the rows above the diagonal
            for (int64_t i = Rp[j]; i < diag_idx; i++)
            {
                SPEX_MPZ_SGN(&sgn, Rx[i]);
                if (sgn == 0)
                {
                    continue;
                }
                // bx[i] = bx[i] - Rx[i]*bx[j]
                SPEX_MPZ_SUBMUL(SPEX_2D(bx, Ri[i], k, mpz),
                                Rx[i], SPEX_2D(bx, j, k, mpz));
            }
        }
    }

    return (SPEX_OK);
}
