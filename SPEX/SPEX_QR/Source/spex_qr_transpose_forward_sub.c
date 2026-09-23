//------------------------------------------------------------------------------
// SPEX_QR/spex_qr_transpose_forward_sub:
//              sparse transpose forward substitution (x = (R'D)\x)
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2026, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs a sparse transpose roundoff-error-free (REF)
 * forward substitution for the transposed REF QR factorization,
 * that is x = (R'D) \x. This is a subroutine for solving the system Ax=b
 * when A is rectangular and contains more columns than rows.
 *
 * A must be full rank. If A is rank deficient, the only caller
 * spex_qr_transpose_backslash, will not call this function and instead
 * exit with an appropriate error.
 *
 * Mathematically, we do not transpose R directly, instead, since R is stored
 * in CSC format, we think of R' being stored in compressed row format.
 * We also assume that x is dense, thus we do not compute the nonzero pattern
 * and each nonzero in x is iterated across. The system that is solved is
 * thus U' D x_output = x_input, overwriting the right hand side with the
 * solution.
 *
 * On output, the SPEX matrix x is modified.
 *
 * This function is heavily based on the SPEX LU transpose solve
 *
 */

#define SPEX_FREE_ALL           \
    SPEX_matrix_free(&h, NULL);

#include "spex_qr_internal.h"

SPEX_info spex_qr_transpose_forward_sub
(
    const SPEX_matrix R,    // upper triangular matrix
    const int64_t rank,     // Rank of A which is also number of rows of zeros
    SPEX_matrix x,          // right hand side matrix of size n*numRHS
    const SPEX_matrix rhos  // sequence of pivots used in factorization
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE(R, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(x, SPEX_DENSE, SPEX_MPZ);
    SPEX_REQUIRE(rhos, SPEX_DENSE, SPEX_MPZ);

    //--------------------------------------------------------------------------

    int64_t i, hx, k, j, jnew;
    int sgn ;

    // Build the history matrix
    SPEX_matrix h = NULL ;
    SPEX_CHECK (SPEX_matrix_allocate(&h, SPEX_DENSE, SPEX_INT64, x->m, x->n,
        x->nzmax, false, true, NULL));

    // initialize entries of history matrix to be -1
    for (i = 0; i < x->nzmax; i++)
    {
        h->x.int64[i] = -1;
    }


    //--------------------------------------------------------------------------
    // Iterate across each RHS vector
    //--------------------------------------------------------------------------

    for (k = 0; k < x->n; k++)
    {

        //----------------------------------------------------------------------
        // Iterate accross all nonzeros in x. Assume x is dense
        //----------------------------------------------------------------------

        for (i = 0; i < rank; i++)
        {

            //------------------------------------------------------------------
            // IPGE updates
            //------------------------------------------------------------------

            // Access row i of R'
            // We are finalizing the value of x[i] which is initially set as b[i]
            // Thus, for an example b[i], we are calculating (in the dense case)
            // R[i,1] x[1] + R[i,2] x[2] + ... + R[i, i-1] x[i-1] + R[i,i] x[i] = b[i]
            // Thus R[i,i] x[i] = b[i] - (R[i,1] x[1] + R[i,2] x[2] + ... + R[i, i-1] x[i-1])
            // The first loop iterates through each nonzero in row i of R and performs
            // this submul IPGE update on b[i].
            // Once we have done so, we perform a history update on b[i] to finalize it

            // Find the diagonal element if A is rank deficient. If A has full rank,
            // the diagonal is located at R->p[i+1]-1 so no extra work is done.
            int64_t diag_idx = -1;
            for (int64_t p = R->p[i+1] - 1; p >= R->p[i]; p--)
            {
                if (R->i[p] == i)
                {
                    diag_idx = p;
                    break;
                }
            }

            // Process only the strict upper triangular part (elements above the diagonal)
            for (j = R->p[i]; j < diag_idx; j++)
            {
                // Column index of R[j]
                jnew = R->i[j];
                ASSERT (jnew <= i);

                // Now we history update x[i] if necessary with respect to jnew
                hx = SPEX_2D(h, i, k, int64);
                if (hx < jnew-1)
                {
                    // x[i] = x[i]*rhos[jnew-1]
                    SPEX_MPZ_MUL(SPEX_2D(x, i, k, mpz),
                                 SPEX_2D(x, i, k, mpz),
                                 SPEX_1D(rhos, jnew-1, mpz));

                    if (hx > -1)
                    {
                        SPEX_MPZ_DIVEXACT(SPEX_2D(x,i,k,mpz),
                                          SPEX_2D(x,i,k,mpz),
                                          SPEX_1D(rhos, hx, mpz));
                    }
                }

                // x[i]*rhos[jnew]
                SPEX_MPZ_MUL(SPEX_2D(x,i,k,mpz), SPEX_2D(x,i,k,mpz),
                             SPEX_1D(rhos, jnew, mpz));

                // Now, we update x[i] using R'[i,j]*x[j]
                SPEX_MPZ_SUBMUL(SPEX_2D(x,i,k,mpz),
                                R->x.mpz[j], SPEX_2D(x,jnew,k,mpz));

                if (jnew > 0)
                {
                    // Divide by rhos[jnew-1]
                    SPEX_MPZ_DIVEXACT(SPEX_2D(x,i,k,mpz),
                                  SPEX_2D(x,i,k,mpz),
                                  SPEX_1D(rhos, jnew-1, mpz));
                }
                // Update history
                SPEX_2D(h,i,k,int64) = jnew;
            }
            hx = SPEX_2D(h, i, k, int64);
            // History update to finalize x[i]
            if (hx < i-1)
            {
                // x[i] = x[i] * rhos[i-1]
                SPEX_MPZ_MUL(SPEX_2D(x, i, k, mpz),
                             SPEX_2D(x, i, k, mpz),
                             SPEX_1D(rhos, i-1, mpz));
                // x[i] = x[i] / rhos[hx]
                if (hx > -1)
                {
                    SPEX_MPZ_DIVEXACT(SPEX_2D(x, i, k, mpz),
                                      SPEX_2D(x, i, k, mpz),
                                      SPEX_1D(rhos, hx, mpz));
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // Free h memory
    //--------------------------------------------------------------------------

    SPEX_FREE_ALL;
    return SPEX_OK;
}

