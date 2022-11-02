//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_forward_sub: sparse forward substitution (x = (LD)\x)
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs sparse roundoff-error-free (REF) forward
 * substitution This is essentially the same as the sparse REF triangular solve
 * applied to each column of the right hand side vectors. Like the normal one,
 * this function expects that the matrix x is dense. As a result,the nonzero
 * pattern is not computed and each nonzero in x is iterated across.  The
 * system to solve is L*D*x_output = x_input, overwriting the right-hand-side
 * with the solution.
 *
 * On output, the SPEX_matrix* x structure is modified.
 */

#define SPEX_FREE_ALL            \
    SPEX_matrix_free(&h, NULL)  ;

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_forward_sub
(
    const SPEX_matrix *L,   // lower triangular matrix
    SPEX_matrix *x,         // right hand side matrix of size n*numRHS
    const SPEX_matrix *rhos // sequence of pivots used in factorization
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE(L, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(x, SPEX_DENSE, SPEX_MPZ);
    SPEX_REQUIRE(rhos, SPEX_DENSE, SPEX_MPZ);

    //--------------------------------------------------------------------------

    int64_t i, hx, k, j, jnew;
    int sgn ;

    // Build the history matrix
    SPEX_matrix *h;
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

        for (i = 0; i < x->m; i++)
        {
            hx = SPEX_2D(h, i, k, int64);
            // If x[i][k] = 0, can skip operations and continue to next i
            SPEX_CHECK(SPEX_mpz_sgn(&sgn, SPEX_2D(x, i, k, mpz)));
            if (sgn == 0) {continue;}

            //------------------------------------------------------------------
            // History Update
            //------------------------------------------------------------------

            if (hx < i-1)
            {
                // x[i] = x[i] * rhos[i-1]
                SPEX_CHECK(SPEX_mpz_mul( SPEX_2D(x, i, k, mpz),
                                         SPEX_2D(x, i, k, mpz),
                                         SPEX_1D(rhos, i-1, mpz)));
                // x[i] = x[i] / rhos[hx]
                if (hx > -1)
                {
                    SPEX_CHECK(SPEX_mpz_divexact( SPEX_2D(x, i, k, mpz),
                                                  SPEX_2D(x, i, k, mpz),
                                                  SPEX_1D(rhos, hx, mpz)));
                }
            }

            //------------------------------------------------------------------
            // IPGE updates
            //------------------------------------------------------------------

            // Access the Lji
            for (j = L->p[i]; j < L->p[i+1]; j++)
            {
                // Location of Lji
                jnew = L->i[j];

                // skip if Lx[j] is zero
                SPEX_CHECK(SPEX_mpz_sgn(&sgn, L->x.mpz[j]));
                if (sgn == 0) {continue;}

                // j > i
                if (jnew > i)
                {
                    // check if x[jnew] is zero
                    SPEX_CHECK(SPEX_mpz_sgn(&sgn, SPEX_2D(x, jnew, k, mpz)));
                    if (sgn == 0)
                    {
                        // x[j] = x[j] - lji xi
                        SPEX_CHECK(SPEX_mpz_submul(SPEX_2D(x, jnew, k, mpz),
                                                   SPEX_1D(L, j, mpz),
                                                   SPEX_2D(x, i, k, mpz)));
                        // x[j] = x[j] / rhos[i-1]
                        if (i > 0)
                        {
                            SPEX_CHECK(
                                SPEX_mpz_divexact(SPEX_2D(x, jnew, k, mpz),
                                                  SPEX_2D(x, jnew, k, mpz),
                                                  SPEX_1D(rhos, i-1, mpz)));
                        }
                    }
                    else
                    {
                        hx = SPEX_2D(h, jnew, k, int64);
                        // History update if necessary
                        if (hx < i-1)
                        {
                            // x[j] = x[j] * rhos[i-1]
                            SPEX_CHECK(SPEX_mpz_mul(SPEX_2D(x, jnew, k, mpz),
                                                    SPEX_2D(x, jnew, k, mpz),
                                                    SPEX_1D(rhos, i-1, mpz)));
                            // x[j] = x[j] / rhos[hx]
                            if (hx > -1)
                            {
                                SPEX_CHECK(
                                    SPEX_mpz_divexact(SPEX_2D(x, jnew, k, mpz),
                                                      SPEX_2D(x, jnew, k, mpz),
                                                      SPEX_1D(rhos, hx, mpz)));
                            }
                        }
                        // x[j] = x[j] * rhos[i]
                        SPEX_CHECK(SPEX_mpz_mul(SPEX_2D(x, jnew, k, mpz),
                                                SPEX_2D(x, jnew, k, mpz),
                                                SPEX_1D(rhos, i, mpz)));
                        // x[j] = x[j] - lmi xi
                        SPEX_CHECK(SPEX_mpz_submul(SPEX_2D(x, jnew, k, mpz),
                                                   SPEX_1D(L, j, mpz),
                                                   SPEX_2D(x, i, k, mpz)));
                        // x[j] = x[j] / rhos[i-1]
                        if (i > 0)
                        {
                            SPEX_CHECK(
                                SPEX_mpz_divexact(SPEX_2D(x, jnew, k, mpz),
                                                  SPEX_2D(x, jnew, k, mpz),
                                                  SPEX_1D(rhos, i-1, mpz)));
                        }
                    }
                    //h[jnew][k] = i;
                    SPEX_2D(h, jnew, k, int64) = i;
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

