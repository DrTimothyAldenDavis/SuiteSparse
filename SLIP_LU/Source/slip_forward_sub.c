//------------------------------------------------------------------------------
// SLIP_LU/slip_forward_sub: sparse forward substitution (x = (LD)\x)
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function performs sparse roundoff-error-free (REF) forward
 * substitution This is essentially the same as the sparse REF triangular solve
 * applied to each column of the right hand side vectors. Like the normal one,
 * this function expects that the matrix x is dense. As a result,the nonzero
 * pattern is not computed and each nonzero in x is iterated across.  The
 * system to solve is L*D*x_output = x_input, overwriting the right-hand-side
 * with the solution.
 *
 * On output, the SLIP_matrix* x structure is modified.
 */

#define SLIP_FREE_ALL            \
    SLIP_matrix_free(&h, NULL)  ;

#include "slip_internal.h"

SLIP_info slip_forward_sub
(
    const SLIP_matrix *L,   // lower triangular matrix
    SLIP_matrix *x,         // right hand side matrix of size n*numRHS
    const SLIP_matrix *rhos // sequence of pivots used in factorization
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SLIP_info info ;
    SLIP_REQUIRE(L, SLIP_CSC, SLIP_MPZ);
    SLIP_REQUIRE(x, SLIP_DENSE, SLIP_MPZ);
    SLIP_REQUIRE(rhos, SLIP_DENSE, SLIP_MPZ);

    //--------------------------------------------------------------------------

    int64_t i, hx, k, j, jnew;
    int sgn ;

    // Build the history matrix
    SLIP_matrix *h;
    SLIP_CHECK (SLIP_matrix_allocate(&h, SLIP_DENSE, SLIP_INT64, x->m, x->n,
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
            hx = SLIP_2D(h, i, k, int64);
            // If x[i][k] = 0, can skip operations and continue to next i
            SLIP_CHECK(SLIP_mpz_sgn(&sgn, SLIP_2D(x, i, k, mpz)));
            if (sgn == 0) {continue;}

            //------------------------------------------------------------------
            // History Update
            //------------------------------------------------------------------

            if (hx < i-1)
            {
                // x[i] = x[i] * rhos[i-1]
                SLIP_CHECK(SLIP_mpz_mul( SLIP_2D(x, i, k, mpz),
                                         SLIP_2D(x, i, k, mpz),
                                         SLIP_1D(rhos, i-1, mpz)));
                // x[i] = x[i] / rhos[hx]
                if (hx > -1)
                {
                    SLIP_CHECK(SLIP_mpz_divexact( SLIP_2D(x, i, k, mpz),
                                                  SLIP_2D(x, i, k, mpz),
                                                  SLIP_1D(rhos, hx, mpz)));
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
                SLIP_CHECK(SLIP_mpz_sgn(&sgn, L->x.mpz[j]));
                if (sgn == 0) {continue;}

                // j > i
                if (jnew > i)
                {
                    // check if x[jnew] is zero
                    SLIP_CHECK(SLIP_mpz_sgn(&sgn, SLIP_2D(x, jnew, k, mpz)));
                    if (sgn == 0)
                    {
                        // x[j] = x[j] - lji xi
                        SLIP_CHECK(SLIP_mpz_submul(SLIP_2D(x, jnew, k, mpz),
                                                   SLIP_1D(L, j, mpz),
                                                   SLIP_2D(x, i, k, mpz)));
                        // x[j] = x[j] / rhos[i-1]
                        if (i > 0)
                        {
                            SLIP_CHECK(
                                SLIP_mpz_divexact(SLIP_2D(x, jnew, k, mpz),
                                                  SLIP_2D(x, jnew, k, mpz),
                                                  SLIP_1D(rhos, i-1, mpz)));
                        }
                    }
                    else
                    {
                        hx = SLIP_2D(h, jnew, k, int64);
                        // History update if necessary
                        if (hx < i-1)
                        {
                            // x[j] = x[j] * rhos[i-1]
                            SLIP_CHECK(SLIP_mpz_mul(SLIP_2D(x, jnew, k, mpz),
                                                    SLIP_2D(x, jnew, k, mpz),
                                                    SLIP_1D(rhos, i-1, mpz)));
                            // x[j] = x[j] / rhos[hx]
                            if (hx > -1)
                            {
                                SLIP_CHECK(
                                    SLIP_mpz_divexact(SLIP_2D(x, jnew, k, mpz),
                                                      SLIP_2D(x, jnew, k, mpz),
                                                      SLIP_1D(rhos, hx, mpz)));
                            }
                        }
                        // x[j] = x[j] * rhos[i]
                        SLIP_CHECK(SLIP_mpz_mul(SLIP_2D(x, jnew, k, mpz),
                                                SLIP_2D(x, jnew, k, mpz),
                                                SLIP_1D(rhos, i, mpz)));
                        // x[j] = x[j] - lmi xi
                        SLIP_CHECK(SLIP_mpz_submul(SLIP_2D(x, jnew, k, mpz),
                                                   SLIP_1D(L, j, mpz),
                                                   SLIP_2D(x, i, k, mpz)));
                        // x[j] = x[j] / rhos[i-1]
                        if (i > 0)
                        {
                            SLIP_CHECK(
                                SLIP_mpz_divexact(SLIP_2D(x, jnew, k, mpz),
                                                  SLIP_2D(x, jnew, k, mpz),
                                                  SLIP_1D(rhos, i-1, mpz)));
                        }
                    }
                    //h[jnew][k] = i;
                    SLIP_2D(h, jnew, k, int64) = i;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // Free h memory
    //--------------------------------------------------------------------------

    SLIP_FREE_ALL;
    return SLIP_OK;
}

