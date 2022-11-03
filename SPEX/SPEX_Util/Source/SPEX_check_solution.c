//------------------------------------------------------------------------------
// SPEX_Util/spex_check_solution: check solution to Ax=b
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: Given a solution vector x, check the solution of the linear system
 * Ax = b. This is done by computing a rational-arthmetic A*x == b. This function
 * is only used for debugging purposes, as the routines within SPEX are gauranteed
 * to be exact.
 */

#define SPEX_FREE_ALL                       \
    SPEX_MPQ_CLEAR(temp);                   \
    SPEX_matrix_free(&b2, NULL);

#include "spex_util_internal.h"

SPEX_info SPEX_check_solution
(
    const SPEX_matrix *A,         // Input matrix
    const SPEX_matrix *x,         // Solution vectors
    const SPEX_matrix *b,         // Right hand side vectors
    const SPEX_options* option    // Command options
)
{
    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    //--------------------------------------------------------------------------
    // check inputs. Input are also checked by the two callers
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ) ;
    SPEX_REQUIRE (x, SPEX_DENSE, SPEX_MPQ) ;
    SPEX_REQUIRE (b, SPEX_DENSE, SPEX_MPZ) ;

    //--------------------------------------------------------------------------
    // Declare vars
    //--------------------------------------------------------------------------

    int64_t p, j, i ;
    SPEX_matrix *b2 = NULL;   // b2 stores the solution of A*x
    mpq_t temp; SPEX_MPQ_SET_NULL(temp);

    SPEX_CHECK (SPEX_mpq_init(temp));
    SPEX_CHECK (SPEX_matrix_allocate(&b2, SPEX_DENSE, SPEX_MPQ, b->m, b->n,
        b->nzmax, false, true, option));

    //--------------------------------------------------------------------------
    // perform SPEX_mpq_addmul in loops to compute b2 = A*x
    //--------------------------------------------------------------------------

    for (j = 0; j < b->n; j++)
    {
        for (i = 0; i < b->m; i++)
        {
            for (p = A->p[i]; p < A->p[i + 1]; p++)
            {
                // temp = A[p][i] (note this must be done seperately since A is
                // mpz and temp is mpq)
                SPEX_CHECK(SPEX_mpq_set_z(temp, A->x.mpz[p]));

                // temp = temp*x[i]
                SPEX_CHECK(SPEX_mpq_mul(temp, temp,
                                        SPEX_2D(x, i, j, mpq)));

                // b2[p] = b2[p]-temp
                SPEX_CHECK(SPEX_mpq_add(SPEX_2D(b2, A->i[p], j, mpq),
                                        SPEX_2D(b2, A->i[p], j, mpq),temp));
            }
        }
    }

    //--------------------------------------------------------------------------
    // check if b==b2
    //--------------------------------------------------------------------------

    for (j = 0; j < b->n; j++)
    {
        for (i = 0; i < b->m; i++)
        {
            // temp = b[i] (correct b)
            SPEX_CHECK(SPEX_mpq_set_z(temp, SPEX_2D(b, i, j, mpz)));

            // set check false if b!=b2
            int r ;
            SPEX_CHECK(SPEX_mpq_equal(&r, temp, SPEX_2D(b2, i, j, mpq)));
            if (r == 0)
            {
                info = SPEX_INCORRECT;
                j = b->n;
                break;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Print info
    //--------------------------------------------------------------------------

    int pr = SPEX_OPTION_PRINT_LEVEL (option) ;
    if (info == SPEX_OK)
    {
        SPEX_PR1 ("Solution is verified to be exact.\n") ;
    }
    else if (info == SPEX_INCORRECT)
    {
        // This can never happen.
        SPEX_PR1 ("ERROR! Solution is wrong. This is a bug; please "
                  "contact the authors of SPEX.\n") ;
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    SPEX_FREE_ALL;
    return info;
}

