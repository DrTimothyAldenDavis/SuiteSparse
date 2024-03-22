//------------------------------------------------------------------------------
// Demo/spex_check_solution: checks whether a given vector exactly solves Ax=b
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_demos.h"

#undef  FREE_WORKSPACE
#define FREE_WORKSPACE                      \
{                                           \
    SPEX_mpq_clear (temp) ;                 \
    SPEX_mpq_clear (scale) ;                \
    SPEX_matrix_free(&b2, NULL);            \
}

SPEX_info spex_demo_check_solution
(
    const SPEX_matrix A,         // Input matrix
    const SPEX_matrix x,         // Solution vectors
    const SPEX_matrix b,         // Right hand side vectors
    const SPEX_options option    // Command options
)
{

    //--------------------------------------------------------------------------
    // Declare vars
    //--------------------------------------------------------------------------

    mpq_t temp;
    mpq_t scale;
    int64_t p, j, i, nz;
    SPEX_matrix b2 = NULL;   // b2 stores the solution of A*x
    int r;
    SPEX_mpq_set_null (temp) ;
    SPEX_mpq_set_null (scale) ;

    //--------------------------------------------------------------------------
    // initialize
    //--------------------------------------------------------------------------

    SPEX_TRY (SPEX_mpq_init(temp));
    SPEX_TRY (SPEX_mpq_init(scale));
    SPEX_TRY (SPEX_matrix_allocate(&b2, SPEX_DENSE, SPEX_MPQ, b->m, b->n,
        b->nzmax, false, true, option));

    //--------------------------------------------------------------------------
    // perform SPEX_mpq_add,mul in loops to compute b2 = A'*x, where A' is the
    // scaled matrix with all entries in integer
    //--------------------------------------------------------------------------

    for (j = 0; j < b->n; j++)
    {
        for (i = 0; i < b->m; i++)
        {
            for (p = A->p[i]; p < A->p[i + 1]; p++)
            {
                // temp = A[p][i] (note this must be done seperately since A is
                // mpz and temp is mpq)
                SPEX_TRY (SPEX_mpq_set_z(temp, A->x.mpz[p]));

                // temp = temp*x[i]
                SPEX_TRY (SPEX_mpq_mul(temp, temp,
                                        SPEX_2D(x, i, j, mpq)));

                // b2[p] = b2[p]-temp
                SPEX_TRY (SPEX_mpq_add(SPEX_2D(b2, A->i[p], j, mpq),
                                        SPEX_2D(b2, A->i[p], j, mpq),temp));
            }
        }
    }

    //--------------------------------------------------------------------------
    // Apply scales of A and b to b2 before comparing the b2 with scaled b'
    //--------------------------------------------------------------------------

    SPEX_TRY (SPEX_mpq_div(scale, b->scale, A->scale));

    // Apply scaling factor, but ONLY if it is not 1
    SPEX_TRY (SPEX_mpq_cmp_ui(&r, scale, 1, 1));
    if (r != 0)
    {
        nz = x->m * x->n;
        for (i = 0; i < nz; i++)
        {
            SPEX_TRY (SPEX_mpq_mul(b2->x.mpq[i], b2->x.mpq[i], scale));
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
            SPEX_TRY (SPEX_mpq_set_z(temp, SPEX_2D(b, i, j, mpz)));

            // set check false if b!=b2
            SPEX_TRY (SPEX_mpq_equal(&r, temp, SPEX_2D(b2, i, j, mpq)));
            if (r == 0)
            {
                // This can "never" happen.
                fprintf (stderr, "ERROR! Solution is wrong. This is a bug;"
                    "please contact the authors of SPEX.\n");
                SPEX_TRY (SPEX_PANIC) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE ;
    return SPEX_OK ;
}

