//------------------------------------------------------------------------------
// Demo/spex_check_solution: checks whether a given vector exactly solves Ax=b
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


#include "spex_demos.h"

#undef  FREE_WORKSPACE
#define FREE_WORKSPACE                      \
{                                           \
    SPEX_MPQ_CLEAR(temp);                   \
    SPEX_MPQ_CLEAR(scale);                  \
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
    if (!SPEX_initialize ( )) return (SPEX_PANIC);

    //--------------------------------------------------------------------------
    // check inputs. Input are also checked by the two callers
    //--------------------------------------------------------------------------

    SPEX_info info ;
    /*SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ);
    SPEX_REQUIRE (x, SPEX_DENSE, SPEX_MPQ);
    SPEX_REQUIRE (b, SPEX_DENSE, SPEX_MPZ);*/

    //--------------------------------------------------------------------------
    // Declare vars
    //--------------------------------------------------------------------------

    int64_t p, j, i, nz;
    SPEX_matrix b2 = NULL;   // b2 stores the solution of A*x
    mpq_t temp; SPEX_MPQ_SET_NULL(temp);
    mpq_t scale; SPEX_MPQ_SET_NULL(scale);

    SPEX_MPQ_INIT(temp);
    SPEX_MPQ_INIT(scale);
    SPEX_CHECK (SPEX_matrix_allocate(&b2, SPEX_DENSE, SPEX_MPQ, b->m, b->n,
        b->nzmax, false, true, option));


    //--------------------------------------------------------------------------
    // perform SPEX_mpq_addmul in loops to compute b2 = A'*x, where A' is the
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
                SPEX_MPQ_SET_Z(temp, A->x.mpz[p]);

                // temp = temp*x[i]
                SPEX_MPQ_MUL(temp, temp,
                                        SPEX_2D(x, i, j, mpq));

                // b2[p] = b2[p]-temp
                SPEX_MPQ_ADD(SPEX_2D(b2, A->i[p], j, mpq),
                                        SPEX_2D(b2, A->i[p], j, mpq),temp);
            }
        }
    }

    //--------------------------------------------------------------------------
    // Apply scales of A and b to b2 before comparing the b2 with scaled b'
    //--------------------------------------------------------------------------

    SPEX_MPQ_DIV(scale, b->scale, A->scale);

    // Apply scaling factor, but ONLY if it is not 1
    int r;
    SPEX_MPQ_CMP_UI(&r, scale, 1, 1);
    if (r != 0)
    {
        nz = x->m * x->n;
        for (i = 0; i < nz; i++)
        {
            SPEX_MPQ_MUL(b2->x.mpq[i], b2->x.mpq[i], scale);
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
            SPEX_MPQ_SET_Z(temp, SPEX_2D(b, i, j, mpz));

            // set check false if b!=b2
            SPEX_MPQ_EQUAL(&r, temp, SPEX_2D(b2, i, j, mpq));
            if (r == 0)
            {
                //printf("ERROR\n");
                info = SPEX_PANIC;
                j = b->n;
                break;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Print info
    //--------------------------------------------------------------------------

    int pr = SPEX_OPTION_PRINT_LEVEL (option);
    if (info == SPEX_OK)
    {
        SPEX_PR1 ("Solution is verified to be exact.\n");
    }
    else if (info == SPEX_PANIC)
    {
        // This can never happen.
        SPEX_PR1 ("ERROR! Solution is wrong. This is a bug; please "
                  "contact the authors of SPEX.\n");
    }

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE ;
    return info;
}
