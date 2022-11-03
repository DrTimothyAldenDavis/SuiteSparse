//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_back_sub: sparse REF backward substitution (x = U\x)
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs sparse REF backward substitution, solving
 * the system Ux = b. Note that prior to this, x is multiplied by
 * the determinant of A.
 *
 * U is a sparse mpz matrix, and bx is a dense mpz matrix.  The diagonal entry
 * of U must appear as the last entry in each column.
 *
 * The input argument bx contains b on input, and it is overwritten on output
 * by the solution x.
 */

#include "spex_left_lu_internal.h"

SPEX_info spex_left_lu_back_sub  // performs sparse REF backward substitution
(
    const SPEX_matrix *U,   // input upper triangular matrix
    SPEX_matrix *bx         // right hand side matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE (U,  SPEX_CSC,   SPEX_MPZ) ;
    SPEX_REQUIRE (bx, SPEX_DENSE, SPEX_MPZ) ;

    //--------------------------------------------------------------------------

    int sgn;
    mpz_t *Ux = U->x.mpz;
    int64_t *Ui = U->i;
    int64_t *Up = U->p;

    for (int64_t k = 0; k < bx->n; k++)
    {
        // Start at bx[n]
        for (int64_t j = U->n-1; j >= 0; j--)
        {
            // If bx[j] is zero skip this iteration
            SPEX_CHECK( SPEX_mpz_sgn( &sgn, SPEX_2D( bx, j, k, mpz)));
            if (sgn == 0) {continue;}

            // Obtain bx[j]
            SPEX_CHECK(SPEX_mpz_divexact( SPEX_2D(bx, j, k, mpz),
                                          SPEX_2D(bx, j, k, mpz),
                                          Ux[Up[j+1]-1]));
            for (int64_t i = Up[j]; i < Up[j+1]-1; i++)
            {
                SPEX_CHECK(SPEX_mpz_sgn(&sgn, Ux[i]));
                if (sgn == 0) {continue;}
                // bx[i] = bx[i] - Ux[i]*bx[j]
                SPEX_CHECK(SPEX_mpz_submul( SPEX_2D(bx, Ui[i], k, mpz),
                                            Ux[i], SPEX_2D(bx, j, k, mpz)));
            }
        }
    }

    return (SPEX_OK) ;
}

