//------------------------------------------------------------------------------
// SPEX_Backslash/SPEX_rank: compute the rank of sparse A
//------------------------------------------------------------------------------

// SPEX_Backslash: (c) 2019-2026, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function computes the exact rank of a sparse matrix A using
 * either LU or QR factorization. If A is square, a rank revealing form
 * of the SPEX LU factorization is computed. Conversely, if A is rectangular,
 * the SPEX QR factorization is used.
 *
 * Note that all information about the factorization is destroyed in this
 * function and thus if the goal is to find a solution to a rank deficient
 * linear system the SPEX QR software package should be used.
 *
 * Input/Output arguments:
 *
 * rank:        A pointer to the rank of A. Undefined on input and on output
 *              contains the rank of A
 *
 * A:           User's input matrix. It must be populated prior to calling this
 *              function.
 *
 * option:      Struct containing various command parameters for the
 *              factorization. If NULL on input, default values are used.
 */


#include "spex_util_internal.h"
#include "SPEX.h"

SPEX_info SPEX_rank
(
        // Output
        int64_t* rank,                  // rank of A
        // Input
        const SPEX_matrix A,            // Input matrix
        const SPEX_options option       // Command options
)
{

    //-------------------------------------------------------------------------
    // check inputs
    //-------------------------------------------------------------------------

    SPEX_info info ;
    if (!spex_initialized ( )) return (SPEX_PANIC);
    
    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    // Algorithm must be default, QR or LU.
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_LU_LEFT && algo != SPEX_QR_GS)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }

    // A must be CSC and MPZ
    SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ);

    // A must have appropriate dimensions
    if (A->m < 0 || A->n < 0)
    {
        return SPEX_INCORRECT_INPUT;
    }

    //-------------------------------------------------------------------------
    // compute rank
    //-------------------------------------------------------------------------

    if (A->m == A->n)
    {
        // A is square and thus the LU rank function should be utilized
        info = SPEX_lu_rank(rank, A, option);
    }
    else
    {
        // A is rectangular and thus we need to use a QR factorization
        info = SPEX_qr_rank(rank, A, option);
    }

    //--------------------------------------------------------------------------
    // Return success
    //--------------------------------------------------------------------------

    return (info);
}

