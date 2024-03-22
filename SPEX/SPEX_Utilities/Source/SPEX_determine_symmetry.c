//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_determine_symmetry: Determine if given matrix is
// *numerically* (thus pattern-wise) symmetric
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: Determine if the input A is *numerically* (thus pattern-wise)
 * symmetric.  Since SPEX is an exact framework, it doesn't make sense to check
 * only pattern symmetry.
 *
 * If the matrix is determined to be symmetric, is_symmetric is returned as
 * true.
 */

#define SPEX_FREE_ALL               \
{                                   \
    SPEX_matrix_free(&T,NULL);      \
    SPEX_matrix_free(&R,NULL);      \
}

#include "spex_util_internal.h"

SPEX_info SPEX_determine_symmetry
(
    bool *is_symmetric,         // true if matrix is symmetric, false otherwise
    const SPEX_matrix A,        // Input matrix to be checked for symmetry
    const SPEX_options option   // Command options
)
{

    SPEX_info info;

    // Check for null pointers
    if (!A || !option || !is_symmetric)
    {
        return SPEX_INCORRECT_INPUT;
    }
    (*is_symmetric) = false ;

    // A must be CSC and mpz_t
    if (A->kind != SPEX_CSC || A->type != SPEX_MPZ)
    {
        return SPEX_INCORRECT_INPUT;
    }

    if (A->n != A->m)
    {
        // matrix is rectangular (and thus unsymmetric)
        return SPEX_OK ;
    }

    // Only used index
    int64_t j;

    // Declare matrices T and R. T = A' and R = T' = A''
    SPEX_matrix T = NULL, R = NULL ;
    // T = A'
    SPEX_CHECK( SPEX_transpose(&T, A, option) );

    // Check if the number of nonzeros in the columns
    // of A are equal to the number of nonzeros in
    // the rows of A. This is a quick check to
    // ensure the matrix is candidate to be symmetric.
    // Moreover, this check is important becuase
    // otherwise the ensuing block could seg-fault
    for (j = 0; j <= A->n; j++)
    {
        if (T->p[j] != A->p[j])
        {
            // Number of nonzeros in column k of A
            // is not the same as the number of nonzeros
            // in row k of A. So free all and exit
            // nnz( A(:,k)) != nnz( A(k,:))
            SPEX_FREE_ALL;
            return SPEX_OK ;
        }
    }

    // Set R = T'
    SPEX_CHECK( SPEX_transpose(&R, T, option) );

    // Check whether A[i][j] = A[j][i] in both pattern and numerics
    for (j = 0; j < R->p[R->n]; j++)
    {
        // Check pattern
        if (T->i[j] != R->i[j])
        {
            // Not pattern symmetric as row indices do not match
            SPEX_FREE_ALL;
            return SPEX_OK ;
        }
        // Check numerics
        int r;
        SPEX_MPZ_CMP(&r, R->x.mpz[j], T->x.mpz[j]);
        if (r != 0)
        {
            // Not numeric symmetric
            SPEX_FREE_ALL;
            return SPEX_OK ;
        }
    }

    // Free memory and return OK, and return is_symmetric as true
    SPEX_FREE_ALL;
    (*is_symmetric) = true ;
    return SPEX_OK;
}

