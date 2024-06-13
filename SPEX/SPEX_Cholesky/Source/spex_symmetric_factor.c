//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_factor: Wrapper for Cholesky/LDL factorization
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_ALL                    \
{                                        \
    SPEX_factorization_free(&F, option); \
}


#include "spex_cholesky_internal.h"

/* Purpose: Compute a symmetric factorization A = LDL'.
 * Only appropriate if A is symmetric with a nonzero diagonal F->kind must be
 * SPEX_CHOLESKY_FACTORIZATION or SPEX_LDL_FACTORIZATION If F->kind is
 * SPEX_CHOLESKY, A must be SPD, otherwise an error code is returned.  On input
 * A contains the user's matrix, option->algo indicates which factorization
 * algorithm is used; up-looking (default) or left-looking On output, L
 * contains the L factor of A, rhos contains the pivot elements and S contains
 * the elimination tree lower triangular matrix and rhos contains the pivots'
 * values used in the factorization.
 *
 * This function performs the either the integer preserving Cholesky or integer
 * preserving LDL factorization.  It allows either the left-looking or
 * up-looking factorization. In order to compute the L matrix, it performs n
 * iterations of a sparse REF symmetric triangular solve function. The overall
 * factorization is PAP' = LDL' The algorithms only differ by if its a Cholesky
 * or LDL factorization.  If it's Cholesky, diagonal elements must be >0. If
 * it's LDL they can be any nonzero.
 *
 * Importantly, this function assumes that A has already been permuted,
 *              and symbolically analyzed.
 *
 * Input arguments of the function:
 *
 * F_handle:    A handle to the factorization struct. Null on input.
 *              On output, contains a pointer to the factorization (this
 *              includes matrix L)
 *
 * S:           Symbolic analysis struct for Cholesky or LDL factorization.
 *              On input it contains the elimination tree and
 *              the number of nonzeros in L.
 *
 * A:           The user's permuted input matrix
 *
 * chol:        True if we are performing a Cholesky factorization
 *              and false if we are performing LDL
 *
 * option:      Command options. Notably, option->algo indicates whether
 *              it is performing a left-looking (SPEX_CHOL_LEFT or
 *              SPEX_LDL_LEFT), or up-looking factorization (SPEX_CHOL_UP or
 *              SPEX_LDL_UP).
 */

SPEX_info spex_symmetric_factor
(
    // Output
    SPEX_factorization *F_handle,   // Factorization struct
    //Input
    const SPEX_symbolic_analysis S, // Symbolic analysis struct containing the
                               // elimination tree of A, the column pointers of
                               // L, and the exact number of nonzeros of L.
    const SPEX_matrix A,       // Matrix to be factored
    bool chol,                 // If true we are attempting a Cholesky
                               // factorization only and thus the pivot
                               // elements must be >0 If false, we try a
                               // general LDL factorization with the pivot
                               // element strictly != 0.
    const SPEX_options option  // Command options
                               // Notably, option->algo indicates whether
                               // SPEX_CHOL_UP, SPEX_CHOL_LEFT, SPEX_LDL_UP, or
                               // SPEX_LDL_LEFT is used.
)
{

    SPEX_info info;

    SPEX_factorization F = NULL ;

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------
    // All inputs have been checked by the caller, asserts are used here as a
    // reminder of the appropriate data types
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);

    // Number of nonzeros in A
    int64_t anz;
    SPEX_CHECK( SPEX_matrix_nnz(&anz, A, option) );
    ASSERT(anz > 0);

    (*F_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    // Dimension of A
    int64_t n = A->n ;

    // Allocate memory for the factorization
    F = (SPEX_factorization) SPEX_calloc(1, sizeof(SPEX_factorization_struct));
    if (F == NULL) return SPEX_OUT_OF_MEMORY;

    // set factorization kind
    if (chol)
        F->kind = SPEX_CHOLESKY_FACTORIZATION;
    else
        F->kind = SPEX_LDL_FACTORIZATION;

    // Allocate and set scale_for_A
    SPEX_MPQ_INIT(F->scale_for_A);
    SPEX_MPQ_SET(F->scale_for_A, A->scale);

    // Inverse pivot ordering
    F->Pinv_perm = (int64_t*) SPEX_malloc ( n*sizeof(int64_t) );
    // row/column permutation, to be copied from S->P_perm
    F->P_perm =    (int64_t*) SPEX_malloc ( n*sizeof(int64_t) );
    if (!(F->Pinv_perm) || !(F->P_perm))
    {
        // out of memory: free everything and return
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }

    // Copy row/column permutation from symbolic analysis to factorization
    memcpy(F->P_perm, S->P_perm, n*sizeof(int64_t));
    memcpy(F->Pinv_perm, S->Pinv_perm, n*sizeof(int64_t));

    //--------------------------------------------------------------------------
    // factorization: up-looking or left-looking
    //--------------------------------------------------------------------------

    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);

    switch(algo)
    {
        default:
        case SPEX_ALGORITHM_DEFAULT:
        case SPEX_CHOL_UP:
        case SPEX_LDL_UP:
            SPEX_CHECK( spex_symmetric_up_factor(&(F->L), &(F->rhos), S, A, chol,
                option));
            break;

        case SPEX_CHOL_LEFT:
        case SPEX_LDL_LEFT:
            SPEX_CHECK( spex_symmetric_left_factor(&(F->L), &(F->rhos), S, A, chol,
                option) );
            break;
    }

    //--------------------------------------------------------------------------
    // Set outputs, return ok
    //--------------------------------------------------------------------------

    (*F_handle) = F ;
    return SPEX_OK;
}
