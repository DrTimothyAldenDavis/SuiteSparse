//------------------------------------------------------------------------------
// SPEX_QR/spex_qr_preorder: symbolic ordering for QR
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2026, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the symbolic ordering for SPEX QR.
 * Currently, there are three options: user-defined order, COLAMD, or AMD.
 * It is *highly* recommended that COLAMD is used for QR factorization.
 *
 * Input/output arguments:
 *
 * S:       Symbolic analysis struct. Undefined on input; contains column
 *          permutation and estimates of nnz on output (is the exact
 *          number of nonzeros if AMD is used)
 *
 * A:       Input matrix, unmodified on input/output
 *
 * option:  option->order tells the function which ordering scheme to use
 */


#define SPEX_FREE_ALL                           \
{                                               \
    SPEX_FREE_WORKSPACE ;                       \
    SPEX_symbolic_analysis_free(&S, option);    \
}

#include "spex_qr_internal.h"

SPEX_info spex_qr_preorder
(
    // Output
    SPEX_symbolic_analysis *S_handle,   // Symbolic analysis data structure
                                        // On input: undefined
                                        // On output: contains the
                                        // row/column permutation and its
                                        // inverse.
    // Input
    const SPEX_matrix A,            // Input matrix
    const SPEX_options option       // Control parameters (use default if NULL)
)
{

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    if ( !spex_initialized() ) return SPEX_PANIC;

    // All inputs have been checked by the caller so ASSERTS are used instead
    // of ifs A can have any data type, but must be in sparse CSC format
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);

    // m >= n for thin QR factorization
    ASSERT(A->n <= A->m);

    // Dimension can't be negative
    ASSERT(A->n >= 0);

    (*S_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Allocate symbolic analysis object
    //--------------------------------------------------------------------------

    SPEX_symbolic_analysis S = NULL;

    // declare indices and dimension of matrix
    int64_t i, k, index, n = A->n;

    int64_t anz; // Number of nonzeros in A
    SPEX_CHECK (SPEX_matrix_nnz(&anz, A, option));

    // Allocate memory for S
    S = (SPEX_symbolic_analysis)
        SPEX_calloc(1, sizeof(SPEX_symbolic_analysis_struct));
    if (S == NULL)
    {
        SPEX_FREE_ALL;
        return (SPEX_OUT_OF_MEMORY);
    }

    S->kind = SPEX_QR_FACTORIZATION ;

    //Check which ordering to use.
    SPEX_preorder order = SPEX_OPTION_ORDER(option);
    switch(order)
    {
        default:
        case SPEX_DEFAULT_ORDERING:
        case SPEX_COLAMD:
        // --- COLAMD ordering is used (DEFAULT)
        // S->q is set as COLAMD's column ordering.
        {
            SPEX_CHECK( spex_colamd(&(S->Q_perm),&(S->rnz),A,option));
        }
        break;

        case SPEX_AMD:
        // ---AMD ordering is used---
        // S->q is set to AMD's column ordering on ATA.
        // The number of nonzeros in R is given as AMD's computed
        // number of nonzeros in the Cholesky factor L of ATA 
        {
            SPEX_CHECK( spex_amd(&(S->Q_perm),&(S->rnz),A,option));
        }
        break;

        case SPEX_NO_ORDERING:
        // ---No ordering is used---
        // S->q is set to [0 ... n] and the number of nonzeros in R is estimated
        // to be 10 times the number of nonzeros in A.
        // This is a very crude estimate on the nnz(R)
        {
            S->Q_perm = (int64_t*)SPEX_malloc( (n+1)*sizeof(int64_t) );
            if (S->Q_perm == NULL)
            {
                SPEX_FREE_ALL;
                return (SPEX_OUT_OF_MEMORY);
            }

            for (i = 0; i < n+1; i++)
            {
                S->Q_perm[i] = i;
            }
            // Very crude estimate for number of R nonzeros
            S->rnz = 10*anz;
        }
        break;
    }

    //--------------------------------------------------------------------------
    // Make sure appropriate space is allocated. It is possible to return
    // estimates which exceed the dimension of R or estimates which are
    // too small for R. In this case, this block of code ensures that the
    // estimates on nnz(R) are at least n and no more than n*n.
    //--------------------------------------------------------------------------
    
    // estimate exceeds max number of nnz in A
    if (S->rnz > (double) n*n)
    {
        int64_t nnz = ceil(0.5*n*n);
        S->rnz =  nnz;
    }
    // If estimate < n, it is possible that the first iteration of triangular
    // solve may fail, so we make sure that the estimate is at least n
    if (S->rnz < n) S->rnz += n;

    // Allocate pinv
    S->Pinv_perm = (int64_t*)SPEX_calloc(n, sizeof(int64_t));
    if(!(S->Pinv_perm))
    {
        SPEX_FREE_ALL;
        return (SPEX_OUT_OF_MEMORY);
    }

    // Populate pinv
    for (k = 0; k < n; k++)
    {
        index = S->Q_perm[k];
        S->Pinv_perm[index] = k;
    }

    //--------------------------------------------------------------------------
    // Set result, report success
    //--------------------------------------------------------------------------

    SPEX_FREE_WORKSPACE ;
    (*S_handle) = S;
    return SPEX_OK;
}
