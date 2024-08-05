//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_lu_analyze: symbolic ordering and analysis for sparse LU
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the symbolic ordering for unsymmetric
 * matrices.  Currently, there are three options: user-defined order, COLAMD,
 * or AMD.
 *
 * Input/output arguments:
 *
 * S:       Symbolic analysis struct. Undefined on input; contains column
 *          permutation and estimates of nnz(L) and nnz(U) nnz on output
 *
 * A:       Input matrix, unmodified on input/output
 *
 * option:  option->order tells the function which ordering scheme to use
 *
 */

#define SPEX_FREE_ALL                           \
{                                               \
    SPEX_symbolic_analysis_free (&S, option);   \
}

#include "spex_lu_internal.h"

SPEX_info SPEX_lu_analyze
(
    SPEX_symbolic_analysis *S_handle,   // symbolic analysis including
                                 // column perm. and nnz of L and U
    const SPEX_matrix A,         // Input matrix
    const SPEX_options option    // Control parameters, if NULL, use default
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!spex_initialized()) return SPEX_PANIC;
    
    // get option->algo, or use SPEX_ALGORITHM_DEFAULT if option is NULL:
    SPEX_factorization_algorithm algo = SPEX_OPTION_ALGORITHM(option);
    if (algo != SPEX_ALGORITHM_DEFAULT && algo != SPEX_LU_LEFT)
    {
        return SPEX_INCORRECT_ALGORITHM;
    }
    
    SPEX_info info ;

    // A can have any data type, but must be in sparse CSC format
    SPEX_REQUIRE_KIND (A, SPEX_CSC);

    if (!S_handle)
    {
        return SPEX_INCORRECT_INPUT;
    }
    (*S_handle) = NULL;
    if (A->n != A->m)
    {
        return SPEX_INCORRECT_INPUT;
    }

    //--------------------------------------------------------------------------
    // allocate symbolic analysis object
    //--------------------------------------------------------------------------

    SPEX_symbolic_analysis S = NULL ;
    int64_t i, n = A->n, anz;
    // SPEX enviroment is checked to be init'ed and A is checked to be not NULL
    // and a SPEX_CSC kind, so there shouldnt be any error from this function
    SPEX_matrix_nnz(&anz, A, option);

    // ALlocate memory for S
    S = (SPEX_symbolic_analysis) SPEX_calloc(1,
        sizeof(SPEX_symbolic_analysis_struct));
    if (S == NULL) {return SPEX_OUT_OF_MEMORY;}
    S->kind = SPEX_LU_FACTORIZATION;

    //--------------------------------------------------------------------------
    // order the matrix and estimate the # of entries in L and U
    //--------------------------------------------------------------------------

    // Get option->order to determine which ordering to use.
    SPEX_preorder order = SPEX_OPTION_ORDER (option);
    switch(order)
    {
        default:
        case SPEX_DEFAULT_ORDERING:
        case SPEX_COLAMD:
        // ---COLAMD ordering is used (DEFAULT)---
        // S->q is set to COLAMD's column ordering on A.
        {
            SPEX_CHECK( spex_colamd(&(S->Q_perm),&(S->unz),A,option));
            S->lnz = S->unz;
        }
        break;

        case SPEX_NO_ORDERING:
        // ---No ordering is used---
        // S->q is set to [0 ... n] and the number of nonzeros in L and U are
        // estimated to be 10 times the number of nonzeros in A.  This is a
        // very crude estimate on nnz(L) and nnz (U).
        {
            S->Q_perm = (int64_t*)SPEX_malloc( (n+1)*sizeof(int64_t) );
            if (S->Q_perm == NULL)
            {
                // out of memory
                SPEX_FREE_ALL;
                return (SPEX_OUT_OF_MEMORY);
            }
            for (i = 0; i < n+1; i++)
            {
                S->Q_perm[i] = i;
            }
            // Very crude estimate for number of L and U nonzeros
            S->lnz = S->unz = 10*anz;
        }
        break;

        case SPEX_AMD:
        // --- AMD ordering is used
        // S->q is set as AMD's ordering.
        {
            SPEX_CHECK( spex_amd(&(S->Q_perm),&(S->unz),A,option));
            S->lnz = S->unz;
        }
        break;
    }

    //--------------------------------------------------------------------------
    // Make sure appropriate space is allocated. It's possible to return
    // estimates which exceed the dimension of L and U or estimates which are
    // too small for L U. In this case, this block of code ensures that the
    // estimates on nnz(L) and nnz(U) are at least n and no more than n*n.
    //--------------------------------------------------------------------------
    // estimate exceeds max number of nnz in A
    if (S->lnz > (double) n*n)
    {
        int64_t nnz = ceil(0.5*n*n);
        S->lnz = S->unz = nnz;
    }
    // If estimate < n, first column of triangular solve may fail
    if (S->lnz < n)
    {
        S->lnz = S->lnz + n;
    }
    if (S->unz < n)
    {
        S->unz = S->unz + n;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*S_handle) = S ;
    return SPEX_OK;
}

