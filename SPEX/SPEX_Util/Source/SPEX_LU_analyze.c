//------------------------------------------------------------------------------
// SPEX_Util/SPEX_LU_analyze: symbolic ordering and analysis for sparse LU
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function performs the symbolic ordering for unsymmetric matrices.
 * Currently, there are three options: user-defined order, COLAMD, or AMD.
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

// SPEX_LU_analyze creates the SPEX_LU_analysis object S.  Use
// SPEX_LU_analysis_free to delete it.

#include "spex_util_internal.h"

SPEX_info SPEX_LU_analyze
(
    SPEX_LU_analysis** S_handle, // symbolic analysis (column perm. and nnz L,U)
    const SPEX_matrix *A,        // Input matrix
    const SPEX_options *option   // Control parameters, if NULL, use default
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    // A can have any data type, but must be in sparse CSC format
    SPEX_REQUIRE_KIND (A, SPEX_CSC) ;

    if (!S_handle || A->n != A->m)
    {
        return SPEX_INCORRECT_INPUT;
    }
    (*S_handle) = NULL ;

    //--------------------------------------------------------------------------
    // allocate symbolic analysis object
    //--------------------------------------------------------------------------

    SPEX_LU_analysis *S = NULL ;
    int64_t i, n = A->n, anz;
    // SPEX enviroment is checked to be init'ed and A is checked to be not NULL
    // and a SPEX_CSC kind, so there shouldnt be any error from this function
    SPEX_matrix_nnz(&anz, A, option);
    // ALlocate memory for S
    S = (SPEX_LU_analysis*) SPEX_malloc(sizeof(SPEX_LU_analysis));
    if (S == NULL) {return SPEX_OUT_OF_MEMORY;}

    // Allocate memory for column permutation
    S->q = (int64_t*) SPEX_malloc((n+1) * sizeof(int64_t));
    if (S->q == NULL)
    {
        SPEX_FREE(S);
        return SPEX_OUT_OF_MEMORY;
    }

    //--------------------------------------------------------------------------
    // No ordering is used. S->q is set to [0 ... n] and the number of nonzeros
    // in L and U is estimated to be 10 times the number of nonzeros in A. This
    // is a very crude estimate on the nnz(L) and nnz(U)
    //--------------------------------------------------------------------------

    SPEX_col_order order = SPEX_OPTION_ORDER (option) ;
    int pr = SPEX_OPTION_PRINT_LEVEL (option) ;

    if (order == SPEX_NO_ORDERING)
    {
        for (i = 0; i < n+1; i++)
        {
            S->q[i] = i;
        }
        // estimates for number of L and U nonzeros
        S->lnz = S->unz = 10*anz;
    }

    //--------------------------------------------------------------------------
    // The AMD ordering is used. S->q is set to AMD's column ordering on
    // A+A'. The number of nonzeros in L and U is given as AMD's computed
    // number of nonzeros in the Cholesky factor L of A+A'
    //--------------------------------------------------------------------------
    else if (order == SPEX_AMD)
    {
        double Control [AMD_CONTROL];           // Declare AMD control
        amd_l_defaults (Control) ;              // Set AMD defaults
        double Info [AMD_INFO];
        // Perform AMD
        amd_l_order(n, (int64_t *) A->p, (int64_t *) A->i,
            (int64_t *) S->q, Control, Info) ;
        S->lnz = S->unz = Info[AMD_LNZ];        // estimate for unz and lnz
        if (pr > 0)   // Output AMD info if desired
        {
            SPEX_PRINTF ("\n****Column Ordering Information****\n") ;
            amd_l_control (Control) ;
            amd_l_info (Info) ;
        }
    }

    //--------------------------------------------------------------------------
    // The COLAMD ordering is used. S->q is set as COLAMD's column ordering.
    // The number of nonzeros in L and U is set as 10 times the number of
    // nonzeros in A. This is a crude estimate.
    //--------------------------------------------------------------------------
    else
    {
        // Declared as per COLAMD documentation
        int64_t Alen = 2*anz + 6 *(n+1) + 6*(n+1) + n;
        int64_t* A2 = (int64_t*) SPEX_malloc(Alen* sizeof(int64_t));
        if (!A2)
        {
            // out of memory
            SPEX_LU_analysis_free (&S, option) ;
            return (SPEX_OUT_OF_MEMORY) ;
        }
        // Initialize S->q as per COLAMD documentation
        for (i = 0; i < n+1; i++)
        {
            S->q[i] = A->p[i];
        }
        // Initialize A2 per COLAMD documentation
        for (i = 0; i < anz; i++)
        {
            A2[i] = A->i[i];
        }
        int64_t stats [COLAMD_STATS];
        colamd_l (n, n, Alen, (int64_t *) A2,
            (int64_t *) S->q, (double *) NULL,
            (int64_t *) stats) ;
        // estimate for lnz and unz
        S->lnz = S->unz = 10*anz;

        // Print stats if desired
        if (pr > 0)
        {
            SPEX_PRINTF ("\n****Column Ordering Information****\n") ;
            colamd_l_report ((int64_t *) stats) ;
            SPEX_PRINTF ("\nEstimated L and U nonzeros: %" PRId64 "\n", S->lnz);
        }
        SPEX_FREE(A2);
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

