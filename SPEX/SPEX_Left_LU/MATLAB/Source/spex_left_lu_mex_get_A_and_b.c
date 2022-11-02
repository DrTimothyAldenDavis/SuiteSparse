//------------------------------------------------------------------------------
// SPEX_Left_LU/MATLAB/spex_left_lu_mex_get_A_and_b: Obtain user's A and b matrices
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function reads in the A matrix and right hand side vectors. */

#include "SPEX_Left_LU_mex.h"

void spex_left_lu_mex_get_A_and_b
(
    SPEX_matrix **A_handle,     // Internal SPEX Mat stored in CSC
    SPEX_matrix **b_handle,     // mpz matrix used internally
    const mxArray* pargin[],    // The input A matrix and options
    SPEX_options* option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!A_handle || !pargin)
    {
        spex_left_lu_mex_error (SPEX_INCORRECT_INPUT, "");
    }
    (*A_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Declare variables
    //--------------------------------------------------------------------------

    SPEX_info status;
    int64_t nA, mA, nb, mb, Anz, k, j;
    int64_t *Ap, *Ai;
    double *Ax, *bx;

    //--------------------------------------------------------------------------
    // Read in A
    //--------------------------------------------------------------------------

    // Read in Ap, Ai, Ax
    Ap = (int64_t *) mxGetJc (pargin[0]) ;
    Ai = (int64_t *) mxGetIr (pargin[0]) ;
    Ax = mxGetDoubles (pargin[0]) ;
    if (!Ai || !Ap || !Ax)
    {
        spex_left_lu_mex_error (SPEX_INCORRECT_INPUT, "") ;
    }

    // Get info about A
    nA = (int64_t) mxGetN (pargin[0]) ;
    mA = (int64_t) mxGetM (pargin[0]) ;
    Anz = Ap[nA];
    if (nA != mA)
    {
        spex_left_lu_mex_error (1, "A must be square") ;
    }

    // check the values of A
    bool A_has_int64_values = spex_left_lu_mex_check_for_inf (Ax, Anz) ;

    SPEX_matrix* A = NULL;
    SPEX_matrix* A_matlab = NULL;

    if (A_has_int64_values)
    {
        // All entries in A can be typecast to int64_t without change in value.
        int64_t *Ax_int64 = (int64_t*) SPEX_malloc (Anz* sizeof (int64_t)) ;
        if (!Ax_int64)
        {
            spex_left_lu_mex_error (SPEX_OUT_OF_MEMORY, "") ;
        }
        for (k = 0; k < Anz; k++)
        {
            // typecast the double Ax into the int64_t Ax_int64
            Ax_int64[k] = (int64_t) Ax[k];
        }

        // Create A_matlab (->x starts as shallow)
        SPEX_matrix_allocate (&A_matlab, SPEX_CSC, SPEX_INT64, mA,
            nA, Anz, true, false, option);

        // transplant A_matlab->x, which is no longer shallow
        A_matlab->x.int64 = Ax_int64;
        A_matlab->x_shallow = false ;

    }
    else
    {
        // Entries in A cannot be typecast to int64_t without changing them.
        // Create A_matlab (->x is shallow)
        SPEX_matrix_allocate (&A_matlab, SPEX_CSC, SPEX_FP64, mA,
            nA, Anz, true, false, option);
        A_matlab->x.fp64 = Ax;
    }

    // the pattern of A_matlab is always shallow
    A_matlab->p = Ap ;
    A_matlab->i = Ai ;

    // scale A and convert to MPZ
    SPEX_MEX_OK (SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, A_matlab, option)) ;

    // free the shallow copy of A
    SPEX_MEX_OK (SPEX_matrix_free (&A_matlab, option)) ;

    //--------------------------------------------------------------------------
    // Read in b
    //--------------------------------------------------------------------------

    SPEX_matrix* b = NULL;
    SPEX_matrix* b_matlab = NULL;

    bx = mxGetDoubles (pargin[1]) ;
    if (!bx)
    {
        spex_left_lu_mex_error (SPEX_INCORRECT_INPUT, "") ;
    }

    // Get info about RHS vector (s)
    nb = mxGetN (pargin[1]) ;
    mb = mxGetM (pargin[1]) ;
    if (mb != mA)
    {
        spex_left_lu_mex_error (1, "dimension mismatch") ;
    }

    int64_t count = 0;

    // check the values of b
    bool b_has_int64_values = spex_left_lu_mex_check_for_inf (bx, nb*mb) ;

    if (b_has_int64_values)
    {

        // Create b_matlab (which is shallow)
        SPEX_matrix_allocate(&b_matlab, SPEX_DENSE, SPEX_INT64, mb,
            nb, mb*nb, true, false, option);

        b_matlab->x.int64 = SPEX_calloc(nb*mb, sizeof(int64_t));
        for (int64_t j = 0; j < mb*nb; j++)
        {
            // typecast b from double to int64
            b_matlab->x.int64[j] = (int64_t) bx[j];
        }

    }
    else
    {

        // Create b_matlab (which is shallow)
        SPEX_matrix_allocate(&b_matlab, SPEX_DENSE, SPEX_FP64, mb,
                nb, mb*nb, true, false, option);

        b_matlab->x.fp64 = bx;
    }

    // scale b and convert to MPZ
    SPEX_MEX_OK (SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ, b_matlab, option)) ;

    // free the shallow copy of b
    SPEX_MEX_OK (SPEX_matrix_free (&b_matlab, option)) ;

    (*A_handle) = A;
    (*b_handle) = b;
}

