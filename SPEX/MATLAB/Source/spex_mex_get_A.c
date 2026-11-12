//------------------------------------------------------------------------------
// SPEX/MATLAB/SPEX_mex_get_A.c: convert A to SPEX matrices
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2026, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function reads in the A matrix */

#include "SPEX_mex.h"

void spex_mex_get_A
(
    SPEX_matrix *A_handle,      // Internal SPEX Mat stored in CSC
    const mxArray* pargin[],    // The input A matrix
    SPEX_options option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!A_handle || !pargin)
    {
        spex_mex_error (SPEX_INCORRECT_INPUT, "");
    }
    (*A_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Declare variables
    //--------------------------------------------------------------------------

    SPEX_info status;
    int64_t nA, mA, Anz, k, j;
    int64_t *Ap, *Ai;
    double *Ax;

    //--------------------------------------------------------------------------
    // Read in A
    //--------------------------------------------------------------------------

    // Read in Ap, Ai, Ax
    Ap = (int64_t *) mxGetJc (pargin[0]);
    Ai = (int64_t *) mxGetIr (pargin[0]);
    Ax = mxGetDoubles (pargin[0]);

    if (!Ai || !Ap || !Ax)
    {
        spex_mex_error (SPEX_INCORRECT_INPUT, "");
    }

    // Get info about A
    nA = (int64_t) mxGetN (pargin[0]);
    mA = (int64_t) mxGetM (pargin[0]);
    Anz = Ap[nA];

    // check the values of A
    bool A_has_int64_values = spex_mex_check_for_inf (Ax, Anz);

    SPEX_matrix A = NULL;
    SPEX_matrix A_matlab = NULL;

    if (A_has_int64_values)
    {
        // All entries in A can be typecast to int64_t without change in value.
        int64_t *Ax_int64 = (int64_t*) SPEX_malloc (Anz* sizeof (int64_t));
        if (!Ax_int64)
        {
            spex_mex_error (SPEX_OUT_OF_MEMORY, "");
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
    SPEX_MEX_OK (SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, A_matlab, option));

    // free the shallow copy of A
    SPEX_MEX_OK (SPEX_matrix_free (&A_matlab, option));

    (*A_handle) = A;
}

