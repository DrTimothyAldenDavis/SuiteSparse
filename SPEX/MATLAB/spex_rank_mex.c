//------------------------------------------------------------------------------
// SPEX/MATLAB/spex_rank_mex: Use SPEX QR within MATLAB
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2026, Chris Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: The .c file defining the SPEX QR MATLAB interfacee
 * This function defines: x = spex_qr_mex_soln (A, b, option)
 */


#include "SPEX_mex.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    //--------------------------------------------------------------------------
    // Initialize SPEX QR library environment
    //--------------------------------------------------------------------------

    SPEX_info status ;
    SPEX_MEX_OK (SPEX_initialize_expert
            (mxMalloc, mxCalloc, mxRealloc, mxFree));

    SuiteSparse_config_printf_func_set (mexPrintf);

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    if (nargout > 1 || nargin > 1 )
    {
        spex_mex_error (1, "Usage: r = spex_rank (A)");
    }

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (mxIsComplex (pargin [0]))
    {
        spex_mex_error (1, "inputs must be real");
    }
    /***/
    if (!mxIsSparse (pargin [0]))     // Is the matrix sparse?
    {
        spex_mex_error (1, "first input must be sparse");
    }


    //--------------------------------------------------------------------------
    // set options
    //--------------------------------------------------------------------------

    SPEX_options option = NULL;
    SPEX_create_default_options(&option);
    if (option == NULL)
    {
        spex_mex_error (SPEX_OUT_OF_MEMORY, "");
    }

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    SPEX_matrix A = NULL ;
    spex_mex_get_A (&A, pargin, option);
/**/
    if (option->print_level > 0)
    {
        printf ("\nScaled integer input matrix A:\n");
        SPEX_matrix_check (A, option);
    }

    //--------------------------------------------------------------------------
    // r = rank(A) using SPEX QR or SPEX LU
    //--------------------------------------------------------------------------

    int64_t r;
    SPEX_MEX_OK( SPEX_rank(&r, A, option));


    //--------------------------------------------------------------------------
    // return r to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateDoubleScalar ((double) r);
    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    SPEX_matrix_free (&A, option);
    SPEX_FREE (option);
    SPEX_finalize ( );
    /**/

}

