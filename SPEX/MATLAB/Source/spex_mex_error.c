//------------------------------------------------------------------------------
// SPEX/MATLAB/SPEX_mex_error.c: Check error codes for MATLAB
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function prints error messages for MATLAB for debugging */

#include "SPEX_mex.h"

void spex_mex_error
(
    SPEX_info status,
    char *message
)
{

    switch (status)
    {
        case SPEX_OK :                  // all is well
            return ;

        case SPEX_OUT_OF_MEMORY :       // out of memory
            SPEX_finalize ( );
            mexErrMsgTxt ("out of memory");

        case SPEX_SINGULAR :            // the input matrix A is singular
            SPEX_finalize ( );
            mexErrMsgTxt ("input matrix is singular");

        case SPEX_INCORRECT_INPUT :     // one or more input arguments are
                                        // incorrect
            SPEX_finalize ( );
            mexErrMsgTxt ("invalid inputs");
            
        case SPEX_NOTSPD :              // input matrix A is not SPD
            SPEX_finalize ( );
            mexErrMsgTxt ("input matrix is not symmetric positive definite");
            
        case SPEX_INCORRECT_ALGORITHM : // incorrect algorithm for function
            SPEX_finalize ( );
            mexErrMsgTxt ("incorrect algorithm");
            
        case SPEX_PANIC :               // SPEX used without proper
                                        // initialization
            SPEX_finalize ( );
            mexErrMsgTxt ("panic");
            
       case SPEX_ZERODIAG :             // input matrix A has a zero in
                                        // the diagonal
            SPEX_finalize ( );
            mexErrMsgTxt ("input matrix has a zero in diagonal");
            
        case SPEX_UNSYMMETRIC:          // input matrix A is unsymmetric
            SPEX_finalize ( );
            mexErrMsgTxt ("input matrix is unsymmetric");
            
        default :
            SPEX_finalize ( );
            mexErrMsgTxt (message);
    }
}

