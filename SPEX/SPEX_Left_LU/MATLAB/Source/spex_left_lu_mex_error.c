//------------------------------------------------------------------------------
// SPEX_Left_LU/MATLAB/spex_left_lu_mex_error: Return error messages to matlab
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function prints error messages for MATLAB for debugging*/

#include "SPEX_Left_LU_mex.h"

void spex_left_lu_mex_error
(
    SPEX_info status,
    char *message
)
{
    
    switch (status)
    {
        case SPEX_OK :                   // all is well
            return ;

        case SPEX_OUT_OF_MEMORY :        // out of memory
            SPEX_finalize ( ) ;
            mexErrMsgTxt ("out of memory") ;

        case SPEX_SINGULAR :             // the input matrix A is singular
            SPEX_finalize ( ) ;
            mexErrMsgTxt ("input matrix is singular") ;

        case SPEX_INCORRECT_INPUT :      // one or more input arguments are incorrect
            SPEX_finalize ( ) ;
            mexErrMsgTxt ("invalid inputs") ;

        case SPEX_INCORRECT :            // The solution is incorrect
            SPEX_finalize ( ) ;
            mexErrMsgTxt ("result invalid") ;

        case SPEX_PANIC :                // SPEX_Left_LU used without proper initialization
            SPEX_finalize ( ) ;
            mexErrMsgTxt ("panic") ;

        default : 
            SPEX_finalize ( ) ;
            mexErrMsgTxt (message) ;
    }
}

