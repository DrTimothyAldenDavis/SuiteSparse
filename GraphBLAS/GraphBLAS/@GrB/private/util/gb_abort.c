//------------------------------------------------------------------------------
// gb_abort: terminate a GraphBLAS function
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

void gb_abort ( void )                    // assertion failure
{
    mexErrMsgIdAndTxt ("GraphBLAS:assert", "assertion failed") ;
}

