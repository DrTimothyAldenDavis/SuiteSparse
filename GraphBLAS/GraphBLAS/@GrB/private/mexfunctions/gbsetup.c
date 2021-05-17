//------------------------------------------------------------------------------
// gbsetup: initialize or finalize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// gbsetup initializes GraphBLAS by calling GxB_init and by setting
// all GraphBLAS global variables to their MATLAB defaults.

#include "gb_matlab.h"

#define USAGE "GrB.setup"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // get test coverage
    //--------------------------------------------------------------------------

    #ifdef GBCOV
    gbcov_get ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // finalize GraphBLAS, if it is already started
    //--------------------------------------------------------------------------

    if (GB_Global_GrB_init_called_get ( ))
    {
        GrB_finalize ( ) ;
    }

    //--------------------------------------------------------------------------
    // allow GraphBLAS to be called again
    //--------------------------------------------------------------------------

    GB_Global_GrB_init_called_set (false) ;

    //--------------------------------------------------------------------------
    // initialize GraphBLAS
    //--------------------------------------------------------------------------

    gb_usage (true, USAGE) ;

    //--------------------------------------------------------------------------
    // save test coverage
    //--------------------------------------------------------------------------

    GB_WRAPUP ;
}

