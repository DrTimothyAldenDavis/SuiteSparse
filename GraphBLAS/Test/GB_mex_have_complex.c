//------------------------------------------------------------------------------
// GB_mex_have_complex: determine if the 'double complex' type is available
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    // 'double complex' requires ANSI C11 or greater
    pargout [0] = mxCreateLogicalScalar (GxB_STDC_VERSION >= 201112L) ;
}

