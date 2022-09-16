//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/sparse2: MATLAB interface to CHOLMOD triplet-to-sparse method
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2022, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Identical to the "sparse" function in MATLAB, just faster.  */

#include "cholmod_matlab.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    sputil_sparse (nargout, pargout, nargin, pargin) ;
}
