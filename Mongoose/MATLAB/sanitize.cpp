//------------------------------------------------------------------------------
// Mongoose/MATLAB/sanitize.cpp
//------------------------------------------------------------------------------

// Mongoose Graph Partitioning Library, Copyright (C) 2017-2018,
// Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
// Mongoose is licensed under Version 3 of the GNU General Public License.
// Mongoose is also available under other licenses; contact authors for details.
// SPDX-License-Identifier: GPL-3.0-only

//------------------------------------------------------------------------------

#include "mongoose_mex.hpp"
#include "Mongoose_Sanitize.hpp"

using namespace Mongoose;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    const char* usage = "Usage: A_safe = sanitize(A, [makeBinary=false])";
    if (nargin < 1 || nargin > 2 || nargout > 1)
    {
        mexErrMsgTxt(usage);
    }

    bool makeBinary = false;
    if (nargin == 2)
    {
        makeBinary = (bool) mxGetScalar(pargin[1]);
    }

    const mxArray *A_matlab = pargin[0];
    cs *A = (cs *) SuiteSparse_malloc(1, sizeof(cs));
    A = cs_mex_get_sparse (A, 0, 1, A_matlab);

    cs *A_safe = sanitizeMatrix(A, false, makeBinary);
    A->p = NULL;
    A->i = NULL;
    A->x = NULL;
    cs_spfree(A);

    pargout[0] = cs_mex_put_sparse (&A_safe) ;
}
