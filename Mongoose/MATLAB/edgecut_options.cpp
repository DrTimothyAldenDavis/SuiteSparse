//------------------------------------------------------------------------------
// Mongoose/MATLAB/edgecut_options.cpp
//------------------------------------------------------------------------------

// Mongoose Graph Partitioning Library, Copyright (C) 2017-2018,
// Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
// Mongoose is licensed under Version 3 of the GNU General Public License.
// Mongoose is also available under other licenses; contact authors for details.
// SPDX-License-Identifier: GPL-3.0-only

//------------------------------------------------------------------------------

#include "mongoose_mex.hpp"

using namespace Mongoose;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    const char* usage = "Usage: O = edgecut_options()";
    if(nargout > 1 || nargin != 0) mexErrMsgTxt(usage);

    EdgeCut_Options *ret = EdgeCut_Options::create();
    if(ret == NULL) mexErrMsgTxt("Out of memory");

    pargout[0] = mex_put_options(ret);
    ret->~EdgeCut_Options();
}
