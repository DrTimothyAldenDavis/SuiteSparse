//------------------------------------------------------------------------------
// Mongoose/MATLAB/mex_util/mex_get_graph.cpp
//------------------------------------------------------------------------------

// Mongoose Graph Partitioning Library, Copyright (C) 2017-2018,
// Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
// Mongoose is licensed under Version 3 of the GNU General Public License.
// Mongoose is also available under other licenses; contact authors for details.
// SPDX-License-Identifier: GPL-3.0-only

//------------------------------------------------------------------------------

#include "mongoose_mex.hpp"

namespace Mongoose
{

EdgeCutProblem *mex_get_graph
(
    const mxArray *Gmatlab, /* The sparse matrix            */
    const mxArray *Amatlab  /* The real-valued vertex weights */
)
{
    // Check for valid sparse matrix
    cs_mex_check (0, -1, -1, 1, 1, 1, Gmatlab) ;

    Int n = mxGetN(Gmatlab);
    Int *Gp = (Int*) mxGetJc(Gmatlab);
    Int *Gi = (Int*) mxGetIr(Gmatlab);
    double *Gx = (double*) mxGetPr(Gmatlab);
    double *Gw = (Amatlab) ? (double*) mxGetPr(Amatlab) : NULL;
    Int nz = Gp[n];
    
    EdgeCutProblem *returner = EdgeCutProblem::create(n, nz, Gp, Gi, Gx, Gw);

    if (!returner)
    {
        return NULL;
    }

    return returner;
}

}
