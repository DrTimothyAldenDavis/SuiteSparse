//------------------------------------------------------------------------------
// Mongoose/MATLAB/mex_util/mex_put_options.cpp
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

#define MEX_STRUCT_PUT(F)        addFieldWithValue(returner, #F, (double) O->F);

mxArray *mex_put_options
(
    const EdgeCut_Options *O
)
{
    mxArray *returner = mxCreateStructMatrix(1, 1, 0, NULL);

    MEX_STRUCT_PUT(random_seed);
    MEX_STRUCT_PUT(coarsen_limit);
    MEX_STRUCT_PUT(matching_strategy);
    MEX_STRUCT_PUT(do_community_matching);
    MEX_STRUCT_PUT(high_degree_threshold);
    
    /** Guess Partitioning Options *******************************************/
    MEX_STRUCT_PUT(initial_cut_type);

    /** Waterdance Options ***************************************************/
    MEX_STRUCT_PUT(num_dances);

    /**** Fidducia-Mattheyes Options *****************************************/
    MEX_STRUCT_PUT(use_FM);
    MEX_STRUCT_PUT(FM_search_depth);
    MEX_STRUCT_PUT(FM_consider_count);
    MEX_STRUCT_PUT(FM_max_num_refinements);

    /**** Quadratic Programming Options **************************************/
    MEX_STRUCT_PUT(use_QP_gradproj);
    MEX_STRUCT_PUT(gradproj_tolerance);
    MEX_STRUCT_PUT(gradproj_iteration_limit);

    /** Final Partition Target Metrics ***************************************/
    MEX_STRUCT_PUT(target_split);
    MEX_STRUCT_PUT(soft_split_tolerance);

    return returner;
}

}
