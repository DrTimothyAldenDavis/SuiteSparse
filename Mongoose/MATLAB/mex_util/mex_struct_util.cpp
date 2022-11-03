//------------------------------------------------------------------------------
// Mongoose/MATLAB/mex_util/mex_struct_util.cpp
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

/*****************************************************************************
Function:
    Add Field with Value

Purpose:
    Adds a field to the given mxArray (assumed to be a 1x1 matlab structure)
    and automatically sets it to the given value.
 *****************************************************************************/
void addFieldWithValue
(
    mxArray* matStruct,     /* The mxArray assumed to be a matlab structure. */
    const char* fieldname,  /* The name of the field to create.              */
    const double value      /* The double value to assign to the new field.  */
)
{
    if(!mxIsStruct(matStruct)) return;

    mxAddField(matStruct, fieldname);
    mxSetField(matStruct, 0, fieldname, mxCreateDoubleScalar(value));
}

/*****************************************************************************
Function:
    Reads Field

Purpose:
    Reads the scalar value out of the specified structure.field.
 *****************************************************************************/
double readField
(
    const mxArray* matStruct,
    const char* fieldname
)
{
    double returner = 0.0;

    mxArray* field = mxGetField(matStruct, 0, fieldname);
    if(field != NULL) returner = mxGetScalar(field);
    return returner;
}

}
