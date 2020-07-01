//------------------------------------------------------------------------------
// gb_is_float: check if a GrB_Type is floating-point
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

bool gb_is_float (const GrB_Type type)
{ 
    return ((type == GrB_FP32 ) ||
            (type == GrB_FP64 ) ||
            (type == GxB_FC32 ) ||
            (type == GxB_FC64 )) ;
}

