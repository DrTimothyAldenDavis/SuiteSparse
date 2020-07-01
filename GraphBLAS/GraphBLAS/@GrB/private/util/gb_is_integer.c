//------------------------------------------------------------------------------
// gb_is_integer: check if a GrB_Type is integer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

bool gb_is_integer (const GrB_Type type)
{ 
    return ((type == GrB_INT8  ) ||
            (type == GrB_INT16 ) ||
            (type == GrB_INT32 ) ||
            (type == GrB_INT64 ) ||
            (type == GrB_UINT8 ) ||
            (type == GrB_UINT16) ||
            (type == GrB_UINT32) ||
            (type == GrB_UINT64)) ;
}

