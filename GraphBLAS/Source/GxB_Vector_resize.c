//------------------------------------------------------------------------------
// GxB_Vector_resize: change the size of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This function now appears in the C API Specification as GrB_Vector_resize.
// The new name is preferred.

#include "GB.h"

GrB_Info GxB_Vector_resize      // change the size of a vector
(
    GrB_Vector u,               // vector to modify
    GrB_Index nrows_new         // new number of rows in vector
)
{ 
    return (GrB_Vector_resize (u, nrows_new)) ;
}

