//------------------------------------------------------------------------------
// GxB_Matrix_resize: change the size of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This function now appears in the C API Specification as GrB_Matrix_resize.
// The new name is preferred.

#include "GB.h"

GrB_Info GxB_Matrix_resize      // change the size of a matrix
(
    GrB_Matrix A,               // matrix to modify
    GrB_Index nrows_new,        // new number of rows in matrix
    GrB_Index ncols_new         // new number of columns in matrix
)
{ 
    return (GrB_Matrix_resize (A, nrows_new, ncols_new)) ;
}

