//------------------------------------------------------------------------------
// GrB_Matrix_wait: wait for a matrix to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Matrix_wait    // finish all work on a matrix
(
    GrB_Matrix *A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GrB_Matrix_wait (&A)") ;
    GB_BURBLE_START ("GrB_Matrix_wait") ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;

    //--------------------------------------------------------------------------
    // finish all pending work on the matrix
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_MATRIX_WAIT (*A) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

