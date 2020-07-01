//------------------------------------------------------------------------------
// GrB_Vector_wait: wait for a vector to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GrB_Vector_wait    // finish all work on a vector
(
    GrB_Vector *v
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GrB_Vector_wait (&v)") ;
    GB_BURBLE_START ("GrB_Vector_wait") ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*v) ;

    //--------------------------------------------------------------------------
    // finish all pending work on the vector
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_VECTOR_WAIT (*v) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

