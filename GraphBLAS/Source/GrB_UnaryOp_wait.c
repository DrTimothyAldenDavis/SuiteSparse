//------------------------------------------------------------------------------
// GrB_UnaryOp_wait: wait for a user-defined GrB_UnaryOp to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GrB_UnaryOp has no pending
// operations to wait for.  All this method does is verify that the op is
// properly initialized.

#include "GB.h"

GrB_Info GrB_UnaryOp_wait   // no work, just check if the GrB_UnaryOp is valid
(
    GrB_UnaryOp *op
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GrB_UnaryOp_wait (&op)") ;
    GB_RETURN_IF_NULL (op) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*op) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

