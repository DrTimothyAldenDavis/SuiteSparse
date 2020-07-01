//------------------------------------------------------------------------------
// GxB_SelectOp_wait: wait for a user-defined GxB_SelectOp to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GxB_SelectOp has no pending
// operations to wait for.  All this method does is verify that the op is
// properly initialized.

#include "GB.h"

GrB_Info GxB_SelectOp_wait   // no work, just check if the GxB_SelectOp is valid
(
    GxB_SelectOp *op
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GxB_SelectOp_wait (&op)") ;
    GB_RETURN_IF_NULL (op) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*op) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

