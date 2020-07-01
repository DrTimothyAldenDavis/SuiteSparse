//------------------------------------------------------------------------------
// GB_block: apply all pending computations if blocking mode enabled
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_Pending.h"

#define GB_FREE_ALL ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_block   // apply all pending computations if blocking mode enabled
(
    GrB_Matrix A,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (A != NULL) ;

    //--------------------------------------------------------------------------
    // check for blocking mode
    //--------------------------------------------------------------------------

    if (GB_shall_block (A))
    { 
        // delete any lingering zombies and assemble any pending tuples
        GB_MATRIX_WAIT (A) ;
    }
    return (GrB_SUCCESS) ;
}

