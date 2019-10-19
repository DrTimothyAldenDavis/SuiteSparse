//------------------------------------------------------------------------------
// GB_mx_put_global: put the GraphBLAS status in MATLAB workspace
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

void GB_mx_put_global
(
    bool malloc_debug
)
{

    //--------------------------------------------------------------------------
    // check nmalloc
    //--------------------------------------------------------------------------

    Complex_finalize ( ) ;

    //--------------------------------------------------------------------------
    // log the statement coverage
    //--------------------------------------------------------------------------

    #ifdef GBCOVER
    gbcover_put ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_finalize ( ) ;

    if (GB_thread_local.nmalloc != 0)
    {
        printf ("GraphBLAS nmalloc "GBd"!\n", GB_thread_local.nmalloc) ;
        mexErrMsgTxt ("memory leak!") ;
    }
}

