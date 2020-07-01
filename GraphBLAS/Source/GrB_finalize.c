//------------------------------------------------------------------------------
// GrB_finalize: finalize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GrB_finalize must be called as the last GraphBLAS function, per the
// GraphBLAS C API Specification.  Only one user thread can call this
// function.  Results are undefined if more than one thread calls this
// function at the same time.

#include "GB.h"

GrB_Info GrB_finalize ( )
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GrB_finalize") ;

    #if defined (USER_POSIX_THREADS)
    { pthread_mutex_destroy (&GB_sync) ; }  // TODO in 4.0: delete
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

