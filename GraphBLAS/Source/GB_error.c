//------------------------------------------------------------------------------
// GB_error: log an error string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_error logs the details of an error to the error string in thread-local
// storage so that it is accessible to GrB_error.  This function is called via
// the GB_ERROR(info,args) macro.

// SuiteSparse:GraphBLAS can generate a GrB_PANIC only in these cases:

//  (1) GrB_init (or GxB*init) is called twice.

//  (2) unrecoverable GPU failure

//  (3) an internal error in the Intel MKL library

//  (4) a failure to allocate thread-local storage for GrB_error
//      (see GB_thread_local_get).

#include "GB_thread_local.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_error           // log an error in thread-local-storage
(
    GrB_Info info,          // error return code from a GraphBLAS function
    GB_Context Context      // pointer to a Context struct, on the stack.
                            // The Context may be NULL, which occurs when a
                            // parallel region calls GB_* functions and
                            // wants them to run with one thread.
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // GrB_SUCCESS and GrB_NO_VALUE are not errors.

    ASSERT (info != GrB_SUCCESS) ;
    ASSERT (info > GrB_NO_VALUE) ;
    ASSERT (info <= GrB_PANIC) ;

    //--------------------------------------------------------------------------
    // quick return if Context is NULL
    //--------------------------------------------------------------------------

    if (Context == NULL)
    { 
        // the error cannot be logged in the Context, inside a parallel region,
        // so just return the error.  The error will be logged when the
        // parallel region exits.
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // get pointer to thread-local storage
    //--------------------------------------------------------------------------

    char *p = GB_thread_local_get ( ) ;
    if (p == NULL) return (GrB_PANIC) ;

    //--------------------------------------------------------------------------
    // write the error to the string p
    //--------------------------------------------------------------------------

    // p now points to thread-local storage (char array of size GB_RLEN+1)
    snprintf (p, GB_RLEN, "GraphBLAS error: %s\nfunction: %s\n%s\n",
        GB_status_code (info),
        (Context == NULL) ? "" : Context->where,
        (Context == NULL) ? "" : Context->details) ;
    return (info) ;
}

