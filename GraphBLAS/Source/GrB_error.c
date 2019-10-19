//------------------------------------------------------------------------------
// GrB_error: return an error string describing the last error
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// All thread local storage is held in a single struct, initialized here
//------------------------------------------------------------------------------

_Thread_local GB_thread_local_struct GB_thread_local =
{
    // error status
    .info = GrB_SUCCESS,        // last info status of user-callable routines
    .row = 0,                   // last row index searched for
    .col = 0,                   // last column index searched for
    .is_matrix = 0,             // last search matrix (true) or vector (false)
    .where = "",                // last user-callable function called
    .file = "",                 // file where error occurred
    .line = 0,                  // line number where error occurred
    .details = "",              // details of the error
    .report = "",               // report created by GrB_error

    // queued matrices with work to do 
    .queue_head = NULL,         // pointer to first queued matrix

    // GraphBLAS mode
    .mode = GrB_NONBLOCKING,    // default is nonblocking

    // malloc tracking
    .nmalloc = 0,               // memory block counter
    .malloc_debug = false,      // do not test memory handling
    .malloc_debug_count = 0,    // counter for testing memory handling

    // workspace
    .Mark = NULL,               // initialized space
    .Mark_flag = 1,             // current watermark in Mark [...]
    .Mark_size = 0,             // current size of Mark array
    .Work = NULL,               // uninitialized space
    .Work_size = 0,             // current size of Work array
    .Flag = NULL,               // initialized space
    .Flag_size = 0,             // size of Flag array

    // random seed
    .seed = 1

} ;

//------------------------------------------------------------------------------
// status_code: convert GrB_Info enum into a string
//------------------------------------------------------------------------------

static const char *status_code ( )
{
    switch (GB_thread_local.info)
    {
        case GrB_SUCCESS             : return ("GrB_SUCCESS") ;
        case GrB_NO_VALUE            : return ("GrB_NO_VALUE") ;
        case GrB_UNINITIALIZED_OBJECT: return ("GrB_UNINITIALIZED_OBJECT") ;
        case GrB_INVALID_OBJECT      : return ("GrB_INVALID_OBJECT") ;
        case GrB_NULL_POINTER        : return ("GrB_NULL_POINTER") ;
        case GrB_INVALID_VALUE       : return ("GrB_INVALID_VALUE") ;
        case GrB_INVALID_INDEX       : return ("GrB_INVALID_INDEX") ;
        case GrB_DOMAIN_MISMATCH     : return ("GrB_DOMAIN_MISMATCH") ;
        case GrB_DIMENSION_MISMATCH  : return ("GrB_DIMENSION_MISMATCH") ;
        case GrB_OUTPUT_NOT_EMPTY    : return ("GrB_OUTPUT_NOT_EMPTY") ;
        case GrB_OUT_OF_MEMORY       : return ("GrB_OUT_OF_MEMORY") ;
        case GrB_INDEX_OUT_OF_BOUNDS : return ("GrB_INDEX_OUT_OF_BOUNDS") ;
        case GrB_PANIC               : return ("GrB_PANIC") ;
        default                      : return ("unknown!") ;
    }
}

//------------------------------------------------------------------------------
// GrB_error
//------------------------------------------------------------------------------

const char *GrB_error ( )       // return a string describing the last error
{

    //--------------------------------------------------------------------------
    // construct a string in thread local storage
    //--------------------------------------------------------------------------

    if (GB_thread_local.info == GrB_SUCCESS)
    {

        //----------------------------------------------------------------------
        // status is OK, print information about GraphBLAS
        //----------------------------------------------------------------------

        snprintf (GB_thread_local.report, GB_RLEN,
        "\n=================================================================\n"
        "%s"
        "SuiteSparse:GraphBLAS version: %d.%d.%d  Date: %s\n"
        "%s"
        "Conforms to GraphBLAS spec:    %d.%d.%d  Date: %s\n"
        "%s"
        "=================================================================\n"
        #ifndef NDEBUG
        "Debugging enabled; GraphBLAS will be very slow\n"
        #endif
        "GraphBLAS status: %s\n"
        "=================================================================\n",
        GRAPHBLAS_ABOUT,
        GRAPHBLAS_IMPLEMENTATION_MAJOR,
        GRAPHBLAS_IMPLEMENTATION_MINOR,
        GRAPHBLAS_IMPLEMENTATION_SUB,
        GRAPHBLAS_DATE,
        GRAPHBLAS_LICENSE,
        GRAPHBLAS_MAJOR,
        GRAPHBLAS_MINOR,
        GRAPHBLAS_SUB,
        GRAPHBLAS_SPEC_DATE,
        GRAPHBLAS_SPEC,
        status_code ( )) ;

    }
    else if (GB_thread_local.info == GrB_NO_VALUE)
    {

        //----------------------------------------------------------------------
        // 'no value' status
        //----------------------------------------------------------------------

        if (GB_thread_local.is_matrix)
        {

        snprintf (GB_thread_local.report, GB_RLEN,
        "\n=================================================================\n"
        "GraphBLAS status: %s\nGraphBLAS function: GrB_Matrix_extractElement\n"
        "No entry A(%" PRIu64 ",%" PRIu64 ") present in the matrix.\n"
        "=================================================================\n",
        status_code ( ), GB_thread_local.row, GB_thread_local.col) ;

        }

        else
        {

        snprintf (GB_thread_local.report, GB_RLEN,
        "\n=================================================================\n"
        "GraphBLAS status: %s\nGraphBLAS function: GrB_Vector_extractElement\n"
        "No entry v(%" PRIu64 ") present in the vector.\n"
        "=================================================================\n",
        status_code ( ), GB_thread_local.row) ;

        }

    }
    else
    {

        //----------------------------------------------------------------------
        // error status
        //----------------------------------------------------------------------

        snprintf (GB_thread_local.report, GB_RLEN,
        "\n=================================================================\n"
        "GraphBLAS error: %s\nfunction: %s\n%s\n"
        #ifdef DEVELOPER
        "Line: %d in file: %s\n"
        #endif
        "=================================================================\n",
        status_code ( ),
        GB_thread_local.where, GB_thread_local.details
        #ifdef DEVELOPER
        , GB_thread_local.line, GB_thread_local.file
        #endif
        ) ;

    }

    //--------------------------------------------------------------------------
    // return the string to the user
    //--------------------------------------------------------------------------

    return (GB_thread_local.report) ;
}

