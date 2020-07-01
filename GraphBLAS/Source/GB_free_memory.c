//------------------------------------------------------------------------------
// GB_free_memory: wrapper for free
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A wrapper for free.  If p is NULL on input, it is not freed.

#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void GB_free_memory
(
    void *p                 // pointer to allocated block of memory to free
)
{
    if (p != NULL)
    { 

        if (GB_Global_malloc_tracking_get ( ))
        {

            //------------------------------------------------------------------
            // for memory usage testing only
            //------------------------------------------------------------------

            GB_Global_nmalloc_decrement ( ) ;
        }

        //----------------------------------------------------------------------
        // free the memory
        //----------------------------------------------------------------------

        GB_Global_free_function (p) ;
    }
}

