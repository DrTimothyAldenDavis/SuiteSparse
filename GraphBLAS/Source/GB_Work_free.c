//------------------------------------------------------------------------------
// GB_Work_free: free the Work workspace array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

void GB_Work_free ( )               // free the Work array
{
    GB_FREE_MEMORY (GB_thread_local.Work) ;
    GB_thread_local.Work_size = 0 ;
}

