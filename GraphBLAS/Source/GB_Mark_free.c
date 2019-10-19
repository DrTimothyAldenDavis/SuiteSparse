//------------------------------------------------------------------------------
// GB_Mark_free: free the Mark workspace array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

void GB_Mark_free ( )               // free the Mark array
{
    GB_FREE_MEMORY (GB_thread_local.Mark) ;
    GB_thread_local.Mark_size = 0 ;
    GB_thread_local.Mark_flag = 1 ;
}

