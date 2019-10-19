//------------------------------------------------------------------------------
// GB_free_memory: wrapper for free
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A wrapper for FREE.  If p is NULL on input, it is not freed.

// By default, FREE is defined in GB.h as free.  For a MATLAB mexFunction, it
// is mxFree.  It can also be defined at compile time with -DFREE=myfreefunc.

#include "GB.h"

void GB_free_memory         // pointer to allocated block of memory to free
(
    void *p
)
{
    if (p != NULL)
    {
        GB_thread_local.nmalloc-- ;

#ifdef PRINT_MALLOC
            printf ("free:    %14p %3d %1d\n",
                p,
                (int) GB_thread_local.nmalloc,
                GB_thread_local.malloc_debug) ;
        if (GB_thread_local.nmalloc < 0 )
            printf (GBd " free    %p negative mallocs!\n",
                GB_thread_local.nmalloc, p) ;
#endif

        FREE (p) ;
        ASSERT (GB_thread_local.nmalloc >= 0) ;
    }
}

