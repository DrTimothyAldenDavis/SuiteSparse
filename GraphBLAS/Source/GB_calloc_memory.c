//------------------------------------------------------------------------------
// GB_calloc_memory: wrapper for calloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A wrapper for CALLOC.  Space is set to zero.

// Parameters are the same as the POSIX calloc, except that asking to allocate
// a block of zero size causes a block of size 1 to be allocated instead.  This
// allows the return pointer p to be checked for the out-of-memory condition,
// even when allocating an object of size zero.

// By default, CALLOC is defined in GB.h as calloc.  For a MATLAB mexFunction,
// it is mxCalloc.  It can also be defined at compile time with
// -DCALLOC=mycallocfunc.

#include "GB.h"

void *GB_calloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item     // sizeof each item
)
{

    void *p ;
    size_t size ;

    // make sure at least one item is allocated
    nitems = IMAX (1, nitems) ;

    // make sure at least one byte is allocated
    size_of_item = IMAX (1, size_of_item) ;

    bool ok = GB_size_t_multiply (&size, nitems, size_of_item) ;
    if (!ok || nitems > GB_INDEX_MAX || size_of_item > GB_INDEX_MAX)
    {
        // overflow
        p = NULL ;
    }
    else
    {

        if (GB_thread_local.malloc_debug)
        {
            // brutal malloc debug; pretend to fail if the count <= 0
            if (GB_thread_local.malloc_debug_count <= 0)
            {
                return (false) ;
            }
        }

        p = (void *) CALLOC (nitems, size_of_item) ;

        if (p != NULL)
        {
            GB_thread_local.nmalloc++ ;

#ifdef PRINT_MALLOC
            printf ("calloc:  %14p %3d %1d n "GBd" size "GBd"\n", 
                p,
                (int) GB_thread_local.nmalloc,
                GB_thread_local.malloc_debug,
                (int64_t) nitems, (int64_t) size_of_item) ;
#endif

            // a malloc has been used up
            if (GB_thread_local.malloc_debug)
            {
                GB_thread_local.malloc_debug_count-- ;
            }
        }

    }
    return (p) ;
}

