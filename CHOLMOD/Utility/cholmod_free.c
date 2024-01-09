//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_free: free (int64/int32)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

void *CHOLMOD(free)     // returns NULL to simplify its usage
(
    // input:
    size_t n,           // number of items
    size_t size,        // size of each item
    // input/output:
    void *p,            // memory to free
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    if (p == NULL) return (NULL) ;      // nothing to do (not an error)

    #ifndef NDEBUG
    size_t size2 = CM_memtable_size (p) ;
    ASSERT (n * size == size2) ;
    #endif

    //--------------------------------------------------------------------------
    // free memory
    //--------------------------------------------------------------------------

    SuiteSparse_free (p) ;

    //--------------------------------------------------------------------------
    // log memory usage and return result
    //--------------------------------------------------------------------------

    Common->memory_inuse -= (n * size) ;
    Common->malloc_count-- ;

    PRINTM (("cholmod_free   %p %g cnt: %g inuse %g\n",
        p, (double) n*size, (double) Common->malloc_count,
        (double) Common->memory_inuse)) ;

    #ifndef NDEBUG
    CM_memtable_remove (p) ;
    #endif

    return (NULL) ;
}

