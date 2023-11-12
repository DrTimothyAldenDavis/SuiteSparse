//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_malloc: malloc/calloc (int64/int32)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// This template creates 4 functions:
//      cholmod_malloc      int32, malloc, using SuiteSparse_malloc
//      cholmod_l_malloc    int64, malloc, using SuiteSparse_malloc
//      cholmod_calloc      int32, calloc, using SuiteSparse_calloc
//      cholmod_l_calloc    int64, calloc, using SuiteSparse_calloc

void *CHOLMOD_ALLOC_FUNCTION    // return pointer to newly allocated memory
(
    // input:
    size_t n,           // number of items
    size_t size,        // size of each item
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;

    //--------------------------------------------------------------------------
    // allocate memory
    //--------------------------------------------------------------------------

    void *p = SUITESPARSE_ALLOC_FUNCTION (n, size) ;   // malloc or calloc

    //--------------------------------------------------------------------------
    // log memory usage and return result
    //--------------------------------------------------------------------------

    if (p != NULL)
    {
        // success: log the change in memory usage
        Common->memory_inuse += (n * size) ;
        Common->memory_usage = MAX (Common->memory_usage, Common->memory_inuse);
        Common->malloc_count++ ;
        PRINTM (("cholmod_malloc %p %g cnt: %g inuse %g\n",
            p, (double) n*size, (double) Common->malloc_count,
            (double) Common->memory_inuse)) ;
        #ifndef NDEBUG
        CM_memtable_add (p, n*size) ;
        #endif
    }
    else
    {
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
    }
    return (p) ;
}

