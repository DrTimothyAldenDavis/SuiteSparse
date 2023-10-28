//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_realloc: realloc (int64/int32)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

void *CHOLMOD(realloc)  // return newly reallocated block of memory
(
    size_t nnew,        // # of items in newly reallocate memory
    size_t size,        // size of each item
    void *p,            // pointer to memory to reallocate (may be NULL)
    size_t *n,          // # of items in p on input; nnew on output if success
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;

    //--------------------------------------------------------------------------
    // realloc the block
    //--------------------------------------------------------------------------

    int ok ;
    bool newly_allocated = (p == NULL) ;
    void *pold = p ;
    size_t nold = (*n) ;

    p = SuiteSparse_realloc (nnew, *n, size, p, &ok) ;

    //--------------------------------------------------------------------------
    // log memory usage and return result
    //--------------------------------------------------------------------------

    if (ok)
    {
        // success: log the change in memory usage and update n to new # items

        if (!newly_allocated)
        {
	    PRINTM (("cholmod_free %p %g cnt: %g inuse %g\n",
		pold, (double) nold*size, (double) Common->malloc_count-1,
                   (double) (Common->memory_inuse - nold*size))) ;
            #ifndef NDEBUG
            size_t size2 = CM_memtable_size (pold) ;
            ASSERT (nold * size == size2) ;
            CM_memtable_remove (pold) ;
            #endif
        }
        Common->memory_inuse += ((nnew-nold) * size) ;
        Common->memory_usage = MAX (Common->memory_usage, Common->memory_inuse);
        if (newly_allocated)
        {
            Common->malloc_count++ ;
        }
        PRINTM (("cholmod_malloc %p %g cnt: %g inuse %g\n",
            p, (double) nnew*size, (double) Common->malloc_count,
            (double) Common->memory_inuse)) ;
        #ifndef NDEBUG
        CM_memtable_add (p, nnew * size) ;
        #endif
        (*n) = nnew ;
    }
    else
    {
        // p is unchanged
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
    }
    return (p) ;
}

