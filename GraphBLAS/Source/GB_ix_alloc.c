//------------------------------------------------------------------------------
// GB_ix_alloc: allocate a matrix to hold a given number of entries
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Does not modify A->p or A->h (unless an error occurs).  Frees A->x and A->i
// and reallocates them to the requested size.  Frees any pending tuples and
// deletes all entries (including zombies, if any).  If numeric is false, then
// A->x is freed but not reallocated.

// If this method fails, all content of A is freed (including A->p and A->h).

#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_ix_alloc        // allocate A->i and A->x space in a matrix
(
    GrB_Matrix A,           // matrix to allocate space for
    const GrB_Index nzmax,  // number of entries the matrix can hold
    const bool numeric,     // if true, allocate A->x, otherwise A->x is NULL
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // GB_new does not always initialize or even allocate A->p
    ASSERT (A != NULL) ;

    if (nzmax > GxB_INDEX_MAX)
    { 
        // problem too large
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // allocate the A->x and A->i content of the matrix
    //--------------------------------------------------------------------------

    // Free the existing A->x and A->i content, if any.
    // Leave A->p and A->h unchanged.
    GB_IX_FREE (A) ;

    // allocate the new A->x and A->i content
    A->nzmax = GB_IMAX (nzmax, 1) ;
    A->i = GB_MALLOC (A->nzmax, int64_t) ;
    if (numeric)
    { 
        A->x = GB_MALLOC (A->nzmax * A->type->size, GB_void) ;
    }

    if (A->i == NULL || (numeric && A->x == NULL))
    { 
        // out of memory
        GB_PHIX_FREE (A) ;
        return (GB_OUT_OF_MEMORY) ;
    }

    return (GrB_SUCCESS) ;
}

