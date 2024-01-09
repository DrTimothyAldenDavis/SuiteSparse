//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_realloc_multiple: multiple realloc (int64/int32)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(realloc_multiple)   // returns true if successful, false otherwise
(
    // input:
    size_t nnew,    // # of items in newly reallocate memory
    int nint,       // 0: do not allocate I or J, 1: just I, 2: both I and J
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    // input/output:
    void **I,       // integer block of memory (int32_t or int64_t)
    void **J,       // integer block of memory (int32_t or int64_t)
    void **X,       // real or complex, double or single, block
    void **Z,       // zomplex only: double or single block
    size_t *n,      // current size of I, J, X, and/or Z blocks on input,
                    // changed to nnew on output, if successful
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;

    //--------------------------------------------------------------------------
    // get the xtype and dtype
    //--------------------------------------------------------------------------

    int xtype = xdtype & 3 ;    // pattern, real, complex, or zomplex
    int dtype = xdtype & 4 ;    // double or single

    if (nint < 1 && xtype == CHOLMOD_PATTERN)
    {
        return (TRUE) ;         // nothing to reallocate
    }

    //--------------------------------------------------------------------------
    // get the problem size
    //--------------------------------------------------------------------------

    size_t ni = (*n) ;  // size of I, if present
    size_t nj = (*n) ;  // size of J, if present
    size_t nx = (*n) ;  // size of X, if present
    size_t nz = (*n) ;  // size of Z, if present

    size_t ei = sizeof (Int) ;
    size_t e = (dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((xtype == CHOLMOD_PATTERN) ? 0 :
                    ((xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * ((xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    if ((nint > 0 && I == NULL) ||
        (nint > 1 && J == NULL) ||
        (ex   > 0 && X == NULL) ||
        (ez   > 0 && Z == NULL))
    {
        // input argument missing
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)
        {
            ERROR (CHOLMOD_INVALID, "argument missing") ;
        }
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // reallocate all of the blocks
    //--------------------------------------------------------------------------

    if (nint > 0) (*I) = CHOLMOD(realloc) (nnew, ei, *I, &ni, Common) ;
    if (nint > 1) (*J) = CHOLMOD(realloc) (nnew, ei, *J, &nj, Common) ;
    if (ex   > 0) (*X) = CHOLMOD(realloc) (nnew, ex, *X, &nx, Common) ;
    if (ez   > 0) (*Z) = CHOLMOD(realloc) (nnew, ez, *Z, &nz, Common) ;

    //--------------------------------------------------------------------------
    // handle any errors
    //--------------------------------------------------------------------------

    if (Common->status < CHOLMOD_OK)
    {
        if ((*n) == 0)
        {
            // free all blocks if they were just freshly allocated
            if (nint > 0) (*I) = CHOLMOD(free) (ni, ei, *I, Common) ;
            if (nint > 1) (*J) = CHOLMOD(free) (nj, ei, *J, Common) ;
            if (ex   > 0) (*X) = CHOLMOD(free) (nx, ex, *X, Common) ;
            if (ez   > 0) (*Z) = CHOLMOD(free) (nz, ez, *Z, Common) ;
        }
        else
        {
            // resize all blocks to their original size
            if (nint > 0) (*I) = CHOLMOD(realloc) ((*n), ei, *I, &ni, Common) ;
            if (nint > 1) (*J) = CHOLMOD(realloc) ((*n), ei, *J, &nj, Common) ;
            if (ex   > 0) (*X) = CHOLMOD(realloc) ((*n), ex, *X, &nx, Common) ;
            if (ez   > 0) (*Z) = CHOLMOD(realloc) ((*n), ez, *Z, &nz, Common) ;
        }
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // clear the first entry of X and Z
    //--------------------------------------------------------------------------

    if ((*n) == 0)
    {
        // X and Z have been freshly allocated.  Set their first entry to zero,
        // so that valgrind doesn't complain about accessing uninitialized
        // space in cholmod_*_xtype.
        if (*X != NULL && ex > 0) memset (*X, 0, ex) ;
        if (*Z != NULL && ez > 0) memset (*Z, 0, ez) ;
    }

    //--------------------------------------------------------------------------
    // log the new size and return result
    //--------------------------------------------------------------------------

    (*n) = nnew ;
    return (TRUE) ;
}

