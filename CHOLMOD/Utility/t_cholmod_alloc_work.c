//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_alloc_work: double/single int32/64 workspace
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Allocates and initializes CHOLMOD workspace in Common, or increases the size
// of the workspace if already allocated.  If the required workspace is already
// allocated, no action is taken.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_work) (Common) ;           \
        return (FALSE) ;                        \
    }

#define RETURN_IF_ALLOC_NOT_ALLOWED             \
    if (Common->no_workspace_reallocate)        \
    {                                           \
        Common->status = CHOLMOD_INVALID ;      \
        return (FALSE) ;                        \
    }

int CHOLMOD(alloc_work)
(
    // input:
    size_t nrow,        // # of rows in the matrix A
    size_t iworksize,   // size of Iwork (# of integers, int32 or int64)
    size_t xworksize,   // size of Xwork (in # of entries, double or single)
    int dtype,          // CHOLMOD_DOUBLE or CHOLMOD_SINGLE
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate Flag (of size nrow Ints) and Head (of size nrow+1 Ints)
    //--------------------------------------------------------------------------

    // ensure at least 1 entry allocated, and check for size_t overflow
    nrow = MAX (1, nrow) ;
    size_t nrow1 = nrow + 1 ;
    if (nrow1 < nrow) Common->status = CHOLMOD_TOO_LARGE ;
    RETURN_IF_ERROR ;

    if (nrow > Common->nrow)
    {
        RETURN_IF_ALLOC_NOT_ALLOWED ;
        Common->Flag = CHOLMOD(free) (Common->nrow, sizeof (Int),
            Common->Flag, Common) ;
        Common->Head = CHOLMOD(free) (Common->nrow+1, sizeof (Int),
            Common->Head, Common) ;
        Common->nrow = nrow ;
        Common->Flag = CHOLMOD(malloc) (nrow,  sizeof (Int), Common) ;
        Common->Head = CHOLMOD(malloc) (nrow1, sizeof (Int), Common) ;
        RETURN_IF_ERROR ;
        // clear the Flag and Head workspace
        Common->mark = 0 ;
        CHOLMOD(set_empty) (Common->Flag, nrow) ;
        CHOLMOD(set_empty) (Common->Head, nrow+1) ;
    }

    //--------------------------------------------------------------------------
    // allocate Iwork (of size iworksize Ints)
    //--------------------------------------------------------------------------

    iworksize = MAX (1, iworksize) ;
    if (iworksize > Common->iworksize)
    {
        RETURN_IF_ALLOC_NOT_ALLOWED ;
        CHOLMOD(free) (Common->iworksize, sizeof (Int), Common->Iwork, Common) ;
        Common->iworksize = iworksize ;
        Common->Iwork = CHOLMOD(malloc) (iworksize, sizeof (Int), Common) ;
        RETURN_IF_ERROR ;
    }

    //--------------------------------------------------------------------------
    // allocate Xwork (xworksize) and set it to 0
    //--------------------------------------------------------------------------

    // make sure xworksize is >= 2
    xworksize = MAX (2, xworksize) ;

    size_t e = (dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;

    if (xworksize > Common->xworkbytes / e)
    {
        RETURN_IF_ALLOC_NOT_ALLOWED ;
        CHOLMOD(free) (Common->xworkbytes, sizeof (uint8_t), Common->Xwork,
            Common) ;
        Common->Xwork = CHOLMOD(malloc) (xworksize, e, Common) ;
        RETURN_IF_ERROR ;
        // clear the Xwork workspace
        Common->xworkbytes = xworksize * e ;
        memset (Common->Xwork, 0, Common->xworkbytes) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (TRUE) ;
}

