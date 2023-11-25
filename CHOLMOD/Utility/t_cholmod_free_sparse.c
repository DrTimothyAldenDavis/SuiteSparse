//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_free_sparse: free sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(free_sparse)
(
    // input/output:
    cholmod_sparse **A,         // handle of sparse matrix to free
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    if (A == NULL || (*A) == NULL)
    {
        // A is already freed; nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = ((*A)->dtype == CHOLMOD_SINGLE) ?
                    sizeof (float) : sizeof (double) ;
    size_t ex = e * (((*A)->xtype == CHOLMOD_PATTERN) ? 0 :
                    (((*A)->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * (((*A)->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    size_t nzmax = (*A)->nzmax ;
    size_t ncol  = (*A)->ncol ;

    //--------------------------------------------------------------------------
    // free the five arrays
    //--------------------------------------------------------------------------

    CHOLMOD(free) (ncol+1, ei, (*A)->p,  Common) ;
    CHOLMOD(free) (ncol,   ei, (*A)->nz, Common) ;
    CHOLMOD(free) (nzmax,  ei, (*A)->i,  Common) ;
    CHOLMOD(free) (nzmax,  ex, (*A)->x,  Common) ;
    CHOLMOD(free) (nzmax,  ez, (*A)->z,  Common) ;

    //--------------------------------------------------------------------------
    // free the header and return result
    //--------------------------------------------------------------------------

    (*A) = CHOLMOD(free) (1, sizeof (cholmod_sparse), (*A), Common) ;
    return (TRUE) ;
}

