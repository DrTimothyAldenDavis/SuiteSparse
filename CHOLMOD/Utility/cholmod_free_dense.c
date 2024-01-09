//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_free_dense: free dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(free_dense)
(
    // input/output:
    cholmod_dense **X,          // handle of dense matrix to free
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    if (X == NULL || (*X) == NULL)
    {
        // X is already freed; nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t e = ((*X)->dtype == CHOLMOD_SINGLE) ?
                    sizeof (float) : sizeof (double) ;
    size_t ex = e * (((*X)->xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;
    size_t ez = e * (((*X)->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;
    size_t nzmax = (*X)->nzmax ;

    //--------------------------------------------------------------------------
    // free the two arrays
    //--------------------------------------------------------------------------

    CHOLMOD(free) (nzmax, ex, (*X)->x,  Common) ;
    CHOLMOD(free) (nzmax, ez, (*X)->z,  Common) ;

    //--------------------------------------------------------------------------
    // free the header and return result
    //--------------------------------------------------------------------------

    (*X) = CHOLMOD(free) (1, sizeof (cholmod_dense), (*X), Common) ;
    return (TRUE) ;
}

