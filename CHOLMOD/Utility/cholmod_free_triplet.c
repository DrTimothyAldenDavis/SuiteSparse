//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_free_triplet: free triplet matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(free_triplet)
(
    // input/output:
    cholmod_triplet **T,        // handle of triplet matrix to free
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    if (T == NULL || (*T) == NULL)
    {
        // T is already freed; nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = ((*T)->dtype == CHOLMOD_SINGLE) ?
                    sizeof (float) : sizeof (double) ;
    size_t ex = e * (((*T)->xtype == CHOLMOD_PATTERN) ? 0 :
                    (((*T)->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * (((*T)->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;
    size_t nz = (*T)->nzmax ;

    //--------------------------------------------------------------------------
    // free the four arrays
    //--------------------------------------------------------------------------

    CHOLMOD(free) (nz, ei, (*T)->i, Common) ;
    CHOLMOD(free) (nz, ei, (*T)->j, Common) ;
    CHOLMOD(free) (nz, ex, (*T)->x, Common) ;
    CHOLMOD(free) (nz, ez, (*T)->z, Common) ;

    //--------------------------------------------------------------------------
    // free the header and return result
    //--------------------------------------------------------------------------

    (*T) = CHOLMOD(free) (1, sizeof (cholmod_triplet), (*T), Common) ;
    return (TRUE) ;
}

