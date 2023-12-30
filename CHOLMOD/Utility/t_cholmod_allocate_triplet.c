//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_allocate_triplet: allocate triplet matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Allocate an empty triplet matrix, with space to hold a given max # of
// entries.  The contents of T->i, T->j, T->x, and T->z are not initialized.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_triplet) (&T, Common) ;    \
        return (NULL) ;                         \
    }

cholmod_triplet *CHOLMOD(allocate_triplet)      // return triplet matrix T
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t nzmax,   // max # of entries the matrix can hold
    int stype,      // the stype of the matrix (unsym, tril, or triu)
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    Common->status = CHOLMOD_OK ;

    if (stype != 0 && nrow != ncol)
    {
        ERROR (CHOLMOD_INVALID, "rectangular matrix with stype != 0 invalid") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // get the xtype and dtype
    //--------------------------------------------------------------------------

    int xtype = xdtype & 3 ;    // pattern, real, complex, or zomplex
    int dtype = xdtype & 4 ;    // double or single

    //--------------------------------------------------------------------------
    // allocate the header for T
    //--------------------------------------------------------------------------

    cholmod_triplet *T = CHOLMOD(calloc) (1, sizeof (cholmod_triplet), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the header
    //--------------------------------------------------------------------------

    T->nrow = nrow ;            // # rows
    T->ncol = ncol ;            // # columns
    T->stype = stype ;          // symmetry type
    T->itype = ITYPE ;          // integer type
    T->xtype = xtype ;          // pattern, real, complex, or zomplex
    T->dtype = dtype ;          // double or single

    //--------------------------------------------------------------------------
    // reallocate the triplet matrix to change T->nzmax from 0 to nzmax
    //--------------------------------------------------------------------------

    CHOLMOD(reallocate_triplet) (nzmax, T, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_triplet) (T, "allocate_triplet:T", Common) >= 0) ;
    return (T) ;
}

