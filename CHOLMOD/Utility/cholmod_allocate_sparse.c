//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_allocate_sparse: allocate a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Allocate a sparse matrix.  The contents A->i, A->x, and A->z (if zomplex)
// exist but are not initialized.  A->p and A->nz (if unpacked) are set to
// zero, giving a valid sparse matrix with no entries.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_sparse) (&A, Common) ;     \
        return (NULL) ;                         \
    }

cholmod_sparse *CHOLMOD(allocate_sparse)
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t nzmax,   // max # of entries the matrix can hold
    int sorted,     // true if columns are sorted
    int packed,     // true if A is be packed (A->nz NULL), false if unpacked
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
    // allocate the header for A
    //--------------------------------------------------------------------------

    cholmod_sparse *A = CHOLMOD(calloc) (1, sizeof (cholmod_sparse), Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the header
    //--------------------------------------------------------------------------

    A->nrow = nrow ;            // # rows
    A->ncol = ncol ;            // # columns
    A->stype = stype ;          // symmetry type
    A->itype = ITYPE ;          // integer type
    A->xtype = xtype ;          // pattern, real, complex, or zomplex
    A->dtype = dtype ;          // double or single

    A->packed = packed ;        // packed or unpacked
    A->sorted = sorted ;        // columns sorted or unsorted

    //--------------------------------------------------------------------------
    // allocate and clear A->p and A->nz
    //--------------------------------------------------------------------------

    A->p = CHOLMOD(calloc) (ncol+1, sizeof (Int), Common) ;
    if (!packed)
    {
        A->nz = CHOLMOD(calloc) (ncol, sizeof (Int), Common) ;
    }
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // reallocate the sparse matrix to change A->nzmax from 0 to nzmax
    //--------------------------------------------------------------------------

    CHOLMOD(reallocate_sparse) (nzmax, A, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (A, "allocate_sparse:A", Common) >= 0) ;
    return (A) ;
}

