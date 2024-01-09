//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_spzeros: all-zero sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Create a sparse matrix with no entries, of any xtype or dtype.  The A->stype
// is zero (unsymmetric) but this can be modified by the caller to either +1 or
// -1, and the matrix will still be valid.

#include "cholmod_internal.h"

cholmod_sparse *CHOLMOD(spzeros)    // return a sparse matrix with no entries
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    size_t nzmax,   // max # of entries the matrix can hold
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
)
{
    return (CHOLMOD(allocate_sparse) (nrow, ncol, nzmax,
        /* A is sorted: */ TRUE,
        /* A is packed: */ TRUE,
        /* A is unsymmetric: */ 0,
        xdtype, Common)) ;
}

