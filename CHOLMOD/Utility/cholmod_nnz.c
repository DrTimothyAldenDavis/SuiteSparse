//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_nnz: # of entries in a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// returns the # of entries held in a sparse matrix data structure.  If A is
// symmetric and held in either upper/lower form, then only those entries
// in the upper/lower part are counted.

#include "cholmod_internal.h"

int64_t CHOLMOD(nnz)            // return # of entries in the sparse matrix
(
    // input:
    cholmod_sparse *A,          // sparse matrix to query
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, EMPTY) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // get the # of entries
    //--------------------------------------------------------------------------

    if (!(A->packed))
    {
        // A is held in unpacked form, so nnz(A) is sum (Anz [0..ncol-1])
        int64_t anz = 0 ;
        int64_t ncol = A->ncol ;
        Int *Anz = (Int *) A->nz ;
        for (int64_t j = 0 ; j < ncol ; j++)
        {
            anz += (int64_t) Anz [j] ;
        }
        return (anz) ;
    }
    else
    {
        // A is held in packed form, so nnz(A) is just Ap [ncol]
        return ((int64_t) ((Int *) A->p) [A->ncol]) ;
    }
}

