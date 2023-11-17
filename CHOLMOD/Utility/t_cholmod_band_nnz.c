//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_band_nnz: # of entries in a band of sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Counts the # of entries with diagonals k1 to k2 of a sparse matrix.
// For example, to count entries in the tridagonal part, use k1=-1 and k2=1.
// To include the diagonal (k = 0), use ignore_diag = false; to exclude it, use
// ignore_diag = true.

#include "cholmod_internal.h"

int64_t CHOLMOD(band_nnz)   // return # of entries in a band (-1 if error)
(
    // input:
    cholmod_sparse *A,      // matrix to examine
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    bool ignore_diag,       // if true, exclude any diagonal entries
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, EMPTY) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ap  = (Int *) A->p ;
    Int *Ai  = (Int *) A->i ;
    Int *Anz = (Int *) A->nz ;
    bool packed = (bool) A->packed ;
    Int nrow = A->nrow ;
    Int ncol = A->ncol ;

    if (A->stype > 0 && k1 < 0)
    {
        // A is symmetric with the strictly lower triangular part ignored
        k1 = 0 ;
    }
    else if (A->stype < 0 && k2 > 0)
    {
        // A is symmetric with the strictly upper triangular part ignored
        k2 = 0 ;
    }

    // ensure k1 and k2 are in range -nrow:ncol
    k1 = RANGE (k1, -nrow, ncol) ;
    k2 = RANGE (k2, -nrow, ncol) ;

    // check for quick return
    if (k1 > k2) return (0) ;

    // columns outside of j1:j2 have no entries in diagonals k1:k2
    Int j1 = MAX (k1, 0) ;
    Int j2 = MIN (k2+nrow, ncol) ;

    //--------------------------------------------------------------------------
    // count entries within the k1:k2 band
    //--------------------------------------------------------------------------

    int64_t bnz = 0 ;
    for (Int j = j1 ; j < j2 ; j++)
    {
        Int p = Ap [j] ;
        Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
        for ( ; p < pend ; p++)
        {
            // A(i,j) is in the kth diagonal, where k = j-i
            Int k = j - Ai [p] ;
            // check if k is in range k1:k2; if k is zero and diagonal is
            // ignored, then skip this entry
            bnz += ((k >= k1) && (k <= k2) && !(k == 0 && ignore_diag)) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (bnz) ;
}

