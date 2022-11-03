// =============================================================================
// === spqr_stranspose2 ========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Construct the numerical values of S = A (p,q) in compressed-row form

#include "spqr.hpp"

template <typename Entry> void spqr_stranspose2
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    int64_t *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    int64_t *Sp,           // size m+1, row pointers of S
    int64_t *PLinv,        // size m, inverse row permutation, PLinv [i] = k

    // output, contents not defined on input
    Entry *Sx,          // size nz, numerical values of S

    // workspace, not defined on input or output
    int64_t *W             // size m
)
{
    int64_t i, j, p, pend, row, col, s, m, n, *Ap, *Ai ;
    Entry *Ax ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    m = A->nrow ;
    n = A->ncol ;
    Ap = (int64_t *) A->p ;
    Ai = (int64_t *) A->i ;
    Ax = (Entry *) A->x ;

    // -------------------------------------------------------------------------
    // create S = A (p,q)', or S=A(p,q) if S is considered to be in row-form
    // -------------------------------------------------------------------------

    for (row = 0 ; row < m ; row++)
    {
        W [row] = Sp [row] ;
    }

    for (col = 0 ; col < n ; col++)     // for each column of A(:,Qfill)
    {
        j = Qfill ? Qfill [col] : col ; // col of S is column j of A
        pend = Ap [j+1] ;
        for (p = Ap [j] ; p < pend ; p++)
        {
            i = Ai [p] ;                // the entry A(i,j)
            row = PLinv [i] ;           // row of S is row i of A
            s = W [row]++ ;             // place S(row,col) in position
            Sx [s] = Ax [p] ;
        }
    }
}


// =============================================================================

template void spqr_stranspose2 <double>
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    int64_t *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    int64_t *Sp,           // size m+1, row pointers of S
    int64_t *PLinv,        // size m, inverse row permutation, PLinv [i] = k

    // output, contents not defined on input
    double *Sx,         // size nz, numerical values of S

    // workspace, not defined on input or output
    int64_t *W             // size m
) ;

// =============================================================================

template void spqr_stranspose2 <Complex>
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    int64_t *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    int64_t *Sp,           // size m+1, row pointers of S
    int64_t *PLinv,        // size m, inverse row permutation, PLinv [i] = k

    // output, contents not defined on input
    Complex *Sx,        // size nz, numerical values of S

    // workspace, not defined on input or output
    int64_t *W             // size m
) ;

