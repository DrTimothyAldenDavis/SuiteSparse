//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_band_worker: extract the band of a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_band_worker)
(
    cholmod_sparse *C,
    cholmod_sparse *A,
    int64_t k1,             // keep entries in k1:k2 diagonals
    int64_t k2,
    bool ignore_diag        // if true, exclude any diagonal entries
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int nrow = A->nrow ;
    Int ncol = A->ncol ;

    Int  *Ap  = (Int  *) A->p ;
    Int  *Anz = (Int  *) A->nz ;
    Int  *Ai  = (Int  *) A->i ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->x ;
    bool packed = A->packed ;

    Int  *Cp = (Int  *) C->p ;
    Int  *Ci = (Int  *) C->i ;
    Real *Cx = (Real *) C->x ;
    Real *Cz = (Real *) C->z ;

    // columns outside of j1:j2 have no entries in diagonals k1:k2
    Int j1 = MAX (k1, 0) ;
    Int j2 = MIN (k2+nrow, ncol) ;

    //--------------------------------------------------------------------------
    // columns 0 to j1-1 are empty
    //--------------------------------------------------------------------------

    memset (Cp, 0, j1 * sizeof (Int)) ;

    //--------------------------------------------------------------------------
    // handle columns j1:j2-1
    //--------------------------------------------------------------------------

    Int cnz = 0 ;
    for (Int j = j1 ; j < j2 ; j++)
    {

        //----------------------------------------------------------------------
        // get A(:,j) and log the start of C(:,j)
        //----------------------------------------------------------------------

        // NOTE: C and A can be aliased
        Int p = Ap [j] ;
        Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
        Cp [j] = cnz ;

        //----------------------------------------------------------------------
        // extract entries in the band of A(:,j)
        //----------------------------------------------------------------------

        for ( ; p < pend ; p++)
        {
            // A(i,j) is in the kth diagonal, where k = j-i
            Int i = Ai [p] ;
            Int k = j - i ;
            // check if k is in range k1:k2; if k is zero and diagonal is
            // ignored, then skip this entry
            if ((k >= k1) && (k <= k2) && !(k == 0 && ignore_diag))
            {
                // C(i,j) = A(i,j)
                ASSIGN (Cx, Cz, cnz, Ax, Az, p) ;
                Ci [cnz++] = i ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // columns j2 to ncol-1 are empty
    //--------------------------------------------------------------------------

    for (Int j = j2 ; j <= ncol ; j++)
    {
        Cp [j] = cnz ;
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

