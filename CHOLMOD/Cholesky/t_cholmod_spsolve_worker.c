//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/t_cholmod_spsolve_worker
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// t_cholmod_spsolve_B_scatter_worker:  scatter the sparse B into the dense B4
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_spsolve_B_scatter_worker)
(
    cholmod_dense *B4,      // output dense matrix
    cholmod_sparse *B,      // input sparse matrix
    Int jfirst,
    Int jlast
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Bp = B->p ;
    Int *Bi = B->i ;
    Real *Bx = B->x ;
    Real *Bz = B->z ;
    Int *Bnz = B->nz ;
    bool packed = B->packed ;

    Real *B4x = B4->x ;
    Real *B4z = B4->z ;

    Int n = B4->nrow ;

    //--------------------------------------------------------------------------
    // B4 = B (:, jfirst:jlast-1)
    //--------------------------------------------------------------------------

    for (Int j = jfirst ; j < jlast ; j++)
    {
        Int p = Bp [j] ;
        Int pend = (packed) ? (Bp [j+1]) : (p + Bnz [j]) ;
        Int j_n = (j-jfirst)*n ;
        for ( ; p < pend ; p++)
        {
            Int q = Bi [p] + j_n ;
            ASSIGN (B4x, B4z, q, Bx, Bz, p) ;
        }
    }
}

//------------------------------------------------------------------------------
// t_cholmod_spsolve_X_worker:  append entries from X4 onto X
//------------------------------------------------------------------------------

static bool TEMPLATE (cholmod_spsolve_X_worker)
(
    cholmod_sparse *X,      // append X4 onto X
    cholmod_dense *X4,
    Int jfirst,
    Int jlast,
    size_t *xnz,            // position to place entries into X
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Xp = X->p ;
    Int *Xi = X->i ;
    Real *Xx = X->x ;
    Real *Xz = X->z ;
    size_t px = (*xnz) ;

    size_t nzmax = X->nzmax ;

    Real *X4x = X4->x ;
    Real *X4z = X4->z ;
    Int n = X4->nrow ;

    //--------------------------------------------------------------------------
    // append nonzeros from X4 onto X
    //--------------------------------------------------------------------------

    for (Int j = jfirst ; j < jlast ; j++)
    {
        Xp [j] = px ;
        Int j_n = (j-jfirst)*n ;
        if (px + n <= nzmax)
        {

            //------------------------------------------------------------------
            // X is guaranteed to be large enough
            //------------------------------------------------------------------

            for (Int i = 0 ; i < n ; i++)
            {
                // append X4 (i,j) to X if nonzero
                Int p = i + j_n ;
                if (ENTRY_IS_NONZERO (X4x, X4z, p))
                {
                    Xi [px] = i ;
                    ASSIGN (Xx, Xz, px, X4x, X4z, p) ;
                    px++ ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // X may need to increase in size
            //------------------------------------------------------------------

            for (Int i = 0 ; i < n ; i++)
            {
                // append X4 (i,j) to X if nonzero
                Int p = i + j_n ;
                if (ENTRY_IS_NONZERO (X4x, X4z, p))
                {
                    if (px >= nzmax)
                    {
                        // increase the size of X
                        nzmax *= 2 ;
                        CHOLMOD(reallocate_sparse) (nzmax, X, Common) ;
                        if (Common->status < CHOLMOD_OK)
                        {
                            return (false) ;
                        }
                        Xi = X->i ;
                        Xx = X->x ;
                        Xz = X->z ;
                    }
                    Xi [px] = i ;
                    ASSIGN (Xx, Xz, px, X4x, X4z, p) ;
                    px++ ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*xnz) = px ;
    return (true) ;
}

//------------------------------------------------------------------------------
// t_cholmod_spsolve_B_clear_worker:  clear B4 for the next iteration
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_spsolve_B_clear_worker)
(
    cholmod_dense *B4,      // output dense matrix
    cholmod_sparse *B,      // input sparse matrix
    Int jfirst,
    Int jlast
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Bp = B->p ;
    Int *Bi = B->i ;
    Int *Bnz = B->nz ;
    bool packed = B->packed ;

    Real *B4x = B4->x ;
    Real *B4z = B4->z ;

    Int n = B4->nrow ;

    //--------------------------------------------------------------------------
    // clear the entries B4 that were scattered from B (:, jfirst:jast-1)
    //--------------------------------------------------------------------------

    for (Int j = jfirst ; j < jlast ; j++)
    {
        Int p = Bp [j] ;
        Int pend = (packed) ? (Bp [j+1]) : (p + Bnz [j]) ;
        Int j_n = (j-jfirst)*n ;
        for ( ; p < pend ; p++)
        {
            Int q = Bi [p] + j_n ;
            CLEAR (B4x, B4z, q) ;
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

