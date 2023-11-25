//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/t_cholmod_psolve_worker: permutations for cholmod_solve
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// This worker is included just twice into cholmod_solve.c, for both dtypes.
// Each method below handles all xtypes (real, complex, and zomplex) itself.

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// perm
//------------------------------------------------------------------------------

// Y = B (P (1:nrow), k1 : min (k1+ncols,ncol)-1) where B is nrow-by-ncol.
//
// Creates a permuted copy of a contiguous set of columns of B.
// Y is already allocated on input.  Y must be of sufficient size.  Let nk be
// the number of columns accessed in B.  Y->xtype determines the complexity of
// the result.
//
// If B is real and Y is complex (or zomplex), only the real part of B is
// copied into Y.  The imaginary part of Y is set to zero.
//
// If B is complex (or zomplex) and Y is real, both the real and imaginary and
// parts of B are returned in Y.  Y is returned as nrow-by-2*nk. The even
// columns of Y contain the real part of B and the odd columns contain the
// imaginary part of B.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
// returned as nrow-by-nk with leading dimension nrow.  Y->nzmax must be >=
// nrow*nk.
//
// The case where the input (B) is real and the output (Y) is zomplex is
// not used.

static void TEMPLATE_DTYPE (perm)
(
    // input:
    cholmod_dense *B,   // input matrix B
    Int *Perm,          // optional input permutation (can be NULL)
    Int k1,             // first column of B to copy
    Int ncols,          // last column to copy is min(k1+ncols,B->ncol)-1
    // input/output:
    cholmod_dense *Y    // output matrix Y, already allocated
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Yx, *Yz, *Bx, *Bz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dj, j2 ;
    size_t dual ;

    ncol = B->ncol ;
    nrow = B->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    dual = (Y->xtype == CHOLMOD_REAL && B->xtype != CHOLMOD_REAL) ? 2 : 1 ;
    d = B->d ;
    Bx = B->x ;
    Bz = B->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    Y->nrow = nrow ;
    Y->ncol = dual*nk ;
    Y->d = nrow ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*dual) ;
    ASSERT (Y->dtype == B->dtype) ;

    //--------------------------------------------------------------------------
    // Y = B (P (1:nrow), k1:k2-1)
    //--------------------------------------------------------------------------

    switch (Y->xtype)
    {

        case CHOLMOD_REAL:

            switch (B->xtype)
            {

                case CHOLMOD_REAL:
                    // Y real, B real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [k + j2] = Bx [p] ;              // real
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y real, B complex. Y is nrow-by-2*nk
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [k + j2       ] = Bx [2*p  ] ;   // real
                            Yx [k + j2 + nrow] = Bx [2*p+1] ;   // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y real, B zomplex. Y is nrow-by-2*nk
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [k + j2       ] = Bx [p] ;       // real
                            Yx [k + j2 + nrow] = Bz [p] ;       // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_COMPLEX:

            switch (B->xtype)
            {

                case CHOLMOD_REAL:
                    // Y complex, B real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [2*k   + j2] = Bx [p] ;          // real
                            Yx [2*k+1 + j2] = 0 ;               // imag
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y complex, B complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [2*k   + j2] = Bx [2*p  ] ;      // real
                            Yx [2*k+1 + j2] = Bx [2*p+1] ;      // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y complex, B zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [2*k   + j2] = Bx [p] ;          // real
                            Yx [2*k+1 + j2] = Bz [p] ;          // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_ZOMPLEX:

            switch (B->xtype)
            {

                case CHOLMOD_COMPLEX:
                    // Y zomplex, B complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [k + j2] = Bx [2*p  ] ;          // real
                            Yz [k + j2] = Bx [2*p+1] ;          // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y zomplex, B zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [k + j2] = Bx [p] ;              // real
                            Yz [k + j2] = Bz [p] ;              // imag
                        }
                    }
                    break ;

            }
            break ;
    }
}

//------------------------------------------------------------------------------
// iperm
//------------------------------------------------------------------------------

// X (P (1:nrow), k1 : min (k1+ncols,ncol)-1) = Y where X is nrow-by-ncol.
//
// Copies and permutes Y into a contiguous set of columns of X.  X is already
// allocated on input.  Y must be of sufficient size.  Let nk be the number
// of columns accessed in X.  X->xtype determines the complexity of the result.
//
// If X is real and Y is complex (or zomplex), only the real part of B is
// copied into X.  The imaginary part of Y is ignored.
//
// If X is complex (or zomplex) and Y is real, both the real and imaginary and
// parts of Y are returned in X.  Y is nrow-by-2*nk. The even
// columns of Y contain the real part of B and the odd columns contain the
// imaginary part of B.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
// nrow-by-nk with leading dimension nrow.  Y->nzmax must be >= nrow*nk.
//
// The case where the input (Y) is complex and the output (X) is real,
// and the case where the input (Y) is zomplex and the output (X) is real,
// are not used.

static void TEMPLATE_DTYPE (iperm)
(
    // input:
    cholmod_dense *Y,   // input matrix Y
    Int *Perm,          // optional input permutation (can be NULL)
    Int k1,             // first column of B to copy
    Int ncols,          // last column to copy is min(k1+ncols,B->ncol)-1
    // input/output:
    cholmod_dense *X    // output matrix X, already allocated
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Yx, *Yz, *Xx, *Xz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dj, j2 ;

    ncol = X->ncol ;
    nrow = X->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    d = X->d ;
    Xx = X->x ;
    Xz = X->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*
            ((X->xtype != CHOLMOD_REAL && Y->xtype == CHOLMOD_REAL) ? 2:1)) ;
    ASSERT (Y->dtype == X->dtype) ;

    //--------------------------------------------------------------------------
    // X (P (1:nrow), k1:k2-1) = Y
    //--------------------------------------------------------------------------

    switch (Y->xtype)
    {

        case CHOLMOD_REAL:

            switch (X->xtype)
            {

                case CHOLMOD_REAL:
                    // Y real, X real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [k + j2] ;              // real
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y real, X complex. Y is nrow-by-2*nk
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [2*p  ] = Yx [k + j2       ] ;   // real
                            Xx [2*p+1] = Yx [k + j2 + nrow] ;   // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y real, X zomplex. Y is nrow-by-2*nk
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [k + j2       ] ;       // real
                            Xz [p] = Yx [k + j2 + nrow] ;       // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_COMPLEX:

            switch (X->xtype)
            {

                case CHOLMOD_COMPLEX:
                    // Y complex, X complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [2*p  ] = Yx [2*k   + j2] ;      // real
                            Xx [2*p+1] = Yx [2*k+1 + j2] ;      // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y complex, X zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * 2 * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [2*k   + j2] ;          // real
                            Xz [p] = Yx [2*k+1 + j2] ;          // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_ZOMPLEX:

            switch (X->xtype)
            {

                case CHOLMOD_COMPLEX:
                    // Y zomplex, X complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [2*p  ] = Yx [k + j2] ;          // real
                            Xx [2*p+1] = Yz [k + j2] ;          // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y zomplex, X zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = nrow * (j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [k + j2] ;              // real
                            Xz [p] = Yz [k + j2] ;              // imag
                        }
                    }
                    break ;

            }
            break ;
    }
}

//------------------------------------------------------------------------------
// ptrans
//------------------------------------------------------------------------------

// Y = B (P (1:nrow), k1 : min (k1+ncols,ncol)-1)' where B is nrow-by-ncol.
//
// Creates a permuted and transposed copy of a contiguous set of columns of B.
// Y is already allocated on input.  Y must be of sufficient size.  Let nk be
// the number of columns accessed in B.  Y->xtype determines the complexity of
// the result.
//
// If B is real and Y is complex (or zomplex), only the real part of B is
// copied into Y.  The imaginary part of Y is set to zero.
//
// If B is complex (or zomplex) and Y is real, both the real and imaginary and
// parts of B are returned in Y.  Y is returned as 2*nk-by-nrow. The even
// rows of Y contain the real part of B and the odd rows contain the
// imaginary part of B.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
// returned as nk-by-nrow with leading dimension nk.  Y->nzmax must be >=
// nrow*nk.
//
// The array transpose is performed, not the complex conjugate transpose.

static void TEMPLATE_DTYPE (ptrans)
(
    // input:
    cholmod_dense *B,   // input matrix B
    Int *Perm,          // optional input permutation (can be NULL)
    Int k1,             // first column of B to copy
    Int ncols,          // last column to copy is min(k1+ncols,B->ncol)-1
    // input/output:
    cholmod_dense *Y    // output matrix Y, already allocated
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Yx, *Yz, *Bx, *Bz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dj, j2 ;
    size_t dual ;

    ncol = B->ncol ;
    nrow = B->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    dual = (Y->xtype == CHOLMOD_REAL && B->xtype != CHOLMOD_REAL) ? 2 : 1 ;
    d = B->d ;
    Bx = B->x ;
    Bz = B->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    Y->nrow = dual*nk ;
    Y->ncol = nrow ;
    Y->d = dual*nk ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*dual) ;
    ASSERT (Y->dtype == B->dtype) ;

    //--------------------------------------------------------------------------
    // Y = B (P (1:nrow), k1:k2-1)'
    //--------------------------------------------------------------------------

    switch (Y->xtype)
    {

        case CHOLMOD_REAL:

            switch (B->xtype)
            {

                case CHOLMOD_REAL:
                    // Y real, B real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2 + k*nk] = Bx [p] ;           // real
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y real, B complex. Y is 2*nk-by-nrow
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2   + k*2*nk] = Bx [2*p  ] ;   // real
                            Yx [j2+1 + k*2*nk] = Bx [2*p+1] ;   // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y real, B zomplex. Y is 2*nk-by-nrow
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2   + k*2*nk] = Bx [p] ;       // real
                            Yx [j2+1 + k*2*nk] = Bz [p] ;       // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_COMPLEX:

            switch (B->xtype)
            {

                case CHOLMOD_REAL:
                    // Y complex, B real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2   + k*2*nk] = Bx [p] ;       // real
                            Yx [j2+1 + k*2*nk] = 0 ;            // imag
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y complex, B complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2   + k*2*nk] = Bx [2*p  ] ;   // real
                            Yx [j2+1 + k*2*nk] = Bx [2*p+1] ;   // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y complex, B zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2   + k*2*nk] = Bx [p] ;       // real
                            Yx [j2+1 + k*2*nk] = Bz [p] ;       // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_ZOMPLEX:

            switch (B->xtype)
            {

                case CHOLMOD_REAL:
                    // Y zomplex, B real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2 + k*nk] = Bx [p] ;           // real
                            Yz [j2 + k*nk] = 0 ;                // imag
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y zomplex, B complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2 + k*nk] = Bx [2*p  ] ;       // real
                            Yz [j2 + k*nk] = Bx [2*p+1] ;       // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y zomplex, B zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Yx [j2 + k*nk] = Bx [p] ;           // real
                            Yz [j2 + k*nk] = Bz [p] ;           // imag
                        }
                    }
                    break ;

            }
            break ;
    }
}

//------------------------------------------------------------------------------
// iptrans
//------------------------------------------------------------------------------

// X (P (1:nrow), k1 : min (k1+ncols,ncol)-1) = Y' where X is nrow-by-ncol.
//
// Copies into a permuted and transposed contiguous set of columns of X.
// X is already allocated on input.  Y must be of sufficient size.  Let nk be
// the number of columns accessed in X.  X->xtype determines the complexity of
// the result.
//
// If X is real and Y is complex (or zomplex), only the real part of Y is
// copied into X.  The imaginary part of Y is ignored.
//
// If X is complex (or zomplex) and Y is real, both the real and imaginary and
// parts of X are returned in Y.  Y is 2*nk-by-nrow. The even
// rows of Y contain the real part of X and the odd rows contain the
// imaginary part of X.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
// nk-by-nrow with leading dimension nk.  Y->nzmax must be >= nrow*nk.
//
// The case where Y is complex or zomplex, and X is real, is not used.
//
// The array transpose is performed, not the complex conjugate transpose.

static void TEMPLATE_DTYPE (iptrans)
(
    // input:
    cholmod_dense *Y,   // input matrix Y
    Int *Perm,          // optional input permutation (can be NULL)
    Int k1,             // first column of X to copy into
    Int ncols,          // last column to copy is min(k1+ncols,X->ncol)-1
    // input/output:
    cholmod_dense *X    // output matrix X, already allocated
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Yx, *Yz, *Xx, *Xz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dj, j2 ;

    ncol = X->ncol ;
    nrow = X->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    d = X->d ;
    Xx = X->x ;
    Xz = X->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*
            ((X->xtype != CHOLMOD_REAL && Y->xtype == CHOLMOD_REAL) ? 2:1)) ;
    ASSERT (Y->dtype == X->dtype) ;

    //--------------------------------------------------------------------------
    // X (P (1:nrow), k1:k2-1) = Y'
    //--------------------------------------------------------------------------

    switch (Y->xtype)
    {

        case CHOLMOD_REAL:

            switch (X->xtype)
            {

                case CHOLMOD_REAL:
                    // Y real, X real
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [j2 + k*nk] ;           // real
                        }
                    }
                    break ;

                case CHOLMOD_COMPLEX:
                    // Y real, X complex. Y is 2*nk-by-nrow
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [2*p  ] = Yx [j2   + k*2*nk] ;   // real
                            Xx [2*p+1] = Yx [j2+1 + k*2*nk] ;   // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y real, X zomplex. Y is 2*nk-by-nrow
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [j2   + k*2*nk] ;       // real
                            Xz [p] = Yx [j2+1 + k*2*nk] ;       // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_COMPLEX:

            switch (X->xtype)
            {

                case CHOLMOD_COMPLEX:
                    // Y complex, X complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [2*p  ] = Yx [j2   + k*2*nk] ;   // real
                            Xx [2*p+1] = Yx [j2+1 + k*2*nk] ;   // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y complex, X zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = 2*(j-k1) ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [j2   + k*2*nk] ;       // real
                            Xz [p] = Yx [j2+1 + k*2*nk] ;       // imag
                        }
                    }
                    break ;

            }
            break ;

        case CHOLMOD_ZOMPLEX:

            switch (X->xtype)
            {

                case CHOLMOD_COMPLEX:
                    // Y zomplex, X complex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [2*p  ] = Yx [j2 + k*nk] ;       // real
                            Xx [2*p+1] = Yz [j2 + k*nk] ;       // imag
                        }
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:
                    // Y zomplex, X zomplex
                    for (j = k1 ; j < k2 ; j++)
                    {
                        dj = d*j ;
                        j2 = j-k1 ;
                        for (k = 0 ; k < nrow ; k++)
                        {
                            p = P(k) + dj ;
                            Xx [p] = Yx [j2 + k*nk] ;           // real
                            Xz [p] = Yz [j2 + k*nk] ;           // imag
                        }
                    }
                    break ;

            }
            break ;
    }
}

