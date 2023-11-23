//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/ldlsolve: MATLAB interface to CHOLMOD LDL' solve
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Solve LDL'x=b given an LDL' factorization computed by ldlchol.
//
// Usage:
//
//      x = ldlsolve (LD,b)
//
// b can be dense or sparse.

#include "sputil2.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0, rcond ;
    int64_t *Lp, *Lnz, *Lprev, *Lnext ;
    cholmod_sparse *Bs, Bspmatrix, *Xs ;
    cholmod_dense *B, Bmatrix, *X ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    int64_t j, k, n, B_is_sparse, head, tail ;

    //--------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil2_config (SPUMONI, cm) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = ldlsolve (LD, b)") ;
    }

    n = mxGetN (pargin [0]) ;
    k = mxGetN (pargin [1]) ;

    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
        mexErrMsgTxt ("ldlsolve: LD must be sparse and square") ;
    }
    if (n != mxGetM (pargin [1]))
    {
        mexErrMsgTxt ("ldlsolve: b wrong dimension") ;
    }

    //--------------------------------------------------------------------------
    // get b
    //--------------------------------------------------------------------------

    // get sparse or dense matrix B
    B = NULL ;
    Bs = NULL ;
    B_is_sparse = mxIsSparse (pargin [1]) ;
    size_t B_xsize = 0 ;
    if (B_is_sparse)
    {
        // get sparse matrix B (unsymmetric)
        Bs = sputil2_get_sparse (pargin [1], 0, CHOLMOD_DOUBLE, &Bspmatrix,
            &B_xsize, cm) ;
    }
    else
    {
        // get dense matrix B
        B = sputil2_get_dense (pargin [1], CHOLMOD_DOUBLE, &Bmatrix,
            &B_xsize, cm) ;
    }

    //--------------------------------------------------------------------------
    // construct a shallow copy of the input sparse matrix L
    //--------------------------------------------------------------------------

    // the construction of the CHOLMOD takes O(n) time and memory

    // allocate the CHOLMOD symbolic L
    L = cholmod_l_allocate_factor (n, cm) ;
    L->ordering = CHOLMOD_NATURAL ;

    // get the MATLAB L
    L->p = mxGetJc (pargin [0]) ;
    L->i = mxGetIr (pargin [0]) ;
    L->x = mxGetData (pargin [0]) ;
    L->z = NULL ;

    // allocate and initialize the rest of L
    L->nz = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
    Lp = L->p ;
    Lnz = L->nz ;
    for (j = 0 ; j < n ; j++)
    {
        Lnz [j] = Lp [j+1] - Lp [j] ;
    }
    L->prev = cholmod_l_malloc (n+2, sizeof (int64_t), cm) ;
    L->next = cholmod_l_malloc (n+2, sizeof (int64_t), cm) ;
    Lprev = L->prev ;
    Lnext = L->next ;

    head = n+1 ;
    tail = n ;
    Lnext [head] = 0 ;
    Lprev [head] = -1 ;
    Lnext [tail] = -1 ;
    Lprev [tail] = n-1 ;
    for (j = 0 ; j < n ; j++)
    {
        Lnext [j] = j+1 ;
        Lprev [j] = j-1 ;
    }
    Lprev [0] = head ;

    L->xtype = (mxIsComplex (pargin [0])) ? CHOLMOD_COMPLEX : CHOLMOD_REAL ;
    L->nzmax = Lp [n] ;

    //--------------------------------------------------------------------------
    // solve and return solution to MATLAB
    //--------------------------------------------------------------------------

    if (B_is_sparse)
    {
        // solve LDL'X=B with sparse X and B; return sparse X to MATLAB.
        // cholmod_l_spsolve returns Xs with no explicit zeros.
        Xs = cholmod_l_spsolve (CHOLMOD_LDLt, L, Bs, cm) ;
        pargout [0] = sputil2_put_sparse (&Xs, mxDOUBLE_CLASS,
            /* already done by cholmod_l_spsolve: */ false, cm) ;
    }
    else
    {
        // solve AX=B with dense X and B; return dense X to MATLAB
        X = cholmod_l_solve (CHOLMOD_LDLt, L, B, cm) ;
        pargout [0] = sputil2_put_dense (&X, mxDOUBLE_CLASS, cm) ;
    }

    rcond = cholmod_l_rcond (L, cm) ;
    if (rcond == 0)
    {
        mexWarnMsgTxt ("Matrix is indefinite or singular to working precision");
    }
    else if (rcond < DBL_EPSILON)
    {
        mexWarnMsgTxt ("Matrix is close to singular or badly scaled.") ;
        mexPrintf ("         Results may be inaccurate. RCOND = %g.\n", rcond) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and the CHOLMOD L, except for what is copied to MATLAB
    //--------------------------------------------------------------------------

    L->p = NULL ;
    L->i = NULL ;
    L->x = NULL ;
    sputil2_free_sparse (&Bs, &Bspmatrix, B_xsize, cm) ;
    sputil2_free_dense  (&B,  &Bmatrix,   B_xsize, cm) ;
    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
}

