//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/sdmult: sparse-times-full using CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Compute C = S*F or S'*F where S is sparse and F is full (C is also sparse).
// S and F must both be real or both be complex.
//
// Usage:
//
//      C = sdmult (S,F) ;              C = S*F
//      C = sdmult (S,F,0) ;            C = S*F
//      C = sdmult (S,F,1) ;            C = S'*F

#include "sputil2.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0, one [2] = {1,0}, zero [2] = {0,0} ;
    cholmod_sparse *S, Smatrix ;
    cholmod_dense *F, Fmatrix, *C ;
    cholmod_common Common, *cm ;
    int64_t srow, scol, frow, fcol, crow, transpose ;

    //--------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil2_config (SPUMONI, cm) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: C = sdmult (S,F,transpose)") ;
    }

    srow = mxGetM (pargin [0]) ;
    scol = mxGetN (pargin [0]) ;
    frow = mxGetM (pargin [1]) ;
    fcol = mxGetN (pargin [1]) ;

    transpose = !((nargin == 2) || (mxGetScalar (pargin [2]) == 0)) ;

    if (frow != (transpose ? srow : scol))
    {
        mexErrMsgTxt ("invalid inner dimensions") ;
    }

    if (!mxIsSparse (pargin [0]) || mxIsSparse (pargin [1]))
    {
        mexErrMsgTxt ("sdmult (S,F): S must be sparse, F must be full") ;
    }

    //--------------------------------------------------------------------------
    // get S and F
    //--------------------------------------------------------------------------

    size_t S_xsize = 0 ;
    S = sputil2_get_sparse (pargin [0], 0, CHOLMOD_DOUBLE, &Smatrix,
        &S_xsize, cm) ;

    size_t F_xsize = 0 ;
    F = sputil2_get_dense (pargin [1], CHOLMOD_DOUBLE, &Fmatrix, &F_xsize, cm) ;

    //--------------------------------------------------------------------------
    // C = S*F or S'*F
    //--------------------------------------------------------------------------

    crow = transpose ? scol : srow ;
    C = cholmod_l_allocate_dense (crow, fcol, crow, F->xtype, cm) ;
    cholmod_l_sdmult (S, transpose, one, zero, F, C, cm) ;
    pargout [0] = sputil2_put_dense (&C, mxDOUBLE_CLASS, cm) ;

    //--------------------------------------------------------------------------
    // free workspace and the CHOLMOD L, except for what is copied to MATLAB
    //--------------------------------------------------------------------------

    sputil2_free_sparse (&S, &Smatrix, S_xsize, cm) ;
    sputil2_free_dense  (&F, &Fmatrix, F_xsize, cm) ;
    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
}

