//------------------------------------------------------------------------------
// UMFPACK/MATLAB/luflop.c: MATLAB mexFunction for computing # flops for LU
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2022, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    f = luflop (L, U) ;

    Given L and U, compute:

        Lnz = full (sum (spones (L))) - 1 ;
        Unz = full (sum (spones (U')))' - 1 ;
        f = 2*Lnz*Unz + sum (Lnz) ;

    without allocating O (lunz) space.
*/

#include "mex.h"
#include "SuiteSparse_config.h"

#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif

void mexFunction
(
    int nlhs,			/* number of left-hand sides */
    mxArray *plhs [ ],		/* left-hand side matrices */
    int nrhs,			/* number of right--hand sides */
    const mxArray *prhs [ ]	/* right-hand side matrices */
)
{

    /* ---------------------------------------------------------------------- */
    /* local variables */
    /* ---------------------------------------------------------------------- */

    double flop_count ;
    double *pflop ;
    int64_t *Lp, *Li, *Up, *Ui, *Unz, n, k, row, col, p, Lnz_k, Unz_k ;
    mxArray *Lmatrix, *Umatrix ;

    /* ---------------------------------------------------------------------- */
    /* get inputs L, U */
    /* ---------------------------------------------------------------------- */

    if (nrhs != 2)
    {
	mexErrMsgTxt ("Usage:  f = luflop (L, U)") ;
    }

    Lmatrix = (mxArray *) prhs [0] ;
    Umatrix = (mxArray *) prhs [1] ;

    n = mxGetM (Lmatrix) ;
    if (n != mxGetN (Lmatrix) || n != mxGetM (Umatrix) || n != mxGetN (Umatrix))
    {
	mexErrMsgTxt ("Usage:  f = luflop (L, U) ;  L and U must be square") ;
    }

    if (!mxIsSparse (Lmatrix) || !mxIsSparse (Umatrix))
    {
	mexErrMsgTxt ("Usage:  f = luflop (L, U) ;  L and U must be sparse") ;
    }

    Lp = (int64_t *) mxGetJc (Lmatrix) ;
    Li = (int64_t *) mxGetIr (Lmatrix) ;

    Up = (int64_t *) mxGetJc (Umatrix) ;
    Ui = (int64_t *) mxGetIr (Umatrix) ;

    Unz = (int64_t *) mxMalloc (n * sizeof (int64_t)) ;

    /* ---------------------------------------------------------------------- */
    /* count the nonzeros in each row of U */
    /* ---------------------------------------------------------------------- */

    for (row = 0 ; row < n ; row++)
    {
	Unz [row] = 0 ;
    }
    for (col = 0 ; col < n ; col++)
    {
	for (p = Up [col] ; p < Up [col+1] ; p++)
	{
	    row = Ui [p] ;
	    Unz [row]++ ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* count the flops */
    /* ---------------------------------------------------------------------- */

    flop_count = 0.0 ;
    for (k = 0 ; k < n ; k++)
    {
	/* off-diagonal nonzeros in column k of L: */
	Lnz_k = Lp [k+1] - Lp [k] - 1 ;
	Unz_k = Unz [k] - 1 ;
	flop_count += (2 * Lnz_k * Unz_k) + Lnz_k ;
    }

    /* ---------------------------------------------------------------------- */
    /* return the result */
    /* ---------------------------------------------------------------------- */

    plhs [0] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
    pflop = mxGetPr (plhs [0]) ;
    pflop [0] = flop_count ;
}
