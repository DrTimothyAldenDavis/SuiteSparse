//------------------------------------------------------------------------------
// LDL/MATLAB/ldlsymbolmex.c: MATLAB interface for LDL symbolic analysis
//------------------------------------------------------------------------------

// LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

/* MATLAB interface for symbolic LDL' factorization using the LDL sparse matrix
 * package.  This mexFunction is not required by the LDL mexFunction.
 *
 * MATLAB calling syntax is:
 *
 *       [Lnz, Parent, flopcount] = ldlsymbol (A)
 *       [Lnz, Parent, flopcount] = ldlsymbol (A, P)
 *
 * P is a permutation of 1:n, an output of AMD, SYMAMD, or SYMRCM, for example.
 * Only the diagonal and upper triangular part of A or A(P,P) is accessed; the
 * lower triangular part is ignored.
 *
 * The elimination tree is returned in the Parent array.  The number of nonzeros
 * in each column of L is returned in Lnz.  This mexFunction replicates the
 * following MATLAB computations, using ldl_l_symbolic:
 *
 *	Lnz = symbfact (A) - 1 ;
 *	Parent = etree (A) ;
 *	flopcount = sum (Lnz .* (Lnz + 2)) ;
 *
 * or, if P is provided,
 *
 *	B = A (P,P) ;
 *	Lnz = symbfact (B) - 1 ;
 *	Parent = etree (B) ;
 *	flopcount = sum (Lnz .* (Lnz + 2)) ;
 *
 * This code is faster than the above MATLAB statements, typically by a factor
 * of 4 to 40 (median speedup of 9) in MATLAB 6.5 on a Pentium 4 Linux laptop
 * (excluding the B=A(P,P) time), on a wide range of symmetric sparse matrices.
 */

#include "ldl.h"
#include "mex.h"

/* ========================================================================== */
/* === LDLSYMBOL mexFunction ================================================ */
/* ========================================================================== */

void mexFunction
(
    int	nargout,
    mxArray *pargout[ ],
    int	nargin,
    const mxArray *pargin[ ]
)
{
    int64_t i, n, *Pattern, *Flag, *Lp, *Ap, *Ai, *Lnz, *Parent,
	*P, *Pinv, nn, k, j ;
    int permute ;
    double flops, *p ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin == 0 || nargin > 2)
    {
	mexErrMsgTxt ("Usage:\n"
	    "  [Lnz, Parent, flopcount] = ldl (A) ;\n"
	    "  [Lnz, Parent, flopcount] = ldl (A, P) ;\n") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0])
	    || mxIsComplex (pargin [0]))
    {
    	mexErrMsgTxt ("ldlsymbol: A must be sparse, square, and real") ;
    }

    nn = (n == 0) ? 1 : n ;

    /* get sparse matrix A */
    Ap = (int64_t *) mxGetJc (pargin [0]) ;
    Ai = (int64_t *) mxGetIr (pargin [0]) ;

    /* get fill-reducing ordering, if present */
    permute = ((nargin > 1) && !mxIsEmpty (pargin [1])) ;
    if (permute)
    {
	if (mxGetM (pargin [1]) * mxGetN (pargin [1]) != n ||
		mxIsSparse (pargin [1]))
	{
	    mexErrMsgTxt ("ldlsymbol: invalid input permutation\n") ;
	}
	P    = (int64_t *) mxMalloc (nn * sizeof (int64_t)) ;
	Pinv = (int64_t *) mxMalloc (nn * sizeof (int64_t)) ;
	p = mxGetPr (pargin [1]) ;
	for (k = 0 ; k < n ; k++)
	{
	    P [k] = p [k] - 1 ;	/* convert to 0-based */
	}
    }
    else
    {
	P    = (int64_t *) NULL ;
	Pinv = (int64_t *) NULL ;
    }

    /* allocate first part of L */
    Lp      = (int64_t *) mxMalloc ((n+1) * sizeof (int64_t)) ;
    Parent  = (int64_t *) mxMalloc (nn * sizeof (int64_t)) ;

    /* get workspace */
    Flag    = (int64_t *) mxMalloc (nn * sizeof (int64_t)) ;
    Pattern = (int64_t *) mxMalloc (nn * sizeof (int64_t)) ;
    Lnz     = (int64_t *) mxMalloc (nn * sizeof (int64_t)) ;

    /* make sure the input P is valid */
    if (permute && !ldl_l_valid_perm (n, P, Flag))
    {
	mexErrMsgTxt ("ldlsymbol: invalid input permutation\n") ;
    }

    /* note that we assume that the input matrix is valid */

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization to get Lp, Parent, Lnz, and optionally Pinv */
    /* ---------------------------------------------------------------------- */

    ldl_l_symbolic (n, Ap, Ai, Lp, Parent, Lnz, Flag, P, Pinv) ;

    /* ---------------------------------------------------------------------- */
    /* create outputs */
    /* ---------------------------------------------------------------------- */

    /* create the output Lnz vector */
    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    p = mxGetPr (pargout [0]) ;
    for (j = 0 ; j < n ; j++)
    {
	p [j] = Lnz [j] ;
    }

    /* return elimination tree (add 1 to change from 0-based to 1-based) */
    if (nargout > 1)
    {
	pargout [1] = mxCreateDoubleMatrix (1, n, mxREAL) ;
	p = mxGetPr (pargout [1]) ;
	for (i = 0 ; i < n ; i++)
	{
	    p [i] = Parent [i] + 1 ;
	}
    }

    if (nargout > 2)
    {
	/* find flop count for ldl_l_numeric */
	flops = 0 ;
	for (k = 0 ; k < n ; k++)
	{
	    flops += ((double) Lnz [k]) * (Lnz [k] + 2) ;
	}
	pargout [2] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
	p = mxGetPr (pargout [2]) ;
	p [0] = flops ;
    }

    if (permute)
    {
	mxFree (P) ;
	mxFree (Pinv) ;
    }
    mxFree (Lp) ;
    mxFree (Parent) ;
    mxFree (Flag) ;
    mxFree (Pattern) ;
    mxFree (Lnz) ;
}
