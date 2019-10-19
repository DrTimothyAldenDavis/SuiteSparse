/* ========================================================================== */
/* === btf mexFunction ====================================================== */
/* ========================================================================== */

/* BTF: Permute a matrix to upper block triangular form with a zero-free
 * diagonal.
 *
 * Usage:
 *
 *	[p,q,r] = btf (A) ;
 *
 * This is essentially identical to
 *
 *	[p,q,r] = dmperm (A)
 *
 * except that p, q, and r will differ.  Both return an upper block triangular
 * form with a zero-free diagonal.  The number and sizes of the blocks will be
 * identical, but the order of the blocks, and the ordering within the blocks,
 * can be different.  If the matrix is structurally singular, both strongcomp
 * and maxtrans return a vector q containing negative entries.  abs(q) is a
 * permutation of 1:n, and find(q<0) gives a list of the indices of the
 * diagonal of A(p,q) that are zero.  That is, C(i,i) is zero if i is in the
 * list find(q<0).
 *
 * Copyright (c) 2004.  Tim Davis, May, 2004, University of Florida,
 * with support from Sandia National Laboratories.  All Rights Reserved.
 *
 * See also maxtrans, strongcomp, dmperm
 */

/* ========================================================================== */

#include "mex.h"
#include "btf.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout[],
    int	nargin,
    const mxArray *pargin[]
)
{
    int b, n, i, k, j, *Ap, *Ai, *P, *R, nblocks, *Work, *Q, nfound ;
    double *Px, *Rx, *Qx ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargout > 3)
    {
	mexErrMsgTxt ("Usage: [p,q,r] = btf (A)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0]))
    {
    	mexErrMsgTxt ("btf: A must be sparse, square, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;

    /* get output arrays */
    Q = mxMalloc (n * sizeof (int)) ;
    P = mxMalloc (n * sizeof (int)) ;
    R = mxMalloc ((n+1) * sizeof (int)) ;

    /* get workspace */
    Work = mxMalloc (5*n * sizeof (int)) ;

    /* ---------------------------------------------------------------------- */
    /* find the permutation to BTF */
    /* ---------------------------------------------------------------------- */

    nblocks = btf_order (n, Ap, Ai, P, Q, R, &nfound, Work) ;

    /* ---------------------------------------------------------------------- */
    /* create outputs and free workspace */
    /* ---------------------------------------------------------------------- */

    /* create P */
    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Px = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < n ; k++)
    {
	Px [k] = P [k] + 1 ;	/* convert to 1-based */
    }

    /* create Q */
    pargout [1] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Qx = mxGetPr (pargout [1]) ;
    for (k = 0 ; k < n ; k++)
    {
	Qx [k] = Q [k] + 1 ;	/* convert to 1-based */
    }

    /* create R */
    pargout [2] = mxCreateDoubleMatrix (1, nblocks+1, mxREAL) ;
    Rx = mxGetPr (pargout [2]) ;
    for (b = 0 ; b <= nblocks ; b++)
    {
	Rx [b] = R [b] + 1 ;	/* convert to 1-based */
    }

    mxFree (P) ;
    mxFree (R) ;
    mxFree (Work) ;
    mxFree (Q) ;
}
