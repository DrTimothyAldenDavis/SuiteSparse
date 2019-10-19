/* ========================================================================== */
/* === klu_lt mexFunction =================================================== */
/* ========================================================================== */

/* Solves L'x=b where L is from klu, x and b are dense matrices.
 *
 * x = klu_lt (L,b)
 */

/* ========================================================================== */

/* #include "klu.h" */
#include "klu_internal.h"
#include "mex.h"
#include "tictoc.h"

#define UNITD(type, n) (int) ceil((sizeof(type) * n) / sizeof(double))

void mexFunction
(
    int	nargout,
    mxArray *pargout[],
    int	nargin,
    const mxArray *pargin[]
)
{
    /* need to fix this interface */
    double *Lx, *X, *B, *LU, *Lxnew, tt [2] ;
    int *Lp, *Li, *Llen, *Lip, *Linew ;
    int k, n, p, j, nrhs, size, count, pend ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 2 || nargout != 1)
    {
	mexErrMsgTxt ("Usage: x = klu_l (L,b)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0])
	|| mxIsComplex (pargin [0]))
    {
    	mexErrMsgTxt ("klu_l: L must be sparse, square, and real") ;
    }

    /* get sparse matrix L */
    Lp = mxGetJc (pargin [0]) ;
    Li = mxGetIr (pargin [0]) ;
    Lx = mxGetPr (pargin [0]) ;

    /* get the right-hand-side b */
    B = mxGetPr (pargin [1]) ;
    nrhs = mxGetN (pargin [1]) ;
    if (mxGetM (pargin [1]) != n)
    {
    	mexErrMsgTxt ("klu_l: b wrong dimension") ;
    }
    /* need to convert into the compressed index-value data structure */
    /* create the Llen array */
    Llen = (int *) mxMalloc(n * sizeof(int)) ;
    if (!Llen)
    {
    	mexErrMsgTxt ("klu_l: malloc failed") ;
    }
    size = 0 ;
    for (k = 0 ; k < n ; k++)
    {
	Llen [k] = Lp [k+1] - Lp [k] ;
	size += Llen [k] ;
    }
    /* size = total no of elements in L */
    size += UNITD(int, size) ;
    LU = (double *) mxMalloc(size * sizeof(double)) ;
    if (! LU)
    {
	mxFree(Llen) ;
    	mexErrMsgTxt ("klu_l: malloc failed") ;
    }
    Lip = (int *) mxMalloc((n+1) * sizeof(int)) ;
    if (! Lip)
    {
	mxFree(Llen) ;
	mxFree(LU) ;
    	mexErrMsgTxt ("klu_l: malloc failed") ;
    }

    count = 0 ;
    for (k= 0 ; k < n ; k++)
    {
	/* store the indices and values in LU and modify the Lp
	 * pointer */
	Lip [k] = count ;
	Linew = (int *) (LU + count) ;
	/*Lxnew = (double *) (LU + count + UNITD (int, Llen [k])) ;*/
	Lxnew = (double *) (LU + count + UNITD(int, Llen [k])) ;
	pend = Lp [k + 1] ;
	mxAssert (Llen [k] == Lp [k + 1] - Lp [k], "") ;
	for (j = 0, p = Lp [k] ; p < pend ; p++, j++)
	{
	    Linew [j] = Li [p] ;
	    Lxnew [j] = Lx [p] ;
	}
	count += Llen [k] + UNITD(int, Llen [k]) ;
    }
    Lip [n] = count ;
    
    /* create the solution, x */
    pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
    X = mxGetPr (pargout [0]) ;

    /* copy the right-hand-side into the solution */
    for (k = 0 ; k < n*nrhs ; k++)
    {
	X [k] = B [k] ;
    }


    /* ---------------------------------------------------------------------- */
    /* solve Lx = b */
    /* ---------------------------------------------------------------------- */

    /* my_tic (tt) ; */
    printf ("b4 calling ltsolve\n") ;
    /* klu_ltsolve (n, Lp, Li, Lx, nrhs, 0, X) ; */
    klu_ltsolve (n, Lip, Llen, LU, nrhs, 0, X) ;
    /* my_toc (tt) ; */
    printf ("cputime: %g  per rhs: %g\n", tt [1], tt [1]/nrhs) ;
    mxFree(Llen) ;
    mxFree(LU) ;
    mxFree(Lip) ;
}
