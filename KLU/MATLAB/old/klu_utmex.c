/* ========================================================================== */
/* === klu_ut mexFunction =================================================== */
/* ========================================================================== */

/* Solves U'x=b where U is from klu, x and b are dense matrices.
 *
 * x = klu_ut (U,b)
 */

/* ========================================================================== */

/* #include "klu.h" */
#include "klu_internal.h"
#include "mex.h"
#include "tictoc.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout[],
    int	nargin,
    const mxArray *pargin[]
)
{
    /* need to fix this interface */
    double *Ux, *X, *B, tt [2], *Udiag ;
    int k, n, *Up, *Ui, nrhs ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 2 || nargout != 1)
    {
	mexErrMsgTxt ("Usage: x = klu_l (U,b)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0])
	|| mxIsComplex (pargin [0]))
    {
    	mexErrMsgTxt ("klu_l: U must be sparse, square, and real") ;
    }

    /* get sparse matrix U */
    Up = mxGetJc (pargin [0]) ;
    Ui = mxGetIr (pargin [0]) ;
    Ux = mxGetPr (pargin [0]) ;

    /* get the right-hand-side b */
    B = mxGetPr (pargin [1]) ;
    nrhs = mxGetN (pargin [1]) ;
    if (mxGetM (pargin [1]) != n)
    {
    	mexErrMsgTxt ("klu_l: b wrong dimension") ;
    }

    /* create the solution, x */
    pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
    X = mxGetPr (pargout [0]) ;

    /* copy the right-hand-side into the solution */
    for (k = 0 ; k < n*nrhs ; k++)
    {
	X [k] = B [k] ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve U'x = b */
    /* ---------------------------------------------------------------------- */

    /* my_tic (tt) ; */
    klu_utsolve (n, Up, Ui, Ux, Udiag, nrhs, 0, X) ;
    /* my_toc (tt) ; */
    printf ("cputime: %g  per rhs: %g\n", tt [1], tt [1]/nrhs) ;
}
