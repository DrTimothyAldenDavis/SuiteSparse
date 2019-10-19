
/* Usage
 *
 * [p,q,cp,ssize] = dp (A)
 *
 */

#define MIN(a,b) (((a) < (b)) ?  (a) : (b))

#include "mex.h"
#include "btf.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    int nrow, ncol, i, *Ap, *Ai, *ATp, *ATi, cp [5], rp [5], *P, *Q, k ;
    double *x ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin != 2 || nargout > 4)
    {
	mexErrMsgTxt ("Usage: [p,q,cp,rp] = dp (A,A')") ;
    }
    nrow = mxGetM (pargin [0]) ;
    ncol = mxGetN (pargin [0]) ;
    if (!mxIsSparse (pargin [0]))
    {
    	mexErrMsgTxt ("maxtrans: A must be sparse, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;

    /* get sparse matrix AT */
    ATp = mxGetJc (pargin [1]) ;
    ATi = mxGetIr (pargin [1]) ;

    P = mxMalloc (nrow * sizeof (int)) ;
    Q = mxMalloc (ncol * sizeof (int)) ;

    printf ("nrow %d ncol %d\n", nrow, ncol) ;
    dmperm (nrow, ncol, Ap, Ai, ATp, ATi, P, Q, cp, rp) ;

    /* ---------------------------------------------------------------------- */
    /* create outputs and free workspace */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateDoubleMatrix (1, nrow, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < nrow ; k++)
    {
	x [k] = P [k] + 1 ;
    }

    pargout [1] = mxCreateDoubleMatrix (1, ncol, mxREAL) ;
    x = mxGetPr (pargout [1]) ;
    for (k = 0 ; k < ncol ; k++)
    {
	x [k] = Q [k] + 1 ;
    }

    pargout [2] = mxCreateDoubleMatrix (1, 5, mxREAL) ;
    x = mxGetPr (pargout [2]) ;
    for (k = 0 ; k < 5 ; k++)
    {
	x [k] = cp [k] + 1 ;
    }

    pargout [3] = mxCreateDoubleMatrix (1, 5, mxREAL) ;
    x = mxGetPr (pargout [3]) ;
    for (k = 0 ; k < 5 ; k++)
    {
	x [k] = rp [k] + 1 ;
    }

    mxFree (P) ;
    mxFree (Q) ;
}
