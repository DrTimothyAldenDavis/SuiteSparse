/* ========================================================================== */
/* === maxtrans mexFunction ================================================= */
/* ========================================================================== */

#define MIN(a,b) (((a) < (b)) ?  (a) : (b))

/* MAXTRANS: Find a column permutation for a zero-free diagonal.
 *
 * Usage:
 *
 * p = maxtrans (A) ;
 *
 * A (:,p) has a zero-free diagonal unless A is structurally singular.
 * If the matrix is structurally singular, the p will contain zeros.  Similar
 * to p = dmperm (A), except that dmperm returns a row permutation.
 */

/* ========================================================================== */

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
    int nrow, ncol, i, *Ap, *Ai, *Match, nfound, *Work ;
    double *Matchx ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 1)
    {
	mexErrMsgTxt ("Usage: p = maxtrans (A)") ;
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

    /* get output array */
    Match = mxMalloc (nrow * sizeof (int)) ;

    /* get workspace of size 5n (recursive version needs only 2n) */
    Work = mxMalloc (5*ncol * sizeof (int)) ;

    /* ---------------------------------------------------------------------- */
    /* perform the maximum transversal */
    /* ---------------------------------------------------------------------- */

    nfound = maxtrans (nrow, ncol, Ap, Ai, Match, Work) ;
    if (nfound < MIN (nrow, ncol))
    {
	printf ("maxtrans: A is structurally rank deficient\n") ;
    }

    /* ---------------------------------------------------------------------- */
    /* create outputs and free workspace */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateDoubleMatrix (1, nrow, mxREAL) ;
    Matchx = mxGetPr (pargout [0]) ;
    for (i = 0 ; i < nrow ; i++)
    {
	Matchx [i] = Match [i] + 1 ;	/* convert to 1-based */
    }

    mxFree (Work) ;
    mxFree (Match) ;
}
