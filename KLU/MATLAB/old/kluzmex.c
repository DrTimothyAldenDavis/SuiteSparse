/* ========================================================================== */
/* === klu mexFunction ====================================================== */
/* ========================================================================== */

/* Factorizes A(p,q) into L*U.  Usage:
 *
 * [L,U,p,t] = klu (A)			% Q assumed to be identity
 * [L,U,p,t] = klu (A, q)
 * [L,U,p,t] = klu (A, q, Control)
 *
 * Control (1): threshold partial pivoting tolerance (0 means force diagonal
 *		pivoting, 1 means conventional partial pivoting.  A value of
 *		0.5 means the diagonal will be picked if its absolute value
 *		is >= 0.5 times the largest value in the column; otherwise
 *		the largest value is selected.  Default: 1.0.
 * Control (2): if positive, this is the initial size of the L matrix, in # of
 *		nonzero entries.  If negative, the initial size of L is
 *		(-Control (2) * nnz (A)).  Default: -10.
 * Control (3): if positive, this is the initial size of the U matrix, in # of
 *		nonzero entries.  If negative, the initial size of U is
 *		(-Control (2) * nnz (A)).  Default: -10.
 * Control (4): memory growth factor
 *
 * t = [cputime noffdiag umin umax], output statistics.
 *
 * If Control is not present, or present but not of the correct size,
 * defaults are used.
 *
 * If klu is compiled with the default options (NRECIPROCAL not defined), then
 * the diagonal of U is returned inverted on output.
 */

/* ========================================================================== */

/* #include "klu.h" */
#include "klu_internal.h"
#include "mex.h"
#include "tictoc.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double Control [KLU_CONTROL], tt [2], umin, umax ;
    double *Ax, *Px, *User_Control, *Lx, *Ux, *T, *Qx ;
    _Complex double *Atemp, *LU, *X ;
    int n, result, k, anz, s, lnz, unz, j, p, col, noffdiag, nrealloc ;
    int *Ap, *Ai, *Li, *Ui, *P, *Pp, *Pi, 
	*Q, *Work, *Llen, *Ulen, *Lip, *Uip, sing_col ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 3 || !(nargout == 3 || nargout == 4))
    {
	mexErrMsgTxt ("Usage: [L,U,P,t] = klu (A,Q,Control), where t, Q, and Control are optional.") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0])
	|| n == 0)
    {
    	mexErrMsgTxt ("klu: A must be sparse, square, complex, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    Az = mxGetPi (pargin [0]) ;
    anz = Ap [n] ;
    Atemp = mxMalloc(anz * sizeof(_Complex double)) ;
    if (!Atemp)		    
        mexErrMsgTxt("malloc failed") ;
    for (k = 0; k < anz ; k++)
        Atemp[k] = Ax [k] + Az [k] * _Complex_I ;
    /* get input column permutation Q */
    if (nargin > 1)
    {
	if (!mxIsDouble (pargin [1]) || n != mxGetNumberOfElements (pargin [1]))
	{
	    mexErrMsgTxt ("klu: Q must be a dense 1-by-n vector") ; 
	}
	Qx = mxGetPr (pargin [1]) ;
	Q = (int *) mxMalloc (n * sizeof (int)) ;
	for (k = 0 ; k < n ; k++)
	{
	    col = (int) (Qx [k]) - 1 ;	/* convert from 1-based to 0-based */
	    if (col < 0 || col >= n)
	    {
		mexErrMsgTxt ("klu: Q not a valid permutation\n") ;
	    }
	    Q [k] = col ;
	}
    }
    else
    {
	/* klu will assume that Q is the identity permutation */
	Q = (int *) NULL ;
    }

    /* get control parameters */
    klu_defaults (Control) ;
    if (nargin > 2)
    {
	if (!mxIsDouble (pargin [2]))
	{
	    mexErrMsgTxt ("klu: Control must be real") ;
	}
	User_Control = mxGetPr (pargin [2]) ;
	s = mxGetNumberOfElements (pargin [2]) ;
	for (k = 0 ; k < s ; k++)
	{
	    Control [k] = User_Control [k] ;
	}
    }

    P  = (int *) mxMalloc (n * sizeof (int)) ;
    Llen = (int *) mxMalloc (n * sizeof (int)) ;
    Ulen = (int *) mxMalloc (n * sizeof (int)) ;
    Lip = (int *) mxMalloc ((n+1) * sizeof (int)) ;
    Uip = (int *) mxMalloc ((n+1) * sizeof (int)) ;

    X    = (_Complex double *) mxMalloc (n * sizeof (_Complex double)) ;
    Work = (int *) mxMalloc (5*n * sizeof (int)) ;

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    /* my_tic (tt) ; */

    result = klu_factor (n, Ap, Ai, Atemp, Q, (double *) NULL,
	    	 &LU, Llen, Ulen, Lip, Uip, P, &lnz, &unz, 
		 &noffdiag, &umin, &umax, &nrealloc, &sing_col, X, Work, 
	    	 /* no BTF or scaling here */
	    	 0, (int *) NULL, (double *) NULL, 0, 0, (int *) NULL, (int *) NULL,
	    	 (double *) NULL) ;

    /* my_toc (tt) ; */

    mxFree (X) ;
    mxFree (Work) ;

    if (nargout == 4)
    {
	pargout [3] = mxCreateDoubleMatrix (1, 4, mxREAL) ;
	T = mxGetPr (pargout [3]) ;
	T [0] = tt [1] ;
	T [1] = noffdiag ;
	T [2] = umin ;
	T [3] = umax ;
    }

    if (result == KLU_OUT_OF_MEMORY)
    {
	mexErrMsgTxt ("klu: out of memory") ;
    }

    /* NOTE: this should be done in a better way, without copying, but when
     * I tried to do it I got segmentation faults, and gave up ... */

    /* create sparse matrix for L */
    pargout [0] = mxCreateSparse (n, n, lnz, mxREAL) ;
    mxSetJc (pargout [0], Lip);
    Li = mxGetIr (pargout [0]) ;
    Lx = mxGetPr (pargout [0]) ;
    KLU_z_convert (LU, Lip, Llen, Li, Lx, n) ; 

    /* create sparse matrix for U */
    pargout [1] = mxCreateSparse (n, n, unz, mxREAL) ;
    mxSetJc (pargout [1], Uip);
    Ui = mxGetIr (pargout [1]) ;
    Ux = mxGetPr (pargout [1]) ;
    KLU_z_convert (LU, Uip, Ulen, Ui, Ux, n) ; 

    mxFree (LU) ; 
    mxFree (Llen) ; 
    mxFree (Ulen) ; 

    /* create permutation vector for P */
    pargout [2] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Px = mxGetPr (pargout [2]) ;
    for (k = 0 ; k < n ; k++)
    {
	Px [k] = P [k] + 1 ;	/* convert to 1-based */
    }

    mxFree (P) ;

}
