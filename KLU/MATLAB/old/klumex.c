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

#include "klu.h" 
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
    double *Ax, *Px, *User_Control, *Lx, *Ux, *T, *Qx, *X,
	*LU, *Aimag, *Limag, *Uimag, *Udiag ;
    double *Az, *Lz, *Uz ;
    int n, result, k, anz, s, lnz, unz, j, p, col, noffdiag, nrealloc ;
    int *Ap, *Ai, *Li, *Ui, *P, *Pp, *Pi, 
	*Q, *Work, *Llen, *Ulen, *Lip, *Uip, sing_col ;
    klu_common Common ;

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
    	mexErrMsgTxt ("klu: A must be sparse, square and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    if (mxIsComplex (pargin [0]))
    {
        Aimag = mxGetPi (pargin [0]) ;
        Az = (double *) mxMalloc (Ap[n] * sizeof(double) * 2) ;
        if (!Az)
            mexErrMsgTxt("Malloc failed") ;
        for(j = 0 , k = 0 ; k < Ap[n] ; k++)
        {
	    Az [j++] = Ax [k] ;
	    Az [j++] = Aimag [k] ;
        }
	Ax = Az ;
    }
    anz = Ap [n] ;

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
    klu_kernel_defaults (Control) ;
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

    if (mxIsComplex (pargin [0]))
    {
        X = (double *) mxMalloc (n * sizeof (double) * 2) ;
        Udiag = (double *) mxMalloc (n * sizeof (double) * 2) ;
    }
    else
    {
        X = (double *) mxMalloc (n * sizeof (double)) ;
        Udiag = (double *) mxMalloc (n * sizeof (double)) ;
    }
    Work = (int *) mxMalloc (5*n * sizeof (int)) ;

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    /* my_tic (tt) ; */
    klu_defaults(&Common) ;
    if (mxIsComplex (pargin [0]))
    {
	result = klu_z_kernel_factor (n, Ap, Ai, Ax, Q, (double *) NULL,
		     &LU, Udiag, Llen, Ulen, Lip, Uip, P, &lnz, &unz, 
		     &umin, &umax, X, Work, 
		     /* no BTF or scaling here */
		     0, (int *) NULL, (double *) NULL, (int *) NULL, 
		     (int *) NULL, (double *) NULL, &Common) ;
    }
    else
    {
	result = klu_kernel_factor (n, Ap, Ai, Ax, Q, (double *) NULL,
		     &LU, Udiag, Llen, Ulen, Lip, Uip, P, &lnz, &unz, 
		     &umin, &umax, X, Work, 
		     /* no BTF or scaling here */
		     0, (int *) NULL, (double *) NULL, (int *) NULL, 
		     (int *) NULL, (double *) NULL, &Common);
    }
    /* my_toc (tt) ; */

    if (mxIsComplex (pargin [0]))
    {
        mxFree (Az) ;
    }
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
    if (mxIsComplex (pargin [0]))
    {
    	pargout [0] = mxCreateSparse (n, n, lnz + n, mxCOMPLEX) ;
	Lz = (double *) mxMalloc((lnz + n) * sizeof(double) * 2) ;
    	mxSetJc (pargout [0], Lip) ;
    	Li = mxGetIr (pargout [0]) ;
    	Lx = mxGetPr (pargout [0]) ;
    	Limag = mxGetPi (pargout [0]) ;
    	klu_z_convert (LU, Lip, Llen, Li, Lz, NULL, n) ; 

	for(j = 0 , k = 0 ; k < lnz + n ; k++)
	{
	    Lx [k] = Lz [j++] ;
	    Limag [k] = Lz [j++] ;
	}
    	/* create sparse matrix for U */
    	pargout [1] = mxCreateSparse (n, n, unz + n, mxCOMPLEX) ;
	Uz = (double *) mxMalloc((unz + n) * sizeof(double) * 2) ;
    	mxSetJc (pargout [1], Uip) ;
    	Ui = mxGetIr (pargout [1]) ;
    	Ux = mxGetPr (pargout [1]) ;
    	Uimag = mxGetPi (pargout [1]) ;
    	klu_z_convert (LU, Uip, Ulen, Ui, Uz, Udiag, n) ; 

	for (j = 0 , k = 0 ; k < unz + n ; k++)
	{
	    Ux [k] = Uz [j++] ;
	    Uimag [k] = Uz [j++] ;
	}

        mxFree (Lz) ;
	mxFree (Uz) ;
    }
    else
    {
    	pargout [0] = mxCreateSparse (n, n, lnz + n, mxREAL) ;
    	mxSetJc (pargout [0], Lip);
    	Li = mxGetIr (pargout [0]) ;
    	Lx = mxGetPr (pargout [0]) ;
    	klu_convert (LU, Lip, Llen, Li, Lx, NULL, n) ; 

    	/* create sparse matrix for U */
    	pargout [1] = mxCreateSparse (n, n, unz + n, mxREAL) ;
    	mxSetJc (pargout [1], Uip);
    	Ui = mxGetIr (pargout [1]) ;
    	Ux = mxGetPr (pargout [1]) ;
    	klu_convert (LU, Uip, Ulen, Ui, Ux, Udiag, n) ; 
    }

    mxFree (LU) ; 
    mxFree (Udiag) ;
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

