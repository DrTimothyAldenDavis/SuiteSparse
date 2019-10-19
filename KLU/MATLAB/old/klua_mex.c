/* ========================================================================== */
/* === klua mexFunction ===================================================== */
/* ========================================================================== */

/* Order A using BTF and AMD (in klu_analyze)
 *
 * [P,Q,R,Lnz,Info] = klua (A)
 * [P,Q,R,Lnz,Info] = klua (A, Control)
 *
 * A (P,Q) is in upper block triangular form, with each block ordered via
 * AMD.  The kth block is in row/column range R (k) to R (k+1)-1.  The nz
 * estimate of the L factor of the kth block is Lnz (k).
 */

/* ========================================================================== */

#include "klu_internal.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout[],
    int	nargin,
    const mxArray *pargin[]
)
{
    int n, *Ap, *Ai, k ;
    double *Px, *Qx, *Rx, *Lnzx, *Lnz, *Info, *Control ;
    klu_common Common ;
    klu_symbolic *Symbolic ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin > 2 || nargout != 5)
    {
	mexErrMsgTxt ("Usage: [P,Q,R,Lnz,Info] = klua (A,Control)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0]))
    {
    	mexErrMsgTxt ("klua: A must be sparse, square, real, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;

    /* get control parameters */
    klu_defaults (&Common) ;
    if (nargin > 1)
    {
	int s ;
	if (!mxIsDouble (pargin [1]))
	{
	    mexErrMsgTxt ("klu: control must be real") ;
	}
	Control = mxGetPr (pargin [1]) ;
	s = mxGetNumberOfElements (pargin [1]) ;
	/* if (s > 0) control.prl      = Control [0] ; */
	if (s > 1) Common.btf         = Control [1] ;
	if (s > 2) Common.scale       = Control [2] ;
	if (s > 3) Common.ordering    = Control [3] ;
	if (s > 4) Common.tol         = Control [4] ;
	if (s > 5) Common.growth      = Control [5] ;
	if (s > 6) Common.initmem_amd = Control [6] ;
	if (s > 7) Common.initmem     = Control [7] ;
    }
    PRINTF (("control: btf %d ord %d tol %g gro %g inita %g init %g\n",
	 Common.btf, Common.ordering, Common.tol, Common.growth,
	 Common.initmem_amd, Common.initmem)) ;

    /* ---------------------------------------------------------------------- */
    /* analyze */
    /* ---------------------------------------------------------------------- */

    PRINTF (("calling klu_analyze\n")) ;
    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    PRINTF (("klu_analyze done\n")) ;

    if (Symbolic == (klu_symbolic *) NULL)
    {
	mexErrMsgTxt ("klua: error") ;
    }

    /* ---------------------------------------------------------------------- */
    /* return the results to MATLAB */
    /* ---------------------------------------------------------------------- */

    /* create Info output */
    pargout [4] = mxCreateDoubleMatrix (1, 90, mxREAL) ;
    Info = mxGetPr (pargout [4]) ;
    for (k = 0 ; k < 90 ; k++) Info [k] = -1 ;

    Info [ 1] = Symbolic->n ;		/* n, dimension of input matrix */
    Info [ 2] = Symbolic->nz ;		/* # entries in input matrix */
    Info [ 3] = Symbolic->nblocks ;	/* # of blocks in BTF form */
    Info [ 4] = Symbolic->maxblock ;	/* dimension of largest block */
    Info [ 7] = Symbolic->nzoff ;	/* nz in off-diagonal blocks of A */
    Info [ 8] = Symbolic->symmetry ;	/* symmetry of largest block */
    Info [10] = Symbolic->lnz ;		/* nz in L, estimated (incl diagonal) */
    Info [11] = Symbolic->unz ;		/* nz in U, estimated (incl diagonal) */
    Info [12] = Symbolic->est_flops ;	/* est. factorization flop count */

    /* create permutation vector for P */
    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Px = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < n ; k++)
    {
	Px [k] = Symbolic->P [k] + 1 ;	/* convert to 1-based */
    }

    /* create permutation vector for Q */
    pargout [1] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Qx = mxGetPr (pargout [1]) ;
    for (k = 0 ; k < n ; k++)
    {
	Qx [k] = Symbolic->Q [k] + 1 ;	/* convert to 1-based */
    }

    /* create vector for R */
    pargout [2] = mxCreateDoubleMatrix (1, Symbolic->nblocks+1, mxREAL) ;
    Rx = mxGetPr (pargout [2]) ;
    for (k = 0 ; k <= Symbolic->nblocks ; k++)
    {
	Rx [k] = Symbolic->R [k] + 1 ;	/* convert to 1-based */
    }

    /* create vector for Lnz */
    pargout [3] = mxCreateDoubleMatrix (1, Symbolic->nblocks, mxREAL) ;
    Lnzx = mxGetPr (pargout [3]) ;
    for (k = 0 ; k < Symbolic->nblocks ; k++)
    {
	Lnzx [k] = Symbolic->Lnz [k] ;
    }

    PRINTF (("Symbolic n %d nzoff %d nblocks %d maxblock %d\n",
	Symbolic->n, Symbolic->nzoff, Symbolic->nblocks,
	Symbolic->maxblock)) ;

    klu_free_symbolic (&Symbolic, &Common) ;
}
