/* ========================================================================== */
/* === klust mexFunction ==================================================== */
/* ========================================================================== */

/* Solve Ax=b using klu_analyze, klu_factor, and klu_tsolve
 *
 * [x, Info] = klust (A, b, control) ;
 *
 * [x, Info] = klust (A, b, control, [ ]) ;
 *
 * [x, Info] = klust (A, b, control, [ ]) ;
 *
 * with 4 arguments, the matrix is factorized and then refactorized, just
 * to test the code.
 *
 * b may be n-by-m with m > 1.  It must be dense.
 */

/* ========================================================================== */

#include "klu_internal.h"
#include "tictoc.h"
#include <complex.h>

double dsecnd_ (void) ;

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double *Az, *Bz ;
    double t, *Ax, *Info, tt [2], *B, *X, *W, *Control, *Aimag, *Bimag, *Xz,
	    *Ximag ;
    int n, j, *Ap, *Ai, k, nrhs ;
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
    klu_common Common ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin > 4 || nargout > 2)
    {
	mexErrMsgTxt ("Usage: [x, Info] = klust (A, b, Control)") ;
    }

    /* get sparse matrix A */
    n = mxGetN (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]) ||
	n == 0)
    {
    	mexErrMsgTxt ("klust: A must be sparse, square and non-empty") ;
    }
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    

    /* get dense vector B */
    B = mxGetPr (pargin [1]) ;
    nrhs = mxGetN (pargin [1]) ;
    if (mxIsSparse (pargin [1]) || n != mxGetM (pargin [1]) ||
	nrhs == 0)
    {
    	mexErrMsgTxt (
	    "klust: B must be dense, non-empty and correct dimensions") ;
    }
    if (mxIsComplex (pargin [0]))
    {
	printf("complex case\n") ;
        Aimag = mxGetPi (pargin [0]) ;
        Az = (double *)mxMalloc (2 * Ap[n] * sizeof(double)) ;
        if (!Az)
	{
            mexErrMsgTxt("Malloc failed") ;
	}
        for(j = 0 , k = 0 ; k < Ap[n] ; k++)
        {
            Az [j++] = Ax [k] ;
	    Az [j++] = Aimag [k] ;
	}		
    
	Bz = (double *) mxMalloc(2 * n * nrhs * sizeof(double)) ;
	if (! Bz)
	{
	    mexErrMsgTxt("Malloc failed") ;
	}
	if (mxIsComplex (pargin [1]))
	{
	    Bimag = mxGetPi (pargin [1]) ;
	    for (j = 0 , k = 0 ; k < (n * nrhs) ; k++)
	    {
		Bz [j++] = B [k] ;
	       	Bz [j++] = Bimag [k] ;
	    }
	}
	else
	{
	    for (j = 0 , k = 0 ; k < (n * nrhs) ; k++)
	    {
		Bz [j++] = B [k] ;
	       	Bz [j++] = 0 ;
	    }
	    
	}
	
    }
    /* get control parameters */
    klu_defaults (&Common) ;
    if (nargin > 2)
    {
	int s ;
	if (!mxIsDouble (pargin [2]))
	{
	    mexErrMsgTxt ("klu: control must be real") ;
	}
	Control = mxGetPr (pargin [2]) ;
	s = mxGetNumberOfElements (pargin [2]) ;
	if (s > 0) Common.tol           = Control [0] ;
	if (s > 1) Common.growth        = Control [1] ;
	if (s > 2) Common.initmem_amd   = Control [2] ;
	if (s > 3) Common.initmem       = Control [3] ;
	if (s > 4) Common.btf           = Control [4] ;
	if (s > 5) Common.ordering      = Control [5] ;
	if (s > 6) Common.scale         = Control [6] ;
	if (s > 7) Common.singular_proc = Control [7] ;
    }

    PRINTF (("control: tol %g gro %g inita %g init %g btf %d ord %d scale %d \
	     sing %d\n",
	     Common.tol, Common.growth, Common.initmem_amd, Common.initmem,
	     Common.btf, Common.ordering, Common.scale, Common.singular_proc
	   )) ;
    /* allocate Info output */
    pargout [1] = mxCreateDoubleMatrix (1, 90, mxREAL) ;
    Info = mxGetPr (pargout [1]) ;
    for (k = 0 ; k < 90 ; k++) Info [k] = -1 ;

    /* ---------------------------------------------------------------------- */
    /* analyze */
    /* ---------------------------------------------------------------------- */

    tt [1] = dsecnd_ ( ) ;

    /* my_tic (tt) ; */
    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    /* my_toc (tt) ; */
    tt [1] = dsecnd_ ( ) - tt [1] ;
    Info [9] = tt [1] ;
    if (Symbolic == (klu_symbolic *) NULL)
    {
	mexErrMsgTxt ("klu_analyze failed") ;
    }

    Info [ 1] = Symbolic->n ;		/* n, dimension of input matrix */
    Info [ 2] = Symbolic->nz ;		/* # entries in input matrix */
    Info [ 3] = Symbolic->nblocks ;	/* # of blocks in BTF form */
    Info [ 4] = Symbolic->maxblock ;	/* dimension of largest block */
    Info [ 7] = Symbolic->nzoff ;	/* nz in off-diagonal blocks of A */
    Info [ 8] = Symbolic->symmetry ;	/* symmetry of largest block */
    Info [10] = Symbolic->lnz ;		/* nz in L, estimated (incl diagonal) */
    Info [11] = Symbolic->unz ;		/* nz in U, estimated (incl diagonal) */
    Info [12] = Symbolic->est_flops ;	/* est. factorization flop count */

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    tt [1] = dsecnd_ ( ) ;
    /* my_tic (tt) ; */
    if (mxIsComplex (pargin [0]))
    {
	/* result = klu_z_factor (Ap, Ai, (double *) Az, Symbolic, &Numeric, 
				   &Common) ; */
	Numeric = klu_z_factor (Ap, Ai, (double *) Az, Symbolic, &Common) ;
    }
    else
    {
	/* result = klu_factor (Ap, Ai, Ax, Symbolic, &Numeric, &Common) ; */
	Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    }
    /* my_toc (tt) ; */
    tt [1] = dsecnd_ ( ) - tt [1] ;
    Info [37] = tt [1] ;
    /*if (Numeric == (klu_numeric *) NULL)*/
    if (Common.status != KLU_OK)
    {
	mexErrMsgTxt ("klu_factor failed") ;
    }
    Info [60] = EMPTY ;

    /* create Info output */
    Info [30] = Numeric->lnz ;		/* nz in L, actual (incl. diagonal) */
    Info [31] = Numeric->unz ;		/* nz in U, actual (incl. diagonal) */
    Info [36] = Common.noffdiag ;	/* number of off-diagonal pivots */
    Info [33] = Numeric->umin ;		/* min abs diagonal entry in U */
    Info [34] = Numeric->umax ;		/* max abs diagonal entry in U */

    /* ---------------------------------------------------------------------- */
    /* refactorize, just to test the code */
    /* ---------------------------------------------------------------------- */

    if (nargin > 3)
    {
	tt [1] = dsecnd_ ( ) ;
	/*my_tic (tt) ; */
	if (mxIsComplex(pargin [0]))
	{
	    klu_z_refactor (Ap, Ai, (double *) Az, Symbolic, 
					 Numeric, &Common) ;
	}
	else
	{
	    klu_refactor (Ap, Ai, Ax, Symbolic, 
			       Numeric, &Common) ;
	}
	/* my_toc (tt) ; */
	tt [1] = dsecnd_ ( ) - tt [1] ;
	Info [60] = tt [1] ;
	if (Common.status != KLU_OK)
	{
	    mexErrMsgTxt ("klu_refactor failed") ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* allocate outputs and set X=B */
    /* ---------------------------------------------------------------------- */
    if (mxIsComplex (pargin [0]))
    {
	pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxCOMPLEX) ;
    }
    else
    {
	pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
	X = mxGetPr (pargout [0]) ;
	for (k = 0 ; k < n*nrhs ; k++) X [k] = B [k] ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve (overwrites right-hand-side with solution) */
    /* ---------------------------------------------------------------------- */

    tt [1] = dsecnd_ ( ) ;
    /*my_tic (tt) ; */
    if (mxIsComplex (pargin [0]))
    {
	/* do a conjugate transpose solve */
	printf("complex transpose solve\n") ;
	klu_z_tsolve (Symbolic, Numeric, n, nrhs, Bz, 1, &Common) ;
    }
    else
    {
	klu_tsolve (Symbolic, Numeric, n, nrhs, X, 0, &Common) ;
    }
    /* my_toc (tt) ; */
    tt [1] = dsecnd_ ( ) - tt [1] ;
    Info [80] = tt [1] ;

    /* ---------------------------------------------------------------------- */
    /* free Symbolic and Numeric objects, and free workspace */
    /* ---------------------------------------------------------------------- */

    klu_free_symbolic (&Symbolic, &Common) ;
    if (mxIsComplex (pargin [0]))
    {
	Xz = mxGetPr (pargout [0]) ;
	Ximag = mxGetPi (pargout [0]) ;
	for (j = 0, k = 0 ; k < n * nrhs ; k++)
	{
	    Xz [k] = Bz [j++] ;
	    Ximag [k] = Bz [j++] ;
	}
	klu_z_free_numeric (&Numeric, &Common) ;
	mxFree(Az) ;
        mxFree(Bz) ;
    }
    else
    {
	klu_free_numeric (&Numeric, &Common) ;
    }
}
