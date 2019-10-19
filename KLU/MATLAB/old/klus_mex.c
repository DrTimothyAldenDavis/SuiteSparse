/* ========================================================================== */
/* === klus mexFunction ===================================================== */
/* ========================================================================== */

/* Solve Ax=b using klu_analyze, klu_factor, and klu_solve
 *
 * [x, Info] = klus (A, b, control) ;
 *
 * [x, Info] = klus (A, b, control, [ ]) ;
 *
 * [x, Info] = klus (A, b, control, [ ]) ;
 *
 * with 4 arguments, the matrix is factorized and then refactorized, just
 * to test the code.
 *
 * b may be n-by-m with m > 1.  It must be dense.
 */

/* ========================================================================== */

#include "klu_internal.h"
#include "tictoc.h"
double dsecnd_ (void) ; 

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double t, *Ax, *Info, tt [2], *B, *X, *W, *Control,
   	   pivot_growth, estimate ;
    int n, *Ap, *Ai, k, nrhs, result ;
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
    klu_common Common ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin > 4 || nargout > 2)
    {
	mexErrMsgTxt ("Usage: [x, Info] = klus (A, b, Control)") ;
    }

    /* get sparse matrix A */
    n = mxGetN (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]) ||
	mxIsComplex (pargin [0]) || n == 0)
    {
    	mexErrMsgTxt ("klus: A must be sparse, square, real, and non-empty") ;
    }
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;

    /* get dense vector B */
    B = mxGetPr (pargin [1]) ;
    nrhs = mxGetN (pargin [1]) ;
    if (mxIsSparse (pargin [1]) || n != mxGetM (pargin [1]) ||
	mxIsComplex (pargin [1]) || nrhs == 0)
    {
    	mexErrMsgTxt (
	    "klus: B must be dense, real, non-empty, and correct dimensions") ;
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
    PRINTF (("control: btf %d ord %d tol %g gro %g inita %g init %g\n",
	 Common.btf, Common.ordering, Common.tol, Common.growth,
	 Common.initmem_amd, Common.initmem)) ;

    /* allocate Info output */
    pargout [1] = mxCreateDoubleMatrix (1, 90, mxREAL) ;
    Info = mxGetPr (pargout [1]) ;
    for (k = 0 ; k < 90 ; k++) Info [k] = -1 ;

    /* ---------------------------------------------------------------------- */
    /* analyze */
    /* ---------------------------------------------------------------------- */

    tt [1] = dsecnd_ ( ) ; 
    /*my_tic (tt) ; */
    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    /*my_toc (tt) ; */
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
    /*my_tic (tt) ; */
    Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    /* result = klu_factor (Ap, Ai, Ax, Symbolic, &Numeric, &Common) ; */
    /* my_toc (tt) ; */
    tt [1] = dsecnd_ ( ) - tt [1] ; 
    Info [37] = tt [1] ;
    /*if (Numeric == (klu_numeric *) NULL)*/
    if (Common.status == KLU_SINGULAR)
    {
	printf("# singular column : %d\n", Common.singular_col) ;
    }
    if (Common.status != KLU_OK)
    {
	mexErrMsgTxt ("klu_factor failed") ;
    }
    
    Info [60] = EMPTY ;

    /* create Info output */
    Info [30] = Numeric->lnz ;		/* nz in L, actual (excl. diagonal) */
    /* printf ("lnz + n  %d\n", Numeric->lnz + n) ; */
    Info [31] = Numeric->unz ;		/* nz in U, actual (excl. diagonal) */
    /* printf ("unz + n  %d\n", Numeric->unz + n) ; */
    Info [36] = Common.noffdiag ;	/* number of off-diagonal pivots */
    /* printf ("nzoff    %d\n", Numeric->noffdiag) ; */
    Info [33] = Numeric->umin ;		/* min abs diagonal entry in U */
    Info [34] = Numeric->umax ;		/* max abs diagonal entry in U */

    /* ---------------------------------------------------------------------- */
    /* refactorize, just to test the code */
    /* ---------------------------------------------------------------------- */

    if (nargin > 3)
    {
	tt [1] = dsecnd_ ( ) ;
	/* my_tic (tt) ;  */
	/* result = klu_refactor (Ap, Ai, Ax, Symbolic, &Common, Numeric) ; */
	result = klu_refactor (Ap, Ai, Ax, Symbolic, Numeric, &Common) ;
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

    pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
    X = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < n*nrhs ; k++) X [k] = B [k] ;

    /* ---------------------------------------------------------------------- */
    /* solve (overwrites right-hand-side with solution) */
    /* ---------------------------------------------------------------------- */

    tt [1] = dsecnd_ ( ) ; 
    /* my_tic (tt) ;*/ 
    result = klu_solve (Symbolic, Numeric, n, nrhs, X, &Common) ;
    /*my_toc (tt) ;*/ 
    tt [1] = dsecnd_ ( ) - tt [1] ; 
    Info [80] = tt [1] ;

    /* get the reciprocal pivot growth */
    result = klu_growth(Ap, Ai, Ax, Symbolic, Numeric, &pivot_growth, &Common);
    Info [81] = pivot_growth ;
   
    /* get the condition no estimate*/
    result = klu_condest(Ap, Ax, Symbolic, Numeric, &estimate,
				      &Common);
    if (Common.status != KLU_OK)
    {
	mexErrMsgTxt ("klu_condest failed") ;
    }
    Info [82] = estimate ;
    
    /* ---------------------------------------------------------------------- */
    /* free Symbolic and Numeric objects, and free workspace */
    /* ---------------------------------------------------------------------- */
    result = klu_free_symbolic (&Symbolic, &Common) ;
    result = klu_free_numeric (&Numeric, &Common) ;
 
}
