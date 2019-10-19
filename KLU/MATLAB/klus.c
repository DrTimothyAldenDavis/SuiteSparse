/* ========================================================================== */
/* === klus mexFunction ===================================================== */
/* ========================================================================== */

/* Solve Ax=b using KLU, with ordering provided by CHOLMOD 
 *
 * [x, Info] = klus (A, b)	    order A+A' for each block
 * [x, Info] = klus (A, b, 0)	    order A'A for each block
 *
 * b may be n-by-m with m > 1.  It must be dense.
 */

/* ========================================================================== */

#include "klu.h"
#include "klu_cholmod.h"
#include "mex.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double condest ;
    double *Ax, *Info, *B, *X ;
    int n, *Ap, *Ai, k, nrhs, result, symmetric ;
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
    klu_common Common ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 2 || nargin > 3 || nargout > 2)
    {
	mexErrMsgTxt ("Usage: [x, Info] = klus (A, b, options)") ;
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

    /* get options */
    symmetric = (nargin == 2) || (mxGetScalar (pargin [2])) ;

    /* get control parameters */
    klu_defaults (&Common) ;
    Common.ordering = 3 ;
    Common.user_order = klu_cholmod ;
    Common.user_data = &symmetric ;

    /* hack
    Common.btf = 0 ; printf ("btf off\n") ;
    */

    /* allocate Info output */
    pargout [1] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    Info = mxGetPr (pargout [1]) ;
    for (k = 0 ; k < 3 ; k++) Info [k] = -1 ;

    /* ---------------------------------------------------------------------- */
    /* analyze */
    /* ---------------------------------------------------------------------- */

    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    if (Symbolic == (klu_symbolic *) NULL)
    {
	mexErrMsgTxt ("klu_analyze failed") ;
    }

    Info [0] = Symbolic->nblocks ;
    Info [1] = Symbolic->maxblock ;

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    if (Common.status == KLU_SINGULAR)
    {
	printf("# singular column : %d\n", Common.singular_col) ;
    }
    if (Common.status != KLU_OK)
    {
	mexErrMsgTxt ("klu_factor failed") ;
    }

    /* nz in L, U, and off-diagonal blocks */
    Info [2] = Numeric->lnz + Numeric->unz + n + Common.noffdiag ;

    klu_condest (Ap, Ax, Symbolic, Numeric, &condest, &Common) ;
    printf ("cond est %g\n", condest) ;

    /* ---------------------------------------------------------------------- */
    /* allocate outputs and set X=B */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
    X = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < n*nrhs ; k++) X [k] = B [k] ;

    /* ---------------------------------------------------------------------- */
    /* solve (overwrites right-hand-side with solution) */
    /* ---------------------------------------------------------------------- */

    result = klu_solve (Symbolic, Numeric, n, nrhs, X, &Common) ;

    /* ---------------------------------------------------------------------- */
    /* free Symbolic and Numeric objects, and free workspace */
    /* ---------------------------------------------------------------------- */

    result = klu_free_symbolic (&Symbolic, &Common) ;
    result = klu_free_numeric (&Numeric, &Common) ;
}
