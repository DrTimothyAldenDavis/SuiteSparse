/* ========================================================================== */
/* === klu_mex mexFunction ================================================== */
/* ========================================================================== */

/* See klu.m for a description.  Usage: [L,U,p,q,R,F,r,info] = klu (A,opts) */

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
    double condest, rcond ;
    int *Ap, *Ai, *Lp, *Li, *Up, *Ui, *Fp, *Fi, *P, *Q, *Rp, *Ri, *R ;
    double *Lx, *Ux, *Fx, *Rs, *Ax, *px ;
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
    klu_common Common ;
    mxArray *field ;
    int n, k, symmetric ;

    static const char *fnames [ ] =
	{ "noffdiag", "nrealloc", "condest", "rcond" } ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 2 || nargout > 8)
    {
	mexErrMsgTxt ("Usage: [L,U,p,q,R,F,r,info] = klu (A,opts)") ;
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

    /* get options */
    klu_defaults (&Common) ;
    if (nargin > 1 && mxIsStruct (pargin [1]))
    {

	if ((field = mxGetField (pargin [1], 0, "tol")) != NULL)
	{
	    Common.tol = mxGetScalar (field) ;
	}

	if ((field = mxGetField (pargin [1], 0, "growth")) != NULL)
	{
	    Common.growth = mxGetScalar (field) ;
	}

	if ((field = mxGetField (pargin [1], 0, "imemamd")) != NULL)
	{
	    Common.initmem_amd = mxGetScalar (field) ;
	}

	if ((field = mxGetField (pargin [1], 0, "imem")) != NULL)
	{
	    Common.initmem = mxGetScalar (field) ;
	}

	if ((field = mxGetField (pargin [1], 0, "btf")) != NULL)
	{
	    Common.btf = mxGetScalar (field) ;
	}

	if ((field = mxGetField (pargin [1], 0, "ordering")) != NULL)
	{
	    Common.ordering = mxGetScalar (field) ;
	}

	if ((field = mxGetField (pargin [1], 0, "scale")) != NULL)
	{
	    Common.scale = mxGetScalar (field) ;
	}
    }

    /* CHOLMOD ordering 3,4 becomes 3, with symmetric 0 or 1 */
    symmetric = (Common.ordering == 4) ;
    if (symmetric) Common.ordering = 3 ;
    Common.user_order = klu_cholmod ;
    Common.user_data = &symmetric ;

    /* memory management routines */
    Common.malloc_memory  = mxMalloc ;
    Common.calloc_memory  = mxCalloc ;
    Common.free_memory    = mxFree ;
    Common.realloc_memory = mxRealloc ;

    /* ---------------------------------------------------------------------- */
    /* analyze */
    /* ---------------------------------------------------------------------- */

    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    if (Symbolic == (klu_symbolic *) NULL)
    {
	mexErrMsgTxt ("klu_analyze failed") ;
    }

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    /*
    if (Common.status == KLU_SINGULAR)
    {
	mexPrintf ("singular column: %d, rank %d\n", Common.singular_col,
	    Common.numerical_rank) ;
    }
    */
    if (Common.status != KLU_OK)
    {
	mexErrMsgTxt ("klu_factor failed") ;
    }

    /* ---------------------------------------------------------------------- */
    /* condition number estimate (both of them) */
    /* ---------------------------------------------------------------------- */

    klu_condest (Ap, Ax, Symbolic, Numeric, &condest, &Common) ;
    klu_rcond (Symbolic, Numeric, &rcond, &Common) ;

    /* ---------------------------------------------------------------------- */
    /* sort L and U */
    /* ---------------------------------------------------------------------- */

    if (nargout > 0)
    {
	klu_sort (Symbolic, Numeric, &Common) ;
    }

    /* ---------------------------------------------------------------------- */
    /* extract factorization */
    /* ---------------------------------------------------------------------- */

    /* L */
    if (nargout > 0)
    {
	pargout [0] = mxCreateSparse (n, n, Numeric->lnz, mxREAL) ;
	Lp = mxGetJc (pargout [0]) ;
	Li = mxGetIr (pargout [0]) ;
	Lx = mxGetPr (pargout [0]) ;
    }
    else
    {
	Lp = NULL ;
	Li = NULL ;
	Lx = NULL ;
    }

    /* U */
    if (nargout > 1)
    {
	pargout [1] = mxCreateSparse (n, n, Numeric->unz, mxREAL) ;
	Up = mxGetJc (pargout [1]) ;
	Ui = mxGetIr (pargout [1]) ;
	Ux = mxGetPr (pargout [1]) ;
    }
    else
    {
	Up = NULL ;
	Ui = NULL ;
	Ux = NULL ;
    }

    /* p */
    if (nargout > 2)
    {
	pargout [2] = mxCreateDoubleMatrix (1, n, mxREAL) ;
	P = mxMalloc (n * sizeof (int)) ;
    }
    else
    {
	P = NULL ;
    }

    /* q */
    if (nargout > 3)
    {
	pargout [3] = mxCreateDoubleMatrix (1, n, mxREAL) ;
	Q = mxMalloc (n * sizeof (int)) ;
    }
    else
    {
	Q = NULL ;
    }

    /* R, as a sparse diagonal matrix */
    if (nargout > 4)
    {
	pargout [4] = mxCreateSparse (n, n, n+1, mxREAL) ;
	Rp = mxGetJc (pargout [4]) ;
	Ri = mxGetIr (pargout [4]) ;
	Rs = mxGetPr (pargout [4]) ;
	for (k = 0 ; k <= n ; k++)
	{
	    Rp [k] = k ;
	    Ri [k] = k ;
	}
    }
    else
    {
	Rs = NULL ;
    }

    /* F, off diagonal blocks */
    if (nargout > 5)
    {
	pargout [5] = mxCreateSparse (n, n, Numeric->Offp [n], mxREAL) ;
	Fp = mxGetJc (pargout [5]) ;
	Fi = mxGetIr (pargout [5]) ;
	Fx = mxGetPr (pargout [5]) ;
    }
    else
    {
	Fp = NULL ;
	Fi = NULL ;
	Fx = NULL ;
    }

    /* r, block boundaries */
    if (nargout > 6)
    {
	pargout [6] = mxCreateDoubleMatrix (1, n+1, mxREAL) ;
	R = mxMalloc ((n+1) * sizeof (int)) ;
    }
    else
    {
	R = NULL ;
    }

    klu_extract (Numeric, Symbolic, Lp, Li, Lx, Up, Ui, Ux, Fp, Fi, Fx,
	P, Q, Rs, R) ;

    /* info */
    if (nargout > 6)
    {
	pargout [7] = mxCreateStructMatrix (1, 1, 4, fnames) ;
	mxSetFieldByNumber (pargout [7], 0, 0,
	    mxCreateScalarDouble (Common.noffdiag)) ;
	mxSetFieldByNumber (pargout [7], 0, 1,
	    mxCreateScalarDouble (Common.nrealloc)) ;
	mxSetFieldByNumber (pargout [7], 0, 2, mxCreateScalarDouble (condest)) ;
	mxSetFieldByNumber (pargout [7], 0, 3, mxCreateScalarDouble (rcond)) ;
    }

    /* p */
    if (nargout > 2)
    {
	pargout [2] = mxCreateDoubleMatrix (1, n, mxREAL) ;
	px = mxGetPr (pargout [2]) ;
	for (k = 0 ; k < n ; k++)
	{
	    px [k] = P [k] + 1 ;
	}
	mxFree (P)  ;
    }

    /* q */
    if (nargout > 3)
    {
	pargout [3] = mxCreateDoubleMatrix (1, n, mxREAL) ;
	px = mxGetPr (pargout [3]) ;
	for (k = 0 ; k < n ; k++)
	{
	    px [k] = Q [k] + 1 ;
	}
	mxFree (Q)  ;
    }

    /* r, block boundaries */
    if (nargout > 6)
    {
	pargout [6] = mxCreateDoubleMatrix (1, Symbolic->nblocks+1, mxREAL) ;
	px = mxGetPr (pargout [6]) ;
	for (k = 0 ; k <= Symbolic->nblocks ; k++)
	{
	    px [k] = R [k] + 1 ;
	}
	mxFree (R)  ;
    }

    if (nargout == 0)
    {
	printf ("nnz(L) %d nnz(U) %d nnz(F) %d\n", Numeric->lnz,
	    Numeric->unz, Numeric->Offp [n]) ;
    }

#if 0
    for (k = 0 ; k < n ; k++)
    {
	int p ;
	printf ("L[%d]: %d to %d\n", k, Lp [k], Lp [k+1]-1) ;
	for (p = Lp [k] ; p < Lp [k+1]-1 ; p++)
	{
	    printf ("   %d %g\n", Li [p], Lx [p]) ;
	}
    }

    for (k = 0 ; k < n ; k++)
    {
	int p ;
	printf ("U[%d]: %d to %d\n", k, Up [k], Up [k+1]-1) ;
	for (p = Up [k] ; p < Up [k+1]-1 ; p++)
	{
	    printf ("   %d %g\n", Ui [p], Ux [p]) ;
	}
    }

    for (k = 0 ; k < n ; k++)
    {
	int p ;
	printf ("F[%d]: %d to %d\n", k, Fp [k], Fp [k+1]-1) ;
	for (p = Fp [k] ; p < Fp [k+1]-1 ; p++)
	{
	    printf ("   %d %g\n", Fi [p], Fx [p]) ;
	}
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* free Symbolic and Numeric objects */
    /* ---------------------------------------------------------------------- */

    klu_free_symbolic (&Symbolic, &Common) ;
    klu_free_numeric (&Numeric, &Common) ;
}
