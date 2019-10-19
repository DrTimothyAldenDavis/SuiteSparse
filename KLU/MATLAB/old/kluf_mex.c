/* ========================================================================== */
/* === kluf mexFunction ===================================================== */
/* ========================================================================== */

/* Factor A using symbolic info from klua.
 *
 * [P,Q,R,Lnz,Info1] = klua (A) ;
 * [L,U,Off,Pnum,Rs,Info2] = kluf (A, P,Q,R,Lnz,Info1, Control) ;
 *
 * The factorization is L*U + Off = Rs (Pnum,Pnum) \ (A (Pnum,Q)), where Rs is
 * a diagonal matrix of row scale factors.  If Pnum and Q are converted to
 * permutation matrices, then L*U + Off = Pnum * (Rs\A) * Q. 
 */


/* ========================================================================== */

#include "klu_internal.h"

#define LUNITS(x,y,n) ((sizeof(x) * n + (sizeof(y) - 1)) / sizeof(y)) 

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    int n, *Ap, *Ai, k, p, col, block, nblocks, *Off2i, *Off2p, *L2i, *L2p, nk,
	*U2p, *U2i, *Li, *Ui, lnz, unz, k1, k2, nz, pl, pu, nzoff,
	*Rsi, *Rsp, i, *Lip, *Llen, *Uip, *Ulen ;
    double *Ax, *Px, *Qx, *Rx, *Lnzx, *Info_in, *Pnumx, *Off2x, *L2x,
	*U2x, *Rsx, *Info, *Control ;
    double *Atemp, *Aimag, *Off2imag, *L2imag, *U2imag, *Offtemp,
	*Lx, *Sing, *Ux, *Ud ;
    double *LU ;
    DoubleComplex *LUz, *Singz, *L2complex, *U2complex, *Lxz, *Uxz, *Udz ;
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
    klu_common Common ;
    int status ;
    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 6 || nargin > 7 || nargout != 6)
    {
	mexErrMsgTxt ("Usage: [L,U,Off,Pnum,Rs,Info] = "
		      "kluf (A,P,Q,R,Lnz,Info_in,Control)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0]))
    {
    	mexErrMsgTxt ("klua: A must be sparse, square, real, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = mxGetJc (pargin [0]) ;
    Ai = mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    if (mxIsComplex (pargin [0]))
    {
    	Aimag = mxGetPi (pargin [0]) ;
        Atemp = mxMalloc (2 * Ap[n] * sizeof(double)) ;
	if (!Atemp)
	    mexErrMsgTxt("Malloc failed") ;
	for(i = 0, k = 0 ; k < Ap[n] ; k++)
	{
	    Atemp[i++] = Ax [k] ;
	    Atemp[i++] = Aimag [k] ;
	}
	Ax = Atemp ;
    }
    nz = Ap [n] ;

    /* get control parameters */
    klu_defaults (&Common) ;

    /* use mxMalloc and related memory management routines */
    Common.malloc_memory  = mxMalloc ;
    Common.free_memory    = mxFree ;
    Common.realloc_memory = mxRealloc ;

    if (nargin > 6)
    {
	int s ;
	if (!mxIsDouble (pargin [6]))
	{
	    mexErrMsgTxt ("klu: control must be real") ;
	}
	Control = mxGetPr (pargin [6]) ;
	s = mxGetNumberOfElements (pargin [6]) ;
	/* if (s > 0) Common.prl      = Control [0] ; */
	if (s > 1) Common.btf         = Control [1] ;
	if (s > 2) Common.scale       = Control [2] ;
	if (s > 3) Common.ordering    = Control [3] ;
	if (s > 4) Common.tol         = Control [4] ;
	if (s > 5) Common.growth      = Control [5] ;
	if (s > 6) Common.initmem_amd = Control [6] ;
	if (s > 7) Common.initmem     = Control [7] ;
    }
    Common.scale = 0 ;
    PRINTF (("Common: btf %d ord %d tol %g gro %g inita %g init %g\n",
	 Common.btf, Common.ordering, Common.tol, Common.growth,
	 Common.initmem_amd, Common.initmem)) ;

    /* ---------------------------------------------------------------------- */
    /* reconstruct the symbolic object */
    /* ---------------------------------------------------------------------- */

    Symbolic = klu_malloc (1, sizeof (klu_symbolic), &Common) ;

    /* get Info */
    Info_in = mxGetPr (pargin [5]) ;
    pargout [5] = mxCreateDoubleMatrix (1, 90, mxREAL) ;
    Info = mxGetPr (pargout [5]) ;
    for (i = 0 ; i < 90 ; i++)
    {
	Info [i] = Info_in [i] ;
    }

    Symbolic->n        = n ;		/* dimension of A */
    Symbolic->nz       = nz ;		/* # entries in input matrix */
    Symbolic->nblocks  = Info_in [ 3] ; /* # of blocks in BTF form */
    Symbolic->maxblock = Info_in [ 4] ; /* dimension of largest block */
    Symbolic->nzoff    = Info_in [ 7] ; /* nz in off-diagonal blocks of A */
    Symbolic->symmetry = Info_in [ 8] ; /* symmetry of largest block */
    Symbolic->lnz      = Info_in [10] ; /* nz in L, estimated (incl diagonal) */
    Symbolic->unz      = Info_in [11] ; /* nz in U, estimated (incl diagonal) */
    Symbolic->est_flops= Info_in [12] ; /* est. factorization flop count */

    PRINTF (("kluf: n %d nzoff %d nblocks %d maxblock %d\n",
	Symbolic->n, Symbolic->nzoff, Symbolic->nblocks,
	Symbolic->maxblock)) ; 

    nblocks = Symbolic->nblocks ;
    ASSERT (nblocks > 0) ;

    Symbolic->P = klu_malloc (n, sizeof (int), &Common) ;
    Symbolic->Q = klu_malloc (n, sizeof (int), &Common) ;
    Symbolic->R = klu_malloc (nblocks+1, sizeof (int), &Common) ;
    Symbolic->Lnz = klu_malloc (nblocks, sizeof (double), &Common) ;

    ASSERT (Symbolic->nzoff >= 0 && Symbolic->nzoff <= nz) ;
    ASSERT (Symbolic->maxblock > 0 && Symbolic->maxblock <= n - nblocks + 1) ;

    /* get P */
    Px = mxGetPr (pargin [1]) ;
    for (k = 0 ; k < n ; k++)
    {
	Symbolic->P [k] = Px [k] - 1 ;	    /* convert to 0-based */
    }

    /* get Q */
    Qx = mxGetPr (pargin [2]) ;
    for (k = 0 ; k < n ; k++)
    {
	Symbolic->Q [k] = Qx [k] - 1 ;	    /* convert to 0-based */
    }

    /* get R */
    Rx = mxGetPr (pargin [3]) ;
    for (k = 0 ; k <= nblocks ; k++)
    {
	Symbolic->R [k] = Rx [k] - 1 ;	    /* convert to 0-based */
    }
    ASSERT (Symbolic->R [nblocks] == n) ;

    /* get Lnz */
    Lnzx = mxGetPr (pargin [4]) ;
    for (k = 0 ; k < nblocks ; k++)
    {
	Symbolic->Lnz [k] = Lnzx [k] ;
    }

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    if (mxIsComplex (pargin [0]))
    {
	/* status = klu_z_factor (Ap, Ai, Ax, Symbolic, &Numeric, &Common) ; */
	printf("Complex case\n") ;
	Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    }
    else
    {
	/* status = klu_factor (Ap, Ai, Ax, Symbolic, &Numeric, &Common) ; */
	Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    }

    if (Common.status == KLU_SINGULAR)
    {
        mexPrintf("# singular column : %d\n", Common.singular_col) ;
    }
    if (Common.status != KLU_OK)
    {
        mexErrMsgTxt ("klu_factor failed") ;
    }	

#ifndef NDEBUG
    /* dump */
    for (block = 0 ; block < nblocks ; block++)
    {
	k1 = Symbolic->R [block] ;
	k2 = Symbolic->R [block+1] ;
	nk = k2 - k1 ;
	PRINTF (("block %d k1 %d k2 %d nk %d\n", block, k1, k2, nk)) ;
	if (nk > 1)
	{
	    Lip = Numeric->Lbip [block] ; 
	    Llen = Numeric->Lblen [block] ; 
	    Uip = Numeric->Ubip [block] ; 
	    Ulen = Numeric->Ublen [block] ; 
	    PRINTF (("\n---- L block %d\n", block)) ; 
	    if (mxIsComplex (pargin [0]))
	    {
		LUz = (DoubleComplex *) Numeric->LUbx [block] ;
		ASSERT (klu_z_valid_LU (nk, TRUE, Lip, Llen, LUz)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_z_valid_LU (nk, FALSE, Uip, Ulen, LUz)) ;
	    }
	    else
	    {
		LU = (double *) Numeric->LUbx [block] ;
		ASSERT (klu_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
	    }
	}
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* return the results to MATLAB */
    /* ---------------------------------------------------------------------- */

    /* create Info output */
    Info [30] = Numeric->lnz ;		/* nz in L, actual (incl. diagonal) */
    Info [31] = Numeric->unz ;		/* nz in U, actual (incl. diagonal) */
    /* Info [36] = Numeric->noffdiag ;	 number of off-diagonal pivots */
    Info [36] = Common.noffdiag ;	/* number of off-diagonal pivots */
    Info [33] = Numeric->umin ;		/* min abs diagonal entry in U */
    Info [34] = Numeric->umax ;		/* max abs diagonal entry in U */

    /* create permutation vector for Pnum */
    pargout [3] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Pnumx = mxGetPr (pargout [3]) ;
    for (k = 0 ; k < n ; k++)
    {
	Pnumx [k] = Numeric->Pnum [k] + 1 ;	/* convert to 1-based */
	PRINTF (("Pnum (%d) = %g\n", k+1, Pnumx [k])) ;
    }

#ifndef NDEBUG
    /* dump again */
    for (block = 0 ; block < nblocks ; block++)
    {
	k1 = Symbolic->R [block] ;
	k2 = Symbolic->R [block+1] ;
	nk = k2 - k1 ;
	PRINTF (("again, block %d k1 %d k2 %d nk %d\n", block, k1, k2, nk)) ;
	if (nk > 1)
	{
	    Lip = Numeric->Lbip [block] ; 
	    Llen = Numeric->Lblen [block] ; 
	    Uip = Numeric->Ubip [block] ; 
	    Ulen = Numeric->Ublen [block] ; 
	    PRINTF (("\n---- L block %d\n", block)) ; 
	    if (mxIsComplex (pargin [0]))
	    {
		LUz = (DoubleComplex *) Numeric->LUbx [block] ;
		ASSERT (klu_z_valid_LU (nk, TRUE, Lip, Llen, LUz)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_z_valid_LU (nk, FALSE, Uip, Ulen, LUz)) ;
	    }
	    else
	    {
		LU = (double *) Numeric->LUbx [block] ;
		ASSERT (klu_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
	    }
	}
    }
#endif

    /* create Off */
    PRINTF (("\n----------------------- Off input:\n")) ;
    nzoff = Symbolic->nzoff ;
    if (mxIsComplex (pargin [0]))
    {
	ASSERT (klu_z_valid (n, Numeric->Offp, Numeric->Offi, Numeric->Offx)) ;
        pargout [2] = mxCreateSparse (n, n, nzoff, mxCOMPLEX) ;
    }	    
    else
    {
	ASSERT (klu_valid (n, Numeric->Offp, Numeric->Offi, Numeric->Offx)) ;
        pargout [2] = mxCreateSparse (n, n, nzoff, mxREAL) ;
    }	    
    Off2p = mxGetJc (pargout [2]) ;
    Off2i = mxGetIr (pargout [2]) ;
    Off2x = mxGetPr (pargout [2]) ;
    for (col = 0 ; col <= n ; col++) Off2p [col] = Numeric->Offp [col] ;
    for (p = 0 ; p < nzoff ; p++)    Off2i [p]   = Numeric->Offi [p] ;
    
    PRINTF (("\n----------------------- Off output:\n")) ;
    Offtemp = (double *) Numeric->Offx ;
    if (mxIsComplex (pargin [0]))
    {
	Off2imag = mxGetPi (pargout [2]) ;	    
    	for (i = 0 , p = 0 ; p < nzoff ; p++)
	{    
	    Off2x [p]   = Offtemp [i++] ;
	    Off2imag [p] = Offtemp [i++] ;
	}
	ASSERT (klu_z_valid (n, Off2p, Off2i, (DoubleComplex *) Offtemp)) ;
    }
    else
    {    
    	for (p = 0 ; p < nzoff ; p++)    Off2x [p]   = Offtemp [p] ;
	ASSERT (klu_valid (n, Off2p, Off2i, Offtemp)) ;
    }
#ifndef NDEBUG
    /* determine # of nonzeros in L and U */
    lnz = 0 ;
    unz = 0 ;
    for (block = 0 ; block < nblocks ; block++)
    {
	k1 = Symbolic->R [block] ;
	k2 = Symbolic->R [block+1] ;
	nk = k2 - k1 ;
	PRINTF (("block %d k1 %d k2 %d nk %d\n", block, k1, k2, nk)) ;
	if (nk == 1)
	{
	    lnz++ ;
	    unz++ ;
	}
	else
	{
	    Lip = Numeric->Lbip [block] ; 
	    Llen = Numeric->Lblen [block] ; 
	    Uip = Numeric->Ubip [block] ; 
	    Ulen = Numeric->Ublen [block] ; 
	    PRINTF (("\n---- L block %d\n", block)) ; 
	    if (mxIsComplex (pargin [0]))
	    {
		LUz = (DoubleComplex *) Numeric->LUbx [block] ;
		ASSERT (klu_z_valid_LU (nk, TRUE, Lip, Llen, LUz)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_z_valid_LU (nk, FALSE, Uip, Ulen, LUz)) ;
	    }
	    else
	    {
		LU = (double *) Numeric->LUbx [block] ;
		ASSERT (klu_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
	    }
	    for (p = 0 ; p < nk ; p++)
	    {
		lnz += Llen [p] + 1 ;
	    }
	    for (p = 0 ; p < nk ; p++)
	    {
		unz += Ulen [p] + 1 ;
	    }
	}
    }
    PRINTF (("Total lnz %d unz %d for all blocks\n", lnz, unz)) ;

    ASSERT (lnz == Numeric->lnz) ;
    ASSERT (unz == Numeric->unz) ;
#endif
    lnz = Numeric->lnz ;
    unz = Numeric->unz ;
    /* create L */
    if (mxIsComplex (pargin [0]))
    {
    	pargout [0] = mxCreateSparse (n, n, lnz+n, mxCOMPLEX) ;
        L2imag = mxGetPi (pargout [0]) ;
	L2complex = (DoubleComplex *) mxMalloc (2 * (lnz + n) * sizeof (double)) ;
    }
    else
    {	    
    	pargout [0] = mxCreateSparse (n, n, lnz+n, mxREAL) ;
    }
    L2p = mxGetJc (pargout [0]) ;
    L2i = mxGetIr (pargout [0]) ;
    L2x = mxGetPr (pargout [0]) ;
    /* create U */
    if (mxIsComplex (pargin [0]))
    {
    	pargout [1] = mxCreateSparse (n, n, unz+n, mxCOMPLEX) ;
        U2imag = mxGetPi (pargout [1]) ;
	U2complex = 
	    (DoubleComplex *) mxMalloc (2 * sizeof (double) * (unz + n)) ;
    }
    else
    {
    	pargout [1] = mxCreateSparse (n, n, unz+n, mxREAL) ;
    }
    U2p = mxGetJc (pargout [1]) ;
    U2i = mxGetIr (pargout [1]) ;
    U2x = mxGetPr (pargout [1]) ;

#ifndef NDEBUG
    /* dump again */
    for (block = 0 ; block < nblocks ; block++)
    {
	k1 = Symbolic->R [block] ;
	k2 = Symbolic->R [block+1] ;
	nk = k2 - k1 ;
	PRINTF (("yet again block %d k1 %d k2 %d nk %d\n", block, k1, k2, nk)) ;
	if (nk > 1)
	{
	    Lip = Numeric->Lbip [block] ; 
	    Llen = Numeric->Lblen [block] ; 
	    Uip = Numeric->Ubip [block] ; 
	    Ulen = Numeric->Ublen [block] ; 
	    PRINTF (("\n---- L block %d\n", block)) ; 
	    if (mxIsComplex (pargin [0]))
	    {
		LUz = (DoubleComplex *) Numeric->LUbx [block] ;
		ASSERT (klu_z_valid_LU (nk, TRUE, Lip, Llen, LUz)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_z_valid_LU (nk, FALSE, Uip, Ulen, LUz)) ;
	    }
	    else
	    {
		LU = (double *) Numeric->LUbx [block] ;
		ASSERT (klu_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
		PRINTF (("\n---- U block %d\n", block)) ; 
		ASSERT (klu_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
	    }
	}
    }
#endif

    /* fill L and U */
    pl = 0 ;
    pu = 0 ;
    if (mxIsComplex (pargin [0]))
    {
	Singz = (DoubleComplex *) Numeric->Singleton ;
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = Symbolic->R [block] ;
	    k2 = Symbolic->R [block+1] ;
	    nk = k2 - k1 ;
	    if (nk == 1)
	    {
		L2p [k1] = pl ;
		L2i [pl] = k1 ;
		L2x [pl] = 1 ;
		L2imag [pl] = 0 ;

		U2p [k1] = pu ;
		U2i [pu] = k1 ;
		U2x [pu] = Singz [block].Real ;
		U2imag [pu] = Singz [block].Imag ;
		
		L2complex [pl].Real = 1 ;
		L2complex [pl].Imag = 0 ;
		U2complex [pu] = Singz [block] ;
		pl++ ;
		pu++ ;
	    }
	    else
	    {
		Lip = Numeric->Lbip [block] ; 
		Llen = Numeric->Lblen [block] ; 
		Uip = Numeric->Ubip [block] ; 
		Ulen = Numeric->Ublen [block] ; 
		LUz = (DoubleComplex *) Numeric->LUbx [block] ; 
		Udz = (DoubleComplex *) Numeric->Udiag [block] ;
		for (k = 0 ; k < nk ; k++)
		{
		    Li = (int *) (LUz + Lip [k]) ;
		    Lxz = (DoubleComplex *) (LUz + Lip [k] + 
				     LUNITS (int, DoubleComplex, Llen [k])) ; 
		    L2p [k+k1] = pl ;
		    for (p = 0 ; p < Llen [k] ; p++)
		    {
			L2i [pl] = Li [p] + k1 ;
			L2x [pl] = Lxz [p].Real ;
			L2imag [pl] = Lxz [p].Imag ;
			L2complex [pl] = Lxz [p] ;
			pl++ ;
		    }
		    L2i [pl] = k1 + k;
		    L2x [pl] = 1 ;
		    L2imag [pl] = 0 ;
		    L2complex [pl].Real = 1 ;
		    L2complex [pl].Imag = 0 ;
		    pl++ ;

		    Ui = (int *) (LUz + Uip [k]) ;
		    Uxz = (DoubleComplex *) (LUz + Uip [k] +
				    LUNITS (int, DoubleComplex, Ulen [k])) ;
		    U2p [k+k1] = pu ;
		    for (p = 0 ; p < Ulen [k] ; p++)
		    {
			U2i [pu] = Ui [p] + k1 ;
			U2x [pu] = Uxz [p].Real ;
			U2imag [pu] = Uxz [p].Imag ;
			U2complex [pu] = Uxz [p] ;
			pu++ ;
		    }
		    U2i [pu] = k1 + k ;
		    U2x [pu] = Udz [k].Real ;
		    U2imag [pu] = Udz [k].Imag ;
		    U2complex [pu] = Udz [k] ;
		    pu++ ;
		}
	    }
	}
    }
    else
    {
	Sing = (double *) Numeric->Singleton ;
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = Symbolic->R [block] ;
	    k2 = Symbolic->R [block+1] ;
	    nk = k2 - k1 ;
	    if (nk == 1)
	    {
		L2p [k1] = pl ;
		L2i [pl] = k1 ;
		L2x [pl] = 1 ;

		U2p [k1] = pu ;
		U2i [pu] = k1 ;
		U2x [pu] = Sing [block] ;
		pl++ ;
		pu++ ;
	    }
	    else
	    {
		Lip = Numeric->Lbip [block] ; 
		Llen = Numeric->Lblen [block] ; 
		Uip = Numeric->Ubip [block] ; 
		Ulen = Numeric->Ublen [block] ; 
		LU = (double *) Numeric->LUbx [block] ; 
		Ud = (double *) Numeric->Udiag [block] ;
		for (k = 0 ; k < nk ; k++)
		{
		    Li = (int *) (LU + Lip [k]) ;
		    Lx = (double *) (LU + Lip [k] + 
			    LUNITS (int, double, Llen [k])) ;
		    L2p [k+k1] = pl ;
		    for (p = 0 ; p < Llen [k] ; p++)
		    {
			L2i [pl] = Li [p] + k1 ;
			ASSERT (Li [p] != k) ;
			L2x [pl] = Lx [p] ;
			pl++ ;
		    }
		    L2i [pl] = k1 + k ;
		    L2x [pl] = 1 ; /* unit diagonal */
		    pl++ ;

		    Ui = (int *) (LU + Uip [k]) ;
		    Ux = (double *) (LU + Uip [k] + 
			    LUNITS (int, double, Ulen [k])) ;
		    U2p [k+k1] = pu ;
		    for (p = 0 ; p < Ulen [k] ; p++)
		    {
			U2i [pu] = Ui [p] + k1 ;
			U2x [pu] = Ux [p] ;
			pu++ ;
		    }
		    U2i [pu] = k1 + k ;
		    U2x [pu] = Ud [k] ;
		    pu++ ;
		}
	    }
	}
    }
    L2p [n] = pl ;
    U2p [n] = pu ;

    /* create Rs */
    pargout [4] = mxCreateSparse (n, n, n, mxREAL) ;
    Rsp = mxGetJc (pargout [4]) ;
    Rsi = mxGetIr (pargout [4]) ;
    Rsx = mxGetPr (pargout [4]) ;
    for (k = 0 ; k < n ; k++)
    {
	Rsp [k] = k ;
	Rsi [k] = k ;
	Rsx [k] = Numeric->Rs [k] ;
	PRINTF (("Rsx [k %d] %g\n", k, Rsx [k])) ;
    }
    Rsp [n] = n ;

    if (mxIsComplex (pargin [0]))
    {
    	PRINTF (("\n------------------ All of output L:\n")) ;
	ASSERT (klu_z_valid (n, L2p, L2i, L2complex)) ;
    	PRINTF (("\n------------------ All of output U:\n")) ;
    	ASSERT (klu_z_valid (n, U2p, U2i, U2complex)) ;
    }
    else
    {
    	PRINTF (("\n------------------ All of output L:\n")) ;
	ASSERT (klu_valid (n, L2p, L2i, L2x)) ;
    	PRINTF (("\n------------------ All of output U:\n")) ;
    	ASSERT (klu_valid (n, U2p, U2i, U2x)) ;
    }    

    /* destroy the symbolic object */
    klu_free_symbolic (&Symbolic, &Common) ;

    /* destroy the numeric object */
    KLU_free_numeric (&Numeric, &Common) ;

    if (mxIsComplex (pargin [0]))
    {
    	mxFree (L2complex) ;
	mxFree (U2complex) ;
	mxFree (Atemp) ;
    }	    
}
