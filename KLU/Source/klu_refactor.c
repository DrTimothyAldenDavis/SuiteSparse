/* ========================================================================== */
/* === klu_refactor ========================================================= */
/* ========================================================================== */

/* Factor the matrix, after ordering and analyzing it with klu_analyze, and
 * factoring it once with klu_factor.  This routine cannot do any numerical
 * pivoting.  The pattern of the input matrix (Ap, Ai) must be identical to
 * the pattern given to klu_factor.
 */

#include "klu_internal.h"

/* ========================================================================== */
/* === compute the kth column of L and U ==================================== */
/* ========================================================================== */

#define COMPUTE_KTH_COLUMN_OF_L_AND_U \
{ \
    /* compute kth column of U, and update kth column of A */ \
    GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, ulen) ; \
    for (up = 0 ; up < ulen ; up++) \
    { \
	j = Ui [up] ; \
	ujk = X [j] ; \
	/* X [j] = 0 ;*/ \
	CLEAR (X [j]) ; \
	Ux [up] = ujk ; \
	GET_POINTER (LU, Lip, Llen, Li, Lx, j, llen) ; \
	for (p = 0 ; p < llen ; p++) \
	{ \
	    /* X [Li [p]] -= Lx [p] * ujk ;*/ \
	    MULT_SUB (X [Li [p]], Lx [p], ujk) ; \
	} \
    } \
    /* get the diagonal entry of U */ \
    ukk = X [k] ; \
    /* X [k] = 0 ;*/ \
    CLEAR (X [k]) ; \
    /* singular case */ \
    if (IS_ZERO (ukk)) \
    { \
	/* matrix is numerically singular */ \
	Common->status = KLU_SINGULAR ; \
	if (Common->numerical_rank == EMPTY) \
	{ \
	    Common->numerical_rank = k+k1 ; \
	    Common->singular_col = Q [k+k1] ; \
	} \
	if (Common->halt_if_singular) \
	{ \
	    /* do not continue the factorization */ \
	    return (FALSE) ; \
	} \
    } \
    Udiag_entry [k] = ukk ; \
    /* gather and divide by pivot to get kth column of L */ \
    GET_POINTER (LU, Lip, Llen, Li, Lx, k, llen) ; \
    for (p = 0 ; p < llen ; p++) \
    { \
	i = Li [p] ; \
	DIV (Lx [p], X [i], ukk) ; \
	CLEAR (X [i]) ; \
    } \
}


/* ========================================================================== */
/* === klu_refactor ========================================================= */
/* ========================================================================== */

int KLU_refactor	/* returns TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    double Ax [ ],
    klu_symbolic *Symbolic,

    /* input/output */
    klu_numeric *Numeric,
    klu_common  *Common
)
{
    Entry ukk, ujk, s ;
    Entry *Singleton, *Offx, *Lx, *Ux, *X, *Az, *Udiag_entry ;
    double *Rs ;
    int *P, *Q, *R, *Pnum, *Offp, *Offi, *Ui, *Li, *Pinv, *Lip, *Uip, *Llen,
	*Ulen ;
    int **Lbip, **Ubip, **Lblen, **Ublen ;
    Unit **LUbx, **Udiag ;
    Unit *LU ;
    int k1, k2, nk, k, block, oldcol, pend, oldrow, n, p, newrow, scale,
	nblocks, poff, i, j, up, ulen, llen, maxblock ;

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    Common->status = KLU_OK ;

    if (Numeric == NULL)
    {
	/* invalid Numeric object */
	Common->status = KLU_INVALID ;
	return (FALSE) ;
    }

    Common->numerical_rank = EMPTY ;
    Common->singular_col = EMPTY ;

    Az = (Entry *) Ax ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Symbolic object */
    /* ---------------------------------------------------------------------- */

    n = Symbolic->n ;
    P = Symbolic->P ;
    Q = Symbolic->Q ;
    R = Symbolic->R ;
    nblocks = Symbolic->nblocks ;
    maxblock = Symbolic->maxblock ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Numeric object */
    /* ---------------------------------------------------------------------- */

    Pnum = Numeric->Pnum ;
    Offp = Numeric->Offp ;
    Offi = Numeric->Offi ;
    Offx = (Entry *) Numeric->Offx ;
    Singleton = (Entry *) Numeric->Singleton ;

    Lbip = Numeric->Lbip ;
    Lblen = Numeric->Lblen ;
    Ubip = Numeric->Ubip ;
    Ublen = Numeric->Ublen ;
    LUbx = (Unit **) Numeric->LUbx ;

    scale = Common->scale ;
    if (scale > 0)
    {
	/* factorization was not scaled, but refactorization is scaled */
	if (Numeric->Rs == NULL)
	{
	    Numeric->Rs = klu_malloc (n, sizeof (double), Common) ;
	    if (Common->status < KLU_OK)
	    {
		Common->status = KLU_OUT_OF_MEMORY ;
		return (FALSE) ;
	    }
	}
    }
    else
    {
	/* no scaling for refactorization; ensure Numeric->Rs is freed.  This
	 * does nothing if Numeric->Rs is already NULL. */
	Numeric->Rs = klu_free (Numeric->Rs, Common) ;
    }
    Rs = Numeric->Rs ;

    Pinv = Numeric->Pinv ;
    X = (Entry *) Numeric->Xwork ;
    Common->nrealloc = 0 ;
    Udiag = (Unit **) Numeric->Udiag ;

    /* ---------------------------------------------------------------------- */
    /* get control parameters */
    /* ---------------------------------------------------------------------- */

    PRINTF (("klu_refactor scale %d:  n %d nzoff %d nblocks %d maxblock %d\n",
	scale, n, Symbolic->nzoff, nblocks, maxblock)) ;

#ifdef TESTING
    /* randomly mangle the numerical values to test the refactor code */
    {
	int block, k1, k2, nk, p ;
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    if (nk == 1)
	    {
		/* note that this will cause a failure for singular matrices */
		Singleton [block] = 100 * block ;
	    }
	    else
	    {
		Entry *Ux, *Lx;
		Lip = Lbip [block] ;
		Llen = Lblen [block] ;
		Uip = Ubip [block] ;
		Ulen = Ublen [block] ;
		LU = LUbx [block] ;
		for (p = 0 ; p < nk ; p++)
		{
		    /*Lx = (Entry *) (LU + Lip [p] + UNITS (int, Llen [p]) ) ;*/
		    GET_X_POINTER (LU, Lip, Llen, Lx, p) ;
		    for (k = 0 ; k < Llen [p] ; k++)
		    {
			Lx [k] = 42 + p * 1e-5 ;	/*check if this is ok*/
	    	    }
		    /*Ux = (Entry *) (LU + Uip [p] + UNITS (int, Ulen [p]) ) ;*/
		    GET_X_POINTER (LU, Uip, Ulen, Ux, p) ;
		    for (k = 0 ; k < Ulen [p] ; k++)
		    {
			Ux [k] = 99 + p * 1e-4 ;	/*check if this is ok*/
	    	    }
		}
	    }
	}
	for (p = 0 ; p < Symbolic->nzoff ; p++) Offx [p] = p ;
	if (Rs != NULL) for (k = 0 ; k < n ; k++) Rs [k] = 1 + k/1000. ;
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* check the input matrix compute the row scale factors, Rs */
    /* ---------------------------------------------------------------------- */

    /* do no scale, or check the input matrix, if scale < 0 */
    if (scale >= 0)
    {
	/* check for out-of-range indices, but do not check for duplicates */
	if (!KLU_scale (scale, n, Ap, Ai, Ax, Rs, NULL, Common))
	{
	    return (FALSE) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* clear workspace X */
    /* ---------------------------------------------------------------------- */

    for (k = 0 ; k < maxblock ; k++)
    {
	/* X [k] = 0 ;*/
	CLEAR (X [k]) ;
    }

    poff = 0 ;

    /* ---------------------------------------------------------------------- */
    /* factor each block */
    /* ---------------------------------------------------------------------- */

    if (scale < 0)
    {

	/* ------------------------------------------------------------------ */
	/* no scaling or error-checking */
	/* ------------------------------------------------------------------ */

	for (block = 0 ; block < nblocks ; block++)
	{

	    /* -------------------------------------------------------------- */
	    /* the block is from rows/columns k1 to k2-1 */
	    /* -------------------------------------------------------------- */

	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    PRINTF (("BLOCK %d, k1 %d k2-1 %d nk %d\n", block, k1,k2-1,nk)) ;
	    ASSERT (nk <= maxblock) ;

	    if (nk == 1)
	    {

		/* ---------------------------------------------------------- */
		/* singleton case */
		/* ---------------------------------------------------------- */

		oldcol = Q [k1] ;
		pend = Ap [oldcol+1] ;
		CLEAR (s) ;
		for (p = Ap [oldcol] ; p < pend ; p++)
		{
		    if (Pinv [Ai [p]] < k1)
		    {
			Offx [poff++] = Az [p] ;
		    }
		    else
		    {
			s = Az [p] ;
		    }
		}
		Singleton [block] = s ;

	    }
	    else
	    {

		/* ---------------------------------------------------------- */
		/* construct and factor the kth block */
		/* ---------------------------------------------------------- */

		Lip = Lbip [block] ;
		Llen = Lblen [block] ;
		Uip = Ubip [block] ;
		Ulen = Ublen [block] ;
		LU = LUbx [block] ;
		Udiag_entry = (Entry *) Udiag [block] ;

		for (k = 0 ; k < nk ; k++)
		{
		    /* scatter kth column of the block into workspace X */
		    oldcol = Q [k+k1] ;
		    pend = Ap [oldcol+1] ;
		    for (p = Ap [oldcol] ; p < pend ; p++)
		    {
			newrow = Pinv [Ai [p]] ;
			if (newrow < k1)
			{
			    /* this is an entry in the off-diagonal part */
			    Offx [poff++] = Az [p] ;
			}
			else
			{
			    /* (newrow-k1,k) is an entry in the block */
			    X [newrow-k1] = Az [p] ;
			}
		    }
		    COMPUTE_KTH_COLUMN_OF_L_AND_U ;
		}
	    }
	}

    }
    else if (scale == 0)
    {

	/* ------------------------------------------------------------------ */
	/* no scaling, but with error-checking */
	/* ------------------------------------------------------------------ */

	for (block = 0 ; block < nblocks ; block++)
	{

	    /* -------------------------------------------------------------- */
	    /* the block is from rows/columns k1 to k2-1 */
	    /* -------------------------------------------------------------- */

	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    PRINTF (("BLOCK %d, k1 %d k2-1 %d nk %d\n", block, k1,k2-1,nk)) ;
	    ASSERT (nk <= maxblock) ;

	    if (nk == 1)
	    {

		/* ---------------------------------------------------------- */
		/* singleton case */
		/* ---------------------------------------------------------- */

		if (Offp [k1] != poff)
		{
		    Common->status = KLU_INVALID ;
		    return (FALSE) ;
		}
		oldcol = Q [k1] ;
		pend = Ap [oldcol+1] ;
		CLEAR (s) ;

		for (p = Ap [oldcol] ; p < pend ; p++)
		{
		    newrow = Pinv [Ai [p]] ;
		    if (newrow < k1)
		    {
			if (Offi [poff] != newrow)
			{
			    Common->status = KLU_INVALID ;
			    return (FALSE) ;
			}
			Offx [poff++] = Az [p] ;
		    }
		    else
		    {
			if (newrow != k1)
			{
			    Common->status = KLU_INVALID ;
			    return (FALSE) ;
			}
			s = Az [p] ;
		    }
		}
		Singleton [block] = s ;

	    }
	    else
	    {

		/* ---------------------------------------------------------- */
		/* construct and factor the kth block */
		/* ---------------------------------------------------------- */

		Lip = Lbip [block] ;
		Llen = Lblen [block] ;
		Uip = Ubip [block] ;
		Ulen = Ublen [block] ;
		LU = LUbx [block] ;
		Udiag_entry = (Entry *) Udiag [block] ;

		for (k = 0 ; k < nk ; k++)
		{
		    /* scatter kth column of the block into workspace X */
		    if (Offp [k+k1] != poff)
		    {
			Common->status = KLU_INVALID ;
			return (FALSE) ;
		    }
		    oldcol = Q [k+k1] ;
		    pend = Ap [oldcol+1] ;

		    for (p = Ap [oldcol] ; p < pend ; p++)
		    {
			oldrow = Ai [p] ;
			newrow = Pinv [oldrow] ;
			if (newrow < k1)
			{
			    /* this is an entry in the off-diagonal part */
			    ASSERT (poff < Symbolic->nzoff) ;
			    if (Offi [poff] != newrow)
			    {
				Common->status = KLU_INVALID ;
				return (FALSE) ;
			    }
			    Offx [poff++] = Az [p] ;
			}
			else
			{
			    /* (newrow,k) is an entry in the block */
			    ASSERT (newrow < k2) ;
			    newrow -= k1 ;
			    if (newrow >= nk)
			    {
				Common->status = KLU_INVALID ;
				return (FALSE) ;
			    }
			    X [newrow] = Az [p] ;
			}
		    }

		    COMPUTE_KTH_COLUMN_OF_L_AND_U ;
		}
	    }
	}

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* scaling and error-checking */
	/* ------------------------------------------------------------------ */

	for (block = 0 ; block < nblocks ; block++)
	{

	    /* -------------------------------------------------------------- */
	    /* the block is from rows/columns k1 to k2-1 */
	    /* -------------------------------------------------------------- */

	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    PRINTF (("BLOCK %d, k1 %d k2-1 %d nk %d\n", block, k1,k2-1,nk)) ;
	    ASSERT (nk <= maxblock) ;

	    if (nk == 1)
	    {

		/* ---------------------------------------------------------- */
		/* singleton case */
		/* ---------------------------------------------------------- */

		if (Offp [k1] != poff)
		{
		    Common->status = KLU_INVALID ;
		    return (FALSE) ;
		}
		oldcol = Q [k1] ;
		pend = Ap [oldcol+1] ;
		CLEAR (s) ;
		for (p = Ap [oldcol] ; p < pend ; p++)
		{
		    oldrow = Ai [p] ;
		    newrow = Pinv [oldrow] ;
		    if (newrow < k1)
		    {
			if (Offi [poff] != newrow)
			{
			    Common->status = KLU_INVALID ;
			    return (FALSE) ;
			}
			/* Offx [poff++] = Az [p] / Rs [oldrow] ;*/
			SCALE_DIV_ASSIGN (Offx [poff++], Az [p], Rs [oldrow]) ;
		    }
		    else
		    {
			if (newrow != k1)
			{
			    Common->status = KLU_INVALID ;
			    return (FALSE) ;
			}
			/* s = Az [p] / Rs [oldrow] */
			SCALE_DIV_ASSIGN (s, Az [p], Rs [oldrow]) ;
		    }
		}
		Singleton [block] = s ;

	    }
	    else
	    {

		/* ---------------------------------------------------------- */
		/* construct and factor the kth block */
		/* ---------------------------------------------------------- */

		Lip = Lbip [block] ;
		Llen = Lblen [block] ;
		Uip = Ubip [block] ;
		Ulen = Ublen [block] ;
		LU = LUbx [block] ;
		Udiag_entry = (Entry *) Udiag [block] ;

		for (k = 0 ; k < nk ; k++)
		{
		    /* scatter kth column of the block into workspace X */
		    if (Offp [k+k1] != poff)
		    {
			Common->status = KLU_INVALID ;
			return (FALSE) ;
		    }
		    oldcol = Q [k+k1] ;
		    pend = Ap [oldcol+1] ;

		    for (p = Ap [oldcol] ; p < pend ; p++)
		    {
			oldrow = Ai [p] ;
			newrow = Pinv [oldrow] ;
			if (newrow < k1)
			{
			    /* this is an entry in the off-diagonal part */
			    ASSERT (poff < Symbolic->nzoff) ;
			    if (Offi [poff] != newrow)
			    {
				Common->status = KLU_INVALID ;
				return (FALSE) ;
			    }
			    SCALE_DIV_ASSIGN (Offx [poff++], Az [p],
					      Rs [oldrow]) ;
			}
			else
			{
			    /* (newrow,k) is an entry in the block */
			    ASSERT (newrow < k2) ;
			    newrow -= k1 ;
			    if (newrow >= nk)
			    {
				Common->status = KLU_INVALID ;
				return (FALSE) ;
			    }
			    SCALE_DIV_ASSIGN (X [newrow], Az [p], Rs [oldrow]) ;
			}
		    }

		    COMPUTE_KTH_COLUMN_OF_L_AND_U ;
		}
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* permute scale factors Rs according to pivotal row order */
    /* ---------------------------------------------------------------------- */

    if (scale > 0)
    {
	for (k = 0 ; k < n ; k++)
	{
	    REAL (X [k]) = Rs [Pnum [k]] ;
	}
	for (k = 0 ; k < n ; k++)
	{
	    Rs [k] = REAL (X [k]) ;
	}
    }

    ASSERT (Offp [n] == poff) ;
    ASSERT (Symbolic->nzoff == poff) ;
    PRINTF (("\n------------------- Off diagonal entries, new:\n")) ;
    ASSERT (KLU_valid (n, Offp, Offi, Offx)) ;

#ifndef NDEBUG
    if (Common->status == KLU_OK)
    {
	int block, k1, k2, nk ;
	PRINTF (("\n ########### KLU_BTF_REFACTOR done, nblocks %d\n",nblocks));
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    PRINTF ((
		"\n================klu_refactor output: k1 %d k2 %d nk %d\n",
		k1, k2, nk)) ;
	    if (nk == 1)
	    {
		PRINTF (("singleton  ")) ;
		PRINT_ENTRY ( Singleton [block] ) ;
	    }
	    else
	    {
		Lip = Lbip [block] ;
                Llen = Lblen [block] ;
                LU = (Unit *) Numeric->LUbx [block] ;
		PRINTF (("\n---- L block %d\n", block)) ;
		ASSERT (KLU_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
		Uip = Ubip [block] ;
                Ulen = Ublen [block] ;
		PRINTF (("\n---- U block %d\n", block)) ;
		ASSERT (KLU_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
	    }
	}
    }
#endif
    return (TRUE) ;
}
