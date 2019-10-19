/* ========================================================================== */
/* === klu_factor =========================================================== */
/* ========================================================================== */

/* Factor the matrix, after ordering and analyzing it with klu_analyze
 * or klu_analyze_given.
 */

#include "klu_internal.h"

/* ========================================================================== */
/* === klu_factor2 ========================================================== */
/* ========================================================================== */

static void klu_factor2
(
    /* inputs, not modified */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    Entry Ax [ ],
    klu_symbolic *Symbolic,

    /* inputs, modified on output: */
    klu_numeric *Numeric,
    klu_common *Common
)
{
    double lsize ;
    double *Lnz, *Rs ;
    int *P, *Q, *R, *Pnum, *Offp, *Offi, *Pblock, *Pinv, *Iwork ;
    int **Lbip, **Ubip, **Lblen, **Ublen ;
    Entry *Offx, *Singleton, *X, s ;
    Unit **LUbx, **Udiag ;
    int k1, k2, nk, k, block, oldcol, pend, oldrow, n, lnz, unz, p, newrow,
	nblocks, poff, nzoff, lnz_block, unz_block, scale, max_lnz_block,
	max_unz_block ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    /* get the contents of the Symbolic object */
    n = Symbolic->n ;
    P = Symbolic->P ;
    Q = Symbolic->Q ;
    R = Symbolic->R ;
    Lnz = Symbolic->Lnz ;
    nblocks = Symbolic->nblocks ;
    nzoff = Symbolic->nzoff ;

    Pnum = Numeric->Pnum ;
    Offp = Numeric->Offp ;
    Offi = Numeric->Offi ;
    Offx = (Entry *) Numeric->Offx ;
    Singleton = (Entry *) Numeric->Singleton ;

    Lbip = Numeric->Lbip ;
    Ubip = Numeric->Ubip ;
    Lblen = Numeric->Lblen ;
    Ublen = Numeric->Ublen ;
    LUbx = (Unit **) Numeric->LUbx ;
    Udiag = (Unit **) Numeric->Udiag ;

    Rs = Numeric->Rs ;
    Pinv = Numeric->Pinv ;
    X = (Entry *) Numeric->Xwork ;		/* X is of size n */
    Iwork = Numeric->Iwork ;			/* 5*maxblock for klu_factor */
						/* 1*maxblock for Pblock */
    Pblock = Iwork + 5*((size_t) Symbolic->maxblock) ;
    Common->nrealloc = 0 ;
    scale = Common->scale ;
    max_lnz_block = 1 ;
    max_unz_block = 1 ;

    /* compute the inverse of P from symbolic analysis.  Will be updated to
     * become the inverse of the numerical factorization when the factorization
     * is done, for use in klu_refactor */
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++)
    {
	Pinv [k] = EMPTY ;
    }
#endif
    for (k = 0 ; k < n ; k++)
    {
	ASSERT (P [k] >= 0 && P [k] < n) ;
	Pinv [P [k]] = k ;
    }
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++) ASSERT (Pinv [k] != EMPTY) ;
#endif

    for (block = 0 ; block < nblocks ; block++)
    {
	/* Singleton [block] = 0 ; */
	CLEAR (Singleton [block]) ;
    }

    lnz = 0 ;
    unz = 0 ;
    Common->noffdiag = 0 ;
    Offp [0] = 0 ;

    /* ---------------------------------------------------------------------- */
    /* optionally check input matrix and compute scale factors */
    /* ---------------------------------------------------------------------- */

    if (scale >= 0)
    {
	/* use Pnum as workspace */
	KLU_scale (scale, n, Ap, Ai, (double *) Ax, Rs, Pnum, Common) ;
	if (Common->status < KLU_OK)
	{
	    /* matrix is invalid */
	    return ;
	}
    }

#ifndef NDEBUG
    if (scale > 0)
    {
	for (k = 0 ; k < n ; k++) PRINTF (("Rs [%d] %g\n", k, Rs [k])) ;
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* factor each block using klu */
    /* ---------------------------------------------------------------------- */

    for (block = 0 ; block < nblocks ; block++)
    {

	/* ------------------------------------------------------------------ */
	/* the block is from rows/columns k1 to k2-1 */
	/* ------------------------------------------------------------------ */

	k1 = R [block] ;
	k2 = R [block+1] ;
	nk = k2 - k1 ;
	PRINTF (("FACTOR BLOCK %d, k1 %d k2-1 %d nk %d\n", block, k1,k2-1,nk)) ;

	if (nk == 1)
	{

	    /* -------------------------------------------------------------- */
	    /* singleton case */
	    /* -------------------------------------------------------------- */

	    poff = Offp [k1] ;
	    oldcol = Q [k1] ;
	    pend = Ap [oldcol+1] ;
	    CLEAR (s) ;

	    if (scale <= 0)
	    {
		/* no scaling */
		for (p = Ap [oldcol] ; p < pend ; p++)
		{
		    oldrow = Ai [p] ;
		    newrow = Pinv [oldrow] ;
		    if (newrow < k1)
		    {
			Offi [poff] = oldrow ;
			Offx [poff] = Ax [p] ;
			poff++ ;
		    }
		    else
		    {
			ASSERT (newrow == k1) ;
			PRINTF (("Singleton block %d", block)) ;
			PRINT_ENTRY (Ax [p]) ;
			s = Ax [p] ;
		    }
		}
	    }
	    else
	    {
		/* row scaling */
		for (p = Ap [oldcol] ; p < pend ; p++)
		{
		    oldrow = Ai [p] ;
		    newrow = Pinv [oldrow] ;
		    if (newrow < k1)
		    {
			Offi [poff] = oldrow ;
			/*Offx [poff] = Ax [p] / Rs [oldrow] ; */
			SCALE_DIV_ASSIGN (Offx [poff], Ax [p], Rs [oldrow]) ;
			poff++ ;
		    }
		    else
		    {
			ASSERT (newrow == k1) ;
			PRINTF (("Singleton block %d ", block)) ;
			PRINT_ENTRY (Ax[p]) ;
			SCALE_DIV_ASSIGN (s, Ax [p], Rs [oldrow]) ;
		    }
		}
	    }

	    Singleton [block] = s ;

	    if (IS_ZERO (s))
	    {
		/* singular singleton */
		Common->status = KLU_SINGULAR ;
		Common->numerical_rank = k1 ;
		Common->singular_col = oldcol ;
		if (Common->halt_if_singular)
		{
		    return ;
		}
	    }

	    Offp [k1+1] = poff ;
	    Pnum [k1] = P [k1] ;
	    lnz++ ;
	    unz++ ;

	}
	else
	{

	    /* -------------------------------------------------------------- */
	    /* construct and factorize the kth block */
	    /* -------------------------------------------------------------- */

	    if (Lnz [block] < 0)
	    {
		/* COLAMD was used - no estimate of fill-in */
		/* use 10 times the nnz in A, plus n */
		lsize = -(Common->initmem) ;
	    }
	    else
	    {
		lsize = Common->initmem_amd * Lnz [block] + nk ;
	    }

	    /* allocates 5 arrays:
	     * Lbip [block], Ubip [block], Lblen [block], Ublen [block],
	     * LUbx [block] */
	    KLU_kernel_factor (nk, Ap, Ai, Ax, Q, lsize,
		    &LUbx [block], Udiag [block], Lblen [block], Ublen [block],
		    Lbip [block], Ubip [block], Pblock, &lnz_block, &unz_block,
		    X, Iwork,
		    /* BTF and scale-related arguments: */
		    k1, Pinv, Rs, Offp, Offi, Offx, Common) ;

	    if (Common->status < KLU_OK ||
	       (Common->status == KLU_SINGULAR && Common->halt_if_singular))
	    {
		/* out of memory, invalid inputs, or singular */
		return ;
	    }

	    PRINTF (("\n----------------------- L %d:\n", block)) ;
	    ASSERT (KLU_valid_LU (nk, TRUE, Lbip [block],
		    Lblen [block], LUbx [block])) ;
	    PRINTF (("\n----------------------- U %d:\n", block)) ;
	    ASSERT (KLU_valid_LU (nk, FALSE, Ubip [block],
		    Ublen [block], LUbx [block])) ;

	    /* -------------------------------------------------------------- */
	    /* get statistics */
	    /* -------------------------------------------------------------- */

	    lnz += lnz_block ;
	    unz += unz_block ;
	    max_lnz_block = MAX (max_lnz_block, lnz_block) ;
	    max_unz_block = MAX (max_unz_block, unz_block) ;

	    if (Lnz [block] == EMPTY)
	    {
		/* revise estimate for subsequent factorization */
		Lnz [block] = MAX (lnz_block, unz_block) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* combine the klu row ordering with the symbolic pre-ordering */
	    /* -------------------------------------------------------------- */

	    PRINTF (("Pnum, 1-based:\n")) ;
	    for (k = 0 ; k < nk ; k++)
	    {
		ASSERT (k + k1 < n) ;
		ASSERT (Pblock [k] + k1 < n) ;
		Pnum [k + k1] = P [Pblock [k] + k1] ;
		PRINTF (("Pnum (%d + %d + 1 = %d) = %d + 1 = %d\n",
		    k, k1, k+k1+1, Pnum [k+k1], Pnum [k+k1]+1)) ;
	    }

	    /* the local pivot row permutation Pblock is no longer needed */
	}
    }
    ASSERT (nzoff == Offp [n]) ;
    PRINTF (("\n------------------- Off diagonal entries:\n")) ;
    ASSERT (KLU_valid (n, Offp, Offi, Offx)) ;

    Numeric->lnz = lnz ;
    Numeric->unz = unz ;
    Numeric->max_lnz_block = max_lnz_block ;
    Numeric->max_unz_block = max_unz_block ;

    /* Numeric->flops = EMPTY ;		TODO not yet computed */

    /* compute the inverse of Pnum */
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++)
    {
	Pinv [k] = EMPTY ;
    }
#endif
    for (k = 0 ; k < n ; k++)
    {
	ASSERT (Pnum [k] >= 0 && Pnum [k] < n) ;
	Pinv [Pnum [k]] = k ;
    }
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++) ASSERT (Pinv [k] != EMPTY) ;
#endif

    /* permute scale factors Rs according to pivotal row order */
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

    PRINTF (("\n------------------- Off diagonal entries, old:\n")) ;
    ASSERT (KLU_valid (n, Offp, Offi, Offx)) ;

    /* apply the pivot row permutations to the off-diagonal entries */
    for (p = 0 ; p < nzoff ; p++)
    {
	ASSERT (Offi [p] >= 0 && Offi [p] < n) ;
	Offi [p] = Pinv [Offi [p]] ;
    }

    PRINTF (("\n------------------- Off diagonal entries, new:\n")) ;
    ASSERT (KLU_valid (n, Offp, Offi, Offx)) ;

#ifndef NDEBUG
    {
	PRINTF (("\n ############# KLU_BTF_FACTOR done, nblocks %d\n",nblocks));
	Entry *singleton = Numeric->Singleton ;
	for (block = 0 ; block < nblocks && Common->status == KLU_OK ; block++)
	{
	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    PRINTF (("\n======================klu_factor output: k1 %d k2 %d nk %d\n",k1,k2,nk)) ;
	    if (nk == 1)
	    {
		PRINTF (("singleton  ")) ;
		/* ENTRY_PRINT (singleton [block]) ; */
		PRINT_ENTRY (singleton [block]) ;
	    }
	    else
	    {
		int *Lip, *Uip, *Llen, *Ulen ;
		Unit *LU ;
		Lip = Lbip [block] ;
		Llen = Lblen [block] ;
		LU = (Unit *) Numeric->LUbx [block] ;
		PRINTF (("\n---- L block %d\n", block));
		ASSERT (KLU_valid_LU (nk, TRUE, Lip, Llen, LU)) ;
		Uip = Ubip [block] ;
		Ulen = Ublen [block] ;
		PRINTF (("\n---- U block %d\n", block)) ;
		ASSERT (KLU_valid_LU (nk, FALSE, Uip, Ulen, LU)) ;
	    }
	}
    }
#endif
}


/* ========================================================================== */
/* === CLEAR_PTR ============================================================ */
/* ========================================================================== */

/* Set an array of pointers to NULL */

#define CLEAR_PTR(Ptr,size) \
{ \
    int ii ; \
    if (Ptr != NULL) \
    { \
	for (ii = 0 ; ii < size ; ii++) \
	{ \
	    Ptr [ii] = NULL ; \
	} \
    } \
}


/* ========================================================================== */
/* === klu_factor =========================================================== */
/* ========================================================================== */

klu_numeric *KLU_factor		/* returns NULL if error, or a valid
				   klu_numeric object if successful */
(
    /* --- inputs --- */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    double Ax [ ],
    klu_symbolic *Symbolic,
    /* -------------- */
    klu_common *Common
)
{
    int n, nzoff, nblocks, maxblock, block, k1, k2, nk, ok = TRUE ;
    int *R ;
    int **Lbip, **Ubip, **Lblen, **Ublen ;
    klu_numeric *Numeric ;
    Unit **Udiag ;
    size_t n1, nzoff1, nunits, s, b6, n3, nk1 ;

    if (Common == NULL)
    {
	return (NULL) ;
    }
    Common->status = KLU_OK ;
    Common->numerical_rank = EMPTY ;
    Common->singular_col = EMPTY ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Symbolic object */
    /* ---------------------------------------------------------------------- */

    /* check for a valid Symbolic object */
    if (Symbolic == NULL)
    {
	Common->status = KLU_INVALID ;
	return (NULL) ;
    }

    n = Symbolic->n ;
    nzoff = Symbolic->nzoff ;
    nblocks = Symbolic->nblocks ;
    maxblock = Symbolic->maxblock ;
    R = Symbolic->R ;
    PRINTF (("klu_factor:  n %d nzoff %d nblocks %d maxblock %d\n",
	n, nzoff, nblocks, maxblock)) ;

    /* ---------------------------------------------------------------------- */
    /* get control parameters and make sure they are in the proper range */
    /* ---------------------------------------------------------------------- */

    Common->initmem_amd = MAX (1.0, Common->initmem_amd) ;
    Common->initmem = MAX (1.0, Common->initmem) ;
    Common->tol = MIN (Common->tol, 1.0) ;
    Common->tol = MAX (0.0, Common->tol) ;
    Common->growth = MAX (1.0, Common->growth) ;

    /* ---------------------------------------------------------------------- */
    /* allocate the Numeric object  */
    /* ---------------------------------------------------------------------- */

    /* this will not cause size_t overflow (already checked by klu_symbolic) */
    n1 = ((size_t) n) + 1 ;
    nzoff1 = ((size_t) nzoff) + 1 ;

    Numeric = klu_malloc (sizeof (klu_numeric), 1, Common) ;
    if (Common->status < KLU_OK)
    {
	/* out of memory */
	Common->status = KLU_OUT_OF_MEMORY ;
	return (NULL) ;
    }
    Numeric->nblocks = nblocks ;
    Numeric->Pnum = klu_malloc (n, sizeof (int), Common) ;
    Numeric->Offp = klu_malloc (n1, sizeof (int), Common) ;
    Numeric->Offi = klu_malloc (nzoff1, sizeof (int), Common) ;
    Numeric->Offx = klu_malloc (nzoff1, sizeof (Entry), Common) ;
    Numeric->Singleton = klu_malloc (nblocks, sizeof (Entry), Common) ;

    Numeric->Lbip  = klu_malloc (nblocks, sizeof (int *), Common) ;
    Numeric->Ubip  = klu_malloc (nblocks, sizeof (int *), Common) ;
    Numeric->Lblen = klu_malloc (nblocks, sizeof (int *), Common) ;
    Numeric->Ublen = klu_malloc (nblocks, sizeof (int *), Common) ;
    Numeric->LUbx  = klu_malloc (nblocks, sizeof (Unit *), Common) ;
    Numeric->Udiag = klu_malloc (nblocks, sizeof (Unit *), Common) ;

    if (Common->scale > 0)
    {
	Numeric->Rs = klu_malloc (n, sizeof (double), Common) ;
    }
    else
    {
	/* no scaling */
	Numeric->Rs = NULL ;
    }

    Numeric->Pinv = klu_malloc (n, sizeof (int), Common) ;

    /* allocate permanent workspace for factorization and solve.
     * Note that the solver will use an Xwork of size 4n, whereas
     * the factorization codes use an Xwork of size n and integer space
     * (Iwork) of size 6n. klu_condest uses an Xwork of size 2n. */

    s = klu_mult_size_t (n, sizeof (Entry), &ok) ;
    n3 = klu_mult_size_t (n, 3 * sizeof (Entry), &ok) ;
    b6 = klu_mult_size_t (maxblock, 6 * sizeof (int), &ok) ;
    Numeric->worksize = klu_add_size_t (s, MAX (n3, b6), &ok) ;
    if (!ok)
    {
	/* problem too large (almost impossible to happen here) */
	Common->status = KLU_TOO_LARGE ;
	KLU_free_numeric (&Numeric, Common) ;
	return (NULL) ;
    }

/*
    Numeric->worksize = n * sizeof (Entry) +
	MAX (n * 3 *sizeof (Entry), 6*maxblock * sizeof (int)) ;
*/

    Numeric->Work = klu_malloc (Numeric->worksize, 1, Common) ;
    Numeric->Xwork = Numeric->Work ;
    Numeric->Iwork = (int *) ((Entry *) Numeric->Xwork + n) ;

    if (Common->status < KLU_OK)
    {
	/* out of memory */
	KLU_free_numeric (&Numeric, Common) ;
	return (NULL) ;
    }

    /* clear the pointer arrays, so that klu_free_numeric works OK */
    CLEAR_PTR (Numeric->Lbip,  nblocks) ;
    CLEAR_PTR (Numeric->Ubip,  nblocks) ;
    CLEAR_PTR (Numeric->Lblen, nblocks) ;
    CLEAR_PTR (Numeric->Ublen, nblocks) ;
    CLEAR_PTR (Numeric->LUbx,  nblocks) ;
    CLEAR_PTR (Numeric->Udiag, nblocks) ;

    /* allocate the column pointer arrays for each block */
    Lbip = Numeric->Lbip ;
    Ubip = Numeric->Ubip ;
    Lblen = Numeric->Lblen ;
    Ublen = Numeric->Ublen ;
    Udiag = (Unit **) Numeric->Udiag ;

    for (block = 0 ; block < nblocks ; block++)
    {
	k1 = R [block] ;
	k2 = R [block+1] ;
	nk = k2 - k1 ;
	nk1 = ((size_t) nk) + 1 ;   /* cannot overflow */
	if (nk > 1)
	{
	    nunits = UNITS (Entry, nk) ;
	    Lbip [block]  = klu_malloc (nk1, sizeof (int), Common) ;
	    Ubip [block]  = klu_malloc (nk1, sizeof (int), Common) ;
	    Lblen [block] = klu_malloc (nk1, sizeof (int), Common) ;
	    Ublen [block] = klu_malloc (nk1, sizeof (int), Common) ;
	    Udiag [block] = klu_malloc (nunits, sizeof (Unit), Common) ;
	    if (Common->status < KLU_OK)
	    {
		/* out of memory */
	 	KLU_free_numeric (&Numeric, Common) ;
		return (NULL) ;
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* factorize the blocks */
    /* ---------------------------------------------------------------------- */

    klu_factor2 (Ap, Ai, (Entry *) Ax, Symbolic, Numeric, Common) ;

    /* ---------------------------------------------------------------------- */
    /* return or free the Numeric object */
    /* ---------------------------------------------------------------------- */

    if (Common->status < KLU_OK)
    {
	/* out of memory or inputs invalid */
	KLU_free_numeric (&Numeric, Common) ;
    }
    else if (Common->status == KLU_SINGULAR)
    {
	if (Common->halt_if_singular)
	{
	    /* Matrix is singular, and the Numeric object is only partially
	     * defined because we halted early.  This is the default case for
	     * a singular matrix. */
	    KLU_free_numeric (&Numeric, Common) ;
	}
    }
    else if (Common->status == KLU_OK)
    {
	/* successful non-singular factorization */
	Common->numerical_rank = n ;
	Common->singular_col = n ;
    }
    return (Numeric) ;
}
