/* ========================================================================== */
/* === klu_analyze ========================================================== */
/* ========================================================================== */

/* Order the matrix using BTF (or not), and then AMD, COLAMD, the natural
 * ordering, or the user-provided-function on the blocks.  Does not support
 * using a given ordering (use klu_analyze_given for that case). */

#include "klu_internal.h"

/* ========================================================================== */
/* === worker =============================================================== */
/* ========================================================================== */

static int worker	/* returns KLU_OK or < 0 if error */
(
    /* inputs, not modified */
    int n,		/* A is n-by-n */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    int nblocks,	/* # of blocks */
    int Pbtf [ ],	/* BTF row permutation */
    int Qbtf [ ],	/* BTF col permutation */
    int R [ ],		/* size n+1, but only Rbtf [0..nblocks] is used */
    int ordering,	/* what ordering to use */

    /* output only, not defined on input */
    int P [ ],		/* size n */
    int Q [ ],		/* size n */
    double Lnz [ ],	/* size n, but only Lnz [0..nblocks-1] is used */

    /* workspace, not defined on input or output */
    int Pamd [ ],	/* size maxblock */
    int Cp [ ],		/* size maxblock+1 */
    int Ci [ ],		/* size nz */
    int Ep [ ],		/* size maxblock+1 */
    int Pinv [ ],	/* size maxblock */

    /* input/output */
    klu_symbolic *Symbolic,
    klu_common *Common
)
{
    double amd_Info [AMD_INFO], lnz, lnz1, flops ;
    int *RowCount, *W, k1, k2, nk, k, block, oldcol, pend, row, newcol,
	result, pc, p, newrow, maxnz, nzoff, ok ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    /* use Pamd as workspace for RowCount and W */
    RowCount = Pamd ;
    W = Pamd ;

    /* compute the inverse of Pbtf */
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++)
    {
	P [k] = EMPTY ;
	Q [k] = EMPTY ;
	Pinv [k] = EMPTY ;
    }
#endif
    for (k = 0 ; k < n ; k++)
    {
	ASSERT (Pbtf [k] >= 0 && Pbtf [k] < n) ;
	Pinv [Pbtf [k]] = k ;
    }
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++) ASSERT (Pinv [k] != EMPTY) ;
#endif
    nzoff = 0 ;
    lnz = 0 ;
    maxnz = 0 ;
    flops = 0 ;
    Symbolic->symmetry = EMPTY ;	/* only computed by AMD */

    /* ---------------------------------------------------------------------- */
    /* order each block */
    /* ---------------------------------------------------------------------- */

    for (block = 0 ; block < nblocks ; block++)
    {

	/* ------------------------------------------------------------------ */
	/* the block is from rows/columns k1 to k2-1 */
	/* ------------------------------------------------------------------ */

	k1 = R [block] ;
	k2 = R [block+1] ;
	nk = k2 - k1 ;
	PRINTF (("BLOCK %d, k1 %d k2-1 %d nk %d\n", block, k1, k2-1, nk)) ;

	if (nk == 1)
	{

	    /* -------------------------------------------------------------- */
	    /* singleton case */
	    /* -------------------------------------------------------------- */

	    Lnz [block] = 1 ;
	    P [k1] = Pbtf [k1] ;
	    Q [k1] = Qbtf [k1] ;
	    oldcol = Q [k1] ;
	    pend = Ap [oldcol+1] ;
	    for (p = Ap [oldcol] ; p < pend ; p++)
	    {
		if (Pinv [Ai [p]] < k1)
		{
		    nzoff++ ;
		}
		else
		{
		    lnz++ ;
		}
	    }

	}
	else
	{

	    /* -------------------------------------------------------------- */
	    /* construct the kth block, C */
	    /* -------------------------------------------------------------- */

	    Lnz [block] = EMPTY ;
	    for (row = 0 ; row < nk ; row++)
	    {
		RowCount [row] = 0 ;
	    }
	    pc = 0 ;
	    for (k = k1 ; k < k2 ; k++)
	    {
		newcol = k-k1 ;
		Cp [newcol] = pc ;
		oldcol = Qbtf [k] ;
		pend = Ap [oldcol+1] ;
		for (p = Ap [oldcol] ; p < pend ; p++)
		{
		    newrow = Pinv [Ai [p]] ;
		    if (newrow < k1)
		    {
			nzoff++ ;
		    }
		    else
		    {
			/* (newrow,newcol) is an entry in the block */
			ASSERT (newrow < k2) ;
			newrow -= k1 ;
			Ci [pc++] = newrow ;
			RowCount [newrow]++ ;
		    }
		}
	    }
	    Cp [nk] = pc ;
	    maxnz = MAX (maxnz, pc) ;
	    ASSERT (KLU_valid (nk, Cp, Ci, NULL)) ;

	    /* -------------------------------------------------------------- */
	    /* order the block C */
	    /* -------------------------------------------------------------- */

	    if (ordering == 0)
	    {

		/* ---------------------------------------------------------- */
		/* order the block with AMD (C+C') */
		/* ---------------------------------------------------------- */

#if 0
		/* since AMD requires a sorted input, compute E = C' */
		Ep [0] = 0 ;
		for (row = 0 ; row < nk ; row++)
		{
		    Ep [row+1] = Ep [row] + RowCount [row] ;
		}
		ASSERT (Ep [nk] == pc) ;
		for (row = 0 ; row < nk ; row++)
		{
		    W [row] = Ep [row] ;
		}
		for (col = 0 ; col < nk ; col++)
		{
		    pend = Cp [col+1] ;
		    for (p = Cp [col] ; p < pend ; p++)
		    {
			ASSERT (W [Ci [p]] >= 0 && W [Ci [p]] < pc) ;
			Ei [W [Ci [p]]++] = col ;
		    }
		}
		ASSERT (KLU_valid (nk, Ep, Ei, NULL)) ;
#endif

		/* AMD memory management routines */
		amd_malloc  = Common->malloc_memory ;
		amd_free    = Common->free_memory ;
		amd_calloc  = Common->calloc_memory ;
		amd_realloc = Common->realloc_memory ;

		result = amd_order (nk, Cp, Ci, Pamd, NULL, amd_Info) ;
		if (result == AMD_OUT_OF_MEMORY)
		{
		    return (KLU_OUT_OF_MEMORY) ;
		}
		else if (result == AMD_INVALID)
		{
		    /* fatal error - something is corrupted in the matrix E */
		    PRINTF (("AMD invalid!\n")) ;
		    return (KLU_INVALID) ;
		}

		/* get the ordering statistics from AMD */
		lnz1 = (int) (amd_Info [AMD_LNZ]) + nk ;
		Lnz [block] = lnz1 ;
		lnz += lnz1 ;
		if (pc == maxnz)
		{
		    /* get the symmetry of the biggest block */
		    Symbolic->symmetry = amd_Info [AMD_SYMMETRY] ;
		}
		flops += 2 * amd_Info [AMD_NMULTSUBS_LU] + amd_Info [AMD_NDIV] ;

	    }
	    else if (ordering == 1)
	    {

		/* ---------------------------------------------------------- */
		/* order the block with COLAMD (C) */
		/* ---------------------------------------------------------- */

		int *AA, cstats [COLAMD_STATS] ;
		size_t Alen ;

		PRINTF (("calling COLAMD\n")) ;
		Alen = colamd_recommended (pc, n, n) ;
		if (Alen == 0)
		{
		    /* problem is too large */
		    return (KLU_TOO_LARGE) ;
		}
		AA = klu_malloc (Alen, sizeof (int), Common) ;
		if (AA == (int *) NULL)
		{
		    /* out of memory */
		    return (KLU_OUT_OF_MEMORY) ;
		}

		/* copy the matrix C into AA and Ep */
		for (k = 0 ; k <= nk ; k++)
		{
		    Ep [k] = Cp [k] ;
		}
		for (p = 0 ; p < pc ; p++)
		{
		    AA [p] = Ci [p] ;
		}

		/* order (and destroy) AA, returning col permutation in Ep */
		ok = colamd (nk, nk, (int) Alen, AA, Ep, NULL, cstats) ;
		PRINTF (("COLAMD done\n")) ;

		/* free the workspace */
		AA = klu_free (AA, Common) ;

		if (!ok)
		{
		    /* fatal error - something is corrupted in the matrix */
		    PRINTF (("COLAMD invalid!\n")) ;
		    return (KLU_INVALID) ;
		}

		/* copy the permutation from Ep to Pamd */
		for (k = 0 ; k < nk ; k++)
		{
		    Pamd [k] = Ep [k] ;
		}

		/* COLAMD does not return an estimate on fill-in or flops */
		flops = EMPTY ;
		lnz = EMPTY ;
		Lnz [block] = EMPTY ;

		PRINTF (("done COLAMD\n")) ;
	    }
	    else if (ordering == 3 && Common->user_order != NULL)
	    {

		/* ---------------------------------------------------------- */
		/* pass the block to the user-provided ordering function */
		/* ---------------------------------------------------------- */

		Lnz [block] = (Common->user_order) (nk, Cp, Ci, Pamd,
			Common->user_data) ;
		if (Lnz [block] == 0)
		{
		    PRINTF (("user ordering function failed\n")) ;
		    return (KLU_INVALID) ;
		}

	    }
	    else
	    {
		PRINTF (("invalid ordering\n")) ;
		return (KLU_INVALID) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* combine the preordering with the BTF ordering */
	    /* -------------------------------------------------------------- */

	    PRINTF (("Pamd, 1-based:\n")) ;
	    for (k = 0 ; k < nk ; k++)
	    {
		ASSERT (k + k1 < n) ;
		ASSERT (Pamd [k] + k1 < n) ;
		Q [k + k1] = Qbtf [Pamd [k] + k1] ;
	    }
	    for (k = 0 ; k < nk ; k++)
	    {
		ASSERT (k + k1 < n) ;
		ASSERT (Pamd [k] + k1 < n) ;
		P [k + k1] = Pbtf [Pamd [k] + k1] ;
	    }
	}
    }

    PRINTF (("nzoff %d  Ap[n] %d\n", nzoff, Ap [n])) ;
    ASSERT (nzoff >= 0 && nzoff <= Ap [n]) ;

    /* return estimates of # of nonzeros in L including diagonal */
    Symbolic->lnz = lnz ;		/* EMPTY if COLAMD used */
    Symbolic->unz = lnz ;		/* EMPTY if COLAMD used */
    Symbolic->nzoff = nzoff ;
    Symbolic->est_flops = flops ;	/* EMPTY if COLAMD used */
    return (KLU_OK) ;
}


/* ========================================================================== */
/* === order_and_analyze ==================================================== */
/* ========================================================================== */

/* Orders the matrix with or with BTF, then orders each block with AMD, COLAMD,
 * or the user ordering function.  Does not handle the natural or given
 * ordering cases. */

static klu_symbolic *order_and_analyze	/* returns NULL if error, or a valid
					   klu_symbolic object if successful */
(
    /* inputs, not modified */
    int n,		/* A is n-by-n */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    /* --------------------- */
    klu_common *Common
)
{
    klu_symbolic *Symbolic ;
    double *Lnz ;
    int *Qbtf, *Cp, *Ci, *Pinv, *Pamd, *Ep, *Pbtf, *P, *Q, *R ;
    int nblocks, nz, block, maxblock, k1, k2, nk, j, i, p, pend, do_btf, nzdiag,
	ordering, k, ok ;
    size_t n1, n5, nz1, m1 ;

    /* ---------------------------------------------------------------------- */
    /* determine if input matrix is valid, and get # of nonzeros */
    /* ---------------------------------------------------------------------- */

    /* A is n-by-n, with n > 0.  Ap [0] = 0 and nz = Ap [n] >= 0 required.
     * Ap [j] <= Ap [j+1] must hold for all j = 0 to n-1.  Row indices in Ai
     * must be in the range 0 to n-1, and no duplicate entries can be present.
     * The list of row indices in each column of A need not be sorted.
     */

    if (n <= 0 || Ap == NULL || Ai == NULL)
    {
	/* Ap and Ai must be present, and n must be > 0 */
	Common->status = KLU_INVALID ;
	return (NULL) ;
    }
    nz = Ap [n] ;
    if (Ap [0] != 0 || nz < 0)
    {
	/* nz must be >= 0 and Ap [0] must equal zero */
	Common->status = KLU_INVALID ;
	return (NULL) ;
    }

    /* check for size_t overflow */
    do_btf = Common->btf ;
    do_btf = (do_btf) ? TRUE : FALSE ;
    ok = TRUE ;
    n1 = klu_add_size_t (n, 1, &ok) ;
    nz1 = klu_add_size_t (nz, 1, &ok) ;
    n5 = do_btf ? (klu_mult_size_t (n, 5, &ok)) : 0 ;
    if (!ok)
    {
	/* problem is too large */
	Common->status = KLU_TOO_LARGE ;
	return (NULL) ;
    }

    for (j = 0 ; j < n ; j++)
    {
	if (Ap [j] > Ap [j+1])
	{
	    /* column pointers must be non-decreasing */
	    Common->status = KLU_INVALID ;
	    return (NULL) ;
	}
    }

    P = klu_malloc (n, sizeof (int), Common) ;
    if (Common->status < KLU_OK)
    {
	return (NULL) ;
    }
    for (i = 0 ; i < n ; i++)
    {
	P [i] = EMPTY ;
    }
    nzdiag = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	for (p = Ap [j] ; p < pend ; p++)
	{
	    i = Ai [p] ;
	    if (i < 0 || i >= n || P [i] == j)
	    {
		/* row index out of range, or duplicate entry */
		P = klu_free (P, Common) ;
		Common->status = KLU_INVALID ;
		return (NULL) ;
	    }
	    if (i == j)
	    {
		/* count the number of diagonal entries */
		nzdiag++ ;
	    }
	    /* flag row i as appearing in column j */
	    P [i] = j ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* allocate the Symbolic object */
    /* ---------------------------------------------------------------------- */

    Symbolic = klu_malloc (sizeof (klu_symbolic), 1, Common) ;
    if (Common->status < KLU_OK)
    {
	klu_free (P, Common) ;
	return (NULL) ;
    }

    Q = klu_malloc (n, sizeof (int), Common) ;
    R = klu_malloc (n1, sizeof (int), Common) ;
    Lnz = klu_malloc (n, sizeof (double), Common) ;

    Symbolic->n = n ;
    Symbolic->nz = nz ;
    Symbolic->P = P ;
    Symbolic->Q = Q ;
    Symbolic->R = R ;
    Symbolic->Lnz = Lnz ;

    if (Common->status < KLU_OK)
    {
	klu_free_symbolic (&Symbolic, Common) ;
	return (NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate workspace for BTF permutation */
    /* ---------------------------------------------------------------------- */

    Pbtf = klu_malloc (n, sizeof (int), Common) ;
    Qbtf = klu_malloc (n, sizeof (int), Common) ;
    if (Common->status < KLU_OK)
    {
	klu_free (Pbtf, Common) ;
	klu_free (Qbtf, Common) ;
	klu_free_symbolic (&Symbolic, Common) ;
	return ((klu_symbolic *) NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the common parameters for BTF and ordering method */
    /* ---------------------------------------------------------------------- */

    ordering = Common->ordering ;
    ordering = MAX (ordering, 0) ;	    /* 0: AMD */
    ordering = MIN (ordering, 3) ;	    /* 1: COLAMD */
    Symbolic->ordering = ordering ;
    Symbolic->do_btf = do_btf ;
    Symbolic->structural_rank = EMPTY ;

    /* ---------------------------------------------------------------------- */
    /* find the block triangular form (if requested) */
    /* ---------------------------------------------------------------------- */

    if (do_btf)
    {

	int *Work ;
	Work = klu_malloc (n5, sizeof (int), Common) ;
	if (Common->status < KLU_OK)
	{
	    /* out of memory */
	    klu_free (Pbtf, Common) ;
	    klu_free (Qbtf, Common) ;
	    klu_free_symbolic (&Symbolic, Common) ;
	    return ((klu_symbolic *) NULL) ;
	}

	nblocks = btf_order (n, Ap, Ai, Pbtf, Qbtf, R,
		&(Symbolic->structural_rank), Work) ;
	Common->structural_rank = Symbolic->structural_rank ;

	klu_free (Work, Common) ;

	/* unflip Qbtf if the matrix does not have full structural rank */
	if (Symbolic->structural_rank < n)
	{
	    for (k = 0 ; k < n ; k++)
	    {
		Qbtf [k] = MAXTRANS_UNFLIP (Qbtf [k]) ;
	    }
	}

	/* find the size of the largest block */
	maxblock = 1 ;
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = R [block] ;
	    k2 = R [block+1] ;
	    nk = k2 - k1 ;
	    PRINTF (("block %d size %d\n", block, nk)) ;
	    maxblock = MAX (maxblock, nk) ;
	}
    }
    else
    {
	/* BTF not requested */
	nblocks = 1 ;
	maxblock = n ;
	R [0] = 0 ;
	R [1] = n ;
	for (k = 0 ; k < n ; k++)
	{
	    Pbtf [k] = k ;
	    Qbtf [k] = k ;
	}
    }

    Symbolic->nblocks = nblocks ;

    PRINTF (("maxblock size %d\n", maxblock)) ;
    Symbolic->maxblock = maxblock ;

    /* TODO: merge adjacent 1-by-1 blocks into an upper triangular block */

    /* ---------------------------------------------------------------------- */
    /* allocate more workspace, for klu_analyze_worker */
    /* ---------------------------------------------------------------------- */

    /* if we get here, n+1 is already safe, so maxblock+1 is safe too */
    m1 = ((size_t) maxblock) + 1 ;

    Pamd = klu_malloc (maxblock, sizeof (int), Common) ;
    Cp   = klu_malloc (m1, sizeof (int), Common) ;
    Ep   = klu_malloc (m1, sizeof (int), Common) ;
    Ci   = klu_malloc (nz1, sizeof (int), Common) ;
    Pinv = klu_malloc (n, sizeof (int), Common) ;

    /* ---------------------------------------------------------------------- */
    /* order each block of the BTF matrix using AMD or COLAMD */
    /* ---------------------------------------------------------------------- */

    if (Common->status == KLU_OK)
    {
	PRINTF (("calling klu_analyze_worker\n")) ;
	Common->status = worker (n, Ap, Ai, nblocks, Pbtf, Qbtf, R,
	    ordering, P, Q, Lnz, Pamd, Cp, Ci, Ep, Pinv, Symbolic, Common) ;
	PRINTF (("klu_analyze_worker done\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free all workspace */
    /* ---------------------------------------------------------------------- */

    klu_free (Pamd, Common) ;
    klu_free (Cp, Common) ;
    klu_free (Ci, Common) ;
    klu_free (Ep, Common) ;
    klu_free (Pinv, Common) ;
    klu_free (Pbtf, Common) ;
    klu_free (Qbtf, Common) ;

    /* ---------------------------------------------------------------------- */
    /* return the symbolic object */
    /* ---------------------------------------------------------------------- */

    if (Common->status < KLU_OK)
    {
	klu_free_symbolic (&Symbolic, Common) ;
    }
    return (Symbolic) ;
}


/* ========================================================================== */
/* === klu_analyze ====================================================== */
/* ========================================================================== */

klu_symbolic *klu_analyze	/* returns NULL if error, or a valid
				   klu_symbolic object if successful */
(
    /* inputs, not modified */
    int n,		/* A is n-by-n */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    /* -------------------- */
    klu_common *Common
)
{

    /* ---------------------------------------------------------------------- */
    /* get the control parameters for BTF and ordering method */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
	return (NULL) ;
    }
    Common->status = KLU_OK ;
    Common->structural_rank = EMPTY ;

    /* ---------------------------------------------------------------------- */
    /* order and analyze */
    /* ---------------------------------------------------------------------- */

    if (Common->ordering == 2)
    {
	/* natural ordering */
	return (klu_analyze_given (n, Ap, Ai, NULL, NULL, Common)) ;
    }
    else
    {
	/* order with P and Q */
	return (order_and_analyze (n, Ap, Ai, Common)) ;
    }
}
