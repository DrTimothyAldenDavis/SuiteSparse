/* ========================================================================== */
/* === klu_diagnostics ====================================================== */
/* ========================================================================== */

/* Linear algebraic diagnostics:
 * klu_growth:	reciprocal pivot growth, takes O(|A|+|U|) time
 * klu_condest:	condition number estimator, takes about O(|A|+5*(|L|+|U|)) time
 * klu_flops:	compute # flops required to factorize A into L*U
 * klu_rcond:	compute a really cheap estimate of the reciprocal of the
 *		condition number, min(abs(diag(U))) / max(abs(diag(U))).
 *		Takes O(n) time.
 */

#include "klu_internal.h"

/* ========================================================================== */
/* === klu_growth =========================================================== */
/* ========================================================================== */

/* Compute the reciprocal pivot growth factor */

int KLU_growth		/* return TRUE if successful, FALSE otherwise */
(
    int *Ap,
    int *Ai,
    double *Ax,
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    double *growth,
    klu_common *Common
)
{
    double temp, max_ai, max_ui, min_block_growth ;
    Entry aik ;
    int *Q, *Ui, *Uip, *Ulen, *Pinv ;
    Unit *LU ;
    Entry *Aentry, *Ux, *Ukk ;
    double *Rs ;
    int **Ubip ;
    int i, newrow, oldrow, k1, k2, nk, j, oldcol, k, pend, len ;

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    Common->status = KLU_OK ;

    if (Numeric == NULL)
    {
	*growth = 0 ;
	Common->status = KLU_SINGULAR ;
	return (TRUE) ;
    }

    Aentry = (Entry *) Ax ;
    Pinv = Numeric->Pinv ;
    Rs = Numeric->Rs ;
    Q = Symbolic->Q ;
    *growth = 1 ;

    /* The method of calculating the reciprocal pivot growth is :
     * Iterate over each of the blocks.  Within each block, iterate over each
     * column to find the minimum value for the block.  Compare the value of
     * the block with the minimum value computed for all the blocks till now,
     * to find out the new minimum.
     */

    for (i = 0 ; i < Symbolic->nblocks ; i++)
    {
	k1 = Symbolic->R[i] ;
	k2 = Symbolic->R[i+1] ;
	nk = k2 - k1 ;
	/* skip singleton blocks*/
	if (nk == 1)
	{
	    continue ;
	}
	LU = (Unit *) Numeric->LUbx[i] ;
	Ubip = Numeric->Ubip ;
	Uip = Ubip [i] ;
	Ulen = Numeric->Ublen [i] ;
	Ukk = (Entry *) Numeric->Udiag [i] ;
	min_block_growth = 1 ;
	for (j = 0 ; j < nk ; j++)
	{
	    max_ai = 0 ;
	    max_ui = 0 ;
	    oldcol = Q[j + k1] ;
	    pend = Ap [oldcol + 1] ;
	    for (k = Ap[oldcol] ; k < pend ; k++)
	    {
		oldrow = Ai [k] ;
		newrow = Pinv [oldrow] ;
		/* skip entry outside the block */
                if (newrow < k1)
		{
	 	    continue ;
		}
		ASSERT (newrow < k2) ;
		if (Rs != NULL)
		{
		    /* aik = Aentry [k] / Rs [oldrow] */
		    SCALE_DIV_ASSIGN (aik, Aentry [k], Rs [oldrow]) ;
		}
		else
		{
		    aik = Aentry [k] ;
		}
		/* temp = ABS (aik) */
		ABS (temp, aik) ;
		if (temp > max_ai)
		{
		    max_ai = temp ;
		}
	    }

	    GET_POINTER (LU, Uip, Ulen, Ui, Ux, j, len) ;
	    for (k = 0 ; k < len ; k++)
	    {
	        /* temp = ABS (Ux [k]) */
	        ABS (temp, Ux [k]) ;
		if (temp > max_ui)
		{
		    max_ui = temp ;
		}
	    }
	    /* consider the diagonal element */
	    ABS (temp, Ukk [j]) ;
	    if (temp > max_ui)
	    {
		max_ui = temp ;
	    }

	    /* if max_ui is 0, skip the column */
	    if (SCALAR_IS_ZERO (max_ui))
	    {
		continue ;
	    }
	    temp = max_ai / max_ui ;
	    if (temp < min_block_growth)
	    {
		min_block_growth = temp ;
	    }
	}

	if (min_block_growth < *growth)
	{
	    *growth = min_block_growth ;
	}
    }
    return (TRUE) ;
}


/* ========================================================================== */
/* === klu_condest ========================================================== */
/* ========================================================================== */

/* Estimate the condition number.  Uses Higham and Tisseur's algorithm
 * (A block algorithm for matrix 1-norm estimation, with applications to
 * 1-norm pseudospectra, SIAM J. Matrix Anal. Appl., 21(4):1185-1201, 2000.
 */

int KLU_condest		/* return TRUE if successful, FALSE otherwise */
(
    int Ap [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    double *condest,
    klu_common *Common
)
{
    double xj, Xmax, csum, anorm, ainv_norm, est_old, est_new, abs_value ;
    Unit **Udiag ;
    Entry *Ukk, *Aentry, *X, *S ;
    int *R ;
    int nblocks, nk, block, i, j, jmax, jnew, pend, n ;
#ifndef COMPLEX
    int unchanged ;
#endif

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    Common->status = KLU_OK ;
    abs_value = 0 ;

    if (Numeric == NULL)
    {
	/* treat this as a singular matrix */
	*condest = 1 / abs_value ;
	Common->status = KLU_SINGULAR ;
	return (TRUE) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    n = Symbolic->n ;
    nblocks = Symbolic->nblocks ;
    R = Symbolic->R ;
    Udiag = (Unit **) Numeric->Udiag ;

    /* ---------------------------------------------------------------------- */
    /* check if diagonal of U has a zero on it */
    /* ---------------------------------------------------------------------- */

    for (block = 0 ; block < nblocks ; block++)
    {
	Ukk = (Entry *) Udiag [block] ;
	nk =  R [block + 1] - R [block] ;
	if (nk == 1)
	{
	    continue ; /* singleton block */
	}
	for (i = 0 ; i < nk ; i++)
	{
	    ABS (abs_value, Ukk [i]) ;
	    if (SCALAR_IS_ZERO (abs_value))
	    {
		*condest = 1 / abs_value ;
		Common->status = KLU_SINGULAR ;
		return (TRUE) ;
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* compute 1-norm (maximum column sum) of the matrix */
    /* ---------------------------------------------------------------------- */

    anorm =  0.0 ;
    Aentry = (Entry *) Ax ;
    for (i = 0 ; i < n ; i++)
    {
	pend = Ap [i + 1] ;
	csum = 0.0 ;
	for (j = Ap [i] ; j < pend ; j++)
	{
	    ABS (abs_value, Aentry [j]) ;
	    csum += abs_value ;
	}
	if (csum > anorm)
	{
	    anorm = csum ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* compute estimate of 1-norm of inv (A) */
    /* ---------------------------------------------------------------------- */

    /* get workspace */
    X = Numeric->Xwork ;	    /* size n space used in klu_solve, tsolve */
    X += n ;			    /* X is size n */
    S = X + n ;			    /* S is size n */

    for (i = 0 ; i < n ; i++)
    {
	CLEAR (S [i]) ;
	CLEAR (X [i]) ;
	REAL (X [i]) = 1.0 / ((double) n) ;
    }
    jmax = 0 ;

    ainv_norm = 0.0 ;
    for (i = 0 ; i < 5 ; i++)
    {
	if (i > 0)
	{
	    /* X [jmax] is the largest entry in X */
	    for (j = 0 ; j < n ; j++)
	    {
		/* X [j] = 0 ;*/
		CLEAR (X [j]) ;
	    }
	    REAL (X [jmax]) = 1 ;
	}

	KLU_solve (Symbolic, Numeric, n, 1, (double *) X, Common) ;
	est_old = ainv_norm ;
	ainv_norm = 0.0 ;

	for (j = 0 ; j < n ; j++)
	{
	    /* ainv_norm += ABS (X [j]) ;*/
	    ABS (abs_value, X [j]) ;
	    ainv_norm += abs_value ;
	}

#ifndef COMPLEX
	unchanged = TRUE ;

	for (j = 0 ; j < n ; j++)
	{
	    double s = (X [j] >= 0) ? 1 : -1 ;
	    if (s != (int) REAL (S [j]))
	    {
		S [j] = s ;
		unchanged = FALSE ;
	    }
	}

	if (i > 0 && (ainv_norm <= est_old || unchanged))
	{
	    break ;
	}
#else
        for (j = 0 ; j < n ; j++)
	{
	    if (IS_NONZERO (X [j]))
	    {
		ABS (abs_value, X [j]) ;
		SCALE_DIV_ASSIGN (S [j], X [j], abs_value) ;
	    }
	    else
	    {
		CLEAR (S [j]) ;
		REAL (S [j]) = 1 ;
	    }
	}

	if (i > 0 && ainv_norm <= est_old)
        {
            break ;
        }
#endif

	for (j = 0 ; j < n ; j++)
	{
	    X [j] = S [j] ;
	}

#ifndef COMPLEX
	/* do a transpose solve */
	KLU_tsolve (Symbolic, Numeric, n, 1, X, Common) ;
#else
	/* do a conjugate transpose solve */
	KLU_tsolve (Symbolic, Numeric, n, 1, (double *) X, 1, Common) ;
#endif

	/* jnew = the position of the largest entry in X */
	jnew = 0 ;
	/* Xmax = ABS (X [0]) ;*/
	ABS (Xmax, X [0]) ;
	for (j = 1 ; j < n ; j++)
	{
	    /* xj = ABS (X [j]) ;*/
	    ABS (xj, X [j]) ;
	    if (xj > Xmax)
	    {
		Xmax = xj ;
		jnew = j ;
	    }
	}
	if (i > 0 && jnew == jmax)
	{
	    /* the position of the largest entry did not change
	     * from the previous iteration */
	    break ;
	}
	jmax = jnew ;
    }

    for (j = 0 ; j < n ; j++)
    {
	CLEAR (X [j]) ;
	if (j % 2)
	{
	    REAL (X [j]) = 1 + ((double) j) / ((double) (n-1)) ;
	}
	else
	{
	    REAL (X [j]) = -1 - ((double) j) / ((double) (n-1)) ;
	}
    }

    KLU_solve (Symbolic, Numeric, n, 1, (double *) X, Common) ;

    est_new = 0.0 ;
    for (j = 0 ; j < n ; j++)
    {
	/* est_new += ABS (X [j]) ;*/
	ABS (abs_value, X [j]) ;
	est_new += abs_value ;
    }
    est_new = 2 * est_new / (3 * n) ;
    if (est_new > ainv_norm)
    {
	ainv_norm = est_new ;
    }

    /* ---------------------------------------------------------------------- */
    /* compute estimate of condition number */
    /* ---------------------------------------------------------------------- */

    *condest = ainv_norm * anorm ;
    return (TRUE) ;
}


/* ========================================================================== */
/* === klu_flops ============================================================ */
/* ========================================================================== */

/* Compute the flop count for the LU factorization (in Common->flops) */

int KLU_flops		/* return TRUE if successful, FALSE otherwise */
(
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    klu_common *Common
)
{
    double flops = 0 ;
    int **Ubip, **Lblen, **Ublen ;
    int *R, *Ui, *Uip, *Llen, *Ulen ;
    Unit **LUbx ;
    Unit *LU ;
    int k, ulen, p, n, nk, block, nblocks ;

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    Common->status = KLU_OK ;
    Common->flops = EMPTY ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Symbolic object */
    /* ---------------------------------------------------------------------- */

    n = Symbolic->n ;
    R = Symbolic->R ;
    nblocks = Symbolic->nblocks ;

    /* ---------------------------------------------------------------------- */
    /* get the contents of the Numeric object */
    /* ---------------------------------------------------------------------- */

    Lblen = Numeric->Lblen ;
    Ubip = Numeric->Ubip ;
    Ublen = Numeric->Ublen ;
    LUbx = (Unit **) Numeric->LUbx ;

    /* ---------------------------------------------------------------------- */
    /* compute the flop count */
    /* ---------------------------------------------------------------------- */

    for (block = 0 ; block < nblocks ; block++)
    {
	nk = R [block+1] - R [block] ;
	if (nk > 1)
	{
	    Llen = Lblen [block] ;
	    Uip = Ubip [block] ;
	    Ulen = Ublen [block] ;
	    LU = LUbx [block] ;
	    for (k = 0 ; k < nk ; k++)
	    {
		/* compute kth column of U, and update kth column of A */
		GET_I_POINTER (LU, Uip, Ui, k) ;
		ulen = Ulen [k] ;
		for (p = 0 ; p < ulen ; p++)
		{
		    flops += 2 * Llen [Ui [p]] ;
		}
		/* gather and divide by pivot to get kth column of L */
		flops += Llen [k] ;
	    }
	}
    }
    Common->flops = flops ;
    return (TRUE) ;
}


/* ========================================================================== */
/* === klu_rcond ============================================================ */
/* ========================================================================== */

/* Compute a really cheap estimate of the reciprocal of the condition number,
 * condition number, min(abs(diag(U))) / max(abs(diag(U))).  If U has a zero
 * pivot, or a NaN pivot, rcond will be zero.  Takes O(n) time.
 */   

int KLU_rcond			/* return TRUE if successful, FALSE otherwise */
(
    klu_symbolic *Symbolic,	/* input, not modified */
    klu_numeric *Numeric,	/* input, not modified */
    double *rcond,		/* output (pointer to a scalar) */
    klu_common *Common
)
{
    double ukk, umin, umax ;
    Entry *Ukk ;
    int block, k1, k2, nk, j ;

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    Common->status = KLU_OK ;

    if (Numeric == NULL)
    {
	*rcond = 0 ;
	Common->status = KLU_SINGULAR ;
	return (TRUE) ;
    }

    for (block = 0 ; block < Symbolic->nblocks ; block++)
    {
	k1 = Symbolic->R [block] ;
	k2 = Symbolic->R [block+1] ;
	nk = k2 - k1 ;
	if (nk == 1)
	{
	    /* get the singleton */
	    Ukk = ((Entry *) Numeric->Singleton) + block ;
	}
	else
	{
	    /* get the diagonal of U for a non-singleton block */
	    Ukk = (Entry *) Numeric->Udiag [block] ;
	}
	for (j = 0 ; j < nk ; j++)
	{
	    /* get the magnitude of the pivot */
	    ABS (ukk, Ukk [j]) ;
	    if (SCALAR_IS_NAN (ukk) || SCALAR_IS_ZERO (ukk))
	    {
		/* if NaN, or zero, the rcond is zero */
		*rcond = 0 ;
		Common->status = KLU_SINGULAR ;
		return (TRUE) ;
	    }
	    if (block == 0 && j == 0)
	    {
		/* first pivot entry in the first block */
		umin = ukk ;
		umax = ukk ;
	    }
	    else
	    {
		/* subsequent pivots */
		umin = MIN (umin, ukk) ;
		umax = MAX (umax, ukk) ;
	    }
	}
    }

    *rcond = umin / umax ;
    if (SCALAR_IS_NAN (*rcond) || SCALAR_IS_ZERO (*rcond))
    {
	/* this can occur if umin or umax are Inf or NaN */
	*rcond = 0 ;
	Common->status = KLU_SINGULAR ;
    }
    return (TRUE) ;
}
