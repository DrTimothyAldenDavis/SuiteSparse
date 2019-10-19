/* ========================================================================== */
/* === klu_extract ========================================================== */
/* ========================================================================== */

/* Extract KLU factorization into conventional compressed-column matrices.
 * If any output array is NULL, that part of the LU factorization is not
 * extracted (this is not an error condition).
 *
 * nnz(L) = Numeric->lnz, nnz(U) = Numeric->unz, and nnz(F) = Numeric->Offp [n]
 */

#include "klu_internal.h"

int KLU_extract	    /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* L */
    int *Lp,	    /* size n+1 */
    int *Li,	    /* size nnz(L) */
    double *Lx,	    /* size nnz(L) */
#ifdef COMPLEX
    double *Lz,	    /* size nnz(L) for the complex case, ignored if real */
#endif

    /* U */
    int *Up,	    /* size n+1 */
    int *Ui,	    /* size nnz(U) */
    double *Ux,	    /* size nnz(U) */
#ifdef COMPLEX
    double *Uz,	    /* size nnz(U) for the complex case, ignored if real */
#endif

    /* F */
    int *Fp,	    /* size n+1 */
    int *Fi,	    /* size nnz(F) */
    double *Fx,	    /* size nnz(F) */
#ifdef COMPLEX
    double *Fz,	    /* size nnz(F) for the complex case, ignored if real */
#endif

    /* P, row permutation */
    int *P,	    /* size n */

    /* Q, column permutation */
    int *Q,	    /* size n */

    /* Rs, scale factors */
    double *Rs,	    /* size n */

    /* R, block boundaries */
    int *R	    /* size nblocks+1 */
)
{
    int *Lip, *Llen, *Uip, *Ulen, *Li2, *Ui2 ;
    Unit *LU ;
    Entry *Lx2, *Ux2, *Ud ;
    int i, k, block, nblocks, n, nz, k1, k2, nk, len, kk, p ;

    if (Symbolic == NULL || Numeric == NULL)
    {
	return (FALSE) ;
    }

    n = Symbolic->n ;
    nblocks = Symbolic->nblocks ;

    /* ---------------------------------------------------------------------- */
    /* extract scale factors */
    /* ---------------------------------------------------------------------- */

    if (Rs != NULL)
    {
	if (Numeric->Rs != NULL)
	{
	    for (i = 0 ; i < n ; i++)
	    {
		Rs [i] = Numeric->Rs [i] ;
	    }
	}
	else
	{
	    /* no scaling */
	    for (i = 0 ; i < n ; i++)
	    {
		Rs [i] = 1 ;
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* extract block boundaries */
    /* ---------------------------------------------------------------------- */

    if (R != NULL)
    {
	for (block = 0 ; block <= nblocks ; block++)
	{
	    R [block] = Symbolic->R [block] ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* extract final row permutation */
    /* ---------------------------------------------------------------------- */

    if (P != NULL)
    {
	for (k = 0 ; k < n ; k++)
	{
	    P [k] = Numeric->Pnum [k] ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* extract column permutation */
    /* ---------------------------------------------------------------------- */

    if (Q != NULL)
    {
	for (k = 0 ; k < n ; k++)
	{
	    Q [k] = Symbolic->Q [k] ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* extract each block of L */
    /* ---------------------------------------------------------------------- */

    if (Lp != NULL && Li != NULL && Lx != NULL
#ifdef COMPLEX
	&& Lz != NULL
#endif
    )
    {
	nz = 0 ;
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = Symbolic->R [block] ;
	    k2 = Symbolic->R [block+1] ;
	    nk = k2 - k1 ;
	    /* printf ("k1 %d k2 %d nk %d\n", k1, k2, nk) ; */
	    if (nk == 1)
	    {
		/* singleton block */
		/* printf ("L singleton %d\n", k1) ; */
		Lp [k1] = nz ;
		Li [nz] = k1 ;
		Lx [nz] = 1 ;
#ifdef COMPLEX
		Lz [nz] = 0 ;
#endif
		nz++ ;
	    }
	    else
	    {
		/* non-singleton block */
		LU = Numeric->LUbx [block] ;
		Lip = Numeric->Lbip [block] ;
		Llen = Numeric->Lblen [block] ;
		for (kk = 0 ; kk < nk ; kk++)
		{
		    /* printf ("k %d\n", k1+kk) ; */
		    Lp [k1+kk] = nz ;
		    /* add the unit diagonal entry */
		    Li [nz] = k1 + kk ;
		    Lx [nz] = 1 ;
#ifdef COMPLEX
		    Lz [nz] = 0 ;
#endif
		    nz++ ;
		    GET_POINTER (LU, Lip, Llen, Li2, Lx2, kk, len) ;
		    for (p = 0 ; p < len ; p++)
		    {
			Li [nz] = k1 + Li2 [p] ;
			Lx [nz] = REAL (Lx2 [p]) ;
#ifdef COMPLEX
			Lz [nz] = IMAG (Lx2 [p]) ;
#endif
			nz++ ;
		    }
		}
	    }
	}
	Lp [n] = nz ;
	ASSERT (nz == Numeric->lnz) ;
	/* printf ("Lnz %d %d\n", nz, Numeric->lnz) ; */
    }

    /* ---------------------------------------------------------------------- */
    /* extract each block of U */
    /* ---------------------------------------------------------------------- */

    if (Up != NULL && Ui != NULL && Ux != NULL
#ifdef COMPLEX
	&& Uz != NULL
#endif
    )
    {
	nz = 0 ;
	for (block = 0 ; block < nblocks ; block++)
	{
	    k1 = Symbolic->R [block] ;
	    k2 = Symbolic->R [block+1] ;
	    nk = k2 - k1 ;
	    if (nk == 1)
	    {
		/* singleton block */
		Up [k1] = nz ;
		Ui [nz] = k1 ;
		Ux [nz] = REAL (((Entry *) Numeric->Singleton) [block]) ;
#ifdef COMPLEX
		Uz [nz] = IMAG (((Entry *) Numeric->Singleton) [block]) ;
#endif
		nz++ ;
	    }
	    else
	    {
		/* non-singleton block */
		LU = Numeric->LUbx [block] ;
		Uip = Numeric->Ubip [block] ;
		Ulen = Numeric->Ublen [block] ;
		Ud = Numeric->Udiag [block] ;
		for (kk = 0 ; kk < nk ; kk++)
		{
		    Up [k1+kk] = nz ;
		    GET_POINTER (LU, Uip, Ulen, Ui2, Ux2, kk, len) ;
		    for (p = 0 ; p < len ; p++)
		    {
			Ui [nz] = k1 + Ui2 [p] ;
			Ux [nz] = REAL (Ux2 [p]) ;
#ifdef COMPLEX
			Uz [nz] = IMAG (Ux2 [p]) ;
#endif
			nz++ ;
		    }
		    /* add the diagonal entry */
		    Ui [nz] = k1 + kk ;
		    Ux [nz] = REAL (Ud [kk]) ;
#ifdef COMPLEX
		    Uz [nz] = IMAG (Ud [kk]) ;
#endif
		    nz++ ;
		}
	    }
	}
	Up [n] = nz ;
	ASSERT (nz == Numeric->unz) ;
	/* printf ("Unz %d %d\n", nz, Numeric->unz) ; */
    }

    /* ---------------------------------------------------------------------- */
    /* extract the off-diagonal blocks, F */
    /* ---------------------------------------------------------------------- */

    if (Fp != NULL && Fi != NULL && Fx != NULL
#ifdef COMPLEX
	&& Fz != NULL
#endif
    )
    {
	for (k = 0 ; k <= n ; k++)
	{
	    Fp [k] = Numeric->Offp [k] ;
	}
	nz = Fp [n] ;
	for (k = 0 ; k < nz ; k++)
	{
	    Fi [k] = Numeric->Offi [k] ;
	}
	for (k = 0 ; k < nz ; k++)
	{
	    Fx [k] = REAL (((Entry *) Numeric->Offx) [k]) ;
#ifdef COMPLEX
	    Fz [k] = IMAG (((Entry *) Numeric->Offx) [k]) ;
#endif
	}
    }

    return (TRUE) ;
}
