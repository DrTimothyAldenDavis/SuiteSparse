/* ========================================================================== */
/* === klu_sort ============================================================= */
/* ========================================================================== */

/* sorts the columns of L and U so that the row indices appear in strictly
 * increasing order.
 */

#include "klu_internal.h"

/* ========================================================================== */
/* === sort ================================================================= */
/* ========================================================================== */

/* Sort L or U using a double-transpose */

static void sort (int n, int *Xip, int *Xlen, Unit *LU, int *Tp, int *Tj,
    Entry *Tx, int *W)
{
    int *Xi ;
    Entry *Xx ;
    int p, i, j, len, nz, tp, xlen, pend ;

    ASSERT (KLU_valid_LU (n, FALSE, Xip, Xlen, LU)) ;

    /* count the number of entries in each row of L or U */ 
    for (i = 0 ; i < n ; i++)
    {
	W [i] = 0 ;
    }
    for (j = 0 ; j < n ; j++)
    {
	GET_POINTER (LU, Xip, Xlen, Xi, Xx, j, len) ;
	for (p = 0 ; p < len ; p++)
	{
	    W [Xi [p]]++ ;
	}
    }

    /* construct the row pointers for T */
    nz = 0 ;
    for (i = 0 ; i < n ; i++)
    {
	Tp [i] = nz ;
	nz += W [i] ;
    }
    Tp [n] = nz ;
    for (i = 0 ; i < n ; i++)
    {
	W [i] = Tp [i] ;
    }

    /* transpose the matrix into Tp, Ti, Tx */
    for (j = 0 ; j < n ; j++)
    {
	GET_POINTER (LU, Xip, Xlen, Xi, Xx, j, len) ;
	for (p = 0 ; p < len ; p++)
	{
	    tp = W [Xi [p]]++ ;
	    Tj [tp] = j ;
	    Tx [tp] = Xx [p] ;
	}
    }

    /* transpose the matrix back into Xip, Xlen, Xi, Xx */
    for (j = 0 ; j < n ; j++)
    {
	W [j] = 0 ;
    }
    for (i = 0 ; i < n ; i++)
    {
	pend = Tp [i+1] ;
	for (p = Tp [i] ; p < pend ; p++)
	{
	    j = Tj [p] ;
	    GET_POINTER (LU, Xip, Xlen, Xi, Xx, j, len) ;
	    xlen = W [j]++ ;
	    Xi [xlen] = i ;
	    Xx [xlen] = Tx [p] ;
	}
    }

    ASSERT (KLU_valid_LU (n, FALSE, Xip, Xlen, LU)) ;
}


/* ========================================================================== */
/* === klu_sort ============================================================= */
/* ========================================================================== */

int KLU_sort
(
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    klu_common *Common
)
{
    int *R, *W, *Tp, *Ti ;
    Entry *Tx ;
    int **Lbip, **Ubip, **Lblen, **Ublen ;
    Unit **LUbx ;
    int n, nk, nz, block, nblocks, maxblock ;
    size_t m1 ;

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    Common->status = KLU_OK ;

    n = Symbolic->n ;
    R = Symbolic->R ;
    nblocks = Symbolic->nblocks ;
    maxblock = Symbolic->maxblock ;

    Lbip = Numeric->Lbip ;
    Lblen = Numeric->Lblen ;
    Ubip = Numeric->Ubip ;
    Ublen = Numeric->Ublen ;
    LUbx = (Unit **) Numeric->LUbx ;

    m1 = ((size_t) maxblock) + 1 ;

    /* allocate workspace */
    nz = MAX (Numeric->max_lnz_block, Numeric->max_unz_block) ;
    W  = klu_malloc (maxblock, sizeof (int), Common) ;
    Tp = klu_malloc (m1, sizeof (int), Common) ;
    Ti = klu_malloc (nz, sizeof (int), Common) ;
    Tx = klu_malloc (nz, sizeof (Entry), Common) ;

    PRINTF (("\n======================= Start sort:\n")) ;

    if (Common->status == KLU_OK)
    {
	/* sort each block of L and U */
	for (block = 0 ; block < nblocks ; block++)
	{
	    nk = R [block+1] - R [block] ;
	    if (nk > 1)
	    {
		PRINTF (("\n-------------------block: %d nk %d\n", block, nk)) ;
		sort (nk, Lbip [block], Lblen [block], LUbx [block],
		    Tp, Ti, Tx, W) ;
		sort (nk, Ubip [block], Ublen [block], LUbx [block],
		    Tp, Ti, Tx, W) ;
	    }
	}
    }

    PRINTF (("\n======================= sort done.\n")) ;

    /* free workspace */
    klu_free (W, Common) ;
    klu_free (Tp, Common) ;
    klu_free (Ti, Common) ;
    klu_free (Tx, Common) ;
    return (Common->status == KLU_OK) ;
}
