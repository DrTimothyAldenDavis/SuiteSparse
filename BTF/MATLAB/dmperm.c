#include "btf.h"
#include "btf_internal.h"
#include "mex.h"

#define CS_MAX(a,b) (((a) > (b)) ? (a) : (b))

/* Given A and A', find RowMatch, ColMatch, and coarse dmperm */
void dmperm (int nrow, int ncol,
	int *Ap,	/* size ncol+1 */
	int *Ai,	/* size nz = Ap [ncol] */
	int *ATp,	/* size nrow+1 */
	int *ATi,	/* size nz */
	int *P,		/* size nrow */
	int *Q,		/* size ncol */
	int *cp,	/* size 5 */
	int *rp		/* size 5 */
	)
{
    int *w, *RowMatch, *ColMatch, *RowFlag, *ColFlag, *queue ;
    int nfound, n, i, j, k, head, tail, i2, j2, kc, kr, p ;

    RowMatch = mxMalloc (nrow * sizeof (int)) ;
    w = mxMalloc (5*ncol * sizeof (int)) ;

    /* find RowMatch */
    nfound = maxtrans (nrow, ncol, Ap, Ai, RowMatch, w) ;

/*
    for (i = 0 ; i < nrow ; i++)
    {
	j = RowMatch [i] ;
	printf ("row i %d matched with col %d (unflipped) %d\n", i, j,
		MAXTRANS_UNFLIP (j)) ;
    }
*/

    mxFree (w) ;

    n = CS_MAX (nrow, ncol) ;
    ColMatch = mxMalloc (ncol * sizeof (int)) ;
    queue    = mxMalloc (n    * sizeof (int)) ;
    RowFlag  = mxMalloc (nrow * sizeof (int)) ;
    ColFlag  = mxMalloc (ncol * sizeof (int)) ;

    /* find ColMatch from RowMatch */
    for (j = 0 ; j < ncol ; j++) ColMatch [j] = EMPTY ;
    for (i = 0 ; i < nrow ; i++)
    {
	j = RowMatch [i] ;
	if (j >= 0)
	{
	    /* printf ("row i %d matched with col %d\n", i, j) ; */
	    /* mxAssert (j < ncol, "") ; */
	    ColMatch [j] = i ;
	}
    }

    for (j = 0 ; j < ncol ; j++) ColFlag [j] = -1 ;
    for (i = 0 ; i < nrow ; i++) RowFlag [i] = -1 ;

    /* printf ("C0:\n") ; */

    /* find all unmatched columns and place them in the queue */
    head = 0 ;
    tail = 0 ;
    for (j = 0 ; j < ncol ; j++)
    {
	if (ColMatch [j] < 0)
	{
	    ColFlag [j] = 0 ;	    /* j in set C0 */
	    queue [tail++] = j ;
	    /* printf ("added %d to queue\n", j) ; */
	}
    }
    /* while queue is not empty */
    while (head < tail)
    {
	/* get the head of the queue */
	j = queue [head++] ;
	/* printf ("got %d from queue\n", j) ; */
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    /* printf ("   row %d \n", i) ; */
	    if (RowFlag [i] >= 0) continue ;
	    RowFlag [i] = 1 ;	    /* i in set R1 */
	    j2 = RowMatch [i] ;
	    /* printf ("		j2 %d\n", j2) ; */
	    if (ColFlag [j2] >= 0) continue ;
	    ColFlag [j2] = 1 ;	    /* j2 in set C1; add to queue */
	    queue [tail++] = j2 ;
	    /* printf ("			    added to queue\n") ; */
	}
    }

    /* printf ("R0:\n") ; */

    /* find all unmatched rows and place them in the queue */
    head = 0 ;
    tail = 0 ;
    for (i = 0 ; i < nrow ; i++)
    {
	if (RowMatch [i] < 0)
	{
	    RowFlag [i] = 0 ;	    /* i in set R0 */
	    queue [tail++] = i ;
	}
    }
    /* while queue is not empty */
    while (head < tail)
    {
	/* get the head of the queue */
	i = queue [head++] ;
	/* printf ("got %d from queue\n", i) ; */
	for (p = ATp [i] ; p < ATp [i+1] ; p++)
	{
	    j = ATi [p] ;
	    /* printf ("   col %d \n", j) ; */
	    if (ColFlag [j] >= 0) continue ;
	    ColFlag [j] = 3 ;	    /* j in set C3 */
	    i2 = ColMatch [j] ;
	    /* printf ("		i2 %d\n", i2) ; */
	    if (RowFlag [i2] >= 0) continue ;
	    RowFlag [i2] = 3 ;	    /* i2 in set R3; add to queue */
	    queue [tail++] = i2 ;
	    /* printf ("			    added to queue\n") ; */
	}
    }

    /* printf ("wrapup:\n") ; */

    cp [0] = 0 ;
    rp [0] = 0 ;

    /* collect the sets into P and Q */
    kc = 0 ;
    kr = 0 ;
    for (j = 0 ; j < ncol ; j++)    /* set C0 */
    {
	if (ColFlag [j] == 0) Q [kc++] = j ;
    }

    cp [1] = kc ;

    for (i = 0 ; i < nrow ; i++)    /* set R1 and C1 */
    {
	if (RowFlag [i] == 1)
	{
	    j = RowMatch [i] ;	    mxAssert (ColFlag [j] == 1, "") ;
	    P [kr++] = i ;
	    Q [kc++] = j ;
	}
    }

    cp [2] = kc ;
    rp [1] = kr ;

    for (i = 0 ; i < nrow ; i++)    /* set R2 and C2 */
    {
	if (RowFlag [i] == -1)
	{
	    j = RowMatch [i] ;	    mxAssert (ColFlag [j] == -1, "") ;
	    P [kr++] = i ;
	    Q [kc++] = j ;
	}
    }

    cp [3] = kc ;
    rp [2] = kr ;

    for (i = 0 ; i < nrow ; i++)    /* set R3 and C3 */
    {
	if (RowFlag [i] == 3)
	{
	    j = RowMatch [i] ;	    mxAssert (ColFlag [j] == 3, "") ;
	    P [kr++] = i ;
	    Q [kc++] = j ;
	}
    }

    cp [4] = kc ;
    rp [3] = kr ;

    for (i = 0 ; i < nrow ; i++)    /* set R0 */
    {
	if (RowFlag [i] == 0) P [kr++] = i ;
    }

    rp [4] = kr ;

    mxAssert (kc == ncol && kr == nrow, "") ;

    mxFree (RowMatch) ;
    mxFree (ColMatch) ;
    mxFree (RowFlag) ;
    mxFree (ColFlag) ;
    mxFree (queue) ;

/*
    printf ("cp %d %d %d %d %d\n", cp [0], cp [1], cp [2], cp [3], cp [4]);
    printf ("rp %d %d %d %d %d\n", rp [0], rp [1], rp [2], rp [3], rp [4]);
*/
}
