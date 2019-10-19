/* ========================================================================== */
/* === klu_cholmod ========================================================== */
/* ========================================================================== */

/* klu_cholmod: user-defined ordering function to interface KLU to CHOLMOD.
 *
 * This routine is an example of a user-provided ordering function for KLU.
 * Its return value is klu_cholmod's estimate of max (nnz(L),nnz(U)):
 *	0 if error,
 *	-1 if OK, but estimate of max (nnz(L),nnz(U)) not computed
 *	> 0 if OK and estimate computed.
 *
 * This function can be assigned to KLU's Common->user_order function pointer.
 */

#include "klu_cholmod.h"
#include "cholmod.h"
#define TRUE 1
#define FALSE 0

int klu_cholmod
(
    /* inputs */
    int n,		    /* A is n-by-n */
    int Ap [ ],		    /* column pointers */
    int Ai [ ],		    /* row indices */
    /* outputs */
    int Perm [ ],	    /* fill-reducing permutation */
    /* user-defined */
    void *user_datap	    /* this is KLU's Common->user_data, unchanged.
			     * You may use this to pass additional parameters
			     * to your ordering routine.  KLU does not access
			     * Common->user_data, except to pass it here. */
)
{
    double one [2] = {1,0}, zero [2] = {0,0} ;
    cholmod_sparse *A, *AT, *S ;
    cholmod_factor *L ;
    cholmod_common *cm, Common ;
    int *P, *user_data ;
    int k, symmetric ;

    /* start CHOLMOD */
    cm = &Common ;
    cholmod_start (cm) ;
    cm->supernodal = CHOLMOD_SIMPLICIAL ;

    /* cm->print = 5 ; */

    /* construct a CHOLMOD version of the input matrix A */
    A = cholmod_malloc (1, sizeof (cholmod_sparse), cm) ;
    if (A == NULL)
    {
	/* out of memory */
	return (0) ;
    }
    A->nrow = n ;		    /* A is n-by-n */
    A->ncol = n ;
    A->nzmax = Ap [n] ;		    /* with nzmax entries */
    A->packed = TRUE ;		    /* there is no A->nz array */
    A->stype = 0 ;		    /* A is unsymmetric */
    A->itype = CHOLMOD_INT ;
    A->xtype = CHOLMOD_PATTERN ;
    A->dtype = CHOLMOD_DOUBLE ;
    A->nz = NULL ;
    A->p = Ap ;			    /* column pointers */
    A->i = Ai ;			    /* row indices */
    A->x = NULL ;		    /* no numerical values */
    A->z = NULL ;
    A->sorted = FALSE ;		    /* columns of A are not sorted */

    /* cholmod_print_sparse (A, "A for user order", cm) ; */

    /* get the user_data */
    user_data = user_datap ;
    symmetric = (user_data == NULL) ? TRUE : (user_data [0] != 0) ;

    /* AT = pattern of A' */
    AT = cholmod_transpose (A, 0, cm) ;

    if (symmetric)
    {
	/* S = the symmetric pattern of A+A' */
	S = cholmod_add (A, AT, one, zero, FALSE, FALSE, cm) ;
	cholmod_free_sparse (&AT, cm) ;
    }
    else
    {
	/* S = A'.  CHOLMOD will order S*S', which is A'*A */
	S = AT ;
    }

    /* free the header for A, but not Ap and Ai */
    A->p = NULL ;
    A->i = NULL ;
    cholmod_free_sparse (&A, cm) ;

    if (S == NULL)
    {
	/* out of memory */
	return (0) ;
    }

    if (symmetric)
    {
	S->stype = 1 ;
    }

    /* cholmod_print_sparse (S, "S", cm) ; */

    cm->nmethods = 10 ;
    /*
    cm->nmethods = 1 ;
    cm->method [0].ordering = CHOLMOD_AMD ;
    cm->method [0].ordering = CHOLMOD_METIS ;
    cm->method [0].ordering = CHOLMOD_NESDIS ;
    */

    /* order and analyze S or S*S' */
    L = cholmod_analyze (S, cm) ;
    if (L == NULL)
    {
	return (0) ;
    }

/*
printf ("L->ordering %d\n", L->ordering) ; 
*/

    /* cholmod_print_factor (L, "L, symbolic analysis only", cm) ; */

    /* copy the permutation from L to the output */
    P = L->Perm ;
    for (k = 0 ; k < n ; k++)
    {
	Perm [k] = P [k] ;
    }

    /* cholmod_print_perm (Perm, n, n, "Final permutation", cm) ; */

    cholmod_free_sparse (&S, cm) ;
    cholmod_free_factor (&L, cm) ;
    cholmod_finish (cm) ;

    /*
    if (n > 100) cholmod_print_common ("Final statistics", cm) ;
    if (cm->malloc_count != 0) printf ("CHOLMOD memory leak!") ;
    */
    return ((int) (cm->lnz)) ;
}
