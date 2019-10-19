/* ========================================================================== */
/* === Demo/cholmod_demo ==================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Demo Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * The CHOLMOD/Demo Module is licensed under Version 2.0 of the GNU
 * General Public License.  See gpl.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

/* Read in a matrix from a file, and use CHOLMOD to solve Ax=b if A is
 * symmetric, or (AA'+beta*I)x=b otherwise.  The file format is a simple
 * triplet format, compatible with most files in the Matrix Market format.
 * See cholmod_read.c for more details.  The readhb.f program reads a
 * Harwell/Boeing matrix (excluding element-types) and converts it into the
 * form needed by this program.  reade.f reads a matrix in Harwell/Boeing
 * finite-element form.
 *
 * Usage:
 *	cholmod_demo matrixfile
 *	cholmod_demo < matrixfile
 *
 * The matrix is assumed to be positive definite (a supernodal LL' or simplicial
 * LDL' factorization is used).
 *
 * Requires the Core, Cholesky, MatrixOps, and Check Modules.
 * Optionally uses the Partition and Supernodal Modules.
 * Does not use the Modify Module.
 *
 * See cholmod_simple.c for a simpler demo program.
 */

#include "cholmod_demo.h"

/* ff is a global variable so that it can be closed by my_handler */
FILE *ff ;

/* halt if an error occurs */
static void my_handler (int status, const char *file, int line,
    const char *message)
{
    printf ("cholmod error: file: %s line: %d status: %d: %s\n",
	    file, line, status, message) ;
    if (status < 0)
    {
	if (ff != NULL) fclose (ff) ;
	exit (0) ;
    }
}

int main (int argc, char **argv)
{
    double resid, t, ta, tf, ts, tot, bnorm, xnorm, anorm, rnorm, fl, anz, 
	axbnorm, rnorm2, resid2 ;
    FILE *f ;
    cholmod_sparse *A ;
    cholmod_dense *X, *B, *W, *R ;
    double one [2], zero [2], minusone [2], beta [2], xlnz ;
    cholmod_common Common, *cm ;
    cholmod_factor *L ;
    double *Bx, *Rx, *Xx ;
    int i, n, isize, xsize, ordering, xtype, s, ss, lnz ;

    /* ---------------------------------------------------------------------- */
    /* get the file containing the input matrix */
    /* ---------------------------------------------------------------------- */

    ff = NULL ;
    if (argc > 1)
    {
	if ((f = fopen (argv [1], "r")) == NULL)
	{
	    my_handler (CHOLMOD_INVALID, __FILE__, __LINE__,
		    "unable to open file") ;
	}
	ff = f ;
    }
    else
    {
	f = stdin ;
    }

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_start (cm) ;

    /* use default parameter settings, except for the error handler.  This
     * demo program terminates if an error occurs (out of memory, not positive
     * definite, ...).  It makes the demo program simpler (no need to check
     * CHOLMOD error conditions).  This non-default parameter setting has no
     * effect on performance. */
    cm->error_handler = my_handler ;

    /* Note that CHOLMOD will do a supernodal LL' or a simplicial LDL' by
     * default, automatically selecting the latter if flop/nnz(L) < 40. */

    /* ---------------------------------------------------------------------- */
    /* create basic scalars */
    /* ---------------------------------------------------------------------- */

    zero [0] = 0 ;
    zero [1] = 0 ;
    one [0] = 1 ;
    one [1] = 0 ;
    minusone [0] = -1 ;
    minusone [1] = 0 ;
    beta [0] = 1e-6 ;
    beta [1] = 0 ;

    /* ---------------------------------------------------------------------- */
    /* read in a matrix */
    /* ---------------------------------------------------------------------- */

    printf ("\n---------------------------------- cholmod_demo:\n") ;
    A = cholmod_read_sparse (f, cm) ;
    if (ff != NULL) fclose (ff) ;
    anorm = cholmod_norm_sparse (A, 0, cm) ;
    xtype = A->xtype ;
    printf ("norm (A,inf) = %g\n", anorm) ;
    printf ("norm (A,1)   = %g\n", cholmod_norm_sparse (A, 1, cm)) ;
    cholmod_print_sparse (A, "A", cm) ;

    if (A->nrow > A->ncol)
    {
	/* Transpose A so that A'A+beta*I will be factorized instead */
	cholmod_sparse *C = cholmod_transpose (A, 2, cm) ;
	cholmod_free_sparse (&A, cm) ;
	A = C ;
	printf ("transposing input matrix\n") ;
    }

    /* ---------------------------------------------------------------------- */
    /* create an arbitrary right-hand-side */
    /* ---------------------------------------------------------------------- */

    n = A->nrow ;
    B = cholmod_zeros (n, 1, xtype, cm) ;
    Bx = B->x ;

#if GHS
    {
	/* b = A*ones(n,1), used by Gould, Hu, and Scott in their experiments */
	cholmod_dense *X0 ;
	X0 = cholmod_ones (A->ncol, 1, xtype, cm) ;
	cholmod_sdmult (A, 0, one, zero, X0, B, cm) ;
	cholmod_free_dense (&X0, cm) ;
    }
#else
    if (xtype == CHOLMOD_REAL)
    {
	/* real case */
	for (i = 0 ; i < n ; i++)
	{
	    double x = n ;
	    Bx [i] = 1 + i / x ;
	}
    }
    else
    {
	/* complex case */
	for (i = 0 ; i < n ; i++)
	{
	    double x = n ;
	    Bx [2*i  ] = 1 + i / x ;		/* real part of B(i) */
	    Bx [2*i+1] = (x/2 - i) / (3*x) ;	/* imag part of B(i) */
	}
    }
#endif

    cholmod_print_dense (B, "B", cm) ;
    bnorm = cholmod_norm_dense (B, 0, cm) ;	/* max norm */
    printf ("bnorm %g\n", bnorm) ;

    /* ---------------------------------------------------------------------- */
    /* analyze, factorize, and solve */
    /* ---------------------------------------------------------------------- */

    t = CPUTIME ;
    L = cholmod_analyze (A, cm) ;
    ta = CPUTIME - t ;
    ta = MAX (ta, 0) ;

    printf ("Analyze: flop %g lnz %g\n", cm->fl, cm->lnz) ;

    if (A->stype == 0)
    {
	printf ("Factorizing A*A'+beta*I\n") ;
	t = CPUTIME ;
	cholmod_factorize_p (A, beta, NULL, 0, L, cm) ;
	tf = CPUTIME - t ;
	tf = MAX (tf, 0) ;
    }
    else
    {
	printf ("Factorizing A\n") ;
	t = CPUTIME ;
	cholmod_factorize (A, L, cm) ;
	tf = CPUTIME - t ;
	tf = MAX (tf, 0) ;
    }

    t = CPUTIME ;

    X = cholmod_solve (CHOLMOD_A, L, B, cm) ;
    ts = CPUTIME - t ;
    ts = MAX (ts, 0) ;
    tot = ta + tf + ts ;

    /* ---------------------------------------------------------------------- */
    /* compute the residual */
    /* ---------------------------------------------------------------------- */

    if (A->stype == 0)
    {
	/* (AA'+beta*I)x=b is the linear system that was solved */
	/* W = A'*X */
	W = cholmod_allocate_dense (A->ncol, 1, A->ncol, xtype, cm) ;
	cholmod_sdmult (A, 2, one, zero, X, W, cm) ;
	/* R = B - beta*X */
	R = cholmod_zeros (n, 1, xtype, cm) ;
	Rx = R->x ;
	Xx = X->x ;
	if (xtype == CHOLMOD_REAL)
	{
	    for (i = 0 ; i < n ; i++)
	    {
		Rx [i] = Bx [i] - beta [0] * Xx [i] ;
	    }
	}
	else
	{
	    /* complex case */
	    for (i = 0 ; i < n ; i++)
	    {
		Rx [2*i  ] = Bx [2*i  ] - beta [0] * Xx [2*i  ] ;
		Rx [2*i+1] = Bx [2*i+1] - beta [0] * Xx [2*i+1] ;
	    }
	}
	/* R = A*W - R */
	cholmod_sdmult (A, 0, one, minusone, W, R, cm) ;
	cholmod_free_dense (&W, cm) ;
    }
    else
    {
	/* Ax=b was factorized and solved, R = B-A*X */
	R = cholmod_copy_dense (B, cm) ;
	cholmod_sdmult (A, 0, minusone, one, X, R, cm) ;
    }
    rnorm = cholmod_norm_dense (R, 0, cm) ;	    /* max abs. entry */
    xnorm = cholmod_norm_dense (X, 0, cm) ;	    /* max abs. entry */

    axbnorm = (anorm * xnorm + bnorm + ((n == 0) ? 1 : 0)) ;
    resid = rnorm / axbnorm ;

    /* ---------------------------------------------------------------------- */
    /* iterative refinement (real symmetric case only) */
    /* ---------------------------------------------------------------------- */

    resid2 = -1 ;
    if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
    {
	cholmod_dense *R2 ;

	/* R2 = A\(B-A*X) */
	R2 = cholmod_solve (CHOLMOD_A, L, R, cm) ;
	/* compute X = X + A\(B-A*X) */
	Xx = X->x ;
	Rx = R2->x ;
	for (i = 0 ; i < n ; i++)
	{
	    Xx [i] = Xx [i] + Rx [i] ;
	}
	cholmod_free_dense (&R2, cm) ;
	cholmod_free_dense (&R, cm) ;

	/* compute the new residual, R = B-A*X */
	R = cholmod_copy_dense (B, cm) ;
	cholmod_sdmult (A, 0, minusone, one, X, R, cm) ;
	rnorm2 = cholmod_norm_dense (R, 0, cm) ;
	resid2 = rnorm2 / axbnorm ;
    }

    cholmod_free_dense (&R, cm) ;

    /* ---------------------------------------------------------------------- */
    /* print results */
    /* ---------------------------------------------------------------------- */

    cholmod_print_factor (L, "L", cm) ;

    /* determine the # of integers's and reals's in L.  See cholmod_free */
    if (L->is_super)
    {
	s = L->nsuper + 1 ;
	xsize = L->xsize ;
	ss = L->ssize ;
	isize =
	    n	/* L->Perm */
	    + n	/* L->ColCount, nz in each column of 'pure' L */
	    + s	/* L->pi, column pointers for L->s */
	    + s	/* L->px, column pointers for L->x */
	    + s	/* L->super, starting column index of each supernode */
	    + ss ;	/* L->s, the pattern of the supernodes */
    }
    else
    {
	/* this space can increase if you change parameters to their non-
	 * default values (cm->final_pack, for example). */
	lnz = L->nzmax ;
	xsize = lnz ;
	isize =
	    n	/* L->Perm */
	    + n	/* L->ColCount, nz in each column of 'pure' L */
	    + n+1	/* L->p, column pointers */
	    + lnz	/* L->i, integer row indices */
	    + n	/* L->nz, nz in each column of L */
	    + n+2	/* L->next, link list */
	    + n+2 ;	/* L->prev, link list */
    }

    anz = cm->anz ;
    for (i = 0 ; i < CHOLMOD_MAXMETHODS ; i++)
    {
	fl = cm->method [i].fl ;
	xlnz = cm->method [i].lnz ;
	cm->method [i].fl = -1 ;
	cm->method [i].lnz = -1 ;
	ordering = cm->method [i].ordering ;
	if (fl >= 0)
	{
	    printf ("Ordering: ") ;
	    if (ordering == CHOLMOD_POSTORDERED) printf ("postordered ") ;
	    if (ordering == CHOLMOD_NATURAL)     printf ("natural ") ;
	    if (ordering == CHOLMOD_GIVEN)	     printf ("user    ") ;
	    if (ordering == CHOLMOD_AMD)	     printf ("AMD     ") ;
	    if (ordering == CHOLMOD_METIS)	     printf ("METIS   ") ;
	    if (ordering == CHOLMOD_NESDIS)      printf ("NESDIS  ") ;
	    if (xlnz > 0)
	    {
		printf ("fl/lnz %10.1f", fl / xlnz) ;
	    }
	    if (anz > 0)
	    {
		printf ("  lnz/anz %10.1f", xlnz / anz) ;
	    }
	    printf ("\n") ;
	}
    }

    printf ("ints in L: %d, doubles in L: %d\n", isize, xsize) ;
    printf ("factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
	    cm->fl, cm->lnz) ;
    if (A->stype == 0)
    {
	printf ("nnz(A):    %15.0f\n", cm->anz) ;
    }
    else
    {
	printf ("nnz(A*A'): %15.0f\n", cm->anz) ;
    }
    if (cm->lnz > 0)
    {
	printf ("flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz) ;
    }
    if (anz > 0)
    {
	printf ("nnz(L) / nnz(A): %8.1f\n", cm->lnz / cm->anz) ;
    }
    printf ("analyze cputime:  %12.4f\n", ta) ;
    printf ("factor  cputime:   %12.4f mflop: %8.1f\n", tf,
	(tf == 0) ? 0 : (1e-6*cm->fl / tf)) ;
    printf ("solve   cputime:   %12.4f mflop: %8.1f\n", ts,
	(ts == 0) ? 0 : (1e-6*4*cm->lnz / ts)) ;
    printf ("overall cputime:   %12.4f mflop: %8.1f\n", 
	    tot, (tot == 0) ? 0 : (1e-6 * (cm->fl + 4 * cm->lnz) / tot)) ;
    printf ("peak memory usage: %12.0f (MB)\n",
	    (double) (cm->memory_usage) / 1048576.) ;
    printf ("residual %8.1e (|Ax-b|/(|A||x|+|b|))\n", resid) ;
    if (resid2 >= 0)
    {
	printf ("residual %8.1e (|Ax-b|/(|A||x|+|b|))"
		" after iterative refinement\n", resid2) ;
    }
    printf ("rcond    %8.1e\n\n", cholmod_rcond (L, cm)) ;
    cholmod_free_factor (&L, cm) ;
    cholmod_free_dense (&X, cm) ;

    /* ---------------------------------------------------------------------- */
    /* free matrices and finish CHOLMOD */
    /* ---------------------------------------------------------------------- */

    cholmod_free_sparse (&A, cm) ;
    cholmod_free_dense (&B, cm) ;
    cholmod_finish (cm) ;
    return (0) ;
}
