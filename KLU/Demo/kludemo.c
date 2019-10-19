/* ========================================================================== */
/* === kludemo ============================================================== */
/* ========================================================================== */

/* Demo program for KLU.  Reads in a column-form 0-based matrix in tmp/Ap,
 * tmp/Ai, and tmp/Ax, whose size and # of nonzeros are in the file tmp/Asize.
 * Then calls KLU to analyze, factor, and solve the system.
 *
 * Example:
 *
 *	kludemo
 *
 * The right-hand-side can be provided in the optional tmp/b file.  The solution
 * is written to tmp/x.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "klu.h"
#include "klu_cholmod.h"

#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define XTRUE(i,n) (1.0 + ((double) i) / ((double) n))
#define ISNAN(x) ((x) != (x))

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#define NTRIAL 1
#define NRHS 24

#include "dsecnd.h"

#include <time.h>
#define CPUTIME (((double) clock ( )) / CLOCKS_PER_SEC)

/* -------------------------------------------------------------------------- */
/* err: compute the relative error, ||x-xtrue||/||xtrue|| */
/* -------------------------------------------------------------------------- */

static double err
(
    int n,
    double x [ ]
)
{
    int i  ;
    double enorm, e, abse, absxtrue, xnorm ;
    enorm = 0 ;
    xnorm = 0 ;

    for (i = 0 ; i < n ; i++)
    {
	if (ISNAN (x [i]))
	{
	    enorm = x [i] ;
	    break ;
	}
	e = x [i] - XTRUE (i,n) ;
	abse = ABS (e) ;
	enorm = MAX (enorm, abse) ;
    }

    for (i = 0 ; i < n ; i++)
    {
	/* XTRUE is positive, but do this in case XTRUE is redefined */
	absxtrue = ABS (XTRUE (i,n)) ;
	xnorm = MAX (xnorm, absxtrue) ;
    }

    if (xnorm == 0)
    {
	xnorm = 1 ;
    }
    return (enorm / xnorm) ;
}


/* -------------------------------------------------------------------------- */
/* resid: compute the relative residual, ||Ax-b||/||b|| or ||A'x-b||/||b|| */
/* -------------------------------------------------------------------------- */

static double resid
(
    int n,
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    double x [ ],
    double r [ ],
    double b [ ],
    int transpose
)
{
    int i, j, p ;
    double rnorm, absr, absb, bnorm ;
    for (i = 0 ; i < n ; i++)
    {
	r [i] = 0 ;
    }

    if (transpose)
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		r [j] += Ax [p] * x [i] ;
	    }
	}
    }
    else
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		r [i] += Ax [p] * x [j] ;
	    }
	}
    }

    for (i = 0 ; i < n ; i++)
    {
	r [i] -= b [i] ;
    }
    rnorm = 0. ;
    bnorm = 0. ;
    for (i = 0 ; i < n ; i++)
    {
	if (ISNAN (r [i]))
	{
	    rnorm = r [i] ;
	    break ;
	}
	absr = ABS (r [i]) ;
	rnorm = MAX (rnorm, absr) ;
    }
    for (i = 0 ; i < n ; i++)
    {
	if (ISNAN (b [i]))
	{
	    bnorm = b [i] ;
	    break ;
	}
	absb = ABS (b [i]) ;
	bnorm = MAX (bnorm, absb) ;
    }
    if (bnorm == 0)
    {
	bnorm = 1 ;
    }
    return (rnorm / bnorm) ;
}


/* -------------------------------------------------------------------------- */
/* Atimesx: compute y = A*x  or A'*x, where x (i) = 1 + i/n */
/* -------------------------------------------------------------------------- */

static void Atimesx
(
    int n,
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    double y [ ],
    int transpose
)
{
    int i, j, p ;
    for (i = 0 ; i < n ; i++)
    {
	y [i] = 0 ;
    }
    if (transpose)
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		y [j] += Ax [p] * XTRUE (i,n) ;
	    }
	}
    }
    else
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		y [i] += Ax [p] * XTRUE (j,n) ;
	    }
	}
    }
}

/* -------------------------------------------------------------------------- */
/* main program */
/* -------------------------------------------------------------------------- */

int main (void)
{
    double *Ax, *b, *x, *r, *Y ;
    double stats [2], condest, growth, t, c, rcond ;
    /* atime [2], ftime [2], rtime [2], stime [2], stime2 [2], */
    int p, i, j, n, nz, *Ap, *Ai, nrow, ncol, rhs, trial, s, k, nrhs, do_sort,
	ldim, ordering, do_btf, scale, rescale, lnz, unz, halt_if_singular ;
    klu_symbolic *Symbolic ;
    klu_numeric  *Numeric ;
    klu_common Kommon, *km ;
    FILE *f ;

    printf ("\nKLU stand-alone demo:\n") ;
    km = &Kommon ;
    klu_defaults (km) ;

    /* ---------------------------------------------------------------------- */
    /* get n and nz */
    /* ---------------------------------------------------------------------- */

    printf ("File: tmp/Asize\n") ;
    f = fopen ("tmp/Asize", "r") ;
    if (!f)
    {
	printf ("Unable to open file\n") ;
	exit (1) ;
    }
    fscanf (f, "%d %d %d\n", &nrow, &ncol, &nz) ;
    fclose (f) ;
    n  = MAX (1, MAX (nrow, ncol)) ;
    nz = MAX (1, nz) ;
    ldim = n + 13 ;
    nrhs = NRHS ;

    /* ---------------------------------------------------------------------- */
    /* allocate space for the matrix A, and for the vectors b, r, and x */
    /* ---------------------------------------------------------------------- */

    Ap = (int *) malloc ((n+1) * sizeof (int)) ;
    Ai = (int *) malloc (nz * sizeof (int)) ;
    Ax = (double *) malloc (nz * sizeof (double)) ;
    b = (double *) malloc (ldim * NRHS * sizeof (double)) ;
    r = (double *) malloc (n * sizeof (double)) ;
    x = (double *) malloc (ldim * NRHS * sizeof (double)) ;

    if (!Ap || !Ai || !Ax || !b || !r || !x)
    {
	printf ("out of memory for input matrix\n") ;
	exit (1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the matrix (tmp/Ap, tmp/Ai, and tmp/Ax) */
    /* ---------------------------------------------------------------------- */

    printf ("File: tmp/Ap\n") ;
    f = fopen ("tmp/Ap", "r") ;
    if (!f)
    {
	printf ("Unable to open file\n") ;
	exit (1) ;
    }
    for (j = 0 ; j <= n ; j++)
    {
	fscanf (f, "%d\n", &Ap [j]) ;
    }
    fclose (f) ;

    printf ("File: tmp/Ai\n") ;
    f = fopen ("tmp/Ai", "r") ;
    if (!f)
    {
	printf ("Unable to open file\n") ;
	exit (1) ;
    }
    for (p = 0 ; p < nz ; p++)
    {
	fscanf (f, "%d\n", &Ai [p]) ;
    }
    fclose (f) ;

    printf ("File: tmp/Ax\n") ;
    f = fopen ("tmp/Ax", "r") ;
    if (!f)
    {
	printf ("Unable to open file\n") ;
	exit (1) ;
    }
    for (p = 0 ; p < nz ; p++)
    {
	fscanf (f, "%lg\n", &Ax [p]) ;
    }
    fclose (f) ;

    printf ("n %d nrow %d ncol %d nz %d\n", n, nrow, ncol, nz) ;

    /* ---------------------------------------------------------------------- */
    /* b = A * xtrue, or read b from a file */
    /* ---------------------------------------------------------------------- */

    for (i = 0 ; i < ldim * NRHS ; i++) b [i] = 0 ;

    rhs = FALSE ;
    f = fopen ("tmp/b", "r") ;
    if (f != (FILE *) NULL)
    {
	printf ("Reading tmp/b\n") ;
	rhs = TRUE ;
	for (i = 0 ; i < n ; i++)
	{
	    fscanf (f, "%lg\n", &b [i]) ;
	}
	fclose (f) ;
    }
    else
    {
	Atimesx (n, Ap, Ai, Ax, b, FALSE) ;
    }

    /* create b (:,2:NRHS) */
    Y = b + ldim ;
    for (s = 1 ; s < NRHS ; s++)
    {
	for (k = 0 ; k < n ; k++)
	{
	    Y [k] = s + ((double) n) / 1000 ;
	}
	Y += ldim ;
    }

    /*
    atime [0] = 0 ;
    atime [1] = 0 ;
    ftime [0] = 0 ;
    ftime [1] = 0 ;
    rtime [0] = 0 ;
    rtime [1] = 0 ;
    stime [0] = 0 ;
    stime [1] = 0 ;
    stime2 [0] = 0 ;
    stime2 [1] = 0 ;
    */
    stats [0] = 0 ;
    stats [1] = 0 ;

    do_btf = 1 ;
    ordering = 0 ;
    scale = -1 ;
    halt_if_singular = TRUE ;
    do_sort = FALSE ;

    /*
for (do_btf = 0 ; do_btf <= 1 ; do_btf++)
{
for (scale = -1 ; scale <= 2 ; scale++)
{
for (halt_if_singular = TRUE ; halt_if_singular >= FALSE ; halt_if_singular--)
{
*/

for (ordering = 0 ; ordering <= 3 ; ordering++)
{
if (ordering == 2 || ordering == 1) continue ;

/*
for (do_sort = FALSE ; do_sort <= TRUE ; do_sort++)
{
*/

klu_defaults (km) ;
km->ordering = ordering ;	    /* 0: amd, 1: colamd, 2: given, 3: user */
km->btf = do_btf ;		    /* 1: BTF , 0: no BTF */
km->scale = scale ;		    /* -1: none (and no error check)
				    0: none, 1: row, 2: max */
km->user_order = klu_cholmod ;
km->halt_if_singular = halt_if_singular ;

printf ("\nKLUDEMO: ordering: %d  BTF: %d scale: %d halt_if_singular: %d\n",
	ordering, do_btf, scale, halt_if_singular) ;

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization */
    /* ---------------------------------------------------------------------- */

for (trial = 0 ; trial < NTRIAL ; trial++)
{
printf ("---------------- trial %d\n", trial) ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    Symbolic = klu_analyze (n, Ap, Ai, km) ;
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    if (Symbolic == NULL)
    {
	printf ("klu_analyze failed\n") ;
	exit (1) ;
    }

    printf ("\nKLU Symbolic analysis:\n"
	    "dimension of matrix:          %d\n"
	    "nz in off diagonal blocks:    %d\n"
	    "# of off diagonal blocks:     %d\n"
	    "largest block dimension:      %d\n"
	    "estimated nz in L:            %g\n"
	    "estimated nz in U:            %g\n"
	    "symbolic analysis time:       %10.6f (wall) %10.6f (cpu)\n",
	    n, Symbolic->nzoff, Symbolic->nblocks,
	    Symbolic->maxblock, Symbolic->lnz, Symbolic->unz,
	    stats [0], stats [1]) ;

    /* ---------------------------------------------------------------------- */
    /* numeric factorization */
    /* ---------------------------------------------------------------------- */

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    Numeric = klu_factor (Ap, Ai, Ax, Symbolic, km) ;
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    if (Numeric == NULL)
    {
	printf ("klu_factor failed\n") ;
    }

    lnz = (Numeric != NULL) ? Numeric->lnz : 0 ;
    unz = (Numeric != NULL) ? Numeric->unz : 0 ;

    printf ("\nKLU Numeric factorization:\n"
	    "actual nz in L:               %d\n"
	    "actual nz in U:               %d\n"
	    "actual nz in L+U:             %d\n"
	    "# of offdiagonal pivots:      %d\n"
	    "numeric factorization time:   %10.6f (wall) %10.6f (cpu)\n",
	    lnz, unz, lnz + unz - n, km->noffdiag, stats [0], stats [1]) ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_flops (Symbolic, Numeric, km) ;
    t = dsecnd_ ( ) - t ;
    c = CPUTIME - c ;
    printf ("flop count %g time %10.6f %10.6f\n", km->flops, t,c) ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_condest (Ap, Ax, Symbolic, Numeric, &condest, km) ;
    t = dsecnd_ ( ) - t ;
    c = CPUTIME - c ;

    printf ("condition number estimate %g time %10.6f %10.6f\n", condest,t,c) ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_growth (Ap, Ai, Ax, Symbolic, Numeric, &growth, km) ;
    t = dsecnd_ ( ) - t ;
    c = CPUTIME - c ;

    printf ("reciprocal pivot growth:  %g time %10.6f %10.6f\n", growth,t,c) ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_rcond (Symbolic, Numeric, &rcond, km) ;
    t = dsecnd_ ( ) - t ;
    c = CPUTIME - c ;

    printf ("cheap reciprocal cond:    %g time %10.6f %10.6f\n", rcond,t,c) ;

    if (do_sort)
    {
	c = CPUTIME ;
	t = dsecnd_ ( ) ;
	klu_sort (Symbolic, Numeric, km) ;
	t = dsecnd_ ( ) - t ;
	c = CPUTIME - c ;
	printf (" -------------------- sort time: %10.6f (wall) %10.6f (cpu)\n",
	    t, c) ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b */
    /* ---------------------------------------------------------------------- */

    for (i = 0 ; i < n ; i++) x [i] = b [i] ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_solve (Symbolic, Numeric, n, 1, x, km) ;  /* does x = A\x */
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    printf ("\nKLU solve:\n"
	    "solve time:                   %10.6f (wall) %10.6f (cpu)\n",
	    stats [0], stats [1]) ;

    printf ("\nrelative maxnorm of residual, ||Ax-b||/||b||:  %8.2e\n",
	resid (n, Ap, Ai, Ax, x, r, b, FALSE)) ;
    if (!rhs)
    {
	printf ("relative maxnorm of error, ||x-xtrue||/||xtrue||: %8.2e\n\n",
	    err (n, x)) ;
    }

    if (trial == 0)
    {
	printf ("Writing solution to file: tmp/x\n") ;
	f = fopen ("tmp/x", "w") ;
	if (!f)
	{
	    printf ("Unable to open file\n") ;
	    exit (1) ;
	}
	for (i = 0 ; i < n ; i++)
	{
	    fprintf (f, "%24.16e\n", x [i]) ;
	}
	fclose (f) ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve A'x=b */
    /* ---------------------------------------------------------------------- */

    for (i = 0 ; i < n ; i++) x [i] = b [i] ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_tsolve (Symbolic, Numeric, n, 1, x, km) ;
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    printf ("\nKLU tsolve:\n"
	    "solve time:                   %10.6f (wall) %10.6f (cpu)\n",
	    stats [0], stats [1]) ;

    printf ("\nrelative maxnorm of residual, ||A'x-b||/||b||: %8.2e\n",
	resid (n, Ap, Ai, Ax, x, r, b, TRUE)) ;

    /* ---------------------------------------------------------------------- */
    /* solve AX=B */
    /* ---------------------------------------------------------------------- */

#if 0

    for (i = 0 ; i < ldim*NRHS ; i++) x [i] = b [i] ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_solve (Symbolic, Numeric, ldim, NRHS, x, km) ;  /* does x = A\x */
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    printf ("\nKLU solve (nrhs = %d:\n"
	    "solve time:                   %10.6f (wall) %10.6f (cpu)\n",
	    NRHS, stats [0], stats [1]) ;

    for (s = 0 ; s < NRHS ; s++)
    {
	printf ("relative maxnorm of residual, ||Ax-b||/||b||:  %8.2e\n",
	    resid (n, Ap, Ai, Ax, x + s*ldim, r, b + s*ldim, FALSE)) ;
    }

    for (i = 0 ; i < ldim*NRHS ; i++) x [i] = b [i] ;

    for (s = 0 ; s < NRHS ; s++)
    {
	klu_solve (Symbolic, Numeric, n, 1, x+s*ldim, km) ;  /* does x = A\x */
    }
    for (s = 0 ; s < NRHS ; s++)
    {
	printf ("relative maxnorm of residual, ||Ax-b||/||b||:  %8.2e\n",
	    resid (n, Ap, Ai, Ax, x + s*ldim, r, b + s*ldim, FALSE)) ;
    }

    for (nrhs = 1 ; nrhs <= NRHS ; nrhs++)
    {
	for (i = 0 ; i < ldim*NRHS ; i++) x [i] = b [i] ;

	klu_solve (Symbolic, Numeric, ldim, nrhs, x, km) ;  /* does x = A\x */

	printf ("\nKLU  solve (nrhs = %2d : ", nrhs) ;

	for (s = 0 ; s < nrhs ; s++)
	{
	    printf ("relative maxnorm of residual, ||Ax-b||/||b||:  %8.2e\n",
		resid (n, Ap, Ai, Ax, x + s*ldim, r, b + s*ldim, FALSE)) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* solve A'X=B */
    /* ---------------------------------------------------------------------- */

    for (i = 0 ; i < ldim*NRHS ; i++) x [i] = b [i] ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_tsolve (Symbolic, Numeric, ldim, NRHS, x, km) ;  /* does x = A'\x */
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    printf ("\nKLU tsolve (nrhs = %d:\n"
	    "solve time:                   %10.6f (wall) %10.6f (cpu)\n",
	    NRHS, stats [0], stats [1]) ;

    for (s = 0 ; s < NRHS ; s++)
    {
	printf ("relative maxnorm of residual, ||A'x-b||/||b||: %8.2e\n",
	    resid (n, Ap, Ai, Ax, x + s*ldim, r, b + s*ldim, TRUE)) ;
    }

    for (i = 0 ; i < ldim*NRHS ; i++) x [i] = b [i] ;

    for (s = 0 ; s < NRHS ; s++)
    {
	klu_tsolve (Symbolic, Numeric, n, 1, x+s*ldim, 0, km) ;
    }
    for (s = 0 ; s < NRHS ; s++)
    {
	printf ("relative maxnorm of residual, ||A'x-b||/||b||: %8.2e\n",
	    resid (n, Ap, Ai, Ax, x + s*ldim, r, b + s*ldim, TRUE)) ;
    }

    for (nrhs = 1 ; nrhs <= NRHS ; nrhs++)
    {
	for (i = 0 ; i < ldim*NRHS ; i++) x [i] = b [i] ;

	klu_tsolve (Symbolic, Numeric, ldim, nrhs, x, 0, km) ;

	printf ("\nKLU tsolve (nrhs = %2d : ", nrhs) ;

	for (s = 0 ; s < nrhs ; s++)
	{
	    printf ("relative maxnorm of residual, ||A'x-b||/||b||: %8.2e\n",
		resid (n, Ap, Ai, Ax, x + s*ldim, r, b + s*ldim, TRUE)) ;
	}
    }

#endif

    /* ---------------------------------------------------------------------- */
    /* numeric refactorization, no pivot (just to test the code) */
    /* ---------------------------------------------------------------------- */

for (rescale = -1 ; rescale <= 2 ; rescale++)
{
    km->scale = rescale ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    if (!klu_refactor (Ap, Ai, Ax, Symbolic, Numeric, km))
    {
	printf ("klu_refactor failed\n") ;
    }
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    printf ("\nKLU Numeric refactorization: scale %d\n"
	    "numeric refactorization time: %10.6f (wall) %10.6f (cpu)\n",
	    km->scale, stats [0], stats [1]) ;

    klu_condest (Ap, Ax, Symbolic, Numeric, &condest, km) ;
    printf ("condition number estimate %g\n", condest) ;
    klu_growth (Ap, Ai, Ax, Symbolic, Numeric, &growth, km) ;
    printf ("reciprocal pivot growth:  %g\n", growth) ;
    klu_rcond (Symbolic, Numeric, &rcond, km) ;
    printf ("cheap reciprocal cond:    %g\n", rcond) ;

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b again */
    /* ---------------------------------------------------------------------- */

    for (i = 0 ; i < n ; i++) x [i] = b [i] ;

    c = CPUTIME ;
    t = dsecnd_ ( ) ;
    klu_solve (Symbolic, Numeric, n, 1, x, km) ;
    stats [0] = dsecnd_ ( ) - t ;
    stats [1] = CPUTIME - c ;

    printf ("\nKLU solve:\n"
	    "solve time:                   %10.6f (wall) %10.6f (cpu)\n",
	    stats [0], stats [1]) ;

    printf ("\nrelative maxnorm of residual, ||Ax-b||/||b||:  %8.2e\n",
	resid (n, Ap, Ai, Ax, x, r, b, FALSE)) ;
    if (!rhs)
    {
	printf ("relative maxnorm of error, ||x-xtrue||/||xtrue||: %8.2e\n\n",
	    err (n, x)) ;
    }

}
    /* restore scale parameter for next trial */
    km->scale = scale ;

    /* ---------------------------------------------------------------------- */
    /* free the Symbolic and Numeric factorization, and all workspace */
    /* ---------------------------------------------------------------------- */

    klu_free_symbolic (&Symbolic, km) ;
    klu_free_numeric (&Numeric, km) ;
}

/*
    printf ("analysis: %12.4f  %12.4f\n", atime[0] / NTRIAL, atime [1] / NTRIAL) ;
    printf ("factor:   %12.4f  %12.4f\n", ftime[0] / NTRIAL, ftime [1] / NTRIAL) ;
    printf ("refactor: %12.4f  %12.4f\n", rtime[0] / NTRIAL, rtime [1] / NTRIAL) ;
    printf ("solve:    %12.4f  %12.4f\n", stime[0] / (4*NTRIAL), stime [1] / (4*NTRIAL)) ;
    printf ("solve2:   %12.4f  %12.4f (per rhs)\n", stime2[0] / (NTRIAL*NRHS), stime2 [1] / (NTRIAL*NRHS)) ;
    */


fflush (stdout) ;

/* } */ }

/*
} } }
*/

    free (Ap) ;
    free (Ai) ;
    free (Ax) ;
    free (b) ;
    free (r) ;
    free (x) ;

    printf ("kludemo done\n") ;
    return (0) ;
}
