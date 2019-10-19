/* ========================================================================== */
/* === ldlmain.c: LDL main program, for demo and testing ==================== */
/* ========================================================================== */

/* LDLMAIN:  this main program has two purposes.  It provides an example of how
 * to use the LDL routines, and it tests the package.  The output of this
 * program is in ldlmain.out (without AMD) and ldlamd.out (with AMD).  If you
 * did not download and install AMD, then the compilation of this program will
 * not work with USE_AMD defined.  Compare your output with ldlmain.out and
 * ldlamd.out.
 *
 * The program reads in a set of matrices, in the Matrix/ subdirectory.
 * The format of the files is as follows:
 *
 *	one line with the matrix description
 *	one line with 2 integers: n jumbled
 *	n+1 lines, containing the Ap array of size n+1 (column pointers)
 *	nz lines, containing the Ai array of size nz (row indices)
 *	nz lines, containing the Ax array of size nz (numerical values)
 *	n lines, containing the P permutation array of size n
 *
 * The Ap, Ai, Ax, and P data structures are described in ldl.c.
 * The jumbled flag is 1 if the matrix could contain unsorted columns and/or
 * duplicate entries, and 0 otherwise.
 *
 * Once the matrix is read in, it is checked to see if it is valid.  Some
 * matrices are invalid by design, to test the error-checking routines.  If
 * valid, the matrix factorized twice (A and P*A*P').  A linear
 * system Ax=b is set up and solved, and the residual computed.
 * If any system is not solved accurately, this test will fail.
 *
 * This program can also be compiled as a MATLAB mexFunction, with the command
 * "mex ldlmain.c ldl.c".  You can then run the program in MATLAB, with the
 * command "ldlmain".
 *
 * LDL Version 1.3, Copyright (c) 2006 by Timothy A Davis,
 * University of Florida.  All Rights Reserved.  See README for the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ldl.h"

#define NMATRICES 30	    /* number of test matrices in Matrix/ directory */
#define LEN 200		    /* string length */

#ifdef USE_AMD		    /* get AMD include file, if using AMD */
#include "amd.h"
#define PROGRAM "ldlamd"
#else
#define PROGRAM "ldlmain"
#endif

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#include "matrix.h"
#define EXIT_ERROR mexErrMsgTxt ("failure") ;
#define EXIT_OK
#else
#define EXIT_ERROR exit (EXIT_FAILURE) ;
#define EXIT_OK exit (EXIT_SUCCESS) ;
#endif

/* -------------------------------------------------------------------------- */
/* ALLOC_MEMORY: allocate a block of memory */
/* -------------------------------------------------------------------------- */

#define ALLOC_MEMORY(p,type,size) \
p = (type *) malloc ((((size) <= 0) ? 1 : (size)) * sizeof (type)) ; \
if (p == (type *) NULL) \
{ \
    printf (PROGRAM ": out of memory\n") ; \
    EXIT_ERROR ; \
}

/* -------------------------------------------------------------------------- */
/* FREE_MEMORY: free a block of memory */
/* -------------------------------------------------------------------------- */

#define FREE_MEMORY(p,type) \
if (p != (type *) NULL) \
{ \
    free (p) ; \
    p = (type *) NULL ; \
}

/* -------------------------------------------------------------------------- */
/* stand-alone main program, or MATLAB mexFunction */
/* -------------------------------------------------------------------------- */

#ifdef MATLAB_MEX_FILE
void mexFunction
(
    int	nargout,
    mxArray *pargout[ ],
    int	nargin,
    const mxArray *pargin[ ]
)
#else
int main (int argc, char **argv)
#endif
{

    /* ---------------------------------------------------------------------- */
    /* local variables */
    /* ---------------------------------------------------------------------- */

#ifdef USE_AMD
    double Info [AMD_INFO] ;
#endif
    double r, rnorm, flops, maxrnorm = 0. ;
    double *Ax, *Lx, *B, *D, *X, *Y ;
    int matrix, *Ai, *Ap, *Li, *Lp, *P, *Pinv, *Perm, *PermInv, n, i, j, p, nz,
	*Flag, *Pattern, *Lnz, *Parent, trial, lnz, d, jumbled ;
    FILE *f ;
    char s [LEN] ;

    /* ---------------------------------------------------------------------- */
    /* check the error-checking routines with null matrices */
    /* ---------------------------------------------------------------------- */

    i = 1 ;
    if (ldl_valid_perm (-1, (int *) NULL, &i)
	|| !ldl_valid_perm (0, (int *) NULL, &i)
	|| ldl_valid_matrix (-1, (int *) NULL, (int *) NULL)
	|| ldl_valid_matrix (0, &i, &i))
    {
	printf (PROGRAM ": ldl error-checking routine failed\n") ;
	EXIT_ERROR ;
    }

    /* ---------------------------------------------------------------------- */
    /* read in a factorize a set of matrices */
    /* ---------------------------------------------------------------------- */

    for (matrix = 1 ; matrix <= NMATRICES ; matrix++)
    {

	/* ------------------------------------------------------------------ */
	/* read in the matrix and the permutation */
	/* ------------------------------------------------------------------ */

	sprintf (s, "Matrix/A%02d", matrix) ;
	if ((f = fopen (s, "r")) == (FILE *) NULL)
	{
	    printf (PROGRAM ": could not open file: %s\n", s) ;
	    EXIT_ERROR ;
	}
	fgets (s, LEN, f) ;
	printf ("\n\n--------------------------------------------------------");
	printf ("\nInput matrix: %s", s) ;
	printf ("--------------------------------------------------------\n\n");
	fscanf (f, "%d %d", &n, &jumbled) ;
	n = (n < 0) ? (0) : (n) ;
	ALLOC_MEMORY (P, int, n) ;
	ALLOC_MEMORY (Ap, int, n+1) ;
	for (j = 0 ; j <= n ; j++)
	{
	    fscanf (f, "%d", &Ap [j]) ;
	}
	nz = Ap [n] ;
	ALLOC_MEMORY (Ai, int, nz) ;
	ALLOC_MEMORY (Ax, double, nz) ;
	for (p = 0 ; p < nz ; p++)
	{
	    fscanf (f, "%d" , &Ai [p]) ;
	}
	for (p = 0 ; p < nz ; p++)
	{
	    fscanf (f, "%lg", &Ax [p]) ;
	}
	for (j = 0 ; j < n  ; j++)
	{
	    fscanf (f, "%d" , &P  [j]) ;
	}
	fclose (f) ;

	/* ------------------------------------------------------------------ */
	/* check the matrix A and the permutation P */
	/* ------------------------------------------------------------------ */

	ALLOC_MEMORY (Flag, int, n) ;

	/* To test the error-checking routines, some of the input matrices
	 * are not valid.  So this error is expected to occur. */
	if (!ldl_valid_matrix (n, Ap, Ai) || !ldl_valid_perm (n, P, Flag))
	{
	    printf (PROGRAM ": invalid matrix and/or permutation\n") ;
	    FREE_MEMORY (P, int) ;
	    FREE_MEMORY (Ap, int) ;
	    FREE_MEMORY (Ai, int) ;
	    FREE_MEMORY (Ax, double) ;
	    FREE_MEMORY (Flag, int) ;
	    continue ;
	}

	/* ------------------------------------------------------------------ */
	/* get the AMD permutation, if available */
	/* ------------------------------------------------------------------ */

#ifdef USE_AMD
	/* recompute the permutation with AMD */
	/* Assume that AMD produces a valid permutation P. */
	if (amd_order (n, Ap, Ai, P, (double *) NULL, Info) < AMD_OK)
	{
	    printf (PROGRAM ": call to AMD failed\n") ;
	    EXIT_ERROR ;
	}
	amd_control ((double *) NULL) ;
	amd_info (Info) ;
#endif

	/* ------------------------------------------------------------------ */
	/* allocate workspace and the first part of LDL factorization */
	/* ------------------------------------------------------------------ */

	ALLOC_MEMORY (Pinv, int, n) ;
	ALLOC_MEMORY (Y, double, n) ;
	ALLOC_MEMORY (Pattern, int, n) ;
	ALLOC_MEMORY (Lnz, int, n) ;
	ALLOC_MEMORY (Lp, int, n+1) ;
	ALLOC_MEMORY (Parent, int, n) ;
	ALLOC_MEMORY (D, double, n) ;
	ALLOC_MEMORY (B, double, n) ;
	ALLOC_MEMORY (X, double, n) ;

	/* ------------------------------------------------------------------ */
	/* factorize twice, with and without permutation */
	/* ------------------------------------------------------------------ */

	for (trial = 1 ; trial <= 2 ; trial++)
	{

	    if (trial == 1)
	    {
		printf ("Factorize PAP'=LDL' and solve Ax=b\n") ;
		Perm = P ;
		PermInv = Pinv ;
	    }
	    else
	    {
		printf ("Factorize A=LDL' and solve Ax=b\n") ;
		Perm = (int *) NULL ;
		PermInv = (int *) NULL ;
	    }

	    /* -------------------------------------------------------------- */
	    /* symbolic factorization to get Lp, Parent, Lnz, and Pinv */
	    /* -------------------------------------------------------------- */

	    ldl_symbolic (n, Ap, Ai, Lp, Parent, Lnz, Flag, Perm, PermInv) ;
	    lnz = Lp [n] ;

	    /* find # of nonzeros in L, and flop count for ldl_numeric */
	    flops = 0 ;
	    for (j = 0 ; j < n ; j++)
	    {
		flops += ((double) Lnz [j]) * (Lnz [j] + 2) ;
	    }
	    printf ("Nz in L: %d  Flop count: %g\n", lnz, flops) ;

	    /* -------------------------------------------------------------- */
	    /* allocate remainder of L, of size lnz */
	    /* -------------------------------------------------------------- */

	    ALLOC_MEMORY (Li, int, lnz) ;
	    ALLOC_MEMORY (Lx, double, lnz) ;

	    /* -------------------------------------------------------------- */
	    /* numeric factorization to get Li, Lx, and D */
	    /* -------------------------------------------------------------- */

	    d = ldl_numeric (n, Ap, Ai, Ax, Lp, Parent, Lnz, Li, Lx, D,
		Y, Flag, Pattern, Perm, PermInv) ;

	    /* -------------------------------------------------------------- */
	    /* solve, or report singular case */
	    /* -------------------------------------------------------------- */

	    if (d != n)
	    {
		printf ("Ax=b not solved since D(%d,%d) is zero.\n", d, d) ;
	    }
	    else
	    {
		/* construct the right-hand-side, B */
		for (i = 0 ; i < n ; i++)
		{
		    B [i] = 1 + ((double) i) / 100 ;
		}

		/* solve Ax=b */
		if (trial == 1)
		{
		    /* the factorization is LDL' = PAP' */
		    ldl_perm (n, Y, B, P) ;			/* y = Pb */
		    ldl_lsolve (n, Y, Lp, Li, Lx) ;		/* y = L\y */
		    ldl_dsolve (n, Y, D) ;			/* y = D\y */
		    ldl_ltsolve (n, Y, Lp, Li, Lx) ;		/* y = L'\y */
		    ldl_permt (n, X, Y, P) ;			/* x = P'y */
		}
		else
		{
		    /* the factorization is LDL' = A */
		    for (i = 0 ; i < n ; i++)			/* x = b */
		    {
			X [i] = B [i] ;
		    }
		    ldl_lsolve (n, X, Lp, Li, Lx) ;		/* x = L\x */
		    ldl_dsolve (n, X, D) ;			/* x = D\x */
		    ldl_ltsolve (n, X, Lp, Li, Lx) ;		/* x = L'\x */
		}

		/* compute the residual y = Ax-b */
		/* note that this code can tolerate a jumbled matrix */
		for (i = 0 ; i < n ; i++)
		{
		    Y [i] = -B [i] ;
		}
		for (j = 0 ; j < n ; j++)
		{
		    for (p = Ap [j] ; p < Ap [j+1] ; p++)
		    {
			Y [Ai [p]] += Ax [p] * X [j] ;
		    }
		}
		/* rnorm = norm (y, inf) */
		rnorm = 0 ;
		for (i = 0 ; i < n ; i++)
		{
		    r = (Y [i] > 0) ? (Y [i]) : (-Y [i]) ;
		    rnorm = (r > rnorm) ? (r) : (rnorm) ;
		}
		maxrnorm = (rnorm > maxrnorm) ? (rnorm) : (maxrnorm) ;
		printf ("relative maxnorm of residual: %g\n", rnorm) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* free the size-lnz part of L */
	    /* -------------------------------------------------------------- */

	    FREE_MEMORY (Li, int) ;
	    FREE_MEMORY (Lx, double) ;

	}

	/* free everything */
	FREE_MEMORY (P, int) ;
	FREE_MEMORY (Ap, int) ;
	FREE_MEMORY (Ai, int) ;
	FREE_MEMORY (Ax, double) ;
	FREE_MEMORY (Pinv, int) ;
	FREE_MEMORY (Y, double) ;
	FREE_MEMORY (Flag, int) ;
	FREE_MEMORY (Pattern, int) ;
	FREE_MEMORY (Lnz, int) ;
	FREE_MEMORY (Lp, int) ;
	FREE_MEMORY (Parent, int) ;
	FREE_MEMORY (D, double) ;
	FREE_MEMORY (B, double) ;
	FREE_MEMORY (X, double) ;
    }

    printf ("\nLargest residual during all tests: %g\n", maxrnorm) ;
    if (maxrnorm < 1e-8)
    {
	printf ("\n" PROGRAM ": all tests passed\n") ;
	EXIT_OK ;
    }
    else
    {
	printf ("\n" PROGRAM ": one more tests failed (residual too high)\n") ;
	EXIT_ERROR ;
    }
}
