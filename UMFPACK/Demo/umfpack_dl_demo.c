//------------------------------------------------------------------------------
// UMFPACK/Demo/umfpack_dl_demo: C demo for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------


/*
  A demo of UMFPACK:   umfpack_dl_* version.

  First, factor and solve a 5-by-5 system, Ax=b, using default parameters.
  Then solve A'x=b using the factors of A.   Modify one entry (A (1,4) = 0,
  where the row and column indices range from 0 to 4.  The pattern of A
  has not changed (it has explicitly zero entry), so a reanalysis with
  umfpack_dl_symbolic does not need to be done.  Refactorize (with
  umfpack_dl_numeric), and solve Ax=b.  Note that the pivot ordering has
  changed.  Next, change all of the entries in A, but not the pattern.

  Finally, compute C = A', and do the symbolic and numeric factorization of C.
  Factorizing A' can sometimes be better than factorizing A itself (less work
  and memory usage).  Solve C'x=b twice; the solution is the same as the
  solution to Ax=b.

  A note about zero-sized arrays:  UMFPACK uses many user-provided arrays of
  size n (order of the matrix), and of size nz (the number of nonzeros in a
  matrix).  n cannot be zero; UMFPACK does not handle zero-dimensioned arrays.
  However, nz can be zero.  If you attempt to malloc an array of size nz = 0,
  however, malloc will return a null pointer which UMFPACK will report as a
  "missing argument."  Thus, nz1 in this code is set to MAX (nz,1), and
  similarly for lnz and unz.  Lnz can never be zero, however, since L is always
  unit diagonal.
*/

/* -------------------------------------------------------------------------- */
/* definitions */
/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include "umfpack.h"

#define ABS(x) ((x) >= 0 ? (x) : -(x))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif

/* -------------------------------------------------------------------------- */
/* triplet form of the matrix.  The triplets can be in any order. */
/* -------------------------------------------------------------------------- */

static SuiteSparse_long n = 5, nz = 12 ;
static SuiteSparse_long Arow [ ] = { 0,  4,  1,  1,   2,   2,  0,  1,  2,  3,  4,  4} ;
static SuiteSparse_long Acol [ ] = { 0,  4,  0,  2,   1,   2,  1,  4,  3,  2,  1,  2} ;
static double Aval [ ] = {2., 1., 3., 4., -1., -3., 3., 6., 2., 1., 4., 2.} ;
static double b [ ] = {8., 45., -3., 3., 19.}, x [5], r [5] ;


/* -------------------------------------------------------------------------- */
/* error: print a message and exit */
/* -------------------------------------------------------------------------- */

static void error
(
    char *message
)
{
    printf ("\n\n====== error: %s =====\n\n", message) ;
    exit (1) ;
}


/* -------------------------------------------------------------------------- */
/* resid: compute the residual, r = Ax-b or r = A'x=b and return maxnorm (r) */
/* -------------------------------------------------------------------------- */

static double resid
(
    int transpose,
    SuiteSparse_long Ap [ ],
    SuiteSparse_long Ai [ ],
    double Ax [ ]
)
{
    SuiteSparse_long i, j, p ;
    double norm ;

    for (i = 0 ; i < n ; i++)
    {
	r [i] = -b [i] ;
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
    norm = 0. ;
    for (i = 0 ; i < n ; i++)
    {
	norm = MAX (ABS (r [i]), norm) ;
    }
    return (norm) ;
}


/* -------------------------------------------------------------------------- */
/* main program */
/* -------------------------------------------------------------------------- */

int main (int argc, char **argv)
{
    double Info [UMFPACK_INFO], Control [UMFPACK_CONTROL], *Ax, *Cx, *Lx, *Ux,
	*W, t [2], *Dx, rnorm, *Rb, *y, *Rs ;
    SuiteSparse_long *Ap, *Ai, *Cp, *Ci, row, col, p, lnz, unz, nr, nc, *Lp, *Li, *Ui, *Up,
	*P, *Q, *Lj, i, j, k, anz, nfr, nchains, *Qinit, fnpiv, lnz1, unz1, nz1,
	status, *Front_npivcol, *Front_parent, *Chain_start, *Wi, *Pinit, n1,
	*Chain_maxrows, *Chain_maxcols, *Front_1strow, *Front_leftmostdesc,
	nzud, do_recip, *Dmap ;
    void *Symbolic, *Numeric ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    umfpack_tic (t) ;

    printf ("\nUMFPACK V%d.%d (%s) demo: _dl_ version\n",
	    UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_DATE) ;

    /* get the default control parameters */
    umfpack_dl_defaults (Control) ;

    /* change the default print level for this demo */
    /* (otherwise, nothing will print) */
    Control [UMFPACK_PRL] = 6 ;

    /* print the license agreement */
    umfpack_dl_report_status (Control, UMFPACK_OK) ;
    Control [UMFPACK_PRL] = 5 ;

    /* print the control parameters */
    umfpack_dl_report_control (Control) ;

    /* ---------------------------------------------------------------------- */
    /* print A and b, and convert A to column-form */
    /* ---------------------------------------------------------------------- */

    /* print the right-hand-side */
    printf ("\nb: ") ;
    (void) umfpack_dl_report_vector (n, b, Control) ;

    /* print the triplet form of the matrix */
    printf ("\nA: ") ;
    (void) umfpack_dl_report_triplet (n, n, nz, Arow, Acol, Aval,
	Control) ;

    /* convert to column form */
    nz1 = MAX (nz,1) ;	/* ensure arrays are not of size zero. */
    Ap = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Ai = (SuiteSparse_long *) malloc (nz1 * sizeof (SuiteSparse_long)) ;
    Ax = (double *) malloc (nz1 * sizeof (double)) ;
    if (!Ap || !Ai || !Ax)
    {
	error ("out of memory") ;
    }

    status = umfpack_dl_triplet_to_col (n, n, nz, Arow, Acol, Aval,
	Ap, Ai, Ax, (SuiteSparse_long *) NULL) ;

    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_triplet_to_col failed") ;
    }

    /* print the column-form of A */
    printf ("\nA: ") ;
    (void) umfpack_dl_report_matrix (n, n, Ap, Ai, Ax, 1, Control) ;

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_symbolic (n, n, Ap, Ai, Ax, &Symbolic,
	Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_info (Control, Info) ;
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_symbolic failed") ;
    }

    /* print the symbolic factorization */

    printf ("\nSymbolic factorization of A: ") ;
    (void) umfpack_dl_report_symbolic (Symbolic, Control) ;

    /* ---------------------------------------------------------------------- */
    /* numeric factorization */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_numeric (Ap, Ai, Ax, Symbolic, &Numeric,
	Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_info (Control, Info) ;
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_numeric failed") ;
    }

    /* print the numeric factorization */
    printf ("\nNumeric factorization of A: ") ;
    (void) umfpack_dl_report_numeric (Numeric, Control) ;

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, x, b,
	Numeric, Control, Info) ;
    umfpack_dl_report_info (Control, Info) ;
    umfpack_dl_report_status (Control, status) ;
    if (status < 0)
    {
	error ("umfpack_dl_solve failed") ;
    }
    printf ("\nx (solution of Ax=b): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (FALSE, Ap, Ai, Ax) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* compute the determinant */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_get_determinant (x, r, Numeric, Info) ;
    umfpack_dl_report_status (Control, status) ;
    if (status < 0)
    {
	error ("umfpack_dl_get_determinant failed") ;
    }
    printf ("determinant: (%g", x [0]) ;
    printf (") * 10^(%g)\n", r [0]) ;

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b, broken down into steps */
    /* ---------------------------------------------------------------------- */

    /* Rb = R*b */
    Rb  = (double *) malloc (n * sizeof (double)) ;
    y   = (double *) malloc (n * sizeof (double)) ;
    if (!Rb || !y) error ("out of memory") ;

    status = umfpack_dl_scale (Rb, b, Numeric) ;
    if (status < 0) error ("umfpack_dl_scale failed") ;
    /* solve Ly = P*(Rb) */
    status = umfpack_dl_solve (UMFPACK_Pt_L, Ap, Ai, Ax, y, Rb,
	Numeric, Control, Info) ;
    if (status < 0) error ("umfpack_dl_solve failed") ;
    /* solve UQ'x=y */
    status = umfpack_dl_solve (UMFPACK_U_Qt, Ap, Ai, Ax, x, y,
	Numeric, Control, Info) ;
    if (status < 0) error ("umfpack_dl_solve failed") ;
    printf ("\nx (solution of Ax=b, solve is split into 3 steps): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (FALSE, Ap, Ai, Ax) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    free (Rb) ;
    free (y) ;

    /* ---------------------------------------------------------------------- */
    /* solve A'x=b */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_solve (UMFPACK_At, Ap, Ai, Ax, x, b,
	Numeric, Control, Info) ;
    umfpack_dl_report_info (Control, Info) ;
    if (status < 0)
    {
	error ("umfpack_dl_solve failed") ;
    }
    printf ("\nx (solution of A'x=b): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (TRUE, Ap, Ai, Ax) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* modify one numerical value in the column-form of A */
    /* ---------------------------------------------------------------------- */

    /* change A (1,4), look for row 1 in column 4. */
    row = 1 ;
    col = 4 ;
    for (p = Ap [col] ; p < Ap [col+1] ; p++)
    {
	if (row == Ai [p])
	{
	    printf ("\nchanging A (%" SuiteSparse_long_idd ",%" SuiteSparse_long_idd ") to zero\n", row, col) ;
	    Ax [p] = 0.0 ;
	    break ;
	}
    }
    printf ("\nmodified A: ") ;
    (void) umfpack_dl_report_matrix (n, n, Ap, Ai, Ax, 1, Control) ;

    /* ---------------------------------------------------------------------- */
    /* redo the numeric factorization */
    /* ---------------------------------------------------------------------- */

    /* The pattern (Ap and Ai) hasn't changed, so the symbolic factorization */
    /* doesn't have to be redone, no matter how much we change Ax. */

    /* We don't need the Numeric object any more, so free it. */
    umfpack_dl_free_numeric (&Numeric) ;

    /* Note that a memory leak would have occurred if the old Numeric */
    /* had not been free'd with umfpack_dl_free_numeric above. */
    status = umfpack_dl_numeric (Ap, Ai, Ax, Symbolic, &Numeric,
	Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_info (Control, Info) ;
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_numeric failed") ;
    }
    printf ("\nNumeric factorization of modified A: ") ;
    (void) umfpack_dl_report_numeric (Numeric, Control) ;

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b, with the modified A */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, x, b,
	Numeric, Control, Info) ;
    umfpack_dl_report_info (Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_solve failed") ;
    }
    printf ("\nx (with modified A): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (FALSE, Ap, Ai, Ax) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* modify all of the numerical values of A, but not the pattern */
    /* ---------------------------------------------------------------------- */

    for (col = 0 ; col < n ; col++)
    {
	for (p = Ap [col] ; p < Ap [col+1] ; p++)
	{
	    row = Ai [p] ;
	    printf ("changing ") ;
	    printf ("A (%" SuiteSparse_long_idd ",%" SuiteSparse_long_idd ") from %g", row, col, Ax [p]) ;
	    Ax [p] = Ax [p] + col*10 - row ;
	    printf (" to %g\n", Ax [p]) ;
	}
    }
    printf ("\ncompletely modified A (same pattern): ") ;
    (void) umfpack_dl_report_matrix (n, n, Ap, Ai, Ax, 1, Control) ;

    /* ---------------------------------------------------------------------- */
    /* save the Symbolic object to file, free it, and load it back in */
    /* ---------------------------------------------------------------------- */

    /* use the default filename, "symbolic.umf" */
    printf ("\nSaving symbolic object:\n") ;
    status = umfpack_dl_save_symbolic (Symbolic, (char *) NULL) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_save_symbolic failed") ;
    }
    printf ("\nFreeing symbolic object:\n") ;
    umfpack_dl_free_symbolic (&Symbolic) ;
    printf ("\nLoading symbolic object:\n") ;
    status = umfpack_dl_load_symbolic (&Symbolic, (char *) NULL) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_load_symbolic failed") ;
    }
    printf ("\nDone loading symbolic object\n") ;

    /* ---------------------------------------------------------------------- */
    /* redo the numeric factorization */
    /* ---------------------------------------------------------------------- */

    umfpack_dl_free_numeric (&Numeric) ;
    status = umfpack_dl_numeric (Ap, Ai, Ax, Symbolic, &Numeric,
	Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_info (Control, Info) ;
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_numeric failed") ;
    }
    printf ("\nNumeric factorization of completely modified A: ") ;
    (void) umfpack_dl_report_numeric (Numeric, Control) ;

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b, with the modified A */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, x, b,
	Numeric, Control, Info) ;
    umfpack_dl_report_info (Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_solve failed") ;
    }
    printf ("\nx (with completely modified A): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (FALSE, Ap, Ai, Ax) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* free the symbolic and numeric factorization */
    /* ---------------------------------------------------------------------- */

    umfpack_dl_free_symbolic (&Symbolic) ;
    umfpack_dl_free_numeric (&Numeric) ;

    /* ---------------------------------------------------------------------- */
    /* C = transpose of A */
    /* ---------------------------------------------------------------------- */

    Cp = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Ci = (SuiteSparse_long *) malloc (nz1 * sizeof (SuiteSparse_long)) ;
    Cx = (double *) malloc (nz1 * sizeof (double)) ;
    if (!Cp || !Ci || !Cx)
    {
	error ("out of memory") ;
    }
    status = umfpack_dl_transpose (n, n, Ap, Ai, Ax,
	(SuiteSparse_long *) NULL, (SuiteSparse_long *) NULL, Cp, Ci, Cx) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_transpose failed: ") ;
    }
    printf ("\nC (transpose of A): ") ;
    (void) umfpack_dl_report_matrix (n, n, Cp, Ci, Cx, 1, Control) ;

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization of C */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_symbolic (n, n, Cp, Ci, Cx, &Symbolic,
	Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_info (Control, Info) ;
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_symbolic failed") ;
    }
    printf ("\nSymbolic factorization of C: ") ;
    (void) umfpack_dl_report_symbolic (Symbolic, Control) ;

    /* ---------------------------------------------------------------------- */
    /* copy the contents of Symbolic into user arrays print them */
    /* ---------------------------------------------------------------------- */

    printf ("\nGet the contents of the Symbolic object for C:\n") ;
    printf ("(compare with umfpack_dl_report_symbolic output, above)\n") ;
    Pinit = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Qinit = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Front_npivcol = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Front_1strow = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Front_leftmostdesc = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Front_parent = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Chain_start = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Chain_maxrows = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Chain_maxcols = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Dmap = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    if (!Pinit || !Qinit || !Front_npivcol || !Front_parent || !Chain_start ||
	!Chain_maxrows || !Chain_maxcols || !Front_1strow || !Dmap ||
	!Front_leftmostdesc)
    {
	error ("out of memory") ;
    }

    status = umfpack_dl_get_symbolic (&nr, &nc, &n1, &anz, &nfr, &nchains,
	Pinit, Qinit, Front_npivcol, Front_parent, Front_1strow,
	Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols,
	Dmap, Symbolic) ;

    if (status < 0)
    {
	error ("symbolic factorization invalid") ;
    }

    printf ("From the Symbolic object, C is of dimension %" SuiteSparse_long_idd
            "-by-%" SuiteSparse_long_idd "\n", nr, nc);
    printf ("   with nz = %" SuiteSparse_long_idd
            ", number of fronts = %" SuiteSparse_long_idd ",\n", nz, nfr) ;
    printf ("   number of frontal matrix chains = %" SuiteSparse_long_idd "\n",
            nchains) ;

    printf ("\nPivot columns in each front, and parent of each front:\n") ;
    k = 0 ;
    for (i = 0 ; i < nfr ; i++)
    {
      fnpiv = Front_npivcol [i] ;
      printf ("    Front %" SuiteSparse_long_idd
              ": parent front: %" SuiteSparse_long_idd
              " number of pivot cols: %" SuiteSparse_long_idd "\n",
              i, Front_parent [i], fnpiv) ;
      for (j = 0 ; j < fnpiv ; j++)
      {
          col = Qinit [k] ;
          printf ("        %" SuiteSparse_long_idd "-th pivot column is column "
                  "%" SuiteSparse_long_idd " in original matrix\n",
                  k, col) ;
          k++ ;
      }
    }

    printf ("\nNote that the column ordering, above, will be refined\n") ;
    printf ("in the numeric factorization below.  The assignment of pivot\n") ;
    printf ("columns to frontal matrices will always remain unchanged.\n") ;

    printf ("\nTotal number of pivot columns in frontal matrices: %" SuiteSparse_long_idd "\n", k) ;

    printf ("\nFrontal matrix chains:\n") ;
    for (j = 0 ; j < nchains ; j++)
    {
      printf ("   Frontal matrices %" SuiteSparse_long_idd " to "
              "%" SuiteSparse_long_idd " are factorized in a single\n",
              Chain_start [j], Chain_start [j+1] - 1) ;
      printf ("        working array of size %" SuiteSparse_long_idd "-by-"
              "%" SuiteSparse_long_idd "\n",
              Chain_maxrows [j], Chain_maxcols [j]) ;
    }

    //--------------------------------------------------------------------------
    // copy the Symbolic object
    //--------------------------------------------------------------------------

    void *Symbolic_copy = NULL ;
    printf ("\nCopying symbolic object:\n") ;
    status = umfpack_dl_copy_symbolic (&Symbolic_copy, Symbolic) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_copy_symbolic failed") ;
    }
    printf ("\nSymbolic factorization of C (copy): ") ;
    (void) umfpack_dl_report_symbolic (Symbolic_copy, Control) ;
    umfpack_dl_free_symbolic (&Symbolic) ;
    Symbolic = Symbolic_copy ;
    printf ("\nDone copying symbolic object\n") ;

    //--------------------------------------------------------------------------
    // serialize/deserialize the Symbolic object
    //--------------------------------------------------------------------------

    // determine the required blobsize
    SuiteSparse_long S_blobsize ;
    status = umfpack_dl_serialize_symbolic_size (&S_blobsize, Symbolic) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_serialize_symbolic_size failed") ;
    }
    printf ("\nSymbolic blob size: %" SuiteSparse_long_idd "\n", S_blobsize) ;
    // allocate the blob
    void *S_blob = malloc (S_blobsize) ;
    if (!S_blob)
    {
	error ("out of memory") ;
    }
    // serialize the blob
    status = umfpack_dl_serialize_symbolic (S_blob, S_blobsize, Symbolic) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_serialize_symbolic failed") ;
    }
    // free the Symbolic object; its contents are preserved in the blob
    umfpack_dl_free_symbolic (&Symbolic) ;
    // deserialize the blob back into the Symbolic object
    status = umfpack_dl_deserialize_symbolic (&Symbolic, S_blob, S_blobsize) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_deserialize_symbolic failed") ;
    }
    printf ("\nDone serialize/deserialize of symbolic object\n") ;
    free (S_blob) ;

    /* ---------------------------------------------------------------------- */
    /* numeric factorization of C */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_numeric (Cp, Ci, Cx, Symbolic, &Numeric,
	Control, Info) ;
    if (status < 0)
    {
	error ("umfpack_dl_numeric failed") ;
    }
    printf ("\nNumeric factorization of C: ") ;
    (void) umfpack_dl_report_numeric (Numeric, Control) ;

    /* ---------------------------------------------------------------------- */
    /* extract the LU factors of C and print them */
    /* ---------------------------------------------------------------------- */

    if (umfpack_dl_get_lunz (&lnz, &unz, &nr, &nc, &nzud, Numeric) < 0)
    {
	error ("umfpack_dl_get_lunz failed") ;
    }
    /* ensure arrays are not of zero size */
    lnz1 = MAX (lnz,1) ;
    unz1 = MAX (unz,1) ;
    Lp = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Lj = (SuiteSparse_long *) malloc (lnz1 * sizeof (SuiteSparse_long)) ;
    Lx = (double *) malloc (lnz1 * sizeof (double)) ;
    Up = (SuiteSparse_long *) malloc ((n+1) * sizeof (SuiteSparse_long)) ;
    Ui = (SuiteSparse_long *) malloc (unz1 * sizeof (SuiteSparse_long)) ;
    Ux = (double *) malloc (unz1 * sizeof (double)) ;
    P = (SuiteSparse_long *) malloc (n * sizeof (SuiteSparse_long)) ;
    Q = (SuiteSparse_long *) malloc (n * sizeof (SuiteSparse_long)) ;
    Dx = (double *) NULL ;	/* D vector not requested */
    Rs  = (double *) malloc (n * sizeof (double)) ;
    if (!Lp || !Lj || !Lx || !Up || !Ui || !Ux || !P || !Q || !Rs)
    {
	error ("out of memory") ;
    }
    status = umfpack_dl_get_numeric (Lp, Lj, Lx, Up, Ui, Ux,
	P, Q, Dx, &do_recip, Rs, Numeric) ;
    if (status < 0)
    {
	error ("umfpack_dl_get_numeric failed") ;
    }

    printf ("\nL (lower triangular factor of C): ") ;
    (void) umfpack_dl_report_matrix (n, n, Lp, Lj, Lx, 0, Control) ;
    printf ("\nU (upper triangular factor of C): ") ;
    (void) umfpack_dl_report_matrix (n, n, Up, Ui, Ux, 1, Control) ;
    printf ("\nP: ") ;
    (void) umfpack_dl_report_perm (n, P, Control) ;
    printf ("\nQ: ") ;
    (void) umfpack_dl_report_perm (n, Q, Control) ;
    printf ("\nScale factors: row i of A is to be ") ;
    if (do_recip)
    {
	printf ("multiplied by the ith scale factor\n") ;
    }
    else
    {
	printf ("divided by the ith scale factor\n") ;
    }
    for (i = 0 ; i < n ; i++) printf ("%" SuiteSparse_long_idd ": %g\n", i, Rs [i]) ;

    /* ---------------------------------------------------------------------- */
    /* convert L to triplet form and print it */
    /* ---------------------------------------------------------------------- */

    /* Note that L is in row-form, so it is the row indices that are created */
    /* by umfpack_dl_col_to_triplet. */

    printf ("\nConverting L to triplet form, and printing it:\n") ;
    Li = (SuiteSparse_long *) malloc (lnz1 * sizeof (SuiteSparse_long)) ;
    if (!Li)
    {
	error ("out of memory") ;
    }
    if (umfpack_dl_col_to_triplet (n, Lp, Li) < 0)
    {
	error ("umfpack_dl_col_to_triplet failed") ;
    }
    printf ("\nL, in triplet form: ") ;
    (void) umfpack_dl_report_triplet (n, n, lnz, Li, Lj, Lx, Control) ;

    /* ---------------------------------------------------------------------- */
    /* save the Numeric object to file, free it, and load it back in */
    /* ---------------------------------------------------------------------- */

    /* use the default filename, "numeric.umf" */
    printf ("\nSaving numeric object:\n") ;
    status = umfpack_dl_save_numeric (Numeric, (char *) NULL) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_save_numeric failed") ;
    }
    printf ("\nFreeing numeric object:\n") ;
    umfpack_dl_free_numeric (&Numeric) ;
    printf ("\nLoading numeric object:\n") ;
    status = umfpack_dl_load_numeric (&Numeric, (char *) NULL) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_load_numeric failed") ;
    }
    printf ("\nDone loading numeric object\n") ;

    /* ---------------------------------------------------------------------- */
    /* solve C'x=b */
    /* ---------------------------------------------------------------------- */

    status = umfpack_dl_solve (UMFPACK_At, Cp, Ci, Cx, x, b,
	Numeric, Control, Info) ;
    umfpack_dl_report_info (Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_solve failed") ;
    }
    printf ("\nx (solution of C'x=b): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (TRUE, Cp, Ci, Cx) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    //--------------------------------------------------------------------------
    // copy the Numeric object
    //--------------------------------------------------------------------------

    void *Numeric_copy = NULL ;
    printf ("\nCopying numeric object:\n") ;
    status = umfpack_dl_copy_numeric (&Numeric_copy, Numeric) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_copy_numeric failed") ;
    }
    printf ("\nNumeric factorization of C (copy): ") ;
    (void) umfpack_dl_report_numeric (Numeric_copy, Control) ;
    umfpack_dl_free_numeric (&Numeric) ;
    Numeric = Numeric_copy ;
    Numeric_copy = NULL ;
    printf ("\nDone copying numeric object\n") ;

    //--------------------------------------------------------------------------
    // serialize/deserialize the Numeric object
    //--------------------------------------------------------------------------

    // determine the required blobsize
    SuiteSparse_long N_blobsize ;
    status = umfpack_dl_serialize_numeric_size (&N_blobsize, Numeric) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_serialize_numeric_size failed") ;
    }
    printf ("\nNumeric blob size: %" SuiteSparse_long_idd "\n", N_blobsize) ;
    // allocate the blob
    void *N_blob = malloc (N_blobsize) ;
    if (!N_blob)
    {
	error ("out of memory") ;
    }
    // serialize the blob
    status = umfpack_dl_serialize_numeric (N_blob, N_blobsize, Numeric) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_serialize_numeric failed") ;
    }
    // free the Numeric object; its contents are preserved in the blob
    umfpack_dl_free_numeric (&Numeric) ;
    // deserialize the blob back into the Numeric object
    status = umfpack_dl_deserialize_numeric (&Numeric, N_blob, N_blobsize) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_deserialize_numeric failed") ;
    }
    printf ("\nDone serialize/deserialize of numeric object\n") ;
    free (N_blob) ;

    //--------------------------------------------------------------------------
    // solve C'x=b again, with the new copy
    //--------------------------------------------------------------------------

    status = umfpack_dl_solve (UMFPACK_At, Cp, Ci, Cx, x, b,
	Numeric, Control, Info) ;
    umfpack_dl_report_info (Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_solve failed") ;
    }
    printf ("\nx (solution of C'x=b): (using the copy) ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (TRUE, Cp, Ci, Cx) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* solve C'x=b again, using umfpack_dl_wsolve instead */
    /* ---------------------------------------------------------------------- */

    printf ("\nSolving C'x=b again, using umfpack_dl_wsolve instead:\n") ;
    Wi = (SuiteSparse_long *) malloc (n * sizeof (SuiteSparse_long)) ;
    W = (double *) malloc (5*n * sizeof (double)) ;
    if (!Wi || !W)
    {
	error ("out of memory") ;
    }

    status = umfpack_dl_wsolve (UMFPACK_At, Cp, Ci, Cx, x, b,
	Numeric, Control, Info, Wi, W) ;
    umfpack_dl_report_info (Control, Info) ;
    if (status < 0)
    {
	umfpack_dl_report_status (Control, status) ;
	error ("umfpack_dl_wsolve failed") ;
    }
    printf ("\nx (solution of C'x=b): ") ;
    (void) umfpack_dl_report_vector (n, x, Control) ;
    rnorm = resid (TRUE, Cp, Ci, Cx) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* free everything */
    /* ---------------------------------------------------------------------- */

    /* This is not strictly required since the process is exiting and the */
    /* system will reclaim the memory anyway.  It's useful, though, just as */
    /* a list of what is currently malloc'ed by this program.  Plus, it's */
    /* always a good habit to explicitly free whatever you malloc. */

    free (Ap) ;
    free (Ai) ;
    free (Ax) ;

    free (Cp) ;
    free (Ci) ;
    free (Cx) ;

    free (Pinit) ;
    free (Qinit) ;
    free (Front_npivcol) ;
    free (Front_1strow) ;
    free (Front_leftmostdesc) ;
    free (Front_parent) ;
    free (Chain_start) ;
    free (Chain_maxrows) ;
    free (Chain_maxcols) ;
    free (Dmap) ;

    free (Lp) ;
    free (Lj) ;
    free (Lx) ;

    free (Up) ;
    free (Ui) ;
    free (Ux) ;

    free (P) ;
    free (Q) ;

    free (Li) ;

    free (Wi) ;
    free (W) ;
    free (Rs) ;

    umfpack_dl_free_symbolic (&Symbolic) ;
    umfpack_dl_free_numeric (&Numeric) ;

    /* ---------------------------------------------------------------------- */
    /* print the total time spent in this demo */
    /* ---------------------------------------------------------------------- */

    umfpack_toc (t) ;
    printf ("\numfpack_dl_demo complete.\nTotal time: %5.2f seconds"
	" (CPU time), %5.2f seconds (wallclock time)\n", t [1], t [0]) ;
    return (0) ;
}
