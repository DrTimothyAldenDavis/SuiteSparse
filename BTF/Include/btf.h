/* ========================================================================== */
/* === BTF package ========================================================== */
/* ========================================================================== */

/* MAXTRANS:  find a column permutation Q to give A*Q a zero-free diagonal
 * STRONGCOMP:  find a symmetric permutation P to put P*A*P' into block upper
 *	triangular form.
 * BTF_ORDER: do both of the above (maxtrans then strongcomp).
 */


/* ========================================================================== */
/* === MAXTRANS ============================================================= */
/* ========================================================================== */

/* MAXTRANS: finds a permutation of the columns of a matrix so that it has a
 * zero-free diagonal.  The input is an n-by-n sparse matrix in compressed
 * column form.  The array Ap of size n+1 gives the starting and ending
 * positions of the columns in the array Ai.  Ap[0] must be zero. The array Ai
 * contains the row indices of the nonzeros of the matrix A, and is of size
 * Ap[n].  The row indices of column j are located in Ai[Ap[j] ... Ap[j+1]-1].
 * Row indices must be in the range 0 to n-1.  Duplicate entries may be present
 * in any given column.  The input matrix  is not checked for validity (row
 * indices out of the range 0 to n-1 will lead to an undeterminate result -
 * possibly a core dump, for example).  Row indices in any given column need
 * not be in sorted order.  However, if they are sorted and the matrix already
 * has a zero-free diagonal, then the identity permutation is returned.
 *
 * The output of maxtrans is an array Match of size n.  If row i is matched with
 * column j, then A(i,j) is nonzero, and then Match[i] = j.  If the matrix is
 * structurally nonzero, all entries in the Match array are unique.  The Match
 * array can also be interpreted as a column permutation.  That is, column k of
 * the original matrix becomes column Match[k] of the permuted matrix.  In
 * MATLAB, this can be expressed as (for non-structurally singular matrices):
 *
 *	Match = maxtrans (A) ;
 *	B = A (:, Match) ;
 *
 * except of course here the A matrix and Match vector are all 0-based (rows
 * and columns in the range 0 to n-1), not 1-based (rows/cols in range 1 to n).
 * The MATLAB dmperm routine returns a row permutation.  See the maxtrans
 * mexFunction for more details.
 *
 * If row i is not matched to any column, then Match[i] is < -1.  To find a
 * complete permutation can be returned, let j = (-(Match[i])-2) if
 * Match[i] < -1. The column index j can be computed using the macro
 *
 *	j = MAXTRANS_UNFLIP (Match [i]) ;
 *
 * Then row i can be considered to be matched to column j, except that the
 * A(i,j) entry is zero if Match[i] < 0.  If any such row i exists, the matrix
 * is structurally singular.  The maxtrans routine returns the number of
 * nonzeros on the permuted matrix.
 *
 * In the MATLAB mexFunction interface to maxtrans, 1 is added to the Match
 * array to obtain a 1-based permutation.  Thus, in MATLAB:
 *
 *	Match = maxtrans (A) ;	% has entries in the range 1:n and -(1:n)
 *	p = abs (Match) ;	% the permutation (either full rank or singular)
 *	B = A (:, p) ;		% permuted matrix (either full rank or singular)
 *	find (Match < 0) ;	% gives a list of zero diagonal entries of B
 *	sum (Match > 0) ;	% same as "sprank (A)"
 *
 * This behaviour differs from p = dmperm (A) in MATLAB, which returns p(i)=0
 * if row i is unmatched.  Thus:
 *
 *	p = dmperm (A) ;	% has entries in the range 0:n
 *	p                       % the permutation (only if full rank)
 *	B = A (:, p) ;		% permuted matrix (only if full rank)
 *	find (p == 0) ;		% gives a list of zero diagonal entries of B
 *	sum (p > 0) ;		% definition of sprank (A)
 *
 * This algorithm is based on the paper "On Algorithms for obtaining a maximum
 * transversal" by Iain Duff, ACM Trans. Mathematical Software, vol 7, no. 1,
 * pp. 315-330, and "Algorithm 575: Permutations for a zero-free diagonal",
 * same issue, pp. 387-390.  Algorithm 575 is MC21A in the Harwell Subroutine
 * Library.  This code is not merely a translation of the Fortran code into C.
 * It is a completely new implementation of the basic underlying method (depth
 * first search over a subgraph with nodes corresponding to columns matched so
 * far, and cheap matching).  This code was written with minimal observation of
 * the MC21A/B code itself.  See comments below for a comparison between the
 * maxtrans and MC21A/B codes.
 *
 * This routine operates on a column-form matrix and produces a column
 * permutation.  MC21A uses a row-form matrix and produces a row permutation.
 * The difference is merely one of convention in the comments and interpretation
 * of the inputs and outputs.  If you want a row permutation, simply pass a
 * compressed-row sparse matrix to this routine and you will get a row
 * permutation (just like MC21A).  Similarly, you can pass a column-oriented
 * matrix to MC21A and it will happily return a column permutation.
 *
 * Copyright (c) 2004.  Tim Davis, May 15, 2004, University of Florida,
 * with support from Sandia National Laboratories.  All Rights Reserved.
 *
 * TODO: test with duplicate entries to make sure it still works OK
 * TODO: extend to rectangular matrices (like dmperm).
 * TODO: check inputs and return -1 if the input matrix is invalid.
 */

#ifndef _BTF_H
#define _BTF_H

/* make it easy for C++ programs to include BTF */
#ifdef __cplusplus
extern "C" {
#endif

int maxtrans	    /* returns # of columns matched */
(
    /* input, not modified: */
    int nrow,	    /* A is nrow-by-ncol in compressed column form */
    int ncol,
    int Ap [ ],	    /* size ncol+1 */
    int Ai [ ],	    /* size nz = Ap [ncol] */

    /* output, not defined on input */
    int Match [ ],  /* size nrow.  Match [i] = j if column j matched to row i
		     * (see above for the singular-matrix case) */

    /* workspace, not defined on input or output */
    int Work [ ]    /* size 5*ncol */
) ;


/* ========================================================================== */
/* === MAXTRANS marking of singular columns ================================= */
/* ========================================================================== */


/* MAXTRANS_FLIP is a "negation about -1", and is used to mark an integer j
 * that is normally non-negative.  MAXTRANS_FLIP (-1) is -1.  MAXTRANS_FLIP of
 * a number > -1 is negative, and MAXTRANS_FLIP of a number < -1 is positive.
 * MAXTRANS_FLIP (MAXTRANS_FLIP (j)) = j for all integers j.  UNFLIP (j) acts
 * like an "absolute value" operation, and is always >= -1.  You can test
 * whether or not an integer j is "flipped" with the MAXTRANS_ISFLIPPED (j)
 * macro.
 */

#define MAXTRANS_FLIP(j) (-(j)-2)
#define MAXTRANS_ISFLIPPED(j) ((j) < -1)
#define MAXTRANS_UNFLIP(j) ((MAXTRANS_ISFLIPPED (j)) ? MAXTRANS_FLIP (j) : (j))

/* ========================================================================== */
/* === STRONGCOMP =========================================================== */
/* ========================================================================== */


int strongcomp	    /* return # of strongly connected components */
(
    /* input, not modified: */
    int n,	    /* A is n-by-n in compressed column form */
    int Ap [ ],	    /* size n+1 */
    int Ai [ ],	    /* size nz = Ap [n] */

    /* optional input, modified (if present) on output: */
    int Q [ ],	    /* size n, input column permutation */

    /* output, not defined on input */
    int P [ ],	    /* size n.  P [k] = j if row and column j are kth row/col
		     * in permuted matrix. */

    int R [ ],	    /* size n+1.  block b is in rows/cols R[b] ... R[b+1]-1 */

    /* workspace, not defined on input or output */
    int Work [ ]    /* size 4n */
) ;


/* ========================================================================== */
/* === BTF_ORDER ============================================================ */
/* ========================================================================== */


int btf_order	    /* returns number of blocks found */
(
    /* input, not modified: */
    int n,	    /* A is n-by-n in compressed column form */
    int Ap [ ],	    /* size n+1 */
    int Ai [ ],	    /* size nz = Ap [n] */

    /* output, not defined on input */
    int Q [ ],	    /* size n, column permutation */
    int P [ ],	    /* size n, row permutation */
    int R [ ],	    /* size n+1.  block b is in rows/cols R[b] ... R[b+1]-1 */
    int *nfound,    /* # nonzeros on diagonal of P*A*Q */

    /* workspace, not defined on input or output */
    int Work [ ]    /* size 5n */
) ;

/* ========================================================================== */
/* === BTF version ========================================================== */
/* ========================================================================== */

/* All versions of BTF include these definitions.
 * As an example, to test if the version you are using is 1.2 or later:
 *
 *	if (BTF_VERSION >= BTF_VERSION_CODE (1,2)) ...
 *
 * This also works during compile-time:
 *
 *	#if (BTF >= BTF_VERSION_CODE (1,2))
 *	    printf ("This is version 1.2 or later\n") ;
 *	#else
 *	    printf ("This is an early version\n") ;
 *	#endif
 */

#define BTF_DATE "May 5, 2006"
#define BTF_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define BTF_MAIN_VERSION 0
#define BTF_SUB_VERSION 9
#define BTF_VERSION BTF_VERSION_CODE(BTF_MAIN_VERSION,BTF_SUB_VERSION)

#ifdef __cplusplus
}
#endif


#endif
