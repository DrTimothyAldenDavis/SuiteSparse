/* ========================================================================== */
/* === Cholesky/cholmod_solve =============================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Cholesky Module.  Version 1.2.  Copyright (C) 2005-2006,
 * Timothy A. Davis
 * The CHOLMOD/Cholesky Module is licensed under Version 2.1 of the GNU
 * Lesser General Public License.  See lesser.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

/* Solve one of the following systems:
 *
 *	Ax=b	    0: CHOLMOD_A	also applies the permutation L->Perm
 *	LDL'x=b	    1: CHOLMOD_LDLt	does not apply L->Perm
 *	LDx=b	    2: CHOLMOD_LD
 *	DL'x=b	    3: CHOLMOD_DLt
 *	Lx=b	    4: CHOLMOD_L
 *	L'x=b	    5: CHOLMOD_Lt
 *	Dx=b	    6: CHOLMOD_D
 *	x=Pb	    7: CHOLMOD_P	apply a permutation (P is L->Perm)
 *	x=P'b	    8: CHOLMOD_Pt	apply an inverse permutation
 *
 * The factorization can be simplicial LDL', simplicial LL', or supernodal LL'.
 * For an LL' factorization, D is the identity matrix.  Thus CHOLMOD_LD and
 * CHOLMOD_L solve the same system if an LL' factorization was performed,
 * for example.
 *
 * The supernodal solver uses BLAS routines dtrsv, dgemv, dtrsm, and dgemm,
 * or their complex counterparts ztrsv, zgemv, ztrsm, and zgemm.
 *
 * If both L and B are real, then X is returned real.  If either is complex
 * or zomplex, X is returned as either complex or zomplex, depending on the
 * Common->prefer_zomplex parameter.
 *
 * Supports any numeric xtype (pattern-only matrices not supported).
 *
 * This routine does not check to see if the diagonal of L or D is zero,
 * because sometimes a partial solve can be done with indefinite or singular
 * matrix.  If you wish to check in your own code, test L->minor.  If
 * L->minor == L->n, then the matrix has no zero diagonal entries.
 * If k = L->minor < L->n, then L(k,k) is zero for an LL' factorization, or
 * D(k,k) is zero for an LDL' factorization.
 *
 * This routine returns X as NULL only if it runs out of memory.  If L is
 * indefinite or singular, then X may contain Inf's or NaN's, but it will
 * exist on output.
 */

#ifndef NCHOLESKY

#include "cholmod_internal.h"
#include "cholmod_cholesky.h"

#ifndef NSUPERNODAL
#include "cholmod_supernodal.h"
#endif


/* ========================================================================== */
/* === TEMPLATE ============================================================= */
/* ========================================================================== */

#define REAL
#include "t_cholmod_solve.c"

#define COMPLEX
#include "t_cholmod_solve.c"

#define ZOMPLEX
#include "t_cholmod_solve.c"

/* ========================================================================== */
/* === Permutation macro ==================================================== */
/* ========================================================================== */

/* If Perm is NULL, it is interpretted as the identity permutation */

#define P(k) ((Perm == NULL) ? (k) : Perm [k])


/* ========================================================================== */
/* === perm ================================================================= */
/* ========================================================================== */

/* Y = B (P (1:nrow), k1 : min (k1+ncols,ncol)-1) where B is nrow-by-ncol.
 *
 * Creates a permuted copy of a contiguous set of columns of B.
 * Y is already allocated on input.  Y must be of sufficient size.  Let nk be
 * the number of columns accessed in B.  Y->xtype determines the complexity of
 * the result.
 *
 * If B is real and Y is complex (or zomplex), only the real part of B is
 * copied into Y.  The imaginary part of Y is set to zero.
 *
 * If B is complex (or zomplex) and Y is real, both the real and imaginary and
 * parts of B are returned in Y.  Y is returned as nrow-by-2*nk. The even
 * columns of Y contain the real part of B and the odd columns contain the
 * imaginary part of B.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
 * returned as nrow-by-nk with leading dimension nrow.  Y->nzmax must be >=
 * nrow*nk.
 *
 * The case where the input (B) is real and the output (Y) is zomplex is
 * not used.
 */

static void perm
(
    /* ---- input ---- */
    cholmod_dense *B,	/* input matrix B */
    Int *Perm,		/* optional input permutation (can be NULL) */
    Int k1,		/* first column of B to copy */
    Int ncols,		/* last column to copy is min(k1+ncols,B->ncol)-1 */
    /* ---- in/out --- */
    cholmod_dense *Y	/* output matrix Y, already allocated */
)
{
    double *Yx, *Yz, *Bx, *Bz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dual, dj, j2 ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    ncol = B->ncol ;
    nrow = B->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    dual = (Y->xtype == CHOLMOD_REAL && B->xtype != CHOLMOD_REAL) ? 2 : 1 ;
    d = B->d ;
    Bx = B->x ;
    Bz = B->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    Y->nrow = nrow ;
    Y->ncol = dual*nk ;
    Y->d = nrow ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*dual) ;

    /* ---------------------------------------------------------------------- */
    /* Y = B (P (1:nrow), k1:k2-1) */
    /* ---------------------------------------------------------------------- */

    switch (Y->xtype)
    {

	case CHOLMOD_REAL:

	    switch (B->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y real, B real */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [k + j2] = Bx [p] ;		/* real */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y real, B complex. Y is nrow-by-2*nk */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [k + j2       ] = Bx [2*p  ] ;	/* real */
			    Yx [k + j2 + nrow] = Bx [2*p+1] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y real, B zomplex. Y is nrow-by-2*nk */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [k + j2       ] = Bx [p] ;	/* real */
			    Yx [k + j2 + nrow] = Bz [p] ;	/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_COMPLEX:

	    switch (B->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y complex, B real */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [2*k   + j2] = Bx [p] ;		/* real */
			    Yx [2*k+1 + j2] = 0 ;		/* imag */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y complex, B complex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [2*k   + j2] = Bx [2*p  ] ;	/* real */
			    Yx [2*k+1 + j2] = Bx [2*p+1] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y complex, B zomplex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [2*k   + j2] = Bx [p] ;		/* real */
			    Yx [2*k+1 + j2] = Bz [p] ;		/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_ZOMPLEX:

	    switch (B->xtype)
	    {

		case CHOLMOD_REAL:
		    /* this case is not used */
		    ASSERT (0) ;
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y zomplex, B complex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [k + j2] = Bx [2*p  ] ;		/* real */
			    Yz [k + j2] = Bx [2*p+1] ;		/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y zomplex, B zomplex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [k + j2] = Bx [p] ;		/* real */
			    Yz [k + j2] = Bz [p] ;		/* imag */
			}
		    }
		    break ;

	    }
	    break ;

    }
}


/* ========================================================================== */
/* === iperm ================================================================ */
/* ========================================================================== */

/* X (P (1:nrow), k1 : min (k1+ncols,ncol)-1) = Y where X is nrow-by-ncol.
 *
 * Copies and permutes Y into a contiguous set of columns of X.  X is already
 * allocated on input.  Y must be of sufficient size.  Let nk be the number
 * of columns accessed in X.  X->xtype determines the complexity of the result.
 *
 * If X is real and Y is complex (or zomplex), only the real part of B is
 * copied into X.  The imaginary part of Y is ignored.
 *
 * If X is complex (or zomplex) and Y is real, both the real and imaginary and
 * parts of Y are returned in X.  Y is nrow-by-2*nk. The even
 * columns of Y contain the real part of B and the odd columns contain the
 * imaginary part of B.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
 * nrow-by-nk with leading dimension nrow.  Y->nzmax must be >= nrow*nk.
 *
 * The case where the input (Y) is complex and the output (X) is real,
 * and the case where the input (Y) is zomplex and the output (X) is real,
 * are not used.
 */

static void iperm
(
    /* ---- input ---- */
    cholmod_dense *Y,	/* input matrix Y */
    Int *Perm,		/* optional input permutation (can be NULL) */
    Int k1,		/* first column of B to copy */
    Int ncols,		/* last column to copy is min(k1+ncols,B->ncol)-1 */
    /* ---- in/out --- */
    cholmod_dense *X	/* output matrix X, already allocated */
)
{
    double *Yx, *Yz, *Xx, *Xz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dj, j2 ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    ncol = X->ncol ;
    nrow = X->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    d = X->d ;
    Xx = X->x ;
    Xz = X->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*
	    ((X->xtype != CHOLMOD_REAL && Y->xtype == CHOLMOD_REAL) ? 2:1)) ;

    /* ---------------------------------------------------------------------- */
    /* X (P (1:nrow), k1:k2-1) = Y */
    /* ---------------------------------------------------------------------- */

    switch (Y->xtype)
    {

	case CHOLMOD_REAL:

	    switch (X->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y real, X real */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [k + j2] ;		/* real */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y real, X complex. Y is nrow-by-2*nk */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [2*p  ] = Yx [k + j2       ] ;	/* real */
			    Xx [2*p+1] = Yx [k + j2 + nrow] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y real, X zomplex. Y is nrow-by-2*nk */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [k + j2       ] ;	/* real */
			    Xz [p] = Yx [k + j2 + nrow] ;	/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_COMPLEX:

	    switch (X->xtype)
	    {

		case CHOLMOD_REAL:
		    /* this case is not used */
		    ASSERT (0) ;
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y complex, X complex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [2*p  ] = Yx [2*k   + j2] ;	/* real */
			    Xx [2*p+1] = Yx [2*k+1 + j2] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y complex, X zomplex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * 2 * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [2*k   + j2] ;		/* real */
			    Xz [p] = Yx [2*k+1 + j2] ;		/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_ZOMPLEX:

	    switch (X->xtype)
	    {

		case CHOLMOD_REAL:
		    /* this case is not used */
		    ASSERT (0) ;
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y zomplex, X complex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [2*p  ] = Yx [k + j2] ;		/* real */
			    Xx [2*p+1] = Yz [k + j2] ;		/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y zomplex, X zomplex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = nrow * (j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [k + j2] ;		/* real */
			    Xz [p] = Yz [k + j2] ;		/* imag */
			}
		    }
		    break ;

	    }
	    break ;

    }
}


/* ========================================================================== */
/* === ptrans =============================================================== */
/* ========================================================================== */

/* Y = B (P (1:nrow), k1 : min (k1+ncols,ncol)-1)' where B is nrow-by-ncol.
 *
 * Creates a permuted and transposed copy of a contiguous set of columns of B.
 * Y is already allocated on input.  Y must be of sufficient size.  Let nk be
 * the number of columns accessed in B.  Y->xtype determines the complexity of
 * the result.
 *
 * If B is real and Y is complex (or zomplex), only the real part of B is
 * copied into Y.  The imaginary part of Y is set to zero.
 *
 * If B is complex (or zomplex) and Y is real, both the real and imaginary and
 * parts of B are returned in Y.  Y is returned as 2*nk-by-nrow. The even
 * rows of Y contain the real part of B and the odd rows contain the
 * imaginary part of B.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
 * returned as nk-by-nrow with leading dimension nk.  Y->nzmax must be >=
 * nrow*nk.
 *
 * The array transpose is performed, not the complex conjugate transpose.
 */

static void ptrans
(
    /* ---- input ---- */
    cholmod_dense *B,	/* input matrix B */
    Int *Perm,		/* optional input permutation (can be NULL) */
    Int k1,		/* first column of B to copy */
    Int ncols,		/* last column to copy is min(k1+ncols,B->ncol)-1 */
    /* ---- in/out --- */
    cholmod_dense *Y	/* output matrix Y, already allocated */
)
{
    double *Yx, *Yz, *Bx, *Bz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dual, dj, j2 ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    ncol = B->ncol ;
    nrow = B->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    dual = (Y->xtype == CHOLMOD_REAL && B->xtype != CHOLMOD_REAL) ? 2 : 1 ;
    d = B->d ;
    Bx = B->x ;
    Bz = B->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    Y->nrow = dual*nk ;
    Y->ncol = nrow ;
    Y->d = dual*nk ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*dual) ;

    /* ---------------------------------------------------------------------- */
    /* Y = B (P (1:nrow), k1:k2-1)' */
    /* ---------------------------------------------------------------------- */

    switch (Y->xtype)
    {

	case CHOLMOD_REAL:

	    switch (B->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y real, B real  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2 + k*nk] = Bx [p] ;		/* real */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y real, B complex. Y is 2*nk-by-nrow */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2   + k*2*nk] = Bx [2*p  ] ;	/* real */
			    Yx [j2+1 + k*2*nk] = Bx [2*p+1] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y real, B zomplex. Y is 2*nk-by-nrow */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2   + k*2*nk] = Bx [p] ;	/* real */
			    Yx [j2+1 + k*2*nk] = Bz [p] ;	/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_COMPLEX:

	    switch (B->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y complex, B real  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2   + k*2*nk] = Bx [p] ;	/* real */
			    Yx [j2+1 + k*2*nk] = 0 ;		/* imag */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y complex, B complex  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2   + k*2*nk] = Bx [2*p  ] ;	/* real */
			    Yx [j2+1 + k*2*nk] = Bx [2*p+1] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y complex, B zomplex  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2   + k*2*nk] = Bx [p] ;	/* real */
			    Yx [j2+1 + k*2*nk] = Bz [p] ;	/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_ZOMPLEX:

	    switch (B->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y zomplex, B real  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2 + k*nk] = Bx [p] ;		/* real */
			    Yz [j2 + k*nk] = 0 ;		/* imag */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y zomplex, B complex  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2 + k*nk] = Bx [2*p  ] ;	/* real */
			    Yz [j2 + k*nk] = Bx [2*p+1] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y zomplex, B zomplex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Yx [j2 + k*nk] = Bx [p] ;		/* real */
			    Yz [j2 + k*nk] = Bz [p] ;		/* imag */
			}
		    }
		    break ;

	    }
	    break ;

    }
}


/* ========================================================================== */
/* === iptrans ============================================================== */
/* ========================================================================== */

/* X (P (1:nrow), k1 : min (k1+ncols,ncol)-1) = Y' where X is nrow-by-ncol.
 *
 * Copies into a permuted and transposed contiguous set of columns of X.
 * X is already allocated on input.  Y must be of sufficient size.  Let nk be
 * the number of columns accessed in X.  X->xtype determines the complexity of
 * the result.
 *
 * If X is real and Y is complex (or zomplex), only the real part of Y is
 * copied into X.  The imaginary part of Y is ignored.
 *
 * If X is complex (or zomplex) and Y is real, both the real and imaginary and
 * parts of X are returned in Y.  Y is 2*nk-by-nrow. The even
 * rows of Y contain the real part of X and the odd rows contain the
 * imaginary part of X.  Y->nzmax must be >= 2*nrow*nk.  Otherise, Y is
 * nk-by-nrow with leading dimension nk.  Y->nzmax must be >= nrow*nk.
 *
 * The case where Y is complex or zomplex, and X is real, is not used.
 *
 * The array transpose is performed, not the complex conjugate transpose.
 */

static void iptrans
(
    /* ---- input ---- */
    cholmod_dense *Y,	/* input matrix Y */
    Int *Perm,		/* optional input permutation (can be NULL) */
    Int k1,		/* first column of X to copy into */
    Int ncols,		/* last column to copy is min(k1+ncols,X->ncol)-1 */
    /* ---- in/out --- */
    cholmod_dense *X	/* output matrix X, already allocated */
)
{
    double *Yx, *Yz, *Xx, *Xz ;
    Int k2, nk, p, k, j, nrow, ncol, d, dj, j2 ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    ncol = X->ncol ;
    nrow = X->nrow ;
    k2 = MIN (k1+ncols, ncol) ;
    nk = MAX (k2 - k1, 0) ;
    d = X->d ;
    Xx = X->x ;
    Xz = X->z ;
    Yx = Y->x ;
    Yz = Y->z ;
    ASSERT (((Int) Y->nzmax) >= nrow*nk*
	    ((X->xtype != CHOLMOD_REAL && Y->xtype == CHOLMOD_REAL) ? 2:1)) ;

    /* ---------------------------------------------------------------------- */
    /* X (P (1:nrow), k1:k2-1) = Y' */
    /* ---------------------------------------------------------------------- */

    switch (Y->xtype)
    {

	case CHOLMOD_REAL:

	    switch (X->xtype)
	    {

		case CHOLMOD_REAL:
		    /* Y real, X real  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [j2 + k*nk] ;		/* real */
			}
		    }
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y real, X complex. Y is 2*nk-by-nrow */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [2*p  ] = Yx [j2   + k*2*nk] ;	/* real */
			    Xx [2*p+1] = Yx [j2+1 + k*2*nk] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y real, X zomplex. Y is 2*nk-by-nrow */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [j2   + k*2*nk] ;	/* real */
			    Xz [p] = Yx [j2+1 + k*2*nk] ;	/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_COMPLEX:

	    switch (X->xtype)
	    {

		case CHOLMOD_REAL:
		    /* this case is not used */
		    ASSERT (0) ;
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y complex, X complex  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [2*p  ] = Yx [j2   + k*2*nk] ;	/* real */
			    Xx [2*p+1] = Yx [j2+1 + k*2*nk] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y complex, X zomplex  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = 2*(j-k1) ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [j2   + k*2*nk] ;	/* real */
			    Xz [p] = Yx [j2+1 + k*2*nk] ;	/* imag */
			}
		    }
		    break ;

	    }
	    break ;

	case CHOLMOD_ZOMPLEX:

	    switch (X->xtype)
	    {

		case CHOLMOD_REAL:
		    /* this case is not used */
		    ASSERT (0) ;
		    break ;

		case CHOLMOD_COMPLEX:
		    /* Y zomplex, X complex  */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [2*p  ] = Yx [j2 + k*nk] ;	/* real */
			    Xx [2*p+1] = Yz [j2 + k*nk] ;	/* imag */
			}
		    }
		    break ;

		case CHOLMOD_ZOMPLEX:
		    /* Y zomplex, X zomplex */
		    for (j = k1 ; j < k2 ; j++)
		    {
			dj = d*j ;
			j2 = j-k1 ;
			for (k = 0 ; k < nrow ; k++)
			{
			    p = P(k) + dj ;
			    Xx [p] = Yx [j2 + k*nk] ;		/* real */
			    Xz [p] = Yz [j2 + k*nk] ;		/* imag */
			}
		    }
		    break ;

	    }
	    break ;

    }
}


/* ========================================================================== */
/* === cholmod_solve ======================================================== */
/* ========================================================================== */

/* Solve a linear system.
 *
 * The factorization can be simplicial LDL', simplicial LL', or supernodal LL'.
 * The Dx=b solve returns silently for the LL' factorizations (it is implicitly
 * identity).
 */

cholmod_dense *CHOLMOD(solve)
(
    /* ---- input ---- */
    int sys,		/* system to solve */
    cholmod_factor *L,	/* factorization to use */
    cholmod_dense *B,	/* right-hand-side */
    /* --------------- */
    cholmod_common *Common
)
{
    cholmod_dense *Y = NULL, *X = NULL ;
    Int *Perm ;
    Int n, nrhs, ncols, ctype, xtype, k1, nr, ytype ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (L, NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, NULL) ;
    RETURN_IF_XTYPE_INVALID (B, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, NULL) ;
    if (sys < CHOLMOD_A || sys > CHOLMOD_Pt)
    {
	ERROR (CHOLMOD_INVALID, "invalid system") ;
	return (NULL) ;
    }
    if (B->d < L->n || B->nrow != L->n)
    {
	ERROR (CHOLMOD_INVALID, "dimensions of L and B do not match") ;
	return (NULL) ;
    }
    DEBUG (CHOLMOD(dump_factor) (L, "L", Common)) ;
    DEBUG (CHOLMOD(dump_dense) (B, "B", Common)) ;
    Common->status = CHOLMOD_OK ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if ((sys == CHOLMOD_P || sys == CHOLMOD_Pt || sys == CHOLMOD_A)
	    && L->ordering != CHOLMOD_NATURAL)
    {
	Perm = L->Perm ;
    }
    else
    {
	/* do not use L->Perm; use the identity permutation instead */
	Perm = NULL ;
    }

    nrhs = B->ncol ;
    n = L->n ;

    /* ---------------------------------------------------------------------- */
    /* allocate the result X */
    /* ---------------------------------------------------------------------- */

    ctype = (Common->prefer_zomplex) ? CHOLMOD_ZOMPLEX : CHOLMOD_COMPLEX ;

    if (sys == CHOLMOD_P || sys == CHOLMOD_Pt)
    {
	/* x=Pb and x=P'b return X real if B is real; X is the preferred
	 * complex/zcomplex type if B is complex or zomplex */
	xtype = (B->xtype == CHOLMOD_REAL) ? CHOLMOD_REAL : ctype ;
    }
    else if (L->xtype == CHOLMOD_REAL && B->xtype == CHOLMOD_REAL)
    {
	/* X is real if both L and B are real */
	xtype = CHOLMOD_REAL ;
    }
    else
    {
	/* X is complex, use the preferred complex/zomplex type */
	xtype = ctype ;
    }

    X = CHOLMOD(allocate_dense) (n, nrhs, n, xtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
	return (NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve using L, D, L', P, or some combination */
    /* ---------------------------------------------------------------------- */

    if (sys == CHOLMOD_P)
    {

	/* ------------------------------------------------------------------ */
	/* x = P*b */
	/* ------------------------------------------------------------------ */

	perm (B, Perm, 0, nrhs, X) ;

    }
    else if (sys == CHOLMOD_Pt)
    {

	/* ------------------------------------------------------------------ */
	/* x = P'*b */
	/* ------------------------------------------------------------------ */

	iperm (B, Perm, 0, nrhs, X) ;

    }
    else if (L->is_super)
    {

	/* ------------------------------------------------------------------ */
	/* solve using a supernodal LL' factorization */
	/* ------------------------------------------------------------------ */

#ifndef NSUPERNODAL
	Int ok ;

	/* allocate workspace */
	cholmod_dense *E ;
	Int dual ;
	dual = (L->xtype == CHOLMOD_REAL && B->xtype != CHOLMOD_REAL) ? 2 : 1 ;
	Y = CHOLMOD(allocate_dense) (n, dual*nrhs, n, L->xtype, Common) ;
	E = CHOLMOD(allocate_dense) (dual*nrhs, L->maxesize, dual*nrhs,
		L->xtype, Common) ;

	if (Common->status < CHOLMOD_OK)
	{
	    /* out of memory */
	    CHOLMOD(free_dense) (&X, Common) ;
	    CHOLMOD(free_dense) (&Y, Common) ;
	    CHOLMOD(free_dense) (&E, Common) ;
	    return (NULL) ;
	}

	perm (B, Perm, 0, nrhs, Y) ;			    /* Y = P*B */

	if (sys == CHOLMOD_A || sys == CHOLMOD_LDLt)
	{
	    ok = CHOLMOD(super_lsolve) (L, Y, E, Common) ;	   /* Y = L\Y */
	    ok = ok && CHOLMOD(super_ltsolve) (L, Y, E, Common) ;  /* Y = L'\Y*/
	}
	else if (sys == CHOLMOD_L || sys == CHOLMOD_LD)
	{
	    ok = CHOLMOD(super_lsolve) (L, Y, E, Common) ;	   /* Y = L\Y */
	}
	else if (sys == CHOLMOD_Lt || sys == CHOLMOD_DLt)
	{
	    ok = CHOLMOD(super_ltsolve) (L, Y, E, Common) ;	   /* Y = L'\Y*/
	}
	CHOLMOD(free_dense) (&E, Common) ;

	iperm (Y, Perm, 0, nrhs, X) ;			    /* X = P'*Y */

	if (CHECK_BLAS_INT && !ok)
	{
	    /* integer overflow in the BLAS */
	    CHOLMOD(free_dense) (&X, Common) ;
	}

#else
	/* CHOLMOD Supernodal module not installed */
	ERROR (CHOLMOD_NOT_INSTALLED,"Supernodal module not installed") ;
#endif

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* solve using a simplicial LL' or LDL' factorization */
	/* ------------------------------------------------------------------ */

	if (L->xtype == CHOLMOD_REAL && B->xtype == CHOLMOD_REAL)
	{
	    /* L, B, and Y are all real */
	    /* solve with up to 4 columns of B at a time */
	    ncols = 4 ;
	    nr = MAX (4, nrhs) ;
	    ytype = CHOLMOD_REAL ;
	}
	else if (L->xtype == CHOLMOD_REAL)
	{
	    /* solve with one column of B (real/imag), at a time */
	    ncols = 1 ;
	    nr = 2 ;
	    ytype = CHOLMOD_REAL ;
	}
	else
	{
	    /* L is complex or zomplex, B is real/complex/zomplex, Y has the
	     * same complexity as L.  Solve with one column of B at a time. */
	    ncols = 1 ;
	    nr = 1 ;
	    ytype = L->xtype ;
	}

	Y = CHOLMOD(allocate_dense) (nr, n, nr, ytype, Common) ;

	if (Common->status < CHOLMOD_OK)
	{
	    /* out of memory */
	    CHOLMOD(free_dense) (&X, Common) ;
	    CHOLMOD(free_dense) (&Y, Common) ;
	    return (NULL) ;
	}

	for (k1 = 0 ; k1 < nrhs ; k1 += ncols)
	{
	    /* -------------------------------------------------------------- */
	    /* Y = B (P, k1:k1+ncols-1)' = (P * B (:,...))' */
	    /* -------------------------------------------------------------- */

	    ptrans (B, Perm, k1, ncols, Y) ;

	    /* -------------------------------------------------------------- */
	    /* solve Y = (L' \ (L \ Y'))', or other system, with template */
	    /* -------------------------------------------------------------- */

	    switch (L->xtype)
	    {
		case CHOLMOD_REAL:
		    r_simplicial_solver (sys, L, Y) ;
		    break ;

		case CHOLMOD_COMPLEX:
		    c_simplicial_solver (sys, L, Y) ;
		    break ;

		case CHOLMOD_ZOMPLEX:
		    z_simplicial_solver (sys, L, Y) ;
		    break ;
	    }

	    /* -------------------------------------------------------------- */
	    /* X (P, k1:k2+ncols-1) = Y' */
	    /* -------------------------------------------------------------- */

	    iptrans (Y, Perm, k1, ncols, X) ;
	}
    }

    CHOLMOD(free_dense) (&Y, Common) ;
    DEBUG (CHOLMOD(dump_dense) (X, "X result", Common)) ;
    return (X) ;
}
#endif
