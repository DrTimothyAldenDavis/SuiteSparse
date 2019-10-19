/* ========================================================================== */
/* === Check/cholmod_read =================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Check Module.  Version 1.2.  Copyright (C) 2005-2006,
 * Timothy A. Davis.
 * The CHOLMOD/Check Module is licensed under Version 2.1 of the GNU
 * Lesser General Public License.  See lesser.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

/* Read a sparse matrix in triplet form.  Compatible with all variations of
 * the Matrix Market "coord" format (http://www.nist.gov/MatrixMarket).
 *
 * If the first line of the file starts with %%MatrixMarket, then it is
 * interpretted as a file in Matrix Market format.  This line must have
 * the following format:
 *
 *	%%MatrixMarket matrix coord type storage
 *
 *	"type" is one of: real, complex, pattern, or integer.
 *	"storage" is one of: general, hermitian, symmetric, or skew-symmetric
 *
 *	The strings are case-insentive.  Only the first character is
 *	significant (or the first two for skew-symmetric).
 *
 *	"coord" can be replaced with "array" in the Matrix Market format, but
 *	this format not supported.  The "integer" type is converted to real.
 *	"type" is ignored; the actual type (real, complex, or pattern) is
 *	inferred from the number of tokens in each line of the file (2: pattern,
 *	3: real, 4: complex).  This is compatible with the Matrix Market format,
 *	since pattern matrices must have two tokens per line, real matrices
 *	must have 3, and complex matrices must have 4.  A storage of "general"
 *	implies an stype of zero (see below).  "symmetric" and "hermitian"
 *	imply an stype of -1. Skew-symmetric and complex symmetric matrices
 *	are returned with both upper and lower triangular parts present, with
 *	an stype of zero.
 *
 * Any other lines starting with "%" are treated as comments, and are ignored.
 * Blank lines are ignored.
 *
 * The first non-comment line contains 3 or 4 integers:
 *
 *	nrow ncol nnz stype
 *
 * where stype is optional (stype does not appear in the Matrix Market format).
 * The matrix is nrow-by-ncol.  The following nnz lines (excluding comments
 * and blank lines) each contain a single entry.  Duplicates are permitted,
 * and are summed in the output matrix.
 *
 * If stype is present, it denotes the storage format for the matrix.
 * stype = 0 denotes an unsymmetric matrix (same as Matrix Market "general").
 * stype = -1 denotes a symmetric or Hermitian matrix whose lower triangular 
 *	entries are stored.  Entries may be present in the upper triangular
 *	part, but these are ignored (same as Matrix Market "symmetric").
 * stype = 1 denotes a symmetric or Hermitian matrix whose upper triangular 
 *	entries are stored.  Entries may be present in the lower triangular
 *	part, but these are ignored.  This option is not in the Matrix Market
 *	format.
 *
 * If stype is not present it is inferred from the rest of the data (stype does
 * not appear in a Matrix Market file).  If the matrix is rectangular, or has
 * entries in both the upper and lower triangular parts, then it is assumed to
 * be unsymmetric (stype=0).  If only entries in the lower triangular part are
 * present, the matrix is assumed to have stype = -1.  If only entries in the
 * upper triangular part are present, the matrix is assumed to have stype = 1.
 *
 * Each nonzero consists of one line with 2, 3, or 4 entries.  All lines must
 * have the same number of entries.  The first two entries are the row and
 * column indices of the nonzero.  If 3 entries are present, the 3rd entry is
 * the numerical value, and the matrix is real.  If 4 entries are present,
 * the 3rd and 4th entries in the line are the real and imaginary parts of
 * a complex value.
 *
 * The matrix can be either 0-based or 1-based.  It is first assumed to be
 * one-based (compatible with Matrix Market), with row indices in the range
 * 1 to ncol and column indices in the range 1 to nrow.  If a row or column
 * index of zero is found, the matrix is assumed to be zero-based (with row
 * indices in the range 0 to ncol-1 and column indices in the range 0 to
 * nrow-1).
 *
 * For symmetric pattern-only matrices, the kth diagonal (if present) is set to
 * one plus the degree of the row/column k, and the off-diagonals are set to -1.
 * A symmetric pattern-only matrix with a zero-free diagonal is thus converted
 * into a symmetric positive definite matrix.  All entries are set to one for
 * an unsymmetric pattern-only matrix.  This differs from the MatrixMarket
 * format (A = mmread ('file') returns a binary pattern for A for symmetric
 * pattern-only matrices).
 */

#ifndef NCHECK

#include "cholmod_internal.h"
#include "cholmod_check.h"
#include <string.h>
#include <ctype.h>

#define MAXLINE 1024

/* ========================================================================== */
/* === is_blank_line ======================================================== */
/* ========================================================================== */

static int is_blank_line	/* TRUE if s is a blank line, FALSE otherwise */
(
    char *s
)
{
    int c, k ;
    for (k = 0 ; k <= MAXLINE ; k++)
    {
	c = s [k] ;
	if (c == '\0')
	{
	    /* end of line */
	    break ;
	}
	if (!isspace (c))
	{
	    /* non-space character */
	    return (FALSE) ;
	}
    }
    return (TRUE) ;
}


/* ========================================================================== */
/* === get_line ============================================================= */
/* ========================================================================== */

/* Get the next input line, discarding comments. */

static int get_line	/* returns # of items read, or -1 if error */
(
    FILE *f,		/* file to read from */
    long long *i,	/* row index */
    long long *j,	/* column index */
    double *x,		/* real part */
    double *z,		/* imaginary part */
    Int *stype)		/* stype, as determined from Matrix Market header,
			 * but with additional options for the skew-symmetric
	* and complex symmetric cases:
	* 1: symmetric, with upper part stored (not in Matrix Market format
	* 0: unsymmetric (Matrix Market "general")
	* -1: real symmetric or complex Hermitian, with lower part stored
	*	(Matrix Market "real symmetric" or "complex hermitian")
	* -2: real or complex skew symmetric
	* -3: complex symmetric
	*/
{
    char *p, s [MAXLINE+1] ;
    int c, c2, is_complex ;
    *i = 0 ;
    *j = 0 ;
    *x = 0 ;
    *z = 0 ;
    for ( ; ; )
    {
	s [0] = '\0' ;
	s [1] = '\0' ;
	s [MAXLINE] = '\0' ;
	if (fgets (s, MAXLINE, f) == NULL)
	{
	    /* end of file */
	    return (EMPTY) ;
	}
	if (s [0] == '%')
	{
	    /* a comment line */
	    if (strncmp (s, "%%MatrixMarket", 14) == 0)
	    {
		/* this is a Matrix Market header, with the format:
		 * %%MatrixMarket matrix coord type storage */
		p = s ;

		/* get "matrix" token */
		while (*p && !isspace (*p)) p++ ;
		while (*p &&  isspace (*p)) p++ ;
		c = tolower (*p) ;
		if (c != 'm')
		{
		    /* bad format */
		    return (EMPTY) ;
		}

		/* get "coord" token */
		while (*p && !isspace (*p)) p++ ;
		while (*p &&  isspace (*p)) p++ ;
		c = tolower (*p) ;
		if (c != 'c')
		{
		    /* bad format, only "coordinate" is supported;
		     * "array" is not supported */
		    return (EMPTY) ;
		}

		/* get type token (real, pattern, complex, integer) */
		while (*p && !isspace (*p)) p++ ;
		while (*p &&  isspace (*p)) p++ ;
		c = tolower (*p) ;
		if (!(c == 'r' || c == 'p' || c == 'c' || c == 'i'))
		{
		    /* bad format */
		    return (EMPTY) ;
		}

		is_complex = (c == 'c') ;

		/* get storage token (general, hermitian, symmetric, or
		 * skew-symmetric) */
		while (*p && !isspace (*p)) p++ ;
		while (*p &&  isspace (*p)) p++ ;
		c = tolower (*p) ;
		c2 = tolower (*(p+1)) ;
		if (c == 'g')
		{
		    /* "general" storage (unsymmetric matrix) */
		    *stype = 0 ;
		}
		else if (c == 's' && c2 == 'y')
		{
		    /* "symmetric" */
		    if (is_complex)
		    {
			/* complex symmetric, lower triangular part present */
			*stype = -3 ;
		    }
		    else
		    {
			/* real symmetric, lower triangular part present */
			*stype = -1 ;
		    }
		}
		else if (c == 'h')
		{
		    /* "hermitian" matrix, lower triangular part present */
		    *stype = -1 ;
		}
		else if (c == 's' && c2 == 'k')
		{
		    /* "skew-symmetric" (real or complex) */
		    *stype = -2 ;
		}
		else
		{
		    /* bad format */
		    return (EMPTY) ;
		}
	    }
	}
	else
	{
	    /* an entry, or a blank line */
	    if (is_blank_line (s))
	    {
		/* the line is blank, continue and get the next line */
		continue ;
	    }
	    /* this line contains an entry */
	    return (sscanf (s, "%lld %lld %lg %lg\n", i, j, x, z)) ;
	}
    }
}


/* ========================================================================== */
/* === cholmod_read_triplet ================================================= */
/* ========================================================================== */

cholmod_triplet *CHOLMOD(read_triplet)
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, assumed to already be open */
    /* --------------- */
    cholmod_common *Common
)
{
    double x, z ;
    double *Tx ;
    Int *Ti, *Tj, *Rdeg, *Cdeg ;
    cholmod_triplet *T ;
    long long l1, l2 ;
    Int nitems, stype, xtype, unknown, k, nshould, is_lower, is_upper,
	one_based, i, j, imax, jmax, ignore, skew_symmetric, p ;
    size_t s, nrow, ncol, nnz, nnz2, extra ;
    int ok = TRUE ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (f, NULL) ;
    Common->status = CHOLMOD_OK ;

    /* ---------------------------------------------------------------------- */
    /* read the header line */
    /* ---------------------------------------------------------------------- */

    stype = 999 ;
    nitems = get_line (f, &l1, &l2, &x, &z, &stype) ;
    nrow = l1 ;
    ncol = l2 ;
    nnz = x ;
    if (nrow != ncol)
    {
	stype = 0 ;
    }
    else if (nitems == 4)
    {
	/* first line contains: m n nnz stype */
	if (z < 0)
	{
	    stype = -1 ;
	}
	else if (z > 0)
	{
	    stype = 1 ;
	}
	else
	{
	    stype = 0 ;
	}
    }
    unknown = (stype == 999) ;

    if (x < 0 || l1 < 0 || l2 < 0 || nitems < 3)
    {
	/* invalid matrix */
	ERROR (CHOLMOD_INVALID, "invalid format") ;
	return (NULL) ;
    }

    if (nrow == 0 || ncol == 0 || nnz == 0)
    {
	/* return an empty matrix */
	return (CHOLMOD(allocate_triplet) (nrow, ncol, 0, 0, CHOLMOD_REAL,
		    Common)) ;
    }

    skew_symmetric = (stype == -2) ;
    extra = 0 ;
    if (stype < -1)
    {
	/* -2:  real or complex skew symmetric converted to unsymmetric */
	/* -3:  complex symmetric converted to unsymmetric */
	stype = 0 ;
	extra = nnz ;
    }
    nnz2 = CHOLMOD(add_size_t) (nnz, extra, &ok) ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    /* s = nrow + ncol */
    s = CHOLMOD(add_size_t) (nrow, ncol, &ok) ;
    if (!ok || nrow > Int_max || ncol > Int_max || nnz > Int_max)
    {
	ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
	return (NULL) ;
    }

    CHOLMOD(allocate_work) (0, s, 0, Common) ;
    Rdeg = Common->Iwork ;	/* size nrow */
    Cdeg = Rdeg + nrow ;	/* size ncol */

    /* ---------------------------------------------------------------------- */
    /* read the nonzero entries */
    /* ---------------------------------------------------------------------- */

    is_lower = TRUE ;
    is_upper = TRUE ;
    one_based = TRUE ;
    imax = 0 ;
    jmax = 0 ;

    Tx = NULL ;
    Ti = NULL ;
    Tj = NULL ;
    xtype = 999 ;
    nshould = 0 ;

    for (k = 0 ; k < (Int) nnz ; k++)
    {
	nitems = get_line (f, &l1, &l2, &x, &z, &ignore) ;

	i = l1 ;
	j = l2 ;

	/* ------------------------------------------------------------------ */
	/* for first entry: determine type and allocate triplet matrix */
	/* ------------------------------------------------------------------ */

	if (k == 0)
	{

	    if (nitems < 2 || nitems > 4)
	    {
		/* invalid matrix */
		ERROR (CHOLMOD_INVALID, "invalid format") ;
		return (NULL) ;
	    }
	    else if (nitems == 2)
	    {
		/* this will be converted into a real matrix later */
		xtype = CHOLMOD_PATTERN ;
	    }
	    else if (nitems == 3)
	    {
		xtype = CHOLMOD_REAL ;
	    }
	    else if (nitems == 4)
	    {
		xtype = CHOLMOD_COMPLEX ;
	    }

	    /* the rest of the lines should have the same number of entries */
	    nshould = nitems ;

	    T = CHOLMOD(allocate_triplet) (nrow, ncol, nnz2, stype,
		    (xtype == CHOLMOD_PATTERN ? CHOLMOD_REAL : xtype), Common) ;
	    if (Common->status < CHOLMOD_OK)
	    {
		/* out of memory */
		return (NULL) ;
	    }
	    Ti = T->i ;
	    Tj = T->j ;
	    Tx = T->x ;
	    T->nnz = nnz ;
	}

	/* ------------------------------------------------------------------ */
	/* save the entry in the triplet matrix */
	/* ------------------------------------------------------------------ */

	if (nitems != nshould || i < 0 || j < 0)
	{
	    /* wrong format, premature end-of-file, or negative indices */
	    CHOLMOD(free_triplet) (&T, Common) ;
	    ERROR (CHOLMOD_INVALID, "invalid matrix file") ;
	    return (NULL) ;
	}

	Ti [k] = i ;
	Tj [k] = j ;

	if (i < j)
	{
	    /* this entry is in the upper triangular part */
	    is_lower = FALSE ;
	}
	if (i > j)
	{
	    /* this entry is in the lower triangular part */
	    is_upper = FALSE ;
	}

	if (xtype == CHOLMOD_REAL)
	{
	    Tx [k] = x ;
	}
	else if (xtype == CHOLMOD_COMPLEX)
	{
	    Tx [2*k  ] = x ;	/* real part */
	    Tx [2*k+1] = z ;	/* imaginary part */
	}

	if (i == 0 || j == 0)
	{
	    one_based = FALSE ;
	}

	imax = MAX (i, imax) ;
	jmax = MAX (j, jmax) ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert to zero-based */
    /* ---------------------------------------------------------------------- */

    if (one_based)
    {
	/* input matrix is one-based; convert matrix to zero-based */
	for (k = 0 ; k < (Int) nnz ; k++)
	{
	    Ti [k]-- ;
	    Tj [k]-- ;
	}
    }

    if (one_based ?
	(imax >  (Int) nrow || jmax >  (Int) ncol) :
	(imax >= (Int) nrow || jmax >= (Int) ncol))
    {
	/* indices out of range */
	CHOLMOD(free_triplet) (&T, Common) ;
	ERROR (CHOLMOD_INVALID, "indices out of range") ;
	return (NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* add the remainder of skew-symmetric or complex symmetric matrices */
    /* ---------------------------------------------------------------------- */

    if (extra > 0)
    {
	p = nnz ;
	for (k = 0 ; k < (Int) nnz ; k++)
	{
	    i = Ti [k] ;
	    j = Tj [k] ;
	    if (i != j)
	    {
		Ti [p] = j ;
		Tj [p] = i ;
		if (xtype == CHOLMOD_REAL)
		{
		    Tx [p] = skew_symmetric ? (-Tx [k]) : (Tx [k]) ;
		}
		else if (xtype == CHOLMOD_COMPLEX)
		{
		    Tx [2*p  ] = skew_symmetric ? (-Tx [2*k  ]) : (Tx [2*k  ]);
		    Tx [2*p+1] = skew_symmetric ? (-Tx [2*k+1]) : (Tx [2*k+1]);
		}
		p++ ;
	    }
	}
	T->nnz = p ;
	nnz = p ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine the stype, if not yet known */
    /* ---------------------------------------------------------------------- */

    if (unknown)
    {
	if (is_lower && is_upper)
	{
	    /* diagonal matrix, symmetric with upper part present */
	    stype = 1 ;
	}
	else if (is_lower && !is_upper)
	{
	    /* symmetric, lower triangular part present */
	    stype = -1 ;
	}
	else if (!is_lower && is_upper)
	{
	    /* symmetric, upper triangular part present */
	    stype = 1 ;
	}
	else
	{
	    /* unsymmetric */
	    stype = 0 ;
	}
    }
    T->stype = stype ;

    /* ---------------------------------------------------------------------- */
    /* create values for a pattern-only matrix */
    /* ---------------------------------------------------------------------- */

    if (xtype == CHOLMOD_PATTERN)
    {
	if (stype == 0)
	{
	    /* unsymmetric case */
	    for (k = 0 ; k < (Int) nnz ; k++)
	    {
		Tx [k] = 1 ;
	    }
	}
	else
	{
	    /* compute the row and columm degrees (excluding the diagonal) */
	    for (i = 0 ; i < (Int) nrow ; i++)
	    {
		Rdeg [i] = 0 ;
	    }
	    for (j = 0 ; j < (Int) ncol ; j++)
	    {
		Cdeg [j] = 0 ;
	    }
	    for (k = 0 ; k < (Int) nnz ; k++)
	    {
		i = Ti [k] ;
		j = Tj [k] ;
		if ((stype < 0 && i > j) || (stype > 0 && i < j))
		{
		    /* both a(i,j) and a(j,i) appear in the matrix */
		    Rdeg [i]++ ;
		    Cdeg [j]++ ;
		    Rdeg [j]++ ;
		    Cdeg [i]++ ;
		}
	    }
	    /* assign the numerical values */
	    for (k = 0 ; k < (Int) nnz ; k++)
	    {
		i = Ti [k] ;
		j = Tj [k] ;
		Tx [k] = (i == j) ? (1 + MAX (Rdeg [i], Cdeg [j])) : (-1) ;
	    }
	}
    }
    return (T) ;
}


/* ========================================================================== */
/* === cholmod_read_sparse ================================================== */
/* ========================================================================== */

/* Read a sparse matrix from a file.  See cholmod_read_triplet for a discussion
 * of the file format.
 *
 * If Common->prefer_upper is TRUE (the default case), a symmetric matrix is
 * returned stored in upper-triangular form (A->stype == 1).
 */

cholmod_sparse *CHOLMOD(read_sparse)
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, assumed to already be open */
    /* --------------- */
    cholmod_common *Common
)
{
    cholmod_sparse *A, *A2 ;
    cholmod_triplet *T ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (f, NULL) ;
    Common->status = CHOLMOD_OK ;

    /* ---------------------------------------------------------------------- */
    /* convert to a sparse matrix in compressed-column form */
    /* ---------------------------------------------------------------------- */

    T = CHOLMOD(read_triplet) (f, Common) ;
    A = CHOLMOD(triplet_to_sparse) (T, 0, Common) ;
    CHOLMOD(free_triplet) (&T, Common) ;

    if (Common->prefer_upper && A != NULL && A->stype == -1)
    {
	/* A=A' */
	A2 = CHOLMOD(transpose) (A, 2, Common) ;
	CHOLMOD(free_sparse) (&A, Common) ;
	A = A2 ;
    }
    return (A) ;
}
#endif
