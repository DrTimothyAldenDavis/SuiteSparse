//------------------------------------------------------------------------------
// UMFPACK/Source/umf_internal.h: internal definitions for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    This file is for internal use in UMFPACK itself, and should not be included
    in user code.  Use umfpack.h instead.  User-accessible file names and
    routine names all start with the letters "umfpack_".  Non-user-accessible
    file names and routine names all start with "umf_".
*/

#ifndef _UMF_INTERNAL
#define _UMF_INTERNAL

/* -------------------------------------------------------------------------- */
/* ANSI standard include files */
/* -------------------------------------------------------------------------- */

/* from float.h:  DBL_EPSILON */
#include <float.h>

/* from string.h: strcmp */
#include <string.h>

/* when debugging, assert.h and the assert macro are used (see umf_dump.h) */

/* -------------------------------------------------------------------------- */
/* Architecture */
/* -------------------------------------------------------------------------- */

#if defined (__sun) || defined (MSOL2) || defined (ARCH_SOL2)
#define UMF_SOL2
#define UMFPACK_ARCHITECTURE "Sun Solaris"

#elif defined (__sgi) || defined (MSGI) || defined (ARCH_SGI)
#define UMF_SGI
#define UMFPACK_ARCHITECTURE "SGI Irix"

#elif defined (__linux) || defined (MGLNX86) || defined (ARCH_GLNX86)
#define UMF_LINUX
#define UMFPACK_ARCHITECTURE "Linux"

#elif defined (__APPLE__)
#define UMF_MAC
#define UMFPACK_ARCHITECTURE "Mac"

#elif defined (_AIX) || defined (MIBM_RS) || defined (ARCH_IBM_RS)
#define UMF_AIX
#define UMFPACK_ARCHITECTURE "IBM AIX"

#elif defined (__alpha) || defined (MALPHA) || defined (ARCH_ALPHA)
#define UMF_ALPHA
#define UMFPACK_ARCHITECTURE "Compaq Alpha"

#elif defined (_WIN32) || defined (WIN32)
#if defined (__MINGW32__)
#define UMF_MINGW
#elif defined (__CYGWIN32__)
#define UMF_CYGWIN
#else
#define UMF_WINDOWS
#endif
#define UMFPACK_ARCHITECTURE "Microsoft Windows"

#elif defined (__hppa) || defined (__hpux) || defined (MHPUX) || defined (ARCH_HPUX)
#define UMF_HP
#define UMFPACK_ARCHITECTURE "HP Unix"

#elif defined (__hp700) || defined (MHP700) || defined (ARCH_HP700)
#define UMF_HP
#define UMFPACK_ARCHITECTURE "HP 700 Unix"

#else
#define UMFPACK_ARCHITECTURE "unknown"
#endif


//------------------------------------------------------------------------------
// debug
//------------------------------------------------------------------------------

// force debugging off
#ifndef NDEBUG
#define NDEBUG
#endif

// To enable debugging, uncomment the following line:
// #undef NDEBUG

//------------------------------------------------------------------------------
// AMD and SuiteSparse_config
//------------------------------------------------------------------------------

#define SUITESPARSE_BLAS_DEFINITIONS
#include "amd.h"
#include "SuiteSparse_config.h"

/* ------------------------------------------------------------------------- */
/* basic definitions */
/* ------------------------------------------------------------------------- */

#define ONES_COMPLEMENT(r) (-(r)-1)

#ifdef FLIP
#undef FLIP
#endif

#ifdef MAX
#undef MAX
#endif

#ifdef MIN
#undef MIN
#endif

#ifdef EMPTY
#undef EMPTY
#endif

#define PRIVATE static

/* FLIP is a "negation about -1", and is used to mark an integer i that is
 * normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
 * is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
 * for all integers i.  UNFLIP (i) is >= EMPTY. */
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) ((i < EMPTY) ? FLIP (i) : (i))

/* for integer MAX/MIN, or for doubles when we don't care how NaN's behave: */
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

/* logical expression of p implies q: */
#define IMPLIES(p,q) (!(p) || (q))

/* Note that the IBM RS 6000 xlc predefines TRUE and FALSE in <types.h>. */
/* The Compaq Alpha also predefines TRUE and FALSE. */
#ifdef TRUE
#undef TRUE
#endif
#ifdef FALSE
#undef FALSE
#endif

#define TRUE (1)
#define FALSE (0)
#define EMPTY (-1)

/* largest value of size_t */
#ifndef SIZE_T_MAX
#ifdef SIZE_MAX
/* C99 only */
#define SIZE_T_MAX SIZE_MAX
#else
#define SIZE_T_MAX ((size_t) (-1))
#endif
#endif

#if defined (HAVE_PRAGMA_GCC_IVDEP)
#  define UMFPACK_IVDEP _Pragma("GCC ivdep")
#elif defined (HAVE_PRAGMA_CLANG_LOOP_VECTORIZE)
#  define UMFPACK_IVDEP _Pragma("clang loop vectorize(enable)")
#elif defined (HAVE_PRAGMA_IVDEP)
#  define UMFPACK_IVDEP _Pragma("ivdep")
#elif defined (HAVE_PRAGMA_LOOP_IVDEP)
#  define UMFPACK_IVDEP _Pragma("loop( ivdep )")
#else
#  define UMFPACK_IVDEP
#endif

#if defined (HAVE_PRAGMA_GCC_NOVECTOR)
#  define UMFPACK_NOVECTOR _Pragma("GCC novector")
#elif defined (HAVE_PRAGMA_CLANG_LOOP_VECTORIZE)
#  define UMFPACK_NOVECTOR _Pragma("clang loop vectorize(disable)")
#elif defined (HAVE_PRAGMA_NOVECTOR)
#  define UMFPACK_NOVECTOR _Pragma("novector")
#elif defined (HAVE_PRAGMA_LOOP_NO_VECTOR)
#  define UMFPACK_NOVECTOR _Pragma("loop( no_vector )")
#else
#  define UMFPACK_NOVECTOR
#endif

/* ------------------------------------------------------------------------- */
/* integer type for AMD: int32_t or int64_t */
/* ------------------------------------------------------------------------- */

#if defined (DLONG) || defined (ZLONG)

    #define Int int64_t
    #define UInt uint64_t
    #define ID  "%" PRId64
    #define Int_MAX INT64_MAX

    #define AMD_order amd_l_order
    #define AMD_defaults amd_l_defaults
    #define AMD_control amd_l_control
    #define AMD_info amd_l_info
    #define AMD_1 amd_l1
    #define AMD_2 amd_l2
    #define AMD_valid amd_l_valid
    #define AMD_aat amd_l_aat
    #define AMD_postorder amd_l_postorder
    #define AMD_post_tree amd_l_post_tree
    #define AMD_debug_init amd_l_debug_init
    #define AMD_preprocess amd_l_preprocess

#else

    #define Int int32_t
    #define UInt uint32_t
    #define ID "%d"
    #define Int_MAX INT32_MAX

    #define AMD_order amd_order
    #define AMD_defaults amd_defaults
    #define AMD_control amd_control
    #define AMD_info amd_info
    #define AMD_1 amd_1
    #define AMD_2 amd_2
    #define AMD_valid amd_valid
    #define AMD_aat amd_aat
    #define AMD_postorder amd_postorder
    #define AMD_post_tree amd_post_tree
    #define AMD_debug_init amd_debug_init
    #define AMD_preprocess amd_preprocess

#endif

/* ------------------------------------------------------------------------- */
/* AMD routine definitions (not user-callable) */
/* ------------------------------------------------------------------------- */

size_t AMD_aat
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int Len [ ],
    Int Tp [ ],
    double Info [ ]
) ;

void AMD_1
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int P [ ],
    Int Pinv [ ],
    Int Len [ ],
    Int slen,
    Int S [ ],
    double Control [ ],
    double Info [ ]
) ;

void AMD_postorder
(
    Int nn,
    Int Parent [ ],
    Int Npiv [ ],
    Int Fsize [ ],
    Int Order [ ],
    Int Child [ ],
    Int Sibling [ ],
    Int Stack [ ]
) ;

Int AMD_post_tree
(
    Int root,
    Int k,
    Int Child [ ],
    const Int Sibling [ ],
    Int Order [ ],
    Int Stack [ ]
#ifndef NDEBUG
    , Int nn
#endif
) ;

void AMD_preprocess
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int Rp [ ],
    Int Ri [ ],
    Int W [ ],
    Int Flag [ ]
) ;

/* ------------------------------------------------------------------------- */
/* debugging definitions */
/* ------------------------------------------------------------------------- */

#ifndef NDEBUG

/* from assert.h:  assert macro */
#include <assert.h>

void AMD_debug_init ( char *s ) ;

#ifdef ASSERT
#undef ASSERT
#endif

/* Use mxAssert if AMD is compiled into a mexFunction */
#ifdef MATLAB_MEX_FILE
#define ASSERT(expression) (mxAssert ((expression), ""))
#else
#define ASSERT(expression) (assert (expression))
#endif

#else

/* no debugging */
#define ASSERT(expression)

#endif

/* -------------------------------------------------------------------------- */
/* MATLAB include files */
/* -------------------------------------------------------------------------- */

/* only used when compiling the UMFPACK mexFunction */
#ifdef MATLAB_MEX_FILE
#include "matrix.h"
#include "mex.h"
#endif

/* -------------------------------------------------------------------------- */
/* Real/complex and int32_t/int64_t definitions, double relops */
/* -------------------------------------------------------------------------- */

#include "umf_version.h"

/* -------------------------------------------------------------------------- */
/* Compile-time configurations */
/* -------------------------------------------------------------------------- */

#include "umf_config.h"

/* -------------------------------------------------------------------------- */
/* umfpack include file */
/* -------------------------------------------------------------------------- */

#include "umfpack.h"

/* -------------------------------------------------------------------------- */
/* for contents of Info.  This must correlate with umfpack.h */
/* -------------------------------------------------------------------------- */

#define ESTIMATE (UMFPACK_NUMERIC_SIZE_ESTIMATE - UMFPACK_NUMERIC_SIZE)
#define ACTUAL 0

/* -------------------------------------------------------------------------- */
/* get a parameter from the Control array */
/* -------------------------------------------------------------------------- */

#define GET_CONTROL(i,default) \
    ((Control != (double *) NULL) ? \
	(SCALAR_IS_NAN (Control [i]) ? default : Control [i]) \
	: default)

/* -------------------------------------------------------------------------- */
/* for clearing the external degree counters */
/* -------------------------------------------------------------------------- */

#define MAX_MARK(n) Int_MAX - (2*(n)+1)

/* -------------------------------------------------------------------------- */
/* convert number of Units to MBytes */
/* -------------------------------------------------------------------------- */

#define MBYTES(units) (((units) * sizeof (Unit)) / 1048576.0)

/* -------------------------------------------------------------------------- */
/* dense row/column macro */
/* -------------------------------------------------------------------------- */

/* In order for a row or column to be treated as "dense", it must have more */
/* entries than the value returned by this macro.  n is the dimension of the */
/* matrix, and alpha is the dense row/column control parameter. */

/* Note: this is not defined if alpha is NaN or Inf: */
#define UMFPACK_DENSE_DEGREE_THRESHOLD(alpha,n) \
    ((Int) MAX (16.0, (alpha) * 16.0 * sqrt ((double) (n))))

/* -------------------------------------------------------------------------- */
/* PRINTF */
/* -------------------------------------------------------------------------- */

#define PRINTFk(k,params) { if (prl >= (k)) { PRINTF (params) ; } }
#define PRINTF1(params) PRINTFk (1, params)
#define PRINTF2(params) PRINTFk (2, params)
#define PRINTF3(params) PRINTFk (3, params)
#define PRINTF4(params) PRINTFk (4, params)
#define PRINTF5(params) PRINTFk (5, params)
#define PRINTF6(params) PRINTFk (6, params)

/* -------------------------------------------------------------------------- */
/* Fixed control parameters */
/* -------------------------------------------------------------------------- */

/* maximum number of columns to consider at one time, in a single front */
#define MAX_CANDIDATES 128

/* reduce Numeric->Memory request by this ratio, if allocation fails */
#define UMF_REALLOC_REDUCTION (0.95)

/* increase Numeric->Memory request by this ratio, if we need more */
#define UMF_REALLOC_INCREASE (1.2)

/* increase the dimensions of the current frontal matrix by this factor
 * when it needs to grow. */
#define UMF_FRONTAL_GROWTH (1.2)

/* largest BLAS block size permitted */
#define MAXNB 64

/* if abs (y) < RECIPROCAL_TOLERANCE, then compute x/y.  Otherwise x*(1/y).
 * Ignored if NRECIPROCAL is defined */
#define RECIPROCAL_TOLERANCE 1e-12

/* -------------------------------------------------------------------------- */
/* Memory allocator */
/* -------------------------------------------------------------------------- */

    /* see SuiteSparse_config */

/* -------------------------------------------------------------------------- */
/* Memory space definitions */
/* -------------------------------------------------------------------------- */

/* for memory alignment - assume double has worst case alignment */
typedef double Align ;

/* get number of bytes required to hold n items of a type: */
/* note that this will not overflow, because sizeof (type) is always */
/* greater than or equal to sizeof (Int) >= 2 */
#define BYTES(type,n) (sizeof (type) * (n))

/* ceiling of (b/u).  Assumes b >= 0 and u > 0 */
#define CEILING(b,u) (((b) + (u) - 1) / (u))

/* get number of Units required to hold n items of a type: */
#define UNITS(type,n) (CEILING (BYTES (type, n), sizeof (Unit)))

/* same as DUNITS, but use double instead of int to avoid overflow */
#define DUNITS(type,n) (ceil (BYTES (type, (double) n) / sizeof (Unit)))

union Unit_union
{	/* memory is allocated in multiples of Unit */
    struct
    {
	Int
	    size,	/* size, in Units, of the block, excl. header block */
			/* size >= 0: block is in use */
			/* size < 0: block is free, of |size| Units */
	    prevsize ;	/* size, in Units, of preceding block in S->Memory */
			/* during garbage_collection, prevsize is set to -e-1 */
			/* for element e, or positive (and thus a free block) */
			/* otherwise */
    } header ;		/* block header */
    Align  xxxxxx ;	/* force alignment of blocks (xxxxxx is never used) */
} ;

typedef union Unit_union Unit ;

/* get the size of an allocated block */
#define GET_BLOCK_SIZE(p) (((p)-1)->header.size)

/* -------------------------------------------------------------------------- */
/* Numeric */
/* -------------------------------------------------------------------------- */

/*
    NUMERIC_VALID and SYMBOLIC_VALID:
    The different values of SYBOLIC_VALID and NUMERIC_VALID are chosen as a
    first defense against corrupted *Symbolic or *Numeric pointers passed to an
    UMFPACK routine.  They also ensure that the objects are used only by the
    same version that created them (umfpack_di_*, umfpack_dl_*, umfpack_zi_*,
    or umfpack_zl_*).  The values have also been changed since prior releases of
    the code to ensure that all routines that operate on the objects are of the
    same release.  The values themselves are purely arbitrary.  They are less
    than the ANSI C required minimums of INT_MAX and LONG_MAX, respectively.
*/

#ifdef DINT
#define NUMERIC_VALID  15977
#define SYMBOLIC_VALID 41937
#endif
#ifdef DLONG
#define NUMERIC_VALID  399789720
#define SYMBOLIC_VALID 399192713
#endif
#ifdef ZINT
#define NUMERIC_VALID  17957
#define SYMBOLIC_VALID 40927
#endif
#ifdef ZLONG
#define NUMERIC_VALID  129987754
#define SYMBOLIC_VALID 110291734
#endif

typedef struct	/* NumericType */
{
    double
	flops,		/* "true" flop count */
	relpt,		/* relative pivot tolerance used */
	relpt2,		/* relative pivot tolerance used for sym. */
	droptol,
	alloc_init,	/* initial allocation of Numeric->memory */
	front_alloc_init, /* frontal matrix allocation parameter */
	rsmin,		/* smallest row sum */
	rsmax,		/* largest row sum  */
	min_udiag,	/* smallest abs value on diagonal of D */
	max_udiag,	/* maximum abs value on diagonal of D */
	rcond ;		/* min (D) / max (D) */

    Int
	scale ;

    Int valid ;		/* set to NUMERIC_VALID, for validity check */

    /* Memory space for A and LU factors */
    Unit
	*Memory ;	/* working memory for A and LU factors */
    Int
	ihead,		/* pointer to tail of LU factors, in Numeric->Memory */
	itail,		/* pointer to top of elements & tuples,  */
			/* in Numeric->Memory */
	ibig,		/* pointer to largest free block seen in tail */
	size ;		/* size of Memory, in Units */

    Int
	*Rperm,		/* pointer to row perm array, size: n+1 */
			/* after UMF_kernel:  Rperm [new] = old */
			/* during UMF_kernel: Rperm [old] = new */
	*Cperm,		/* pointer to col perm array, size: n+1 */
			/* after UMF_kernel:  Cperm [new] = old */
			/* during UMF_kernel: Cperm [old] = new */

	*Upos,		/* see UMFPACK_get_numeric for a description */
	*Lpos,
	*Lip,
	*Lilen,
	*Uip,
	*Uilen,
	*Upattern ;	/* pattern of last row of U (if singular) */

    Int
	ulen,		/* length of Upattern */
	npiv,		/* number of structural pivots found (sprank approx) */
	nnzpiv ;	/* number of numerical (nonzero) pivots found */

    Entry
	*D ;		/* D [i] is the diagonal entry of U */

    Int do_recip ;
    double *Rs ;	/* scale factors for the rows of A and b */
			/* do_recip FALSE: Divide row i by Rs [i] */
			/* do_recip TRUE:  Multiply row i by Rs [i] */

    Int
	n_row, n_col,	/* A is n_row-by-n_row */
	n1 ;		/* number of singletons */

    /* for information only: */
    Int
	tail_usage,	/* amount of memory allocated in tail */
			/* head_usage is Numeric->ihead */
	init_usage,	/* memory usage just after UMF_kernel_init */
	max_usage,	/* peak memory usage (excludes internal and external */
			/* fragmentation in the tail) */
	ngarbage,	/* number of garbage collections performed */
	nrealloc,	/* number of reallocations performed */
	ncostly,	/* number of costly reallocations performed */
	isize,		/* size of integer pattern of L and U */
	nLentries,	/* number of entries in L, excluding diagonal */
	nUentries,	/* number of entries in U, including diagonal */
			/* Some entries may be numerically zero. */
	lnz,		/* number of nonzero entries in L, excl. diagonal */
	all_lnz,	/* lnz plus entries dropped from L */
	unz,		/* number of nonzero entries in U, excl. diagonal */
	all_unz,	/* unz plus entries dropped form U */
	maxfrsize ;	/* largest actual front size */

    Int maxnrows, maxncols ;	/* not the same as Symbolic->maxnrows/cols* */

} NumericType ;



/* -------------------------------------------------------------------------- */
/* Element tuples for connecting elements together in a matrix */
/* -------------------------------------------------------------------------- */

typedef struct	/* Tuple */
{
    /* The (e,f) tuples for the element lists */
    Int
	e,		/* element */
	f ;		/* contribution to the row/col appears at this offset */

} Tuple ;

#define TUPLES(t) MAX (4, (t) + 1)

/* Col_degree is aliased with Cperm, and Row_degree with Rperm */
#define NON_PIVOTAL_COL(col) (Col_degree [col] >= 0)
#define NON_PIVOTAL_ROW(row) (Row_degree [row] >= 0)

/* -------------------------------------------------------------------------- */
/* An element */
/* -------------------------------------------------------------------------- */

typedef struct	/* Element */
{
    Int

	cdeg,		/* external column degree + cdeg0 offset */
	rdeg,		/* external row degree    + rdeg0 offset */
	nrowsleft,	/* number of rows remaining */
	ncolsleft,	/* number of columns remaining */
	nrows,		/* number of rows */
	ncols,		/* number of columns */
	next ;		/* for list link of sons, used during assembly only */

    /* followed in memory by:
    Int
	col [0..ncols-1],	column indices of this element
	row [0..nrows-1] ;	row indices of this element
    Entry			(suitably aligned, see macro below)
	C [0...nrows-1, 0...ncols-1] ;
	size of C is nrows*ncols Entry's
    */

} Element ;

/* macros for computing pointers to row/col indices, and contribution block: */

#define GET_ELEMENT_SIZE(nr,nc) \
(UNITS (Element, 1) + UNITS (Int, (nc) + (nr)) + UNITS (Entry, (nc) * (nr)))

#define DGET_ELEMENT_SIZE(nr,nc) \
(DUNITS (Element, 1) + DUNITS (Int, (nc) + (nr)) + DUNITS (Entry, (nc) * (nr)))

#define GET_ELEMENT_COLS(ep,p,Cols) { \
    ASSERT (p != (Unit *) NULL) ; \
    ASSERT (p >= Numeric->Memory + Numeric->itail) ; \
    ASSERT (p <= Numeric->Memory + Numeric->size) ; \
    ep = (Element *) p ; \
    p += UNITS (Element, 1) ; \
    Cols = (Int *) p ; \
}

#define GET_ELEMENT_PATTERN(ep,p,Cols,Rows,ncm) { \
    GET_ELEMENT_COLS (ep, p, Cols) ; \
    ncm = ep->ncols ; \
    Rows = Cols + ncm ; \
}

#define GET_ELEMENT(ep,p,Cols,Rows,ncm,nrm,C) { \
    GET_ELEMENT_PATTERN (ep, p, Cols, Rows, ncm) ; \
    nrm = ep->nrows ; \
    p += UNITS (Int, ncm + nrm) ; \
    C = (Entry *) p ; \
}

/* -------------------------------------------------------------------------- */
/* Work data structure */
/* -------------------------------------------------------------------------- */

/*
    This data structure holds items needed only during factorization.
    All of this is freed when UMFPACK_numeric completes.  Note that some of
    it is stored in the tail end of Numeric->S (namely, the Tuples and the
    Elements).
*/

typedef struct	/* WorkType */
{

    /* ---------------------------------------------------------------------- */
    /* information about each row and col of A */
    /* ---------------------------------------------------------------------- */

    /*
	Row_tuples:	pointer to tuple list (alias with Numeric->Uip)
	Row_tlen:	number of tuples (alias with Numeric->Uilen)
	Col_tuples:	pointer to tuple list (alias with Numeric->Lip)
	Col_tlen:	number of tuples (alias with Numeric->Lilen)
	Row_degree:	degree of the row or column (alias Numeric->Rperm)
	Col_degree:	degree of the row or column (alias Numeric->Cperm)

	The Row_degree and Col_degree are MATLAB-style colmmd approximations,
	are equal to the sum of the sizes of the elements (contribution blocks)
	in each row and column.  They are maintained when elements are created
	and assembled.  They are used only during the pivot row and column
	search.  They are not needed to represent the pattern of the remaining
	matrix.
    */

    /* ---------------------------------------------------------------------- */
    /* information about each element */
    /* ---------------------------------------------------------------------- */

    Int	*E ;		/* E [0 .. Work->elen-1] element "pointers" */
			/* (offsets in Numeric->Memory) */

    /* ---------------------------------------------------------------------- */
    /* generic workspace */
    /* ---------------------------------------------------------------------- */

    Entry *Wx, *Wy ;	/* each of size maxnrows+1 */

    Int			/* Sizes:  nn = MAX (n_row, n_col) */
	*Wp,		/* nn+1 */
	*Wrp,		/* n_col+1 */
	*Wm,		/* maxnrows+1 */
	*Wio,		/* maxncols+1 */
	*Woi,		/* maxncols+1 */
	*Woo,		/* MAX (maxnrows,maxncols)+1 */
	*Wrow,		/* pointer to Fcols, Wio, or Woi */
	*NewRows,	/* list of rows to scan */
	*NewCols ;	/* list of cols to scan */

    /* ---------------------------------------------------------------------- */

    Int
	*Lpattern,	/* pattern of column of L, for one Lchain */
	*Upattern,	/* pattern of row of U, for one Uchain */
	ulen, llen ;	/* length of Upattern and Lpattern */

    Int
	*Diagonal_map,	/* used for symmetric pivoting, of size nn+1 */
	*Diagonal_imap ;/* used for symmetric pivoting, of size nn+1 */

    /* ---------------------------------------------------------------------- */

    Int
	n_row, n_col,	/* matrix is n_row-by-n_col */
	nz,		/* nonzeros in the elements for this matrix */
	n1,		/* number of row and col singletons */
	elen,		/* max possible number of elements */
	npiv,		/* number of pivot rows and columns so far */
	ndiscard,	/* number of discarded pivot columns */
	Wrpflag,
	nel,		/* elements in use are in the range 1..nel */
	noff_diagonal,
	prior_element,
	rdeg0, cdeg0,
	rrdeg, ccdeg,
	Candidates [MAX_CANDIDATES],	 /* current candidate pivot columns */
	nCandidates,	/* number of candidates in Candidate set */
	ksuper,
	firstsuper,
	jsuper,
	ncand,		/* number of candidates (some not in Candidates[ ]) */
	nextcand,	/* next candidate to place in Candidate search set */
	lo,
	hi,
	pivrow,		/* current pivot row */
	pivcol,		/* current pivot column */
	do_extend,	/* true if the next pivot extends the current front */
	do_update,	/* true if update should be applied */
	nforced,	/* number of forced updates because of frontal growth */
	any_skip,
	do_scan2row,
	do_scan2col,
	do_grow,
	pivot_case,
	frontid,	/* id of current frontal matrix */
	nfr ;		/* number of frontal matrices */

    /* ---------------------------------------------------------------------- */
    /* For row-merge tree */
    /* ---------------------------------------------------------------------- */

    Int
	*Front_new1strow ;

    /* ---------------------------------------------------------------------- */
    /* current frontal matrix, F */
    /* ---------------------------------------------------------------------- */

    Int Pivrow [MAXNB],
	Pivcol [MAXNB] ;

    Entry
	*Flublock,	/* LU block, nb-by-nb */
	*Flblock,	/* L block,  fnr_curr-by-nb */
	*Fublock,	/* U block,  nb-by-fnc_curr, or U' fnc_curr-by-nb */
	*Fcblock ;	/* C block,  fnr_curr-by-fnc_curr */

    Int
	*Frows,		/* Frows [0.. ]: row indices of F */

	*Fcols,		/* Fcols [0.. ]: column indices of F */

	*Frpos,		/* position of row indices in F, or -1 if not present */
			/* if Frows[i] == row, then Frpos[row] == i */

	*Fcpos,		/* position of col indices in F, or -1 if not present */
			/* if Fcols[j] == col, then */
			/* Fcpos[col] == j*Work->fnr_curr */

	fnrows,		/* number of rows in contribution block in F */
	fncols,		/* number of columns in contribution block in F */
	fnr_curr,	/* maximum # of rows in F (leading dimension) */
	fnc_curr,	/* maximum # of columns in F */
	fcurr_size,	/* current size of F */
	fnrows_max,	/* max possible column-dimension (max # of rows) of F */
	fncols_max,	/* max possible row-dimension (max # of columns) of F */
	nb,
	fnpiv,		/* number of pivots in F */
	fnzeros,	/* number of explicit zero entries in LU block */
	fscan_row,	/* where to start scanning rows of F in UMF_assemble */
	fscan_col,	/* where to start scanning cols of F in UMF_assemble */
	fnrows_new,	/* number of new row indices in F after pivot added */
	fncols_new,	/* number of new col indices in F after pivot added */
	pivrow_in_front,	/* true if current pivot row in Frows */
	pivcol_in_front ;	/* true if current pivot column in Fcols */

    /* ----------------------------------------------------------------------
     * Current frontal matrix
     * ----------------------------------------------------------------------
     * The current frontal matrix is held as a single block of memory allocated
     * from the "tail" end of Numeric->Memory.  It is subdivided into four
     * parts: an LU block, an L block, a U block, and a C block.
     *
     * Let k = fnpiv, r = fnrows, and c = fncols for the following discussion.
     * Let dr = fnr_curr and dc = fnc_curr.  Note that r <= dr and c <= dc.
     *
     * The LU block is of dimension nb-by-nb.  The first k-by-k part holds the
     * "diagonal" part of the LU factors for these k pivot rows and columns.
     * The k pivot row and column indices in this part are Pivrow [0..k-1] and
     * Pivcol [0..k-1], respectively.
     *
     * The L block is of dimension dr-by-nb.  It holds the k pivot columns,
     * except for the leading k-by-k part in the LU block.  Only the leading
     * r-by-k part is in use.
     *
     * The U block is of dimension dc-by-nb.  It holds the k pivot rows,
     * except for the leading k-by-k part in the LU block.  It is stored in
     * row-oriented form.  Only the leading c-by-k part is in use.
     *
     * The C block is of dimension dr-by-dc.  It holds the current contribution
     * block.  Only the leading r-by-c part is in use.  The column indices in
     * the C block are Fcols [0..c-1], and the row indices are Frows [0..r-1].
     *
     * dr is always odd, to avoid bad cache behavior.
     */

} WorkType ;


/* -------------------------------------------------------------------------- */
/* Symbolic */
/* -------------------------------------------------------------------------- */

/*
    This is is constructed by UMFPACK_symbolic, and is needed by UMFPACK_numeric
    to factor the matrix.
*/

typedef struct	/* SymbolicType */
{

    double
	num_mem_usage_est,	/* estimated max Numeric->Memory size */
	num_mem_size_est,	/* estimated final Numeric->Memory size */
	peak_sym_usage,		/* peak Symbolic and SymbolicWork usage */
	sym,			/* symmetry of pattern */
	dnum_mem_init_usage,	/* min Numeric->Memory for UMF_kernel_init */
	amd_lunz,	/* nz in LU for AMD, with symmetric pivoting */
	lunz_bound ;	/* max nx in LU, for arbitrary row pivoting */

    Int valid,		/* set to SYMBOLIC_VALID, for validity check */
	max_nchains,
	nchains,
	*Chain_start,
	*Chain_maxrows,
	*Chain_maxcols,
	maxnrows,		/* largest number of rows in any front */
	maxncols,		/* largest number of columns in any front */
	*Front_npivcol,		/* Front_npivcol [j] = size of jth supercolumn*/
	*Front_1strow,		/* first row in front j */
	*Front_leftmostdesc,	/* leftmost desc of front j */
	*Front_parent,		/* super-column elimination tree */
	*Cperm_init,		/* initial column ordering */
	*Rperm_init,		/* initial row ordering */
	*Cdeg, *Rdeg,
	*Esize,
	dense_row_threshold,
	n1,			/* number of singletons */
    n1r,                    /* number of row singletons */
    n1c,                    /* number of column singletons */
	nempty,			/* MIN (nempty_row, nempty_col) */
	*Diagonal_map,		/* initial "diagonal" */
	esize,			/* size of Esize array */
	nfr,
	n_row, n_col,		/* matrix A is n_row-by-n_col */
	nz,			/* nz of original matrix */
	nb,			/* block size for BLAS 3 */
	num_mem_init_usage,	/* min Numeric->Memory for UMF_kernel_init */
	nempty_row, nempty_col,

	strategy,
	ordering,
	fixQ,
	prefer_diagonal,
	nzaat,
	nzdiag,
	amd_dmax ;

} SymbolicType ;

/* -------------------------------------------------------------------------- */
/* SW Type: used internally in umfpack_qsymbolic */
/* -------------------------------------------------------------------------- */

typedef struct	/* SWType */
{
    Int *Front_npivcol ;    /* size n_col + 1 */
    Int *Front_nrows ;	    /* size n_col */
    Int *Front_ncols ;	    /* size n_col */
    Int *Front_parent ;	    /* size n_col */
    Int *Front_cols ;	    /* size n_col */
    Int *InFront ;	    /* size n_row */
    Int *Ci ;		    /* size Clen */
    Int *Cperm1 ;	    /* size n_col */
    Int *Rperm1 ;	    /* size n_row */
    Int *InvRperm1 ;	    /* size n_row */
    Int *Si ;		    /* size nz */
    Int *Sp ;		    /* size n_col + 1 */
    double *Rs ;	    /* size n_row */

} SWType ;

/* -------------------------------------------------------------------------- */
/* for debugging only: */
/* -------------------------------------------------------------------------- */

#include "umf_dump.h"

/* -------------------------------------------------------------------------- */
/* for statement coverage testing only: */
/* -------------------------------------------------------------------------- */

#ifdef TESTING

/* for testing integer overflow: */
#ifdef TEST_FOR_INTEGER_OVERFLOW
#undef MAX_MARK
#define MAX_MARK(n) (3*(n))
#endif

/* for testing out-of-memory conditions: */
#define UMF_TCOV_TEST

extern int umf_fail, umf_fail_lo, umf_fail_hi ;
extern int umf_realloc_fail, umf_realloc_lo, umf_realloc_hi ;

/* for testing malloc count: */
#define UMF_MALLOC_COUNT

#endif

#endif
