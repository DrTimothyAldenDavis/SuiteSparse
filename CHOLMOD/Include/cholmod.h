//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod.h: include file for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod.h.  Copyright (C) 2005-2022, Timothy A. Davis.
// All Rights Reserved.

// Each Module of CHOLMOD has its own license, and a shared cholmod.h file.

// CHOLMOD/Check:      SPDX-License-Identifier: LGPL-2.1+
// CHOLMOD/Cholesky:   SPDX-License-Identifier: LGPL-2.1+
// CHOLMOD/Core:       SPDX-License-Identifier: LGPL-2.1+
// CHOLMOD/Partition:  SPDX-License-Identifier: LGPL-2.1+

// CHOLMOD/Demo:       SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/GPU:        SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/MATLAB:     SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/MatrixOps:  SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Modify:     SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Supernodal: SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Tcov:       SPDX-License-Identifier: GPL-2.0+
// CHOLMOD/Valgrind:   SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* CHOLMOD consists of a set of Modules, each with their own license: either
 * LGPL-2.1+ or GPL-2.0+.  This cholmod.h file includes defintions of the
 * CHOLMOD API for all Modules, and this cholmod.h file itself is provided to
 * you with a permissive license (Apache-2.0).  You are permitted to provide
 * the hooks for an optional interface to CHOLMOD in a non-GPL/non-LGPL code,
 * without requiring you to agree to the GPL/LGPL license of the Modules, as
 * long as you don't use the *.c files in the relevant Modules.  The Modules
 * themselves can only be functional if their GPL or LGPL licenses are used.
 *
 * Portions of CHOLMOD (the Core and Partition Modules) are copyrighted by the
 * University of Florida.  The Modify Module is co-authored by William W.
 * Hager, Univ. of Florida.
 *
 * Acknowledgements:  this work was supported in part by the National Science
 * Foundation (NFS CCR-0203270 and DMS-9803599), and a grant from Sandia
 * National Laboratories (Dept. of Energy) which supported the development of
 * CHOLMOD's Partition Module.
 * -------------------------------------------------------------------------- */

/* CHOLMOD include file, for inclusion user programs.
 *
 * The include files listed below include a short description of each user-
 * callable routine.  Each routine in CHOLMOD has a consistent interface.
 * More details about the CHOLMOD data types is in the cholmod_core.h file.
 *
 * Naming convention:
 * ------------------
 *
 *	All routine names, data types, and CHOLMOD library files use the
 *	cholmod_ prefix.  All macros and other #define's use the CHOLMOD
 *	prefix.
 * 
 * Return value:
 * -------------
 *
 *	Most CHOLMOD routines return an int (TRUE (1) if successful, or FALSE
 *	(0) otherwise.  An int32_t, int64_t, or double return value is >= 0 if
 *	successful, or -1 otherwise.  A size_t return value is > 0 if
 *	successful, or 0 otherwise.
 *
 *	If a routine returns a pointer, it is a pointer to a newly allocated
 *	object or NULL if a failure occured, with one exception.  cholmod_free
 *	always returns NULL.
 *
 * "Common" parameter:
 * ------------------
 *
 *	The last parameter in all CHOLMOD routines is a pointer to the CHOLMOD
 *	"Common" object.  This contains control parameters, statistics, and
 *	workspace used between calls to CHOLMOD.  It is always an input/output
 *	parameter.
 *
 * Input, Output, and Input/Output parameters:
 * -------------------------------------------
 *
 *	Input parameters are listed first.  They are not modified by CHOLMOD.
 *
 *	Input/output are listed next.  They must be defined on input, and
 *	are modified on output.
 *
 *	Output parameters are listed next.  If they are pointers, they must
 *	point to allocated space on input, but their contents are not defined
 *	on input.
 *
 *	Workspace parameters appear next.  They are used in only two routines
 *	in the Supernodal module.
 *
 *	The cholmod_common *Common parameter always appears as the last
 *	parameter.  It is always an input/output parameter.
 */

#ifndef CHOLMOD_H
#define CHOLMOD_H

#define CHOLMOD_DATE "Oct 15, 2023"
#define CHOLMOD_MAIN_VERSION   4
#define CHOLMOD_SUB_VERSION    2
#define CHOLMOD_SUBSUB_VERSION 2

/* ========================================================================== */
/* === Include/cholmod_io64 ================================================= */
/* ========================================================================== */

/* assume large file support.  If problems occur, compile with -DNLARGEFILE */

/* Definitions required for large file I/O, which must come before any other
 * #includes.  These are not used if -DNLARGEFILE is defined at compile time.
 * Large file support may not be portable across all platforms and compilers;
 * if you encounter an error here, compile your code with -DNLARGEFILE.  In
 * particular, you must use -DNLARGEFILE for MATLAB 6.5 or earlier (which does
 * not have the io64.h include file).
 */

/* skip all of this if NLARGEFILE is defined at the compiler command line */
#ifndef NLARGEFILE

#if defined(MATLAB_MEX_FILE) || defined(MATHWORKS)

/* CHOLMOD is being compiled as a MATLAB mexFunction, or for use in MATLAB */
#include "io64.h"

#else

/* CHOLMOD is being compiled in a stand-alone library */
#undef  _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64

#endif

#endif

/* ========================================================================== */
/* === SuiteSparse_config.h ================================================= */
/* ========================================================================== */

#include "SuiteSparse_config.h"

/* ========================================================================== */
/* === Include/cholmod_config.h ============================================= */
/* ========================================================================== */

/* CHOLMOD configuration file, for inclusion in user programs.
 *
 * You do not have to edit any CHOLMOD files to compile and install CHOLMOD.
 * However, if you do not use all of CHOLMOD's modules, you need to compile
 * with the appropriate flag, or edit this file to add the appropriate #define.
 *
 * Compiler flags for CHOLMOD:
 *
 * -DNCHECK	    do not include the Check module.
 * -DNCHOLESKY	    do not include the Cholesky module.
 * -DNPARTITION	    do not include the Partition module.
 * -DNCAMD          do not include the interfaces to CAMD,
 *                  CCOLAMD, CSYMAND in Partition module.
 * -DNMATRIXOPS	    do not include the MatrixOps module.
 * -DNMODIFY	    do not include the Modify module.
 * -DNSUPERNODAL    do not include the Supernodal module.
 *
 * -DNPRINT	    do not print anything
 *
 * The Core Module is always included in the CHOLMOD library.
 */

/* Use the compiler flag, or uncomment the definition(s), if you want to use
 * one or more non-default installation options: */

/*
#define NCHECK
#define NCHOLESKY
#define NCAMD
#define NPARTITION
#define NMATRIXOPS
#define NMODIFY
#define NSUPERNODAL
#define NPRINT
#define NGPL
*/

/* The NGPL option disables the MatrixOps, Modify, and Supernodal modules.  The
    existence of this #define here, and its use in these 3 modules, does not
    affect the license itself; see CHOLMOD/Doc/License.txt for your actual
    license.
 */

#ifdef NGPL
#undef  NMATRIXOPS
#define NMATRIXOPS
#undef  NMODIFY
#define NMODIFY
#undef  NSUPERNODAL
#define NSUPERNODAL
#endif


/* ========================================================================== */
/* === Include/cholmod_core.h =============================================== */
/* ========================================================================== */

/* CHOLMOD Core module: basic CHOLMOD objects and routines.
 * Required by all CHOLMOD modules.  Requires no other module or package.
 *
 * The CHOLMOD modules are:
 *
 * Core		basic data structures and definitions
 * Check	check/print the 5 CHOLMOD objects, & 3 types of integer vectors
 * Cholesky	sparse Cholesky factorization
 * Modify	sparse Cholesky update/downdate/row-add/row-delete
 * MatrixOps	sparse matrix functions (add, multiply, norm, ...)
 * Supernodal	supernodal sparse Cholesky factorization
 * Partition	graph-partitioning based orderings
 *
 * The CHOLMOD objects:
 * --------------------
 *
 * cholmod_common   parameters, statistics, and workspace
 * cholmod_sparse   a sparse matrix in compressed column form
 * cholmod_factor   an LL' or LDL' factorization
 * cholmod_dense    a dense matrix
 * cholmod_triplet  a sparse matrix in "triplet" form
 *
 * The Core module described here defines the CHOLMOD data structures, and
 * basic operations on them.  To create and solve a sparse linear system Ax=b,
 * the user must create A and b, populate them with values, and then pass them
 * to the routines in the CHOLMOD Cholesky module.  There are two primary
 * methods for creating A: (1) allocate space for a column-oriented sparse
 * matrix and fill it with pattern and values, or (2) create a triplet form
 * matrix and convert it to a sparse matrix.  The latter option is simpler.
 *
 * The matrices b and x are typically dense matrices, but can also be sparse.
 * You can allocate and free them as dense matrices with the
 * cholmod_allocate_dense and cholmod_free_dense routines.
 *
 * The cholmod_factor object contains the symbolic and numeric LL' or LDL'
 * factorization of sparse symmetric matrix.  The matrix must be positive
 * definite for an LL' factorization.  It need only be symmetric and have well-
 * conditioned leading submatrices for it to have an LDL' factorization
 * (CHOLMOD does not pivot for numerical stability).  It is typically created
 * with the cholmod_factorize routine in the Cholesky module, but can also
 * be initialized to L=D=I in the Core module and then modified by the Modify
 * module.  It must be freed with cholmod_free_factor, defined below.
 *
 * The Core routines for each object are described below.  Each list is split
 * into two parts: the primary routines and secondary routines.
 *
 * ============================================================================
 * === cholmod_common =========================================================
 * ============================================================================
 *
 * The Common object contains control parameters, statistics, and
 * You must call cholmod_start before calling any other CHOLMOD routine, and
 * must call cholmod_finish as your last call to CHOLMOD, with two exceptions:
 * you may call cholmod_print_common and cholmod_check_common in the Check
 * module after calling cholmod_finish.
 *
 * cholmod_start		first call to CHOLMOD
 * cholmod_finish		last call to CHOLMOD
 * -----------------------------
 * cholmod_defaults		restore default parameters
 * cholmod_maxrank		maximum rank for update/downdate
 * cholmod_allocate_work	allocate workspace in Common
 * cholmod_free_work		free workspace in Common
 * cholmod_clear_flag		clear Flag workspace in Common
 * cholmod_error		called when CHOLMOD encounters an error
 * cholmod_dbound		for internal use in CHOLMOD only
 * cholmod_hypot		compute sqrt (x*x + y*y) accurately
 * cholmod_divcomplex		complex division, c = a/b
 *
 * ============================================================================
 * === cholmod_sparse =========================================================
 * ============================================================================
 *
 * A sparse matrix is held in compressed column form.  In the basic type
 * ("packed", which corresponds to a MATLAB sparse matrix), an n-by-n matrix
 * with nz entries is held in three arrays: p of size n+1, i of size nz, and x
 * of size nz.  Row indices of column j are held in i [p [j] ... p [j+1]-1] and
 * in the same locations in x.  There may be no duplicate entries in a column.
 * Row indices in each column may be sorted or unsorted (CHOLMOD keeps track).
 * A->stype determines the storage mode: 0 if both upper/lower parts are stored,
 * -1 if A is symmetric and just tril(A) is stored, +1 if symmetric and triu(A)
 * is stored.
 *
 * cholmod_allocate_sparse	allocate a sparse matrix
 * cholmod_free_sparse		free a sparse matrix
 * -----------------------------
 * cholmod_reallocate_sparse	change the size (# entries) of sparse matrix
 * cholmod_nnz			number of nonzeros in a sparse matrix
 * cholmod_speye		sparse identity matrix
 * cholmod_spzeros		sparse zero matrix
 * cholmod_transpose		transpose a sparse matrix
 * cholmod_ptranspose		transpose/permute a sparse matrix
 * cholmod_transpose_unsym	transpose/permute an unsymmetric sparse matrix
 * cholmod_transpose_sym	transpose/permute a symmetric sparse matrix
 * cholmod_sort			sort row indices in each column of sparse matrix
 * cholmod_band			C = tril (triu (A,k1), k2)
 * cholmod_band_inplace		A = tril (triu (A,k1), k2)
 * cholmod_aat			C = A*A'
 * cholmod_copy_sparse		C = A, create an exact copy of a sparse matrix
 * cholmod_copy			C = A, with possible change of stype
 * cholmod_add			C = alpha*A + beta*B
 * cholmod_sparse_xtype		change the xtype of a sparse matrix
 *
 * ============================================================================
 * === cholmod_factor =========================================================
 * ============================================================================
 *
 * The data structure for an LL' or LDL' factorization is too complex to
 * describe in one sentence.  This object can hold the symbolic analysis alone,
 * or in combination with a "simplicial" (similar to a sparse matrix) or
 * "supernodal" form of the numerical factorization.  Only the routine to free
 * a factor is primary, since a factor object is created by the factorization
 * routine (cholmod_factorize).  It must be freed with cholmod_free_factor.
 *
 * cholmod_free_factor		free a factor
 * -----------------------------
 * cholmod_allocate_factor	allocate a factor (LL' or LDL')
 * cholmod_reallocate_factor	change the # entries in a factor
 * cholmod_change_factor	change the type of factor (e.g., LDL' to LL')
 * cholmod_pack_factor		pack the columns of a factor
 * cholmod_reallocate_column	resize a single column of a factor
 * cholmod_factor_to_sparse	create a sparse matrix copy of a factor
 * cholmod_copy_factor		create a copy of a factor
 * cholmod_factor_xtype		change the xtype of a factor
 *
 * Note that there is no cholmod_sparse_to_factor routine to create a factor
 * as a copy of a sparse matrix.  It could be done, after a fashion, but a
 * lower triangular sparse matrix would not necessarily have a chordal graph,
 * which would break the many CHOLMOD routines that rely on this property.
 *
 * ============================================================================
 * === cholmod_dense ==========================================================
 * ============================================================================
 *
 * The solve routines and some of the MatrixOps and Modify routines use dense
 * matrices as inputs.  These are held in column-major order.  With a leading
 * dimension of d, the entry in row i and column j is held in x [i+j*d].
 *
 * cholmod_allocate_dense	allocate a dense matrix
 * cholmod_free_dense		free a dense matrix
 * -----------------------------
 * cholmod_zeros		allocate a dense matrix of all zeros
 * cholmod_ones			allocate a dense matrix of all ones
 * cholmod_eye			allocate a dense identity matrix
 * cholmod_sparse_to_dense	create a dense matrix copy of a sparse matrix
 * cholmod_dense_to_sparse	create a sparse matrix copy of a dense matrix
 * cholmod_copy_dense		create a copy of a dense matrix
 * cholmod_copy_dense2		copy a dense matrix (pre-allocated)
 * cholmod_dense_xtype		change the xtype of a dense matrix
 * cholmod_ensure_dense  	ensure a dense matrix has a given size and type
 *
 * ============================================================================
 * === cholmod_triplet ========================================================
 * ============================================================================
 *
 * A sparse matrix held in triplet form is the simplest one for a user to
 * create.  It consists of a list of nz entries in arbitrary order, held in
 * three arrays: i, j, and x, each of length nk.  The kth entry is in row i[k],
 * column j[k], with value x[k].  There may be duplicate values; if A(i,j)
 * appears more than once, its value is the sum of the entries with those row
 * and column indices.
 *
 * cholmod_allocate_triplet	allocate a triplet matrix
 * cholmod_triplet_to_sparse	create a sparse matrix copy of a triplet matrix
 * cholmod_free_triplet		free a triplet matrix
 * -----------------------------
 * cholmod_reallocate_triplet	change the # of entries in a triplet matrix
 * cholmod_sparse_to_triplet	create a triplet matrix copy of a sparse matrix
 * cholmod_copy_triplet		create a copy of a triplet matrix
 * cholmod_triplet_xtype	change the xtype of a triplet matrix
 *
 * ============================================================================
 * === memory management ======================================================
 * ============================================================================
 *
 * cholmod_malloc		malloc wrapper
 * cholmod_calloc		calloc wrapper
 * cholmod_free			free wrapper
 * cholmod_realloc		realloc wrapper
 * cholmod_realloc_multiple	realloc wrapper for multiple objects
 *
 * ============================================================================
 * === Core CHOLMOD prototypes ================================================
 * ============================================================================
 *
 * All CHOLMOD routines (in all modules) use the following protocol for return
 * values, with one exception:
 *
 * int			TRUE (1) if successful, or FALSE (0) otherwise.
 *			(exception: cholmod_divcomplex)
 * int32_t              a value >= 0 if successful, or -1 otherwise.
 * int64_t              a value >= 0 if successful, or -1 otherwise.
 * double		a value >= 0 if successful, or -1 otherwise.
 * size_t		a value > 0 if successful, or 0 otherwise.
 * void *		a non-NULL pointer to newly allocated memory if
 *			successful, or NULL otherwise.
 * cholmod_sparse *	a non-NULL pointer to a newly allocated matrix
 *			if successful, or NULL otherwise.
 * cholmod_factor *	a non-NULL pointer to a newly allocated factor
 *			if successful, or NULL otherwise.
 * cholmod_triplet *	a non-NULL pointer to a newly allocated triplet
 *			matrix if successful, or NULL otherwise.
 * cholmod_dense *	a non-NULL pointer to a newly allocated triplet
 *			matrix if successful, or NULL otherwise.
 *
 * The last parameter to all routines is always a pointer to the CHOLMOD
 * Common object.
 *
 * TRUE and FALSE are not defined here, since they may conflict with the user
 * program.  A routine that described here returning TRUE or FALSE returns 1
 * or 0, respectively.  Any TRUE/FALSE parameter is true if nonzero, false if
 * zero.
 */

/* ========================================================================== */
/* === CHOLMOD version ====================================================== */
/* ========================================================================== */

/* All versions of CHOLMOD will include the following definitions.
 * As an example, to test if the version you are using is 1.3 or later:
 *
 *	if (CHOLMOD_VERSION >= CHOLMOD_VER_CODE (1,3)) ...
 *
 * This also works during compile-time:
 *
 *	#if CHOLMOD_VERSION >= CHOLMOD_VER_CODE (1,3)
 *	    printf ("This is version 1.3 or later\n") ;
 *	#else
 *	    printf ("This is version is earlier than 1.3\n") ;
 *	#endif
 */

#define CHOLMOD_HAS_VERSION_FUNCTION
#define CHOLMOD_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define CHOLMOD_VERSION \
    CHOLMOD_VER_CODE(CHOLMOD_MAIN_VERSION,CHOLMOD_SUB_VERSION)


/* ========================================================================== */
/* === CUDA BLAS for the GPU ================================================ */
/* ========================================================================== */

/* Define buffering parameters for GPU processing */
#ifndef SUITESPARSE_GPU_EXTERN_ON
#ifdef SUITESPARSE_CUDA
#include <cublas_v2.h>
#endif
#endif

#define CHOLMOD_DEVICE_SUPERNODE_BUFFERS 6
#define CHOLMOD_HOST_SUPERNODE_BUFFERS 8
#define CHOLMOD_DEVICE_STREAMS 2

/* ========================================================================== */
/* === CHOLMOD objects ====================================================== */
/* ========================================================================== */

/* Each CHOLMOD object has its own type code. */

#define CHOLMOD_COMMON 0
#define CHOLMOD_SPARSE 1
#define CHOLMOD_FACTOR 2
#define CHOLMOD_DENSE 3
#define CHOLMOD_TRIPLET 4

/* ========================================================================== */
/* === CHOLMOD Common ======================================================= */
/* ========================================================================== */

/* itype defines the types of integer used: */
#define CHOLMOD_INT 0		/* all integer arrays are int32_t */
#define CHOLMOD_INTLONG 1	/* most are int32_t, some are int64_t */
#define CHOLMOD_LONG 2		/* all integer arrays are int64_t */

/* The itype of all parameters for all CHOLMOD routines must match.
 * FUTURE WORK: CHOLMOD_INTLONG is not yet supported.
 */

/* dtype defines what the numerical type is (double or float): */
#define CHOLMOD_DOUBLE 0	/* all numerical values are double */
#define CHOLMOD_SINGLE 1	/* all numerical values are float */

/* The dtype of all parameters for all CHOLMOD routines must match.
 *
 * Scalar floating-point values are always passed as double arrays of size 2
 * (for the real and imaginary parts).  They are typecast to float as needed.
 * FUTURE WORK: the float case is not supported yet.
 */

/* xtype defines the kind of numerical values used: */
#define CHOLMOD_PATTERN 0	/* pattern only, no numerical values */
#define CHOLMOD_REAL 1		/* a real matrix */
#define CHOLMOD_COMPLEX 2	/* a complex matrix (ANSI C99 compatible) */
#define CHOLMOD_ZOMPLEX 3	/* a complex matrix (MATLAB compatible) */

/* The xtype of all parameters for all CHOLMOD routines must match.
 *
 * CHOLMOD_PATTERN: x and z are ignored.
 * CHOLMOD_DOUBLE:  x is non-null of size nzmax, z is ignored.
 * CHOLMOD_COMPLEX: x is non-null of size 2*nzmax doubles, z is ignored.
 * CHOLMOD_ZOMPLEX: x and z are non-null of size nzmax
 *
 * In the real case, z is ignored.  The kth entry in the matrix is x [k].
 * There are two methods for the complex case.  In the ANSI C99-compatible
 * CHOLMOD_COMPLEX case, the real and imaginary parts of the kth entry
 * are in x [2*k] and x [2*k+1], respectively.  z is ignored.  In the
 * MATLAB-compatible CHOLMOD_ZOMPLEX case, the real and imaginary
 * parts of the kth entry are in x [k] and z [k].
 *
 * Scalar floating-point values are always passed as double arrays of size 2
 * (real and imaginary parts).  The imaginary part of a scalar is ignored if
 * the routine operates on a real matrix.
 *
 * These Modules support complex and zomplex matrices, with a few exceptions:
 *
 *	Check	    all routines
 *	Cholesky    all routines
 *	Core	    all except cholmod_aat, add, band, copy
 *	Demo	    all routines
 *	Partition   all routines
 *	Supernodal  all routines support any real, complex, or zomplex input.
 *			There will never be a supernodal zomplex L; a complex
 *			supernodal L is created if A is zomplex.
 *	Tcov	    all routines
 *	Valgrind    all routines
 *
 * These Modules provide partial support for complex and zomplex matrices:
 *
 *	MATLAB	    all routines support real and zomplex only, not complex,
 *			with the exception of ldlupdate, which supports
 *			real matrices only.  This is a minor constraint since
 *			MATLAB's matrices are all real or zomplex.
 *	MatrixOps   only norm_dense, norm_sparse, and sdmult support complex
 *			and zomplex
 *
 * These Modules do not support complex and zomplex matrices at all:
 *
 *	Modify	    all routines support real matrices only
 */

/* Definitions for cholmod_common: */
#define CHOLMOD_MAXMETHODS 9	/* maximum number of different methods that */
				/* cholmod_analyze can try. Must be >= 9. */

/* Common->status values.  zero means success, negative means a fatal error,
 * positive is a warning. */
#define CHOLMOD_OK 0			/* success */
#define CHOLMOD_NOT_INSTALLED (-1)	/* failure: method not installed */
#define CHOLMOD_OUT_OF_MEMORY (-2)	/* failure: out of memory */
#define CHOLMOD_TOO_LARGE (-3)		/* failure: integer overflow occured */
#define CHOLMOD_INVALID (-4)		/* failure: invalid input */
#define CHOLMOD_GPU_PROBLEM (-5)        /* failure: GPU fatal error */
#define CHOLMOD_NOT_POSDEF (1)		/* warning: matrix not pos. def. */
#define CHOLMOD_DSMALL (2)		/* warning: D for LDL'  or diag(L) or */
					/* LL' has tiny absolute value */

/* ordering method (also used for L->ordering) */
#define CHOLMOD_NATURAL 0	/* use natural ordering */
#define CHOLMOD_GIVEN 1		/* use given permutation */
#define CHOLMOD_AMD 2		/* use minimum degree (AMD) */
#define CHOLMOD_METIS 3		/* use METIS' nested dissection */
#define CHOLMOD_NESDIS 4	/* use CHOLMOD's version of nested dissection:*/
				/* node bisector applied recursively, followed
				 * by constrained minimum degree (CSYMAMD or
				 * CCOLAMD) */
#define CHOLMOD_COLAMD 5	/* use AMD for A, COLAMD for A*A' */

/* POSTORDERED is not a method, but a result of natural ordering followed by a
 * weighted postorder.  It is used for L->ordering, not method [ ].ordering. */
#define CHOLMOD_POSTORDERED 6	/* natural ordering, postordered. */

/* supernodal strategy (for Common->supernodal) */
#define CHOLMOD_SIMPLICIAL 0	/* always do simplicial */
#define CHOLMOD_AUTO 1		/* select simpl/super depending on matrix */
#define CHOLMOD_SUPERNODAL 2	/* always do supernodal */

/* make it easy for C++ programs to include CHOLMOD */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cholmod_common_struct
{
    /* ---------------------------------------------------------------------- */
    /* parameters for symbolic/numeric factorization and update/downdate */
    /* ---------------------------------------------------------------------- */

    double dbound ;	/* Smallest absolute value of diagonal entries of D
			 * for LDL' factorization and update/downdate/rowadd/
	* rowdel, or the diagonal of L for an LL' factorization.
	* Entries in the range 0 to dbound are replaced with dbound.
	* Entries in the range -dbound to 0 are replaced with -dbound.  No
	* changes are made to the diagonal if dbound <= 0.  Default: zero */

    double grow0 ;	/* For a simplicial factorization, L->i and L->x can
			 * grow if necessary.  grow0 is the factor by which
	* it grows.  For the initial space, L is of size MAX (1,grow0) times
	* the required space.  If L runs out of space, the new size of L is
	* MAX(1.2,grow0) times the new required space.   If you do not plan on
	* modifying the LDL' factorization in the Modify module, set grow0 to
	* zero (or set grow2 to 0, see below).  Default: 1.2 */

    double grow1 ;

    size_t grow2 ;	/* For a simplicial factorization, each column j of L
			 * is initialized with space equal to
	* grow1*L->ColCount[j] + grow2.  If grow0 < 1, grow1 < 1, or grow2 == 0,
	* then the space allocated is exactly equal to L->ColCount[j].  If the
	* column j runs out of space, it increases to grow1*need + grow2 in
	* size, where need is the total # of nonzeros in that column.  If you do
	* not plan on modifying the factorization in the Modify module, set
	* grow2 to zero.  Default: grow1 = 1.2, grow2 = 5. */

    size_t maxrank ;	/* rank of maximum update/downdate.  Valid values:
			 * 2, 4, or 8.  A value < 2 is set to 2, and a
	* value > 8 is set to 8.  It is then rounded up to the next highest
	* power of 2, if not already a power of 2.  Workspace (Xwork, below) of
	* size nrow-by-maxrank double's is allocated for the update/downdate.
	* If an update/downdate of rank-k is requested, with k > maxrank,
	* it is done in steps of maxrank.  Default: 8, which is fastest.
	* Memory usage can be reduced by setting maxrank to 2 or 4.
	*/

    double supernodal_switch ;	/* supernodal vs simplicial factorization */
    int supernodal ;		/* If Common->supernodal <= CHOLMOD_SIMPLICIAL
				 * (0) then cholmod_analyze performs a
	* simplicial analysis.  If >= CHOLMOD_SUPERNODAL (2), then a supernodal
	* analysis is performed.  If == CHOLMOD_AUTO (1) and
	* flop/nnz(L) < Common->supernodal_switch, then a simplicial analysis
	* is done.  A supernodal analysis done otherwise.
	* Default:  CHOLMOD_AUTO.  Default supernodal_switch = 40 */

    int final_asis ;	/* If TRUE, then ignore the other final_* parameters
			 * (except for final_pack).
			 * The factor is left as-is when done.  Default: TRUE.*/

    int final_super ;	/* If TRUE, leave a factor in supernodal form when
			 * supernodal factorization is finished.  If FALSE,
			 * then convert to a simplicial factor when done.
			 * Default: TRUE */

    int final_ll ;	/* If TRUE, leave factor in LL' form when done.
			 * Otherwise, leave in LDL' form.  Default: FALSE */

    int final_pack ;	/* If TRUE, pack the columns when done.  If TRUE, and
			 * cholmod_factorize is called with a symbolic L, L is
	* allocated with exactly the space required, using L->ColCount.  If you
	* plan on modifying the factorization, set Common->final_pack to FALSE,
	* and each column will be given a little extra slack space for future
	* growth in fill-in due to updates.  Default: TRUE */

    int final_monotonic ;   /* If TRUE, ensure columns are monotonic when done.
			 * Default: TRUE */

    int final_resymbol ;/* if cholmod_factorize performed a supernodal
			 * factorization, final_resymbol is true, and
	* final_super is FALSE (convert a simplicial numeric factorization),
	* then numerically zero entries that resulted from relaxed supernodal
	* amalgamation are removed.  This does not remove entries that are zero
	* due to exact numeric cancellation, since doing so would break the
	* update/downdate rowadd/rowdel routines.  Default: FALSE. */

    /* supernodal relaxed amalgamation parameters: */
    double zrelax [3] ;
    size_t nrelax [3] ;

	/* Let ns be the total number of columns in two adjacent supernodes.
	 * Let z be the fraction of zero entries in the two supernodes if they
	 * are merged (z includes zero entries from prior amalgamations).  The
	 * two supernodes are merged if:
	 *    (ns <= nrelax [0]) || (no new zero entries added) ||
	 *    (ns <= nrelax [1] && z < zrelax [0]) ||
	 *    (ns <= nrelax [2] && z < zrelax [1]) || (z < zrelax [2])
	 *
	 * Default parameters result in the following rule:
	 *    (ns <= 4) || (no new zero entries added) ||
	 *    (ns <= 16 && z < 0.8) || (ns <= 48 && z < 0.1) || (z < 0.05)
	 */

    int prefer_zomplex ;    /* X = cholmod_solve (sys, L, B, Common) computes
			     * x=A\b or solves a related system.  If L and B are
	 * both real, then X is real.  Otherwise, X is returned as
	 * CHOLMOD_COMPLEX if Common->prefer_zomplex is FALSE, or
	 * CHOLMOD_ZOMPLEX if Common->prefer_zomplex is TRUE.  This parameter
	 * is needed because there is no supernodal zomplex L.  Suppose the
	 * caller wants all complex matrices to be stored in zomplex form
	 * (MATLAB, for example).  A supernodal L is returned in complex form
	 * if A is zomplex.  B can be real, and thus X = cholmod_solve (L,B)
	 * should return X as zomplex.  This cannot be inferred from the input
	 * arguments L and B.  Default: FALSE, since all data types are
	 * supported in CHOLMOD_COMPLEX form and since this is the native type
	 * of LAPACK and the BLAS.  Note that the MATLAB/cholmod.c mexFunction
	 * sets this parameter to TRUE, since MATLAB matrices are in
	 * CHOLMOD_ZOMPLEX form.
	 */

    int prefer_upper ;	    /* cholmod_analyze and cholmod_factorize work
			     * fastest when a symmetric matrix is stored in
	 * upper triangular form when a fill-reducing ordering is used.  In
	 * MATLAB, this corresponds to how x=A\b works.  When the matrix is
	 * ordered as-is, they work fastest when a symmetric matrix is in lower
	 * triangular form.  In MATLAB, R=chol(A) does the opposite.  This
	 * parameter affects only how cholmod_read returns a symmetric matrix.
	 * If TRUE (the default case), a symmetric matrix is always returned in
	 * upper-triangular form (A->stype = 1).  */

    int quick_return_if_not_posdef ;	/* if TRUE, the supernodal numeric
					 * factorization will return quickly if
	* the matrix is not positive definite.  Default: FALSE. */

    int prefer_binary ;	    /* cholmod_read_triplet converts a symmetric
			     * pattern-only matrix into a real matrix.  If
	* prefer_binary is FALSE, the diagonal entries are set to 1 + the degree
	* of the row/column, and off-diagonal entries are set to -1 (resulting
	* in a positive definite matrix if the diagonal is zero-free).  Most
	* symmetric patterns are the pattern a positive definite matrix.  If
	* this parameter is TRUE, then the matrix is returned with a 1 in each
	* entry, instead.  Default: FALSE.  Added in v1.3. */

    /* ---------------------------------------------------------------------- */
    /* printing and error handling options */
    /* ---------------------------------------------------------------------- */

    int print ;		/* print level. Default: 3 */
    int precise ;	/* if TRUE, print 16 digits.  Otherwise print 5 */

    /* CHOLMOD print_function replaced with SuiteSparse_config print_func */

    int try_catch ;	/* if TRUE, then ignore errors; CHOLMOD is in the middle
			 * of a try/catch block.  No error message is printed
	 * and the Common->error_handler function is not called. */

    void (*error_handler) (int status, const char *file,
        int line, const char *message) ;

	/* Common->error_handler is the user's error handling routine.  If not
	 * NULL, this routine is called if an error occurs in CHOLMOD.  status
	 * can be CHOLMOD_OK (0), negative for a fatal error, and positive for
	 * a warning. file is a string containing the name of the source code
	 * file where the error occured, and line is the line number in that
	 * file.  message is a string describing the error in more detail. */

    /* ---------------------------------------------------------------------- */
    /* ordering options */
    /* ---------------------------------------------------------------------- */

    /* The cholmod_analyze routine can try many different orderings and select
     * the best one.  It can also try one ordering method multiple times, with
     * different parameter settings.  The default is to use three orderings,
     * the user's permutation (if provided), AMD which is the fastest ordering
     * and generally gives good fill-in, and METIS.  CHOLMOD's nested dissection
     * (METIS with a constrained AMD) usually gives a better ordering than METIS
     * alone (by about 5% to 10%) but it takes more time.
     *
     * If you know the method that is best for your matrix, set Common->nmethods
     * to 1 and set Common->method [0] to the set of parameters for that method.
     * If you set it to 1 and do not provide a permutation, then only AMD will
     * be called.
     *
     * If METIS is not available, the default # of methods tried is 2 (the user
     * permutation, if any, and AMD).
     *
     * To try other methods, set Common->nmethods to the number of methods you
     * want to try.  The suite of default methods and their parameters is
     * described in the cholmod_defaults routine, and summarized here:
     *
     *	    Common->method [i]:
     *	    i = 0: user-provided ordering (cholmod_analyze_p only)
     *	    i = 1: AMD (for both A and A*A')
     *	    i = 2: METIS
     *	    i = 3: CHOLMOD's nested dissection (NESDIS), default parameters
     *	    i = 4: natural
     *	    i = 5: NESDIS with nd_small = 20000
     *	    i = 6: NESDIS with nd_small = 4, no constrained minimum degree
     *	    i = 7: NESDIS with no dense node removal
     *	    i = 8: AMD for A, COLAMD for A*A'
     *
     * You can modify the suite of methods you wish to try by modifying
     * Common.method [...] after calling cholmod_start or cholmod_defaults.
     *
     * For example, to use AMD, followed by a weighted postordering:
     *
     *	    Common->nmethods = 1 ;
     *	    Common->method [0].ordering = CHOLMOD_AMD ;
     *	    Common->postorder = TRUE ;
     *
     * To use the natural ordering (with no postordering):
     *
     *	    Common->nmethods = 1 ;
     *	    Common->method [0].ordering = CHOLMOD_NATURAL ;
     *	    Common->postorder = FALSE ;
     *
     * If you are going to factorize hundreds or more matrices with the same
     * nonzero pattern, you may wish to spend a great deal of time finding a
     * good permutation.  In this case, try setting Common->nmethods to 9.
     * The time spent in cholmod_analysis will be very high, but you need to
     * call it only once.
     *
     * cholmod_analyze sets Common->current to a value between 0 and nmethods-1.
     * Each ordering method uses the set of options defined by this parameter.
     */

    int nmethods ;	/* The number of ordering methods to try.  Default: 0.
			 * nmethods = 0 is a special case.  cholmod_analyze
	* will try the user-provided ordering (if given) and AMD.  Let fl and
	* lnz be the flop count and nonzeros in L from AMD's ordering.  Let
	* anz be the number of nonzeros in the upper or lower triangular part
	* of the symmetric matrix A.  If fl/lnz < 500 or lnz/anz < 5, then this
	* is a good ordering, and METIS is not attempted.  Otherwise, METIS is
	* tried.   The best ordering found is used.  If nmethods > 0, the
	* methods used are given in the method[ ] array, below.  The first
	* three methods in the default suite of orderings is (1) use the given
	* permutation (if provided), (2) use AMD, and (3) use METIS.  Maximum
	* allowed value is CHOLMOD_MAXMETHODS.  */

    int current ;	/* The current method being tried.  Default: 0.  Valid
			 * range is 0 to nmethods-1. */

    int selected ;	/* The best method found. */

    /* The suite of ordering methods and parameters: */

    struct cholmod_method_struct
    {
	/* statistics for this method */
	double lnz ;	    /* nnz(L) excl. zeros from supernodal amalgamation,
			     * for a "pure" L */

	double fl ;	    /* flop count for a "pure", real simplicial LL'
			     * factorization, with no extra work due to
	    * amalgamation.  Subtract n to get the LDL' flop count.   Multiply
	    * by about 4 if the matrix is complex or zomplex. */

	/* ordering method parameters */
	double prune_dense ;/* dense row/col control for AMD, SYMAMD, CSYMAMD,
			     * and NESDIS (cholmod_nested_dissection).  For a
	    * symmetric n-by-n matrix, rows/columns with more than
	    * MAX (16, prune_dense * sqrt (n)) entries are removed prior to
	    * ordering.  They appear at the end of the re-ordered matrix.
	    *
	    * If prune_dense < 0, only completely dense rows/cols are removed.
	    *
	    * This paramater is also the dense column control for COLAMD and
	    * CCOLAMD.  For an m-by-n matrix, columns with more than
	    * MAX (16, prune_dense * sqrt (MIN (m,n))) entries are removed prior
	    * to ordering.  They appear at the end of the re-ordered matrix.
	    * CHOLMOD factorizes A*A', so it calls COLAMD and CCOLAMD with A',
	    * not A.  Thus, this parameter affects the dense *row* control for
	    * CHOLMOD's matrix, and the dense *column* control for COLAMD and
	    * CCOLAMD.
	    *
	    * Removing dense rows and columns improves the run-time of the
	    * ordering methods.  It has some impact on ordering quality
	    * (usually minimal, sometimes good, sometimes bad).
	    *
	    * Default: 10. */

	double prune_dense2 ;/* dense row control for COLAMD and CCOLAMD.
			    *  Rows with more than MAX (16, dense2 * sqrt (n))
	    * for an m-by-n matrix are removed prior to ordering.  CHOLMOD's
	    * matrix is transposed before ordering it with COLAMD or CCOLAMD,
	    * so this controls the dense *columns* of CHOLMOD's matrix, and
	    * the dense *rows* of COLAMD's or CCOLAMD's matrix.
	    *
	    * If prune_dense2 < 0, only completely dense rows/cols are removed.
	    *
	    * Default: -1.  Note that this is not the default for COLAMD and
	    * CCOLAMD.  -1 is best for Cholesky.  10 is best for LU.  */

	double nd_oksep ;   /* in NESDIS, when a node separator is computed, it
			     * discarded if nsep >= nd_oksep*n, where nsep is
	    * the number of nodes in the separator, and n is the size of the
	    * graph being cut.  Valid range is 0 to 1.  If 1 or greater, the
	    * separator is discarded if it consists of the entire graph.
	    * Default: 1 */

	double other_1 [4] ; /* future expansion */

	size_t nd_small ;    /* do not partition graphs with fewer nodes than
			     * nd_small, in NESDIS.  Default: 200 (same as
			     * METIS) */

	size_t other_2 [4] ; /* future expansion */

	int aggressive ;    /* Aggresive absorption in AMD, COLAMD, SYMAMD,
			     * CCOLAMD, and CSYMAMD.  Default: TRUE */

	int order_for_lu ;  /* CCOLAMD can be optimized to produce an ordering
			     * for LU or Cholesky factorization.  CHOLMOD only
	    * performs a Cholesky factorization.  However, you may wish to use
	    * CHOLMOD as an interface for CCOLAMD but use it for your own LU
	    * factorization.  In this case, order_for_lu should be set to FALSE.
	    * When factorizing in CHOLMOD itself, you should *** NEVER *** set
	    * this parameter FALSE.  Default: TRUE. */

	int nd_compress ;   /* If TRUE, compress the graph and subgraphs before
			     * partitioning them in NESDIS.  Default: TRUE */

	int nd_camd ;	    /* If 1, follow the nested dissection ordering
			     * with a constrained minimum degree ordering that
	    * respects the partitioning just found (using CAMD).  If 2, use
	    * CSYMAMD instead.  If you set nd_small very small, you may not need
	    * this ordering, and can save time by setting it to zero (no
	    * constrained minimum degree ordering).  Default: 1. */

	int nd_components ; /* The nested dissection ordering finds a node
			     * separator that splits the graph into two parts,
	    * which may be unconnected.  If nd_components is TRUE, each of
	    * these connected components is split independently.  If FALSE,
	    * each part is split as a whole, even if it consists of more than
	    * one connected component.  Default: FALSE */

	/* fill-reducing ordering to use */
	int ordering ;

	size_t other_3 [4] ; /* future expansion */

    } method [CHOLMOD_MAXMETHODS + 1] ;

    int postorder ;	/* If TRUE, cholmod_analyze follows the ordering with a
			 * weighted postorder of the elimination tree.  Improves
	* supernode amalgamation.  Does not affect fundamental nnz(L) and
	* flop count.  Default: TRUE. */

    int default_nesdis ;    /* Default: FALSE.  If FALSE, then the default
			     * ordering strategy (when Common->nmethods == 0)
	* is to try the given ordering (if present), AMD, and then METIS if AMD
	* reports high fill-in.  If Common->default_nesdis is TRUE then NESDIS
	* is used instead in the default strategy. */

    /* ---------------------------------------------------------------------- */
    /* memory management, complex divide, and hypot function pointers moved */
    /* ---------------------------------------------------------------------- */

    /* Function pointers moved from here (in CHOLMOD 2.2.0) to
       SuiteSparse_config.[ch].  See CHOLMOD/Include/cholmod_back.h
       for a set of macros that can be #include'd or copied into your
       application to define these function pointers on any version of CHOLMOD.
       */

    /* ---------------------------------------------------------------------- */
    /* METIS workarounds */
    /* ---------------------------------------------------------------------- */

    /* These workarounds were put into place for METIS 4.0.1.  They are safe
       to use with METIS 5.1.0, but they might not longer be necessary. */

    double metis_memory ;   /* This is a parameter for CHOLMOD's interface to
			     * METIS, not a parameter to METIS itself.  METIS
	* uses an amount of memory that is difficult to estimate precisely
	* beforehand.  If it runs out of memory, it terminates your program.
	* All routines in CHOLMOD except for CHOLMOD's interface to METIS
	* return an error status and safely return to your program if they run
	* out of memory.  To mitigate this problem, the CHOLMOD interface
	* can allocate a single block of memory equal in size to an empirical
	* upper bound of METIS's memory usage times the Common->metis_memory
	* parameter, and then immediately free it.  It then calls METIS.  If
	* this pre-allocation fails, it is possible that METIS will fail as
	* well, and so CHOLMOD returns with an out-of-memory condition without
	* calling METIS.
	*
	* METIS_NodeND (used in the CHOLMOD_METIS ordering option) with its
	* default parameter settings typically uses about (4*nz+40n+4096)
	* times sizeof(int) memory, where nz is equal to the number of entries
	* in A for the symmetric case or AA' if an unsymmetric matrix is
	* being ordered (where nz includes both the upper and lower parts
	* of A or AA').  The observed "upper bound" (with 2 exceptions),
	* measured in an instrumented copy of METIS 4.0.1 on thousands of
	* matrices, is (10*nz+50*n+4096) * sizeof(int).  Two large matrices
	* exceeded this bound, one by almost a factor of 2 (Gupta/gupta2).
	*
	* If your program is terminated by METIS, try setting metis_memory to
	* 2.0, or even higher if needed.  By default, CHOLMOD assumes that METIS
	* does not have this problem (so that CHOLMOD will work correctly when
	* this issue is fixed in METIS).  Thus, the default value is zero.
	* This work-around is not guaranteed anyway.
	*
	* If a matrix exceeds this predicted memory usage, AMD is attempted
	* instead.  It, too, may run out of memory, but if it does so it will
	* not terminate your program.
	*/

    double metis_dswitch ;	/* METIS_NodeND in METIS 4.0.1 gives a seg */
    size_t metis_nswitch ;	/* fault with one matrix of order n = 3005 and
				 * nz = 6,036,025.  This is a very dense graph.
     * The workaround is to use AMD instead of METIS for matrices of dimension
     * greater than Common->metis_nswitch (default 3000) or more and with
     * density of Common->metis_dswitch (default 0.66) or more.
     * cholmod_nested_dissection has no problems with the same matrix, even
     * though it uses METIS_ComputeVertexSeparator on this matrix.  If this
     * seg fault does not affect you, set metis_nswitch to zero or less,
     * and CHOLMOD will not switch to AMD based just on the density of the
     * matrix (it will still switch to AMD if the metis_memory parameter
     * causes the switch).
     */

    /* ---------------------------------------------------------------------- */
    /* workspace */
    /* ---------------------------------------------------------------------- */

    /* CHOLMOD has several routines that take less time than the size of
     * workspace they require.  Allocating and initializing the workspace would
     * dominate the run time, unless workspace is allocated and initialized
     * just once.  CHOLMOD allocates this space when needed, and holds it here
     * between calls to CHOLMOD.  cholmod_start sets these pointers to NULL
     * (which is why it must be the first routine called in CHOLMOD).
     * cholmod_finish frees the workspace (which is why it must be the last
     * call to CHOLMOD).
     */

    size_t nrow ;	/* size of Flag and Head */
    int64_t mark ;	/* mark value for Flag array */
    size_t iworksize ;	/* size of Iwork.  Upper bound: 6*nrow+ncol */
    size_t xworksize ;	/* size of Xwork,  in bytes.
			 * maxrank*nrow*sizeof(double) for update/downdate.
			 * 2*nrow*sizeof(double) otherwise */

    /* initialized workspace: contents needed between calls to CHOLMOD */
    void *Flag ;	/* size nrow, an integer array.  Kept cleared between
			 * calls to cholmod rouines (Flag [i] < mark) */

    void *Head ;	/* size nrow+1, an integer array. Kept cleared between
			 * calls to cholmod routines (Head [i] = EMPTY) */

    void *Xwork ; 	/* a double array.  Its size varies.  It is nrow for
			 * most routines (cholmod_rowfac, cholmod_add,
	* cholmod_aat, cholmod_norm, cholmod_ssmult) for the real case, twice
	* that when the input matrices are complex or zomplex.  It is of size
	* 2*nrow for cholmod_rowadd and cholmod_rowdel.  For cholmod_updown,
	* its size is maxrank*nrow where maxrank is 2, 4, or 8.  Kept cleared
	* between calls to cholmod (set to zero). */

    /* uninitialized workspace, contents not needed between calls to CHOLMOD */
    void *Iwork ;	/* size iworksize, 2*nrow+ncol for most routines,
			 * up to 6*nrow+ncol for cholmod_analyze. */

    int itype ;		/* If CHOLMOD_LONG, Flag, Head, and Iwork are
                         * int64_t.  Otherwise all three are int. */

    int dtype ;		/* double or float */

	/* Common->itype and Common->dtype are used to define the types of all
	 * sparse matrices, triplet matrices, dense matrices, and factors
	 * created using this Common struct.  The itypes and dtypes of all
	 * parameters to all CHOLMOD routines must match.  */

    int no_workspace_reallocate ;   /* this is an internal flag, used as a
	* precaution by cholmod_analyze.  It is normally false.  If true,
	* cholmod_allocate_work is not allowed to reallocate any workspace;
	* they must use the existing workspace in Common (Iwork, Flag, Head,
	* and Xwork).  Added for CHOLMOD v1.1 */

    /* ---------------------------------------------------------------------- */
    /* statistics */
    /* ---------------------------------------------------------------------- */

    /* fl and lnz are set only in cholmod_analyze and cholmod_rowcolcounts,
     * in the Cholesky modudle.  modfl is set only in the Modify module. */

    int status ;	    /* error code */
    double fl ;		    /* LL' flop count from most recent analysis */
    double lnz ;	    /* fundamental nz in L */
    double anz ;	    /* nonzeros in tril(A) if A is symmetric/lower,
			     * triu(A) if symmetric/upper, or tril(A*A') if
			     * unsymmetric, in last call to cholmod_analyze. */
    double modfl ;	    /* flop count from most recent update/downdate/
			     * rowadd/rowdel (excluding flops to modify the
			     * solution to Lx=b, if computed) */
    size_t malloc_count ;   /* # of objects malloc'ed minus the # free'd*/
    size_t memory_usage ;   /* peak memory usage in bytes */
    size_t memory_inuse ;   /* current memory usage in bytes */

    double nrealloc_col ;   /* # of column reallocations */
    double nrealloc_factor ;/* # of factor reallocations due to col. reallocs */
    double ndbounds_hit ;   /* # of times diagonal modified by dbound */

    double rowfacfl ;	    /* # of flops in last call to cholmod_rowfac */
    double aatfl ;	    /* # of flops to compute A(:,f)*A(:,f)' */

    int called_nd ;	    /* TRUE if the last call to
			     * cholmod_analyze called NESDIS or METIS. */
    int blas_ok ;           /* FALSE if SUITESPARSE_BLAS_INT overflow; 
                               TRUE otherwise */

    /* ---------------------------------------------------------------------- */
    /* SuiteSparseQR control parameters: */
    /* ---------------------------------------------------------------------- */

    double SPQR_grain ;      /* task size is >= max (total flops / grain) */
    double SPQR_small ;      /* task size is >= small */
    int SPQR_shrink ;        /* controls stack realloc method */
    int SPQR_nthreads ;      /* number of TBB threads, 0 = auto */

    /* ---------------------------------------------------------------------- */
    /* SuiteSparseQR statistics */
    /* ---------------------------------------------------------------------- */

    /* was other1 [0:3] */
    double SPQR_flopcount ;         /* flop count for SPQR */
    double SPQR_analyze_time ;      /* analysis time in seconds for SPQR */
    double SPQR_factorize_time ;    /* factorize time in seconds for SPQR */
    double SPQR_solve_time ;        /* backsolve time in seconds */

    /* was SPQR_xstat [0:3] */
    double SPQR_flopcount_bound ;   /* upper bound on flop count */
    double SPQR_tol_used ;          /* tolerance used */
    double SPQR_norm_E_fro ;        /* Frobenius norm of dropped entries */

    /* was SPQR_istat [0:9] */
    int64_t SPQR_istat [10] ;

    /* ---------------------------------------------------------------------- */
    /* GPU configuration and statistics */
    /* ---------------------------------------------------------------------- */

    /*  useGPU:  1 if gpu-acceleration is requested */
    /*           0 if gpu-acceleration is prohibited */
    /*          -1 if gpu-acceleration is undefined in which case the */
    /*             environment CHOLMOD_USE_GPU will be queried and used. */
    /*             useGPU=-1 is only used by CHOLMOD and treated as 0 by SPQR */
    int useGPU;

    /* for CHOLMOD: */
    size_t maxGpuMemBytes;
    double maxGpuMemFraction;

    /* for SPQR: */
    size_t gpuMemorySize;       /* Amount of memory in bytes on the GPU */
    double gpuKernelTime;       /* Time taken by GPU kernels */
    int64_t gpuFlops;           /* Number of flops performed by the GPU */
    int gpuNumKernelLaunches;   /* Number of GPU kernel launches */

    /* If not using the GPU, these items are not used, but they should be
       present so that the CHOLMOD Common has the same size whether the GPU
       is used or not.  This way, all packages will agree on the size of
       the CHOLMOD Common, regardless of whether or not they are compiled
       with the GPU libraries or not */

#ifdef SUITESPARSE_CUDA
    /* in CUDA, these three types are pointers */
    #define CHOLMOD_CUBLAS_HANDLE cublasHandle_t
    #define CHOLMOD_CUDASTREAM    cudaStream_t
    #define CHOLMOD_CUDAEVENT     cudaEvent_t
#else
    /* ... so make them void * pointers if the GPU is not being used */
    #define CHOLMOD_CUBLAS_HANDLE void *
    #define CHOLMOD_CUDASTREAM    void *
    #define CHOLMOD_CUDAEVENT     void *
#endif

    CHOLMOD_CUBLAS_HANDLE cublasHandle ;

    /* a set of streams for general use */
    CHOLMOD_CUDASTREAM    gpuStream[CHOLMOD_HOST_SUPERNODE_BUFFERS];

    CHOLMOD_CUDAEVENT     cublasEventPotrf [3] ;
    CHOLMOD_CUDAEVENT     updateCKernelsComplete;
    CHOLMOD_CUDAEVENT     updateCBuffersFree[CHOLMOD_HOST_SUPERNODE_BUFFERS];

    void *dev_mempool;    /* pointer to single allocation of device memory */
    size_t dev_mempool_size;

    void *host_pinned_mempool;  /* pointer to single allocation of pinned mem */
    size_t host_pinned_mempool_size;

    size_t devBuffSize;
    int    ibuffer;

    double syrkStart ;          /* time syrk started */

    /* run times of the different parts of CHOLMOD (GPU and CPU) */
    double cholmod_cpu_gemm_time ;
    double cholmod_cpu_syrk_time ;
    double cholmod_cpu_trsm_time ;
    double cholmod_cpu_potrf_time ;
    double cholmod_gpu_gemm_time ;
    double cholmod_gpu_syrk_time ;
    double cholmod_gpu_trsm_time ;
    double cholmod_gpu_potrf_time ;
    double cholmod_assemble_time ;
    double cholmod_assemble_time2 ;

    /* number of times the BLAS are called on the CPU and the GPU */
    size_t cholmod_cpu_gemm_calls ;
    size_t cholmod_cpu_syrk_calls ;
    size_t cholmod_cpu_trsm_calls ;
    size_t cholmod_cpu_potrf_calls ;
    size_t cholmod_gpu_gemm_calls ;
    size_t cholmod_gpu_syrk_calls ;
    size_t cholmod_gpu_trsm_calls ;
    size_t cholmod_gpu_potrf_calls ;

    double chunk ;      // chunksize for computing # of threads to use.
                        // Given nwork work to do, # of threads is
                        // max (1, min (floor (work / chunk), nthreads_max))
    int nthreads_max ;  // max # of threads to use in CHOLMOD.  Defaults to
                        // SUITESPARSE_OPENMP_MAX_THREADS.

} cholmod_common ;

/* size_t BLAS statistcs in Common: */
#define CHOLMOD_CPU_GEMM_CALLS      cholmod_cpu_gemm_calls
#define CHOLMOD_CPU_SYRK_CALLS      cholmod_cpu_syrk_calls
#define CHOLMOD_CPU_TRSM_CALLS      cholmod_cpu_trsm_calls
#define CHOLMOD_CPU_POTRF_CALLS     cholmod_cpu_potrf_calls
#define CHOLMOD_GPU_GEMM_CALLS      cholmod_gpu_gemm_calls
#define CHOLMOD_GPU_SYRK_CALLS      cholmod_gpu_syrk_calls
#define CHOLMOD_GPU_TRSM_CALLS      cholmod_gpu_trsm_calls
#define CHOLMOD_GPU_POTRF_CALLS     cholmod_gpu_potrf_calls

/* double BLAS statistics in Common: */
#define CHOLMOD_CPU_GEMM_TIME       cholmod_cpu_gemm_time
#define CHOLMOD_CPU_SYRK_TIME       cholmod_cpu_syrk_time
#define CHOLMOD_CPU_TRSM_TIME       cholmod_cpu_trsm_time
#define CHOLMOD_CPU_POTRF_TIME      cholmod_cpu_potrf_time
#define CHOLMOD_GPU_GEMM_TIME       cholmod_gpu_gemm_time
#define CHOLMOD_GPU_SYRK_TIME       cholmod_gpu_syrk_time
#define CHOLMOD_GPU_TRSM_TIME       cholmod_gpu_trsm_time
#define CHOLMOD_GPU_POTRF_TIME      cholmod_gpu_potrf_time
#define CHOLMOD_ASSEMBLE_TIME       cholmod_assemble_time
#define CHOLMOD_ASSEMBLE_TIME2      cholmod_assemble_time2

/* for supernodal analysis */
#define CHOLMOD_ANALYZE_FOR_SPQR     0
#define CHOLMOD_ANALYZE_FOR_CHOLESKY 1
#define CHOLMOD_ANALYZE_FOR_SPQRGPU  2

/* -------------------------------------------------------------------------- */
/* cholmod_start:  first call to CHOLMOD */
/* -------------------------------------------------------------------------- */

int cholmod_start
(
    cholmod_common *Common
) ;

int cholmod_l_start (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_finish:  last call to CHOLMOD */
/* -------------------------------------------------------------------------- */

int cholmod_finish
(
    cholmod_common *Common
) ;

int cholmod_l_finish (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_defaults:  restore default parameters */
/* -------------------------------------------------------------------------- */

int cholmod_defaults
(
    cholmod_common *Common
) ;

int cholmod_l_defaults (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_maxrank:  return valid maximum rank for update/downdate */
/* -------------------------------------------------------------------------- */

size_t cholmod_maxrank	/* returns validated value of Common->maxrank */
(
    /* ---- input ---- */
    size_t n,		/* A and L will have n rows */
    /* --------------- */
    cholmod_common *Common
) ;

size_t cholmod_l_maxrank (size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_allocate_work:  allocate workspace in Common */
/* -------------------------------------------------------------------------- */

int cholmod_allocate_work
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_allocate_work (size_t, size_t, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_free_work:  free workspace in Common */
/* -------------------------------------------------------------------------- */

int cholmod_free_work
(
    cholmod_common *Common
) ;

int cholmod_l_free_work (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_clear_flag:  clear Flag workspace in Common */
/* -------------------------------------------------------------------------- */

/* use a macro for speed */
#define CHOLMOD_CLEAR_FLAG(Common) \
{ \
    Common->mark++ ; \
    if (Common->mark <= 0) \
    { \
	Common->mark = EMPTY ; \
	CHOLMOD (clear_flag) (Common) ; \
    } \
}

int64_t cholmod_clear_flag
(
    cholmod_common *Common
) ;

int64_t cholmod_l_clear_flag (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_error:  called when CHOLMOD encounters an error */
/* -------------------------------------------------------------------------- */

int cholmod_error
(
    /* ---- input ---- */
    int status,		/* error status */
    const char *file,	/* name of source code file where error occured */
    int line,		/* line number in source code file where error occured*/
    const char *message,/* error message */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_error (int, const char *, int, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_dbound:  for internal use in CHOLMOD only */
/* -------------------------------------------------------------------------- */

double cholmod_dbound	/* returns modified diagonal entry of D or L */
(
    /* ---- input ---- */
    double dj,		/* diagonal entry of D for LDL' or L for LL' */
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_dbound (double, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_hypot:  compute sqrt (x*x + y*y) accurately */
/* -------------------------------------------------------------------------- */

double cholmod_hypot
(
    /* ---- input ---- */
    double x, double y
) ;

double cholmod_l_hypot (double, double) ;

/* -------------------------------------------------------------------------- */
/* cholmod_divcomplex:  complex division, c = a/b */
/* -------------------------------------------------------------------------- */

int cholmod_divcomplex		/* return 1 if divide-by-zero, 0 otherise */
(
    /* ---- input ---- */
    double ar, double ai,	/* real and imaginary parts of a */
    double br, double bi,	/* real and imaginary parts of b */
    /* ---- output --- */
    double *cr, double *ci	/* real and imaginary parts of c */
) ;

int cholmod_l_divcomplex (double, double, double, double, double *, double *) ;


/* ========================================================================== */
/* === Core/cholmod_sparse ================================================== */
/* ========================================================================== */

/* A sparse matrix stored in compressed-column form. */

typedef struct cholmod_sparse_struct
{
    size_t nrow ;	/* the matrix is nrow-by-ncol */
    size_t ncol ;
    size_t nzmax ;	/* maximum number of entries in the matrix */

    /* pointers to int32_t or int64_t: */
    void *p ;		/* p [0..ncol], the column pointers */
    void *i ;		/* i [0..nzmax-1], the row indices */

    /* for unpacked matrices only: */
    void *nz ;		/* nz [0..ncol-1], the # of nonzeros in each col.  In
			 * packed form, the nonzero pattern of column j is in
	* A->i [A->p [j] ... A->p [j+1]-1].  In unpacked form, column j is in
	* A->i [A->p [j] ... A->p [j]+A->nz[j]-1] instead.  In both cases, the
	* numerical values (if present) are in the corresponding locations in
	* the array x (or z if A->xtype is CHOLMOD_ZOMPLEX). */

    /* pointers to double or float: */
    void *x ;		/* size nzmax or 2*nzmax, if present */
    void *z ;		/* size nzmax, if present */

    int stype ;		/* Describes what parts of the matrix are considered:
			 *
	* 0:  matrix is "unsymmetric": use both upper and lower triangular parts
	*     (the matrix may actually be symmetric in pattern and value, but
	*     both parts are explicitly stored and used).  May be square or
	*     rectangular.
	* >0: matrix is square and symmetric, use upper triangular part.
	*     Entries in the lower triangular part are ignored.
	* <0: matrix is square and symmetric, use lower triangular part.
	*     Entries in the upper triangular part are ignored.
	*
	* Note that stype>0 and stype<0 are different for cholmod_sparse and
	* cholmod_triplet.  See the cholmod_triplet data structure for more
	* details.
	*/

    int itype ;		/* CHOLMOD_INT:     p, i, and nz are int32_t.
			 * CHOLMOD_INTLONG: p is int64_t,
                         *                  i and nz are int32_t.
			 * CHOLMOD_LONG:    p, i, and nz are int64_t */

    int xtype ;		/* pattern, real, complex, or zomplex */
    int dtype ;		/* x and z are double or float */
    int sorted ;	/* TRUE if columns are sorted, FALSE otherwise */
    int packed ;	/* TRUE if packed (nz ignored), FALSE if unpacked
			 * (nz is required) */

} cholmod_sparse ;

typedef struct cholmod_descendant_score_t
{
    double score ;
    int64_t d ;
}
descendantScore ;

/* For sorting descendant supernodes with qsort */
int cholmod_score_comp (struct cholmod_descendant_score_t *i,
			struct cholmod_descendant_score_t *j) ;

int cholmod_l_score_comp (struct cholmod_descendant_score_t *i,
			  struct cholmod_descendant_score_t *j) ;

/* -------------------------------------------------------------------------- */
/* cholmod_allocate_sparse:  allocate a sparse matrix */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_allocate_sparse
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    size_t nzmax,	/* max # of nonzeros of A */
    int sorted,		/* TRUE if columns of A sorted, FALSE otherwise */
    int packed,		/* TRUE if A will be packed, FALSE otherwise */
    int stype,		/* stype of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_allocate_sparse (size_t, size_t, size_t, int, int,
    int, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_free_sparse:  free a sparse matrix */
/* -------------------------------------------------------------------------- */

int cholmod_free_sparse
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_free_sparse (cholmod_sparse **, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_reallocate_sparse:  change the size (# entries) of sparse matrix */
/* -------------------------------------------------------------------------- */

int cholmod_reallocate_sparse
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_reallocate_sparse ( size_t, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_nnz:  return number of nonzeros in a sparse matrix */
/* -------------------------------------------------------------------------- */

int64_t cholmod_nnz
(
    /* ---- input ---- */
    cholmod_sparse *A,
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_nnz (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_speye:  sparse identity matrix */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_speye
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_speye (size_t, size_t, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_spzeros:  sparse zero matrix */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_spzeros
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    size_t nzmax,	/* max # of nonzeros of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_spzeros (size_t, size_t, size_t, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_transpose:  transpose a sparse matrix */
/* -------------------------------------------------------------------------- */

/* Return A' or A.'  The "values" parameter is 0, 1, or 2 to denote the pattern
 * transpose, the array transpose (A.'), and the complex conjugate transpose
 * (A').
 */

cholmod_sparse *cholmod_transpose
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to transpose */
    int values,		/* 0: pattern, 1: array transpose, 2: conj. transpose */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_transpose (cholmod_sparse *, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_transpose_unsym:  transpose an unsymmetric sparse matrix */
/* -------------------------------------------------------------------------- */

/* Compute F = A', A (:,f)', or A (p,f)', where A is unsymmetric and F is
 * already allocated.  See cholmod_transpose for a simpler routine. */

int cholmod_transpose_unsym
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to transpose */
    int values,		/* 0: pattern, 1: array transpose, 2: conj. transpose */
    int32_t *Perm,	/* size nrow, if present (can be NULL) */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    cholmod_sparse *F,	/* F = A', A(:,f)', or A(p,f)' */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_transpose_unsym (cholmod_sparse *, int, int64_t *,
    int64_t *, size_t, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_transpose_sym:  transpose a symmetric sparse matrix */
/* -------------------------------------------------------------------------- */

/* Compute F = A' or A (p,p)', where A is symmetric and F is already allocated.
 * See cholmod_transpose for a simpler routine. */

int cholmod_transpose_sym
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to transpose */
    int values,		/* 0: pattern, 1: array transpose, 2: conj. transpose */
    int32_t *Perm,	/* size nrow, if present (can be NULL) */
    /* ---- output --- */
    cholmod_sparse *F,	/* F = A' or A(p,p)' */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_transpose_sym (cholmod_sparse *, int, int64_t *,
    cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_ptranspose:  transpose a sparse matrix */
/* -------------------------------------------------------------------------- */

/* Return A' or A(p,p)' if A is symmetric.  Return A', A(:,f)', or A(p,f)' if
 * A is unsymmetric. */

cholmod_sparse *cholmod_ptranspose
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to transpose */
    int values,		/* 0: pattern, 1: array transpose, 2: conj. transpose */
    int32_t *Perm,	/* if non-NULL, F = A(p,f) or A(p,p) */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_ptranspose (cholmod_sparse *, int, int64_t *,
    int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_sort:  sort row indices in each column of sparse matrix */
/* -------------------------------------------------------------------------- */

int cholmod_sort
(
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to sort */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_sort (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_band:  C = tril (triu (A,k1), k2) */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_band
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to extract band matrix from */
    int64_t k1,         /* ignore entries below the k1-st diagonal */
    int64_t k2,         /* ignore entries above the k2-nd diagonal */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag) */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_band (cholmod_sparse *, int64_t,
    int64_t, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_band_inplace:  A = tril (triu (A,k1), k2) */
/* -------------------------------------------------------------------------- */

int cholmod_band_inplace
(
    /* ---- input ---- */
    int64_t k1,         /* ignore entries below the k1-st diagonal */
    int64_t k2,         /* ignore entries above the k2-nd diagonal */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag) */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix from which entries not in band are removed */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_band_inplace (int64_t, int64_t, int,
    cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_aat:  C = A*A' or A(:,f)*A(:,f)' */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_aat
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* input matrix; C=A*A' is constructed */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag),
			 * -2: pattern only, no diagonal, add 50%+n extra
			 * space to C */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_aat (cholmod_sparse *, int64_t *, size_t,
    int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_copy_sparse:  C = A, create an exact copy of a sparse matrix */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_copy_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_copy_sparse (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_copy:  C = A, with possible change of stype */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_copy
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    int stype,		/* requested stype of C */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag) */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_copy (cholmod_sparse *, int, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_add: C = alpha*A + beta*B */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_add
(
    /* ---- input ---- */
    cholmod_sparse *A,	    /* matrix to add */
    cholmod_sparse *B,	    /* matrix to add */
    double alpha [2],	    /* scale factor for A */
    double beta [2],	    /* scale factor for B */
    int values,		    /* if TRUE compute the numerical values of C */
    int sorted,		    /* if TRUE, sort columns of C */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_add (cholmod_sparse *, cholmod_sparse *, double *,
    double *, int, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_sparse_xtype: change the xtype of a sparse matrix */
/* -------------------------------------------------------------------------- */

int cholmod_sparse_xtype
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (pattern, real, complex, zomplex) */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* sparse matrix to change */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_sparse_xtype (int, cholmod_sparse *, cholmod_common *) ;


/* ========================================================================== */
/* === Core/cholmod_factor ================================================== */
/* ========================================================================== */

/* A symbolic and numeric factorization, either simplicial or supernodal.
 * In all cases, the row indices in the columns of L are kept sorted. */

typedef struct cholmod_factor_struct
{
    /* ---------------------------------------------------------------------- */
    /* for both simplicial and supernodal factorizations */
    /* ---------------------------------------------------------------------- */

    size_t n ;		/* L is n-by-n */

    size_t minor ;	/* If the factorization failed, L->minor is the column
			 * at which it failed (in the range 0 to n-1).  A value
			 * of n means the factorization was successful or
			 * the matrix has not yet been factorized. */

    /* ---------------------------------------------------------------------- */
    /* symbolic ordering and analysis */
    /* ---------------------------------------------------------------------- */

    void *Perm ;	/* size n, permutation used */
    void *ColCount ;	/* size n, column counts for simplicial L */

    void *IPerm ;       /* size n, inverse permutation.  Only created by
                         * cholmod_solve2 if Bset is used. */

    /* ---------------------------------------------------------------------- */
    /* simplicial factorization */
    /* ---------------------------------------------------------------------- */

    size_t nzmax ;	/* size of i and x */

    void *p ;		/* p [0..ncol], the column pointers */
    void *i ;		/* i [0..nzmax-1], the row indices */
    void *x ;		/* x [0..nzmax-1], the numerical values */
    void *z ;
    void *nz ;		/* nz [0..ncol-1], the # of nonzeros in each column.
			 * i [p [j] ... p [j]+nz[j]-1] contains the row indices,
			 * and the numerical values are in the same locatins
			 * in x. The value of i [p [k]] is always k. */

    void *next ;	/* size ncol+2. next [j] is the next column in i/x */
    void *prev ;	/* size ncol+2. prev [j] is the prior column in i/x.
			 * head of the list is ncol+1, and the tail is ncol. */

    /* ---------------------------------------------------------------------- */
    /* supernodal factorization */
    /* ---------------------------------------------------------------------- */

    /* Note that L->x is shared with the simplicial data structure.  L->x has
     * size L->nzmax for a simplicial factor, and size L->xsize for a supernodal
     * factor. */

    size_t nsuper ;	/* number of supernodes */
    size_t ssize ;	/* size of s, integer part of supernodes */
    size_t xsize ;	/* size of x, real part of supernodes */
    size_t maxcsize ;	/* size of largest update matrix */
    size_t maxesize ;	/* max # of rows in supernodes, excl. triangular part */

    void *super ;	/* size nsuper+1, first col in each supernode */
    void *pi ;		/* size nsuper+1, pointers to integer patterns */
    void *px ;		/* size nsuper+1, pointers to real parts */
    void *s ;		/* size ssize, integer part of supernodes */

    /* ---------------------------------------------------------------------- */
    /* factorization type */
    /* ---------------------------------------------------------------------- */

    int ordering ;	/* ordering method used */

    int is_ll ;		/* TRUE if LL', FALSE if LDL' */
    int is_super ;	/* TRUE if supernodal, FALSE if simplicial */
    int is_monotonic ;	/* TRUE if columns of L appear in order 0..n-1.
			 * Only applicable to simplicial numeric types. */

    /* There are 8 types of factor objects that cholmod_factor can represent
     * (only 6 are used):
     *
     * Numeric types (xtype is not CHOLMOD_PATTERN)
     * --------------------------------------------
     *
     * simplicial LDL':  (is_ll FALSE, is_super FALSE).  Stored in compressed
     *	    column form, using the simplicial components above (nzmax, p, i,
     *	    x, z, nz, next, and prev).  The unit diagonal of L is not stored,
     *	    and D is stored in its place.  There are no supernodes.
     *
     * simplicial LL': (is_ll TRUE, is_super FALSE).  Uses the same storage
     *	    scheme as the simplicial LDL', except that D does not appear.
     *	    The first entry of each column of L is the diagonal entry of
     *	    that column of L.
     *
     * supernodal LDL': (is_ll FALSE, is_super TRUE).  Not used.
     *	    FUTURE WORK:  add support for supernodal LDL'
     *
     * supernodal LL': (is_ll TRUE, is_super TRUE).  A supernodal factor,
     *	    using the supernodal components described above (nsuper, ssize,
     *	    xsize, maxcsize, maxesize, super, pi, px, s, x, and z).
     *
     *
     * Symbolic types (xtype is CHOLMOD_PATTERN)
     * -----------------------------------------
     *
     * simplicial LDL': (is_ll FALSE, is_super FALSE).  Nothing is present
     *	    except Perm and ColCount.
     *
     * simplicial LL': (is_ll TRUE, is_super FALSE).  Identical to the
     *	    simplicial LDL', except for the is_ll flag.
     *
     * supernodal LDL': (is_ll FALSE, is_super TRUE).  Not used.
     *	    FUTURE WORK:  add support for supernodal LDL'
     *
     * supernodal LL': (is_ll TRUE, is_super TRUE).  A supernodal symbolic
     *	    factorization.  The simplicial symbolic information is present
     *	    (Perm and ColCount), as is all of the supernodal factorization
     *	    except for the numerical values (x and z).
     */

    int itype ; /* The integer arrays are Perm, ColCount, p, i, nz,
                 * next, prev, super, pi, px, and s.  If itype is
		 * CHOLMOD_INT, all of these are int arrays.
		 * CHOLMOD_INTLONG: p, pi, px are int64_t, others int.
		 * CHOLMOD_LONG:    all integer arrays are int64_t. */
    int xtype ; /* pattern, real, complex, or zomplex */
    int dtype ; /* x and z double or float */

    int useGPU; /* Indicates the symbolic factorization supports
		 * GPU acceleration */

} cholmod_factor ;


/* -------------------------------------------------------------------------- */
/* cholmod_allocate_factor: allocate a factor (symbolic LL' or LDL') */
/* -------------------------------------------------------------------------- */

cholmod_factor *cholmod_allocate_factor
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_allocate_factor (size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_free_factor:  free a factor */
/* -------------------------------------------------------------------------- */

int cholmod_free_factor
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_free_factor (cholmod_factor **, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_reallocate_factor:  change the # entries in a factor */
/* -------------------------------------------------------------------------- */

int cholmod_reallocate_factor
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_reallocate_factor (size_t, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_change_factor:  change the type of factor (e.g., LDL' to LL') */
/* -------------------------------------------------------------------------- */

int cholmod_change_factor
(
    /* ---- input ---- */
    int to_xtype,	/* to CHOLMOD_PATTERN, _REAL, _COMPLEX, _ZOMPLEX */
    int to_ll,		/* TRUE: convert to LL', FALSE: LDL' */
    int to_super,	/* TRUE: convert to supernodal, FALSE: simplicial */
    int to_packed,	/* TRUE: pack simplicial columns, FALSE: do not pack */
    int to_monotonic,	/* TRUE: put simplicial columns in order, FALSE: not */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_change_factor ( int, int, int, int, int, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_pack_factor:  pack the columns of a factor */
/* -------------------------------------------------------------------------- */

/* Pack the columns of a simplicial factor.  Unlike cholmod_change_factor,
 * it can pack the columns of a factor even if they are not stored in their
 * natural order (non-monotonic). */

int cholmod_pack_factor
(
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_pack_factor (cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_reallocate_column:  resize a single column of a factor */
/* -------------------------------------------------------------------------- */

int cholmod_reallocate_column
(
    /* ---- input ---- */
    size_t j,		/* the column to reallocate */
    size_t need,	/* required size of column j */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_reallocate_column (size_t, size_t, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_factor_to_sparse:  create a sparse matrix copy of a factor */
/* -------------------------------------------------------------------------- */

/* Only operates on numeric factors, not symbolic ones */

cholmod_sparse *cholmod_factor_to_sparse
(
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to copy, converted to symbolic on output */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_factor_to_sparse (cholmod_factor *,
	cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_copy_factor:  create a copy of a factor */
/* -------------------------------------------------------------------------- */

cholmod_factor *cholmod_copy_factor
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to copy */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_copy_factor (cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_factor_xtype: change the xtype of a factor */
/* -------------------------------------------------------------------------- */

int cholmod_factor_xtype
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (real, complex, or zomplex) */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to change */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_factor_xtype (int, cholmod_factor *, cholmod_common *) ;


/* ========================================================================== */
/* === Core/cholmod_dense =================================================== */
/* ========================================================================== */

/* A dense matrix in column-oriented form.  It has no itype since it contains
 * no integers.  Entry in row i and column j is located in x [i+j*d].
 */

typedef struct cholmod_dense_struct
{
    size_t nrow ;	/* the matrix is nrow-by-ncol */
    size_t ncol ;
    size_t nzmax ;	/* maximum number of entries in the matrix */
    size_t d ;		/* leading dimension (d >= nrow must hold) */
    void *x ;		/* size nzmax or 2*nzmax, if present */
    void *z ;		/* size nzmax, if present */
    int xtype ;		/* pattern, real, complex, or zomplex */
    int dtype ;		/* x and z double or float */

} cholmod_dense ;

/* -------------------------------------------------------------------------- */
/* cholmod_allocate_dense:  allocate a dense matrix (contents uninitialized) */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_allocate_dense
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_allocate_dense (size_t, size_t, size_t, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_zeros: allocate a dense matrix and set it to zero */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_zeros
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_zeros (size_t, size_t, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_ones: allocate a dense matrix and set it to all ones */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_ones
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_ones (size_t, size_t, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_eye: allocate a dense matrix and set it to the identity matrix */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_eye
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_eye (size_t, size_t, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_free_dense:  free a dense matrix */
/* -------------------------------------------------------------------------- */

int cholmod_free_dense
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_free_dense (cholmod_dense **, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_ensure_dense:  ensure a dense matrix has a given size and type */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_ensure_dense
(
    /* ---- input/output ---- */
    cholmod_dense **XHandle,    /* matrix handle to check */
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_ensure_dense (cholmod_dense **, size_t, size_t, size_t,
    int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_sparse_to_dense:  create a dense matrix copy of a sparse matrix */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_sparse_to_dense
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_sparse_to_dense (cholmod_sparse *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_dense_to_sparse:  create a sparse matrix copy of a dense matrix */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_dense_to_sparse
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_dense_to_sparse (cholmod_dense *, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_copy_dense:  create a copy of a dense matrix */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_copy_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_copy_dense (cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_copy_dense2:  copy a dense matrix (pre-allocated) */
/* -------------------------------------------------------------------------- */

int cholmod_copy_dense2
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    /* ---- output --- */
    cholmod_dense *Y,	/* copy of matrix X */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_copy_dense2 (cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_dense_xtype: change the xtype of a dense matrix */
/* -------------------------------------------------------------------------- */

int cholmod_dense_xtype
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (real, complex,or zomplex) */
    /* ---- in/out --- */
    cholmod_dense *X,	/* dense matrix to change */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_dense_xtype (int, cholmod_dense *, cholmod_common *) ;


/* ========================================================================== */
/* === Core/cholmod_triplet ================================================= */
/* ========================================================================== */

/* A sparse matrix stored in triplet form. */

typedef struct cholmod_triplet_struct
{
    size_t nrow ;	/* the matrix is nrow-by-ncol */
    size_t ncol ;
    size_t nzmax ;	/* maximum number of entries in the matrix */
    size_t nnz ;	/* number of nonzeros in the matrix */

    void *i ;		/* i [0..nzmax-1], the row indices */
    void *j ;		/* j [0..nzmax-1], the column indices */
    void *x ;		/* size nzmax or 2*nzmax, if present */
    void *z ;		/* size nzmax, if present */

    int stype ;		/* Describes what parts of the matrix are considered:
			 *
	* 0:  matrix is "unsymmetric": use both upper and lower triangular parts
	*     (the matrix may actually be symmetric in pattern and value, but
	*     both parts are explicitly stored and used).  May be square or
	*     rectangular.
	* >0: matrix is square and symmetric.  Entries in the lower triangular
	*     part are transposed and added to the upper triangular part when
	*     the matrix is converted to cholmod_sparse form.
	* <0: matrix is square and symmetric.  Entries in the upper triangular
	*     part are transposed and added to the lower triangular part when
	*     the matrix is converted to cholmod_sparse form.
	*
	* Note that stype>0 and stype<0 are different for cholmod_sparse and
	* cholmod_triplet.  The reason is simple.  You can permute a symmetric
	* triplet matrix by simply replacing a row and column index with their
	* new row and column indices, via an inverse permutation.  Suppose
	* P = L->Perm is your permutation, and Pinv is an array of size n.
	* Suppose a symmetric matrix A is represent by a triplet matrix T, with
	* entries only in the upper triangular part.  Then the following code:
	*
	*	Ti = T->i ;
	*	Tj = T->j ;
	*	for (k = 0 ; k < n  ; k++) Pinv [P [k]] = k ;
	*	for (k = 0 ; k < nz ; k++) Ti [k] = Pinv [Ti [k]] ;
	*	for (k = 0 ; k < nz ; k++) Tj [k] = Pinv [Tj [k]] ;
	*
	* creates the triplet form of C=P*A*P'.  However, if T initially
	* contains just the upper triangular entries (T->stype = 1), after
	* permutation it has entries in both the upper and lower triangular
	* parts.  These entries should be transposed when constructing the
	* cholmod_sparse form of A, which is what cholmod_triplet_to_sparse
	* does.  Thus:
	*
	*	C = cholmod_triplet_to_sparse (T, 0, &Common) ;
	*
	* will return the matrix C = P*A*P'.
	*
	* Since the triplet matrix T is so simple to generate, it's quite easy
	* to remove entries that you do not want, prior to converting T to the
	* cholmod_sparse form.  So if you include these entries in T, CHOLMOD
	* assumes that there must be a reason (such as the one above).  Thus,
	* no entry in a triplet matrix is ever ignored.
	*/

    int itype ; /* CHOLMOD_LONG: i and j are int64_t.  Otherwise int */
    int xtype ; /* pattern, real, complex, or zomplex */
    int dtype ; /* x and z are double or float */

} cholmod_triplet ;

/* -------------------------------------------------------------------------- */
/* cholmod_allocate_triplet:  allocate a triplet matrix */
/* -------------------------------------------------------------------------- */

cholmod_triplet *cholmod_allocate_triplet
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of T */
    size_t ncol,	/* # of columns of T */
    size_t nzmax,	/* max # of nonzeros of T */
    int stype,		/* stype of T */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_triplet *cholmod_l_allocate_triplet (size_t, size_t, size_t, int, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_free_triplet:  free a triplet matrix */
/* -------------------------------------------------------------------------- */

int cholmod_free_triplet
(
    /* ---- in/out --- */
    cholmod_triplet **T,    /* triplet matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_free_triplet (cholmod_triplet **, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_reallocate_triplet:  change the # of entries in a triplet matrix */
/* -------------------------------------------------------------------------- */

int cholmod_reallocate_triplet
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in T */
    /* ---- in/out --- */
    cholmod_triplet *T,	/* triplet matrix to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_reallocate_triplet (size_t, cholmod_triplet *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_sparse_to_triplet:  create a triplet matrix copy of a sparse matrix*/
/* -------------------------------------------------------------------------- */

cholmod_triplet *cholmod_sparse_to_triplet
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_triplet *cholmod_l_sparse_to_triplet (cholmod_sparse *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_triplet_to_sparse:  create a sparse matrix copy of a triplet matrix*/
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_triplet_to_sparse
(
    /* ---- input ---- */
    cholmod_triplet *T,	/* matrix to copy */
    size_t nzmax,	/* allocate at least this much space in output matrix */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_triplet_to_sparse (cholmod_triplet *, size_t,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_copy_triplet:  create a copy of a triplet matrix */
/* -------------------------------------------------------------------------- */

cholmod_triplet *cholmod_copy_triplet
(
    /* ---- input ---- */
    cholmod_triplet *T,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_triplet *cholmod_l_copy_triplet (cholmod_triplet *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_triplet_xtype: change the xtype of a triplet matrix */
/* -------------------------------------------------------------------------- */

int cholmod_triplet_xtype
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (pattern, real, complex,or zomplex)*/
    /* ---- in/out --- */
    cholmod_triplet *T,	/* triplet matrix to change */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_triplet_xtype (int, cholmod_triplet *, cholmod_common *) ;


/* ========================================================================== */
/* === Core/cholmod_memory ================================================== */
/* ========================================================================== */

/* The user may make use of these, just like malloc and free.  You can even
 * malloc an object and safely free it with cholmod_free, and visa versa
 * (except that the memory usage statistics will be corrupted).  These routines
 * do differ from malloc and free.  If cholmod_free is given a NULL pointer,
 * for example, it does nothing (unlike the ANSI free).  cholmod_realloc does
 * not return NULL if given a non-NULL pointer and a nonzero size, even if it
 * fails (it returns the original pointer and sets an error code in
 * Common->status instead).
 *
 * CHOLMOD keeps track of the amount of memory it has allocated, and so the
 * cholmod_free routine also takes the size of the object being freed.  This
 * is only used for statistics.  If you, the user of CHOLMOD, pass the wrong
 * size, the only consequence is that the memory usage statistics will be
 * corrupted.
 */

void *cholmod_malloc	/* returns pointer to the newly malloc'd block */
(
    /* ---- input ---- */
    size_t n,		/* number of items */
    size_t size,	/* size of each item */
    /* --------------- */
    cholmod_common *Common
) ;

void *cholmod_l_malloc (size_t, size_t, cholmod_common *) ;

void *cholmod_calloc	/* returns pointer to the newly calloc'd block */
(
    /* ---- input ---- */
    size_t n,		/* number of items */
    size_t size,	/* size of each item */
    /* --------------- */
    cholmod_common *Common
) ;

void *cholmod_l_calloc (size_t, size_t, cholmod_common *) ;

void *cholmod_free	/* always returns NULL */
(
    /* ---- input ---- */
    size_t n,		/* number of items */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to free */
    /* --------------- */
    cholmod_common *Common
) ;

void *cholmod_l_free (size_t, size_t, void *, cholmod_common *) ;

void *cholmod_realloc	/* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated block */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to realloc */
    size_t *n,		/* current size on input, nnew on output if successful*/
    /* --------------- */
    cholmod_common *Common
) ;

void *cholmod_l_realloc (size_t, size_t, void *, size_t *, cholmod_common *) ;

int cholmod_realloc_multiple
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated blocks */
    int nint,		/* number of int32_t/int64_t blocks */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* ---- in/out --- */
    void **Iblock,	/* int32_t or int64_t block */
    void **Jblock,	/* int32_t or int64_t block */
    void **Xblock,	/* complex, double, or float block */
    void **Zblock,	/* zomplex case only: double or float block */
    size_t *n,		/* current size of the I,J,X,Z blocks on input,
			 * nnew on output if successful */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_realloc_multiple (size_t, int, int, void **, void **, void **,
    void **, size_t *, cholmod_common *) ;

/* ========================================================================== */
/* === version control ====================================================== */
/* ========================================================================== */

int cholmod_version     /* returns CHOLMOD_VERSION */
(
    /* output, contents not defined on input.  Not used if NULL.
        version [0] = CHOLMOD_MAIN_VERSION
        version [1] = CHOLMOD_SUB_VERSION
        version [2] = CHOLMOD_SUBSUB_VERSION
    */
    int version [3]
) ;

int cholmod_l_version (int version [3]) ;

/* Versions prior to 2.1.1 do not have the above function.  The following
   code fragment will work with any version of CHOLMOD:
   #ifdef CHOLMOD_HAS_VERSION_FUNCTION
   v = cholmod_version (NULL) ;
   #else
   v = CHOLMOD_VERSION ;
   #endif
*/

/* ========================================================================== */
/* === symmetry types ======================================================= */
/* ========================================================================== */

#define CHOLMOD_MM_RECTANGULAR 1
#define CHOLMOD_MM_UNSYMMETRIC 2
#define CHOLMOD_MM_SYMMETRIC 3
#define CHOLMOD_MM_HERMITIAN 4
#define CHOLMOD_MM_SKEW_SYMMETRIC 5
#define CHOLMOD_MM_SYMMETRIC_POSDIAG 6
#define CHOLMOD_MM_HERMITIAN_POSDIAG 7

/* ========================================================================== */
/* === Numerical relop macros =============================================== */
/* ========================================================================== */

/* These macros correctly handle the NaN case.
 *
 *  CHOLMOD_IS_NAN(x):
 *	True if x is NaN.  False otherwise.  The commonly-existing isnan(x)
 *	function could be used, but it's not in Kernighan & Ritchie 2nd edition
 *	(ANSI C89).  It may appear in <math.h>, but I'm not certain about
 *	portability.  The expression x != x is true if and only if x is NaN,
 *	according to the IEEE 754 floating-point standard.
 *
 *  CHOLMOD_IS_ZERO(x):
 *	True if x is zero.  False if x is nonzero, NaN, or +/- Inf.
 *	This is (x == 0) if the compiler is IEEE 754 compliant.
 *
 *  CHOLMOD_IS_NONZERO(x):
 *	True if x is nonzero, NaN, or +/- Inf.  False if x zero.
 *	This is (x != 0) if the compiler is IEEE 754 compliant.
 *
 *  CHOLMOD_IS_LT_ZERO(x):
 *	True if x is < zero or -Inf.  False if x is >= 0, NaN, or +Inf.
 *	This is (x < 0) if the compiler is IEEE 754 compliant.
 *
 *  CHOLMOD_IS_GT_ZERO(x):
 *	True if x is > zero or +Inf.  False if x is <= 0, NaN, or -Inf.
 *	This is (x > 0) if the compiler is IEEE 754 compliant.
 *
 *  CHOLMOD_IS_LE_ZERO(x):
 *	True if x is <= zero or -Inf.  False if x is > 0, NaN, or +Inf.
 *	This is (x <= 0) if the compiler is IEEE 754 compliant.
 */

#ifdef CHOLMOD_WINDOWS

/* Yes, this is exceedingly ugly.  Blame Microsoft, which hopelessly */
/* violates the IEEE 754 floating-point standard in a bizarre way. */
/* If you're using an IEEE 754-compliant compiler, then x != x is true */
/* iff x is NaN.  For Microsoft, (x < x) is true iff x is NaN. */
/* So either way, this macro safely detects a NaN. */
#define CHOLMOD_IS_NAN(x)	(((x) != (x)) || (((x) < (x))))
#define CHOLMOD_IS_ZERO(x)	(((x) == 0.) && !CHOLMOD_IS_NAN(x))
#define CHOLMOD_IS_NONZERO(x)	(((x) != 0.) || CHOLMOD_IS_NAN(x))
#define CHOLMOD_IS_LT_ZERO(x)	(((x) < 0.) && !CHOLMOD_IS_NAN(x))
#define CHOLMOD_IS_GT_ZERO(x)	(((x) > 0.) && !CHOLMOD_IS_NAN(x))
#define CHOLMOD_IS_LE_ZERO(x)	(((x) <= 0.) && !CHOLMOD_IS_NAN(x))

#else

/* These all work properly, according to the IEEE 754 standard ... except on */
/* a PC with windows.  Works fine in Linux on the same PC... */
#define CHOLMOD_IS_NAN(x)	((x) != (x))
#define CHOLMOD_IS_ZERO(x)	((x) == 0.)
#define CHOLMOD_IS_NONZERO(x)	((x) != 0.)
#define CHOLMOD_IS_LT_ZERO(x)	((x) < 0.)
#define CHOLMOD_IS_GT_ZERO(x)	((x) > 0.)
#define CHOLMOD_IS_LE_ZERO(x)	((x) <= 0.)

#endif




/* ========================================================================== */
/* === Include/cholmod_check.h ============================================== */
/* ========================================================================== */

/* CHOLMOD Check module.
 *
 * Routines that check and print the 5 basic data types in CHOLMOD, and 3 kinds
 * of integer vectors (subset, perm, and parent), and read in matrices from a
 * file:
 *
 * cholmod_check_common	    check/print the Common object
 * cholmod_print_common
 *
 * cholmod_check_sparse	    check/print a sparse matrix in column-oriented form
 * cholmod_print_sparse
 *
 * cholmod_check_dense	    check/print a dense matrix
 * cholmod_print_dense
 *
 * cholmod_check_factor	    check/print a Cholesky factorization
 * cholmod_print_factor
 *
 * cholmod_check_triplet    check/print a sparse matrix in triplet form
 * cholmod_print_triplet
 *
 * cholmod_check_subset	    check/print a subset (integer vector in given range)
 * cholmod_print_subset
 *
 * cholmod_check_perm	    check/print a permutation (an integer vector)
 * cholmod_print_perm
 *
 * cholmod_check_parent	    check/print an elimination tree (an integer vector)
 * cholmod_print_parent
 *
 * cholmod_read_triplet	    read a matrix in triplet form (any Matrix Market
 *			    "coordinate" format, or a generic triplet format).
 *
 * cholmod_read_sparse	    read a matrix in sparse form (same file format as
 *			    cholmod_read_triplet).
 *
 * cholmod_read_dense	    read a dense matrix (any Matrix Market "array"
 *			    format, or a generic dense format).
 *
 * cholmod_write_sparse	    write a sparse matrix to a Matrix Market file.
 *
 * cholmod_write_dense	    write a dense matrix to a Matrix Market file.
 *
 * cholmod_print_common and cholmod_check_common are the only two routines that
 * you may call after calling cholmod_finish.
 *
 * Requires the Core module.  Not required by any CHOLMOD module, except when
 * debugging is enabled (in which case all modules require the Check module).
 *
 * See cholmod_read.c for a description of the file formats supported by the
 * cholmod_read_* routines.
 */

#ifndef NCHECK

/* -------------------------------------------------------------------------- */
/* cholmod_check_common:  check the Common object */
/* -------------------------------------------------------------------------- */

int cholmod_check_common
(
    cholmod_common *Common
) ;

int cholmod_l_check_common (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_common:  print the Common object */
/* -------------------------------------------------------------------------- */

int cholmod_print_common
(
    /* ---- input ---- */
    const char *name,	/* printed name of Common object */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_common (const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_gpu_stats:  print the GPU / CPU statistics */
/* -------------------------------------------------------------------------- */

int cholmod_gpu_stats   (cholmod_common *) ;
int cholmod_l_gpu_stats (cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_sparse:  check a sparse matrix */
/* -------------------------------------------------------------------------- */

int cholmod_check_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_sparse (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_sparse */
/* -------------------------------------------------------------------------- */

int cholmod_print_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to print */
    const char *name,	/* printed name of sparse matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_sparse (cholmod_sparse *, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_dense:  check a dense matrix */
/* -------------------------------------------------------------------------- */

int cholmod_check_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* dense matrix to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_dense (cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_dense:  print a dense matrix */
/* -------------------------------------------------------------------------- */

int cholmod_print_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* dense matrix to print */
    const char *name,	/* printed name of dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_dense (cholmod_dense *, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_factor:  check a factor */
/* -------------------------------------------------------------------------- */

int cholmod_check_factor
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_factor (cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_factor:  print a factor */
/* -------------------------------------------------------------------------- */

int cholmod_print_factor
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to print */
    const char *name,	/* printed name of factor */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_factor (cholmod_factor *, const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_triplet:  check a sparse matrix in triplet form */
/* -------------------------------------------------------------------------- */

int cholmod_check_triplet
(
    /* ---- input ---- */
    cholmod_triplet *T,	/* triplet matrix to check */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_triplet (cholmod_triplet *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_triplet:  print a triplet matrix */
/* -------------------------------------------------------------------------- */

int cholmod_print_triplet
(
    /* ---- input ---- */
    cholmod_triplet *T,	/* triplet matrix to print */
    const char *name,	/* printed name of triplet matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_triplet (cholmod_triplet *, const char *, cholmod_common *);

/* -------------------------------------------------------------------------- */
/* cholmod_check_subset:  check a subset */
/* -------------------------------------------------------------------------- */

int cholmod_check_subset
(
    /* ---- input ---- */
    int32_t *Set,	/* Set [0:len-1] is a subset of 0:n-1.  Duplicates OK */
    int64_t len,        /* size of Set (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_subset (int64_t *, int64_t, size_t,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_subset:  print a subset */
/* -------------------------------------------------------------------------- */

int cholmod_print_subset
(
    /* ---- input ---- */
    int32_t *Set,	/* Set [0:len-1] is a subset of 0:n-1.  Duplicates OK */
    int64_t len,        /* size of Set (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    const char *name,	/* printed name of Set */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_subset (int64_t *, int64_t, size_t,
    const char *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_perm:  check a permutation */
/* -------------------------------------------------------------------------- */

int cholmod_check_perm
(
    /* ---- input ---- */
    int32_t *Perm,	/* Perm [0:len-1] is a permutation of subset of 0:n-1 */
    size_t len,		/* size of Perm (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_perm (int64_t *, size_t, size_t, cholmod_common *);

/* -------------------------------------------------------------------------- */
/* cholmod_print_perm:  print a permutation vector */
/* -------------------------------------------------------------------------- */

int cholmod_print_perm
(
    /* ---- input ---- */
    int32_t *Perm,	/* Perm [0:len-1] is a permutation of subset of 0:n-1 */
    size_t len,		/* size of Perm (an integer array) */
    size_t n,		/* 0:n-1 is valid range */
    const char *name,	/* printed name of Perm */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_perm (int64_t *, size_t, size_t, const char *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_check_parent:  check an elimination tree */
/* -------------------------------------------------------------------------- */

int cholmod_check_parent
(
    /* ---- input ---- */
    int32_t *Parent,	/* Parent [0:n-1] is an elimination tree */
    size_t n,		/* size of Parent */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_check_parent (int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_print_parent */
/* -------------------------------------------------------------------------- */

int cholmod_print_parent
(
    /* ---- input ---- */
    int32_t *Parent,	/* Parent [0:n-1] is an elimination tree */
    size_t n,		/* size of Parent */
    const char *name,	/* printed name of Parent */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_print_parent (int64_t *, size_t, const char *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_sparse: read a sparse matrix from a file */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_read_sparse
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_read_sparse (FILE *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_triplet: read a triplet matrix from a file */
/* -------------------------------------------------------------------------- */

cholmod_triplet *cholmod_read_triplet
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_triplet *cholmod_l_read_triplet (FILE *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_read_dense: read a dense matrix from a file */
/* -------------------------------------------------------------------------- */

cholmod_dense *cholmod_read_dense
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_read_dense (FILE *, cholmod_common *) ; 

/* -------------------------------------------------------------------------- */
/* cholmod_read_matrix: read a sparse or dense matrix from a file */
/* -------------------------------------------------------------------------- */

void *cholmod_read_matrix
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    int prefer,		/* If 0, a sparse matrix is always return as a
			 *	cholmod_triplet form.  It can have any stype
			 *	(symmetric-lower, unsymmetric, or
			 *	symmetric-upper).
			 * If 1, a sparse matrix is returned as an unsymmetric
			 *	cholmod_sparse form (A->stype == 0), with both
			 *	upper and lower triangular parts present.
			 *	This is what the MATLAB mread mexFunction does,
			 *	since MATLAB does not have an stype.
			 * If 2, a sparse matrix is returned with an stype of 0
			 *	or 1 (unsymmetric, or symmetric with upper part
			 *	stored).
			 * This argument has no effect for dense matrices.
			 */
    /* ---- output---- */
    int *mtype,		/* CHOLMOD_TRIPLET, CHOLMOD_SPARSE or CHOLMOD_DENSE */
    /* --------------- */
    cholmod_common *Common
) ;

void *cholmod_l_read_matrix (FILE *, int, int *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_write_sparse: write a sparse matrix to a file */
/* -------------------------------------------------------------------------- */

int cholmod_write_sparse
(
    /* ---- input ---- */
    FILE *f,		    /* file to write to, must already be open */
    cholmod_sparse *A,	    /* matrix to print */
    cholmod_sparse *Z,	    /* optional matrix with pattern of explicit zeros */
    const char *comments,   /* optional filename of comments to include */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_write_sparse (FILE *, cholmod_sparse *, cholmod_sparse *,
    const char *c, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_write_dense: write a dense matrix to a file */
/* -------------------------------------------------------------------------- */

int cholmod_write_dense
(
    /* ---- input ---- */
    FILE *f,		    /* file to write to, must already be open */
    cholmod_dense *X,	    /* matrix to print */
    const char *comments,   /* optional filename of comments to include */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_write_dense (FILE *, cholmod_dense *, const char *,
    cholmod_common *) ;
#endif






/* ========================================================================== */
/* === Include/cholmod_cholesky.h =========================================== */
/* ========================================================================== */

/* CHOLMOD Cholesky module.
 *
 * Sparse Cholesky routines: analysis, factorization, and solve.
 *
 * The primary routines are all that a user requires to order, analyze, and
 * factorize a sparse symmetric positive definite matrix A (or A*A'), and
 * to solve Ax=b (or A*A'x=b).  The primary routines rely on the secondary
 * routines, the CHOLMOD Core module, and the AMD and COLAMD packages.  They
 * make optional use of the CHOLMOD Supernodal and Partition modules, the
 * METIS package, and the CCOLAMD package.
 *
 * Primary routines:
 * -----------------
 *
 * cholmod_analyze		order and analyze (simplicial or supernodal)
 * cholmod_factorize		simplicial or supernodal Cholesky factorization
 * cholmod_solve		solve a linear system (simplicial or supernodal)
 * cholmod_solve2		like cholmod_solve, but reuse workspace
 * cholmod_spsolve		solve a linear system (sparse x and b)
 *
 * Secondary routines:
 * ------------------
 *
 * cholmod_analyze_p		analyze, with user-provided permutation or f set
 * cholmod_factorize_p		factorize, with user-provided permutation or f
 * cholmod_analyze_ordering	analyze a fill-reducing ordering
 * cholmod_etree		find the elimination tree
 * cholmod_rowcolcounts		compute the row/column counts of L
 * cholmod_amd			order using AMD
 * cholmod_colamd		order using COLAMD
 * cholmod_rowfac		incremental simplicial factorization
 * cholmod_rowfac_mask		rowfac, specific to LPDASA
 * cholmod_rowfac_mask2         rowfac, specific to LPDASA
 * cholmod_row_subtree		find the nonzero pattern of a row of L
 * cholmod_resymbol		recompute the symbolic pattern of L
 * cholmod_resymbol_noperm	recompute the symbolic pattern of L, no L->Perm
 * cholmod_postorder		postorder a tree
 *
 * Requires the Core module, and two packages: AMD and COLAMD.
 * Optionally uses the Supernodal and Partition modules.
 * Required by the Partition module.
 */

#ifndef NCHOLESKY

/* -------------------------------------------------------------------------- */
/* cholmod_analyze:  order and analyze (simplicial or supernodal) */
/* -------------------------------------------------------------------------- */

/* Orders and analyzes A, AA', PAP', or PAA'P' and returns a symbolic factor
 * that can later be passed to cholmod_factorize. */

cholmod_factor *cholmod_analyze 
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order and analyze */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_analyze (cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_analyze_p:  analyze, with user-provided permutation or f set */
/* -------------------------------------------------------------------------- */

/* Orders and analyzes A, AA', PAP', PAA'P', FF', or PFF'P and returns a
 * symbolic factor that can later be passed to cholmod_factorize, where
 * F = A(:,fset) if fset is not NULL and A->stype is zero.
 * UserPerm is tried if non-NULL.  */

cholmod_factor *cholmod_analyze_p
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order and analyze */
    int32_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_analyze_p (cholmod_sparse *, int64_t *,
    int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_analyze_p2:  analyze for sparse Cholesky or sparse QR */
/* -------------------------------------------------------------------------- */

cholmod_factor *cholmod_analyze_p2
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to order and analyze */
    int32_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_factor *cholmod_l_analyze_p2 (int, cholmod_sparse *, int64_t *,
    int64_t *, size_t, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_factorize:  simplicial or supernodal Cholesky factorization */
/* -------------------------------------------------------------------------- */

/* Factorizes PAP' (or PAA'P' if A->stype is 0), using a factor obtained
 * from cholmod_analyze.  The analysis can be re-used simply by calling this
 * routine a second time with another matrix.  A must have the same nonzero
 * pattern as that passed to cholmod_analyze. */

int cholmod_factorize 
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    /* ---- in/out --- */
    cholmod_factor *L,	/* resulting factorization */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_factorize (cholmod_sparse *, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_factorize_p:  factorize, with user-provided permutation or fset */
/* -------------------------------------------------------------------------- */

/* Same as cholmod_factorize, but with more options. */

int cholmod_factorize_p
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- in/out --- */
    cholmod_factor *L,	/* resulting factorization */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_factorize_p (cholmod_sparse *, double *, int64_t *,
    size_t, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_solve:  solve a linear system (simplicial or supernodal) */
/* -------------------------------------------------------------------------- */

/* Solves one of many linear systems with a dense right-hand-side, using the
 * factorization from cholmod_factorize (or as modified by any other CHOLMOD
 * routine).  D is identity for LL' factorizations. */

#define CHOLMOD_A    0		/* solve Ax=b */
#define CHOLMOD_LDLt 1		/* solve LDL'x=b */
#define CHOLMOD_LD   2		/* solve LDx=b */
#define CHOLMOD_DLt  3		/* solve DL'x=b */
#define CHOLMOD_L    4		/* solve Lx=b */
#define CHOLMOD_Lt   5		/* solve L'x=b */
#define CHOLMOD_D    6		/* solve Dx=b */
#define CHOLMOD_P    7		/* permute x=Px */
#define CHOLMOD_Pt   8		/* permute x=P'x */

cholmod_dense *cholmod_solve	/* returns the solution X */
(
    /* ---- input ---- */
    int sys,		/* system to solve */
    cholmod_factor *L,	/* factorization to use */
    cholmod_dense *B,	/* right-hand-side */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_dense *cholmod_l_solve (int, cholmod_factor *, cholmod_dense *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_solve2:  like cholmod_solve, but with reusable workspace */
/* -------------------------------------------------------------------------- */

int cholmod_solve2     /* returns TRUE on success, FALSE on failure */
(
    /* ---- input ---- */
    int sys,		            /* system to solve */
    cholmod_factor *L,	            /* factorization to use */
    cholmod_dense *B,               /* right-hand-side */
    cholmod_sparse *Bset,
    /* ---- output --- */
    cholmod_dense **X_Handle,       /* solution, allocated if need be */
    cholmod_sparse **Xset_Handle,
    /* ---- workspace  */
    cholmod_dense **Y_Handle,       /* workspace, or NULL */
    cholmod_dense **E_Handle,       /* workspace, or NULL */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_solve2 (int, cholmod_factor *, cholmod_dense *, cholmod_sparse *,
    cholmod_dense **, cholmod_sparse **, cholmod_dense **, cholmod_dense **,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_spsolve:  solve a linear system with a sparse right-hand-side */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_spsolve
(
    /* ---- input ---- */
    int sys,		/* system to solve */
    cholmod_factor *L,	/* factorization to use */
    cholmod_sparse *B,	/* right-hand-side */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_spsolve (int, cholmod_factor *, cholmod_sparse *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_etree: find the elimination tree of A or A'*A */
/* -------------------------------------------------------------------------- */

int cholmod_etree
(
    /* ---- input ---- */
    cholmod_sparse *A,
    /* ---- output --- */
    int32_t *Parent,	/* size ncol.  Parent [j] = p if p is the parent of j */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_etree (cholmod_sparse *, int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowcolcounts: compute the row/column counts of L */
/* -------------------------------------------------------------------------- */

int cholmod_rowcolcounts
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int32_t *Parent,	/* size nrow.  Parent [i] = p if p is the parent of i */
    int32_t *Post,	/* size nrow.  Post [k] = i if i is the kth node in
			 * the postordered etree. */
    /* ---- output --- */
    int32_t *RowCount,	/* size nrow. RowCount [i] = # entries in the ith row of
			 * L, including the diagonal. */
    int32_t *ColCount,	/* size nrow. ColCount [i] = # entries in the ith
			 * column of L, including the diagonal. */
    int32_t *First,	/* size nrow.  First [i] = k is the least postordering
			 * of any descendant of i. */
    int32_t *Level,	/* size nrow.  Level [i] is the length of the path from
			 * i to the root, with Level [root] = 0. */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowcolcounts (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, int64_t *,
    int64_t *, int64_t *, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_analyze_ordering:  analyze a fill-reducing ordering */
/* -------------------------------------------------------------------------- */

int cholmod_analyze_ordering
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int ordering,	/* ordering method used */
    int32_t *Perm,	/* size n, fill-reducing permutation to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Parent,	/* size n, elimination tree */
    int32_t *Post,	/* size n, postordering of elimination tree */
    int32_t *ColCount,	/* size n, nnz in each column of L */
    /* ---- workspace  */
    int32_t *First,	/* size nworkspace for cholmod_postorder */
    int32_t *Level,	/* size n workspace for cholmod_postorder */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_analyze_ordering (cholmod_sparse *, int, int64_t *,
    int64_t *, size_t, int64_t *, int64_t *, int64_t *, int64_t *, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_amd:  order using AMD */
/* -------------------------------------------------------------------------- */

/* Finds a permutation P to reduce fill-in in the factorization of P*A*P'
 * or P*A*A'P' */

int cholmod_amd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_amd (cholmod_sparse *, int64_t *, size_t,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_colamd:  order using COLAMD */
/* -------------------------------------------------------------------------- */

/* Finds a permutation P to reduce fill-in in the factorization of P*A*A'*P'.
 * Orders F*F' where F = A (:,fset) if fset is not NULL */

int cholmod_colamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with a coletree postorder */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_colamd (cholmod_sparse *, int64_t *, size_t, int,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowfac:  incremental simplicial factorization */
/* -------------------------------------------------------------------------- */

/* Partial or complete simplicial factorization.  Rows and columns kstart:kend-1
 * of L and D must be initially equal to rows/columns kstart:kend-1 of the
 * identity matrix.   Row k can only be factorized if all descendants of node
 * k in the elimination tree have been factorized. */

int cholmod_rowfac
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    size_t kstart,	/* first row to factorize */
    size_t kend,	/* last row to factorize is kend-1 */
    /* ---- in/out --- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowfac (cholmod_sparse *, cholmod_sparse *, double *, size_t,
    size_t, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowfac_mask:  incremental simplicial factorization */
/* -------------------------------------------------------------------------- */

/* cholmod_rowfac_mask is a version of cholmod_rowfac that is specific to
 * LPDASA.  It is unlikely to be needed by any other application. */

int cholmod_rowfac_mask
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    size_t kstart,	/* first row to factorize */
    size_t kend,	/* last row to factorize is kend-1 */
    int32_t *mask,	/* if mask[i] >= 0, then set row i to zero */
    int32_t *RLinkUp,	/* link list of rows to compute */
    /* ---- in/out --- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowfac_mask (cholmod_sparse *, cholmod_sparse *, double *, size_t,
    size_t, int64_t *, int64_t *, cholmod_factor *,
    cholmod_common *) ;

int cholmod_rowfac_mask2
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    double beta [2],	/* factorize beta*I+A or beta*I+A'*A */
    size_t kstart,	/* first row to factorize */
    size_t kend,	/* last row to factorize is kend-1 */
    int32_t *mask,	/* if mask[i] >= maskmark, then set row i to zero */
    int32_t maskmark,
    int32_t *RLinkUp,	/* link list of rows to compute */
    /* ---- in/out --- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowfac_mask2 (cholmod_sparse *, cholmod_sparse *, double *,
    size_t, size_t, int64_t *, int64_t, int64_t *,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_row_subtree:  find the nonzero pattern of a row of L */
/* -------------------------------------------------------------------------- */

/* Find the nonzero pattern of x for the system Lx=b where L = (0:k-1,0:k-1)
 * and b = kth column of A or A*A' (rows 0 to k-1 only) */

int cholmod_row_subtree
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    cholmod_sparse *F,	/* used for A*A' case only. F=A' or A(:,fset)' */
    size_t k,		/* row k of L */
    int32_t *Parent,	/* elimination tree */
    /* ---- output --- */
    cholmod_sparse *R,	/* pattern of L(k,:), n-by-1 with R->nzmax >= n */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_row_subtree (cholmod_sparse *, cholmod_sparse *, size_t,
    int64_t *, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_lsolve_pattern: find the nonzero pattern of x=L\b */
/* -------------------------------------------------------------------------- */

int cholmod_lsolve_pattern
(
    /* ---- input ---- */
    cholmod_sparse *B,	/* sparse right-hand-side (a single sparse column) */
    cholmod_factor *L,	/* the factor L from which parent(i) is derived */
    /* ---- output --- */
    cholmod_sparse *X,	/* pattern of X=L\B, n-by-1 with X->nzmax >= n */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_lsolve_pattern (cholmod_sparse *, cholmod_factor *,
    cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_row_lsubtree:  find the nonzero pattern of a row of L */
/* -------------------------------------------------------------------------- */

/* Identical to cholmod_row_subtree, except that it finds the elimination tree
 * from L itself. */

int cholmod_row_lsubtree
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *Fi, size_t fnz, /* nonzero pattern of kth row of A', not required
			      * for the symmetric case.  Need not be sorted. */
    size_t k,		/* row k of L */
    cholmod_factor *L,	/* the factor L from which parent(i) is derived */
    /* ---- output --- */
    cholmod_sparse *R,	/* pattern of L(k,:), n-by-1 with R->nzmax >= n */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_row_lsubtree (cholmod_sparse *, int64_t *, size_t,
    size_t, cholmod_factor *, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_resymbol:  recompute the symbolic pattern of L */
/* -------------------------------------------------------------------------- */

/* Remove entries from L that are not in the factorization of P*A*P', P*A*A'*P',
 * or P*F*F'*P' (depending on A->stype and whether fset is NULL or not).
 *
 * cholmod_resymbol is the same as cholmod_resymbol_noperm, except that it
 * first permutes A according to L->Perm.  A can be upper/lower/unsymmetric,
 * in contrast to cholmod_resymbol_noperm (which can be lower or unsym). */

int cholmod_resymbol 
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int pack,		/* if TRUE, pack the columns of L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factorization, entries pruned on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_resymbol (cholmod_sparse *, int64_t *, size_t, int,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_resymbol_noperm:  recompute the symbolic pattern of L, no L->Perm */
/* -------------------------------------------------------------------------- */

/* Remove entries from L that are not in the factorization of A, A*A',
 * or F*F' (depending on A->stype and whether fset is NULL or not). */

int cholmod_resymbol_noperm
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int pack,		/* if TRUE, pack the columns of L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factorization, entries pruned on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_resymbol_noperm (cholmod_sparse *, int64_t *, size_t, int,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rcond:  compute rough estimate of reciprocal of condition number */
/* -------------------------------------------------------------------------- */

double cholmod_rcond	    /* return min(diag(L)) / max(diag(L)) */
(
    /* ---- input ---- */
    cholmod_factor *L,
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_rcond (cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_postorder: Compute the postorder of a tree */
/* -------------------------------------------------------------------------- */

int32_t cholmod_postorder	/* return # of nodes postordered */
(
    /* ---- input ---- */
    int32_t *Parent,	/* size n. Parent [j] = p if p is the parent of j */
    size_t n,
    int32_t *Weight_p,	/* size n, optional. Weight [j] is weight of node j */
    /* ---- output --- */
    int32_t *Post,	/* size n. Post [k] = j is kth in postordered tree */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_postorder (int64_t *, size_t,
    int64_t *, int64_t *, cholmod_common *) ;

#endif


/* ========================================================================== */
/* === Include/cholmod_matrixops.h ========================================== */
/* ========================================================================== */

/* CHOLMOD MatrixOps module.
 *
 * Basic operations on sparse and dense matrices.
 *
 * cholmod_drop		    A = entries in A with abs. value >= tol
 * cholmod_norm_dense	    s = norm (X), 1-norm, inf-norm, or 2-norm
 * cholmod_norm_sparse	    s = norm (A), 1-norm or inf-norm
 * cholmod_horzcat	    C = [A,B]
 * cholmod_scale	    A = diag(s)*A, A*diag(s), s*A or diag(s)*A*diag(s)
 * cholmod_sdmult	    Y = alpha*(A*X) + beta*Y or alpha*(A'*X) + beta*Y
 * cholmod_ssmult	    C = A*B
 * cholmod_submatrix	    C = A (i,j), where i and j are arbitrary vectors
 * cholmod_vertcat	    C = [A ; B]
 *
 * A, B, C: sparse matrices (cholmod_sparse)
 * X, Y: dense matrices (cholmod_dense)
 * s: scalar or vector
 *
 * Requires the Core module.  Not required by any other CHOLMOD module.
 */

#ifndef NMATRIXOPS

/* -------------------------------------------------------------------------- */
/* cholmod_drop:  drop entries with small absolute value */
/* -------------------------------------------------------------------------- */

int cholmod_drop
(
    /* ---- input ---- */
    double tol,		/* keep entries with absolute value > tol */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to drop entries from */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_drop (double, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_norm_dense:  s = norm (X), 1-norm, inf-norm, or 2-norm */
/* -------------------------------------------------------------------------- */

double cholmod_norm_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_norm_dense (cholmod_dense *, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_norm_sparse:  s = norm (A), 1-norm or inf-norm */
/* -------------------------------------------------------------------------- */

double cholmod_norm_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
    /* --------------- */
    cholmod_common *Common
) ;

double cholmod_l_norm_sparse (cholmod_sparse *, int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_horzcat:  C = [A,B] */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_horzcat
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to concatenate */
    cholmod_sparse *B,	/* right matrix to concatenate */
    int values,		/* if TRUE compute the numerical values of C */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_horzcat (cholmod_sparse *, cholmod_sparse *, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_scale:  A = diag(s)*A, A*diag(s), s*A or diag(s)*A*diag(s) */
/* -------------------------------------------------------------------------- */

/* scaling modes, selected by the scale input parameter: */
#define CHOLMOD_SCALAR 0	/* A = s*A */
#define CHOLMOD_ROW 1		/* A = diag(s)*A */
#define CHOLMOD_COL 2		/* A = A*diag(s) */
#define CHOLMOD_SYM 3		/* A = diag(s)*A*diag(s) */

int cholmod_scale
(
    /* ---- input ---- */
    cholmod_dense *S,	/* scale factors (scalar or vector) */
    int scale,		/* type of scaling to compute */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to scale */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_scale (cholmod_dense *, int, cholmod_sparse *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_sdmult:  Y = alpha*(A*X) + beta*Y or alpha*(A'*X) + beta*Y */
/* -------------------------------------------------------------------------- */

/* Sparse matrix times dense matrix */

int cholmod_sdmult
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to multiply */
    int transpose,	/* use A if 0, or A' otherwise */
    double alpha [2],   /* scale factor for A */
    double beta [2],    /* scale factor for Y */
    cholmod_dense *X,	/* dense matrix to multiply */
    /* ---- in/out --- */
    cholmod_dense *Y,	/* resulting dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_sdmult (cholmod_sparse *, int, double *, double *,
    cholmod_dense *, cholmod_dense *Y, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_ssmult:  C = A*B */
/* -------------------------------------------------------------------------- */

/* Sparse matrix times sparse matrix */

cholmod_sparse *cholmod_ssmult
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to multiply */
    cholmod_sparse *B,	/* right matrix to multiply */
    int stype,		/* requested stype of C */
    int values,		/* TRUE: do numerical values, FALSE: pattern only */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_ssmult (cholmod_sparse *, cholmod_sparse *, int, int,
    int, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_submatrix:  C = A (r,c), where i and j are arbitrary vectors */
/* -------------------------------------------------------------------------- */

/* rsize < 0 denotes ":" in MATLAB notation, or more precisely 0:(A->nrow)-1.
 * In this case, r can be NULL.  An rsize of zero, or r = NULL and rsize >= 0,
 * denotes "[ ]" in MATLAB notation (the empty set).
 * Similar rules hold for csize.
 */

cholmod_sparse *cholmod_submatrix
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to subreference */
    int32_t *rset,	/* set of row indices, duplicates OK */
    int64_t rsize,	/* size of r; rsize < 0 denotes ":" */
    int32_t *cset,	/* set of column indices, duplicates OK */
    int64_t csize,	/* size of c; csize < 0 denotes ":" */
    int values,		/* if TRUE compute the numerical values of C */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_submatrix (cholmod_sparse *, int64_t *,
    int64_t, int64_t *, int64_t, int, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_vertcat:  C = [A ; B] */
/* -------------------------------------------------------------------------- */

cholmod_sparse *cholmod_vertcat
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to concatenate */
    cholmod_sparse *B,	/* right matrix to concatenate */
    int values,		/* if TRUE compute the numerical values of C */
    /* --------------- */
    cholmod_common *Common
) ;

cholmod_sparse *cholmod_l_vertcat (cholmod_sparse *, cholmod_sparse *, int,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_symmetry: determine if a sparse matrix is symmetric */
/* -------------------------------------------------------------------------- */

int cholmod_symmetry
(
    /* ---- input ---- */
    cholmod_sparse *A,
    int option,
    /* ---- output ---- */
    int32_t *xmatched,
    int32_t *pmatched,
    int32_t *nzoffdiag,
    int32_t *nzdiag,
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_symmetry (cholmod_sparse *, int, int64_t *,
    int64_t *, int64_t *, int64_t *,
    cholmod_common *) ;

#endif




/* ========================================================================== */
/* === Include/cholmod_modify.h ============================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_modify.h.
 * Copyright (C) 2005-2006, Timothy A. Davis and William W. Hager
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* CHOLMOD Modify module.
 *
 * Sparse Cholesky modification routines: update / downdate / rowadd / rowdel.
 * Can also modify a corresponding solution to Lx=b when L is modified.  This
 * module is most useful when applied on a Cholesky factorization computed by
 * the Cholesky module, but it does not actually require the Cholesky module.
 * The Core module can create an identity Cholesky factorization (LDL' where
 * L=D=I) that can then by modified by these routines.
 *
 * Primary routines:
 * -----------------
 *
 * cholmod_updown	    multiple rank update/downdate
 * cholmod_rowadd	    add a row to an LDL' factorization
 * cholmod_rowdel	    delete a row from an LDL' factorization
 *
 * Secondary routines:
 * -------------------
 *
 * cholmod_updown_solve	    update/downdate, and modify solution to Lx=b
 * cholmod_updown_mark	    update/downdate, and modify solution to partial Lx=b
 * cholmod_updown_mask	    update/downdate for LPDASA
 * cholmod_updown_mask2     update/downdate for LPDASA
 * cholmod_rowadd_solve	    add a row, and update solution to Lx=b
 * cholmod_rowadd_mark	    add a row, and update solution to partial Lx=b
 * cholmod_rowdel_solve	    delete a row, and downdate Lx=b
 * cholmod_rowdel_mark	    delete a row, and downdate solution to partial Lx=b
 *
 * Requires the Core module.  Not required by any other CHOLMOD module.
 */


#ifndef NMODIFY

/* -------------------------------------------------------------------------- */
/* cholmod_updown:  multiple rank update/downdate */
/* -------------------------------------------------------------------------- */

/* Compute the new LDL' factorization of LDL'+CC' (an update) or LDL'-CC'
 * (a downdate).  The factor object L need not be an LDL' factorization; it
 * is converted to one if it isn't. */

int cholmod_updown 
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown (int, cholmod_sparse *, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_updown_solve:  update/downdate, and modify solution to Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_updown, except that it also updates/downdates the
 * solution to Lx=b+DeltaB.  x and b must be n-by-1 dense matrices.  b is not
 * need as input to this routine, but a sparse change to b is (DeltaB).  Only
 * entries in DeltaB corresponding to columns modified in L are accessed; the
 * rest must be zero. */

int cholmod_updown_solve
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_solve (int, cholmod_sparse *, cholmod_factor *,
    cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_updown_mark:  update/downdate, and modify solution to partial Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_updown_solve, except only part of L is used in
 * the update/downdate of the solution to Lx=b.  This routine is an "expert"
 * routine.  It is meant for use in LPDASA only.  See cholmod_updown.c for
 * a description of colmark. */

int cholmod_updown_mark
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_mark (int, cholmod_sparse *, int64_t *,
    cholmod_factor *, cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_updown_mask:  update/downdate, for LPDASA */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_updown_mark, except has an additional "mask"
 * argument.  This routine is an "expert" routine.  It is meant for use in
 * LPDASA only.  See cholmod_updown.c for a description of mask. */

int cholmod_updown_mask
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    int32_t *mask,	/* size n */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_mask (int, cholmod_sparse *, int64_t *,
    int64_t *, cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

int cholmod_updown_mask2
(
    /* ---- input ---- */
    int update,		/* TRUE for update, FALSE for downdate */
    cholmod_sparse *C,	/* the incoming sparse update */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    int32_t *mask,	/* size n */
    int32_t maskmark,
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_updown_mask2 (int, cholmod_sparse *, int64_t *,
    int64_t *, int64_t, cholmod_factor *, cholmod_dense *,
    cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowadd:  add a row to an LDL' factorization (a rank-2 update) */
/* -------------------------------------------------------------------------- */

/* cholmod_rowadd adds a row to the LDL' factorization.  It computes the kth
 * row and kth column of L, and then updates the submatrix L (k+1:n,k+1:n)
 * accordingly.  The kth row and column of L must originally be equal to the
 * kth row and column of the identity matrix.  The kth row/column of L is
 * computed as the factorization of the kth row/column of the matrix to
 * factorize, which is provided as a single n-by-1 sparse matrix R. */

int cholmod_rowadd 
(
    /* ---- input ---- */
    size_t k,		/* row/column index to add */
    cholmod_sparse *R,	/* row/column of matrix to factorize (n-by-1) */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowadd (size_t, cholmod_sparse *, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowadd_solve:  add a row, and update solution to Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowadd, and also updates the solution to Lx=b
 * See cholmod_updown for a description of how Lx=b is updated.  There is on
 * additional parameter:  bk specifies the new kth entry of b. */

int cholmod_rowadd_solve
(
    /* ---- input ---- */
    size_t k,		/* row/column index to add */
    cholmod_sparse *R,	/* row/column of matrix to factorize (n-by-1) */
    double bk [2],	/* kth entry of the right-hand-side b */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowadd_solve (size_t, cholmod_sparse *, double *,
    cholmod_factor *, cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowadd_mark:  add a row, and update solution to partial Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowadd_solve, except only part of L is used in
 * the update/downdate of the solution to Lx=b.  This routine is an "expert"
 * routine.  It is meant for use in LPDASA only.  */

int cholmod_rowadd_mark
(
    /* ---- input ---- */
    size_t k,		/* row/column index to add */
    cholmod_sparse *R,	/* row/column of matrix to factorize (n-by-1) */
    double bk [2],	/* kth entry of the right hand side, b */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowadd_mark (size_t, cholmod_sparse *, double *,
    int64_t *, cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowdel:  delete a row from an LDL' factorization (a rank-2 update) */
/* -------------------------------------------------------------------------- */

/* Sets the kth row and column of L to be the kth row and column of the identity
 * matrix, and updates L(k+1:n,k+1:n) accordingly.   To reduce the running time,
 * the caller can optionally provide the nonzero pattern (or an upper bound) of
 * kth row of L, as the sparse n-by-1 vector R.  Provide R as NULL if you want
 * CHOLMOD to determine this itself, which is easier for the caller, but takes
 * a little more time.
 */

int cholmod_rowdel 
(
    /* ---- input ---- */
    size_t k,		/* row/column index to delete */
    cholmod_sparse *R,	/* NULL, or the nonzero pattern of kth row of L */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowdel (size_t, cholmod_sparse *, cholmod_factor *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowdel_solve:  delete a row, and downdate Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowdel, but also downdates the solution to Lx=b.
 * When row/column k of A is "deleted" from the system A*y=b, this can induce
 * a change to x, in addition to changes arising when L and b are modified.
 * If this is the case, the kth entry of y is required as input (yk) */

int cholmod_rowdel_solve
(
    /* ---- input ---- */
    size_t k,		/* row/column index to delete */
    cholmod_sparse *R,	/* NULL, or the nonzero pattern of kth row of L */
    double yk [2],	/* kth entry in the solution to A*y=b */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowdel_solve (size_t, cholmod_sparse *, double *,
    cholmod_factor *, cholmod_dense *, cholmod_dense *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_rowdel_mark:  delete a row, and downdate solution to partial Lx=b */
/* -------------------------------------------------------------------------- */

/* Does the same as cholmod_rowdel_solve, except only part of L is used in
 * the update/downdate of the solution to Lx=b.  This routine is an "expert"
 * routine.  It is meant for use in LPDASA only.  */

int cholmod_rowdel_mark
(
    /* ---- input ---- */
    size_t k,		/* row/column index to delete */
    cholmod_sparse *R,	/* NULL, or the nonzero pattern of kth row of L */
    double yk [2],	/* kth entry in the solution to A*y=b */
    int32_t *colmark,	/* array of size n.  See cholmod_updown.c */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factor to modify */
    cholmod_dense *X,	/* solution to Lx=b (size n-by-1) */
    cholmod_dense *DeltaB,  /* change in b, zero on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_rowdel_mark (size_t, cholmod_sparse *, double *,
    int64_t *, cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

#endif



/* ========================================================================== */
/* === Include/cholmod_camd.h =============================================== */
/* ========================================================================== */

/* CHOLMOD Partition module, interface to CAMD, CCOLAMD, and CSYMAMD
 *
 * An interface to CCOLAMD and CSYMAMD, constrained minimum degree ordering
 * methods which order a matrix following constraints determined via nested
 * dissection.
 *
 * These functions do not require METIS.  They are installed unless NCAMD
 * is defined:
 * cholmod_ccolamd		interface to CCOLAMD ordering
 * cholmod_csymamd		interface to CSYMAMD ordering
 * cholmod_camd			interface to CAMD ordering
 *
 * Requires the Core and Cholesky modules, and two packages: CAMD,
 * and CCOLAMD.  Used by functions in the Partition Module.
 */

#ifndef NCAMD

/* -------------------------------------------------------------------------- */
/* cholmod_ccolamd */
/* -------------------------------------------------------------------------- */

/* Order AA' or A(:,f)*A(:,f)' using CCOLAMD. */

int cholmod_ccolamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int32_t *Cmember,	/* size A->nrow.  Cmember [i] = c if row i is in the
			 * constraint set c.  c must be >= 0.  The # of
			 * constraint sets is max (Cmember) + 1.  If Cmember is
			 * NULL, then it is interpretted as Cmember [i] = 0 for
			 * all i */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_ccolamd (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_csymamd */
/* -------------------------------------------------------------------------- */

/* Order A using CSYMAMD. */

int cholmod_csymamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    /* ---- output --- */
    int32_t *Cmember,	/* size nrow.  see cholmod_ccolamd above */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_csymamd (cholmod_sparse *, int64_t *,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_camd */
/* -------------------------------------------------------------------------- */

/* Order A using CAMD. */

int cholmod_camd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Cmember,	/* size nrow.  see cholmod_ccolamd above */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_camd (cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, cholmod_common *) ;

#endif

/* ========================================================================== */
/* === Include/cholmod_partition.h ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_partition.h.
 * Copyright (C) 2005-2013, Univ. of Florida.  Author: Timothy A. Davis
 * -------------------------------------------------------------------------- */

/* CHOLMOD Partition module.
 *
 * Graph partitioning and graph-partition-based orderings.  Includes an
 * interface to CCOLAMD and CSYMAMD, constrained minimum degree ordering
 * methods which order a matrix following constraints determined via nested
 * dissection.
 *
 * These functions require METIS:
 * cholmod_nested_dissection	CHOLMOD nested dissection ordering
 * cholmod_metis		METIS nested dissection ordering (METIS_NodeND)
 * cholmod_bisect		graph partitioner (currently based on METIS)
 * cholmod_metis_bisector	direct interface to METIS_ComputeVertexSeparator
 *
 * Requires the Core and Cholesky modules, and three packages: METIS, CAMD,
 * and CCOLAMD.  Optionally used by the Cholesky module.
 *
 * Note that METIS does not have a version that uses int64_t integers.  If you
 * try to use cholmod_nested_dissection, cholmod_metis, cholmod_bisect, or
 * cholmod_metis_bisector on a matrix that is too large, an error code will be
 * returned.  METIS does have an "idxtype", which could be redefined as
 * int64_t, if you wish to edit METIS or use compile-time flags to redefine
 * idxtype.
 */

// These routines still exist if CHOLMOD is compiled with -DNPARTITION,
// but they return Common->status = CHOLMOD_NOT_INSTALLED.

/* -------------------------------------------------------------------------- */
/* cholmod_nested_dissection */
/* -------------------------------------------------------------------------- */

/* Order A, AA', or A(:,f)*A(:,f)' using CHOLMOD's nested dissection method
 * (METIS's node bisector applied recursively to compute the separator tree
 * and constraint sets, followed by CCOLAMD using the constraints).  Usually
 * finds better orderings than METIS_NodeND, but takes longer.
 */

int64_t cholmod_nested_dissection	/* returns # of components */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    int32_t *CParent,	/* size A->nrow.  On output, CParent [c] is the parent
			 * of component c, or EMPTY if c is a root, and where
			 * c is in the range 0 to # of components minus 1 */
    int32_t *Cmember,	/* size A->nrow.  Cmember [j] = c if node j of A is
			 * in component c */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_nested_dissection (cholmod_sparse *,
    int64_t *, size_t, int64_t *, int64_t *,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_metis */
/* -------------------------------------------------------------------------- */

/* Order A, AA', or A(:,f)*A(:,f)' using METIS_NodeND. */

int cholmod_metis
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with etree or coletree postorder */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_metis (cholmod_sparse *, int64_t *, size_t, int,
    int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_bisect */
/* -------------------------------------------------------------------------- */

/* Finds a node bisector of A, A*A', A(:,f)*A(:,f)'. */

int64_t cholmod_bisect	/* returns # of nodes in separator */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to bisect */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int compress,	/* if TRUE, compress the graph first */
    /* ---- output --- */
    int32_t *Partition,	/* size A->nrow.  Node i is in the left graph if
			 * Partition [i] = 0, the right graph if 1, and in the
			 * separator if 2. */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_bisect (cholmod_sparse *, int64_t *,
    size_t, int, int64_t *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_metis_bisector */
/* -------------------------------------------------------------------------- */

/* Find a set of nodes that bisects the graph of A or AA' (direct interface
 * to METIS_ComputeVertexSeperator). */

int64_t cholmod_metis_bisector	/* returns separator size */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to bisect */
    int32_t *Anw,	/* size A->nrow, node weights, can be NULL, */
                        /* which means the graph is unweighted. */ 
    int32_t *Aew,	/* size nz, edge weights (silently ignored). */
                        /* This option was available with METIS 4, but not */
                        /* in METIS 5.  This argument is now unused, but */
                        /* it remains for backward compatibilty, so as not */
                        /* to change the API for cholmod_metis_bisector. */
    /* ---- output --- */
    int32_t *Partition,	/* size A->nrow */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_metis_bisector (cholmod_sparse *,
    int64_t *, int64_t *, int64_t *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_collapse_septree */
/* -------------------------------------------------------------------------- */

/* Collapse nodes in a separator tree. */

int64_t cholmod_collapse_septree
(
    /* ---- input ---- */
    size_t n,		/* # of nodes in the graph */
    size_t ncomponents,	/* # of nodes in the separator tree (must be <= n) */
    double nd_oksep,    /* collapse if #sep >= nd_oksep * #nodes in subtree */
    size_t nd_small,    /* collapse if #nodes in subtree < nd_small */
    /* ---- in/out --- */
    int32_t *CParent,	/* size ncomponents; from cholmod_nested_dissection */
    int32_t *Cmember,	/* size n; from cholmod_nested_dissection */
    /* --------------- */
    cholmod_common *Common
) ;

int64_t cholmod_l_collapse_septree (size_t, size_t, double, size_t,
    int64_t *, int64_t *, cholmod_common *) ;

/* ========================================================================== */
/* === Include/cholmod_supernodal.h ========================================= */
/* ========================================================================== */

/* CHOLMOD Supernodal module.
 *
 * Supernodal analysis, factorization, and solve.  The simplest way to use
 * these routines is via the Cholesky module.  It does not provide any
 * fill-reducing orderings, but does accept the orderings computed by the
 * Cholesky module.  It does not require the Cholesky module itself, however.
 *
 * Primary routines:
 * -----------------
 * cholmod_super_symbolic	supernodal symbolic analysis
 * cholmod_super_numeric	supernodal numeric factorization
 * cholmod_super_lsolve		supernodal Lx=b solve
 * cholmod_super_ltsolve	supernodal L'x=b solve
 *
 * Prototypes for the BLAS and LAPACK routines that CHOLMOD uses are listed
 * below, including how they are used in CHOLMOD.
 *
 * BLAS routines:
 * --------------
 * dtrsv	solve Lx=b or L'x=b, L non-unit diagonal, x and b stride-1
 * dtrsm	solve LX=B or L'X=b, L non-unit diagonal
 * dgemv	y=y-A*x or y=y-A'*x (x and y stride-1)
 * dgemm	C=A*B', C=C-A*B, or C=C-A'*B
 * dsyrk	C=tril(A*A')
 *
 * LAPACK routines:
 * ----------------
 * dpotrf	LAPACK: A=chol(tril(A))
 *
 * Requires the Core module, and two external packages: LAPACK and the BLAS.
 * Optionally used by the Cholesky module.
 */

#ifndef NSUPERNODAL

/* -------------------------------------------------------------------------- */
/* cholmod_super_symbolic */
/* -------------------------------------------------------------------------- */

/* Analyzes A, AA', or A(:,f)*A(:,f)' in preparation for a supernodal numeric
 * factorization.  The user need not call this directly; cholmod_analyze is
 * a "simple" wrapper for this routine.
 */

int cholmod_super_symbolic
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to analyze */
    cholmod_sparse *F,	/* F = A' or A(:,f)' */
    int32_t *Parent,	/* elimination tree */
    /* ---- in/out --- */
    cholmod_factor *L,	/* simplicial symbolic on input,
			 * supernodal symbolic on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_symbolic (cholmod_sparse *, cholmod_sparse *,
    int64_t *, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_symbolic2 */
/* -------------------------------------------------------------------------- */

/* Analyze for supernodal Cholesky or multifrontal QR */

int cholmod_super_symbolic2
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to analyze */
    cholmod_sparse *F,	/* F = A' or A(:,f)' */
    int32_t *Parent,	/* elimination tree */
    /* ---- in/out --- */
    cholmod_factor *L,	/* simplicial symbolic on input,
			 * supernodal symbolic on output */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_symbolic2 (int, cholmod_sparse *, cholmod_sparse *,
    int64_t *, cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_numeric */
/* -------------------------------------------------------------------------- */

/* Computes the numeric LL' factorization of A, AA', or A(:,f)*A(:,f)' using
 * a BLAS-based supernodal method.  The user need not call this directly;
 * cholmod_factorize is a "simple" wrapper for this routine.
 */

int cholmod_super_numeric
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to factorize */
    cholmod_sparse *F,	/* F = A' or A(:,f)' */
    double beta [2],	/* beta*I is added to diagonal of matrix to factorize */
    /* ---- in/out --- */
    cholmod_factor *L,	/* factorization */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_numeric (cholmod_sparse *, cholmod_sparse *, double *,
    cholmod_factor *, cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_lsolve */
/* -------------------------------------------------------------------------- */

/* Solve Lx=b where L is from a supernodal numeric factorization.  The user
 * need not call this routine directly.  cholmod_solve is a "simple" wrapper
 * for this routine. */

int cholmod_super_lsolve
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to use for the forward solve */
    /* ---- output ---- */
    cholmod_dense *X,	/* b on input, solution to Lx=b on output */
    /* ---- workspace   */
    cholmod_dense *E,	/* workspace of size nrhs*(L->maxesize) */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_lsolve (cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

/* -------------------------------------------------------------------------- */
/* cholmod_super_ltsolve */
/* -------------------------------------------------------------------------- */

/* Solve L'x=b where L is from a supernodal numeric factorization.  The user
 * need not call this routine directly.  cholmod_solve is a "simple" wrapper
 * for this routine. */

int cholmod_super_ltsolve
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factor to use for the backsolve */
    /* ---- output ---- */
    cholmod_dense *X,	/* b on input, solution to L'x=b on output */
    /* ---- workspace   */
    cholmod_dense *E,	/* workspace of size nrhs*(L->maxesize) */
    /* --------------- */
    cholmod_common *Common
) ;

int cholmod_l_super_ltsolve (cholmod_factor *, cholmod_dense *, cholmod_dense *,
    cholmod_common *) ;

#endif

#ifdef __cplusplus
}
#endif


/* ========================================================================== */
/* === Include/cholmod_gpu.h ================================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_gpu.h.
 * Copyright (C) 2014, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* CHOLMOD GPU module
 */

#ifdef SUITESPARSE_CUDA

#include "omp.h"
#include <fenv.h>
#ifndef SUITESPARSE_GPU_EXTERN_ON
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/* CHOLMOD_GPU_PRINTF: for printing GPU debug error messages */
/*
#define CHOLMOD_GPU_PRINTF(args) printf args
*/
#define CHOLMOD_GPU_PRINTF(args)

/* define supernode requirements for processing on GPU */
#define CHOLMOD_ND_ROW_LIMIT 256 /* required descendant rows */
#define CHOLMOD_ND_COL_LIMIT 32  /* required descendnat cols */
#define CHOLMOD_POTRF_LIMIT  512  /* required cols for POTRF & TRSM on GPU */

/* # of host supernodes to perform before checking for free pinned buffers */
#define CHOLMOD_GPU_SKIP     3    

#define CHOLMOD_HANDLE_CUDA_ERROR(e,s) {if (e) {ERROR(CHOLMOD_GPU_PROBLEM,s);}}

/* make it easy for C++ programs to include CHOLMOD */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cholmod_gpu_pointers
{
    double *h_Lx [CHOLMOD_HOST_SUPERNODE_BUFFERS] ;
    double *d_Lx [CHOLMOD_DEVICE_STREAMS] ;
    double *d_C ;
    double *d_A [CHOLMOD_DEVICE_STREAMS] ;
    void   *d_Ls ;
    void   *d_Map ;
    void   *d_RelativeMap ;

} cholmod_gpu_pointers ;

int cholmod_gpu_memorysize   /* GPU memory size available, 1 if no GPU */
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;

int cholmod_l_gpu_memorysize /* GPU memory size available, 1 if no GPU */
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;
 
int cholmod_gpu_probe   ( cholmod_common *Common ) ;
int cholmod_l_gpu_probe ( cholmod_common *Common ) ;

int cholmod_gpu_deallocate   ( cholmod_common *Common ) ;
int cholmod_l_gpu_deallocate ( cholmod_common *Common ) ;

void cholmod_gpu_end   ( cholmod_common *Common ) ;
void cholmod_l_gpu_end ( cholmod_common *Common ) ;

int cholmod_gpu_allocate   ( cholmod_common *Common ) ;
int cholmod_l_gpu_allocate ( cholmod_common *Common ) ;

#ifdef __cplusplus
}
#endif

#endif

#endif
