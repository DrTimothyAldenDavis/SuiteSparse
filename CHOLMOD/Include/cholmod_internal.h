//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod_internal.h
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod_internal.h. Copyright (C) 2005-2022,
// Timothy A. Davis.  All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/* CHOLMOD internal include file.
 *
 * This file contains internal definitions for CHOLMOD, not meant to be included
 * in user code.  They define macros that are not prefixed with CHOLMOD_.  This
 * file can safely #include'd in user code if you want to make use of the
 * macros defined here, and don't mind the possible name conflicts with your
 * code, however.
 *
 * Required by all CHOLMOD routines.  Not required by any user routine that
 * uses CHOLMOMD.  Unless debugging is enabled, this file does not require any
 * CHOLMOD module (not even the Core module).
 *
 * If debugging is enabled, all CHOLMOD modules require the Check module.
 * Enabling debugging requires that this file be editted.  Debugging cannot be
 * enabled with a compiler flag.  This is because CHOLMOD is exceedingly slow
 * when debugging is enabled.  Debugging is meant for development of CHOLMOD
 * itself, not by users of CHOLMOD.
 */

#ifndef CHOLMOD_INTERNAL_H
#define CHOLMOD_INTERNAL_H

#define SUITESPARSE_BLAS_DEFINITIONS
#include "cholmod.h"

/* ========================================================================== */
/* === debugging and basic includes ========================================= */
/* ========================================================================== */

/* turn off debugging */
#ifndef NDEBUG
#define NDEBUG
#endif

/* Uncomment this line to enable debugging.  CHOLMOD will be very slow.
#undef NDEBUG
 */

/* ========================================================================== */
/* === basic definitions ==================================================== */
/* ========================================================================== */

/* Some non-conforming compilers insist on defining TRUE and FALSE. */
#undef TRUE
#undef FALSE
#define TRUE 1
#define FALSE 0
#define BOOLEAN(x) ((x) ? TRUE : FALSE)

/* NULL should already be defined, but ensure it is here. */
#ifndef NULL
#define NULL ((void *) 0)
#endif

/* FLIP is a "negation about -1", and is used to mark an integer i that is
 * normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
 * is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
 * for all integers i.  UNFLIP (i) is >= EMPTY. */
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))

/* MAX and MIN are not safe to use for NaN's */
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MAX3(a,b,c) (((a) > (b)) ? (MAX (a,c)) : (MAX (b,c)))
#define MAX4(a,b,c,d) (((a) > (b)) ? (MAX3 (a,c,d)) : (MAX3 (b,c,d)))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define IMPLIES(p,q) (!(p) || (q))

/* find the sign: -1 if x < 0, 1 if x > 0, zero otherwise.
 * Not safe for NaN's */
#define SIGN(x) (((x) < 0) ? (-1) : (((x) > 0) ? 1 : 0))

/* round up an integer x to a multiple of s */
#define ROUNDUP(x,s) ((s) * (((x) + ((s) - 1)) / (s)))

#define ERROR(status,msg) \
    CHOLMOD(error) (status, __FILE__, __LINE__, msg, Common)

/* Check a pointer and return if null.  Set status to invalid, unless the
 * status is already "out of memory" */
#define RETURN_IF_NULL(A,result) \
{ \
    if ((A) == NULL) \
    { \
	if (Common->status != CHOLMOD_OUT_OF_MEMORY) \
	{ \
	    ERROR (CHOLMOD_INVALID, "argument missing") ; \
	} \
	return (result) ; \
    } \
}

/* Return if Common is NULL or invalid */
#define RETURN_IF_NULL_COMMON(result) \
{ \
    if (Common == NULL) \
    { \
	return (result) ; \
    } \
    if (Common->itype != ITYPE || Common->dtype != DTYPE) \
    { \
	Common->status = CHOLMOD_INVALID ; \
	return (result) ; \
    } \
}

#define IS_NAN(x)	CHOLMOD_IS_NAN(x)
#define IS_ZERO(x)	CHOLMOD_IS_ZERO(x)
#define IS_NONZERO(x)	CHOLMOD_IS_NONZERO(x)
#define IS_LT_ZERO(x)	CHOLMOD_IS_LT_ZERO(x)
#define IS_GT_ZERO(x)	CHOLMOD_IS_GT_ZERO(x)
#define IS_LE_ZERO(x)	CHOLMOD_IS_LE_ZERO(x)

/* 1e308 is a huge number that doesn't take many characters to print in a
 * file, in CHOLMOD/Check/cholmod_read and _write.  Numbers larger than this
 * are interpretted as Inf, since sscanf doesn't read in Inf's properly.
 * This assumes IEEE double precision arithmetic.  DBL_MAX would be a little
 * better, except that it takes too many digits to print in a file. */
#define HUGE_DOUBLE 1e308

/* ========================================================================== */
/* === int/long and double/float definitions ================================ */
/* ========================================================================== */

/* CHOLMOD is designed for 3 types of integer variables:
 *
 *	(1) all integers are int
 *	(2) most integers are int, some are SuiteSparse_long
 *	(3) all integers are SuiteSparse_long
 *
 * and two kinds of floating-point values:
 *
 *	(1) double
 *	(2) float
 *
 * the complex types (ANSI-compatible complex, and MATLAB-compatable zomplex)
 * are based on the double or float type, and are not selected here.  They
 * are typically selected via template routines.
 *
 * This gives 6 different modes in which CHOLMOD can be compiled (only the
 * first two are currently supported):
 *
 *	DINT	double, int			prefix: cholmod_
 *	DLONG	double, SuiteSparse_long	prefix: cholmod_l_
 *	DMIX	double, mixed int/SuiteSparse_long	prefix: cholmod_m_
 *	SINT	float, int			prefix: cholmod_si_
 *	SLONG	float, SuiteSparse_long		prefix: cholmod_sl_
 *	SMIX	float, mixed int/log		prefix: cholmod_sm_
 *
 * These are selected with compile time flags (-DDLONG, for example).  If no
 * flag is selected, the default is DINT.
 *
 * All six versions use the same include files.  The user-visible include files
 * are completely independent of which int/long/double/float version is being
 * used.  The integer / real types in all data structures (sparse, triplet,
 * dense, common, and triplet) are defined at run-time, not compile-time, so
 * there is only one "cholmod_sparse" data type.  Void pointers are used inside
 * that data structure to point to arrays of the proper type.  Each data
 * structure has an itype and dtype field which determines the kind of basic
 * types used.  These are defined in Include/cholmod_core.h.
 *
 * FUTURE WORK: support all six types (float, and mixed int/long)
 */

/* -------------------------------------------------------------------------- */
/* routines for doing arithmetic on size_t, and checking for overflow */
/* -------------------------------------------------------------------------- */

size_t cholmod_add_size_t (size_t a, size_t b, int *ok) ;
size_t cholmod_mult_size_t (size_t a, size_t k, int *ok) ;
size_t cholmod_l_add_size_t (size_t a, size_t b, int *ok) ;
size_t cholmod_l_mult_size_t (size_t a, size_t k, int *ok) ;

/* -------------------------------------------------------------------------- */
/* double (also complex double), SuiteSparse_long */
/* -------------------------------------------------------------------------- */

#ifdef DLONG
#define Real double
#define Int SuiteSparse_long
#define UInt SuiteSparse_ulong
#define Int_max SuiteSparse_long_max
#define CHOLMOD(name) cholmod_l_ ## name
#define LONG
#define DOUBLE
#define ITYPE CHOLMOD_LONG
#define DTYPE CHOLMOD_DOUBLE
#define ID "%" SuiteSparse_long_idd

/* -------------------------------------------------------------------------- */
/* double (also complex double), int: this is the default */
/* -------------------------------------------------------------------------- */

#else

#ifndef DINT
#define DINT
#endif
#define INT
#define DOUBLE

#define Real double
#define Int int32_t
#define UInt uint32_t
#define Int_max INT32_MAX
#define CHOLMOD(name) cholmod_ ## name
#define ITYPE CHOLMOD_INT
#define DTYPE CHOLMOD_DOUBLE
#define ID "%d"

/* GPU acceleration is not available for the int version of CHOLMOD */
#undef SUITESPARSE_CUDA

#endif


/* ========================================================================== */
/* === Include/cholmod_complexity.h ========================================= */
/* ========================================================================== */

/* Define operations on pattern, real, complex, and zomplex objects.
 *
 * The xtype of an object defines it numerical type.  A qttern object has no
 * numerical values (A->x and A->z are NULL).  A real object has no imaginary
 * qrt (A->x is used, A->z is NULL).  A complex object has an imaginary qrt
 * that is stored interleaved with its real qrt (A->x is of size 2*nz, A->z
 * is NULL).  A zomplex object has both real and imaginary qrts, which are
 * stored seqrately, as in MATLAB (A->x and A->z are both used).
 *
 * XTYPE is CHOLMOD_PATTERN, _REAL, _COMPLEX or _ZOMPLEX, and is the xtype of
 * the template routine under construction.  XTYPE2 is equal to XTYPE, except
 * if XTYPE is CHOLMOD_PATTERN, in which case XTYPE is CHOLMOD_REAL.
 * XTYPE and XTYPE2 are defined in cholmod_template.h.  
 */

/* -------------------------------------------------------------------------- */
/* pattern */
/* -------------------------------------------------------------------------- */

#define P_TEMPLATE(name)		p_ ## name
#define P_ASSIGN2(x,z,p,ax,az,q)	x [p] = 1
#define P_PRINT(k,x,z,p)		PRK(k, ("1"))

/* -------------------------------------------------------------------------- */
/* real */
/* -------------------------------------------------------------------------- */

#define R_TEMPLATE(name)			r_ ## name
#define R_ASSEMBLE(x,z,p,ax,az,q)		x [p] += ax [q]
#define R_ASSIGN(x,z,p,ax,az,q)			x [p]  = ax [q]
#define R_ASSIGN_CONJ(x,z,p,ax,az,q)		x [p]  = ax [q]
#define R_ASSIGN_REAL(x,p,ax,q)			x [p]  = ax [q]
#define R_XTYPE_OK(type)			((type) == CHOLMOD_REAL)
#define R_IS_NONZERO(ax,az,q)			IS_NONZERO (ax [q])
#define R_IS_ZERO(ax,az,q)			IS_ZERO (ax [q])
#define R_IS_ONE(ax,az,q)			(ax [q] == 1)
#define R_MULT(x,z,p, ax,az,q, bx,bz,r)		x [p]  = ax [q] * bx [r]
#define R_MULTADD(x,z,p, ax,az,q, bx,bz,r)	x [p] += ax [q] * bx [r]
#define R_MULTSUB(x,z,p, ax,az,q, bx,bz,r)	x [p] -= ax [q] * bx [r]
#define R_MULTADDCONJ(x,z,p, ax,az,q, bx,bz,r)	x [p] += ax [q] * bx [r]
#define R_MULTSUBCONJ(x,z,p, ax,az,q, bx,bz,r)	x [p] -= ax [q] * bx [r]
#define R_ADD(x,z,p, ax,az,q, bx,bz,r)		x [p]  = ax [q] + bx [r]
#define R_ADD_REAL(x,p, ax,q, bx,r)		x [p]  = ax [q] + bx [r]
#define R_CLEAR(x,z,p)				x [p]  = 0
#define R_CLEAR_IMAG(x,z,p)
#define R_DIV(x,z,p,ax,az,q)			x [p] /= ax [q]
#define R_LLDOT(x,p, ax,az,q)			x [p] -= ax [q] * ax [q]
#define R_PRINT(k,x,z,p)			PRK(k, ("%24.16e", x [p]))

#define R_DIV_REAL(x,z,p, ax,az,q, bx,r)	x [p] = ax [q] / bx [r]
#define R_MULT_REAL(x,z,p, ax,az,q, bx,r)	x [p] = ax [q] * bx [r]

#define R_LDLDOT(x,p, ax,az,q, bx,r)		x [p] -=(ax[q] * ax[q])/ bx[r]

/* -------------------------------------------------------------------------- */
/* complex */
/* -------------------------------------------------------------------------- */

#define C_TEMPLATE(name)		c_ ## name
#define CT_TEMPLATE(name)		ct_ ## name

#define C_ASSEMBLE(x,z,p,ax,az,q) \
    x [2*(p)  ] += ax [2*(q)  ] ; \
    x [2*(p)+1] += ax [2*(q)+1]

#define C_ASSIGN(x,z,p,ax,az,q) \
    x [2*(p)  ] = ax [2*(q)  ] ; \
    x [2*(p)+1] = ax [2*(q)+1]

#define C_ASSIGN_REAL(x,p,ax,q)			x [2*(p)]  = ax [2*(q)]

#define C_ASSIGN_CONJ(x,z,p,ax,az,q) \
    x [2*(p)  ] =  ax [2*(q)  ] ; \
    x [2*(p)+1] = -ax [2*(q)+1]

#define C_XTYPE_OK(type)		((type) == CHOLMOD_COMPLEX)

#define C_IS_NONZERO(ax,az,q) \
    (IS_NONZERO (ax [2*(q)]) || IS_NONZERO (ax [2*(q)+1]))

#define C_IS_ZERO(ax,az,q) \
    (IS_ZERO (ax [2*(q)]) && IS_ZERO (ax [2*(q)+1]))

#define C_IS_ONE(ax,az,q) \
    ((ax [2*(q)] == 1) && IS_ZERO (ax [2*(q)+1]))

#define C_IMAG_IS_NONZERO(ax,az,q)  (IS_NONZERO (ax [2*(q)+1]))

#define C_MULT(x,z,p, ax,az,q, bx,bz,r) \
x [2*(p)  ] = ax [2*(q)  ] * bx [2*(r)] - ax [2*(q)+1] * bx [2*(r)+1] ; \
x [2*(p)+1] = ax [2*(q)+1] * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

#define C_MULTADD(x,z,p, ax,az,q, bx,bz,r) \
x [2*(p)  ] += ax [2*(q)  ] * bx [2*(r)] - ax [2*(q)+1] * bx [2*(r)+1] ; \
x [2*(p)+1] += ax [2*(q)+1] * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

#define C_MULTSUB(x,z,p, ax,az,q, bx,bz,r) \
x [2*(p)  ] -= ax [2*(q)  ] * bx [2*(r)] - ax [2*(q)+1] * bx [2*(r)+1] ; \
x [2*(p)+1] -= ax [2*(q)+1] * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

/* s += conj(a)*b */
#define C_MULTADDCONJ(x,z,p, ax,az,q, bx,bz,r) \
x [2*(p)  ] +=   ax [2*(q)  ]  * bx [2*(r)] + ax [2*(q)+1] * bx [2*(r)+1] ; \
x [2*(p)+1] += (-ax [2*(q)+1]) * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

/* s -= conj(a)*b */
#define C_MULTSUBCONJ(x,z,p, ax,az,q, bx,bz,r) \
x [2*(p)  ] -=   ax [2*(q)  ]  * bx [2*(r)] + ax [2*(q)+1] * bx [2*(r)+1] ; \
x [2*(p)+1] -= (-ax [2*(q)+1]) * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

#define C_ADD(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] = ax [2*(q)  ] + bx [2*(r)  ] ; \
    x [2*(p)+1] = ax [2*(q)+1] + bx [2*(r)+1]

#define C_ADD_REAL(x,p, ax,q, bx,r) \
    x [2*(p)] = ax [2*(q)] + bx [2*(r)]

#define C_CLEAR(x,z,p) \
    x [2*(p)  ] = 0 ; \
    x [2*(p)+1] = 0

#define C_CLEAR_IMAG(x,z,p) \
    x [2*(p)+1] = 0

/* s = s / a */
#define C_DIV(x,z,p,ax,az,q) \
    SuiteSparse_config_divcomplex ( \
	      x [2*(p)],  x [2*(p)+1], \
	     ax [2*(q)], ax [2*(q)+1], \
	     &x [2*(p)], &x [2*(p)+1])

/* s -= conj(a)*a ; note that the result of conj(a)*a is real */
#define C_LLDOT(x,p, ax,az,q) \
    x [2*(p)] -= ax [2*(q)] * ax [2*(q)] + ax [2*(q)+1] * ax [2*(q)+1]

#define C_PRINT(k,x,z,p) PRK(k, ("(%24.16e,%24.16e)", x [2*(p)], x [2*(p)+1]))

#define C_DIV_REAL(x,z,p, ax,az,q, bx,r) \
    x [2*(p)  ] = ax [2*(q)  ] / bx [2*(r)] ; \
    x [2*(p)+1] = ax [2*(q)+1] / bx [2*(r)]

#define C_MULT_REAL(x,z,p, ax,az,q, bx,r) \
    x [2*(p)  ] = ax [2*(q)  ] * bx [2*(r)] ; \
    x [2*(p)+1] = ax [2*(q)+1] * bx [2*(r)]

/* s -= conj(a)*a/t */
#define C_LDLDOT(x,p, ax,az,q, bx,r) \
    x [2*(p)] -= (ax [2*(q)] * ax [2*(q)] + ax [2*(q)+1] * ax [2*(q)+1]) / bx[r]

/* -------------------------------------------------------------------------- */
/* zomplex */
/* -------------------------------------------------------------------------- */

#define Z_TEMPLATE(name)		z_ ## name
#define ZT_TEMPLATE(name)		zt_ ## name

#define Z_ASSEMBLE(x,z,p,ax,az,q) \
    x [p] += ax [q] ; \
    z [p] += az [q]

#define Z_ASSIGN(x,z,p,ax,az,q) \
    x [p] = ax [q] ; \
    z [p] = az [q]

#define Z_ASSIGN_REAL(x,p,ax,q)			x [p]  = ax [q]

#define Z_ASSIGN_CONJ(x,z,p,ax,az,q) \
    x [p] =  ax [q] ; \
    z [p] = -az [q]

#define Z_XTYPE_OK(type)		((type) == CHOLMOD_ZOMPLEX)

#define Z_IS_NONZERO(ax,az,q) \
    (IS_NONZERO (ax [q]) || IS_NONZERO (az [q]))

#define Z_IS_ZERO(ax,az,q) \
    (IS_ZERO (ax [q]) && IS_ZERO (az [q]))

#define Z_IS_ONE(ax,az,q) \
    ((ax [q] == 1) && IS_ZERO (az [q]))

#define Z_IMAG_IS_NONZERO(ax,az,q)  (IS_NONZERO (az [q]))

#define Z_MULT(x,z,p, ax,az,q, bx,bz,r) \
    x [p] = ax [q] * bx [r] - az [q] * bz [r] ; \
    z [p] = az [q] * bx [r] + ax [q] * bz [r]

#define Z_MULTADD(x,z,p, ax,az,q, bx,bz,r) \
    x [p] += ax [q] * bx [r] - az [q] * bz [r] ; \
    z [p] += az [q] * bx [r] + ax [q] * bz [r]

#define Z_MULTSUB(x,z,p, ax,az,q, bx,bz,r) \
    x [p] -= ax [q] * bx [r] - az [q] * bz [r] ; \
    z [p] -= az [q] * bx [r] + ax [q] * bz [r]

#define Z_MULTADDCONJ(x,z,p, ax,az,q, bx,bz,r) \
    x [p] +=   ax [q]  * bx [r] + az [q] * bz [r] ; \
    z [p] += (-az [q]) * bx [r] + ax [q] * bz [r]

#define Z_MULTSUBCONJ(x,z,p, ax,az,q, bx,bz,r) \
    x [p] -=   ax [q]  * bx [r] + az [q] * bz [r] ; \
    z [p] -= (-az [q]) * bx [r] + ax [q] * bz [r]

#define Z_ADD(x,z,p, ax,az,q, bx,bz,r) \
	x [p] = ax [q] + bx [r] ; \
	z [p] = az [q] + bz [r]

#define Z_ADD_REAL(x,p, ax,q, bx,r) \
	x [p] = ax [q] + bx [r]

#define Z_CLEAR(x,z,p) \
    x [p] = 0 ; \
    z [p] = 0

#define Z_CLEAR_IMAG(x,z,p) \
    z [p] = 0

/* s = s / a */
#define Z_DIV(x,z,p,ax,az,q) \
    SuiteSparse_config_divcomplex \
        (x [p], z [p], ax [q], az [q], &x [p], &z [p])

/* s -= conj(a)*a ; note that the result of conj(a)*a is real */
#define Z_LLDOT(x,p, ax,az,q) \
    x [p] -= ax [q] * ax [q] + az [q] * az [q]

#define Z_PRINT(k,x,z,p)	PRK(k, ("(%24.16e,%24.16e)", x [p], z [p]))

#define Z_DIV_REAL(x,z,p, ax,az,q, bx,r) \
    x [p] = ax [q] / bx [r] ; \
    z [p] = az [q] / bx [r]

#define Z_MULT_REAL(x,z,p, ax,az,q, bx,r) \
    x [p] = ax [q] * bx [r] ; \
    z [p] = az [q] * bx [r]

/* s -= conj(a)*a/t */
#define Z_LDLDOT(x,p, ax,az,q, bx,r) \
    x [p] -= (ax [q] * ax [q] + az [q] * az [q]) / bx[r]

/* -------------------------------------------------------------------------- */
/* all classes */
/* -------------------------------------------------------------------------- */

/* Check if A->xtype and the two arrays A->x and A->z are valid.  Set status to
 * invalid, unless status is already "out of memory".  A can be a sparse matrix,
 * dense matrix, factor, or triplet. */

#define RETURN_IF_XTYPE_INVALID(A,xtype1,xtype2,result) \
{ \
    if ((A)->xtype < (xtype1) || (A)->xtype > (xtype2) || \
        ((A)->xtype != CHOLMOD_PATTERN && ((A)->x) == NULL) || \
	((A)->xtype == CHOLMOD_ZOMPLEX && ((A)->z) == NULL)) \
    { \
	if (Common->status != CHOLMOD_OUT_OF_MEMORY) \
	{ \
	    ERROR (CHOLMOD_INVALID, "invalid xtype") ; \
	} \
	return (result) ; \
    } \
}

/* ========================================================================== */
/* === Architecture and BLAS ================================================ */
/* ========================================================================== */

#if defined (__sun) || defined (MSOL2) || defined (ARCH_SOL2)
#define CHOLMOD_SOL2
#define CHOLMOD_ARCHITECTURE "Sun Solaris"

#elif defined (__sgi) || defined (MSGI) || defined (ARCH_SGI)
#define CHOLMOD_SGI
#define CHOLMOD_ARCHITECTURE "SGI Irix"

#elif defined (__linux) || defined (MGLNX86) || defined (ARCH_GLNX86)
#define CHOLMOD_LINUX
#define CHOLMOD_ARCHITECTURE "Linux"

#elif defined (__APPLE__)
#define CHOLMOD_MAC
#define CHOLMOD_ARCHITECTURE "Mac"

#elif defined (_AIX) || defined (MIBM_RS) || defined (ARCH_IBM_RS)
#define CHOLMOD_AIX
#define CHOLMOD_ARCHITECTURE "IBM AIX"

#elif defined (__alpha) || defined (MALPHA) || defined (ARCH_ALPHA)
#define CHOLMOD_ALPHA
#define CHOLMOD_ARCHITECTURE "Compaq Alpha"

#elif defined (_WIN32) || defined (WIN32) || defined (_WIN64) || defined (WIN64)
#if defined (__MINGW32__) || defined (__MINGW32__)
#define CHOLMOD_MINGW
#elif defined (__CYGWIN32__) || defined (__CYGWIN32__)
#define CHOLMOD_CYGWIN
#else
#define CHOLMOD_WINDOWS
#endif
#define CHOLMOD_ARCHITECTURE "Microsoft Windows"

#elif defined (__hppa) || defined (__hpux) || defined (MHPUX) || defined (ARCH_HPUX)
#define CHOLMOD_HP
#define CHOLMOD_ARCHITECTURE "HP Unix"

#elif defined (__hp700) || defined (MHP700) || defined (ARCH_HP700)
#define CHOLMOD_HP
#define CHOLMOD_ARCHITECTURE "HP 700 Unix"

#else
#define CHOLMOD_ARCHITECTURE "unknown"
#endif

//==============================================================================
//=== openmp support ===========================================================
//==============================================================================

static inline int cholmod_nthreads  // returns # of OpenMP threads to use
(
    double work,                    // total work to do
    cholmod_common *Common
)
{ 
    #ifdef _OPENMP
    double chunk = Common->chunk ;  // give each thread at least this much work
    int nthreads_max = Common->nthreads_max ;   // max # of threads to use
    if (nthreads_max <= 0)
    {
        nthreads_max = SUITESPARSE_OPENMP_MAX_THREADS ;
    }
    work  = MAX (work, 1) ;
    chunk = MAX (chunk, 1) ;
    SuiteSparse_long nthreads = (SuiteSparse_long) floor (work / chunk) ;
    nthreads = MIN (nthreads, nthreads_max) ;
    nthreads = MAX (nthreads, 1) ;
    return ((int) nthreads) ;
    #else
    return (1) ;
    #endif
}

/* ========================================================================== */
/* === debugging definitions ================================================ */
/* ========================================================================== */

#ifndef NDEBUG

#include <assert.h>

/* The cholmod_dump routines are in the Check module.  No CHOLMOD routine
 * calls the cholmod_check_* or cholmod_print_* routines in the Check module,
 * since they use Common workspace that may already be in use.  Instead, they
 * use the cholmod_dump_* routines defined there, which allocate their own
 * workspace if they need it. */

#ifndef EXTERN
#define EXTERN extern
#endif

/* double, int */
EXTERN int cholmod_dump ;
EXTERN int cholmod_dump_malloc ;
SuiteSparse_long cholmod_dump_sparse (cholmod_sparse  *, const char *,
    cholmod_common *) ;
int  cholmod_dump_factor (cholmod_factor  *, const char *, cholmod_common *) ;
int  cholmod_dump_triplet (cholmod_triplet *, const char *, cholmod_common *) ;
int  cholmod_dump_dense (cholmod_dense   *, const char *, cholmod_common *) ;
int  cholmod_dump_subset (int *, size_t, size_t, const char *,
    cholmod_common *) ;
int  cholmod_dump_perm (int *, size_t, size_t, const char *, cholmod_common *) ;
int  cholmod_dump_parent (int *, size_t, const char *, cholmod_common *) ;
void cholmod_dump_init (const char *, cholmod_common *) ;
int  cholmod_dump_mem (const char *, SuiteSparse_long, cholmod_common *) ;
void cholmod_dump_real (const char *, Real *, SuiteSparse_long,
    SuiteSparse_long, int, int, cholmod_common *) ;
void cholmod_dump_super (SuiteSparse_long, int *, int *, int *, int *, double *,
    int, cholmod_common *) ;
int  cholmod_dump_partition (SuiteSparse_long, int *, int *, int *, int *,
    SuiteSparse_long, cholmod_common *) ;
int  cholmod_dump_work(int, int, SuiteSparse_long, cholmod_common *) ;

/* double, SuiteSparse_long */
EXTERN int cholmod_l_dump ;
EXTERN int cholmod_l_dump_malloc ;
SuiteSparse_long cholmod_l_dump_sparse (cholmod_sparse  *, const char *,
    cholmod_common *) ;
int  cholmod_l_dump_factor (cholmod_factor  *, const char *, cholmod_common *) ;
int  cholmod_l_dump_triplet (cholmod_triplet *, const char *, cholmod_common *);
int  cholmod_l_dump_dense (cholmod_dense   *, const char *, cholmod_common *) ;
int  cholmod_l_dump_subset (SuiteSparse_long *, size_t, size_t, const char *,
    cholmod_common *) ;
int  cholmod_l_dump_perm (SuiteSparse_long *, size_t, size_t, const char *,
    cholmod_common *) ;
int  cholmod_l_dump_parent (SuiteSparse_long *, size_t, const char *,
    cholmod_common *) ;
void cholmod_l_dump_init (const char *, cholmod_common *) ;
int  cholmod_l_dump_mem (const char *, SuiteSparse_long, cholmod_common *) ;
void cholmod_l_dump_real (const char *, Real *, SuiteSparse_long,
    SuiteSparse_long, int, int, cholmod_common *) ;
void cholmod_l_dump_super (SuiteSparse_long, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *,
    double *, int, cholmod_common *) ;
int  cholmod_l_dump_partition (SuiteSparse_long, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long, cholmod_common *) ;
int  cholmod_l_dump_work(int, int, SuiteSparse_long, cholmod_common *) ;

#define DEBUG_INIT(s,Common)  { CHOLMOD(dump_init)(s, Common) ; }
#define ASSERT(expression) (assert (expression))

#define PRK(k,params)                                           \
{                                                               \
    if (CHOLMOD(dump) >= (k)                                    \
    {                                                           \
        int (*printf_func) (const char *, ...) ;                \
        printf_func = SuiteSparse_config_printf_func_get ( ) ;  \
        if (printf_func != NULL)                                \
        {                                                       \
            (void) (printf_func) params ;                       \
        }                                                       \
    }                                                           \
}

#define PRINT0(params) PRK (0, params)
#define PRINT1(params) PRK (1, params)
#define PRINT2(params) PRK (2, params)
#define PRINT3(params) PRK (3, params)

#define PRINTM(params) \
{ \
    if (CHOLMOD(dump_malloc) > 0) \
    { \
	printf params ; \
    } \
}

#define DEBUG(statement) statement

#else

/* Debugging disabled (the normal case) */
#define PRK(k,params)
#define DEBUG_INIT(s,Common)
#define PRINT0(params)
#define PRINT1(params)
#define PRINT2(params)
#define PRINT3(params)
#define PRINTM(params)
#define ASSERT(expression)
#define DEBUG(statement)
#endif

#endif
