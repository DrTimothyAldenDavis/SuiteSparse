//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod_types.h
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod_types.h. Copyright (C) 2005-2023,
// Timothy A. Davis.  All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CHOLMOD internal include file: defining integer and floating-point types.
// This file is suitable for inclusion in C and C++ codes.  It can be
// #include'd more than once.

// The #include'ing file defines one of four macros (DINT, DLONG, SINT, or
// SLONG) before #include'ing this file.

/* ========================================================================== */
/* === int/long and double/float definitions ================================ */
/* ========================================================================== */

/* CHOLMOD is designed for 3 types of integer variables:
 *
 *	(1) all integers are int
 *	(2) most integers are int, some are int64_t
 *	(3) all integers are int64_t
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
 *	DLONG	double, int64_t	                prefix: cholmod_l_
 *	DMIX	double, mixed int/int64_t	prefix: cholmod_m_
 *	SINT	float, int			prefix: cholmod_si_
 *	SLONG	float, int64_t		        prefix: cholmod_sl_
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
 * SINT and SLONG are in progress.
 */

// -----------------------------------------------------------------------------

#undef Real
#undef Int
#undef UInt
#undef Int_max
#undef CHOLMOD
#undef ITYPE
#undef DTYPE
#undef ID

#if defined ( SLONG )

    //--------------------------------------------------------------------------
    // SLONG: float (also complex float), int64_t
    //--------------------------------------------------------------------------

    #define Real float
    #define Int int64_t
    #define UInt uint64_t
    #define Int_max INT64_MAX
    #define CHOLMOD(name) cholmod_sl_ ## name
    #define ITYPE CHOLMOD_LONG
    #define DTYPE CHOLMOD_SINGLE
    #define ID "%" PRId64

#elif defined ( SINT )

    //--------------------------------------------------------------------------
    // SINT: float (also complex float), int32_t
    //--------------------------------------------------------------------------

    #define Real float
    #define Int int32_t
    #define UInt uint32_t
    #define Int_max INT32_MAX
    #define CHOLMOD(name) cholmod_si_ ## name
    #define ITYPE CHOLMOD_INT
    #define DTYPE CHOLMOD_SINGLE
    #define ID "%d"

#elif defined ( DLONG )

    //--------------------------------------------------------------------------
    // DLONG: double (also complex double), int64_t
    //--------------------------------------------------------------------------

    #define Real double
    #define Int int64_t
    #define UInt uint64_t
    #define Int_max INT64_MAX
    #define CHOLMOD(name) cholmod_l_ ## name
    #define ITYPE CHOLMOD_LONG
    #define DTYPE CHOLMOD_DOUBLE
    #define ID "%" PRId64

#else

    //--------------------------------------------------------------------------
    // DINT: double (also complex double), int32
    //--------------------------------------------------------------------------

    #ifndef DINT
    #define DINT
    #endif

    #define Real double
    #define Int int32_t
    #define UInt uint32_t
    #define Int_max INT32_MAX
    #define CHOLMOD(name) cholmod_ ## name
    #define ITYPE CHOLMOD_INT
    #define DTYPE CHOLMOD_DOUBLE
    #define ID "%d"

#endif

