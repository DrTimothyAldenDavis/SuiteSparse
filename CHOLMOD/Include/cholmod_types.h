//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod_types.h
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod_types.h. Copyright (C) 2005-2023,
// Timothy A. Davis.  All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CHOLMOD internal include file for defining integer types.  This file is
// suitable for inclusion in C and C++ codes.  It can be #include'd more than
// once.  The #include'ing file defines one of two macros (CHOLMOD_INT32 or
// CHOLMOD_INT64).  CHOLMOD is designed for 2 types of integer variables:
// int32_t or int64_t.
//
// The complex types (ANSI-compatible complex, and MATLAB-compatible zomplex)
// are based on the double or float type, and are not selected here.  They are
// typically selected via template routines.

//------------------------------------------------------------------------------

#undef Int
#undef UInt
#undef Int_max
#undef CHOLMOD
#undef ITYPE
#undef ID
#undef CLEAR_FLAG
// #undef I_GOTCHA
// #undef L_GOTCHA

#if defined ( CHOLMOD_INT64 )

    //--------------------------------------------------------------------------
    // CHOLMOD_INT64: int64_t
    //--------------------------------------------------------------------------

    #define Int int64_t
    #define UInt uint64_t
    #define Int_max INT64_MAX
    #define CHOLMOD(name) cholmod_l_ ## name
    #define ITYPE CHOLMOD_LONG
    #define ID "%" PRId64

    // #define L_GOTCHA GOTCHA
    // #define I_GOTCHA ;

    #define CLEAR_FLAG(Common)                  \
    {                                           \
        Common->mark++ ;                        \
        if (Common->mark <= 0)                  \
        {                                       \
            Common->mark = EMPTY ;              \
            cholmod_l_clear_flag (Common) ;     \
        }                                       \
    }

#else

    //--------------------------------------------------------------------------
    // CHOLMOD_INT32: int32
    //--------------------------------------------------------------------------

    #ifndef CHOLMOD_INT32
    #define CHOLMOD_INT32
    #endif

    #define Int int32_t
    #define UInt uint32_t
    #define Int_max INT32_MAX
    #define CHOLMOD(name) cholmod_ ## name
    #define ITYPE CHOLMOD_INT
    #define ID "%d"

    // #define L_GOTCHA ;
    // #define I_GOTCHA GOTCHA

    #define CLEAR_FLAG(Common)                              \
    {                                                       \
        Common->mark++ ;                                    \
        if (Common->mark <= 0 || Common->mark > INT32_MAX)  \
        {                                                   \
            Common->mark = EMPTY ;                          \
            cholmod_clear_flag (Common) ;                   \
        }                                                   \
    }

#endif

//------------------------------------------------------------------------------
// check for BLAS integer overflow
//------------------------------------------------------------------------------

// The conversion of a CHOLMOD integer (Int) to a BLAS/LAPACK integer (the
// SUITESPARSE_BLAS_INT can result in an integer overflow.  This is detected by
// the SUITESPARSE_TO_BLAS_INT macro in SuiteSparse_config.h.  If the error
// condition occurs, that macro sets Common->blas_ok to false, and that call
// and any subsequent calls to the BLAS/LAPACK will be skipped.  From that
// point on, Common->blas_ok will remain false for that call to CHOLMOD.  The
// following macro sets CHOLMOD status to CHOLMOD_TOO_LARGE if the BLAS
// conversion has failed.  This is done only once for a particular call to any
// given CHOLMOD method.

#define CHECK_FOR_BLAS_INTEGER_OVERFLOW                         \
{                                                               \
    if ((sizeof (SUITESPARSE_BLAS_INT) < sizeof (Int)) &&       \
        (Common->status == CHOLMOD_OK) && !(Common->blas_ok))   \
    {                                                           \
        ERROR (CHOLMOD_TOO_LARGE, "BLAS integer overflow") ;    \
    }                                                           \
}

