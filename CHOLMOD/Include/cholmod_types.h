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
// The complex types (ANSI-compatible complex, and MATLAB-compatable zomplex)
// are based on the double or float type, and are not selected here.  They are
// typically selected via template routines.

// -----------------------------------------------------------------------------

#undef Int
#undef UInt
#undef Int_max
#undef CHOLMOD
#undef ITYPE
#undef ID
#undef CLEAR_FLAG

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

