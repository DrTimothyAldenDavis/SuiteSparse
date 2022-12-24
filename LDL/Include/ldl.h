//------------------------------------------------------------------------------
// LDL/Include/ldl.h: include file for the LDL package
//------------------------------------------------------------------------------

// LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "SuiteSparse_config.h"

//------------------------------------------------------------------------------
// importing/exporting symbols for Microsoft Visual Studio
//------------------------------------------------------------------------------

#if SUITESPARSE_COMPILER_MSC

    // dllimport/dllexport on Windows
    #if defined ( LDL_LIBRARY )
        // compiling SuiteSparse itself, exporting symbols to user apps
        #define LDL_PUBLIC extern __declspec ( dllexport )
    #elif defined ( LDL_STATIC )
        // compiling static library, no dllimport or dllexport
        #define LDL_PUBLIC extern
    #else
        // compiling the user application, importing symbols from SuiteSparse
        #define LDL_PUBLIC extern __declspec ( dllimport )
    #endif

#else

    // for other compilers
    #define LDL_PUBLIC extern

#endif


#ifdef LDL_LONG
#define LDL_int int64_t
#define LDL_ID  "%" PRId64

#define LDL_symbolic ldl_l_symbolic
#define LDL_numeric ldl_l_numeric
#define LDL_lsolve ldl_l_lsolve
#define LDL_dsolve ldl_l_dsolve
#define LDL_ltsolve ldl_l_ltsolve
#define LDL_perm ldl_l_perm
#define LDL_permt ldl_l_permt
#define LDL_valid_perm ldl_l_valid_perm
#define LDL_valid_matrix ldl_l_valid_matrix

#else
#define LDL_int int32_t
#define LDL_ID "%d"

#define LDL_symbolic ldl_symbolic
#define LDL_numeric ldl_numeric
#define LDL_lsolve ldl_lsolve
#define LDL_dsolve ldl_dsolve
#define LDL_ltsolve ldl_ltsolve
#define LDL_perm ldl_perm
#define LDL_permt ldl_permt
#define LDL_valid_perm ldl_valid_perm
#define LDL_valid_matrix ldl_valid_matrix

#endif

/* ========================================================================== */
/* === int32_t version ========================================================== */
/* ========================================================================== */

LDL_PUBLIC
void ldl_symbolic (int32_t n, int32_t Ap [ ], int32_t Ai [ ], int32_t Lp [ ],
    int32_t Parent [ ], int32_t Lnz [ ], int32_t Flag [ ], int32_t P [ ],
    int32_t Pinv [ ]) ;

LDL_PUBLIC
int32_t ldl_numeric (int32_t n, int32_t Ap [ ], int32_t Ai [ ], double Ax [ ],
    int32_t Lp [ ], int32_t Parent [ ], int32_t Lnz [ ], int32_t Li [ ],
    double Lx [ ], double D [ ], double Y [ ], int32_t Pattern [ ],
    int32_t Flag [ ], int32_t P [ ], int32_t Pinv [ ]) ;

LDL_PUBLIC
void ldl_lsolve (int32_t n, double X [ ], int32_t Lp [ ], int32_t Li [ ],
    double Lx [ ]) ;

LDL_PUBLIC
void ldl_dsolve (int32_t n, double X [ ], double D [ ]) ;

LDL_PUBLIC
void ldl_ltsolve (int32_t n, double X [ ], int32_t Lp [ ], int32_t Li [ ],
    double Lx [ ]) ;

LDL_PUBLIC
void ldl_perm  (int32_t n, double X [ ], double B [ ], int32_t P [ ]) ;
LDL_PUBLIC
void ldl_permt (int32_t n, double X [ ], double B [ ], int32_t P [ ]) ;

LDL_PUBLIC
int32_t ldl_valid_perm (int32_t n, int32_t P [ ], int32_t Flag [ ]) ;
LDL_PUBLIC
int32_t ldl_valid_matrix ( int32_t n, int32_t Ap [ ], int32_t Ai [ ]) ;

/* ========================================================================== */
/* === int64_t version ====================================================== */
/* ========================================================================== */

LDL_PUBLIC
void ldl_l_symbolic (int64_t n, int64_t Ap [ ], int64_t Ai [ ], int64_t Lp [ ],
    int64_t Parent [ ], int64_t Lnz [ ], int64_t Flag [ ], int64_t P [ ],
    int64_t Pinv [ ]) ;

LDL_PUBLIC
int64_t ldl_l_numeric (int64_t n, int64_t Ap [ ], int64_t Ai [ ], double Ax [ ],
    int64_t Lp [ ], int64_t Parent [ ], int64_t Lnz [ ], int64_t Li [ ],
    double Lx [ ], double D [ ], double Y [ ], int64_t Pattern [ ],
    int64_t Flag [ ], int64_t P [ ], int64_t Pinv [ ]) ;

LDL_PUBLIC
void ldl_l_lsolve (int64_t n, double X [ ], int64_t Lp [ ], int64_t Li [ ],
    double Lx [ ]) ;

LDL_PUBLIC
void ldl_l_dsolve (int64_t n, double X [ ], double D [ ]) ;

LDL_PUBLIC
void ldl_l_ltsolve (int64_t n, double X [ ], int64_t Lp [ ], int64_t Li [ ],
    double Lx [ ]) ;

LDL_PUBLIC
void ldl_l_perm  (int64_t n, double X [ ], double B [ ], int64_t P [ ]) ;

LDL_PUBLIC
void ldl_l_permt (int64_t n, double X [ ], double B [ ], int64_t P [ ]) ;

LDL_PUBLIC
int64_t ldl_l_valid_perm (int64_t n, int64_t P [ ], int64_t Flag [ ]) ;

LDL_PUBLIC
int64_t ldl_l_valid_matrix ( int64_t n, int64_t Ap [ ], int64_t Ai [ ]) ;

/* ========================================================================== */
/* === LDL version ========================================================== */
/* ========================================================================== */

#define LDL_DATE "Dec 9, 2022"
#define LDL_MAIN_VERSION   3
#define LDL_SUB_VERSION    0
#define LDL_SUBSUB_VERSION 2

#define LDL_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define LDL_VERSION LDL_VERSION_CODE(LDL_MAIN_VERSION,LDL_SUB_VERSION)

