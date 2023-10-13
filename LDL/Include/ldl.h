//------------------------------------------------------------------------------
// LDL/Include/ldl.h: include file for the LDL package
//------------------------------------------------------------------------------

// LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#ifndef LDL_H
#define LDL_H

/* make it easy for C++ programs to include LDL */
#ifdef __cplusplus
extern "C" {
#endif

#include "SuiteSparse_config.h"

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
/* === int32_t version ====================================================== */
/* ========================================================================== */

void ldl_symbolic (int32_t n, int32_t Ap [ ], int32_t Ai [ ], int32_t Lp [ ],
    int32_t Parent [ ], int32_t Lnz [ ], int32_t Flag [ ], int32_t P [ ],
    int32_t Pinv [ ]) ;

int32_t ldl_numeric (int32_t n, int32_t Ap [ ], int32_t Ai [ ], double Ax [ ],
    int32_t Lp [ ], int32_t Parent [ ], int32_t Lnz [ ], int32_t Li [ ],
    double Lx [ ], double D [ ], double Y [ ], int32_t Pattern [ ],
    int32_t Flag [ ], int32_t P [ ], int32_t Pinv [ ]) ;

void ldl_lsolve (int32_t n, double X [ ], int32_t Lp [ ], int32_t Li [ ],
    double Lx [ ]) ;

void ldl_dsolve (int32_t n, double X [ ], double D [ ]) ;

void ldl_ltsolve (int32_t n, double X [ ], int32_t Lp [ ], int32_t Li [ ],
    double Lx [ ]) ;

void ldl_perm  (int32_t n, double X [ ], double B [ ], int32_t P [ ]) ;
void ldl_permt (int32_t n, double X [ ], double B [ ], int32_t P [ ]) ;

int32_t ldl_valid_perm (int32_t n, int32_t P [ ], int32_t Flag [ ]) ;
int32_t ldl_valid_matrix ( int32_t n, int32_t Ap [ ], int32_t Ai [ ]) ;

/* ========================================================================== */
/* === int64_t version ====================================================== */
/* ========================================================================== */

void ldl_l_symbolic (int64_t n, int64_t Ap [ ], int64_t Ai [ ], int64_t Lp [ ],
    int64_t Parent [ ], int64_t Lnz [ ], int64_t Flag [ ], int64_t P [ ],
    int64_t Pinv [ ]) ;

int64_t ldl_l_numeric (int64_t n, int64_t Ap [ ], int64_t Ai [ ], double Ax [ ],
    int64_t Lp [ ], int64_t Parent [ ], int64_t Lnz [ ], int64_t Li [ ],
    double Lx [ ], double D [ ], double Y [ ], int64_t Pattern [ ],
    int64_t Flag [ ], int64_t P [ ], int64_t Pinv [ ]) ;

void ldl_l_lsolve (int64_t n, double X [ ], int64_t Lp [ ], int64_t Li [ ],
    double Lx [ ]) ;

void ldl_l_dsolve (int64_t n, double X [ ], double D [ ]) ;

void ldl_l_ltsolve (int64_t n, double X [ ], int64_t Lp [ ], int64_t Li [ ],
    double Lx [ ]) ;

void ldl_l_perm  (int64_t n, double X [ ], double B [ ], int64_t P [ ]) ;
void ldl_l_permt (int64_t n, double X [ ], double B [ ], int64_t P [ ]) ;

int64_t ldl_l_valid_perm (int64_t n, int64_t P [ ], int64_t Flag [ ]) ;

int64_t ldl_l_valid_matrix ( int64_t n, int64_t Ap [ ], int64_t Ai [ ]) ;

/* ========================================================================== */
/* === LDL version ========================================================== */
/* ========================================================================== */

#define LDL_DATE "Sept 18, 2023"
#define LDL_MAIN_VERSION   3
#define LDL_SUB_VERSION    2
#define LDL_SUBSUB_VERSION 1

#define LDL_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define LDL_VERSION LDL_VERSION_CODE(LDL_MAIN_VERSION,LDL_SUB_VERSION)

#ifdef __cplusplus
}
#endif

#endif
