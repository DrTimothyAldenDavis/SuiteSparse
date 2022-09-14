/* ========================================================================== */
/* === ldl.h:  include file for the LDL package ============================= */
/* ========================================================================== */

/* Copyright (c) Timothy A Davis, http://www.suitesparse.com.
 * All Rights Reserved.  See LDL/Doc/License.txt for the License.
 */

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
/* === int version ========================================================== */
/* ========================================================================== */

SUITESPARSE_PUBLIC 
void ldl_symbolic (int n, int Ap [ ], int Ai [ ], int Lp [ ],
    int Parent [ ], int Lnz [ ], int Flag [ ], int P [ ], int Pinv [ ]) ;

SUITESPARSE_PUBLIC 
int ldl_numeric (int n, int Ap [ ], int Ai [ ], double Ax [ ],
    int Lp [ ], int Parent [ ], int Lnz [ ], int Li [ ], double Lx [ ],
    double D [ ], double Y [ ], int Pattern [ ], int Flag [ ],
    int P [ ], int Pinv [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_lsolve (int n, double X [ ], int Lp [ ], int Li [ ],
    double Lx [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_dsolve (int n, double X [ ], double D [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_ltsolve (int n, double X [ ], int Lp [ ], int Li [ ],
    double Lx [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_perm  (int n, double X [ ], double B [ ], int P [ ]) ;
SUITESPARSE_PUBLIC 
void ldl_permt (int n, double X [ ], double B [ ], int P [ ]) ;

SUITESPARSE_PUBLIC 
int ldl_valid_perm (int n, int P [ ], int Flag [ ]) ;
SUITESPARSE_PUBLIC 
int ldl_valid_matrix ( int n, int Ap [ ], int Ai [ ]) ;

/* ========================================================================== */
/* === long version ========================================================= */
/* ========================================================================== */

SUITESPARSE_PUBLIC 
void ldl_l_symbolic (int64_t n, int64_t Ap [ ],
    int64_t Ai [ ], int64_t Lp [ ],
    int64_t Parent [ ], int64_t Lnz [ ],
    int64_t Flag [ ], int64_t P [ ],
    int64_t Pinv [ ]) ;

SUITESPARSE_PUBLIC 
int64_t ldl_l_numeric (int64_t n, int64_t Ap [ ],
    int64_t Ai [ ], double Ax [ ], int64_t Lp [ ],
    int64_t Parent [ ], int64_t Lnz [ ],
    int64_t Li [ ], double Lx [ ], double D [ ], double Y [ ],
    int64_t Pattern [ ], int64_t Flag [ ],
    int64_t P [ ], int64_t Pinv [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_l_lsolve (int64_t n, double X [ ], int64_t Lp [ ],
    int64_t Li [ ], double Lx [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_l_dsolve (int64_t n, double X [ ], double D [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_l_ltsolve (int64_t n, double X [ ], int64_t Lp [ ],
    int64_t Li [ ], double Lx [ ]) ;

SUITESPARSE_PUBLIC 
void ldl_l_perm  (int64_t n, double X [ ], double B [ ],
    int64_t P [ ]) ;
void ldl_l_permt (int64_t n, double X [ ], double B [ ],
    int64_t P [ ]) ;

SUITESPARSE_PUBLIC 
int64_t ldl_l_valid_perm (int64_t n, int64_t P [ ],
    int64_t Flag [ ]) ;

SUITESPARSE_PUBLIC 
int64_t ldl_l_valid_matrix ( int64_t n,
    int64_t Ap [ ], int64_t Ai [ ]) ;

/* ========================================================================== */
/* === LDL version ========================================================== */
/* ========================================================================== */

#define LDL_DATE "May 4, 2016"
#define LDL_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define LDL_MAIN_VERSION 2
#define LDL_SUB_VERSION 2
#define LDL_SUBSUB_VERSION 6
#define LDL_VERSION LDL_VERSION_CODE(LDL_MAIN_VERSION,LDL_SUB_VERSION)

