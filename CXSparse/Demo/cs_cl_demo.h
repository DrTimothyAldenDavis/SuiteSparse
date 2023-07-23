// CXSparse/Demo/cs_cl_demo.h: include file for CXSparse demo programs
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
typedef struct problem_struct
{
    cs_cl *A ;
    cs_cl *C ;
    SuiteSparse_long sym ;
    cs_complex_t *x ;
    cs_complex_t *b ;
    cs_complex_t *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
SuiteSparse_long demo2 (problem *Prob) ;
SuiteSparse_long demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
