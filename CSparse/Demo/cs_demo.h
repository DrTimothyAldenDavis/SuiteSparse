// CSparse/Demo/cs_demo.h: include file for CSparse demo programs
// CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
typedef struct problem_struct
{
    cs *A ;
    cs *C ;
    csi sym ;
    double *x ;
    double *b ;
    double *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
csi demo2 (problem *Prob) ;
csi demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
