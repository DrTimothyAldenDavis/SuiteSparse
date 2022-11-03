// CSparse/Demo/cs_cl_demo3: demo program for CXSparse (complex int64_t)
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs_cl_demo.h"
/* cs_cl_demo3: read a matrix and test Cholesky update/downdate */
int main (void)
{
    problem *Prob = get_problem (stdin, 0) ;
    demo3 (Prob) ;
    free_problem (Prob) ;
    return (0) ;
}
