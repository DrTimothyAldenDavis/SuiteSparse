//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/CXSparse/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "cs.h"

int main (void)
{
    #define N 2
    #define NNZ 4
    int64_t n = N;
    int64_t nzmax = NNZ;
    cs_dl *A = cs_dl_spalloc (n, n, nzmax, true, false);
    int64_t *Ap = A->p;
    int64_t *Ai = A->i;
    double  *Ax = A->x;
    Ap [0] = 0;
    Ap [1] = 2;
    Ap [2] = NNZ;
    Ai [0] = 0;
    Ai [1] = 1;
    Ai [2] = 0;
    Ai [3] = 1;
    Ax [0] = 11.0;
    Ax [1] = 21.0;
    Ax [2] = 12.0;
    Ax [3] = 22.0;

    cs_dl_print (A, false);
    cs_dl_spfree (A);

    return 0;
}
