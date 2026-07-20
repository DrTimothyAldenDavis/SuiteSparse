//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/BTF/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "btf.h"

int main (void)
{
    #define N 2
    #define NNZ 4
    int64_t n = N;
    int64_t nzmax = NNZ;
    int64_t Ap[N+1];
    int64_t Ai[NNZ];
    Ap [0] = 0;
    Ap [1] = 2;
    Ap [2] = NNZ;
    Ai [0] = 0;
    Ai [1] = 1;
    Ai [2] = 0;
    Ai [3] = 1;

    int64_t P [N];
    double work;
    int64_t nmatch;
    int64_t Q [N], R [N+1], Work [5*N];
    int64_t nblocks = btf_l_order (n, Ap, Ai, -1, &work, P, Q, R, &nmatch, Work);
    for (int k = 0; k < n; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;
    for (int k = 0; k < n; k++)
      std::cout << "Q [" << k << "] = " << Q [k] << std::endl;
    std::cout << "nblocks " << nblocks << std::endl;

    return 0;
}
