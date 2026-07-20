//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/CAMD/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "camd.h"

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
    int64_t Cmem [N] ;
    for (int k = 0; k < n; k++)
      Cmem [k] = 0;
    camd_l_order (n, Ap, Ai, P, nullptr, nullptr, Cmem);
    for (int k = 0; k < n; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;

    return 0;
}
