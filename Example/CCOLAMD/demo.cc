//------------------------------------------------------------------------------
// SuiteSparse/Example/CCOLAMD/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example program

#include <iostream>

#include "ccolamd.h"

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
    int64_t Alen = ccolamd_l_recommended (NNZ, n, n);
    int64_t *Awork = (int64_t *) malloc (Alen * sizeof (int64_t));
    memcpy (Awork, Ai, NNZ * sizeof (int64_t));
    ccolamd_l (n, n, Alen, Awork, P, nullptr, nullptr, Cmem);
    for (int k = 0; k < n; k++)
      std::cout << "P [" << k << "] = " << P [k] << std::endl;
    free (Awork);

    return 0;
}
