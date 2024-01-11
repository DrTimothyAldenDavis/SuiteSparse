//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/KLU/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>
#include <math.h>

#include "klu.h"

int main (void)
{
    #define N 2
    #define NNZ 4
    int64_t n = N;
    int64_t nzmax = NNZ;
    int64_t Ap[N+1];
    int64_t Ai[NNZ];
    double Ax[NNZ];
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

    double b [N] = {8., 45.};
    double xgood [N] = {36.4, -32.7};
    double x [N];

    klu_l_symbolic *Symbolic;
    klu_l_numeric *Numeric;
    klu_l_common Common;
    klu_l_defaults (&Common);
    Symbolic = klu_l_analyze (n, Ap, Ai, &Common);
    Numeric = klu_l_factor (Ap, Ai, Ax, Symbolic, &Common);
    memcpy (x, b, N * sizeof (double));
    klu_l_solve (Symbolic, Numeric, 5, 1, x, &Common);
    klu_l_free_symbolic (&Symbolic, &Common);
    klu_l_free_numeric (&Numeric, &Common);
    double err = 0;
    for (int i = 0; i < n; i++)
    {
      std::cout << "x [" << i << "] = " << x [i] << std::endl;
      err = fmax (err, fabs (x [i] - xgood [i]));
    }
    std::cout << "error: " << err << std::endl;

    return 0;
}
