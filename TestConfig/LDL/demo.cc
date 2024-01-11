//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/LDL/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>
#include <math.h>

#include "ldl.h"

int main (void)
{
    #define N 2
    int64_t n = N;
    int64_t P [N];

    double xgood [N] = {36.4, -32.7};
    double x [N];
    P [0] = 0 ;
    P [1] = 1 ;
    ldl_l_perm (n, x, xgood, P);
    double err = 0;
    for (int i = 0; i < n; i++)
    {
      std::cout << "x2 [" << i << "] = " << x [i] << std::endl;
      err = fmax (err, fabs (x [i] - xgood [i]));
    }
    std::cout << "error: " << err << std::endl;

    return 0;
}
