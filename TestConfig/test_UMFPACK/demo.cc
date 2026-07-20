//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/UMFPACK/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "umfpack.h"

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
    double xgood [N] = {36.4, -32.7} ;
    double x [N] ;

    double Control [UMFPACK_CONTROL] ;
    double Info [UMFPACK_INFO] ;
    umfpack_dl_defaults (Control) ;
    Control [UMFPACK_PRL] = 6 ;

    void *Sym, *Num;
    (void) umfpack_dl_symbolic (n, n, Ap, Ai, Ax, &Sym, Control, Info);
    (void) umfpack_dl_numeric (Ap, Ai, Ax, Sym, &Num, Control, Info);
    umfpack_dl_free_symbolic (&Sym);
    int result = umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Num, Control, Info);
    umfpack_dl_free_numeric (&Num);
    for (int i = 0 ; i < n ; i++)
      std::cout << "x [" << i << "] = " << x [i] << std::endl;
    double err = 0;
    for (int i = 0; i < n; i++)
    {
        err = fmax (err, fabs (x [i] - xgood [i]));
    }
    std::cout << "error: " << err << std::endl;

    umfpack_dl_report_status (Control, result) ;
    umfpack_dl_report_info (Control, Info) ;

    return 0;
}
