//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/SPQR/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "SuiteSparseQR_C.h"

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

    cholmod_sparse *A, A_struct;
    cholmod_dense *B, B_struct;
    cholmod_dense *X;

    // make a shallow CHOLMOD copy of A
    A = &A_struct;
    A->nrow = n;
    A->ncol = n;
    A->p = Ap;
    A->i = Ai;
    A->x = Ax;
    A->z = nullptr;
    A->nzmax = NNZ;
    A->packed = true;
    A->sorted = true;
    A->nz = nullptr;
    A->itype = CHOLMOD_LONG;
    A->dtype = CHOLMOD_DOUBLE;
    A->xtype = CHOLMOD_REAL;
    A->stype = 0;

    // make a shallow CHOLMOD copy of b
    B = &B_struct;
    B->nrow = n;
    B->ncol = 1;
    B->x = b;
    B->z = nullptr;
    B->d = n;
    B->nzmax = n;
    B->dtype = CHOLMOD_DOUBLE;
    B->xtype = CHOLMOD_REAL;

    cholmod_common cc;
    cholmod_l_start (&cc);

    X = SuiteSparseQR_C_backslash_default (A, B, &cc);
    cc.print = 5 ;
    cholmod_l_print_dense (X, "X from QR", &cc);
    cholmod_l_free_dense (&X, &cc) ;

    cholmod_l_finish (&cc);

    return 0;
}
