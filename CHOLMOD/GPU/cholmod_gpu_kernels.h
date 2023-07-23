//------------------------------------------------------------------------------
// CHOLMOD/GPU/cholmod_gpu_kernels.h: include file for GPU utilities
//------------------------------------------------------------------------------

// CHOLMOD/GPU Module.  Copyright (C) 2005-2022, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* -----------------------------------------------------------------------------
 * CUDA kernel support routines for CHOLMOD
 * -------------------------------------------------------------------------- */

#ifndef CHOLMODGPUKERNELS_H
#define CHOLMODGPUKERNELS_H

/* make it easy for C++ programs to include CHOLMOD */
#ifdef __cplusplus
extern "C" {
#endif

#include "SuiteSparse_config.h"

int createMapOnDevice ( SuiteSparse_long *d_Map, SuiteSparse_long *d_Ls,
    SuiteSparse_long psi, SuiteSparse_long nsrow ) ;

int createRelativeMapOnDevice ( SuiteSparse_long *d_Map,
    SuiteSparse_long *d_Ls, SuiteSparse_long *d_RelativeMap,
    SuiteSparse_long pdi1, SuiteSparse_long ndrow, cudaStream_t* astream ) ;

int addUpdateOnDevice ( double *d_A, double *devPtrC,
    SuiteSparse_long *d_RelativeMap,
    SuiteSparse_long ndrow1, SuiteSparse_long ndrow2, SuiteSparse_long nsrow,
    cudaStream_t* astream ) ;

int addComplexUpdateOnDevice ( double *d_A, double *devPtrC, 
    SuiteSparse_long *d_RelativeMap, SuiteSparse_long ndrow1, 
    SuiteSparse_long ndrow2, SuiteSparse_long nsrow, 
    cudaStream_t* astream ) ;

int sumAOnDevice ( double *a1, double *a2, const double alpha, int nsrow,
    int nscol );

int sumComplexAOnDevice ( double *a1, double *a2, const double alpha,
    int nsrow, int nscol );

#ifdef __cplusplus
}
#endif

#endif
