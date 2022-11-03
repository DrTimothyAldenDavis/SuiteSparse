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

int createMapOnDevice ( int64_t *d_Map, int64_t *d_Ls, int64_t psi, int64_t nsrow ); 

int createRelativeMapOnDevice ( int64_t *d_Map, int64_t *d_Ls, 
				  int64_t *d_RelativeMap,int64_t  pdi1, 
				  int64_t ndrow, cudaStream_t* astream ) ;

int addUpdateOnDevice ( double *d_A, double *devPtrC, int64_t *d_RelativeMap,
    int64_t ndrow1, int64_t ndrow2, int64_t nsrow, cudaStream_t* astream ) ;

int addComplexUpdateOnDevice ( double *d_A, double *devPtrC, 
				 int64_t *d_RelativeMap, int64_t ndrow1, 
				 int64_t ndrow2, int64_t nsrow, 
				 cudaStream_t* astream ) ;

int sumAOnDevice ( double *a1, double *a2, const double alpha, int nsrow,
    int nscol );

int sumComplexAOnDevice ( double *a1, double *a2, const double alpha,
    int nsrow, int nscol );

#ifdef __cplusplus
}
#endif

#endif
