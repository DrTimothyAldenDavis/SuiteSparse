//------------------------------------------------------------------------------
// GB_bld:  hard-coded functions for builder methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "FactoryKernels/GB_bld__include.h"

// dup operator: Tx [k] += Sx [i], no typecast here
#define GB_BLD_DUP(Tx,k,Sx,i)  if ((Sx [i] > Tx [k]) || (Tx [k] != Tx [k])) { Tx [k] = Sx [i] ; }
#define GB_BLD_COPY(Tx,k,Sx,i) Tx [k] = Sx [i]

// array types for S and T
#define GB_S_TYPE double
#define GB_T_TYPE double

// operator types: z = dup (x,y)
#define GB_Z_TYPE double
#define GB_X_TYPE double
#define GB_Y_TYPE double

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_MAX) || defined(GxB_NO_FP64) || defined(GxB_NO_MAX_FP64))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "omp/include/GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// build a non-iso matrix
//------------------------------------------------------------------------------

GrB_Info GB (_bld__max_fp64)
(
    GB_T_TYPE *restrict Tx,
    int64_t  *restrict Ti,
    const GB_S_TYPE *restrict Sx,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "builder/template/GB_bld_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

