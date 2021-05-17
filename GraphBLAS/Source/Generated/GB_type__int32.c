//------------------------------------------------------------------------------
// GB_type:  hard-coded functions for each built-in type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_type__include.h"

// The operation is defined by the following types and operators:

// functions:
// C<M>=x (C is dense):          GB (_Cdense_05d__int32)
// C<A>=A (C is dense):          GB (_Cdense_06d__int32)
// C<M>=A (C is empty, A dense): GB (_Cdense_25__int32)
// convert sparse to bitmap:     GB (_convert_s2b__int32)

// C type:   int32_t

#define GB_CTYPE \
    int32_t

// C must have the same type as A or the scalar x
#define GB_ATYPE GB_CTYPE

#define GB_CX(p) Cx [p]

// Cx [p] = scalar
#define GB_COPY_SCALAR_TO_C(p,x) Cx [p] = x

// Cx [p] = Ax [pA]
#define GB_COPY_A_TO_C(Cx,p,Ax,pA) Cx [p] = Ax [pA]

// test the mask condition with Ax [pA]
#define GB_AX_MASK(Ax,pA,asize) \
    (Ax [pA] != 0)

// hard-coded loops can be vectorized
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_INT32)

//------------------------------------------------------------------------------
// C<M>=x, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_05d__int32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GB_void *p_cwork,     // scalar of type C->type
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    int32_t cwork = (*((int32_t *) p_cwork)) ;
    #include "GB_dense_subassign_05d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<A>=A, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_06d__int32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool Mask_struct,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (C->type == A->type) ;
    #include "GB_dense_subassign_06d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M>=A, when C is empty and A is dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_25__int32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (C->type == A->type) ;
    #include "GB_dense_subassign_25_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// convert sparse to bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_convert_s2b__int32)
(
    GrB_Matrix A,
    GB_void *restrict Ax_new_void,
    int8_t  *restrict Ab,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    int32_t *restrict Ax_new = (int32_t *) Ax_new_void ;
    #include "GB_convert_sparse_to_bitmap_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#endif

