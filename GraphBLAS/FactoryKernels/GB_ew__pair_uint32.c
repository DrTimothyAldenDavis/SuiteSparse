//------------------------------------------------------------------------------
// GB_ew: ewise kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "ewise/GB_emult.h"
#include "slice/GB_ek_slice.h"
#include "assign/GB_bitmap_assign_methods.h"
#include "FactoryKernels/GB_ew__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = 1
#define GB_Z_TYPE uint32_t
#define GB_X_TYPE uint32_t
#define GB_Y_TYPE uint32_t

// A matrix:
#define GB_A_TYPE uint32_t
#define GB_A2TYPE void
#define GB_DECLAREA(aij) uint32_t aij
#define GB_GETA(aij,Ax,pA,A_iso)

// B matrix:
#define GB_B_TYPE uint32_t
#define GB_B2TYPE void
#define GB_DECLAREB(bij) uint32_t bij
#define GB_GETB(bij,Bx,pB,B_iso)

// C matrix:
#define GB_C_TYPE uint32_t

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_PAIR) || defined(GxB_NO_UINT32) || defined(GxB_NO_PAIR_UINT32))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "ewise/include/GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cewise_fulln__pair_uint32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "ewise/template/GB_ewise_fulln_template.c"
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// eWiseAdd: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AaddB__pair_uint32)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #define GB_IS_EWISEUNION 0
    // for the "easy mask" condition:
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_B = GB_all_aliased (M, B) ;
    #include "ewise/template/GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseUnion: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

