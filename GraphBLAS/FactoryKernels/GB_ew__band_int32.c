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
#define GB_BINOP(z,x,y,i,j) z = ((x) & (y))
#define GB_Z_TYPE int32_t
#define GB_X_TYPE int32_t
#define GB_Y_TYPE int32_t

// A matrix:
#define GB_A_TYPE int32_t
#define GB_A2TYPE int32_t
#define GB_DECLAREA(aij) int32_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [(A_iso) ? 0 : (pA)]

// B matrix:
#define GB_B_TYPE int32_t
#define GB_B2TYPE int32_t
#define GB_DECLAREB(bij) int32_t bij
#define GB_GETB(bij,Bx,pB,B_iso) bij = Bx [(B_iso) ? 0 : (pB)]

// C matrix:
#define GB_C_TYPE int32_t

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_BAND) || defined(GxB_NO_INT32) || defined(GxB_NO_BAND_INT32))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "ewise/include/GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cewise_fulln__band_int32)
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

GrB_Info GB (_AaddB__band_int32)
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

GrB_Info GB (_AunionB__band_int32)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
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
    GB_X_TYPE alpha_scalar = (*((GB_X_TYPE *) alpha_scalar_in)) ;
    GB_Y_TYPE beta_scalar  = (*((GB_Y_TYPE *) beta_scalar_in )) ;
    #define GB_IS_EWISEUNION 1
    // for the "easy mask" condition:
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_B = GB_all_aliased (M, B) ;
    #include "ewise/template/GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C=A.*B, C<M>=A.*B, or C<M!>=A.*B where C is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_08__band_int32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "ewise/template/GB_emult_08_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<#> = A.*B when A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_02__band_int32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "ewise/template/GB_emult_02_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<M> = A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_04__band_int32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "ewise/template/GB_emult_04_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C=A.*B, C<M>=A.*B, C<!M>=A.*B where C is bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_bitmap__band_int32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads,
    const int C_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "ewise/template/GB_emult_bitmap_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// Cx = op (x,Bx):  apply a binary operator to a matrix with scalar bind1st
//------------------------------------------------------------------------------

GrB_Info GB (_bind1st__band_int32)
(
    GB_void *Cx_output,         // Cx and Bx may be aliased
    const GB_void *x_input,
    const GB_void *Bx_input,
    const int8_t *restrict Bb,
    int64_t bnz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "apply/template/GB_apply_bind1st_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// Cx = op (Ax,y):  apply a binary operator to a matrix with scalar bind2nd
//------------------------------------------------------------------------------

GrB_Info GB (_bind2nd__band_int32)
(
    GB_void *Cx_output,         // Cx and Ax may be aliased
    const GB_void *Ax_input,
    const GB_void *y_input,
    const int8_t *restrict Ab,
    int64_t anz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "apply/template/GB_apply_bind2nd_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C = op (x, A'): transpose and apply a binary operator
//------------------------------------------------------------------------------

// cij = op (x, aij)
#undef  GB_APPLY_OP
#define GB_APPLY_OP(pC,pA)                      \
{                                               \
    GB_DECLAREB (aij) ;                         \
    GB_GETB (aij, Ax, pA, false) ;              \
    GB_EWISEOP (Cx, pC, x, aij, 0, 0) ;         \
}

GrB_Info GB (_bind1st_tran__band_int32)
(
    GrB_Matrix C,
    const GB_void *x_input,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #define GB_BIND_1ST
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_X_TYPE x = (*((const GB_X_TYPE *) x_input)) ;
    #include "transpose/template/GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
    #undef GB_BIND_1ST
}

//------------------------------------------------------------------------------
// C = op (A', y): transpose and apply a binary operator
//------------------------------------------------------------------------------

// cij = op (aij, y)
#undef  GB_APPLY_OP
#define GB_APPLY_OP(pC,pA)                      \
{                                               \
    GB_DECLAREA (aij) ;                         \
    GB_GETA (aij, Ax, pA, false) ;              \
    GB_EWISEOP (Cx, pC, aij, y, 0, 0) ;         \
}

GrB_Info GB (_bind2nd_tran__band_int32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *y_input,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_Y_TYPE y = (*((const GB_Y_TYPE *) y_input)) ;
    #include "transpose/template/GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

