//------------------------------------------------------------------------------
// GB_uop.c:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "FactoryKernels/GB_uop__include.h"

// unary operator: z = f(x)
#define GB_UNARYOP(z,x) z = x
#define GB_Z_TYPE uint16_t
#define GB_X_TYPE uint16_t

// A matrix
#define GB_A_TYPE uint16_t
#define GB_DECLAREA(aij) uint16_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]

// C matrix
#define GB_C_TYPE uint16_t

// cij = op (aij)
#define GB_APPLY_OP(pC,pA)          \
{                                   \
    /* aij = Ax [pA] */             \
    GB_DECLAREA (aij) ;             \
    GB_GETA (aij, Ax, pA, false) ;  \
    /* Cx [pC] = unaryop (aij) */      \
    GB_UNARYOP (Cx [pC], aij) ;     \
}

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_IDENTITY) || defined(GxB_NO_UINT16))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "apply/include/GB_apply_shared_definitions.h"

//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_uop_tran__identity_uint16_uint16)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "transpose/template/GB_transpose_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

