//------------------------------------------------------------------------------
// GB_unop:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_unop__include.h"

// C=unop(A) is defined by the following types and operators:

// op(A)  function:  GB_unop_apply__cimag_fp32_fc32
// op(A') function:  GB_unop_tran__cimag_fp32_fc32

// C type:   float
// A type:   GxB_FC32_t
// cast:     GxB_FC32_t cij = (aij)
// unaryop:  cij = cimagf (aij)

#define GB_ATYPE \
    GxB_FC32_t

#define GB_CTYPE \
    float

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA) \
    GxB_FC32_t aij = Ax [pA]

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x) \
    z = cimagf (x) ;

// casting
#define GB_CAST(z, aij) \
    GxB_FC32_t z = (aij) ;

// cij = op (aij)
#define GB_CAST_OP(pC,pA)           \
{                                   \
    /* aij = Ax [pA] */             \
    GxB_FC32_t aij = Ax [pA] ;          \
    /* Cx [pC] = op (cast (aij)) */ \
    GxB_FC32_t z = (aij) ;               \
    Cx [pC] = cimagf (z) ;        \
}

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_CIMAG || GxB_NO_FP32 || GxB_NO_FC32)

//------------------------------------------------------------------------------
// Cx = op (cast (Ax)): apply a unary operator
//------------------------------------------------------------------------------



GrB_Info GB_unop_apply__cimag_fp32_fc32
(
    float *Cx,       // Cx and Ax may be aliased
    const GxB_FC32_t *Ax,
    int64_t anz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    {
        GxB_FC32_t aij = Ax [p] ;
        GxB_FC32_t z = (aij) ;
        Cx [p] = cimagf (z) ;
    }
    return (GrB_SUCCESS) ;
    #endif
}



//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB_unop_tran__cimag_fp32_fc32
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *GB_RESTRICT *Rowcounts,
    GBI_single_iterator Iter,
    const int64_t *GB_RESTRICT A_slice,
    int naslice
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #define GB_PHASE_2_OF_2
    #include "GB_unop_transpose.c"
    return (GrB_SUCCESS) ;
    #endif
}

#endif

