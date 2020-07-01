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

// op(A)  function:  GB_unop_apply__isfinite_bool_fc64
// op(A') function:  GB_unop_tran__isfinite_bool_fc64

// C type:   bool
// A type:   GxB_FC64_t
// cast:     GxB_FC64_t cij = (aij)
// unaryop:  cij = GB_cisfinite (aij)

#define GB_ATYPE \
    GxB_FC64_t

#define GB_CTYPE \
    bool

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA) \
    GxB_FC64_t aij = Ax [pA]

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x) \
    z = GB_cisfinite (x) ;

// casting
#define GB_CAST(z, aij) \
    GxB_FC64_t z = (aij) ;

// cij = op (aij)
#define GB_CAST_OP(pC,pA)           \
{                                   \
    /* aij = Ax [pA] */             \
    GxB_FC64_t aij = Ax [pA] ;          \
    /* Cx [pC] = op (cast (aij)) */ \
    GxB_FC64_t z = (aij) ;               \
    Cx [pC] = GB_cisfinite (z) ;        \
}

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_ISFINITE || GxB_NO_BOOL || GxB_NO_FC64)

//------------------------------------------------------------------------------
// Cx = op (cast (Ax)): apply a unary operator
//------------------------------------------------------------------------------



GrB_Info GB_unop_apply__isfinite_bool_fc64
(
    bool *Cx,       // Cx and Ax may be aliased
    const GxB_FC64_t *Ax,
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
        GxB_FC64_t aij = Ax [p] ;
        GxB_FC64_t z = (aij) ;
        Cx [p] = GB_cisfinite (z) ;
    }
    return (GrB_SUCCESS) ;
    #endif
}



//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB_unop_tran__isfinite_bool_fc64
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

