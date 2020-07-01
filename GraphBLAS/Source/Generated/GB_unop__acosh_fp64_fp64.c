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

// op(A)  function:  GB_unop_apply__acosh_fp64_fp64
// op(A') function:  GB_unop_tran__acosh_fp64_fp64

// C type:   double
// A type:   double
// cast:     double cij = aij
// unaryop:  cij = acosh (aij)

#define GB_ATYPE \
    double

#define GB_CTYPE \
    double

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA) \
    double aij = Ax [pA]

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x) \
    z = acosh (x) ;

// casting
#define GB_CAST(z, aij) \
    double z = aij ;

// cij = op (aij)
#define GB_CAST_OP(pC,pA)           \
{                                   \
    /* aij = Ax [pA] */             \
    double aij = Ax [pA] ;          \
    /* Cx [pC] = op (cast (aij)) */ \
    double z = aij ;               \
    Cx [pC] = acosh (z) ;        \
}

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_ACOSH || GxB_NO_FP64)

//------------------------------------------------------------------------------
// Cx = op (cast (Ax)): apply a unary operator
//------------------------------------------------------------------------------



GrB_Info GB_unop_apply__acosh_fp64_fp64
(
    double *Cx,       // Cx and Ax may be aliased
    const double *Ax,
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
        double aij = Ax [p] ;
        double z = aij ;
        Cx [p] = acosh (z) ;
    }
    return (GrB_SUCCESS) ;
    #endif
}



//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB_unop_tran__acosh_fp64_fp64
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

