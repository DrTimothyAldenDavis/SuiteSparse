//------------------------------------------------------------------------------
// GB_binop:  hard-coded functions for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_dense.h"
#include "GB_mkl.h"
#include "GB_binop__include.h"

// C=binop(A,B) is defined by the following types and operators:

// A+B function (eWiseAdd):         GB_AaddB
// A.*B function (eWiseMult):       GB_AemultB
// A*D function (colscale):         GB_AxD
// D*A function (rowscale):         GB_DxB
// C+=B function (dense accum):     GB_Cdense_accumB
// C+=b function (dense accum):     GB_Cdense_accumb
// C+=A+B function (dense ewise3):  GB_Cdense_ewise3_accum
// C=A+B function (dense ewise3):   GB_Cdense_ewise3_noaccum
// C=scalar+B                       GB_bind1st
// C=scalar+B'                      GB_bind1st_tran
// C=A+scalar                       GB_bind2nd
// C=A'+scalar                      GB_bind2nd_tran

// C type:   GB_ctype
// A type:   GB_atype
// B,b type: GB_btype
// BinaryOp: GB_binaryop(cij,aij,bij)

#define GB_ATYPE \
    GB_atype

#define GB_BTYPE \
    GB_btype

#define GB_CTYPE \
    GB_ctype

// true if the types of A and B are identical
#define GB_ATYPE_IS_BTYPE \
    GB_atype_is_btype

// true if the types of C and A are identical
#define GB_CTYPE_IS_ATYPE \
    GB_ctype_is_atype

// true if the types of C and B are identical
#define GB_CTYPE_IS_BTYPE \
    GB_ctype_is_btype

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA)  \
    GB_geta(aij,Ax,pA)

// bij = Bx [pB]
#define GB_GETB(bij,Bx,pB)  \
    GB_getb(bij,Bx,pB)

// declare scalar of the same type as C
#define GB_CTYPE_SCALAR(t)  \
    GB_ctype t

// cij = Ax [pA]
#define GB_COPY_A_TO_C(cij,Ax,pA) \
    GB_copy_a_to_c(cij,Ax,pA)

// cij = Bx [pB]
#define GB_COPY_B_TO_C(cij,Bx,pB) \
    GB_copy_b_to_c(cij,Bx,pB)

#define GB_CX(p) Cx [p]

// binary operator
#define GB_BINOP(z, x, y)   \
    GB_binaryop(z, x, y) ;

// op is second
#define GB_OP_IS_SECOND \
    GB_op_is_second

// op is plus_fp32 or plus_fp64
#define GB_OP_IS_PLUS_REAL \
    GB_op_is_plus_real

// op is minus_fp32 or minus_fp64
#define GB_OP_IS_MINUS_REAL \
    GB_op_is_minus_real

// GB_cblas_*axpy gateway routine, if it exists for this operator and type:
#define GB_CBLAS_AXPY \
    GB_cblas_axpy

// do the numerical phases of GB_add and GB_emult
#define GB_PHASE_2_OF_2

// hard-coded loops can be vectorized
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    GB_disable

//------------------------------------------------------------------------------
// C += A+B, all 3 matrices dense
//------------------------------------------------------------------------------

if_is_binop_subset

// The op must be MIN, MAX, PLUS, MINUS, RMINUS, TIMES, DIV, or RDIV.

void GB_Cdense_ewise3_accum
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "GB_dense_ewise3_accum_template.c"
}

endif_is_binop_subset

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB_Cdense_ewise3_noaccum
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_dense_ewise3_noaccum_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB_Cdense_accumB
(
    GrB_Matrix C,
    const GrB_Matrix B,
    const int64_t *GB_RESTRICT kfirst_slice,
    const int64_t *GB_RESTRICT klast_slice,
    const int64_t *GB_RESTRICT pstart_slice,
    const int ntasks,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    if_C_dense_update
    { 
        #include "GB_dense_subassign_23_template.c"
    }
    endif_C_dense_update
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += b, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB_Cdense_accumb
(
    GrB_Matrix C,
    const GB_void *p_bwork,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    if_C_dense_update
    { 
        // get the scalar b for C += b, of type GB_btype
        GB_btype bwork = (*((GB_btype *) p_bwork)) ;
        #include "GB_dense_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    endif_C_dense_update
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C = A*D, column scale with diagonal D matrix
//------------------------------------------------------------------------------

if_binop_is_semiring_multiplier

GrB_Info GB_AxD
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix D, bool D_is_pattern,
    const int64_t *GB_RESTRICT kfirst_slice,
    const int64_t *GB_RESTRICT klast_slice,
    const int64_t *GB_RESTRICT pstart_slice,
    const int ntasks,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_ctype *GB_RESTRICT Cx = (GB_ctype *) C->x ;
    #include "GB_AxB_colscale_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

endif_binop_is_semiring_multiplier

//------------------------------------------------------------------------------
// C = D*B, row scale with diagonal D matrix
//------------------------------------------------------------------------------

if_binop_is_semiring_multiplier

GrB_Info GB_DxB
(
    GrB_Matrix C,
    const GrB_Matrix D, bool D_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_ctype *GB_RESTRICT Cx = (GB_ctype *) C->x ;
    #include "GB_AxB_rowscale_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

endif_binop_is_semiring_multiplier

//------------------------------------------------------------------------------
// eWiseAdd: C = A+B or C<M> = A+B
//------------------------------------------------------------------------------

GrB_Info GB_AaddB
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *GB_RESTRICT C_to_M,
    const int64_t *GB_RESTRICT C_to_A,
    const int64_t *GB_RESTRICT C_to_B,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C = A.*B or C<M> = A.*B
//------------------------------------------------------------------------------

GrB_Info GB_AemultB
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *GB_RESTRICT C_to_M,
    const int64_t *GB_RESTRICT C_to_A,
    const int64_t *GB_RESTRICT C_to_B,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_emult_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// Cx = op (x,Bx):  apply a binary operator to a matrix with scalar bind1st
//------------------------------------------------------------------------------

if_binop_bind1st_is_enabled

GrB_Info GB_bind1st
(
    GB_void *Cx_output,         // Cx and Bx may be aliased
    const GB_void *x_input,
    const GB_void *Bx_input,
    int64_t anz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_ctype *Cx = (GB_ctype *) Cx_output ;
    GB_atype   x = (*((GB_atype *) x_input)) ;
    GB_btype *Bx = (GB_btype *) Bx_input ;
    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    {
        GB_getb(bij, Bx, p) ;
        GB_binaryop(Cx [p], x, bij) ;
    }
    return (GrB_SUCCESS) ;
    #endif
}

endif_binop_bind1st_is_enabled

//------------------------------------------------------------------------------
// Cx = op (Ax,y):  apply a binary operator to a matrix with scalar bind2nd
//------------------------------------------------------------------------------

if_binop_bind2nd_is_enabled

GrB_Info GB_bind2nd
(
    GB_void *Cx_output,         // Cx and Ax may be aliased
    const GB_void *Ax_input,
    const GB_void *y_input,
    int64_t anz,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    int64_t p ;
    GB_ctype *Cx = (GB_ctype *) Cx_output ;
    GB_atype *Ax = (GB_atype *) Ax_input ;
    GB_btype   y = (*((GB_btype *) y_input)) ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    {
        GB_geta(aij, Ax, p) ;
        GB_binaryop(Cx [p], aij, y) ;
    }
    return (GrB_SUCCESS) ;
    #endif
}

endif_binop_bind2nd_is_enabled

//------------------------------------------------------------------------------
// C = op (x, A'): transpose and apply a binary operator
//------------------------------------------------------------------------------

if_binop_bind1st_is_enabled

// cij = op (x, aij), no typcasting (in spite of the macro name)
#undef  GB_CAST_OP
#define GB_CAST_OP(pC,pA)               \
{                                       \
    GB_getb(aij, Ax, pA) ;              \
    GB_binaryop(Cx [pC], x, aij) ;      \
}

GrB_Info GB_bind1st_tran
(
    GrB_Matrix C,
    const GB_void *x_input,
    const GrB_Matrix A,
    int64_t *GB_RESTRICT *Rowcounts,
    GBI_single_iterator Iter,
    const int64_t *GB_RESTRICT A_slice,
    int naslice
)
{ 
    // GB_unop_transpose.c uses GB_ATYPE, but A is
    // the 2nd input to binary operator z=f(x,y).
    #undef  GB_ATYPE
    #define GB_ATYPE \
    GB_btype
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_atype x = (*((const GB_atype *) x_input)) ;
    #define GB_PHASE_2_OF_2
    #include "GB_unop_transpose.c"
    return (GrB_SUCCESS) ;
    #endif
    #undef  GB_ATYPE
    #define GB_ATYPE \
    GB_atype
}

endif_binop_bind1st_is_enabled

//------------------------------------------------------------------------------
// C = op (A', y): transpose and apply a binary operator
//------------------------------------------------------------------------------

if_binop_bind2nd_is_enabled

// cij = op (aij, y), no typcasting (in spite of the macro name)
#undef  GB_CAST_OP
#define GB_CAST_OP(pC,pA)               \
{                                       \
    GB_geta(aij, Ax, pA) ;              \
    GB_binaryop(Cx [pC], aij, y) ;      \
}

GrB_Info GB_bind2nd_tran
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *y_input,
    int64_t *GB_RESTRICT *Rowcounts,
    GBI_single_iterator Iter,
    const int64_t *GB_RESTRICT A_slice,
    int naslice
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_btype y = (*((const GB_btype *) y_input)) ;
    #define GB_PHASE_2_OF_2
    #include "GB_unop_transpose.c"
    return (GrB_SUCCESS) ;
    #endif
}

endif_binop_bind2nd_is_enabled

#endif

