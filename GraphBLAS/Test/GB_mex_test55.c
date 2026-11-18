//------------------------------------------------------------------------------
// GB_mex_test55: indexed operand flip in generic saxpy method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Regression test for a bug in versions 10.5.3 and earlier, in GrB_mxm
// with a index binary operator as the multiplicative op of the semiring.
// See issue #479.

#include "GB_mex.h"
#include "GB_mex_errors.h"

typedef struct { uint64_t value [8] ; } Big ;

static void multiply
(
    void *z, const void *x, GrB_Index ix, GrB_Index jx,
    const void *y, GrB_Index iy, GrB_Index jy, const void *theta
)
{
    (void) theta ;
    *(uint64_t *) z = *(const bool *) y ?
        ((const Big *) x)->value [7] + 100 * ix + 10 * jx + iy + jy : 0 ;
}

//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free (&A) ;              \
    GrB_Matrix_free (&B) ;              \
    GrB_Matrix_free (&C) ;              \
    GrB_Descriptor_free (&descriptor) ; \
}

static GrB_Info GB_mex_test55_worker
(
    bool by_row,
    int sparsity,
    GrB_Type big,
    GrB_Semiring semiring
)
{

    //--------------------------------------------------------------------------
    // construct test matrices
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Descriptor descriptor = NULL ;
    OK (GrB_Matrix_new (&A, big, 2, 3)) ;
    OK (GrB_Matrix_new (&B, GrB_BOOL, 3, 4)) ;
    OK (GrB_Matrix_new (&C, GrB_UINT64, 2, 4)) ;

    int orientation = by_row ? GrB_ROWMAJOR : GrB_COLMAJOR ;
    OK (GrB_set (A, orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_set (B, orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_set (C, orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GxB_Matrix_Option_set_INT32 (C, GxB_SPARSITY_CONTROL, sparsity)) ;
    Big a = { { 0 } } ;
    a.value [7] = 1234 ;
    OK (GrB_Matrix_setElement_UDT (A, &a, 0, 1)) ;
    a.value [7] = 5678 ;
    OK (GrB_Matrix_setElement_UDT (A, &a, 1, 2)) ;
    OK (GrB_Matrix_setElement_BOOL (B, true, 1, 2)) ;
    OK (GrB_Matrix_setElement_BOOL (B, true, 2, 0)) ;
    OK (GrB_Descriptor_new (&descriptor)) ;
    OK (GrB_Descriptor_set (descriptor, GxB_AxB_METHOD, GxB_AxB_SAXPY)) ;
    OK (GrB_mxm (C, NULL, NULL, semiring, A, B, descriptor)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    uint64_t x = 0, y = 0 ;
    GrB_Index count = 0 ;
    OK (GrB_Matrix_extractElement_UINT64 (&x, C, 0, 2)) ;
    OK (GrB_Matrix_extractElement_UINT64 (&y, C, 1, 0)) ;
    OK (GrB_Matrix_nvals (&count, C)) ;
    bool ok = (x == 1247 && y == 5800 && count == 2) ;
    if (!ok)
    {
        printf ("Incorrect result: row=%d format=%d x=%llu y=%llu count=%llu\n",
            by_row, sparsity, (unsigned long long) x, (unsigned long long) y,
            (unsigned long long) count) ;
    }

    //--------------------------------------------------------------------------
    // free everything and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    return (ok ? GrB_SUCCESS : GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GB_mex_test54 mexFunction
//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_free (&semiring) ;          \
    GrB_free (&op) ;                \
    GrB_free (&indexed) ;           \
    GrB_free (&theta) ;             \
    GrB_free (&big) ;               \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Type big = NULL ;
    GxB_IndexBinaryOp indexed = NULL ;
    GrB_BinaryOp op = NULL ;
    GrB_Scalar theta = NULL ;
    GrB_Semiring semiring = NULL ;

    //--------------------------------------------------------------------------
    // construct the type, ops, and semiring
    //--------------------------------------------------------------------------

    // Exercise the generic kernels rather than JIT-generated multiplication.
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;

    OK (GrB_Type_new (&big, sizeof (Big))) ;
    OK (GxB_IndexBinaryOp_new (&indexed, multiply, GrB_UINT64, big, GrB_BOOL,
        GrB_UINT64, "test_indexed_saxpy_flip", NULL)) ;
    OK (GrB_Scalar_new (&theta, GrB_UINT64)) ;
    OK (GrB_Scalar_setElement_UINT64 (theta, 0)) ;
    OK (GxB_BinaryOp_new_IndexOp (&op, indexed, theta)) ;
    OK (GrB_Semiring_new (&semiring, GrB_PLUS_MONOID_UINT64, op)) ;

    //--------------------------------------------------------------------------
    // run the tests
    //--------------------------------------------------------------------------

    for (int by_row = 0 ; by_row <= 1 ; by_row++)
    {
        OK (GB_mex_test55_worker (by_row, GxB_SPARSE, big, semiring)) ;
        OK (GB_mex_test55_worker (by_row, GxB_BITMAP, big, semiring)) ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    OK (GrB_set (GrB_GLOBAL, false, GxB_BURBLE)) ;
    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test55: all tests passed\n\n") ;
}

