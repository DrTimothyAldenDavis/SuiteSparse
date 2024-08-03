//------------------------------------------------------------------------------
// GB_split_bitmap_jit: split A into a bitmap tile C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_SPLIT_BITMAP_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_split_bitmap_jit      // split A into a bitmap tile C
(
    // input/output
    GrB_Matrix C,
    // input:
    const GB_Operator op,
    const GrB_Matrix A,
    int64_t avstart,
    int64_t aistart,
    const int C_nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_SPLIT_BITMAP, GxB_BITMAP, true, C->type, op, false, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "split_bitmap",
        hash, &encoding, suffix, NULL, NULL,
        op, C->type, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, avstart, aistart, C_nthreads)) ;
}

