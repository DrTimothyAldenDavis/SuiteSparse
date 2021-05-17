//------------------------------------------------------------------------------
// GB_subassign_05f: C(:,:)<C,struct> = scalar ; no S, C anything, M structural
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 05f: C(:,:)<C,struct> = scalar ; no S
// compare with Method 05e

// M:           present and aliased with C
// Mask_comp:   false
// Mask_struct: true
// C_replace:   false or effectively false
// accum:       NULL
// A:           scalar
// S:           none

// C is aliased to M, and can have any sparsity on input, so M is not provided
// here.  All values in C are replaced by the scalar.  C can have any sparsity
// format (hyper/sparse/bitmap/full).  If C is bitmap, only assignments where
// only assignments where (Cb [pC] == 1) are needed, but it's faster to just
// assign the scalar to all entries in Cx.

// TODO::: when uniform-valued matrices are supported, this method will take
// O(1) time.

// TODO: this method can be merged with 05d

#include "GB_subassign_methods.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL

GrB_Info GB_subassign_05f
(
    GrB_Matrix C,
    // input:
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for subassign method_05f", GB0) ;

    // C can be jumbled, in which case it remains so C on output 
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (!GB_PENDING (C)) ;

    const GB_Type_code ccode = C->type->code ;
    const size_t csize = C->type->size ;
    GB_GET_SCALAR ;

    int64_t cnz = GB_NNZ_HELD (C) ;

    //--------------------------------------------------------------------------
    // Method 05f: C(:,:)<C,s> = x ; C anything, x is a scalar, structural mask
    //--------------------------------------------------------------------------

    // Time: Optimal:  the method must iterate over all entries in C, and the
    // time is O(nnz(C)).  When uniform-valued matrices are supported, this
    // method will take O(1) time.

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    int64_t pC ;

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    // worker for built-in types
    #define GB_WORKER(ctype)                                                \
    {                                                                       \
        ctype *restrict Cx = (ctype *) C->x ;                            \
        ctype x = (*(ctype *) cwork) ;                                      \
        GB_PRAGMA (omp parallel for num_threads(nthreads) schedule(static)) \
        for (pC = 0 ; pC < cnz ; pC++)                                      \
        {                                                                   \
            Cx [pC] = x ;                                                   \
        }                                                                   \
    }                                                                       \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    // TODO: use fewer cases (1, 2, 4, 8, 16 bytes, and uint's)

    switch (C->type->code)
    {
        case GB_BOOL_code   : GB_WORKER (bool) ;
        case GB_INT8_code   : GB_WORKER (int8_t) ;
        case GB_INT16_code  : GB_WORKER (int16_t) ;
        case GB_INT32_code  : GB_WORKER (int32_t) ;
        case GB_INT64_code  : GB_WORKER (int64_t) ;
        case GB_UINT8_code  : GB_WORKER (uint8_t) ;
        case GB_UINT16_code : GB_WORKER (uint16_t) ;
        case GB_UINT32_code : GB_WORKER (uint32_t) ;
        case GB_UINT64_code : GB_WORKER (uint64_t) ;
        case GB_FP32_code   : GB_WORKER (float) ;
        case GB_FP64_code   : GB_WORKER (double) ;
        case GB_FC32_code   : GB_WORKER (GxB_FC32_t) ;
        case GB_FC64_code   : GB_WORKER (GxB_FC64_t) ;
        default:
            {
                // worker for all user-defined types
                GB_BURBLE_N (cnz, "(generic C(:,:)<C,struct>=x assign) ") ;
                GB_void *restrict Cx = (GB_void *) C->x ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (pC = 0 ; pC < cnz ; pC++)
                { 
                    memcpy (Cx +((pC)*csize), cwork, csize) ;
                }
            }
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C output for subassign method_05f", GB0) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    return (GrB_SUCCESS) ;
}

