//------------------------------------------------------------------------------
// GB_unjumble: unjumble the vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_sort.h"

GrB_Info GB_unjumble        // unjumble a matrix
(
    GrB_Matrix A,           // matrix to unjumble
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to unjumble", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;      // zombies must be killed first
    ASSERT (GB_PENDING_OK (A)) ;    // pending tuples are not modified

    if (!A->jumbled)
    {
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // full and bitmap matrices are never jumbled 
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_NNZ (A) ;
    const int64_t *restrict Ap = A->p ;
    int64_t *restrict Ai = A->i ;
    const size_t asize = A->type->size ;

    GB_void   *Ax   = (GB_void *) A->x ;
    uint8_t   *Ax1  = (uint8_t *) A->x ;
    uint16_t  *Ax2  = (uint16_t *) A->x ;
    uint32_t  *Ax4  = (uint32_t *) A->x ;
    uint64_t  *Ax8  = (uint64_t *) A->x ;
    GB_blob16 *Ax16 = (GB_blob16 *) A->x ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + anvec, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (32 * nthreads) ;
    ntasks = GB_IMIN (ntasks, anvec) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    //--------------------------------------------------------------------------
    // slice the work
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_slice, int64_t) ;
    GB_WERK_PUSH (A_slice, ntasks + 1, int64_t) ;
    if (A_slice == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_pslice (A_slice, Ap, anvec, ntasks, false) ;

    //--------------------------------------------------------------------------
    // sort the vectors
    //--------------------------------------------------------------------------

    switch (asize)
    {
        case 1 : 
            // GrB_BOOL, GrB_UINT8, GrB_INT8, and user defined types of size 1
            #define GB_QSORT \
                GB_qsort_1b_size1 (Ai+pA_start, Ax1+pA_start, aknz) ;
            #include "GB_unjumbled_template.c"
            break ;

        case 2 : 
            // GrB_UINT16, GrB_INT16, and user-defined types of size 2
            #define GB_QSORT \
                GB_qsort_1b_size2 (Ai+pA_start, Ax2+pA_start, aknz) ;
            #include "GB_unjumbled_template.c"
            break ;

        case 4 : 
            // GrB_UINT32, GrB_INT32, GrB_FP32, and user-defined types of size 4
            #define GB_QSORT \
                GB_qsort_1b_size4 (Ai+pA_start, Ax4+pA_start, aknz) ;
            #include "GB_unjumbled_template.c"
            break ;

        case 8 : 
            // GrB_UINT64, GrB_INT64, GrB_FP64, GxB_FC32, and user-defined
            // types of size 8
            #define GB_QSORT \
                GB_qsort_1b_size8 (Ai+pA_start, Ax8+pA_start, aknz) ;
            #include "GB_unjumbled_template.c"
            break ;

        case 16 : 
            // GxB_FC64, and user-defined types of size 16
            #define GB_QSORT \
                GB_qsort_1b_size16 (Ai+pA_start, Ax16+pA_start, aknz) ;
            #include "GB_unjumbled_template.c"
            break ;

        default : 
            // user-defined types of arbitrary size
            #define GB_QSORT \
                GB_qsort_1b (Ai+pA_start, Ax+pA_start*asize, asize, aknz) ;
            #include "GB_unjumbled_template.c"
            break ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_WERK_POP (A_slice, int64_t) ;
    A->jumbled = false ;        // A has been unjumbled
    ASSERT_MATRIX_OK (A, "A unjumbled", GB0) ;
    return (GrB_SUCCESS) ;
}

