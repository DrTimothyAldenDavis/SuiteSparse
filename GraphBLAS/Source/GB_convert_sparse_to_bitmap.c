//------------------------------------------------------------------------------
// GB_convert_sparse_to_bitmap: convert from sparse/hypersparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ek_slice.h"
#ifndef GBCOMPACT
#include "GB_type__include.h"
#endif

#define GB_FREE_WORK                                \
{                                                   \
    GB_WERK_POP (A_ek_slicing, int64_t) ;           \
}

#define GB_FREE_ALL                                 \
{                                                   \
    GB_FREE_WORK ;                                  \
    if (!in_place) GB_FREE (&Ax_new, Ax_new_size) ; \
    GB_FREE (&Ab, Ab_size) ;                        \
}

GrB_Info GB_convert_sparse_to_bitmap    // convert sparse/hypersparse to bitmap
(
    GrB_Matrix A,               // matrix to convert from sparse to bitmap
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int8_t  *restrict Ab     = NULL ; size_t Ab_size = 0 ;
    GB_void *restrict Ax_new = NULL ; size_t Ax_new_size = 0 ;

    ASSERT_MATRIX_OK (A, "A converting sparse/hypersparse to bitmap", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // A can be jumbled on input
    ASSERT (GB_ZOMBIES_OK (A)) ;        // A can have zombies on input
    GBURBLE ("(sparse to bitmap) ") ;

    //--------------------------------------------------------------------------
    // determine the maximum number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // determine if the conversion can be done in-place
    //--------------------------------------------------------------------------

    // if in_place is true, then A->x does not change if A is as-if-full
    bool in_place = GB_as_if_full (A) ;

    //--------------------------------------------------------------------------
    // allocate A->b
    //--------------------------------------------------------------------------

    const int64_t anz = GB_NNZ (A) ;
    const int64_t avdim = A->vdim ;
    const int64_t avlen = A->vlen ;
    const int64_t anvec = A->nvec ;
    int64_t anzmax ;
    if (!GB_Index_multiply ((GrB_Index *) &anzmax, avdim, avlen))
    { 
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }
    anzmax = GB_IMAX (anzmax, 1) ;

    if (in_place || anz > 0)
    { 
        // if any entries exist, use malloc since all of Ab will be set below.
        Ab = GB_MALLOC (anzmax, int8_t, &Ab_size) ;
    }
    else
    { 
        // calloc Ab so all bitmap entries are zero; no need to touch them.
        // This case occurs when setting the GxB_SPARSITY_CONTROL of a new
        // matrix to GxB_BITMAP, with no entries.
        Ab = GB_CALLOC (anzmax, int8_t, &Ab_size) ;
    }

    if (Ab == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // allocate the new A->x
    //--------------------------------------------------------------------------

    const size_t asize = A->type->size ;
    bool Ax_shallow ;

    if (in_place)
    { 
        // keep the existing A->x
        Ax_new = (GB_void *) A->x ;
        Ax_shallow = A->x_shallow ;
        Ax_new_size = A->x_size ;
    }
    else
    {
        // A->x must be modified to fit the bitmap structure
        Ax_new = GB_MALLOC (anzmax * asize, GB_void, &Ax_new_size) ;
        Ax_shallow = false ;
        if (Ax_new == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // scatter the pattern and values into the new bitmap
    //--------------------------------------------------------------------------

    int64_t nzombies = A->nzombies ;
    if (in_place)
    { 

        //----------------------------------------------------------------------
        // the sparse A has all entries: convert in-place
        //----------------------------------------------------------------------

        ASSERT (nzombies == 0) ;
        // set all of Ab [0..anz-1] to 1, in parallel
        GB_memset (Ab, 1, anz, nthreads_max) ;

    }
    else if (anz > 0)
    {

        //----------------------------------------------------------------------
        // set all of Ab to zero
        //----------------------------------------------------------------------

        GB_memset (Ab, 0, anzmax, nthreads_max) ;

        //----------------------------------------------------------------------
        // scatter the values and pattern of A into the bitmap
        //----------------------------------------------------------------------

        int A_nthreads, A_ntasks ;
        GB_SLICE_MATRIX (A, 8, chunk) ;
        bool done = false ;

        #ifndef GBCOMPACT

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_convert_s2b_(cname) GB (_convert_s2b_ ## cname)
            #define GB_WORKER(cname)                                        \
            {                                                               \
                info = GB_convert_s2b_(cname) (A, Ax_new, Ab,               \
                    A_ek_slicing, A_ntasks, A_nthreads) ;                   \
                done = (info != GrB_NO_VALUE) ;                             \
            }                                                               \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            GB_Type_code acode = A->type->code ;
            if (acode < GB_UDT_code)
            { 
                switch (acode)
                {
                    case GB_BOOL_code   : GB_WORKER (_bool )
                    case GB_INT8_code   : GB_WORKER (_int8 )
                    case GB_INT16_code  : GB_WORKER (_int16 )
                    case GB_INT32_code  : GB_WORKER (_int32 )
                    case GB_INT64_code  : GB_WORKER (_int64 )
                    case GB_UINT8_code  : GB_WORKER (_uint8 )
                    case GB_UINT16_code : GB_WORKER (_uint16)
                    case GB_UINT32_code : GB_WORKER (_uint32)
                    case GB_UINT64_code : GB_WORKER (_uint64)
                    case GB_FP32_code   : GB_WORKER (_fp32  )
                    case GB_FP64_code   : GB_WORKER (_fp64  )
                    case GB_FC32_code   : GB_WORKER (_fc32  )
                    case GB_FC64_code   : GB_WORKER (_fc64  )
                    default: ;
                }
            }

        #endif

        if (!done)
        { 
            // Ax_new [pnew] = Ax [p]
            #define GB_COPY_A_TO_C(Ax_new,pnew,Ax,p) \
                memcpy (Ax_new +(pnew)*asize, Ax +(p)*asize, asize)
            #define GB_ATYPE GB_void
            #include "GB_convert_sparse_to_bitmap_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    if (in_place)
    {
        // if done in-place, remove A->x from A so it is not freed
        A->x = NULL ;
        A->x_shallow = false ;
    }

    GB_phbix_free (A) ;

    A->b = Ab ; A->b_size = Ab_size ;
    A->b_shallow = false ;
    Ab = NULL ;

    A->x = Ax_new ; A->x_size = Ax_new_size ;
    A->x_shallow = Ax_shallow ;
    Ax_new = NULL ;

    A->nzmax = anzmax ;
    A->nvals = anz - nzombies ;
    ASSERT (A->nzombies == 0) ;

    A->plen = -1 ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    ASSERT_MATRIX_OK (A, "A converted from sparse to bitmap", GB0) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

