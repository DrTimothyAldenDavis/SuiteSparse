//------------------------------------------------------------------------------
// GB_split_full: split a full matrix into an array of matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL         \
    GB_Matrix_free (&C) ;

#include "GB_split.h"

GrB_Info GB_split_full              // split a full matrix
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    const GrB_Matrix A,             // input matrix
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (GB_is_dense (A)) ;
    GrB_Matrix C = NULL ;

    int sparsity_control = A->sparsity ;
    float hyper_switch = A->hyper_switch ;
    bool csc = A->is_csc ;
    GrB_Type atype = A->type ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    size_t asize = atype->size ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    const int64_t *Tile_vdim = csc ? Tile_cols : Tile_rows ;
    const int64_t *Tile_vlen = csc ? Tile_rows : Tile_cols ;

    //--------------------------------------------------------------------------
    // split A into tiles
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {

        const int64_t avstart = Tile_vdim [outer] ;
        const int64_t avend   = Tile_vdim [outer+1] ;

        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // allocate the tile C
            //------------------------------------------------------------------

            // The tile appears in vectors avstart:avend-1 of A, and indices
            // aistart:aiend-1.

            const int64_t aistart = Tile_vlen [inner] ;
            const int64_t aiend   = Tile_vlen [inner+1] ;
            const int64_t cvdim = avend - avstart ;
            const int64_t cvlen = aiend - aistart ;
            int64_t cnz = cvdim * cvlen ;

            C = NULL ;
            GB_OK (GB_new_bix (&C, false,      // new header
                atype, cvlen, cvdim, GB_Ap_null, csc, GxB_FULL, false,
                hyper_switch, 0, cnz, true, Context)) ;
            C->sparsity = sparsity_control ;
            C->hyper_switch = hyper_switch ;
            int C_nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

            //------------------------------------------------------------------
            // copy the tile from A into C
            //------------------------------------------------------------------

            bool done = false ;

            #ifndef GBCOMPACT
            {
                // no typecasting needed
                switch (asize)
                {
                    #define GB_COPY(pC,pA) Cx [pC] = Ax [pA]

                    case 1 : // uint8, int8, bool, or 1-byte user-defined
                        #define GB_CTYPE uint8_t
                        #include "GB_split_full_template.c"
                        break ;

                    case 2 : // uint16, int16, or 2-byte user-defined
                        #define GB_CTYPE uint16_t
                        #include "GB_split_full_template.c"
                        break ;

                    case 4 : // uint32, int32, float, or 4-byte user-defined
                        #define GB_CTYPE uint32_t
                        #include "GB_split_full_template.c"
                        break ;

                    case 8 : // uint64, int64, double, float complex,
                             // or 8-byte user defined
                        #define GB_CTYPE uint64_t
                        #include "GB_split_full_template.c"
                        break ;

                    case 16 : // double complex or 16-byte user-defined
                        #define GB_CTYPE uint64_t
                        #undef  GB_COPY
                        #define GB_COPY(pC,pA)                      \
                            Cx [2*pC  ] = Ax [2*pA  ] ;             \
                            Cx [2*pC+1] = Ax [2*pA+1] ;
                        #include "GB_split_full_template.c"
                        break ;

                    default:;
                }
            }
            #endif

            if (!done)
            { 
                // user-defined types
                #define GB_CTYPE GB_void
                #undef  GB_COPY
                #define GB_COPY(pC,pA)  \
                    memcpy (Cx + (pC)*asize, Ax + (pA)*asize, asize) ;
                #include "GB_split_full_template.c"
            }

            //------------------------------------------------------------------
            // conform the tile and save it in the Tiles array
            //------------------------------------------------------------------

            C->magic = GB_MAGIC ;
            ASSERT_MATRIX_OK (C, "C for GB_split", GB0) ;
            GB_OK (GB_conform (C, Context)) ;
            if (csc)
            {
                GB_TILE (Tiles, inner, outer) = C ;
            }
            else
            {
                GB_TILE (Tiles, outer, inner) = C ;
            }
            C = NULL ;
        }
    }

    return (GrB_SUCCESS) ;
}

