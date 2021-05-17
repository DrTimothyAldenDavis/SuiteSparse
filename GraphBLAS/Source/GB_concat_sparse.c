//------------------------------------------------------------------------------
// GB_concat_sparse: concatenate an array of matrices into a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORK                            \
    if (S != NULL)                              \
    {                                           \
        for (int64_t k = 0 ; k < m * n ; k++)   \
        {                                       \
            GB_Matrix_free (&(S [k])) ;         \
        }                                       \
    }                                           \
    GB_FREE_WERK (&S, S_size) ;                 \
    GB_FREE_WERK (&Work, Work_size) ;           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;

#define GB_FREE_ALL         \
    GB_FREE_WORK ;          \
    GB_phbix_free (C) ;

#include "GB_concat.h"

GrB_Info GB_concat_sparse           // concatenate into a sparse matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const int64_t cnz,              // # of entries in C
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n,
    const GrB_Index m,
    const GrB_Index n,
    const int64_t *restrict Tile_rows,  // size m+1
    const int64_t *restrict Tile_cols,  // size n+1
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // allocate C as a sparse matrix
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int64_t *Work = NULL ;
    size_t Work_size = 0 ;
    GrB_Matrix *S = NULL ;
    size_t S_size = 0 ;

    GrB_Type ctype = C->type ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    bool csc = C->is_csc ;
    size_t csize = ctype->size ;
    GB_Type_code ccode = ctype->code ;

    float hyper_switch = C->hyper_switch ;
    float bitmap_switch = C->bitmap_switch ;
    int sparsity_control = C->sparsity ;
    bool static_header = C->static_header ;
    GB_phbix_free (C) ;
    GB_OK (GB_new_bix (&C, static_header,   // prior static or dynamic header
        ctype, cvlen, cvdim, GB_Ap_malloc, csc, GxB_SPARSE, false,
        hyper_switch, cvdim, cnz, true, Context)) ;
    C->bitmap_switch = bitmap_switch ;
    C->sparsity = sparsity_control ;
    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;
    Work = GB_CALLOC_WERK (ninner * cvdim, int64_t, &Work_size) ;
    S = GB_CALLOC_WERK (m * n, GrB_Matrix, &S_size) ;
    if (S == NULL || Work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count entries in each vector of each tile
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A; transpose and typecast, if needed
            //------------------------------------------------------------------

            A = csc ? GB_TILE (Tiles, inner, outer)
                    : GB_TILE (Tiles, outer, inner) ;
            GrB_Matrix T = NULL ;
            if (csc != A->is_csc)
            {
                // T = (ctype) A', not in-place, using a dynamic header
                GB_OK (GB_transpose (&T, ctype, csc, A,
                    NULL, NULL, NULL, false, Context)) ;
                // save T in array S
                if (csc)
                { 
                    GB_TILE (S, inner, outer) = T ;
                }
                else
                { 
                    GB_TILE (S, outer, inner) = T ;
                }
                A = T ;
                GB_MATRIX_WAIT (A) ;
            }
            ASSERT (C->is_csc == A->is_csc) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;

            //------------------------------------------------------------------
            // ensure the tile is not bitmap
            //------------------------------------------------------------------

            if (GB_IS_BITMAP (A))
            {
                if (T == NULL)
                {
                    // copy A into T
                    GB_OK (GB_dup2 (&T, A, true, NULL, Context)) ;
                    // save T in array S
                    if (csc)
                    { 
                        GB_TILE (S, inner, outer) = T ;
                    }
                    else
                    { 
                        GB_TILE (S, outer, inner) = T ;
                    }
                }
                // convert T from bitmap to sparse
                GB_OK (GB_convert_bitmap_to_sparse (T, Context)) ;
                A = T ;
            }

            ASSERT (!GB_IS_BITMAP (A)) ;

            //------------------------------------------------------------------
            // log the # of entries in each vector of the tile A
            //------------------------------------------------------------------

            const int64_t anvec = A->nvec ;
            const int64_t avlen = A->vlen ;
            int64_t cvstart = csc ?  Tile_cols [outer] : Tile_rows [outer] ;
            int64_t *restrict W = Work + inner * cvdim + cvstart ;
            int nth = GB_nthreads (anvec, chunk, nthreads_max) ;
            if (GB_IS_FULL (A))
            { 
                // A is full
                int64_t j ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (j = 0 ; j < anvec ; j++)
                {
                    // W [j] = # of entries in A(:,j), which is just avlen
                    W [j] = avlen ;
                }
            }
            else
            { 
                // A is sparse or hyper
                int64_t k ;
                int64_t *restrict Ah = A->h ;
                int64_t *restrict Ap = A->p ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (k = 0 ; k < anvec ; k++)
                {
                    // W [j] = # of entries in A(:,j), the kth column of A
                    int64_t j = GBH (Ah, k) ;
                    W [j] = Ap [k+1] - Ap [k] ; 
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of entries in each tile
    //--------------------------------------------------------------------------

    int nth = GB_nthreads (ninner*cvdim, chunk, nthreads_max) ;
    int64_t k ;
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (k = 0 ; k < cvdim ; k++)
    {
        int64_t s = 0 ;
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        { 
            int64_t p = inner * cvdim + k ;
            int64_t c = Work [p] ;
            Work [p] = s ;
            s += c ;
        }
        // total number of entries in C(:,k)
        Cp [k] = s ;
    }

    GB_cumsum (Cp, cvdim, &(C->nvec_nonempty), nthreads_max, Context) ; 

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (k = 0 ; k < cvdim ; k++)
    {
        int64_t pC = Cp [k] ;
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        { 
            int64_t p = inner * cvdim + k ;
            Work [p] += pC ;
        }
    }

    //--------------------------------------------------------------------------
    // concatenate all matrices into C
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A, either the temporary matrix T or the original A
            //------------------------------------------------------------------

            A = csc ? GB_TILE (S, inner, outer)
                    : GB_TILE (S, outer, inner) ;
            if (A == NULL)
            { 
                A = csc ? GB_TILE (Tiles, inner, outer)
                        : GB_TILE (Tiles, outer, inner) ;
            }

            ASSERT (!GB_IS_BITMAP (A)) ;
            ASSERT (C->is_csc == A->is_csc) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;
            GB_Type_code acode = A->type->code ;

            //------------------------------------------------------------------
            // determine where to place the tile in C
            //------------------------------------------------------------------

            // The tile A appears in vectors cvstart:cvend-1 of C, and indices
            // cistart:ciend-1.

            int64_t cvstart, cvend, cistart, ciend ;
            if (csc)
            { 
                // C and A are held by column
                // Tiles is row-major and accessed in column order
                cvstart = Tile_cols [outer] ;
                cvend   = Tile_cols [outer+1] ;
                cistart = Tile_rows [inner] ;
                ciend   = Tile_rows [inner+1] ;
            }
            else
            { 
                // C and A are held by row
                // Tiles is row-major and accessed in row order
                cvstart = Tile_rows [outer] ;
                cvend   = Tile_rows [outer+1] ;
                cistart = Tile_cols [inner] ;
                ciend   = Tile_cols [inner+1] ;
            }

            // get the workspace pointer array W for this tile
            int64_t *restrict W = Work + inner * cvdim + cvstart ;

            //------------------------------------------------------------------
            // slice the tile
            //------------------------------------------------------------------

            int64_t avdim = cvend - cvstart ;
            int64_t avlen = ciend - cistart ;
            ASSERT (avdim == A->vdim) ;
            ASSERT (avlen == A->vlen) ;
            int A_nthreads, A_ntasks ;
            const int64_t *restrict Ap = A->p ;
            const int64_t *restrict Ah = A->h ;
            const int64_t *restrict Ai = A->i ;
            GB_SLICE_MATRIX (A, 1, chunk) ;

            //------------------------------------------------------------------
            // copy the tile A into C
            //------------------------------------------------------------------

            bool done = false ;

            #ifndef GBCOMPACT
                if (ccode == acode)
                {
                    // no typecasting needed
                    switch (csize)
                    {
                        #define GB_COPY(pC,pA) Cx [pC] = Ax [pA]

                        case 1 : // uint8, int8, bool, or 1-byte user-defined
                            #define GB_CTYPE uint8_t
                            #include "GB_concat_sparse_template.c"
                            break ;

                        case 2 : // uint16, int16, or 2-byte user-defined
                            #define GB_CTYPE uint16_t
                            #include "GB_concat_sparse_template.c"
                            break ;

                        case 4 : // uint32, int32, float, or 4-byte user-defined
                            #define GB_CTYPE uint32_t
                            #include "GB_concat_sparse_template.c"
                            break ;

                        case 8 : // uint64, int64, double, float complex,
                                 // or 8-byte user defined
                            #define GB_CTYPE uint64_t
                            #include "GB_concat_sparse_template.c"
                            break ;

                        case 16 : // double complex or 16-byte user-defined
                            #define GB_CTYPE uint64_t
                            #undef  GB_COPY
                            #define GB_COPY(pC,pA)                      \
                                Cx [2*pC  ] = Ax [2*pA  ] ;             \
                                Cx [2*pC+1] = Ax [2*pA+1] ;
                            #include "GB_concat_sparse_template.c"
                            break ;

                        default:;
                    }
                }
            #endif

            if (!done)
            { 
                // with typecasting or user-defined types
                GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;
                size_t asize = A->type->size ;
                #define GB_CTYPE GB_void
                #undef  GB_COPY
                #define GB_COPY(pC,pA)  \
                    cast_A_to_C (Cx + (pC)*csize, Ax + (pA)*asize, asize) ;
                #include "GB_concat_sparse_template.c"
            }
    
            GB_WERK_POP (A_ek_slicing, int64_t) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    C->magic = GB_MAGIC ;
    return (GrB_SUCCESS) ;
}

