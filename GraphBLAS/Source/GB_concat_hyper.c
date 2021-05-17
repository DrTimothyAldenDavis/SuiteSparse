//------------------------------------------------------------------------------
// GB_concat_hyper: concatenate an array of matrices into a hypersparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL                 \
    GB_FREE (&Wi, Wi_size) ;        \
    GB_FREE_WERK (&Wj, Wj_size) ;   \
    GB_FREE_WERK (&Wx, Wx_size) ;   \
    GB_phbix_free (C) ;

#include "GB_concat.h"

GrB_Info GB_concat_hyper            // concatenate into a hypersparse matrix
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
    // allocate triplet workspace to construct C as hypersparse
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;

    int64_t *restrict Wi = NULL ; size_t Wi_size = 0 ;
    int64_t *restrict Wj = NULL ; size_t Wj_size = 0 ;
    GB_void *restrict Wx = NULL ; size_t Wx_size = 0 ;

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

    Wi = GB_MALLOC (cnz, int64_t, &Wi_size) ;               // becomes C->i
    Wj = GB_MALLOC_WERK (cnz, int64_t, &Wj_size) ;          // freed below
    Wx = GB_MALLOC_WERK (cnz * csize, GB_void, &Wx_size) ;  // freed below
    if (Wi == NULL || Wj == NULL || Wx == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    int64_t nouter = csc ? n : m ;
    int64_t ninner = csc ? m : n ;

    //--------------------------------------------------------------------------
    // concatenate all matrices into the list of triplets
    //--------------------------------------------------------------------------

    int64_t pC = 0 ;
    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {
        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A
            //------------------------------------------------------------------

            A = csc ? GB_TILE (Tiles, inner, outer)
                    : GB_TILE (Tiles, outer, inner) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;

            //------------------------------------------------------------------
            // determine where to place the tile in C
            //------------------------------------------------------------------

            // The tile A appears in vectors cvstart:cvend-1 of C, and indices
            // cistart:ciend-1.

            int64_t cvstart, cistart ;
            if (csc)
            { 
                // C is held by column
                // Tiles is row-major and accessed in column order
                cvstart = Tile_cols [outer] ;
                cistart = Tile_rows [inner] ;
            }
            else
            { 
                // C is held by row
                // Tiles is row-major and accessed in row order
                cvstart = Tile_rows [outer] ;
                cistart = Tile_cols [inner] ;
            }

            //------------------------------------------------------------------
            // extract the tuples from tile A
            //------------------------------------------------------------------

            int64_t anz = GB_NNZ (A) ;
            GB_OK (GB_extractTuples (
                (GrB_Index *) ((csc ? Wi : Wj) + pC),
                (GrB_Index *) ((csc ? Wj : Wi) + pC),
                Wx + pC * csize, (GrB_Index *) (&anz), ccode, A, Context)) ;

            //------------------------------------------------------------------
            // adjust the indices to reflect their new place in C
            //------------------------------------------------------------------

            int nth = GB_nthreads (anz, chunk, nthreads_max) ;
            if (cistart > 0 && cvstart > 0)
            { 
                int64_t pA ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (pA = 0 ; pA < anz ; pA++)
                {
                    Wi [pC + pA] += cistart ;
                    Wj [pC + pA] += cvstart ;
                }
            }
            else if (cistart > 0)
            { 
                int64_t pA ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (pA = 0 ; pA < anz ; pA++)
                {
                    Wi [pC + pA] += cistart ;
                }
            }
            else if (cvstart > 0)
            { 
                int64_t pA ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (pA = 0 ; pA < anz ; pA++)
                {
                    Wj [pC + pA] += cvstart ;
                }
            }

            //------------------------------------------------------------------
            // advance the tuple counter
            //------------------------------------------------------------------

            pC += anz ;
        }
    }

    //--------------------------------------------------------------------------
    // build C from the triplets
    //--------------------------------------------------------------------------

    GB_OK (GB_builder
    (
        C,                      // create C using a static or dynamic header
        ctype,                  // C->type
        cvlen,                  // C->vlen
        cvdim,                  // C->vdim
        csc,                    // C->is_csc
        (int64_t **) &Wi,       // Wi becomes C->i on output, or freed on error
        &Wi_size,
        (int64_t **) &Wj,       // Wj, free on output
        &Wj_size,
        (GB_void **) &Wx,       // Wx, free on output
        &Wx_size,
        false,                  // tuples need to be sorted
        true,                   // no duplicates
        cnz,                    // size of Wi and Wj in # of tuples
        true,                   // is_matrix: unused
        NULL, NULL, NULL,       // original I,J,S tuples, not used here
        cnz,                    // # of tuples
        NULL,                   // op for assembling duplicates (there are none)
        ccode,                  // type of Wx
        Context
    )) ;

    C->hyper_switch = hyper_switch ;
    C->bitmap_switch = bitmap_switch ;
    C->sparsity = sparsity_control ;
    ASSERT (C->static_header == static_header) ;
    ASSERT (GB_IS_HYPERSPARSE (C)) ;

    // workspace has been freed by GB_builder, or transplanted into C
    ASSERT (Wi == NULL) ;
    ASSERT (Wj == NULL) ;
    ASSERT (Wx == NULL) ;

    return (GrB_SUCCESS) ;
}

