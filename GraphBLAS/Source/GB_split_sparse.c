//------------------------------------------------------------------------------
// GB_split_sparse: split a sparse/hypersparse matrix into tiles 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORK                        \
    GB_WERK_POP (C_ek_slicing, int64_t) ;   \
    GB_FREE_WERK (&Wp, Wp_size) ;

#define GB_FREE_ALL                         \
    GB_FREE_WORK ;                          \
    GB_Matrix_free (&C) ;

#include "GB_split.h"

GrB_Info GB_split_sparse            // split a sparse matrix
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
    int A_sparsity = GB_sparsity (A) ;
    bool A_is_hyper = (A_sparsity == GxB_HYPERSPARSE) ;
    ASSERT (A_is_hyper || A_sparsity == GxB_SPARSE) ;
    GrB_Matrix C = NULL ;
    GB_WERK_DECLARE (C_ek_slicing, int64_t) ;
    ASSERT_MATRIX_OK (A, "A sparse for split", GB0) ;

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

    int64_t anvec = A->nvec ;
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    size_t Wp_size = 0 ;
    int64_t *restrict Wp = NULL ;
    Wp = GB_MALLOC_WERK (anvec, int64_t, &Wp_size) ;
    if (Wp == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_memcpy (Wp, Ap, anvec * sizeof (int64_t), nthreads_max) ;

    //--------------------------------------------------------------------------
    // split A into tiles
    //--------------------------------------------------------------------------

    int64_t akend = 0 ;

    for (int64_t outer = 0 ; outer < nouter ; outer++)
    {

        //----------------------------------------------------------------------
        // find the starting and ending vector of these tiles
        //----------------------------------------------------------------------

        // The tile appears in vectors avstart:avend-1 of A, and indices
        // aistart:aiend-1.

        const int64_t avstart = Tile_vdim [outer] ;
        const int64_t avend   = Tile_vdim [outer+1] ;
        int64_t akstart = akend ;

        if (A_is_hyper)
        {
            // A is hypersparse: look for vector avend in the A->h hyper list.
            // The vectors to handle for this outer loop are in
            // Ah [akstart:akend-1].
            akend = akstart ;
            int64_t pright = anvec - 1 ;
            bool found ;
            GB_SPLIT_BINARY_SEARCH (avend, Ah, akend, pright, found) ;
//          printf ("anvec %ld akstart: %ld akend: %ld avend %ld\n",
//              anvec, akstart, akend, avend) ;
            ASSERT (GB_IMPLIES (akstart <= akend-1, Ah [akend-1] < avend)) ;
//          for (int64_t k = akstart ; k < akend ; k++)
//          {
//              printf ("   Ah [%ld] = %ld\n", k, Ah [k]) ;
//          }
        }
        else
        {
            // A is sparse; the vectors to handle are akstart:akend-1
            akend = avend ;
        }

        // # of vectors in all tiles in this outer loop
        int64_t cnvec = akend - akstart ;
        int nth = GB_nthreads (cnvec, chunk, nthreads_max) ;
// printf ("akend %ld\n", akend) ;

        //----------------------------------------------------------------------
        // create all tiles for vectors akstart:akend-1 in A
        //----------------------------------------------------------------------

        for (int64_t inner = 0 ; inner < ninner ; inner++)
        {
// printf ("\ninner: %ld, cnvec %ld akstart %ld akend %ld\n",
    // inner, cnvec, akstart, akend) ;

            //------------------------------------------------------------------
            // allocate C, C->p, and C->h for this tile
            //------------------------------------------------------------------

            const int64_t aistart = Tile_vlen [inner] ;
            const int64_t aiend   = Tile_vlen [inner+1] ;
            const int64_t cvdim = avend - avstart ;
            const int64_t cvlen = aiend - aistart ;

            C = NULL ;
            GB_OK (GB_new (&C, false,      // new header
                atype, cvlen, cvdim, GB_Ap_malloc, csc, A_sparsity,
                hyper_switch, cnvec, Context)) ;
            C->sparsity = sparsity_control ;
            C->hyper_switch = hyper_switch ;
            C->nvec = cnvec ;
            int64_t *restrict Cp = C->p ;
            int64_t *restrict Ch = C->h ;

            //------------------------------------------------------------------
            // determine the boundaries of this tile
            //------------------------------------------------------------------

            int64_t k ;
            #pragma omp parallel for num_threads(nth) schedule(static)
            for (k = akstart ; k < akend ; k++)
            {
                // printf ("look in kth vector of A: k = %ld\n", k) ;
                int64_t pA = Wp [k] ;
                int64_t pA_end = Ap [k+1] ;
                int64_t aknz = pA_end - pA ;
                // printf ("Wp [%ld] = %ld\n", k, Wp [k]) ;
                // printf ("Ap [%ld+1] = %ld\n", k, Ap [k+1]) ;
                // printf ("aknz %ld\n", aknz) ;
                if (aknz == 0 || Ai [pA] >= aiend)
                {
                    // this vector of C is empty
                    // printf ("empty\n") ;
                }
                else if (aknz > 256)
                {
                    // use binary search to find aiend
                    bool found ;
                    int64_t pright = pA_end - 1 ;
                    GB_SPLIT_BINARY_SEARCH (aiend, Ai, pA, pright, found) ;
                    #ifdef GB_DEBUG
                    // check the results with a linear search
                    int64_t p2 = Wp [k] ;
                    for ( ; p2 < Ap [k+1] ; p2++)
                    {
                        if (Ai [p2] >= aiend) break ;
                    }
                    ASSERT (pA == p2) ;
                    #endif
                }
                else
                { 
                    // use a linear-time search to find aiend
                    for ( ; pA < pA_end ; pA++)
                    {
                        if (Ai [pA] >= aiend) break ;
                    }
                    #ifdef GB_DEBUG
                    // check the results with a binary search
                    bool found ;
                    int64_t p2 = Wp [k] ;
                    int64_t p2_end = Ap [k+1] - 1 ;
                    GB_SPLIT_BINARY_SEARCH (aiend, Ai, p2, p2_end, found) ;
                    ASSERT (pA == p2) ;
                    #endif
                }
                Cp [k-akstart] = (pA - Wp [k]) ; // # of entries in this vector
                if (A_is_hyper)
                {
                    Ch [k-akstart] = Ah [k] - avstart ;
                }
                // printf ("cknz is %ld\n", Cp [k-akstart]) ;
            }

            GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nth, Context) ;
            int64_t cnz = Cp [cnvec] ;

            // for (int64_t k = 0 ; k <= cnvec ; k++)
            // {
                // printf ("Cp [%ld] = %ld\n", k, Cp [k]) ;
            // }

            //------------------------------------------------------------------
            // allocate C->i and C->x for this tile
            //------------------------------------------------------------------

            GB_OK (GB_bix_alloc (C, cnz, false, false, true, true, Context)) ;
            int64_t *restrict Ci = C->i ;
            // memset (Ci, 255, cnz * sizeof (int64_t)) ;

            //------------------------------------------------------------------
            // copy the tile from A into C
            //------------------------------------------------------------------

            int C_ntasks, C_nthreads ;
            GB_SLICE_MATRIX (C, 8, chunk) ;
// printf ("C_ntasks %d C_nthreads %d\n", C_ntasks, C_nthreads) ;
// printf ("C->nzmax %ld C->nvec %ld\n", C->nzmax, C->nvec) ;

            bool done = false ;

            #ifndef GBCOMPACT
            {
                // no typecasting needed
                switch (asize)
                {
                    #define GB_COPY(pC,pA) Cx [pC] = Ax [pA]

                    case 1 : // uint8, int8, bool, or 1-byte user-defined
                        #define GB_CTYPE uint8_t
                        #include "GB_split_sparse_template.c"
                        break ;

                    case 2 : // uint16, int16, or 2-byte user-defined
                        #define GB_CTYPE uint16_t
                        #include "GB_split_sparse_template.c"
                        break ;

                    case 4 : // uint32, int32, float, or 4-byte user-defined
                        #define GB_CTYPE uint32_t
                        #include "GB_split_sparse_template.c"
                        break ;

                    case 8 : // uint64, int64, double, float complex,
                             // or 8-byte user defined
                        #define GB_CTYPE uint64_t
                        #include "GB_split_sparse_template.c"
                        break ;

                    case 16 : // double complex or 16-byte user-defined
                        #define GB_CTYPE uint64_t
                        #undef  GB_COPY
                        #define GB_COPY(pC,pA)                      \
                            Cx [2*pC  ] = Ax [2*pA  ] ;             \
                            Cx [2*pC+1] = Ax [2*pA+1] ;
                        #include "GB_split_sparse_template.c"
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
                #include "GB_split_sparse_template.c"
            }

            //------------------------------------------------------------------
            // free workspace
            //------------------------------------------------------------------

            GB_WERK_POP (C_ek_slicing, int64_t) ;

            //------------------------------------------------------------------
            // advance to the next tile
            //------------------------------------------------------------------

            if (inner < ninner - 1)
            {
                int64_t k ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (k = akstart ; k < akend ; k++)
                {
                    int64_t ck = k - akstart ;
                    int64_t cknz = Cp [ck+1] - Cp [ck] ;
                    Wp [k] += cknz ;
                }
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

    GB_FREE_WORK ;
    return (GrB_SUCCESS) ;
}

