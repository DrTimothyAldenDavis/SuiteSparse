//------------------------------------------------------------------------------
// GB_dense_subassign_23_template: C += B where C is dense; B is sparse or dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// All entries in C+=B are computed fully in parallel, using the same kind of
// parallelism as Template/GB_AxB_colscale.c.

#include "GB_unused.h"

{

    //--------------------------------------------------------------------------
    // get C and B
    //--------------------------------------------------------------------------

    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) B->x ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    ASSERT (GB_is_dense (C)) ;

    if (kfirst_slice == NULL)
    {

        //----------------------------------------------------------------------
        // C += B when both C and B are dense
        //----------------------------------------------------------------------

        ASSERT (GB_is_dense (B)) ;
        const int64_t cnz = GB_NNZ (C) ;

        #if defined ( GB_HAS_CBLAS ) && GB_OP_IS_PLUS_REAL

            // C += B via GB_cblas_daxpy or GB_cblas_saxpy
            GB_CBLAS_AXPY           // Y += alpha*X
            (
                cnz,                // length of X and Y (note: int64_t)
                (GB_CTYPE) 1,       // alpha is 1.0
                Bx,                 // X, always stride 1
                Cx,                 // Y, always stride 1
                nthreads            // maximum # of threads to use
            ) ;

        #elif defined ( GB_HAS_CBLAS ) && GB_OP_IS_MINUS_REAL

            // C -= B via GB_cblas_daxpy or GB_cblas_saxpy
            GB_CBLAS_AXPY           // Y += alpha*X
            (
                cnz,                // length of X and Y (note: int64_t)
                (GB_CTYPE) -1,      // alpha is -1.0
                Bx,                 // X, always stride 1
                Cx,                 // Y, always stride 1
                nthreads            // maximum # of threads to use
            ) ;

        #else

            int64_t p ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            { 
                GB_GETB (bij, Bx, p) ;                  // bij = B(i,j)
                GB_BINOP (GB_CX (p), GB_CX (p), bij) ;  // C(i,j) += bij
            }

        #endif
    }
    else
    {

        //----------------------------------------------------------------------
        // C += B when C is dense and B is sparse
        //----------------------------------------------------------------------

        const int64_t *GB_RESTRICT Bp = B->p ;
        const int64_t *GB_RESTRICT Bh = B->h ;
        const int64_t *GB_RESTRICT Bi = B->i ;
        const int64_t cvlen = C->vlen ;

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            // if kfirst > klast then taskid does no work at all
            int64_t kfirst = kfirst_slice [taskid] ;
            int64_t klast  = klast_slice  [taskid] ;

            //------------------------------------------------------------------
            // C(:,kfirst:klast) += B(:,kfirst:klast)
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // find the part of B(:,k) and C(:,k) for this task
                //--------------------------------------------------------------

                int64_t j = (Bh == NULL) ? k : Bh [k] ;
                int64_t my_pB_start, my_pB_end ;
                GB_get_pA_and_pC (&my_pB_start, &my_pB_end, NULL,
                    taskid, k, kfirst, klast, pstart_slice, NULL, NULL, Bp) ;

                int64_t pB_start = Bp [k] ;
                bool ajdense = ((Bp [k+1] - pB_start) == cvlen) ;

                // pC points to the start of C(:,j) if C is dense
                int64_t pC = j * cvlen ;

                //--------------------------------------------------------------
                // C(:,j) += B(:,j)
                //--------------------------------------------------------------

                if (ajdense)
                { 

                    //----------------------------------------------------------
                    // both C(:,j) and B(:,j) are dense
                    //----------------------------------------------------------

                    #if defined ( GB_HAS_CBLAS ) && GB_OP_IS_PLUS_REAL

                        // y += x via GB_cblas_daxpy or GB_cblas_saxpy.
                        // use a single thread since this is already in a
                        // parallel region.

                        int64_t len = my_pB_end - my_pB_start ;
                        int64_t i = my_pB_start - pB_start ;
                        int64_t p = pC + i ;

                        GB_CBLAS_AXPY           // Y += alpha*X
                        (
                            len,                // length of X and Y
                            (GB_CTYPE) 1,       // alpha is 1.0
                            Bx + my_pB_start,   // X, always stride 1
                            Cx + p,             // Y, always stride 1
                            1                   // use a single thread
                        ) ;

                    #elif defined ( GB_HAS_CBLAS ) && GB_OP_IS_MINUS_REAL

                        // y -= x via GB_cblas_daxpy or GB_cblas_saxpy.
                        // use a single thread since this is already in a
                        // parallel region.

                        int64_t len = my_pB_end - my_pB_start ;
                        int64_t i = my_pB_start - pB_start ;
                        int64_t p = pC + i ;

                        GB_CBLAS_AXPY           // Y += alpha*X
                        (
                            len,                // length of X and Y
                            (GB_CTYPE) -1,      // alpha is -1.0
                            Bx + my_pB_start,   // X, always stride 1
                            Cx + p,             // Y, always stride 1
                            1                   // use a single thread
                        ) ;

                    #else

                        GB_PRAGMA_SIMD_VECTORIZE
                        for (int64_t pB = my_pB_start ; pB < my_pB_end ; pB++)
                        { 
                            int64_t i = pB - pB_start ;
                            int64_t p = pC + i ;
                            // bij = B(i,j)
                            GB_GETB (bij, Bx, pB) ;
                            // C(i,j) += bij
                            GB_BINOP (GB_CX (p), GB_CX (p), bij) ;
                        }

                    #endif

                }
                else
                {

                    //----------------------------------------------------------
                    // C(:,j) is dense; B(:,j) is sparse 
                    //----------------------------------------------------------

                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int64_t pB = my_pB_start ; pB < my_pB_end ; pB++)
                    { 
                        int64_t i = Bi [pB] ;
                        int64_t p = pC + i ;
                        GB_GETB (bij, Bx, pB) ;                 // bij = B(i,j)
                        GB_BINOP (GB_CX (p), GB_CX (p), bij) ;  // C(i,j) += bij
                    }
                }
            }
        }
    }
}

