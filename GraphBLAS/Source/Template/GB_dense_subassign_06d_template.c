//------------------------------------------------------------------------------
// GB_dense_subassign_06d_template: C<A> = A where C is dense or bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get C and A
    //--------------------------------------------------------------------------

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    const int64_t  *restrict Ap = A->p ;
    const int64_t  *restrict Ah = A->h ;
    const int64_t  *restrict Ai = A->i ;
    const int8_t   *restrict Ab = A->b ;
    const GB_ATYPE *restrict Ax = (GB_ATYPE *) A->x ;
    const int64_t avlen = A->vlen ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_dense = GB_as_if_full (A) ;
    const int64_t anz = GB_NNZ_HELD (A) ;

    GB_CTYPE *restrict Cx = (GB_CTYPE *) C->x ;
    int8_t   *restrict Cb = C->b ;
    const int64_t cvlen = C->vlen ;
    const bool C_is_bitmap = GB_IS_BITMAP (C) ;

    //--------------------------------------------------------------------------
    // C<A> = A
    //--------------------------------------------------------------------------

    int64_t cnvals = C->nvals ;     // for C bitmap

    if (A_is_dense)
    { 

        //----------------------------------------------------------------------
        // A is dense: all entries present
        //----------------------------------------------------------------------

        if (C_is_bitmap)
        { 

            //------------------------------------------------------------------
            // C is bitmap, A is dense
            //------------------------------------------------------------------

            if (Mask_struct)
            {
                // C<A,struct>=A with C bitmap, A dense
                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                { 
                    // Cx [p] = Ax [p]
                    GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                }
                GB_memset (Cb, 1, anz, A_nthreads) ;
                cnvals = anz ;
            }
            else
            {
                // C<A>=A with C bitmap, A dense
                int tid ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static) reduction(+:cnvals)
                for (tid = 0 ; tid < A_nthreads ; tid++)
                {
                    int64_t pA_start, pA_end, task_cnvals = 0 ;
                    GB_PARTITION (pA_start, pA_end, anz, tid, A_nthreads) ;
                    for (int64_t p = pA_start ; p < pA_end ; p++)
                    {
                        if (GB_AX_MASK (Ax, p, asize))
                        { 
                            // Cx [p] = Ax [p]
                            GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C is hypersparse, sparse, or full, with all entries present
            //------------------------------------------------------------------

            if (Mask_struct)
            {
                // C<A,struct>=A with C sparse/hyper/full
                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                { 
                    // Cx [p] = Ax [p]
                    GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                }
            }
            else
            {
                // C<A>=A with C sparse/hyper/full
                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                {
                    if (GB_AX_MASK (Ax, p, asize))
                    { 
                        // Cx [p] = Ax [p]
                        GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                    }
                }
            }
        }

    }
    else if (A_is_bitmap)
    {
        //----------------------------------------------------------------------
        // A is bitmap
        //----------------------------------------------------------------------

        if (C_is_bitmap)
        {

            //------------------------------------------------------------------
            // C is bitmap, A is bitmap
            //------------------------------------------------------------------

            if (Mask_struct)
            {
                // C<A,struct>=A with A and C bitmap
                int tid ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static) reduction(+:cnvals)
                for (tid = 0 ; tid < A_nthreads ; tid++)
                {
                    int64_t pA_start, pA_end, task_cnvals = 0 ;
                    GB_PARTITION (pA_start, pA_end, anz, tid, A_nthreads) ;
                    for (int64_t p = pA_start ; p < pA_end ; p++)
                    {
                        if (Ab [p])
                        { 
                            // Cx [p] = Ax [p]
                            GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {
                // C<A>=A with A and C bitmap
                int tid ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static) reduction(+:cnvals)
                for (tid = 0 ; tid < A_nthreads ; tid++)
                {
                    int64_t pA_start, pA_end, task_cnvals = 0 ;
                    GB_PARTITION (pA_start, pA_end, anz, tid, A_nthreads) ;
                    for (int64_t p = pA_start ; p < pA_end ; p++)
                    {
                        if (Ab [p] && GB_AX_MASK (Ax, p, asize))
                        { 
                            // Cx [p] = Ax [p]
                            GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C is hypersparse, sparse, or full, with all entries present
            //------------------------------------------------------------------

            if (Mask_struct)
            {
                // C<A,struct>=A with A bitmap, and C hyper/sparse/full
                // this method is used by LAGraph_bfs_parent when q is
                // a bitmap and pi is full.
                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                {
                    // Cx [p] = Ax [p]
                    if (Ab [p])
                    { 
                        GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                    }
                }
            }
            else
            {
                // C<A>=A with A bitmap, and C hyper/sparse/full
                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                {
                    if (Ab [p] && GB_AX_MASK (Ax, p, asize))
                    { 
                        // Cx [p] = Ax [p]
                        GB_COPY_A_TO_C (Cx, p, Ax, p) ;
                    }
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A is hypersparse or sparse; C is dense or a bitmap
        //----------------------------------------------------------------------

        const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
        const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
        const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;
        int taskid ;

        if (Mask_struct)
        {
            if (C_is_bitmap)
            {

                //--------------------------------------------------------------
                // C is bitmap, mask is structural
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1) reduction(+:cnvals)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    int64_t task_cnvals = 0 ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH (Ah, k) ;
                        int64_t pA_start, pA_end ;
                        GB_get_pA (&pA_start, &pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice, Ap, avlen) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C bitmap, A sparse
                        GB_PRAGMA_SIMD_REDUCTION (+,task_cnvals)
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        { 
                            int64_t p = pC + Ai [pA] ;
                            // Cx [p] = Ax [pA]
                            GB_COPY_A_TO_C (Cx, p, Ax, pA) ;
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is full, mask is structural
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH (Ah, k) ;
                        int64_t pA_start, pA_end ;
                        GB_get_pA (&pA_start, &pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice, Ap, avlen) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C full, A sparse
                        GB_PRAGMA_SIMD_VECTORIZE
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        { 
                            int64_t p = pC + Ai [pA] ;
                            // Cx [p] = Ax [pA]
                            GB_COPY_A_TO_C (Cx, p, Ax, pA) ;
                        }
                    }
                }
            }

        }
        else
        {
            if (C_is_bitmap)
            {

                //--------------------------------------------------------------
                // C is bitmap, mask is valued
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1) reduction(+:cnvals)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    int64_t task_cnvals = 0 ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH (Ah, k) ;
                        int64_t pA_start, pA_end ;
                        GB_get_pA (&pA_start, &pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice, Ap, avlen) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C bitmap, A sparse
                        GB_PRAGMA_SIMD_REDUCTION (+,task_cnvals)
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        {
                            if (GB_AX_MASK (Ax, pA, asize))
                            { 
                                int64_t p = pC + Ai [pA] ;
                                // Cx [p] = Ax [pA]
                                GB_COPY_A_TO_C (Cx, p, Ax, pA) ;
                                task_cnvals += (Cb [p] == 0) ;
                                Cb [p] = 1 ;
                            }
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is full, mask is valued
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1) reduction(+:cnvals)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH (Ah, k) ;
                        int64_t pA_start, pA_end ;
                        GB_get_pA (&pA_start, &pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice, Ap, avlen) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C dense, A sparse
                        GB_PRAGMA_SIMD_VECTORIZE
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        {
                            if (GB_AX_MASK (Ax, pA, asize))
                            { 
                                int64_t p = pC + Ai [pA] ;
                                // Cx [p] = Ax [pA]
                                GB_COPY_A_TO_C (Cx, p, Ax, pA) ;
                            }
                        }
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // log the number of entries in the C bitmap
    //--------------------------------------------------------------------------

    if (C_is_bitmap)
    { 
        C->nvals = cnvals ;
    }
}

