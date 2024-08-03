//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase1.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// build nanobuckets, hunt for pre-zombies

// dot3, phase1: symbolic load balancing and data partition
// to assign work to different 'buckets' for later compute

// This kernel scans the non-zero pattern in A and B, takes into account the
// mask and computes total work required to form C. Then it classifies each dot
// product into a set of buckets for efficient compute.

// GB_AxB_cuda_dot3_phase1 is a CUDA kernel that scans all entries in C and
// assigns them to each of the NBUCKETS buckets.  The output is a
// NBUCKETS-by-blockDim array of bucket counts, per threadblock (the nanobucket
// array).  Each of the blockDim.x threads has its own set of NBUCKETS bucket
// counts.  Each threadblock in this kernel then computes the first part of the
// cumulative sum of the nanobuckets, and writes it to global memory.

// The kernel also computes Ci, of size nnz(C), which contains the
// zombie assignment or bucket assignment for non-zombies in C.

// Assigns the dot product C(i,j) = A(:,i)'*B(:,j) to a specific bucket.  Both
// A(:,i) and B(:,j) are non-empty when this method is called.
// GB_BUCKET_ZOMBIE:    C(i,j) is a prezombie, either A(:,i) or B(:,j) are
//                      empty.
// GB_BUCKET_VSVS       both A(:,i) and B(:,j) are very sparse.
// GB_BUCKET_MERGEPATH  both A(:,i) and B(:,j) are sparse, but neither are
//                      very sparse

// FIXME: What if all entries are in one bucket;
// can we skip the bucket creation?

__global__ void GB_jit_AxB_dot3_phase1_kernel
(
    // outputs, preallocated in global memory:
    int64_t *nanobuckets,   // array of size NBUCKETS-blockDim.x-by-gridDim.x
    int64_t *blockbucket,   // bucket counts, of size NBUCKETS-by-gridDim.x
    // input/output:
    GrB_Matrix C,           // final output matrix
    // inputs, not modified:
    const GrB_Matrix M,     // mask matrix
    const GrB_Matrix A,     // input matrix
    const GrB_Matrix B      // input matrix
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    #if GB_M_IS_HYPER
    const int64_t *__restrict__ Mh = M->h ;
    #endif
    const int64_t *__restrict__ Mp = M->p ;
    const int64_t *__restrict__ Mi = M->i ;
    #if !GB_MASK_STRUCT
    const GB_M_TYPE *__restrict__ Mx = (GB_M_TYPE *) M->x ;
    #endif
    const int64_t mnvec = M->nvec ;
    // const int64_t mvlen = M->vlen ;
    const GB_M_NVALS (mnz) ;
    ASSERT (GB_M_IS_SPARSE || GB_M_IS_HYPER) ;

    #if GB_A_IS_SPARSE || GB_A_IS_HYPER
    const int64_t *__restrict__ Ap = A->p ;
    #endif

    #if GB_B_IS_SPARSE || GB_B_IS_HYPER
    const int64_t *__restrict__ Bp = B->p ;
    #endif

    #if GB_A_IS_HYPER
    const int64_t anvec = A->nvec ;
    const int64_t *__restrict__ Ah = A->h ;
    const int64_t *__restrict__ A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const int64_t *__restrict__ A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const int64_t *__restrict__ A_Yx = (int64_t *)
        ((A->Y == NULL) ? NULL : A->Y->x) ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
    #endif

    #if GB_B_IS_HYPER
    const int64_t bnvec = B->nvec ;
    const int64_t *__restrict__ Bh = B->h ;
    const int64_t *__restrict__ B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
    const int64_t *__restrict__ B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
    const int64_t *__restrict__ B_Yx = (int64_t *)
        ((B->Y == NULL) ? NULL : B->Y->x) ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;
    #endif

    // int64_t *restrict Cp = C->p ;    // copy of Mp
    // int64_t *restrict Ch = C->h ;    // copy of Mh
    int64_t *__restrict__ Ci = C->i ;   // for zombies, or bucket assignment

    // FIXME: use (k << 2) not (k << 4)

    // Ci [p] for an entry C(i,j) contains either GB_FLIP(i) if C(i,j) is a
    // zombie, or (k << 4) + bucket otherwise, where C(:,j) is the kth vector
    // of C (j = Ch [k] if hypersparse or j = k if standard sparse), and
    // where bucket is the bucket assignment for C(i,j).
    // bucket can be recovered from Ci by bucket = Ci & 0xF

    //--------------------------------------------------------------------------
    // clear the bucket counters
    //--------------------------------------------------------------------------

    int64_t my_bucket [NBUCKETS] ;
    // each thread uses NBUCKETS bucket counters, held in register
    #pragma unroll
    for (int b = 0 ; b < NBUCKETS ; b++)
    {
        my_bucket [b] = 0 ;
    }

    //--------------------------------------------------------------------------
    // assign buckets to all entries in C(i,j), one chunk at a time
    //--------------------------------------------------------------------------

    // FIXME: tune this loop (and all others) for GPU architectures, where # of
    // threadblocks can differ on different GPUs.

    // grid-stride loop for each threadblock:
    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < mnz ;
                 pfirst += gridDim.x << log2_chunk_size)
    {

        //----------------------------------------------------------------------
        // find the vector k that contains each entry C(i,j) in this chunk
        //----------------------------------------------------------------------

        // This threadblock works on Mi/Mx and Ci/Mx, in positions pfirst to
        // pfirst + my_chunk_size - 1.
        int64_t my_chunk_size, mnvec1 ;
        float slope ;
        int64_t kfirst = GB_cuda_ek_slice_setup (Mp, mnvec, mnz, pfirst,
            chunk_size, &my_chunk_size, &mnvec1, &slope) ;

        //----------------------------------------------------------------------
        // assign entries in C(i,j) to the buckets
        //----------------------------------------------------------------------

        // block-stride loop for all threads in a threadblock, to do the whole
        // chunk assigned to this threadblock.
        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {

            //------------------------------------------------------------------
            // determine the kth vector that contains the pth entry
            //------------------------------------------------------------------

            // get the pM and k value of Mi,Mx [pM]
            int64_t pM ;    // = pfirst + pdelta
            int64_t k = GB_cuda_ek_slice_entry (&pM, pdelta, pfirst, Mp, mnvec1,
                kfirst, slope) ;

            //------------------------------------------------------------------
            // get C(i,j): zombie if A(:,i) and B(:,j) are empty or M(i,j) false
            //------------------------------------------------------------------

            // C(i,j) is in the kth vector of C, where j == k if C is sparse,
            // or j = Mh [k] if C is hypersparse

            GB_bucket_code bucket = GB_BUCKET_ZOMBIE ;
            int64_t i = Mi [pM] ;

            if (GB_MCAST (Mx, pM, ))        // if (M (i,j) is true):
            {

                //--------------------------------------------------------------
                // get B(:,j)
                //--------------------------------------------------------------

                #if GB_B_IS_SPARSE || GB_B_IS_HYPER
                int64_t j = GBH_M (Mh, k) ; // that Ch and Mh are the same
                int64_t pB, pB_end, bjnz ;
                #endif

                #if GB_B_IS_HYPER
                GB_hyper_hash_lookup (Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx,
                    B_hash_bits, j, &pB, &pB_end) ;
                bjnz = pB_end - pB ;
                if (bjnz > 0)
                #elif GB_B_IS_SPARSE
                pB     = Bp [j] ;
                pB_end = Bp [j+1] ;
                bjnz = pB_end - pB ;        // # of entries in B(:,j)
                if (bjnz > 0)
                #else
                // B is bitmap or full: no need to look up B(:,j)
                #endif
                {

                    //----------------------------------------------------------
                    // get A(:,i)
                    //----------------------------------------------------------

                    #if GB_A_IS_SPARSE || GB_A_IS_HYPER
                    int64_t pA, pA_end, ainz ;
                    #endif

                    #if GB_A_IS_HYPER
                    GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx,
                        A_hash_bits, i, &pA, &pA_end) ;
                    ainz = pA_end - pA ;
                    if (ainz > 0)
                    #elif GB_A_IS_SPARSE
                    pA     = Ap [i] ;
                    pA_end = Ap [i+1] ;
                    ainz = pA_end - pA ;        // # of entries in A(:,i)
                    if (ainz > 0)
                    #else
                    // A is bitmap or full: no need to look up A(:,i)
                    #endif
                    {
                        // determine the bucket for C(i,j)
                        #if (GB_A_IS_SPARSE || GB_A_IS_HYPER) && \
                            (GB_B_IS_SPARSE || GB_B_IS_HYPER)
                        // A and B are both sparse/hyper
                        bool vsvs = (ainz + bjnz <= 128) ;
                        bucket = (GB_bucket_code)
                           (  ((int) ( vsvs)) * ((int) GB_BUCKET_VSVS)
                            + ((int) (!vsvs)) * ((int) GB_BUCKET_MERGEPATH)) ;
                        #elif (GB_A_IS_SPARSE || GB_A_IS_HYPER) && \
                              (GB_B_IS_BITMAP || GB_B_IS_FULL)
                        // A is sparse/hyper, B is bitmap/full
                        bool vsvs = (ainz <= 128) ;
                        bucket = (GB_bucket_code)
                           (  ((int) ( vsvs)) * ((int) GB_BUCKET_VSDN)
                            + ((int) (!vsvs)) * ((int) GB_BUCKET_SPDN)) ;
                        #else
                        // A is bitmap/full, B is sparse/hyper
                        bool vsvs = (bjnz <= 128) ;
                        bucket = (GB_bucket_code)
                           (  ((int) ( vsvs)) * ((int) GB_BUCKET_VSDN)
                            + ((int) (!vsvs)) * ((int) GB_BUCKET_SPDN)) ;
                        #endif
                    }
                }
            }

            //------------------------------------------------------------------
            // assign C(i,j) to its bucket
            //------------------------------------------------------------------

            // encode the bucket or zombie status in the row index of C(i,j)
            Ci [pM] = (bucket == GB_BUCKET_ZOMBIE) * ( GB_FLIP(i) << 4)
                    + (bucket != GB_BUCKET_ZOMBIE) * ((k << 4) + bucket) ;

            // each thread counts its own bucket sizes
            my_bucket [bucket]++ ;
        }
    }

    this_thread_block().sync() ;

    //--------------------------------------------------------------------------
    // cumulative sum of each bucket
    //--------------------------------------------------------------------------

    typedef cub::BlockScan<int64_t, 32, cub::BLOCK_SCAN_WARP_SCANS> BlockCumSum;
    __shared__ typename BlockCumSum::TempStorage temp_storage ;

    // The taskbucket for this thread block is an array of size
    // NBUCKETS-by-blockDim.x, held by row.  Each thread owns one column of
    // this taskbucket, the nanobucket.  The nanobucket is a column of length
    // NBUCKETS, with stride equal to blockDim.x.
    int64_t *nanobucket =
        nanobuckets + blockIdx.x * (NBUCKETS * blockDim.x) + threadIdx.x ;

    #pragma unroll
    for (int b = 0 ; b < NBUCKETS ; b++)
    {
        if ( threadIdx.x == blockDim.x-1)
        {
            blockbucket [blockIdx.x + b * gridDim.x] = my_bucket[b] ;
        }
        this_thread_block().sync() ;

        BlockCumSum(temp_storage).ExclusiveSum( my_bucket[b], my_bucket[b]) ;

        this_thread_block().sync() ;

        nanobucket [b * blockDim.x] = my_bucket[b] ;
    }

    // The last thread now has the sum of all nanobuckets, which is then saved
    // to the global bucket counts.   blockbucket is an array of size
    // NBUCKETS-by-gridDim.x, held by row, with one column per thread block.
    // The last thread saves its result in the column of this thread block.
    // Note that this write to global memory is not coalesced.

    if (threadIdx.x == blockDim.x - 1 )
    {
        #pragma unroll
        for (int b = 0; b < NBUCKETS; ++b)
        {
            blockbucket [b * gridDim.x + blockIdx.x] += my_bucket[b];
        }
    }
}

