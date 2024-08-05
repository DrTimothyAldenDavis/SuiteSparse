//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase2end.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase2end_kernel: fill the global buckets
//------------------------------------------------------------------------------

__global__ void GB_cuda_AxB_dot3_phase2end_kernel
(
    // input, not modified:
    const int64_t *__restrict__ nanobuckets,  // array of size
                                              // NBUCKETS-blockDim.x-by-nblocks
    const int64_t *__restrict__ blockbucket,  // global bucket count, of size
                                              // NBUCKETS*nblocks
    // output:
    const int64_t *__restrict__ bucketp,      // global bucket cumsum,
                                              // of size NBUCKETS+1
          int64_t *__restrict__ bucket,       // global buckets, of size
                                              // cnz == mnz
    const int64_t *__restrict__ offset,       // global offsets for each bucket
    // inputs, not modified:
    const GrB_Matrix C,      // output matrix
    const int64_t cnz        // number of entries in C and M
)
{

    //--------------------------------------------------------------------------
    // get C information 
    //--------------------------------------------------------------------------

    // Ci [p] for an entry C(i,j) contains either GB_FLIP(i) if C(i,j) is a
    // zombie, or (k << 4) + bucket otherwise, where C(:,j) is the kth vector
    // of C (j = Ch [k] if hypersparse or j = k if standard sparse), and
    // where bucket is the bucket assignment for C(i,j).  This phase does not
    // need k, just the bucket for each entry C(i,j).

    int64_t *__restrict__ Ci = C->i ;       // for zombies, or bucket assignment
    //int64_t *Mp = C->p ;       // for offset calculations
    //int64_t mnvec = C->nvec;

    //--------------------------------------------------------------------------
    // load and shift the nanobuckets for this thread block
    //--------------------------------------------------------------------------

    // The taskbucket for this threadblock is an array of size
    // NBUCKETS-by-blockDim.x, held by row.  It forms a 2D array within the 3D
    // nanobuckets array.
    const int64_t *taskbucket = nanobuckets +
        blockIdx.x * (NBUCKETS * blockDim.x) ;

    // Each thread in this threadblock owns one column of this taskbucket, for
    // its set of NBUCKETS nanobuckets.  The nanobuckets are a column of length
    // NBUCKETS, with stride equal to blockDim.x.

    const int64_t *nanobucket = taskbucket + threadIdx.x ;

    // Each thread loads its NBUCKETS nanobucket values into registers.
    int64_t my_bucket [NBUCKETS] ;

    #pragma unroll 
    for (int b = 0 ; b < NBUCKETS ; b++)
    {
        my_bucket [b] = nanobucket [b * blockDim.x]
                      + blockbucket [b * gridDim.x + blockIdx.x]
                      + bucketp [b] ;
    }

    // Now each thread has an index into the global set of NBUCKETS buckets,
    // held in bucket, of where to place its own entries.

    //--------------------------------------------------------------------------
    // construct the global buckets
    //--------------------------------------------------------------------------

    // The slice for task blockIdx.x contains entries pfirst:plast-1 of M and
    // C, which is the part of C operated on by this threadblock.

    // FIXME: why is bucket_idx needed?
    __shared__ int64_t bucket_idx [chunk_size] ;

//  int64_t chunk_max = (cnz + chunk_size -1) / chunk_size ;
//  for (int64_t chunk = blockIdx.x ; chunk < chunk_max ; chunk += gridDim.x)

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < cnz ;
                 pfirst += gridDim.x << log2_chunk_size)
    {

        // pfirst = chunk_size * chunk ;
        // plast  = GB_IMIN( chunk_size * (chunk+1), cnz ) ;
        int64_t plast = pfirst + chunk_size ;
        plast = GB_IMIN (plast, cnz) ;

        for (int64_t p = pfirst + threadIdx.x ; p < plast ; p += blockDim.x)
        {
            // get the entry C(i,j), and extract its bucket.  Then
            // place the entry C(i,j) in the global bucket it belongs to.
            int tid = p - pfirst ;

            // TODO: these writes to global are not coalesced.  Instead: each
            // threadblock could buffer its writes to NBUCKETS buffers and when
            // the buffers are full they can be written to global.

            int ibucket = Ci [p] & 0xF;

            //bucket[my_bucket[ibucket]++] = p;
            //int idx = (my_bucket[ibucket]  - pfirst); 
            //my_bucket[ibucket] +=  1; //blockDim.x ;
            //int idx = (my_bucket[ibucket]++ - pfirst) & 0x7F;
            //bucket_s[ibucket][ idx ] = p;

            bucket_idx [tid] = my_bucket [ibucket]++ ;
            Ci [p] = (ibucket==0) * (Ci [p] >> 4) + (ibucket > 0) * Ci [p] ;

            //if(ibucket == 0) {
            ////    bucket[my_bucket[0]++] = p;
            //    Ci[p] = Ci[p] >> 4;
            //} else {
            //  bucket[my_bucket[ibucket]++] = p;
            //}
        }

        // FIXME: can't this be merged with the loop above?  Or is it a
        // partial implementation of a coalesced write to the global bucket
        // array?

        for (int64_t p = pfirst + threadIdx.x ; p < plast ; p += blockDim.x)
        {
            int tid = p - pfirst ;
            bucket [bucket_idx [tid]] = p ;
        }
    }
}

