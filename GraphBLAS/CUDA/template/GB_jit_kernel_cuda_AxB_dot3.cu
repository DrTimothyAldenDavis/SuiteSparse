//------------------------------------------------------------------------------
// GraphBLAS/CUDA/jit_kernels/GB_jit_kernel_cuda_AxB_dot3.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_jit_kernel_cuda_AxB_dot3: C<M>=A'*B using the dot3 method on the GPU.

#define GB_FREE_ALL ;

#if GB_C_ISO
// FIXME
#error "kernel undefined for C iso"
#endif

// FIXME: Figure out how to use graphblas-specific INFINITY macro
#ifndef INFINITY
#define INFINITY std::numeric_limits<double>::max()
#endif

//------------------------------------------------------------------------------
// dot3 kernel launch geometry
//------------------------------------------------------------------------------

// FIXME: some duplicates here
// FIXME: tune these values.  Bigger chunk_size leads to fewer binary searches
// with GB_cuda_ek_slice_setup, for example.
#define chunk_size 128
#define log2_chunk_size 7
#define tile_sz 32 
#define log2_tile_sz 5 
#define shared_vector_size 128 
#define blocksize  32
#define threads_per_block 32

//------------------------------------------------------------------------------
// operators
//------------------------------------------------------------------------------

#if GB_C_ISO

    #define GB_DOT_TERMINAL( c ) break
    #define GB_DOT_MERGE(pA,pB)                                         \
    {                                                                   \
        cij_exists = true ;                                             \
    }
    #define GB_CIJ_EXIST_POSTCHECK

#else

    #define GB_DOT_TERMINAL( c ) GB_IF_TERMINAL_BREAK ( c, zterminal )

    #if GB_IS_PLUS_PAIR_REAL_SEMIRING

        // cij += A(k,i) * B(k,j), for merge operation (plus_pair_real semiring)
        #if GB_Z_IGNORE_OVERFLOW
            // plus_pair for int64, uint64, float, or double
            #define GB_DOT_MERGE(pA,pB) cij++ ;
            #define GB_CIJ_EXIST_POSTCHECK cij_exists = (cij != 0) ;
        #else
            // plus_pair semiring for small integers
            #define GB_DOT_MERGE(pA,pB)                                     \
            {                                                               \
                cij_exists = true ;                                         \
                cij++ ;                                                     \
            }
            #define GB_CIJ_EXIST_POSTCHECK
        #endif

    #else

        // cij += A(k,i) * B(k,j), for merge operation (general case)
        #define GB_DOT_MERGE(pA,pB)                                         \
        {                                                                   \
            GB_GETA ( aki, Ax, pA, ) ;      /* aki = A(k,i) */              \
            GB_GETB ( bkj, Bx, pB, ) ;      /* bkj = B(k,j) */              \
            cij_exists = true ;                                             \
            GB_MULTADD ( cij, aki, bkj, i, k, j ) ;  /* cij += aki * bkj */ \
        }
        #define GB_CIJ_EXIST_POSTCHECK

    #endif

#endif

//------------------------------------------------------------------------------
// dot3 buckets
//------------------------------------------------------------------------------

#define NBUCKETS 3

// NBUCKETS buckets: computed by up to NBUCKETS-1 kernel launches (zombies need
// no work...), each using different kernels (with different configurations
// depending on the bucket).

// dot3:  C<M>=A'B, M is sparse or hyper, C is sparse or hyper
// 32 kernels A,B: (hyper,sparse,bitmap,full)^2 x (M and C are sparse/hyper)

typedef enum
{
    GB_BUCKET_ZOMBIE = 0,       // C(i,j) is a zombie (not a bucket)
    // both A and B are sparse/hyper:
    GB_BUCKET_VSVS = 1,         // vsvs: both A(:,i) and B(:,j) are very sparse
    GB_BUCKET_MERGEPATH = 2,    // mp: use the merge-path method
    // A is sparse/hyper and B is bitmap/full, or
    // A is bitmap/full  and B is sparse/hyper
    GB_BUCKET_VSDN = 1,         // vsdn: the sparse vector is very sparse
    GB_BUCKET_SPDN = 2,         // spdn: sparse vector has lots of entries;
                                // use a whole warp for each dot product
}
GB_bucket_code ;    // FIXME: rename GB_dot3_bucket_code

// These may use another bucket enum:

    // two full/(sparse,hyper) kernels:
    //  // CUDA kernel: spdn, handles 4 buckets:
    //  // A(:,i) is dense and B(:,j) is very sparse (< 256 entries)
    //  GB_BUCKET_DNVS = 2,
    //  // A(:,i) is dense and B(:,j) is sparse (>= 256 entries)
    //  GB_BUCKET_DNSP = 3,

    // a sparse/full kernel
    //  // A(:,i) is very sparse (< 256 entries) and B(:,j) is dense
    //  GB_BUCKET_VSDN = 4,
    //  // A(:,i) is sparse (>= 256 entries) and B(:,j) is dense
    //  GB_BUCKET_SPDN = 5,

    // a sparse/bitmap kernel
    // a bitmap/bitmap kernel
    // a bitmap/sparse kernel
    // ...

#include "GB_cuda_tile_sum_uint64.cuh"
#include "GB_cuda_tile_reduce_ztype.cuh"

//------------------------------------------------------------------------------
// CUDA device kernels for each case
//------------------------------------------------------------------------------

#include "GB_cuda_ek_slice.cuh"

#if ((GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL))
    // dense-dense
    #include "GB_cuda_jit_AxB_dot3_dense_phase1.cuh"
    #include "GB_cuda_jit_AxB_dot3_phase3_dndn.cuh"
#else
    // sparse-sparse, sparse-dense, or dense-sparse

    #undef  GB_FREE_ALL
    #define GB_FREE_ALL                         \
    {                                           \
        GB_FREE_WORK (&Nanobuckets, Nb_size) ;  \
        GB_FREE_WORK (&Blockbucket, Bb_size) ;  \
        GB_FREE_WORK (&Bucketp, Bup_size) ;     \
        GB_FREE_WORK (&offset, O_size) ;        \
        GB_FREE_WORK (&Bucket, Bu_size) ;       \
    }

    #include "GB_cuda_jit_AxB_dot3_phase1.cuh"
    #include "GB_cuda_jit_AxB_dot3_phase2.cuh"
    #include "GB_cuda_jit_AxB_dot3_phase2end.cuh"
    #if ((GB_A_IS_SPARSE || GB_A_IS_HYPER) && \
         (GB_B_IS_SPARSE || GB_B_IS_HYPER))
        // sparse-sparse
        #include "GB_cuda_jit_AxB_dot3_phase3_mp.cuh"
        #include "GB_cuda_jit_AxB_dot3_phase3_vsvs.cuh"
    #else
        // sparse-dense or dense-sparse
        #include "GB_cuda_jit_AxB_dot3_phase3_spdn.cuh"
        #include "GB_cuda_jit_AxB_dot3_phase3_vsdn.cuh"
    #endif
#endif

//------------------------------------------------------------------------------
// host function to launch the CUDA kernels for dot3 on the GPU
//------------------------------------------------------------------------------

// #include "GB_cuda_timer.hpp"

extern "C"
{
    GB_JIT_CUDA_KERNEL_DOT3_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_DOT3_PROTO (GB_jit_kernel)
{

    // GpuTimer kernel_timer ;

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;
    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    #endif

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    #if ((GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL))
    // dense-dense case requires no workspace
    #else
    // sparse-sparse, sparse-dense, and dense-sparse requires workspace
    int64_t *Nanobuckets = NULL ; size_t Nb_size  = 0 ;
    int64_t *Blockbucket = NULL ; size_t Bb_size  = 0 ;
    int64_t *Bucket = NULL      ; size_t Bu_size  = 0 ;
    int64_t *Bucketp = NULL     ; size_t Bup_size = 0 ;
    int64_t *offset = NULL      ; size_t O_size   = 0 ;
    #endif

    //--------------------------------------------------------------------------
    // get problem size
    //--------------------------------------------------------------------------

    const GB_M_NVALS (mnz) ;
    int nblks_1 = (mnz + chunk_size - 1) / chunk_size ;
    int number_of_blocks_1 = GB_IMIN (nblks_1,  chunk_size * number_of_sms) ;

    // most methods can use these launch geometries:
    dim3 grid_1 (number_of_blocks_1) ;
    dim3 block (threads_per_block) ;

    //--------------------------------------------------------------------------
    // C<M>=A'*B via jitified kernels
    //--------------------------------------------------------------------------

    #if ((GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL))
    {

        //----------------------------------------------------------------------
        // (full or bitmap) times (full or bitmap)
        //----------------------------------------------------------------------

        // full/bitmap cases, which means we don't need buckets and zombies.
        // This is a much simpler kernel as a result, it only does the i,j
        // lookup and stores the values in Mi and Ci. 

        // Idea is to have each task work on a continguous block of columns of
        // C Note: for small tests, mnz is small so ntasks is be governed by
        // chunk_size, not chunk_size*number_of_sms.  For large problems in
        // production, chunk_size is less important since ntasks will likely be
        // bounded by chunk_size*number_of_sms (say 128*80 = 10,240 on a V100,
        // for the default chunk_size of 128).

        //----------------------------------------------------------------------
        // dense case, phase 1
        //----------------------------------------------------------------------

        // kernel_timer.Start();
        GB_cuda_AxB_dot3_dense_phase1_kernel <<<grid_1, block, 0, stream>>>
            (C, M) ;

        CUDA_OK (cudaStreamSynchronize(stream)) ;  // is this needed?

        // kernel_timer.Stop();
        // printf ("(GPU phase1 %12.6g ms )\n", kernel_timer.Elapsed()) ;

        //----------------------------------------------------------------------
        // dense case, phase "3" (FIXME: rename to dense_phase2)
        //----------------------------------------------------------------------

        int work_per_thread = 8 ;
        int blocksz = 64 ;
        work_per_thread = 8 ;
        if (mnz > 1024)
        {
            blocksz = 512 ;
            work_per_thread = 64 ;
        }
        int gridsz = GB_ICEIL (mnz, work_per_thread*blocksz) ;
        dim3 grid_2 (gridsz) ;

        // kernel_timer.Start();

        GB_cuda_AxB_dot3_phase3_dndn_kernel <<grid_2, block, 0, stream>>
            (C, M, A, B) ;

    }
    #else
    {

        //----------------------------------------------------------------------
        // (sparse or hyper) times (sparse or hyper)
        // (sparse or hyper) times (bitmap or full)
        // (bitmap or full) times (sparse or hyper)
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // construct the tasks for phase1 and phase2
        //----------------------------------------------------------------------

        // # of threads in phase1 and phase2 kernel launches are related
        // # by the size of the warp.  ph2_task = ph1_task/32 for example

        int64_t blockbuckets_size = NBUCKETS * number_of_blocks_1 ;
        int64_t nanobuckets_size = blockbuckets_size * threads_per_block ;

        Nanobuckets = GB_MALLOC_WORK (nanobuckets_size, int64_t, &Nb_size) ;
        Blockbucket = GB_MALLOC_WORK (blockbuckets_size, int64_t, &Bb_size) ;
        Bucketp = GB_MALLOC_WORK (NBUCKETS+1, int64_t, &Bup_size) ;
        offset = GB_MALLOC_WORK (NBUCKETS, int64_t, &O_size) ;
        Bucket = GB_MALLOC_WORK (mnz, int64_t, &Bu_size) ;

        if (Nanobuckets == NULL || Blockbucket == NULL || Bucketp == NULL
            || Bucket == NULL || offset == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // FIXME: do async with streams
        // FIXME: do we need any of these?
        //CUDA_OK (cudaMemsetAsync(Nanobuckets, 0,
        //    nanobuckets_size * sizeof(int64_t), stream));
        //CUDA_OK (cudaMemsetAsync(Blockbucket, 0,
        //    blockbuckets_size * sizeof(int64_t), stream));
        CUDA_OK (cudaMemsetAsync(Bucketp, 0,
            (NBUCKETS+1) * sizeof(int64_t), stream));
        CUDA_OK (cudaMemsetAsync(offset, 0,
            NBUCKETS * sizeof(int64_t), stream));
        //CUDA_OK (cudaMemsetAsync(Bucket, 0,
        //    mnz * sizeof(int64_t), stream));

        //----------------------------------------------------------------------
        // phase1 and phase2: place each C(i,j) in a bucket
        //----------------------------------------------------------------------

        CUDA_OK (cudaMemAdvise( Bucketp, (NBUCKETS+1) * sizeof ( int64_t),
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_OK (cudaMemAdvise( Bucketp, (NBUCKETS+1) * sizeof ( int64_t),
            cudaMemAdviseSetAccessedBy, device));

        CUDA_OK (cudaMemAdvise( offset, NBUCKETS * sizeof ( int64_t),
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_OK (cudaMemAdvise( offset, NBUCKETS * sizeof ( int64_t),
            cudaMemAdviseSetAccessedBy, device));

        //----------------------------------------------------------------------
        // phase1: assign each C(i,j) to a bucket, and count them
        //----------------------------------------------------------------------

        // kernel_timer.Start();

        GB_jit_AxB_dot3_phase1_kernel <<<grid_1, block, 0, stream>>>
            (Nanobuckets, Blockbucket, C, M, A, B) ;

        CUDA_OK (cudaStreamSynchronize (stream)) ;

        // kernel_timer.Stop();
        // printf ("(GPU phase1 %12.6g ms )\n", kernel_timer.Elapsed()) ;

        //----------------------------------------------------------------------
        // phase2: cumsum across the blockbuckets, propagate to thread level
        //----------------------------------------------------------------------

        // # of blocks for phase2:
        int number_of_blocks_2 = (number_of_blocks_1 + threads_per_block - 1)
            / threads_per_block ;

        dim3 grid_2 (number_of_blocks_2) ;

        // kernel_timer.Start();

        GB_cuda_AxB_dot3_phase2_kernel <<<grid_2, block, 0, stream>>>
            (Blockbucket, offset, number_of_blocks_1) ;

        CUDA_OK (cudaStreamSynchronize (stream)) ;

        int64_t s = offset [0] ;
        C->nzombies = s ;
        bool all_in_one = false ;
        for (int bucket = 1 ; bucket < NBUCKETS+1 ; bucket++)
        {
            Bucketp[bucket] = s ; 
            s += offset [bucket] ;
            if ((Bucketp [bucket] - Bucketp [bucket-1] ) == mnz)
            {
                all_in_one = true ;
            }
        }

        // kernel_timer.Stop();
        // printf ("(GPU phase2 %12.6g ms )\n", kernel_timer.Elapsed()) ;

        //----------------------------------------------------------------------
        // phase2end
        //----------------------------------------------------------------------

        if (!all_in_one) 
        {
            // kernel_timer.Start();

            GB_cuda_AxB_dot3_phase2end_kernel <<<grid_1, block, 0, stream>>>
                (Nanobuckets, Blockbucket, Bucketp, Bucket, offset, C, mnz) ;

            CUDA_OK (cudaStreamSynchronize (stream)) ;
            // kernel_timer.Stop();
            // printf ("(GPU phase2end %12.6g ms)\n",kernel_timer.Elapsed());
        }

        //----------------------------------------------------------------------
        // phase3: do the numerical work
        //----------------------------------------------------------------------

        // kernel_timer.Start();

        for (int bucket = 1 ; bucket < NBUCKETS ; bucket++)
        {
            int64_t start = Bucketp [bucket] ;
            int64_t end   = Bucketp [bucket + 1] ;
            int64_t cnz_in_bucket = end - start ;
            int gridsz, blocksz, work_per_thread ;
            if (cnz_in_bucket > 0)
            {

                #if ((GB_A_IS_SPARSE || GB_A_IS_HYPER) && \
                     (GB_B_IS_SPARSE || GB_B_IS_HYPER))

                    switch (bucket)
                    {

                        //------------------------------------------------------
                        // vsvs bucket: both vectors very sparse
                        //------------------------------------------------------

                        case GB_BUCKET_VSVS :
                        {
                            // FIXME: should be a function of cuda architecture
                            blocksz = 256 ;
                            work_per_thread = 4 ;
                            if (cnz_in_bucket > (2<<12))
                            {
                                blocksz = 512 ;
                            }
                            gridsz = GB_ICEIL (cnz_in_bucket,
                                work_per_thread*blocksz) ;
                            gridsz = GB_IMIN (gridsz, 256*number_of_sms) ;
                            dim3 grid_3 (gridsz) ;
                            GB_cuda_AxB_dot3_phase3_vsvs_kernel
                                <<<grid_3, block, 0, stream>>>
                                (start, end, Bucket, C, M, A, B) ;
                        }
                        break ;

                        //------------------------------------------------------
                        // mergepath bucket:
                        //------------------------------------------------------

                        case GB_BUCKET_MERGEPATH :
                        {
                            // FIXME: should be a function of cuda architecture
                            blocksz = 32 ;
                            work_per_thread = 256 ;
                            if (cnz_in_bucket > (2<<20))
                            {
                                work_per_thread = 1024 ;
                            }
                            gridsz = GB_ICEIL (cnz_in_bucket, work_per_thread) ;
                            if ((gridsz < number_of_sms) &&
                                (cnz_in_bucket > (2<<20)))
                            {
                                gridsz = number_of_sms ;
                            }
                            gridsz = GB_IMIN (gridsz, 256*number_of_sms) ;
                            dim3 grid_3 (gridsz) ;
                            GB_cuda_AxB_dot3_phase3_mp_kernel
                                <<<grid_3, block, 0, stream>>>
                                (start, end, Bucket, C, M, A, B) ;
                        }
                        break ;
                    }

                #else

                    switch (bucket)
                    {

                        //------------------------------------------------------
                        // vsdn bucket: one thread per C(i,j) dot product
                        //------------------------------------------------------

                        case GB_BUCKET_VSDN :
                        {
                            // FIXME: should be a function of cuda architecture
                            blocksz = 256 ;
                            work_per_thread = 4 ;
                            if (cnz_in_bucket > (2<<12))
                            {
                                blocksz = 512 ;
                            }
                            gridsz = GB_ICEIL (cnz_in_bucket,
                                work_per_thread*blocksz) ;
                            gridsz = GB_IMIN (gridsz, 256*number_of_sms) ;
                            dim3 grid_3 (gridsz) ;
                            GB_cuda_AxB_dot3_phase3_vsdn_kernel
                                <<<grid_3, block, 0, stream>>>
                                (start, end, Bucket, C, M, A, B) ;
                        }
                        break ;

                        //------------------------------------------------------
                        // spdn bucket: one warp per C(i,j) dot product
                        //------------------------------------------------------

                        case GB_BUCKET_SPDN :
                        {
                            // FIXME: should be a function of cuda architecture
                            blocksz = 32 ;
                            work_per_thread = 256 ;
                            if (cnz_in_bucket > (2<<20))
                            {
                                work_per_thread = 1024 ;
                            }
                            gridsz = GB_ICEIL (cnz_in_bucket, work_per_thread) ;
                            if ((gridsz < number_of_sms) &&
                                (cnz_in_bucket > (2<<20)))
                            {
                                gridsz = number_of_sms ;
                            }
                            gridsz = GB_IMIN (gridsz, 256*number_of_sms) ;
                            dim3 grid_3 (gridsz) ;
                            GB_cuda_AxB_dot3_phase3_spdn_kernel
                                <<<grid_3, block, 0, stream>>>
                                (start, end, Bucket, C, M, A, B) ;
                            break ;
                        }
                    }
                #endif
            }
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // kernel_timer.Stop();
    // printf ("(GPU phase3 %12.6g ms, rate=%12.6g)\n",
    //     kernel_timer.Elapsed(), mnz/(1000*kernel_timer.Elapsed())) ; 

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

