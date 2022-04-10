//------------------------------------------------------------------------------
// templates/GB_AxB_cuda_dot3_phase2: fill the global buckets
//------------------------------------------------------------------------------

// TODO describe me
#pragma once

#define GB_CUDA_KERNEL

#include "GB_cuda_buckets.h"
#include "matrix.h"
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>

using namespace cooperative_groups;

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp
{
   // Running prefix
   int64_t running_total;
   // Constructor
   __device__ BlockPrefixCallbackOp(int64_t running_total) : running_total(running_total) {}

   // Callback operator to be entered by the first warp of threads in the block.
   // Thread-0 is responsible for returning a value for seeding the block-wide scan.
   __device__ int64_t operator()(int64_t block_aggregate)
   {
     int64_t old_prefix = running_total;
     running_total += block_aggregate;
     return old_prefix;
   }
};

__inline__ 
__device__ void blockBucketExclusiveSum(int bucketId, int64_t *d_data, int nblocks)
{
   #define blocksize  32

   // Specialize BlockScan for a 1D block of 32 threads
   typedef cub::BlockScan<int64_t, 32, cub::BLOCK_SCAN_WARP_SCANS> BlockScan; 

   // Allocate shared memory for BlockScan
   __shared__ typename BlockScan::TempStorage temp_storage;

   // Initialize running total
   BlockPrefixCallbackOp prefix_op(0);

   // Have the block iterate over segments of items
   int64_t data=0;

   int64_t *blockbucket= d_data;

   for (int block_id = 0; block_id < nblocks; block_id += blocksize)
   {
    // Load a segment of consecutive items that are blocked across threads

    //printf("block %d entering sum\n",blockIdx.x);
      int loc = block_id + threadIdx.x;
      if ( loc < nblocks)
      { 
        //printf("block %di loading tid=%d\n",block_id,tid);
        data  = blockbucket[bucketId*nblocks    +loc ] ; 
      }
      __syncthreads();

      //printf("bb%d_%d s0 before prefix= %ld \n", block_id,bucketId, 
      //                     blockbucket[bucketId*nblocks + block_id+threadIdx.x] )  ; 
      // Collectively compute the block-wide exclusive prefix sum
      BlockScan(temp_storage).ExclusiveSum( data, data, prefix_op);
      __syncthreads();

      if ( loc < nblocks)
      { 
        blockbucket[bucketId*nblocks   +loc ]  = data  ; 
      }
      __syncthreads();

        //printf("bb%d_%d = %ld \n", block_id, bucketId, blockbucket[bucketId*nblocks+block_id+threadIdx.x] )  ; 
      
      data = 0;
   }
}


template< typename T, int tile_sz>
__inline__ __device__ T warp_ReduceSumPlus( thread_block_tile<tile_sz> tile, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = tile.size() / 2; i > 0; i /= 2) {
        val +=  tile.shfl_down( val, i);
    }
    return val; // note: only thread 0 will return full sum
}

template<typename T, int warpSize>
__inline__ __device__ T block_ReduceSum(thread_block g, T val)
{
  static __shared__ T shared[warpSize]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  thread_block_tile<warpSize> tile = tiled_partition<warpSize>( g );

  // Each warp performs partial reduction
  val = warp_ReduceSumPlus<T, warpSize>( tile, val);    

  // Wait for all partial reductions
  if (lane==0) { 
     //printf("thd%d warp%d sum is %d\n", threadIdx.x, wid, val);
     shared[wid]=val; // Write reduced value to shared memory
     //printf("thd%d stored warp %d sum %d\n", threadIdx.x, wid, val);
  }
  __syncthreads();              // Wait for all partial reductions

  if (wid > 0 ) return val ;
  //Final reduce within first warp
  if (wid==0) val = warp_ReduceSumPlus<T, warpSize>( tile, val) ; 

  return val;
}

// GB_AxB_cuda_dot3_phase2 is a CUDA kernel that takes as input the
// nanobuckets and blockbucket arrays computed by the first phase kernel,
// GB_AxB_cuda_dot3_phase1.  The launch geometry of this kernel must match the
// GB_AxB_cuda_dot3_phase1 kernel, with the same # of threads and threadblocks.

__global__ void AxB_phase2
(
    // input, not modified:
    int64_t *__restrict__ blockbucket,    // global bucket count, of size 12*nblocks
    // output:
    int64_t *__restrict__ offset,         // global offsets, for each bucket
    // inputs, not modified:
    const int nblocks         // input number of blocks to reduce
)
{

    //--------------------------------------------------------------------------
    // sum up the bucket counts of prior threadblocks
    //--------------------------------------------------------------------------

    // blockbucket is an array of size 12-by-nblocks, held by row.  The
    // entry blockbucket [bucket * nblocks + t] holds the # of entries
    // in the bucket (in range 0 to 11) found by threadblock t.


    //__shared__ uint64_t offset [12] ;
    uint64_t s_0=0;
    uint64_t s_1=0;
    uint64_t s_2=0;
    uint64_t s_3=0;
    uint64_t s_4=0;
    uint64_t s_5=0;
    uint64_t s_6=0;
    uint64_t s_7=0;
    uint64_t s_8=0;
    uint64_t s_9=0;
    uint64_t s_10=0;
    uint64_t s_11=0;

    thread_block_tile<32> tile = tiled_partition<32>(this_thread_block() );

    //printf("block %d entering sum\n",blockIdx.x);
    int tid = threadIdx.x  + blockIdx.x * blockDim.x;
    #define reduceBucket( B )    \
     for( tid = threadIdx.x + blockIdx.x*blockDim.x; \
          tid < nblocks;  \
          tid += blockDim.x*gridDim.x) \
     {                           \
        s_ ## B  += blockbucket[  B *nblocks +tid] ;  \
     } \
     __syncthreads(); \
     s_ ## B  = warp_ReduceSumPlus<uint64_t , 32>( tile, s_ ## B); 

     reduceBucket( 0 )
     reduceBucket( 1 )
     reduceBucket( 2 )
     reduceBucket( 3 )
     reduceBucket( 4 )
     reduceBucket( 5 )
     reduceBucket( 6 )
     reduceBucket( 7 )
     reduceBucket( 8 )
     reduceBucket( 9 )
     reduceBucket( 10 )
     reduceBucket( 11 )


        //printf("summing blk,tid=%d,%d\n",blockIdx.x,threadIdx.x);
       if (threadIdx.x ==0 )
       {
           printf("s_0: %ld, s_1=%ld, s_10=%ld, s_11=%ld\n", s_0, s_1, s_10, s_11);
          atomicAdd( (unsigned long long int*)&(offset[0]), s_0);
          atomicAdd( (unsigned long long int*)&(offset[1]), s_1);
          atomicAdd( (unsigned long long int*)&(offset[2]), s_2);
          atomicAdd( (unsigned long long int*)&(offset[3]), s_3);
          atomicAdd( (unsigned long long int*)&(offset[4]), s_4);
          atomicAdd( (unsigned long long int*)&(offset[5]), s_5);
          atomicAdd( (unsigned long long int*)&(offset[6]), s_6);
          atomicAdd( (unsigned long long int*)&(offset[7]), s_7);
          atomicAdd( (unsigned long long int*)&(offset[8]), s_8);
          atomicAdd( (unsigned long long int*)&(offset[9]), s_9);
          atomicAdd( (unsigned long long int*)&(offset[10]),s_10);
          atomicAdd( (unsigned long long int*)&(offset[11]),s_11);
       }
       __syncthreads();
       


    if( gridDim.x >= 12)
    {
        // Cumulative sum across blocks for each bucket 
        if (blockIdx.x <12)
           blockBucketExclusiveSum( blockIdx.x, blockbucket, nblocks ) ;
    }
    else
    {
        if (blockIdx.x == 0)
        {
           blockBucketExclusiveSum( 0, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 1, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 2, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 3, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 4, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 5, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 6, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 7, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 8, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 9, blockbucket, nblocks ) ;
           blockBucketExclusiveSum( 10, blockbucket, nblocks) ;
           blockBucketExclusiveSum( 11, blockbucket, nblocks) ;
        }
    }
} // phase2
