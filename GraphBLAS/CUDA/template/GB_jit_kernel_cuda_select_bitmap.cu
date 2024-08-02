using namespace cooperative_groups ;

#define tile_sz 32
#define log2_tile_sz 5

#include "GB_cuda_atomics.cuh"

#include "GB_cuda_tile_sum_uint64.cuh"


#include "GB_cuda_threadblock_sum_uint64.cuh"

__global__ void GB_cuda_select_bitmap_kernel
(
    int8_t *Cb_out,
    uint64_t *cnvals_out,
    GrB_Matrix A,
    const GB_void *thunk
)
{
    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    #if ( GB_A_IS_BITMAP )
    const int8_t *__restrict__ Ab = A->b ;
    #endif

    GB_A_NHELD (anz) ;
    int64_t nrows = A->vlen ;

    uint64_t my_keep = 0 ;
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;
    for (int64_t p = tid ; p < anz ; p += nthreads)
    {
        Cb_out [p] = 0 ;
        if (!GBB_A (Ab, p)) { continue; }

        #if ( GB_DEPENDS_ON_I )
        int64_t i = (p % nrows) ;
        #endif

        #if ( GB_DEPENDS_ON_J )
        int64_t j = (p / nrows) ;
        #endif
        
        #if ( GB_DEPENDS_ON_Y )
        GB_Y_TYPE y = * ((GB_Y_TYPE *) thunk) ;
        #endif

        GB_TEST_VALUE_OF_ENTRY (keep, p) ;
        if (keep) 
        {
            my_keep++ ;
            Cb_out [p] = 1 ;    
        } 
    }
    
    // compute cnvals for this block
    // IMPORTANT: every thread in the threadblock must participate in the warp reduction
    // for thread 0 to obtain the right result
    uint64_t block_keep = GB_cuda_threadblock_sum_uint64 (my_keep) ;

    if (threadIdx.x == 0)
    {
        // thread 0 updates global cnvals with atomics
        GB_cuda_atomic_add (cnvals_out, block_keep) ;
    }
}


extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_BITMAP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_SELECT_BITMAP_PROTO (GB_jit_kernel)
{
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    GB_cuda_select_bitmap_kernel <<<grid, block, 0, stream>>> (Cb, cnvals, A, ythunk) ;
    return (GrB_SUCCESS) ;
}
