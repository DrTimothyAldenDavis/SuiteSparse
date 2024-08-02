using namespace cooperative_groups ;

// do not #include functions inside of other functions!
#include "GB_cuda_ek_slice.cuh"

#define log2_chunk_size 10
#define chunk_size 1024

__global__ void GB_cuda_colscale_kernel
(
    GrB_Matrix C,
    GrB_Matrix A,
    GrB_Matrix D
)
{

    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *__restrict__ Dx = (GB_B_TYPE *) D->x ;
    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) C->x ;

    #if ( GB_A_IS_SPARSE || GB_A_IS_HYPER )
    const int64_t *__restrict__ Ap = A->p ;
        #if ( GB_A_IS_HYPER )
        const int64_t *__restrict__ Ah = A->h ;
        #endif
    #endif

    #if ( GB_A_IS_BITMAP )
    const int8_t *__restrict__ Ab = A->b ;
    #endif

    GB_A_NHELD (anz) ;

    #if (GB_A_IS_BITMAP || GB_A_IS_FULL)
        const int64_t avlen = A->vlen ;
        // bitmap/full case
        int nthreads_in_entire_grid = blockDim.x * gridDim.x ;
        int tid = blockIdx.x * blockDim.x + threadIdx.x ;
        for (int64_t p = tid ; p < anz ; p += nthreads_in_entire_grid)
        {
            if (!GBB_A (Ab, p)) continue ;
            // the pth entry in A is A(i,j) where i = p%avlen and j = p/avlen
            int64_t col_idx = p / avlen ;
    //      int64_t row_idx = p % avlen ;
            GB_DECLAREB (djj) ;
            GB_GETB (djj, Dx, col_idx, ) ;
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, ) ;
            // C has same sparsity as A; ewise op code does not change
            GB_EWISEOP (Cx, p, aij, djj, 0, 0) ;
        }

    #else
        const int64_t anvec = A->nvec ;
        // sparse/hypersparse case (cuda_ek_slice only works for sparse/hypersparse)
        for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                    pfirst < anz ;
                    pfirst += gridDim.x << log2_chunk_size )
            {
                int64_t my_chunk_size, anvec_sub1 ;
                float slope ;
                int64_t kfirst = GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst,
                    chunk_size, &my_chunk_size, &anvec_sub1, &slope) ;

                for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
                {
                    int64_t p_final ;
                    int64_t k = GB_cuda_ek_slice_entry (&p_final, pdelta, pfirst, Ap, anvec_sub1, kfirst, slope) ;
                    int64_t j = GBH_A (Ah, k) ;

                    GB_DECLAREB (djj) ;
                    GB_GETB (djj, Dx, j, ) ;
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, p_final, ) ;
                    GB_EWISEOP (Cx, p_final, aij, djj, 0, 0) ;
                }
            }
    #endif

    // not needed because threads do entirely independent work:
    // this_thread_block ( ).sync( ) ;
}

extern "C" {
    GB_JIT_CUDA_KERNEL_COLSCALE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_COLSCALE_PROTO (GB_jit_kernel)
{
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (!GB_IS_BITMAP (D)) ;
    ASSERT (!GB_IS_FULL (D)) ;
    ASSERT (!C->iso) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    
    GB_cuda_colscale_kernel <<<grid, block, 0, stream>>> (C, A, D) ;

    return (GrB_SUCCESS) ;
}
