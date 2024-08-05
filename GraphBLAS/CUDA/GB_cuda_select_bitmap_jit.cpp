#include "GB_cuda_select.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_SELECT_BITMAP_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_select_bitmap_jit
(
    // output:
    int8_t *Cb,
    uint64_t *cnvals,
    // input:
    const bool C_iso,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *ythunk,
    const GrB_IndexUnaryOp op,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_select (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_SELECT_BITMAP, C_iso, false, op, flipij, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_select_family, "cuda_select_bitmap",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) op, A->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cb, cnvals, A, ythunk, stream, gridsz, blocksz)) ;
}
